# AED 显存利用率与调度模型分析

日期：2026-08-22  
范围：GPU0、BF16、双 FireRed AED worker、独立 FireRedPunc owner、默认 overlap 关闭。

## 结论

当前方案是安全、稳定但明显保守的基线，不能称为显存和端到端耗时的最优解。

用户看到的约 2 GiB 额外余量，主要来自成本模型的系统性高估；同时，当前预测指标与安全约束使用的显存口径不一致，生产形状预热也不充分。因此不能直接把显存预测系数按实测 allocated 增量缩小到三分之一后上线。

“AED 阶段后约 60% 接近峰值”表示采样值达到本次 AED 阶段观测峰值的 95%，不是达到安全上限的 95%。它主要反映 CUDA allocator、workspace 和 runtime 懒加载完成后的平台区间，不能证明调度已经最优。

## 实测证据

来源：

- [三次连续任务结果](../../outputs/measurements/20260822T-three-repeat-liuyangyang-fixed/)
- [AED 显存时序结果](../../outputs/measurements/20260822T-aed-memory-trace-liuyangyang-v2/diagnostics.json)

最新 deep 任务的 AED 阶段数据如下：

| 指标 | 数值 |
|---|---:|
| GPU 总显存 | 24576 MiB |
| AED 开始时空闲显存 | 12516 MiB |
| 安全余量 | 1536 MiB |
| 调度瞬时预算 | 10980 MiB |
| AED 开始时已用显存 | 12059 MiB |
| AED 阶段峰值已用显存 | 20881 MiB |
| AED 阶段实际驱动可见增量 | 8822 MiB |
| 峰值空闲显存 | 3695 MiB |
| 预算利用率 | 80.3% |
| 超出安全余量的剩余显存 | 2159 MiB |

调度器预计两个 lane 的并发增量为 10942 MiB，几乎填满 10980 MiB 预算；实际驱动可见增量为 8822 MiB，说明当前模型在这一输入上保守约 2.1 GiB。

同一任务的时序采样共有 135 个点，AED 阶段平均全卡利用率约 79.1%，峰值约 85.0%。达到观测峰值 95% 的采样比例约 62.2%；项目净显存对应比例约 60.7%。

## 根因拆分

### 1. 成本模型系统性高估

当前 `BatchCostModel` 使用固定公式：

```python
predicted_memory = 384 + 0.045 * padded_frames
predicted_time = 0.05 + 0.0003 * padded_frames + 0.01 * item_count
```

此前三次 20 batch 实测中，预测瞬时显存平均约 1037 MiB，实际 PyTorch allocated 增量平均约 349 MiB，实际/预测约 33.5%。预测总耗时约 96.2 秒，实际约 65–67 秒。

最新四批次大 batch 实测中：

| 批次 | 预测显存增量 | 实际 allocated 增量 |
|---|---:|---:|
| 1 | 5466 MiB | 2179 MiB |
| 2 | 5468 MiB | 2547 MiB |
| 3 | 5476 MiB | 3129 MiB |
| 4 | 2946 MiB | 2210 MiB |

四批次预测总耗时 127.13 秒，实际耗时 87.59 秒。

`BatchCostModel.observe()` 虽然存在，但仓库中没有调用点，`observations` 不会增长，模型不会吸收历史任务的实际数据。

### 2. 显存统计口径不一致

调度安全预算来自 GPU free/`nvidia-smi` 驱动可见显存；批次预测准确性目前主要使用：

```text
peak_allocated - resident_allocated
```

这只反映 PyTorch allocated，不包含完整的 CUDA context、allocator reserved、cuBLAS workspace 和驱动层缓存。

最新时序中，两个 AED worker 的驱动可见显存大致从 2812 MiB 增长到 7140/7300 MiB，增长约 4.3–4.5 GiB；这明显高于 allocated 增量，但低于当前模型预测的约 5.5 GiB。

因此，不能直接把模型系数从 `0.045` 降为 allocated 实测斜率后用于生产安全决策。

### 3. 预热没有覆盖生产形状

当前显式预热主要使用短的 `hello_zh.wav` 单条样本，没有覆盖：

- 数百个片段组成的大 batch；
- 8–15 秒长片段；
- 大量 timestamp/token 的解码形状；
- 双 AED 并发 workspace。

这解释了 AED 开始后显存从约 2.8 GiB/worker 逐步阶跃到 7.1–7.3 GiB/worker，并在后段形成平台。首次生产形状推理触发的懒分配不应被误认为调度 batch 的瞬时需求。

### 4. 调度算法存在结构限制

当前算法是“按时长升序的贪心装箱 + 固定两路预算 + 完整 batch 的 LPT 分配”。它不是全局最短 makespan 优化，具体限制包括：

- 时间模型只使用 padded frames 和 item count，没有充分使用最长片段、总帧数和 decoder/token 特征；
- 先装箱再分 worker，完整 batch 不能拆分；
- 两个 lane 固定均分预算，没有依据 worker 的实际状态动态分配；
- 以每个 lane 最大单批增量判断并发安全，隐含前一批显存完全释放的假设；
- CUDA caching allocator 的 reserved/driver 显存可能跨批次保留或累积。

因此，调度器是当前损失的一部分，但不是 2 GiB 余量的首要原因；首先应修正观测和成本模型。

### 5. 诊断净显存需要按 PID 去重

`service` 和 `speaker` 当前是同一个服务 PID，但 trace 中按 owner 名称相加会重复计入约 618 MiB。全卡 `used_mib/free_mib` 不受影响；项目净显存时序和 owner 合计需要改为按 PID 去重。

## 实施顺序

1. 统一 allocated、reserved、driver-visible 三套显存口径，修正 trace 的 PID 去重，并新增 AED 阶段实际增量、预算利用率和超出安全余量指标。
2. 让深度诊断记录每批 allocated/reserved 和可观测的 driver/PID 峰值，区分首次懒加载基线与批次瞬时增量。
3. 增加代表性生产形状预热，至少覆盖短片段大 batch、中等片段、长片段和双 AED 并发；预热和模型加载仍排除在作业耗时之外。
4. 用已有语料和深度 trace 离线拟合显存与耗时的稳健上界模型：显存加入 padded frames、total frames、最长片段等特征，耗时加入最长片段、item count、总帧数和 token 代理特征；优先使用 P95/P99 残差而不是单次平均值。
5. 将校准结果接入后续任务的成本模型，不能在当前任务中改变已经确定的完整计划；第一版校准状态只保存在服务生命周期内，验证后再考虑持久化。
6. 在校准后重新设计两路联合装箱和 makespan 分配，目标是安全约束下的最短实际 AED wall time，而不是单纯填满显存。
7. 在相同常驻服务和预热状态下，对 60%、70%、80%、90%、95% 的 transient budget 做重复 A/B，检查耗时、显存、OOM、输出 fingerprint 和准确性。

## 当前不应做的事情

- 不直接把 `gpu_safety_margin_mib` 从 1536 MiB 调小；
- 不直接把 `memory_per_padded_frame_mib` 降到当前 allocated 实测斜率；
- 不把 near-peak 60% 解读成 60% 时间接近安全上限；
- 不把“大 batch 占用更多显存”当成必然更快；
- 不使用其他项目 PID 的数据做本项目调度参数。

## 预算矩阵初测结果

工具：[benchmark_aed_budget_matrix.py](../../tools/benchmark_aed_budget_matrix.py)。

在同一常驻服务中按预算比例运行时，低预算和高预算都暴露了跨任务状态影响：

- 0.60 候选前两次成功，AED 分别约 38.2 秒和 33.7 秒，但第三次出现 `FireRed returned text without sentences`；其中一次安全采样已经低于 1536 MiB，不能作为合格候选。
- 0.70 候选因单段最低预测需求 384 MiB 大于当时 lane budget 338 MiB，被安全拒绝。
- 0.80 候选首次运行约 75.2 秒 AED、峰值空闲约 2463 MiB，后续因前序任务造成的常驻/缓存增长而无法继续稳定运行，并出现过 OOM 证据。
- 在重启本项目服务后的新一组测试中，0.90 连续三次完成，AED 约 46.0、70.6、70.6 秒；0.95 至少两次完成，AED 约 63.9、62.5 秒。高预算组的前序状态曾出现申请 2.90 GiB、GPU 仅余 2.44 GiB，以及申请 274 MiB、GPU 仅余 88 MiB 的 OOM 记录。

这些结果不能直接选出新的生产默认比例，因为输出 fingerprint 在部分重复中变化，且服务内 CUDA reserved/driver 显存会跨任务增长。它们确认了两个实施要求：

1. 预算矩阵必须把安全余量不足、预算利用率超过 100%、OOM 和协议失败标为不合格，而不是只比较耗时；
2. 最终 A/B 必须以“同一新鲜服务状态下的代表性预热 + 候选重复”，并在每个候选组之间记录或重置常驻状态，不能把跨组累积的 allocator 状态混在一起。

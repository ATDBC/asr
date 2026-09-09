# 双 AED 固定权重方案扩展性分析

日期：2026-09-09
分支：`codex/server-weighted-aed-batching`
实现提交：`49a9822960df0dccf871f7076962bd806749b046`

## 结论

第二个 AED worker 增加了并发请求和模型副本，但仍与第一个 worker 共享同一块物理 GPU0。实测表明两条 lane 的装箱已经足够均衡，主要损失来自同 GPU 并发时每条 lane 自身变慢，而不是 batch 分配不均或显存容量不足。

单 AED 的稳态 AED wall 为 43.95 秒。若两个 worker 能以单 worker 原速完全并行，理论值应为 21.98 秒；实际双 AED 为 35.59 秒，只缩短 19.01%，即 1.235 倍加速。实际节省 8.36 秒，只兑现理论可节省时间的 38.03%。

## 测量范围

- 输入：`audio_test/刘洋洋.wav`，时长约一小时；
- 设备：物理 GPU0，NVIDIA GeForce RTX 3090；
- 精度：`torch.bfloat16`；
- 调度器：`weighted_duration`；
- 固定权重：longest `0.12854663332839417`，padded `0.008409640427755745`；
- batch 形状限制：最长 200 秒、最多 32 items；
- 双 AED 模式不使用显存规划上限、显存 admission 或串行回退；
- 两个 AED lane 独立推进，一个独立 Punc owner 与 AED 重叠；
- 对比值取三次稳态运行的中位数。

证据：

- 双 AED：`outputs/measurements/20260909T-server-weighted-dual-aed-punc-overlap-liuyangyang/summary.json`
  - SHA256：`873f1a259077f7489d237b5b7b049a6b0cb5d13a8bbccc14a3b37aea35456e1a`
- 单 AED + Punc overlap：`outputs/measurements/20260909T-server-weighted-single-aed-punc-overlap-liuyangyang/summary.json`
  - SHA256：`30386011146768cdad24f7727e81a499023dfd14d5a58198647fb95bc73ca3db`

## 结果分解

| 指标 | 单 AED | 双 AED | 变化 |
|---|---:|---:|---:|
| AED wall | 43.95 秒 | 35.59 秒 | -19.01% |
| ASR | 67.23 秒 | 59.03 秒 | -12.19% |
| 端到端 | 101.29 秒 | 93.47 秒 | -7.73% |
| GPU0 设备级采样峰值 | 11319 MiB | 14522 MiB | +3203 MiB |

| 运行 | worker-0 | worker-1 | 合计 | AED wall |
|---|---:|---:|---:|---:|
| steady-1 | 35.02 秒 | 35.85 秒 | 70.86 秒 | 35.85 秒 |
| steady-2 | 34.85 秒 | 35.59 秒 | 70.44 秒 | 35.59 秒 |
| steady-3 | 34.82 秒 | 35.59 秒 | 70.41 秒 | 35.59 秒 |

两条 lane 的 worker 时间合计中位数为 70.44 秒，比单 AED 的 43.95 秒增加 60.28%。因此主要问题不是两路没有同时运行，而是两路同时运行后，各自完成相同预测工作所需的时间显著增加。

## 原因判断

### 1. 同一块 GPU 的计算资源没有增加

两个 worker 共享 GPU0 的 SM、Tensor Core、显存带宽和 kernel 调度资源。第二份模型能放入显存只解决容量问题，不会复制计算吞吐。单 AED 已使用最大 32 items、200 秒的批次，单个 batch 本身已有较高 GPU 并行度，留给第二个进程填充的空闲执行槽有限。

### 2. 普通多进程 CUDA context 会产生竞争

当前实现由两个独立进程各自创建 CUDA context。项目配置、启动路径和本次日志均没有显式设置 `CUDA_MPS_*`，也没有调用 `nvidia-cuda-mps-control`；因此本项目没有主动启用 NVIDIA MPS。除非系统管理员在项目外全局启用了 MPS，本次结果应按普通双 CUDA context 解释。

NVIDIA 文档说明，普通不同 CUDA context 的工作存在时间片调度；MPS 可以让多进程共享调度资源、减少 context 切换，并在单进程未占满 GPU 时改善 kernel 重叠。RTX 3090 的计算能力为 8.6，服务器为 Linux，满足 MPS 基本条件，但满足条件不等于已经启用，也不保证该 workload 接近两倍加速。

参考：

- <https://docs.nvidia.com/deploy/mps/architecture.html>
- <https://docs.nvidia.com/deploy/mps/when-to-use-mps.html>
- <https://developer.nvidia.com/cuda/gpus>

### 3. 装箱已经接近均衡

- 27 个 batch 固定分为 13/14；
- 预测 lane 不平衡为 0.30%；
- 实测 lane 不平衡中位数为 2.14%；
- 两条 lane 的实测差约 0.74 秒。

即使进一步做到完全均衡，也只能回收亚秒级时间，无法解释相对于 21.98 秒理想值的 13.62 秒差距。因此不应优先重算两个装箱权重。

### 4. Punc overlap 可能带来次要竞争

双 AED 的稳态 Punc 模型耗时中位数为 2.32 秒。两种对比方案都启用了独立 Punc owner 和 overlap，因此 Punc 不是新增变量；但双 AED 高负载期间的重叠形态可能改变，仍可能占用少量计算或显存带宽。现有 basic diagnostics 无法精确分离这部分。

## 已验证与待验证边界

已验证：

- 双 lane 确实并发执行，计划在首个 AED 请求前固定；
- batch 分配稳定且均衡；
- 双 lane worker 时间合计相对单 AED 膨胀约 60.28%；
- 四次双 AED 完整输出一致，并与当前单 AED + Punc overlap 输出一致；
- 无 OOM、warning 或安全失败；
- GPU0 在服务停止后恢复到 admission 基线。

尚未直接验证：

- SM occupancy、Tensor Core 利用率和显存带宽各自贡献；
- 两个独立 CUDA context 的调度损耗占比；
- Punc 在 AED 峰值区间的精确资源占用；
- 不同短/长片段比例下 1.60 倍并发膨胀系数是否稳定；
- MPS 或单进程双 stream 能否改善当前 workload。

这些问题需要项目内 batch 级时间线或 profiler 证据。当前设备级 `nvidia-smi` 采样只能说明 GPU 峰值和恢复状态，不能区分 kernel、SM 或带宽瓶颈。

## 后续验证顺序

1. 只在独占维护窗口或专用测试 GPU 上进行，不在共享生产 GPU0 上直接启用 MPS。
2. 保持同一输入、BF16、固定权重、27 个 batch 和 Punc overlap，仅改变 CUDA 多进程执行方式。
3. 对比普通双进程与 MPS 双进程；若改动成本可控，再增加单进程双 stream。
4. 每组至少执行一次 fresh 和三次 steady，同时记录 AED wall、两 lane worker 时间和、lane 不平衡、Punc tail、GPU0 峰值、输出 fingerprint、warning、OOM 和安全结果。
5. 只有输出完全一致、无 OOM/安全回归且稳态 AED wall 稳定改善时，才考虑调整生产执行方式。

MPS daemon、GPU compute mode 和监控归属属于设备级行为，可能影响同一 GPU 上的其他项目。当前安全边界禁止检查或管理其他项目进程，因此不能在现有共享环境中直接启用或试验 MPS。

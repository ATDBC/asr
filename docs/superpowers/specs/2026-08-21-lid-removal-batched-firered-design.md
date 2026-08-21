# 移除 FireRedLID 与 VAD 段批处理设计

## 目标

普通转录路径不再依赖或加载 FireRedLID，固定为：

```text
FireRedVAD -> FireRedASR2-AED -> FireRedPunc
```

在 VAD 结束后，使用 FireRed 官方 `asr_batch_size` 和 `punc_batch_size` 将多个互相
独立的 VAD 语音段放入同一次模型调用，以实现单个 GPU worker 内的批量并行计算。服务
仍只消费一个 GPU 作业，仍只使用物理 GPU 0。

本设计不通过线程并发调用同一个 FireRed 模型，也不启动多个 FireRed worker。

## 已确认事实与边界

- 上游官方 `FireRedAsr2System` 在 `enable_lid=False` 时不会构造或加载
  `FireRedLid`；因此可实际减少模型占用，而不只是跳过一次调用。
- 当前每个 VAD 段最长 60 秒，AED、LID 和 Punc 的 batch 大小均为 1。AED 和 LID
  逐段运行，随后 Punc 再逐段运行。
- 本次仅移除 LID 运行时、代码接口、模型校验/下载及 LID 专属验收项。已经下载到本机
  的 `models/fireredasr2s/FireRedLID` 保留且不触碰；它已不再被程序引用。
- 3D-Speaker、说话人归属、可选 overlap、FIFO、自动启动/停止服务、诊断的资源 profile
  均不在本次改变范围内。

## 方案选择

### 采用：官方批处理

新增 `[asr] segment_batch_size`，同时配置官方的 `asr_batch_size` 和
`punc_batch_size`。VAD 段按原始顺序累积到一个 batch；AED 处理该 batch，随后 Punc
处理对应的结果 batch，最终仍按原始 VAD 顺序格式化、验证和导出。

这属于 GPU 的真实批量并行：多个段同时成为一个模型前向调用的 batch 维度，而不是由
多个 Python 线程争抢同一模型和 CUDA context。默认值保持 1，直至校准实测明确选出可
以提升吞吐、保持输出质量并符合显存预算的值。

### 不采用：多线程/多 CUDA stream 调用一个模型

上游 API 没有声明并发调用安全性。这样会引入共享状态、输出排序、CUDA 资源竞争和难以
复现的结果，不以生产路径实现或作为基准方案。

### 不采用：多个 FireRed worker

每个 worker 都会复制 AED、Punc 和 VAD 模型，显著增加 GPU 0 占用；这会破坏与同卡
其他项目共存的显存约束。

## 运行时与数据契约变更

1. `firered_worker.py` 删除 `FireRedLidConfig`、`lid_model_dir` 和 LID inventory，
   并构造官方系统时传入 `enable_lid=False`。
2. worker 稳定 JSONL 响应、`RecognizedSentence`、FireRed contract 和客户端解析器删除
   `language` 与 `language_confidence`；句子保留文字、时间范围和 `asr_confidence`。
3. pipeline 不再生成语言汇总或 LID 置信度告警；`diagnostics.asr` 保留时长、VAD 段、
   句子文字/时间和 AED 置信度。
4. 配置删除 `lid_warning_confidence`，新增已校验的正整数
   `segment_batch_size`。默认配置与本机校验配置显式保持该值。
5. 模型注册表、下载、完整性验证、doctor、架构文档、测试和 RAMC/发布验收统一从
   `vad/lid/aed/punc` 改为 `vad/aed/punc`，并删除语言覆盖率等 LID gate。

现有 transcript 的业务格式不含 LID 字段，因此文本、token、speaker 轮次及 JSON 输出的
核心结构保持兼容；仅 `diagnostics.json` 的 LID 专属数据移除。

## 批量输出与失败语义

FireRed 官方系统仍在一个调用内完成 VAD、AED 和 Punc。每个 batch 内的 utterance ID
包含原始 VAD 起止时间，最后格式化逻辑按这些时间排序。因此增大 batch 不改变音频时间
轴、文本拼接顺序或 token—speaker 对齐的输入顺序。

批处理失败时，整个录音作业按现有 ASR 错误语义失败；不静默退回单段模式，避免把性能
或输出变化隐藏在同一份结果中。

## 真实 GPU 0 校准

使用已忽略的 RAMC 420 秒录音，关闭 LID、模型加载后先以短音频预热。对 batch
`1, 2, 4, 8` 分别运行三次；每次记录：

- FireRed `process()` 端到端时间，以及 VAD、AED、Punc 的同步计时；
- 本项目 FireRed PID 的峰值显存和 GPU 0 空闲量；
- VAD 段数、文本、时间戳、token 和句子数量。

每个候选必须通过稳定 contract、保持 VAD 边界和规范化全文与 batch 1 基线一致，且没有
更高的错误率。内存不足、CUDA 错误、超时或输出不一致的候选立即淘汰；测试只观察自身
创建的 PID，绝不操作同卡其他项目。通过的候选中，选择端到端中位数最低且峰值显存仍
满足现有 GPU 0 准入余量的值作为新默认；若没有候选优于 1，则保留 1。

测量输出写入 `outputs/measurements/`，不会写入输入语料的 `artifacts/` 目录。

## 验证

- 单元测试覆盖无 LID 的模型路径、响应/contract、配置、诊断和 registry；任何 LID
  字段或模型不应成为运行成功的条件。
- 集成测试覆盖 worker 的 VAD/AED/Punc 完整结果、服务自动启动和 LID 模型目录不存在时
  仍可成功。
- 基准测试覆盖所有 batch 候选、输出基线比较、显存记录和淘汰规则。
- 完整测试套件与真实 GPU 0 校准都必须在提交前通过；若默认 batch 因实测结果改变，再
  以新默认运行一次真实服务验收。

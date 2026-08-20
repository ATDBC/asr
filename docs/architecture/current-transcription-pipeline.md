# 当前录音转录流程

本文记录 FireRedASR2S 迁移后的生产流程。核心原则是三个 GPU 模型阶段严格串行，
主进程永不导入 FireRed 运行时，任何阶段开始前都只读检查物理 GPU 0 的容量。

## 生命周期与数据流

```text
原始音频
  │
  ├─ FFmpeg → 16 kHz / 单声道 / PCM_16
  │
  ├─ GPU 0 准入：≥23 GiB total，≥12 GiB free
  │    └─ .venv-asr / FireRedWorkerClient（主音频）
  │         ├─ FireRedVAD
  │         ├─ FireRedLID
  │         ├─ FireRedASR2-AED
  │         └─ FireRedPunc
  │              ↓
  │       全文、完整 token 时间戳、句级 LID/ASR 置信度
  │
  ├─ FireRed worker 关闭 → 只读采样 GPU 释放
  │
  ├─ GPU 0 准入
  │    └─ .venv / SpeakerDiarizer
  │         ├─ FSMN-VAD
  │         ├─ CAMPPlus embedding
  │         ├─ 全局聚类与标签规范化
  │         └─ overlap detect/separate/声纹匹配准备（可选）
  │
  ├─ SpeakerDiarizer 关闭 → 只读采样 GPU 释放
  │
  ├─ token 与 Speaker 时间段对齐
  │
  ├─ 仅当存在候选 overlap 轨道：
  │    ├─ GPU 0 准入
  │    └─ .venv-asr / 第二个 FireRedWorkerClient
  │         └─ 一个 worker 串行识别全部候选轨道
  │
  ├─ 第二个 FireRed worker 关闭 → 只读采样 GPU 释放
  │
  ├─ 接受或逐窗口降级 overlap 结果
  ├─ 构建并合并说话人轮次
  └─ 所有 GPU owner 关闭后导出 TXT / JSON / diagnostics
```

## FireRed worker 边界

主环境通过严格 JSON Lines 协议调用 `.venv-asr` 中的 worker。每个请求都有唯一 ID，
支持 `transcribe`、`verify` 和 `shutdown`。ID 不匹配、重复响应、非法 JSON、超时或
worker 提前退出都会成为转录错误，不会切换旧 ASR。

worker 固定加载 FireRedASR2S 提交
`4e7d9aaf4482a47cec1724807026b9b151926eb5` 的四模块。官方结果只有句级
`asr_confidence`；稳定协议把所在句的值传给每个 token，不虚构词级分数。标点
附着到带时间戳 token 后，协议会检查全文、句子、token、VAD 和音频时长的一致性。

## Speaker 与 overlap

Speaker 路线独立运行 FSMN-VAD，并在重叠短窗上提取 CAMPPlus embedding；聚类标签
按首次出现顺序规范化为 1、2、3。主 ASR token 以时间覆盖率为主、最近区间为后备
规则对齐。FireRed 成功结果必须有完整 token 时间戳，因此没有局部补时间戳路线。

overlap 分为两个生命周期阶段：

1. Speaker 驻留时执行检测、双路分离、RMS 过滤、质心计算和说话人匹配，不调用 ASR。
2. Speaker 释放后，如有候选轨道，启动一个 FireRed worker 识别所有轨道并回填全局时间。

无候选轨道时 FireRed 只加载一次；有候选轨道时最多加载两次。任一轨道失败只让其
所在窗口保留原 token 并产生 warning，主音频识别失败则整个任务失败。

## GPU 安全与诊断

- 生产固定物理 GPU 0（`cuda:0`），拒绝其他设备或不兼容的可见设备重映射。
- `nvidia-smi` 查询只读；程序从不终止、暂停、重启或迁移其他项目进程。
- 每个主要阶段要求 GPU 总显存至少 23 GiB、空闲显存至少 12 GiB。FireRed
  四模块在 18.39 秒连续语音段上的 `nvidia-smi` 净增峰值实测为 9.04 GiB，
  12 GiB 门槛保留约 3 GiB 运行余量。
- diagnostics 记录准入前/退出后空闲 MiB、worker 峰值显存、加载/推理耗时和
  `released` 判断。Speaker 关闭后允许最多 512 MiB 的 CUDA context 与采样
  误差；本机实测模型释放后残留 261 MiB，进程退出后恢复到原基线。
- 显存不足或监控不可用属于环境错误；不会为了继续执行而降低门槛。

## LID 与文本完整性

句级 FireRedLID 结果按持续时间汇总主语言和语言占比。主语言不是
`zh mandarin`，或任一句语言置信度低于配置阈值时写入 diagnostics warning。
没有 overlap 修改时，最终轮次全文必须与 FireRed 全文在 NFKC 和忽略空白后完全
一致；不一致即失败。

## 主要实现位置

- `src/funasr_transcriber/audio.py`：输入探测和音频归一化。
- `src/funasr_transcriber/firered_worker.py`：隔离四模块运行时和 JSONL 服务。
- `src/funasr_transcriber/firered_client.py`：持久 worker、超时、关联和关闭。
- `src/funasr_transcriber/firered_contract.py`：官方响应规范化与完整性验证。
- `src/funasr_transcriber/gpu.py`：GPU 0 准入和只读释放采样。
- `src/funasr_transcriber/speaker_diarization.py`：Speaker 模型与显式释放。
- `src/funasr_transcriber/overlap.py`：重叠准备和轨道识别应用。
- `src/funasr_transcriber/pipeline.py`：对齐、LID 汇总、轮次和完整性检查。
- `src/funasr_transcriber/cli.py`：三阶段生命周期编排和导出。

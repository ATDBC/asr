# 当前录音转录流程

本文记录 FireRedASR2 迁移之前，独立命令行转录程序的真实处理流程。它是迁移
设计的基线，不代表 FireRedASR2 的目标架构。

## 流程图

```text
原始音频文件
      ↓
FFmpeg 归一化
16 kHz / 单声道 / PCM
      ↓
┌─────────────────────────────────────────────┐
│ ASR 路线（先执行）                          │
│                                             │
│ 整条归一化音频                              │
│      ↓                                      │
│ FunASR AutoModel                            │
│  ├─ 内部 FSMN-VAD                           │
│  ├─ Paraformer ASR                          │
│  └─ 标点模型                                │
│      ↓                                      │
│ 全文 + 句级时间 + token 级时间戳            │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ Speaker 路线（随后执行）                    │
│                                             │
│ 再次读取整条归一化音频                      │
│      ↓                                      │
│ 独立执行 FSMN-VAD                           │
│      ↓                                      │
│ 每个 VAD 区间切成重叠短窗                   │
│ 默认 1.0 秒窗口、0.5 秒步长                 │
│      ↓                                      │
│ CAMPPlus speaker embedding                  │
│      ↓                                      │
│ 全录音谱聚类                                │
│      ↓                                      │
│ 过分聚类合并、标签重排、时间段平滑          │
│      ↓                                      │
│ Speaker 1 / 2 / 3 时间区间                  │
└──────────────────────┬──────────────────────┘
                       ↓
            ASR token 与 Speaker 区间对齐
                       ↓
     按覆盖率归属；覆盖不足时使用最近 Speaker 区间
                       ↓
     无可靠 token 时间戳的句子执行局部重新识别
                       ↓
       可选：重叠检测、双路分离和局部重新识别
                       ↓
                  构建说话人轮次
                       ↓
       同一说话人且间隔不超过 1 秒则合并
                       ↓
       [00:00] 说话人1：……
       [00:05] 说话人2：……
```

## 关键事实

- ASR 和 Speaker 路线没有共享同一份 VAD 结果，两边会分别执行 VAD。
- 主程序对整条归一化音频调用一次 FunASR；Python 编排层没有逐个 VAD 段调用
  ASR，分段和批处理由 FunASR 内部完成。
- Speaker embedding 不是对整个 VAD 段只提取一次，而是对 VAD 段内部的重叠
  短窗逐一提取，再对全录音的 embedding 做聚类。
- 两条路线在算法上可以独立运行，但当前代码是先执行 ASR、再执行 Speaker
  diarization，并未并行。
- 最终对齐的基本单位是带时间戳的 ASR token，不是整句文本。
- Speaker 编号按照聚类标签在录音中的首次出现顺序规范化为 1、2、3。

## 主要实现位置

- `standalone_transcriber/src/funasr_transcriber/audio.py`：输入探测和音频归一化。
- `standalone_transcriber/src/funasr_transcriber/asr.py`：FunASR 结果解析和 token
  时间戳保留。
- `standalone_transcriber/src/funasr_transcriber/speaker_diarization.py`：独立 VAD、
  短窗 embedding、全局聚类和时间段平滑。
- `standalone_transcriber/src/funasr_transcriber/alignment.py`：token 与 Speaker
  时间区间对齐以及局部重新识别。
- `standalone_transcriber/src/funasr_transcriber/overlap.py`：可选重叠检测、分离、
  轨道说话人匹配和双路文本回填。
- `standalone_transcriber/src/funasr_transcriber/turns.py`：相邻同说话人轮次合并。
- `standalone_transcriber/src/funasr_transcriber/pipeline.py`：上述阶段的执行顺序。

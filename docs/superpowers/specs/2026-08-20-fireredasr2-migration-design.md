# FireRedASR2S ASR 路线迁移设计

> **执行状态更新（2026-08-21）**：本文件中的 FireRed 模型、隔离环境和
> 稳定响应契约仍然有效；第 8 节的“每个任务关闭模型”的 GPU 生命周期已由
> [常驻并行转录服务设计](2026-08-21-persistent-parallel-service-design.md)
> 取代。正式 CLI、RAMC 验收和压力验收均通过同一个常驻服务运行；只有显式
> `service stop` 或启用 overlap 的受控卸载路径才会释放热模型。

## 1. 背景与目标

独立命令行转录程序当前使用 FunASR 的 FSMN-VAD、Paraformer 和标点模型完成
ASR，再将带时间戳 token 与 3D-Speaker 说话人时间段对齐。本文定义将整条 ASR
路线替换成 FireRedASR2S 的方案：

```text
FireRedVAD → FireRedLID → FireRedASR2-AED → FireRedPunc
```

迁移后的外部目标保持不变：输入最长两小时的音频文件，输出类似
`[00:00] 说话人1：……` 的完整普通话转录，并保留 JSON、诊断信息、重叠双路
恢复、1 秒同说话人轮次合并和现有命令行用法。

## 2. 已确认决策

- 使用 FireRedASR2-AED，不使用 FireRedASR2-LLM。AED 官方接口提供字词级
  时间戳和置信度，能够满足 Speaker 时间对齐要求。
- 启用 FireRedVAD、FireRedLID、FireRedASR2-AED、FireRedPunc 四个模块。
- FireRedASR2S 运行在独立 `.venv-asr`，主程序通过本地 worker 调用。
- `.venv-asr` 使用 Python 3.11。官方 README 的 Python 3.10 示例与当前包声明
  不一致；本项目以当前 `pyproject.toml` 的 `requires-python >=3.11` 为准。
- 目标服务器为 Linux、单张 24 GB NVIDIA GPU、单任务串行执行。
- 不保留 FunASR ASR 回退。FireRed 失败时任务明确失败，不能静默切回旧模型。
- Speaker 路线继续使用已经验证过的 FSMN-VAD、CAMPPlus 和 3D-Speaker。
- 普通话仍是第一阶段验收范围；LID 的其他语言或方言结果记录为诊断，不作为
  当前任务的硬失败条件。

## 3. 方案比较

### 方案 A：官方完整系统 + 常驻 worker（采用）

在 `.venv-asr` 内加载官方 `FireRedAsr2System`，显式启用 VAD、LID、AED 和
Punc。主程序使用 JSON Lines 协议复用同一 worker。该方案直接利用官方已经
实现的长音频 VAD 分段、LID、标点及全局时间戳拼接。

### 方案 B：独立组合四个模块

自行调用 FireRedVAD、LID、AED 和 Punc，可以精细控制每个阶段，但需要重新
实现 60 秒 AED 上限、分段上下文、标点回填和全局时间戳拼接，维护面更大。

### 方案 C：每次调用官方 CLI

集成代码最少，但主识别、局部识别和重叠轨道都会重复加载模型，无法满足长录音
和多重叠窗口的性能要求。

## 4. 目标数据流

```text
原始音频
    ↓
FFmpeg 归一化：16 kHz / 单声道 / PCM
    ↓
┌─────────────────────────────┴─────────────────────────────┐
↓                                                           ↓
ASR worker 路线                                      Speaker 路线
独立 Python 3.11 环境                               主 Python 3.11 环境
↓                                                           ↓
FireRedVAD                                           独立 FSMN-VAD
↓                                                           ↓
安全长度语音段                                      1.0 秒重叠短窗
↓                                                           ↓
FireRedLID                                           CAMPPlus embedding
↓                                                           ↓
FireRedASR2-AED                                      全局谱聚类与平滑
↓                                                           ↓
字词文本、置信度、时间戳                             Speaker 时间区间
↓                                                           │
FireRedPunc                                                │
↓                                                           │
带标点正文 + 全局字词时间戳                                 │
└─────────────────────────────┬─────────────────────────────┘
                              ↓
                    token—speaker 时间对齐
                              ↓
                可选重叠检测、分离与 FireRed 重识别
                              ↓
              同一说话人间隔不超过 1 秒则合并
                              ↓
                   [00:00] 说话人1：……
```

图中的左右分支表示两类数据产物，不表示并行执行。针对 24 GB 单卡，实际执行按
第 8 节的模型生命周期串行进行。

FireRedASR2-AED 官方单次输入上限为 60 秒。FireRedVAD 的连续语音上限采用官方
示例的 `max_speech_frame=2000`，使 AED 输入明显低于 60 秒。worker 必须拒绝
任何超过 60 秒的实际 AED 子段，不能冒险直接推理。

## 5. 进程边界与 worker 协议

### 5.1 进程边界

主程序不导入 FireRedASR2S、其 PyTorch 或 Transformers 依赖。新增 ASR worker
脚本只在 `.venv-asr` 中运行。主程序启动 worker 后，通过标准输入发送一行一个
JSON 请求，从标准输出读取同请求 ID 的一行 JSON 响应。日志只写标准错误，禁止
污染协议标准输出。

### 5.2 请求

worker 支持以下动作：

- `transcribe`：识别一条已归一化 WAV；参数包含请求 ID 和绝对音频路径。
- `verify`：加载四个模型，检查 CUDA，并对 doctor 音频执行完整四模块推理。
- `shutdown`：正常释放模型并退出。

主进程为每个请求生成唯一 ID。响应 ID 不匹配、重复响应、非法 JSON、worker
提前退出或超时均视为任务失败。

### 5.3 规范化响应

worker 输出以下稳定内部契约，不将官方对象直接暴露给主程序：

```json
{
  "id": "request-id",
  "ok": true,
  "text": "加好标点的完整文本。",
  "tokens": [
    {
      "text": "加",
      "start_ms": 120,
      "end_ms": 280,
      "confidence": 0.97
    }
  ],
  "sentences": [
    {
      "text": "加好标点的完整文本。",
      "start_ms": 120,
      "end_ms": 1800,
      "language": "zh mandarin",
      "language_confidence": 0.99,
      "asr_confidence": 0.95
    }
  ],
  "vad_segments_ms": [[100, 1850]],
  "duration_ms": 2320
}
```

所有浮点秒时间戳在 worker 内四舍五入为整数毫秒。token、sentence 和 VAD 区间
必须起止合法、单调并位于音频时长内。

## 6. 文本、标点和时间戳完整性

FireRedPunc 返回的正文带标点，而 AED 的 `words` 通常不包含标点。适配器按句子
边界将标点附着到前一个字词 token；句首标点附着到后一个 token。比较正文时执行
NFKC 规范化并忽略空白，但不删除正文标点。

每个成功结果必须同时满足：

- 正文非空时，token 列表非空。
- 拼接 token 文本经过相同规范化后与完整正文一致。
- 每个 token 的 `0 <= start_ms < end_ms <= duration_ms`。
- token 按开始时间和结束时间单调排列。
- sentence 正文与其覆盖 token 的正文一致。
- FireRed 返回正文却缺少完整字词时间戳时直接失败，不生成低置信占位 token。

现有 `AsrResult` 和 `TimedToken` 继续作为主程序边界。FireRed 适配器将规范化响应
转换为这些类型，因此 Speaker 对齐、turn 构建和导出格式无需了解模型实现。

## 7. LID 行为

- FireRedLID 在每个 VAD 语音段上启用。
- worker 保留每句 `language` 和 `language_confidence`。
- 主程序按 sentence 有效时长汇总录音主要语言及各标签时长占比。
- LID 信息写入 `diagnostics.json`，不改变 Speaker ID，也不参与聚类。
- 第一阶段普通话验收中，主要标签不是 `zh mandarin` 或低置信度只产生明确警告，
  不直接中止任务。
- 后续方言验收可以直接按 LID 标签分组统计，不需要再次修改 worker 协议。

## 8. 24 GB GPU 模型生命周期

第一版只运行单任务，并将 GPU 模型分阶段驻留：

### 阶段 1：主 ASR

启动 FireRed worker，加载 VAD、LID、AED、Punc，识别整条录音，返回完整结果后
关闭 worker并确认进程退出，释放 FireRed 显存。

### 阶段 2：Speaker diarization

加载 3D-Speaker，执行独立 VAD、短窗 embedding、全局聚类和说话人中心计算。

### 阶段 3：可选重叠处理

1. pyannote 检测重叠窗口。
2. ClearVoice 分离双路。
3. 仍由 3D-Speaker 计算分离轨道 embedding，并匹配全局说话人中心。
4. 释放 3D-Speaker 和重叠模型。
5. 若存在通过声纹匹配和能量检查的轨道，重新启动一次 FireRed worker，在同一
   worker 会话中依次识别所有轨道并回填全局时间线，然后关闭 worker。

无重叠窗口时 FireRed 只加载一次；有重叠窗口时最多加载两次。禁止为每个轨道
分别启动 worker。

`doctor` 要求 GPU 总显存至少 23 GiB，并在每个主要 GPU 阶段开始前要求至少
12 GiB 可用显存。该门槛由 GPU 0 实测校准：四模块处理一个包含 18.39 秒连续
语音段的 22.016 秒官方样本时，本项目的 `nvidia-smi` 净增峰值为 9,256 MiB
（9.04 GiB），worker 报告的 PyTorch 峰值为 9,048,353,280 bytes；12 GiB
保留约 3 GiB 余量。显存被其他任务占用时立即失败并报告当前占用，不尝试进入
可能 OOM 的推理。诊断记录每阶段模型加载耗时、推理耗时、峰值 GPU 显存和
退出后的显存释放状态。

释放判断允许退出后空闲显存相对准入前最多减少 512 MiB，用于容纳同一主进程中
无法卸载的 PyTorch CUDA context 和采样误差。Speaker 模型关闭后的实测差值为
261 MiB，doctor 进程退出后恢复到原基线；超过 512 MiB 仍视为模型未正确释放。

## 9. 安装、源码和模型布局

```text
standalone_transcriber/
├── .venv/               # 主程序与 3D-Speaker，Python 3.11
├── .venv-asr/           # FireRedASR2S，Python 3.11
├── .venv-overlap/       # pyannote 与 ClearVoice
└── models/
    └── fireredasr2s/
        ├── source/
        ├── FireRedVAD/
        ├── FireRedLID/
        ├── FireRedASR2-AED/
        └── FireRedPunc/
```

官方源码固定到提交 `4e7d9aaf4482a47cec1724807026b9b151926eb5`。安装脚本
必须校验实际 HEAD，不能静默使用更新版本。`.venv-asr` 按该提交的包声明安装，
与主环境和 overlap 环境的 PyTorch 依赖完全隔离。

overlap 环境固定使用 `torch==2.8.0`、`torchaudio==2.8.0` 和配套的
`torchvision==0.23.0`（CUDA 12.8 wheel），配合 `pyannote.audio==3.3.2`。
不得将 overlap 环境无条件升级到
torchaudio 2.9 及以上：这些版本移除了 pyannote 3.3.2 仍使用的
`AudioMetaData`、`info` 和音频 backend API。安装测试必须同时检查版本约束和
真实 `pyannote.audio` 导入，避免仅由 `pip check` 无法发现的运行时 API 冲突。

overlap 模型默认从 ModelScope 镜像获取：`pyannote/speaker-diarization-3.1`、
`pyannote/segmentation-3.0`、`pyannote/wespeaker-voxceleb-resnet34-LM` 和
`alibabasglab/MossFormer2_SS_16K`。主环境将它们原子下载到
`models/huggingface/` 下的固定目录；worker 用绝对本地 checkpoint 生成临时
pyannote pipeline 配置，不再访问 Hugging Face，也不要求 token。本地资产不完整
时才使用原有 Hugging Face 路线并要求 `HF_TOKEN`。ClearVoice 初始化后必须检查
`last_best_checkpoint` 指向的权重存在，不能把上游吞掉下载异常后的空模型当成功。
所有 overlap worker 子进程都设置 `CUDA_VISIBLE_DEVICES=0`，使 ClearVoice 的
逻辑 `cuda:0` 始终映射到物理 GPU 0。

默认使用官方推荐的 ModelScope 模型：

- `xukaituo/FireRedVAD`
- `xukaituo/FireRedLID`
- `xukaituo/FireRedASR2-AED`
- `xukaituo/FireRedPunc`

同时提供对应 `FireRedTeam/*` Hugging Face 仓库作为显式备用下载源。下载采用临时
目录，权重完整校验成功后原子发布到最终模型目录。安装和卸载脚本不得自动删除
已有输出或用户配置。

## 10. 失败处理

- 模型、源码、CUDA 或 Python 环境缺失：启动前失败。
- VAD 无语音：返回明确的无语音错误，不生成空转写成功结果。
- AED 子段超过 60 秒：拒绝该任务并报告区间。
- worker 超时或退出：终止任务，清理 worker 和临时文件。
- LID 低置信或非普通话：记录警告，继续识别。
- Punc 结果无法与 AED 字词对齐：任务失败。
- 正文和 token 完整性不一致：任务失败。
- GPU 显存不足或无法有效监控：正式任务失败。
- 重叠增强失败：维持现有策略，保留基础 ASR token 并记录降级警告；基础 ASR
  失败则绝不降级。

## 11. 测试策略

### 11.1 无真实模型测试

- FireRed 官方响应到稳定 worker 响应的转换。
- 秒到毫秒的四舍五入、边界和单调性。
- 中文、英文、数字、空格和标点回填。
- LID 标签、置信度和按时长汇总。
- 正文缺 token、时间戳越界、乱序、句子不一致时失败。
- JSON Lines 请求 ID、超时、崩溃、非法 JSON 和 shutdown。
- 主 ASR、Speaker、重叠三阶段的模型加载和释放顺序。
- 安装脚本、源码提交、模型目录和 Linux 命令静态验证。
- 生产 ASR 路线不再导入 FunASR `AutoModel`。

### 11.2 Linux/NVIDIA 真实模型测试

- doctor 音频经过 VAD、LID、AED、Punc 后得到正文和完整字词时间戳。
- 20 条 RAMC 普通话核心录音全部完成。
- 30 条合成重叠、10 条真实标记观察、10 条正常换人负样本全部执行。
- WAV、MP3、M4A、FLAC、AAC、OGG 的 48 种采样率与声道组合全部输出四件套。
- 30、60、120 分钟压力录音各运行两次。
- 每个 GPU 阶段获得有效显存样本，worker 退出后显存和临时文件得到释放。

## 12. 发布门槛

- 20 条 RAMC 全部完成。
- 基础 ASR 正文到最终 turns 的文字保留率为 100%。
- 话轮 purity 不低于 0.92。
- DER-like 不高于 0.22。
- 说话人数 MAE 不高于 0.25，准确率不低于 0.80。
- cpCER 相比全文 CER 的增量不高于 0.05。
- 低置信文字占比不高于 0.05。
- 合成重叠恢复率和正常换人误报率继续使用现有发布门槛。
- 两小时录音两次结果哈希一致。
- FireRed、LID、时间戳、GPU 监控或任何必需阶段缺失时，汇总不得标记通过。

实现完成后重新构建 `artifacts/release_validation_corpus`，确保包内 Linux runner
使用 FireRedASR2S 环境和模型检查。生成语料继续保持 Git 忽略。

## 13. 官方依据

- FireRedASR2S 官方仓库：<https://github.com/FireRedTeam/FireRedASR2S>
- FireRedASR2-AED 官方 Python 示例包含 `return_timestamp=True`，并返回字词、
  起止时间和置信度。
- 官方完整系统返回 `text`、`sentences`、`vad_segments_ms` 和 `words`。
- 官方 FAQ 规定 FireRedASR2-AED 单段输入不超过 60 秒。
- 当前官方 `pyproject.toml`：
  <https://github.com/FireRedTeam/FireRedASR2S/blob/main/pyproject.toml>

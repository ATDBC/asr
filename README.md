# Standalone FireRedASR2S Transcriber

Linux 上的独立普通话录音转录工具。输入音频，输出带毫秒时间戳和匿名说话人编号的
TXT、结构化 JSON 与诊断文件。

默认流程为：3D-Speaker 说话人分离 → 对超过 10 秒的说话人区间使用 FireRedVAD
细分 → FireRed AED → FireRedPunc。AED 可配置一个或两个常驻 worker，并由一个独立 Punc
owner 处理标点；使用物理 GPU 0（`cuda:0`）与 BF16。Nemotron 3 是显式选择的可选说话人后端，
不会取代默认 3D-Speaker。

## 1. 使用前确认

- Linux、Python 3.11、Git、FFmpeg、NVIDIA GPU。启动前要求 GPU0 总显存至少 23 GiB；
  Nemotron 与常驻 FireRed 共驻时，还要求当前空闲显存至少 13,312 MiB。
- 只支持物理 GPU 0，不要用 `CUDA_VISIBLE_DEVICES` 把其他 GPU 重映射成
  `cuda:0`。
- 支持 WAV、MP3、M4A、FLAC、AAC、OGG，单个输入最长两小时。
- 程序只管理本项目登记的进程；显存不足时会报错，不会尝试终止或清理其他项目的进程。

先检查基础环境：

```bash
python3.11 --version
ffmpeg -version
ffprobe -version
nvidia-smi --id=0 --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
```

## 2. 克隆仓库

```bash
git clone git@gitlab.hmccloud.net:wujiajun/standalone_transcriber.git
cd standalone_transcriber
```

安装脚本默认从 `PATH` 查找 `python3.11`。如果机器上的 Python 3.11 位于其他路径，
先显式指定：

```bash
export PYTHON_BIN=/absolute/path/to/python3.11
"$PYTHON_BIN" --version
```

## 3. 安装

默认把模型放在仓库的 `models/`：

```bash
export FUNASR_TRANSCRIBER_MODEL_ROOT="$PWD/models"
chmod +x scripts/install_linux.sh
./scripts/install_linux.sh
```

默认使用 CUDA 12.8 的 PyTorch wheel。需要其他官方 wheel 源时，在安装前设置：

```bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
```

安装器创建三个隔离环境：

- `.venv`：主 CLI、3D-Speaker 和调度逻辑。
- `.venv-asr`：FireRedVAD、FireRed AED、FireRedPunc。
- `.venv-overlap`：可选的 pyannote/ClearVoice 重叠语音处理。

FireRedASR2S 源码固定到提交
`4e7d9aaf4482a47cec1724807026b9b151926eb5`。安装器不会修改系统 Python，也不会
覆盖提交不匹配的已有 FireRed 源码目录。

后续命令可以直接使用 `.venv/bin/...`，无需激活虚拟环境。

## 4. 创建本机配置

不要直接修改仓库默认配置。创建一个本机专用、被 Git 忽略的配置：

```bash
mkdir -p outputs/runtime
cp config/default.toml outputs/runtime/config.toml
```

至少确认以下配置使用绝对路径：

```toml
[general]
device = "cuda:0"

[paths]
model_dir = "/absolute/path/to/standalone_transcriber/models"
work_dir = "/absolute/path/to/standalone_transcriber/outputs/runtime/work"

[asr]
worker_python = "/absolute/path/to/standalone_transcriber/.venv-asr/bin/python"
use_half = true
aed_worker_count = 2
aed_scheduler = "weighted_duration"
aed_weight_artifact = ""

[overlap]
worker_python = "/absolute/path/to/standalone_transcriber/.venv-overlap/bin/python"
```

`aed_weight_artifact` 先留空。完成第 6 节的新设备权重初始化后，再填入生成文件的
绝对路径。

配置优先级为：

```text
命令行 > FUNASR_TRANSCRIBER_* 环境变量 > TOML > 内置默认值
```

## 5. 下载并验证模型

```bash
.venv/bin/funasr-transcribe models download \
  --config "$PWD/outputs/runtime/config.toml"

.venv/bin/funasr-transcribe models verify \
  --config "$PWD/outputs/runtime/config.toml"

.venv/bin/funasr-transcribe doctor \
  --config "$PWD/outputs/runtime/config.toml"
```

默认模型布局：

```text
models/fireredasr2s/
├── source/
├── FireRedVAD/VAD/
├── FireRedASR2-AED/
└── FireRedPunc/
```

默认下载只包含 FireRedVAD/VAD、FireRedASR2-AED、FireRedPunc 的 PyTorch
权重、FSMN-VAD 和 CAMPPlus；不下载语言识别权重、FireRedVAD 的其他变体或
FireRedPunc 的 TensorFlow 权重。`overlap.enabled = false` 时也不会下载
pyannote/ClearVoice 模型，只有显式启用 overlap 后才按需获取。

模型默认从 ModelScope 获取。只有对应资产无法从 ModelScope 获取时，才使用
Hugging Face fallback；token 只从环境变量读取，不写入配置或日志。

`doctor` 会实际检查主环境、FireRed worker、说话人分离和 overlap 环境。它失败时先
修复对应问题，不要用 `--skip-cuda` 作为生产通过证据。

### 5.1 可选 Nemotron 3 说话人后端

默认安装和默认 `models download` **不会**安装或下载 Nemotron。只有明确选择该后端时，
才执行下面的额外步骤。其隔离环境只为 Nemotron 服务，不改变 .venv、.venv-asr 或默认
3D-Speaker 路径：

```bash
bash scripts/install_nemotron_linux.sh
```

此脚本沿用 `PYTHON_BIN`（Python 3.11）和 `TORCH_INDEX_URL`，创建
`.venv-nemotron`，固定 PyTorch 2.8.0、Transformers 提交
`f324707307757d9c0b8dac1c4462eceff911fa2f`。模型固定为 Hugging Face
`nvidia/Nemotron-3-Diarization` 的修订版
`a435e9867d79e789e90053f9b6d6834053af564a`；该可选资产不是从
ModelScope 下载。只有下面显式选择的 `models download` 才会获取快照，并验证
checkpoint SHA-256；运行时只读本地快照、禁止联网下载。

在专用 `outputs/runtime/nemotron.toml` 中复制上述本机配置，保留绝对路径，并设置：

```toml
[diarization]
backend = "nemotron3"
speaker_count = "auto"

[nemotron]
worker_python = "/absolute/path/to/standalone_transcriber/.venv-nemotron/bin/python"
activity_threshold = 0.5

[asr]
aed_scheduler = "weighted_duration"
aed_worker_count = 1  # 也可用 2；不能选择 memory_first

[overlap]
enabled = false
```

`[paths].model_dir`、`[paths].work_dir` 和 `[asr].worker_python` 仍需指向实际
绝对路径。使用选中配置显式下载、验证和冒烟检查：

```bash
.venv/bin/funasr-transcribe models download --config "$PWD/outputs/runtime/nemotron.toml"
.venv/bin/funasr-transcribe models verify --config "$PWD/outputs/runtime/nemotron.toml"
.venv/bin/funasr-transcribe doctor --config "$PWD/outputs/runtime/nemotron.toml"
.venv/bin/funasr-transcribe /data/recording.wav --config "$PWD/outputs/runtime/nemotron.toml"
```

Nemotron 最多提供八个说话人通道，不能可靠发现未被模型观察到的第九人。每段说话人区间
都从**原始归一化混合音频**单独裁剪给 AED；例如 0–6 秒与 5–8 秒会产生两份输入，
5–6 秒重叠声音在两份中都保留。这里不做 ClearVoice 波形分离、重叠去重或重新匹配
说话人。`--enable-overlap` 与 Nemotron 互斥。短于 100 ms 的预测区间会跳过；
超过 10 秒的区间先尝试 FireRedVAD，失败或不安全时固定切为不超过 10 秒的片段，
不是模型的 30 秒上限。JSON/TXT 可以保留同时发生的两条说话人转写，全文可能含重复词。
这是可选验证路径，不代表生产准确率已获验收。

需要一次 GPU0 验收时，设置一个**从未使用过**的输出目录，显式执行：

```bash
RUN_NEMOTRON_GPU_TEST=1 \
NEMOTRON_BENCHMARK_CONFIG="$PWD/outputs/runtime/nemotron.toml" \
NEMOTRON_BENCHMARK_OUTPUT_ROOT="$PWD/outputs/measurements/nemotron-direct-crop-unique-run" \
.venv/bin/python -m pytest -q tests/integration/test_nemotron_gpu_smoke.py
```

该入口只运行本项目的 GPU0 作业，并在新目录保留设备准入/结束快照、各阶段耗时、
本项目 owner 的显存峰值和转录结果。真实重叠片段的人工文字只在推理结束后用于观察性
对照，不参与训练或调参；粗粒度参考时间不支持宣称生产准确率通过。

要回退，先用**当前配置**停止本项目服务，再改用原来的 `backend = "3d_speaker"`
配置重新启动。不要关闭或管理其他项目进程。

### 5.2 可选 Hybrid refinement

Hybrid 路径保留 3D-Speaker 作为全局说话人 ID 空间，同时用 Nemotron 补充局部短插话。
它不是 `backend = "nemotron3"` 的 direct-crop 路径；配置继续使用默认
`backend = "3d_speaker"`，并显式执行：

```bash
.venv/bin/funasr-transcribe /data/recording.wav \
  --config "$PWD/outputs/runtime/config.toml" \
  --enable-nemotron-interruption
```

对超过 10 秒的输入，服务会在 3D-Speaker 和 Nemotron 推理期间并行预取一次全局
FireRedVAD。Hybrid refinement 完成后，正常 `refine_vad` 请求复用 worker-0 中按音频路径
缓存的全局结果，只执行 speaker 区间求交；三路全部完成后才冻结 AED 计划。该优化不改变
VAD 区间、AED item 或 AED 计划。LLM 纠错本身可能在重复请求间返回不同 patch，因此最终文本
不承诺逐字相同。短于或等于 10 秒的输入不启动预取。

## 6. 在新设备上初始化 AED 两个时间权重

weighted-duration 调度只需要两个设备相关权重：

```text
预计耗时 = 最长片段秒数 × (longest_weight + padded_weight × 片段数)
```

初始化命令只计算这两个权重。它不估计显存上限、双 worker 争用、VAD 或 Punc
耗时，也不会自动修改配置或启动转录服务。

### 6.1 准备已切分片段清单

清单是 JSON 文件，入口为 `metadata.source_inventory`。每个源录音包含原始音频路径
和已经确定的片段边界；不需要转录文本或标签。

```json
{
  "metadata": {
    "source_inventory": [
      {
        "audio_path": "meeting-001.wav",
        "source_recording_group": "meeting-001",
        "source_checksum": "<音频文件的 sha256>",
        "segments_checksum": "<segments 的 canonical JSON sha256>",
        "segments": [
          {"index": 0, "start_ms": 1200, "end_ms": 5400},
          {"index": 1, "start_ms": 6100, "end_ms": 13800}
        ]
      }
    ]
  }
}
```

清单必须满足：

- 至少 10 个不同的 `source_recording_group`。
- 每个 group 名称唯一，每个 group 内 `index` 唯一。
- `start_ms >= 0` 且 `end_ms > start_ms`；单个片段不超过 200 秒。
- 自动生成的有效 case 至少覆盖 4 种片段数，其中必须包含 1，最大至少为 16。
- 为覆盖生产的固定边界，建议部分源录音至少提供 32 个代表性片段。
- 音频与 `segments` 的 SHA-256 必须匹配；数据应代表该设备的实际负载。

相对 `audio_path` 会相对于 `--audio-root` 解析；绝对路径不需要
`--audio-root`。`segments_checksum` 使用排序键、无额外空格的 UTF-8 JSON 计算，例如：

```bash
.venv/bin/python - <<'PY'
import hashlib, json

segments = [
    {"index": 0, "start_ms": 1200, "end_ms": 5400},
    {"index": 1, "start_ms": 6100, "end_ms": 13800},
]
payload = json.dumps(
    segments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
```

### 6.2 运行初始化

确保物理 GPU 0 可用，然后执行：

```bash
.venv/bin/python tools/calibrate_weighted_aed.py \
  --manifest /data/calibration/segments-plan.json \
  --audio-root /data/calibration/audio \
  --model-dir "$PWD/models" \
  --worker-python "$PWD/.venv-asr/bin/python" \
  --output "$PWD/outputs/device-calibration"
```

校准固定使用 `cuda:0`、一个 AED worker、生产精度，并关闭 VAD/Punc。每个 case 先
执行一次不参与拟合的预热，再记录 3 次正式测量。它只读取音频和片段边界，不读取
validation labels、blind holdout 或其他标签。

输出目录使用 create-only/append-only 语义，包含：

```text
outputs/device-calibration/
├── plan.json
├── records.jsonl
├── calibration-report.json
├── weighted-aed-calibration.json
└── firered-calibration.log
```

正常中断后可以续跑同一计划：

```bash
.venv/bin/python tools/calibrate_weighted_aed.py \
  --manifest /data/calibration/segments-plan.json \
  --audio-root /data/calibration/audio \
  --model-dir "$PWD/models" \
  --worker-python "$PWD/.venv-asr/bin/python" \
  --output "$PWD/outputs/device-calibration" \
  --resume
```

已完成 case 不会重跑；已经写入的失败记录不会被覆盖或忽略。发生失败时保留现场，
修复原因后使用新的输出目录重新开始。

### 6.3 启用生成的权重

先检查结果为 ready：

```bash
jq '{status, weights, runtime_identity, fit, evidence}' \
  outputs/device-calibration/weighted-aed-calibration.json
```

再修改 `outputs/runtime/config.toml`：

```toml
[asr]
aed_scheduler = "weighted_duration"
aed_weight_artifact = "/absolute/path/to/standalone_transcriber/outputs/device-calibration/weighted-aed-calibration.json"
```

服务启动时会严格校验设备、精度、GPU 型号、计算能力、FireRed 源码提交和 AED
checkpoint。严格身份不匹配时会记录 warning 并回退到内置权重；driver、CUDA 或
Torch 版本变化只记录 warning。权重在运行期间不会在线更新。

## 7. 第一次转录

```bash
.venv/bin/funasr-transcribe /data/recording.mp3 \
  --config "$PWD/outputs/runtime/config.toml"
```

指定输出目录：

```bash
.venv/bin/funasr-transcribe /data/recording.mp3 \
  --config "$PWD/outputs/runtime/config.toml" \
  --output-dir /data/results/recording
```

首次请求会自动启动本项目后台服务。首次模型加载不计入作业耗时，但首次代表性预热
计入第一个作业；每个作业结束前的两个 AED worker cache 清理也计入端到端耗时。

查看或停止本项目服务：

```bash
.venv/bin/funasr-transcribe service status \
  --config "$PWD/outputs/runtime/config.toml"

.venv/bin/funasr-transcribe service stop \
  --config "$PWD/outputs/runtime/config.toml"
```

`service stop` 只停止本项目登记的服务和 worker，不会操作其他项目进程。

## 8. 输出与诊断

默认输出：

```text
recording_transcript/
├── transcript.txt
├── transcript.json
└── diagnostics.json
```

诊断级别：

- `off`：最小成功/失败摘要。
- `basic`：默认；阶段耗时、计数和调度摘要。
- `resources`：增加本项目 owner 的 GPU/RSS 峰值。
- `deep`：增加完整 scheduler、逐 batch trace 和 AED 显存时序。

例如：

```bash
.venv/bin/funasr-transcribe /data/recording.mp3 \
  --config "$PWD/outputs/runtime/config.toml" \
  --diagnostics deep
```

深度诊断写入 `diagnostics/scheduler.json` 和
`diagnostics/batch-events.jsonl`。资源采样只关联本项目登记的 PID。

重叠增强默认关闭；需要时显式增加 `--enable-overlap`。启用后会在受控生命周期内
关闭并重新加载本项目常驻模型，以运行 pyannote/ClearVoice。一个 overlap 作业中
FireRed 最多加载两次：常驻主 worker，以及确有接受轨道时启动的临时 worker。

## 9. 常见问题

### 找不到 Python 3.11

使用绝对路径重新安装：

```bash
export PYTHON_BIN=/absolute/path/to/python3.11
./scripts/install_linux.sh
```

### CUDA 不可用

分别检查三个环境：

```bash
.venv/bin/python -c 'import torch; print(torch.cuda.is_available())'
.venv-asr/bin/python -c 'import torch; print(torch.cuda.is_available())'
.venv-overlap/bin/python -c 'import torch; print(torch.cuda.is_available())'
```

### GPU 0 显存不足

只做设备级检查：

```bash
nvidia-smi --id=0 --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
```

等待 GPU 0 自然释放后重试。不要检查、暂停、终止或迁移其他项目进程。
旧串行实现的准入门槛为 12 GiB，FireRed 四模块的历史净增峰值为 9.04 GiB；它们
不是当前两个 AED worker 常驻架构的准入依据，不要据此降低当前门槛。

### 没有生成权重 artifact

查看 `firered-calibration.log`、`records.jsonl` 和 `calibration-report.json`。如果有失败记录，保留原目录
用于诊断，修复后换一个新的 `--output` 目录；不要删除或覆盖旧记录。

### 未配置或无法加载权重 artifact

服务会记录 warning，并使用版本内置权重继续运行。该回退可以保证可用性，但新设备
应先完成一次第 6 节的初始化。

### 输出目录已存在

程序默认拒绝覆盖。确认目标后显式增加 `--overwrite`。

## 10. 开发验证

```bash
.venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_firered_client.py
.venv/bin/python -m pytest -q
git diff --check
```

退出码：`0` 成功，`2` 参数或配置错误，`3` 输入错误，`4` 环境/GPU/模型错误，
`5` 主转录失败，`6` 输出写入失败，`130` 用户中断。

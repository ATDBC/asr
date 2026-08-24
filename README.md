# Standalone FireRedASR2S Transcriber

独立的 Linux 命令行录音转录工具。输入以普通话为主的音频，输出带毫秒时间戳、
匿名说话人编号的 TXT、结构化 JSON 与诊断文件。生产 ASR 使用
FireRedASR2S；不存在旧 FunASR ASR fallback。

## 支持范围

- Linux、Python 3.11、NVIDIA GPU。
- 固定使用物理 GPU 0（`cuda:0`）。
- GPU 总显存至少 23 GiB；常驻服务热加载前至少 13,312 MiB 空闲。
- WAV、MP3、M4A、FLAC、AAC、OGG，最长两小时。
- 自动估计 1～10 位说话人；服务按 FIFO 串行消费作业。单个作业先完成说话人分离，
  其段落直接作为 ASR 切分。
- 生产转录以说话人分离段落经 FireRed AED → Punc 识别，默认启用 BF16（3090 上为
  `torch.bfloat16`）；不加载语言识别模型。
  doctor 与 overlap 重解码仍使用 FireRed VAD。

本工具不依赖 FastAPI、MySQL、原项目 `.env`、讯飞接口或声纹角色识别。

## 系统准备

~~~bash
python3.11 --version
ffmpeg -version
ffprobe -version
nvidia-smi
git --version
~~~

NVIDIA 驱动必须支持安装脚本选择的 CUDA PyTorch。默认 wheel 源为 cu128；如需
使用其他官方源，安装前设置：

~~~bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
~~~

安装、doctor 和转录都只读取 GPU 状态。显存不足时程序会失败并报告当前数值，
不会尝试终止或清理其他项目的进程，也不会暂停、重启其他任务。请等待 GPU 0
自然释放到门槛以上后重试。

## 安装

安装脚本默认把模型放在仓库的 `models/`。如果配置准备使用其他可写目录，先让
安装器使用同一路径：

~~~bash
cd standalone_transcriber
export FUNASR_TRANSCRIBER_MODEL_ROOT="$PWD/models"
chmod +x scripts/install_linux.sh
./scripts/install_linux.sh
source .venv/bin/activate
~~~

脚本创建三个互相隔离的 Python 3.11 环境：

- `.venv`：主 CLI、3D-Speaker、FSMN-VAD 和 CAMPPlus。
- `.venv-asr`：固定源码提交的 FireRedASR2S worker。
- `.venv-overlap`：pyannote OSD 与 ClearVoice。

主环境使用 PyTorch 2.11；overlap 环境固定使用 PyTorch/torchaudio 2.8 和
配套的 torchvision 0.23，兼容 `pyannote.audio==3.3.2` 所需的旧 torchaudio
音频 API。两个环境均从配置的 CUDA 12.8 wheel 源安装，依赖不会交叉导入。
overlap 环境只安装 worker 所需依赖，不安装主项目 editable 包，以避免主项目的
NumPy/音频依赖覆盖 overlap 的兼容性锁定。

FireRedASR2S 固定到提交
`4e7d9aaf4482a47cec1724807026b9b151926eb5`。安装器验证已有源码的 HEAD；
目录不完整或提交不符时直接失败，不覆盖已有内容。不使用系统 site-packages，
也不修改系统 Python。

## 配置

复制默认配置，并确保路径和安装位置一致：

~~~bash
cp config/default.toml /etc/funasr-transcriber.toml
~~~

~~~toml
[general]
device = "cuda:0"

[paths]
model_dir = "/path/to/standalone_transcriber/models"
work_dir = "/var/tmp/funasr-transcriber"

[asr]
worker_python = "/path/to/standalone_transcriber/.venv-asr/bin/python"
timeout_seconds = 1800
use_half = true                 # GPU 支持时由 FireRed 选择 BF16
aed_worker_count = 2            # 两个 AED；worker-0 独占 VAD/Punc
gpu_safety_margin_mib = 1536       # 预热后空闲显存保留的安全余量

[service]
queue_limit = 16
startup_timeout_seconds = 180
profile_sampling_interval_ms = 200
minimum_total_mib = 23552
minimum_free_mib = 13312

[overlap]
worker_python = "/path/to/standalone_transcriber/.venv-overlap/bin/python"
~~~

如果设置 `CUDA_VISIBLE_DEVICES`，首个可见设备必须仍是物理 GPU 0；不支持把物理
GPU 1 重映射成逻辑 `cuda:0`。配置优先级为：

~~~text
命令行 > FUNASR_TRANSCRIBER_* 环境变量 > TOML > 内置默认值
~~~

AED 不再配置固定 batch 数量、帧数或静态显存上限。说话人分离/VAD 完成且两个 AED 已预热后读取
GPU 0 空闲显存，用 `free_mib - gpu_safety_margin_mib` 得到共享瞬时预算；memory-first 装箱器按
片段时长、padding 后帧数和实际 worker 常驻基线，逐个决定每个 batch 的具体片段，直到再加入片段会超出预算。
batch 数量和每个 batch 的 cardinality 都是计算结果。完整计划、实际观测和预测误差写入
`diagnostics.json`，执行期间不重排或重新分 batch。

生产默认常驻两个 AED worker，但只加载一个 Punc 模型。说话人分离完成后，worker-0 对全部
speaker 区间执行一次全局 FireRedVAD；主进程随后生成一份完整 batch 计划，并按预测耗时用
LPT 将完整 batch 分到两个 AED。两个 worker 不重新分组，结果按原始片段索引合并后再交给
worker-0 的单个 Punc 模型。`aed_worker_count = 1` 可回退到单 worker；当前 24 GiB GPU
必须通过现有 GPU 空间准入检查后才会启动两路 AED。AED 开始前会再次只读检查 GPU 0：预算足够时并发，
预算只能容纳单路时按固定计划串行；若连一个计划批次也容纳不下则在 AED 前拒绝该作业，不会触碰其他项目进程。

每个 memory-first 批次都会返回预计/实际耗时、预计/实际增量显存（worker 预热后的 PyTorch allocated 基线）和误差，
并汇总到 `diagnostics.json`，因此可以直接验证“按音频段长度预测”的准确性。`100 MiB` 不作为默认余量；
CUDA allocator 和其他用户显存变化需要更大的安全空间，当前默认 `1536 MiB`。

服务启动只负责加载模型；服务生命周期内的第一个作业会执行代表性 AED/Punc 预热，并把预热计入该
作业耗时。每个作业完成结果组装后会并行执行两个 AED worker 的
`gc.collect() + cuda.synchronize() + empty_cache() + cuda.synchronize()`，确认完成后才结束
端到端计时；独立 Punc owner 不参与这项逐任务清理。`diagnostics.json` 的 `stage_seconds` 记录
`warmup`（仅首任务）、`cache_cleanup`、`output`、`end_to_end_after_load` 以及 diarization、
VAD、AED 计划/推理、Punc、对齐；端到端包含正式输出的持久化与目录发布，
`end_to_end_after_warmup` 暂时作为同值兼容别名。启用
`--profile-resources` 时，`resource_profile` 还记录 GPU 总占用峰值、
本服务登记 PID 峰值，以及减去作业开始时其他项目基线后的 `net_service_peak_mib`。

生产默认 `use_half = true`。FireRed 上游在 GPU 支持 BF16 时将该选项解析为
`torch.bfloat16`；不支持 BF16 的设备才会退化到上游的 FP16 路径。设为 `false` 可恢复
FP32。当前物理 GPU 0 为 RTX 3090，生产路径实际使用 BF16。

## 模型准备

FireRed 模型默认从 ModelScope 获取，目录如下：

~~~text
models/fireredasr2s/
├── source/                 # 固定 commit 的 FireRedASR2S
├── FireRedVAD/VAD/
├── FireRedASR2-AED/
└── FireRedPunc/
~~~

overlap 模型也默认从 ModelScope 获取，保存在 `models/huggingface/`：

~~~text
pyannote/speaker-diarization-3.1/
pyannote/segmentation-3.0/
pyannote/wespeaker-voxceleb-resnet34-LM/
checkpoints/MossFormer2_SS_16K/
~~~

默认下载和验证：

~~~bash
funasr-transcribe models download --config /etc/funasr-transcriber.toml
funasr-transcribe models verify --config /etc/funasr-transcriber.toml
funasr-transcribe doctor --config /etc/funasr-transcriber.toml
~~~

使用上述完整 ModelScope 镜像时，pyannote 重叠检测不需要 Hugging Face token。
只有本地 overlap 资产不完整、显式切换到 Hugging Face fallback 时，才需要先接受
`segmentation-3.0` 与 `speaker-diarization-3.1` 的访问条件。`HF_TOKEN` 只从环境
读取，不写入配置、日志或 diagnostics。

仅当 ModelScope 无法获取 FireRed VAD/AED/Punc 资产时，才显式使用 Hugging Face fallback：

~~~bash
.venv-asr/bin/python -c \
  'from pathlib import Path; from funasr_transcriber.model_registry import acquire_firered_assets; acquire_firered_assets(Path("/path/to/models"), source="huggingface")'
~~~

fallback 仍会校验三个模型目录，3D-Speaker 相关资产继续由 ModelScope 获取。不要
混用不完整目录。

`doctor` 对内置 `hello_zh.wav` 依次执行说话人分离阶段的 FireRedVAD、FireRed
AED/Punc worker 全链推理；关闭 worker 后检查显存释放，再按顺序
验证 Speaker 和 overlap 环境。`--skip-cuda` 只用于环境型测试，会明确显示模型
smoke 未运行，不能作为发布通过证据。

## 转录

~~~bash
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml
~~~

可选参数：

~~~bash
# 指定输出
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml \
  --output-dir /data/results/recording

# 保留本项目中间文件
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml \
  --keep-intermediate

# 显式启用重叠增强（默认关闭）
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml \
  --enable-overlap

# 为本次作业记录服务自身的资源档案
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml \
  --profile-resources

# 深度性能诊断：资源采样、完整调度计划和逐 batch trace
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml \
  --diagnostics deep
~~~

首次转录会自动启动本机后台服务。服务在 `work_dir/service/` 保存自己的 Unix socket、
状态和日志，并常驻两个 FireRed AED worker、一个独立 FireRedPunc owner 与一个 3D-Speaker owner。FireRedVAD
功能上属于说话人分离阶段，但目前由同一个 `.venv-asr` worker 进程承载。普通作业先跑说话人分离、
再以其段落做 ASR，多个 CLI 请求按 FIFO 排队；每次 CLI 返回不会释放热模型。

```bash
# 查看本项目服务是否 ready
funasr-transcribe service status --config /etc/funasr-transcriber.toml

# 停止接收新作业、排空已接收队列并释放本项目模型
funasr-transcribe service stop --config /etc/funasr-transcriber.toml
```

只有 `service stop` 会关闭本项目的服务和其 FireRed 子进程。它不会也不能影响其他项目。
overlap 默认关闭；显式 `--enable-overlap` 时，服务先完成主分支、关闭两项热模型并确认释放，
才启动 pyannote/ClearVoice；完成后自动热加载。主 ASR 失败会返回退出码 5，不切换旧模型；
单个重叠窗口失败只保留该窗口原文并写入 warning。一个 overlap 作业中 FireRed 最多加载两次：
常驻主 worker 与仅在有接受轨道时启动的临时 worker。

## 输出

~~~text
recording_transcript/
├── transcript.txt
├── transcript.json
└── diagnostics.json
~~~

TXT 示例：

~~~text
[00:00:03] 说话人1：我们先确认一下今天讨论的问题。
[00:00:11] 说话人2：好的，主要有三个方面。
~~~

JSON 保留毫秒级起止时间、说话人、来源、对齐置信度与重叠标记。diagnostics 默认记录
低开销阶段耗时和调度摘要；`--diagnostics resources` 额外记录 service、speaker、两个 AED、
Punc 与临时 overlap owner 的资源峰值；`--diagnostics deep` 还会写出完整 scheduler 和逐 batch
事件 trace。旧的 `--profile-resources` 等价于 `--diagnostics resources`。所有资源采样只包含本项目
登记的 PID，不包含其他项目进程。

### AED 预算矩阵

需要比较同一常驻服务下不同瞬时预算比例时，使用专用工具。它通过请求级比例控制 AED
transient budget，不修改安全余量，也不管理其他项目进程：

```bash
.venv/bin/python tools/benchmark_aed_budget_matrix.py \
  audio_test/刘洋洋.wav \
  --config outputs/measurements/20260822T-memory-first-aed-liuyangyang/config.toml \
  --output-root outputs/measurements/aed-budget-matrix \
  --fractions 0.60 0.70 0.80 0.90 0.95 \
  --repeats 3 \
  --diagnostics deep
```

矩阵记录每次的 AED/端到端耗时、输出 fingerprint、显存峰值、预算利用率、安全余量和失败原因。
安全余量不足、预算利用率超过 100%、OOM 或协议失败的候选不能作为生产默认值。生产服务会在每项
作业末尾清理两个 AED allocator 的未使用 cache；正式比较仍须使用同一常驻服务并保留首任务预热
与后续任务不预热的真实差异。

## 退出码

| 退出码 | 含义 |
|---:|---|
| 0 | 成功，可能包含已记录的重叠降级 |
| 2 | 参数或配置错误 |
| 3 | 输入音频错误 |
| 4 | 环境、GPU 容量或模型错误 |
| 5 | 主转录失败 |
| 6 | 输出写入失败 |
| 130 | 用户中断 |

## 常见问题

### doctor 报告 CUDA 不可用

在三个虚拟环境中分别检查：

~~~bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
.venv-asr/bin/python -c "import torch; print(torch.cuda.is_available())"
.venv-overlap/bin/python -c "import torch; print(torch.cuda.is_available())"
~~~

### doctor 报告 GPU 0 空闲显存不足

用 `nvidia-smi -i 0` 只读确认状态。不要杀死、暂停或迁移其他项目进程；等待其自然
结束。常驻服务门槛是 23 GiB 总显存、13,312 MiB 空闲显存。实际作业峰值应使用
`--profile-resources` 在本机实测；不要在未重新实测时降低该准入门槛。
旧的串行阶段门槛为 12 GiB；FireRed 四模块历史净增峰值记录为 9.04 GiB，它们不是
常驻并行服务的当前准入依据。

### 重叠增强不可用

主 ASR 和 diarization 仍可完成，diagnostics 会记录 warning。检查 `HF_TOKEN`、
pyannote 授权、ClearVoice checkpoint 和 overlap worker 路径。

### 输出目录已存在

默认拒绝覆盖。确认目标内容后显式增加 `--overwrite`。

### 清理本项目模型和环境

只在本项目没有运行任务时处理 `.venv`、`.venv-asr`、`.venv-overlap` 和配置中的
`model_dir`。安装、doctor 和转录命令不会删除输出目录或管理其他项目进程。

## 构建发布验收语料包

本仓库已有被 Git 忽略的语料目录：
`artifacts/release_validation_corpus`。需要从 RAMC 重建时运行：

~~~bash
PYTHONPATH=src .venv/bin/python tools/build_release_validation_corpus.py \
  --source-root /path/to/MDT2021S003 \
  --output-dir artifacts/release_validation_corpus \
  --overwrite
PYTHONPATH=src .venv/bin/python tools/verify_release_validation_corpus.py \
  artifacts/release_validation_corpus
~~~

构建器生成 20 条 420 秒核心录音、30 条合成重叠、10 条真实重叠观察窗、10 条
正常换人负样本和三个派生测试源；构建过程只读源数据，并校验相对路径和 SHA-256。

在 GPU 0 自然满足门槛时执行完整 Linux 验收：

~~~bash
artifacts/release_validation_corpus/run_validation_linux.sh \
  --project-root "$PWD" \
  --config /etc/funasr-transcriber.toml
~~~

正式发布不得使用 `--skip-stress`。结果写入
`outputs/release_validation/<运行时间>-<进程号>/`，最新汇总写入
`outputs/release_validation/summary.json` 和 `outputs/release_validation/summary.txt`。

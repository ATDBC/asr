# Standalone FireRedASR2S Transcriber

独立的 Linux 命令行录音转录工具。输入以普通话为主的音频，输出带毫秒时间戳、
匿名说话人编号的 TXT、结构化 JSON 与诊断文件。生产 ASR 使用
FireRedASR2S；不存在旧 FunASR ASR fallback。

## 支持范围

- Linux、Python 3.11、NVIDIA GPU。
- 固定使用物理 GPU 0（`cuda:0`）。
- GPU 总显存至少 23 GiB；常驻服务热加载前至少 13,312 MiB 空闲。
- WAV、MP3、M4A、FLAC、AAC、OGG，最长两小时。
- 自动估计 1～10 位说话人；服务按 FIFO 串行消费作业，但单个作业的 ASR 与
  diarization 并行。
- FireRed 流程固定为 VAD → AED → Punc；不加载语言识别模型。

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
segment_batch_size = 4

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

`segment_batch_size` 是 VAD 完成后 AED 与 Punc 的官方批处理大小。每个 VAD 段仍按
原时间顺序导出；默认值 1 代表逐段处理，只有在本机 GPU 0 校准证明结果不变且显存满足
预算后才应提高。

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

仅当 ModelScope 无法获取 FireRed 三模型时，才显式使用 Hugging Face fallback：

~~~bash
.venv-asr/bin/python -c \
  'from pathlib import Path; from funasr_transcriber.model_registry import acquire_firered_assets; acquire_firered_assets(Path("/path/to/models"), source="huggingface")'
~~~

fallback 仍会校验三个模型目录，3D-Speaker 相关资产继续由 ModelScope 获取。不要
混用不完整目录。

`doctor` 对内置 `hello_zh.wav` 依次执行 FireRedVAD、FireRedASR2-AED、FireRedPunc
全链推理；关闭 worker 后检查显存释放，再按顺序
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
~~~

首次转录会自动启动本机后台服务。服务在 `work_dir/service/` 保存自己的 Unix socket、
状态和日志，并常驻一个 FireRed worker 与一个 3D-Speaker owner。普通作业让两条分支并行，
多个 CLI 请求按 FIFO 排队；每次 CLI 返回不会释放热模型。

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

JSON 保留毫秒级起止时间、说话人、来源、对齐置信度与重叠标记。diagnostics
记录句级 AED/ASR 置信度、VAD 区间、阶段耗时、GPU 准入/释放样本和降级告警。默认
`resource_profile` 仅为 `{"enabled": false}`；使用 `--profile-resources` 后还会记录服务、
FireRed 与临时 overlap owner 的资源峰值、模型清单和并行分支时间，绝不包含其他项目 PID。

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

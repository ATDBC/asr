# Standalone FireRedASR2S Transcriber

独立的 Linux 命令行录音转录工具。输入以普通话为主的音频，输出带毫秒时间戳、
匿名说话人编号、语言识别信息的 TXT、结构化 JSON 与诊断文件。生产 ASR 使用
FireRedASR2S；不存在旧 FunASR ASR fallback。

## 支持范围

- Linux、Python 3.11、NVIDIA GPU。
- 固定使用物理 GPU 0（`cuda:0`）。
- GPU 总显存至少 23 GiB；每个 GPU 模型阶段开始前至少 20 GiB 空闲。
- WAV、MP3、M4A、FLAC、AAC、OGG，最长两小时。
- 自动估计 1～10 位说话人，单任务串行。
- 主要面向普通话；FireRedLID 会记录句级语言和置信度，非普通话或低置信度会告警。

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
lid_warning_confidence = 0.80

[overlap]
worker_python = "/path/to/standalone_transcriber/.venv-overlap/bin/python"
~~~

如果设置 `CUDA_VISIBLE_DEVICES`，首个可见设备必须仍是物理 GPU 0；不支持把物理
GPU 1 重映射成逻辑 `cuda:0`。配置优先级为：

~~~text
命令行 > FUNASR_TRANSCRIBER_* 环境变量 > TOML > 内置默认值
~~~

## 模型准备

FireRed 模型默认从 ModelScope 获取，目录如下：

~~~text
models/fireredasr2s/
├── source/                 # 固定 commit 的 FireRedASR2S
├── FireRedVAD/VAD/
├── FireRedLID/
├── FireRedASR2-AED/
└── FireRedPunc/
~~~

默认下载和验证：

~~~bash
export HF_TOKEN=hf_xxx
funasr-transcribe models download --config /etc/funasr-transcriber.toml
funasr-transcribe models verify --config /etc/funasr-transcriber.toml
funasr-transcribe doctor --config /etc/funasr-transcriber.toml
~~~

pyannote 重叠检测需要先在 Hugging Face 接受 `segmentation-3.0` 与
`speaker-diarization-3.1` 的访问条件。`HF_TOKEN` 只从环境读取，不写入配置、
日志或 diagnostics。

仅当 ModelScope 无法获取 FireRed 四模型时，才显式使用 Hugging Face fallback：

~~~bash
.venv-asr/bin/python -c \
  'from pathlib import Path; from funasr_transcriber.model_registry import acquire_firered_assets; acquire_firered_assets(Path("/path/to/models"), source="huggingface")'
~~~

fallback 仍会校验四个模型目录，3D-Speaker 相关资产继续由 ModelScope 获取。不要
混用不完整目录。

`doctor` 对内置 `hello_zh.wav` 依次执行 FireRedVAD、FireRedLID、
FireRedASR2-AED、FireRedPunc 全链推理；关闭 worker 后检查显存释放，再按顺序
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

# 禁用重叠增强
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml \
  --disable-overlap
~~~

主音频先由一个 FireRed worker 识别并关闭；随后加载 Speaker 并准备重叠轨道；
Speaker 关闭后，只有存在候选轨道时才启动第二个 FireRed worker。无重叠时
FireRed 只加载一次，有重叠时最多加载两次，所有候选轨道复用第二个 worker。
主 ASR 失败会返回退出码 5，不切换旧模型；单个重叠窗口失败只保留该窗口原文并
写入 warning。

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
记录句级 LID/ASR 置信度、VAD 区间、阶段耗时、GPU 准入/释放样本和降级告警。

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
结束。门槛是 23 GiB 总显存、20 GiB 空闲显存，不能通过降低门槛绕过。

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
`validation_results/<运行时间>-<进程号>/`，最新汇总写入
`validation_results/summary.json` 和 `validation_results/summary.txt`。

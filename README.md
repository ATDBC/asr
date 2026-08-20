# FunASR Standalone Transcriber

独立的 Linux 命令行录音转录工具。输入普通话音频，输出带时间戳和匿名说话人编号的 TXT、结构化 JSON 与诊断文件。

## 支持范围

- Linux + NVIDIA GPU。
- Python 3.11。
- WAV、MP3、M4A、FLAC、AAC、OGG。
- 最长两小时。
- 自动估计 1～10 位说话人。
- 单任务串行。
- 普通话。

本工具不依赖 FastAPI、MySQL、原项目 .env、讯飞接口或声纹角色识别。

## 系统准备

安装以下系统组件：

~~~bash
python3.11 --version
ffmpeg -version
ffprobe -version
nvidia-smi
git --version
~~~

NVIDIA 驱动必须支持安装脚本选择的 CUDA PyTorch。默认使用 cu128；需要其他官方 PyTorch wheel 源时设置：

~~~bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
~~~

## 安装

~~~bash
cd standalone_transcriber
chmod +x scripts/install_linux.sh
./scripts/install_linux.sh
source .venv/bin/activate
~~~

安装脚本创建：

- .venv：FunASR、3D-Speaker 和主 CLI。
- .venv-overlap：pyannote OSD 与 ClearVoice。

不使用系统 site-packages，也不修改系统 Python。

## 配置

复制默认配置：

~~~bash
cp config/default.toml /etc/funasr-transcriber.toml
~~~

确认并修改：

~~~toml
[paths]
model_dir = "/opt/funasr-transcriber/models"
work_dir = "/var/tmp/funasr-transcriber"

[overlap]
worker_python = "/path/to/standalone_transcriber/.venv-overlap/bin/python"
~~~

配置优先级：

~~~text
命令行 > FUNASR_TRANSCRIBER_* 环境变量 > TOML > 内置默认值
~~~

## 模型准备

pyannote 重叠检测需要先在 Hugging Face 接受 `segmentation-3.0` 与
`speaker-diarization-3.1` 的模型访问条件，并创建 read token：

~~~bash
export HF_TOKEN=hf_xxx
funasr-transcribe models download --config /etc/funasr-transcriber.toml
funasr-transcribe models verify --config /etc/funasr-transcriber.toml
funasr-transcribe doctor --config /etc/funasr-transcriber.toml
~~~

HF_TOKEN 只从环境读取，不写入配置、日志或 diagnostics。

## 转录

~~~bash
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml
~~~

指定输出：

~~~bash
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml \
  --output-dir /data/results/recording
~~~

保留中间文件：

~~~bash
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml \
  --keep-intermediate
~~~

禁用重叠增强：

~~~bash
funasr-transcribe /data/recording.mp3 \
  --config /etc/funasr-transcriber.toml \
  --disable-overlap
~~~

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

JSON 保留毫秒级起止时间、说话人、来源、对齐置信度与重叠标记。diagnostics 记录模型、版本、配置快照、各阶段耗时和降级告警。

## 退出码

| 退出码 | 含义 |
|---:|---|
| 0 | 成功，可能包含已记录的重叠降级 |
| 2 | 参数或配置错误 |
| 3 | 输入音频错误 |
| 4 | 环境或模型错误 |
| 5 | 主转录失败 |
| 6 | 输出写入失败 |
| 130 | 用户中断 |

## 常见问题

### doctor 报告 CUDA 不可用

确认 nvidia-smi 正常，并在两个虚拟环境中运行：

~~~bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
.venv-overlap/bin/python -c "import torch; print(torch.cuda.is_available())"
~~~

### 重叠增强不可用

普通 ASR 和 diarization 仍会运行，diagnostics 会记录 warning。检查 HF_TOKEN、pyannote 模型授权、ClearVoice checkpoint 和 overlap worker 路径。

### 输出目录已存在

默认拒绝覆盖。确认目标内容后显式增加 --overwrite。

### 清理模型和环境

停止任务后删除 .venv、.venv-overlap 和配置中的 model_dir。输出目录不会被安装或卸载脚本删除。

## 构建发布验收语料包

完整 RAMC 数据准备好后，在仓库根目录运行：

~~~powershell
$env:PYTHONPATH='standalone_transcriber/src'
python standalone_transcriber/tools/build_release_validation_corpus.py `
  --source-root G:\MagicData-RAMC\extracted\MDT2021S003 `
  --output-dir artifacts\release_validation_corpus
~~~

构建器生成20条420秒核心录音、30条合成重叠、10条真实重叠观察窗、
10条正常换人负样本和三个派生测试源。构建过程只读源数据，完成后自动校验
音频、标注、相对路径和 SHA-256。生成目录已被 Git 忽略。

复制 `standalone_transcriber` 和 `artifacts/release_validation_corpus` 到 Linux，
进入验收包目录后运行：

~~~bash
chmod +x ./run_validation_linux.sh
./run_validation_linux.sh \
  --project-root /opt/FunASR_API/standalone_transcriber \
  --config /etc/funasr-transcriber.toml
~~~

脚本先验证校验和与模型，再执行短普通话冒烟、20条 RAMC、重叠集、输入格式
矩阵和30/60/120分钟压力测试。每次运行的阶段产物保留在
`validation_results/<运行时间>-<进程号>/`，最新总结果同步写入
`validation_results/summary.json` 和 `validation_results/summary.txt`。
临时只验证效果、不执行长录音时，可增加 `--skip-stress`；正式发布不得跳过。

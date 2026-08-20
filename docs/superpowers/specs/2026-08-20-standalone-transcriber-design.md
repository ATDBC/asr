# 独立离线录音转录工具设计

日期：2026-08-20

## 1. 目标

建设一个可独立部署在 Linux + NVIDIA GPU 服务器上的命令行工具。输入一份普通话音频文件，输出带时间戳和匿名说话人编号的完整转录文本，同时保存结构化 JSON 和运行诊断。

成品 TXT 示例：

~~~text
[00:00:03] 说话人1：我们先确认一下今天讨论的问题。
[00:00:11] 说话人2：好的，主要有三个方面。
~~~

第一版约束：

- 只处理本地音频文件，不处理视频、URL、Web 请求或实时音频。
- 普通话作为唯一验收语言。
- 系统自动判断说话人数，范围为 1 至 10 人。
- 典型录音约 30 分钟，必须支持最长 2 小时。
- 第一版只允许单任务串行执行。
- 相同说话人前一句结束与后一句开始间隔不超过 1 秒时合并为同一话轮。
- 重叠语音尽力恢复双路内容，但增强失败不得破坏普通转录结果。
- 不依赖 FastAPI、MySQL、原项目 .env、讯飞接口或声纹角色识别。

## 2. 交付边界

在仓库内建立可单独复制、安装和运行的子项目：

~~~text
standalone_transcriber/
├── pyproject.toml
├── README.md
├── config/
│   └── default.toml
├── scripts/
│   └── install_linux.sh
├── src/
│   └── funasr_transcriber/
│       ├── cli.py
│       ├── config.py
│       ├── schemas.py
│       ├── pipeline.py
│       ├── audio.py
│       ├── model_registry.py
│       ├── asr.py
│       ├── speaker_diarization.py
│       ├── overlap.py
│       ├── alignment.py
│       ├── turns.py
│       ├── exporters.py
│       └── diagnostics.py
└── tests/
~~~

模块职责：

- audio.py：输入探测、FFmpeg 转换、时长及音频属性校验。
- model_registry.py：显式加载、缓存和验证所有模型。
- asr.py：输出全文、句子和字词级时间信息。
- speaker_diarization.py：自动人数、固定随机种子、聚类、过分割合并和抖动平滑。
- overlap.py：重叠检测、ClearVoice 双路分离、结果验收和回退。
- alignment.py：将带时间的文字归属到全局说话人轨迹。
- turns.py：构建和合并最终话轮。
- exporters.py：原子生成 TXT 和 JSON。
- diagnostics.py：记录版本、配置、耗时、资源、告警和降级信息。
- pipeline.py：只负责编排阶段，不包含具体算法。
- cli.py：命令行解析、退出码、进度和任务锁。

新子项目不得从旧 app.py、db.py 或 Web 服务反向导入。已经验证的 diarization 算法应被抽取并固化到新包中，避免把服务端副作用带入离线工具。

## 3. 安装与运行

服务器环境：

- Linux。
- NVIDIA GPU。
- Python 3.11 虚拟环境。
- 可以访问 ModelScope、Hugging Face 等外网模型源。
- 直接安装到主机，不使用 Docker。

主程序和重叠分离使用两个虚拟环境，隔离 PyTorch 和音频依赖冲突：

~~~text
standalone_transcriber/.venv/
standalone_transcriber/.venv-overlap/
~~~

安装入口：

~~~bash
cd standalone_transcriber
./scripts/install_linux.sh
source .venv/bin/activate
~~~

安装脚本检查 Python、FFmpeg、NVIDIA 驱动和 CUDA，创建两个虚拟环境，安装锁定依赖并验证 PyTorch 能识别 GPU。脚本不得修改系统 Python。

模型通过显式命令准备，正式转录时不得隐式升级或临时下载：

~~~bash
funasr-transcribe models download
funasr-transcribe models verify
funasr-transcribe doctor
~~~

doctor 检查运行环境、模型完整性、两个虚拟环境、目录权限、磁盘空间和 CUDA，并使用内置短音频完成一次最小 ASR 与 diarization 冒烟测试。

## 4. CLI 契约

最简调用：

~~~bash
funasr-transcribe /data/recording.mp3
~~~

完整调用：

~~~bash
funasr-transcribe /data/recording.mp3 \
  --output-dir /data/results/recording \
  --config /etc/funasr-transcriber/config.toml \
  --device cuda:0 \
  --keep-intermediate
~~~

第一版参数：

- INPUT_AUDIO：必填，本地音频路径。
- --output-dir：指定结果目录。
- --config：指定 TOML 配置。
- --device：覆盖 GPU 配置。
- --keep-intermediate：保留标准 WAV、重叠窗口和分离音轨。
- --overwrite：明确允许覆盖已有结果。
- --disable-overlap：关闭重叠增强。
- --log-level：设置日志等级。

默认结果目录为输入文件旁的 文件名_transcript。目录已存在时默认拒绝覆盖。

退出码：

- 0：成功，允许包含明确记录的重叠增强降级。
- 2：命令行或配置错误。
- 3：输入音频错误。
- 4：环境或模型错误。
- 5：主转录失败。
- 6：输出写入失败。
- 130：用户中断。

## 5. 配置

默认配置使用 TOML。优先级为命令行参数、FUNASR_TRANSCRIBER_ 前缀环境变量、指定 TOML、default.toml。不得读取原项目 .env。

核心默认值：

~~~toml
[general]
language = "zh"
device = "cuda:0"
max_audio_seconds = 7200
random_seed = 0

[paths]
model_dir = "/opt/funasr-transcriber/models"
work_dir = "/var/tmp/funasr-transcriber"

[diarization]
speaker_count = "auto"
min_speakers = 1
max_speakers = 10
chunk_seconds = 1.0
chunk_step_seconds = 0.5
merge_cosine = 0.23

[turns]
merge_gap_seconds = 1.0

[overlap]
enabled = true
detector_model = "pyannote/overlapped-speech-detection"
hf_token_env = "HF_TOKEN"
worker_python = "/opt/funasr-transcriber/.venv-overlap/bin/python"
timeout_seconds = 1200
~~~

未指定 --output-dir 时，结果目录固定为输入文件旁的 文件名_transcript；配置文件不另设隐含输出根目录。

## 6. 主数据流

### 6.1 音频标准化

支持 WAV、MP3、M4A、FLAC、AAC 和 OGG。使用 FFmpeg 统一转换为 16 kHz、单声道、PCM signed 16-bit WAV。

在模型加载前检查文件可读、时长大于零、时长不超过 7200 秒、解码结果属性正确和临时目录空间充足。

### 6.2 主 ASR

对标准 WAV 执行一次完整 Paraformer 转写，保留全文、句子、标点和字词级时间戳。ASR 适配器的正式输出契约是 TimedToken：

~~~text
TimedToken(text, start_ms, end_ms)
~~~

模型启动后必须验证时间戳能力。个别句子缺少细粒度时间戳时进入第 8 节的局部重解码流程。

### 6.3 全局说话人分离

对整段录音执行 VAD、1.0 秒窗口及 0.5 秒步长的 speaker embedding 提取、自动人数谱聚类、阈值 0.23 的过分割合并和 A-B-A 短抖动平滑。

每次聚类前固定随机种子 0。自动人数限制为 1 至 10。最终说话人编号按照首次发言时间重排，最早出现的簇为说话人1。

第一版对 0 至 2 小时录音统一使用全局聚类，不预设按时长切换算法。embedding 必须分批提取，控制 GPU 显存。30 分钟、1 小时和2小时压力测试决定当前服务器是否满足上线条件。如果2小时全局谱聚类无法稳定完成，分层聚类作为后续单独设计，不在第一版中静默启用。

### 6.4 对齐、重叠和话轮

主 ASR token 与全局说话人轨迹按时间重叠进行归属。高置信重叠窗口可进入专用 OSD 与 ClearVoice 增强。最终 token 重建为话轮，并按同人且间隔不超过 1 秒的规则合并。

## 7. 文本与说话人对齐

统一中间结构：

~~~text
TimedToken(text, start_ms, end_ms)
SpeakerSegment(speaker_id, start_ms, end_ms)
AttributedToken(text, start_ms, end_ms, speaker_id, confidence, source)
~~~

每个 token 计算与所有 SpeakerSegment 的重叠时长，归给重叠最多的说话人，并记录最佳覆盖比例、第一名与第二名差距以及是否跨越切换边界。

初始置信规则：

- high：最佳覆盖比例不低于 70%，且无明显第二说话人竞争。
- medium：最佳覆盖比例不低于 50%，或 token 中点明确落在某个说话人段中。
- low：两个说话人覆盖接近、没有有效 speaker segment，或 token 明显跨越边界。

阈值写入配置，并通过 RAMC 端到端验证后锁定。标点附着到前一个有效文字 token，不独立判断说话人。

短插话不因字数少而自动归给前后长句。只要时间证据明确指向另一说话人，就保留为独立发言。

## 8. 缺失时间戳与局部重解码

某句话缺少字词级时间戳时：

1. 只有一个明显主说话人时，整句归给该人。
2. 跨越多个有效说话人时，按 diarization 边界切成子音频。
3. 每个子音频局部重新 ASR。
4. 局部结果与原始整句进行文本对齐，移除切割边缘重复字。
5. 局部结果归给对应说话人。
6. 局部重解码失败时保留原始整句，归给时间占比最高的说话人并标记 low。

生产路径不得使用按字符比例猜测说话人边界的旧逻辑。

## 9. 重叠语音增强

重叠候选主要由 pyannote/overlapped-speech-detection 提供，附近 diarization 或 ASR 冲突只用于确认和扩展，不能单独把正常换人判成重叠。该模型需要在 Hugging Face 接受访问条件，并通过 HF_TOKEN 环境变量授权；models download 和 models verify 必须明确检查这一条件。令牌不得写入 TOML、诊断文件或日志。

候选窗口合并相邻 OSD 片段并适当前后扩展，同时限制最短、最长和单任务累计处理时长。每个窗口通过 ClearVoice 产生两路音轨，两路分别接受音量、VAD、ASR、embedding 和全局 speaker centroid 匹配检查。

只有以下条件全部满足时才替换普通结果：

- 两路均有有效语音和非空文字。
- 两路匹配到两个不同的全局说话人。
- 最佳相似度达到阈值。
- 最佳与次佳相似度差达到阈值。
- 两路不是明显重复内容或同一残留音轨。

如果只有一路通过、匹配不可靠、OSD 或 ClearVoice 不可用，则保留普通结果并记录 overlap_status=fallback。不因单个短重叠窗口新增全局说话人。

重叠话轮可在 JSON 中时间交叉。TXT 按开始时间排序；开始时间相同时按说话人编号排序。重叠话轮不参与跨窗口的 1 秒同人合并。

## 10. 输出契约

成功输出：

~~~text
<output-dir>/
├── transcript.txt
├── transcript.json
├── diagnostics.json
└── transcribe.log
~~~

transcript.txt 使用 UTF-8，每行格式为：

~~~text
[HH:MM:SS] 说话人N：文本
~~~

transcript.json 至少包含：

- schema_version。
- status。
- 输入路径、格式和时长。
- language。
- speaker_count。
- full_text。
- turns。
- warnings。

每个 turn 包含 index、speaker、start_ms、end_ms、text、source、alignment_confidence 和 overlap。

diagnostics.json 记录音频属性、模型及版本、CUDA 和依赖版本、配置快照、各阶段耗时、VAD 时长、原始及合并后簇数、低置信文字数量、重叠窗口统计、降级原因、峰值 GPU 显存和可可靠采集的进程峰值内存。配置快照不得包含密钥。

输出先写临时文件，结构检查通过后原子改名。硬失败不得留下看似成功的 transcript.txt 或 transcript.json。

## 11. 失败与降级

以下情况硬失败：

- 输入无法解码或超过 2 小时。
- ASR 模型或主转写失败。
- 3D-Speaker 模型或全局分离失败。
- 没有得到有效文字。
- 文字完整性检查发现无法解释的丢失或重复。
- 输出无法安全写入。

以下情况允许降级成功并产生 warning：

- OSD 不可用。
- ClearVoice 某个窗口失败。
- 少量句子缺少细粒度时间戳并采用低置信归属。
- 某些分离音轨无法可靠匹配全局说话人。

说话人分离失败不得静默退回其他 speaker 标签后仍报告成功。

## 12. 运行资源与任务管理

第一版使用带 PID 的任务锁，确保同一安装实例只运行一个 GPU 转录任务。活跃锁存在时拒绝新任务；陈旧锁经 PID 检查后清理。

正式处理前估算标准 WAV、中间音轨和临时文件所需空间，检查 GPU 可用显存和临时目录权限。用户中断时终止重叠子进程并清理临时文件，保留日志和错误诊断。

CLI 输出七个阶段的进度：标准化、ASR、embedding、聚类、重叠增强、对齐与话轮、导出。

## 13. 验证与上线标准

### 13.1 RAMC 完整端到端

使用现有20条、每条420秒的 RAMC 音频，通过最终 CLI 并关闭重叠增强。评估全文 CER、最终话轮 purity、DER-like、人数准确率及 MAE、cpCER 和文字保留率。

上线门槛：

- 最终话轮 weighted purity 不低于 0.92。
- 最终话轮 DER-like 不高于 0.22。
- 人数绝对误差 MAE 不高于 0.25。
- 人数完全正确率不低于 80%。
- cpCER 不得比全文 CER 高超过 0.05。
- 非重叠流程中文字保留率为 100%，忽略空格和统一标点差异。
- 低置信度文字占比不高于 5%。

cpCER 将每个预测说话人的文字串联，与每个真实说话人的文字进行最优匿名说话人映射后计算总字符错误率。

### 13.2 可复现性

选择短、中、长三条录音，各连续运行两次。transcript.txt 必须字节一致；transcript.json 中除运行标识外的业务字段一致；人数、speaker 编号和话轮边界一致。

### 13.3 重叠语音

从不同 RAMC 说话人的干净语句构造约50个合成重叠样本，覆盖不同音量差、重叠比例和持续时间，并加入正常换人负样本。

门槛：

- 正常换人误触发率不高于 5%。
- 合成重叠双路有效恢复率不低于 60%。
- 被接受的双路结果全部匹配到两个不同全局说话人。
- 降级窗口不产生主转录文字重复。

### 13.4 输入兼容性

覆盖 WAV、MP3、M4A、FLAC、AAC、OGG，8k、16k、44.1k、48k，单声道、双声道，中文及特殊文件名，长静音、全静音、损坏文件、零字节和超过2小时文件。

有效文件必须被规范化并完成处理；无效文件返回对应退出码；全静音不得生成伪造说话人文本；成功和失败场景均按配置清理或保留中间文件。

### 13.5 长录音压力

分别运行30分钟、1小时和2小时录音。三者均使用同一全局聚类，要求稳定完成、无 GPU OOM、无进程被系统杀死、embedding 保持批处理、临时文件正确清理、结果可复现。记录总时长、阶段耗时、GPU峰值显存和进程峰值内存。

服务器硬件型号尚未固定，因此第一版发布门槛不设置固定实时率；门槛是目标服务器上的2小时任务能够完成并留下完整资源报告。

### 13.6 Linux 全新安装

在无项目历史环境的 Linux + NVIDIA 服务器上，从代码检出开始执行安装、模型下载、模型验证、doctor 和示例转录。不得依赖数据库、Web 服务、Windows 路径或手工修改 Python 源码。ClearVoice 环境损坏时，普通转录仍应成功并明确记录重叠增强不可用。

## 14. 测试分层

- unit：配置、时间处理、对齐、话轮合并、格式化、错误处理。
- integration：FFmpeg、模型适配器、ClearVoice worker、短音频管线。
- benchmark：20条 RAMC、合成重叠集、长音频压力。
- deployment：全新 Linux 安装与 doctor。

日常提交运行 unit 和轻量 integration。模型基准及2小时压力测试作为发布验收。

## 15. 非目标

第一版不实现：

- Web API、网页或数据库任务系统。
- 实时或流式转录。
- URL 下载。
- 角色身份识别或注册声纹。
- 中文方言专项模型路由。
- 多 GPU 调度或并发任务队列。
- 未经性能测试自动启用的分层聚类。

## 16. 设计完成判定

当独立子项目可以在全新 Linux + NVIDIA 环境中安装，通过 doctor，完成最长2小时普通话音频的确定性全局转录，生成满足契约的 TXT、JSON 和诊断文件，并通过第13节全部门槛时，第一版目标完成。

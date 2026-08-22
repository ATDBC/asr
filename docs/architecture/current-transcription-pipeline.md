# 当前录音转录流程

生产转录由本机常驻服务执行。客户端通过 Unix socket 提交作业；服务只使用物理 GPU 0，
长期持有两个 AED FireRedWorkerClient、一个独立 Punc owner 和一个 SpeakerDiarizer。除非收到 `service stop` 或
执行显式 overlap 作业，模型不会在每次 CLI 调用后卸载。

## 服务生命周期与普通作业

```text
funasr-transcribe recording.wav
  │
  ├─ health Unix socket
  ├─ 服务不存在：持有本项目启动锁，启动后台服务并等待 ready
  ├─ FIFO 提交一个作业（服务一次只消费一个作业）
  │
  └─ 服务内：FFmpeg 归一化为 16 kHz / 单声道 / PCM_16
       │
       ├─ SpeakerDiarizer.diarize(normalized.wav)
       │        ↓ 长 speaker 段共享一次全局 FireRedVAD，再与各段求交
       └─ worker-0 全局 FireRedVAD → 主进程固定 batch 计划
             ┌─ FireRed AED worker-0 ─┐
             └─ FireRed AED worker-1 ─┘
                    ↓ batch 完成即入有界队列
                单个独立 FireRedPunc owner
                    ↓ 按原始 index 合并
              ↑
       diarization 先行完成；ASR 只使用 diarization 段落；超过 10 秒的长 turn
       触发一次全局 FireRedVAD，结果与每个 speaker 区间求交后再进入 AED/Punc；
       FireRedVAD 无输出时按 10 秒硬切，60 秒仍是 AED 的绝对上限
              │
       token 按所在段直接归属说话人、轮次构建、原子导出 TXT / JSON / diagnostics
              │
       所有热模型继续常驻，供 FIFO 中的下一个作业复用
```

服务启动会先只读检查 GPU 0 的并行预算，然后并发加载两个 FireRed AED、一个独立 Punc owner 和 3D-Speaker；
worker-0 只承载 FireRedVAD，worker-1 不加载 VAD/Punc，Punc owner 只加载 Punc。三个 FireRed owner 的首个 `status`
请求完成后，服务用短音频预热 AED/Punc；预热不计入后续作业耗时。FireRed 仍在独立的
`.venv-asr` 子进程；主服务进程不导入 FireRed 运行时。服务停止时会先进入 draining，
拒绝新的提交、完成已经接受的作业，再关闭热模型；因此 `funasr-transcribe service stop`
是释放本项目常驻显存的唯一正常操作。

本机协议是每行一个 JSON 的 Unix socket 协议，状态、socket、启动锁和日志位于
`paths.work_dir/service/`。客户端只能清理由记录的服务 PID 已不存在的陈旧状态；不会
向任何其他项目的 PID 发信号。

## 命令

```bash
# 自动启动服务（如尚未运行），提交并等待结果
funasr-transcribe /data/recording.mp3 --config /etc/funasr-transcriber.toml

# 仅查询本项目服务是否 ready
funasr-transcribe service status --config /etc/funasr-transcriber.toml

# 排空已接受的队列并正常关闭本项目模型
funasr-transcribe service stop --config /etc/funasr-transcriber.toml
```

`--enable-overlap` 和 `--disable-overlap` 互斥。配置中的 overlap 默认值为关闭，所以普通
作业不会创建 pyannote 或 ClearVoice worker。诊断默认级别为 `basic`；`--diagnostics resources`
启用资源采样，`--diagnostics deep` 额外写出完整调度和逐 batch trace；`--profile-resources` 是
`resources` 的兼容别名。默认输出的 `resource_profile` 为 `{ "enabled": false }`。

## FireRed worker 边界与功能归类

本文中“FireRed worker”专指隔离常驻的 FireRed owner；生产部署为两个 AED worker 和一个唯一的
Punc owner。Punc owner 不加载 AED/VAD。
FireRedVAD 在功能上归属于说话人分离阶段：它只负责把超过 10 秒的 speaker 区间细分为
可供 AED 解码的语音区间。由于官方运行时和模型依赖，FireRedVAD 目前仍由同一个
`.venv-asr` 子进程承载；这属于进程部署方式，不改变它在流程和指标中的说话人分离角色。

服务通过严格 JSON Lines 调用 `.venv-asr` 中的 FireRed worker。每个请求都有唯一 ID，
支持 `warmup`、`refine_vad`、固定 `aed_plan`、`punctuate`、`transcribe`、`verify`、`status` 和 `shutdown`。ID 不匹配、重复响应、非法 JSON、
超时或 worker 提前退出都会成为转录错误，不会切换旧 ASR。

生产流程先由 worker-0 对 `vad_segments_ms`（diarization 段落原样发送）执行一次全局 VAD；
主进程按预测耗时将完整 batch 计划分到两个 AED worker。worker 进程中的
说话人分离 VAD helper 用脚本化 VAD 重放这些段落，其中超过 10 秒的长 turn 触发一次全局
神经 FireRedVAD，随后
只保留落在各 speaker 段内部的交集（无交集时退化为 10 秒硬切分）。官方 AED/Punc 批处理与 uttid
时间戳路径保持不变，响应回显细分后的段落。未携带该字段的请求（doctor 的 `verify`、
overlap 临时 worker 对分离轨道的重解码）仍运行 FireRedVAD。

worker-0 固定加载 FireRedASR2S 提交 `4e7d9aaf4482a47cec1724807026b9b151926eb5` 的
FireRedASR2-AED 和 FireRedVAD；worker-1 只加载 FireRedASR2-AED；独立 Punc owner 只加载
FireRedPunc。语言识别模型不加载。主进程按时长排序后用 memory-first 装箱器根据预热后的实际
worker 常驻显存和共享预算决定每个 batch 的具体片段；任一 AED batch 完成后立即进入有界 Punc
队列，Punc 按 token 预算组成微批，最后按原始时间索引恢复，因此响应顺序和 speaker 归属不变。
在第一个 AED batch 开始前，两个 AED worker 和 Punc owner 预热完成、说话人分离/VAD 结束；主进程按时长排序并基于初始成本模型一次性生成完整批次计划。
计划预算来自预热后 GPU 0 空闲显存减去 `[asr].gpu_safety_margin_mib=1536`，使用每个 worker 的实际常驻基线和预计瞬时增量显存。AED 开始前再次采样：满足两路预计峰值则并发，
只能容纳一路则按相同固定计划串行，连单个批次都放不下则提前失败；这些门控不会重排或重分批。
两个 AED worker 后续 batch 严格按各自固定计划执行，互相没有 batch 屏障；每个完成事件立即进入有界 Punc 队列。
batch 完成后的实际耗时和 PyTorch 峰值只用于更新诊断成本模型，不会重新排序或改变后续批次。该读取只调用 CUDA 只读接口，
不枚举、暂停或操作其他项目进程。
基础级别保留完整计划的摘要、两路分配和预热基线；逐批时间/增量显存误差只有 `deep` 级别才写入
`diagnostics/scheduler.json` 和 `diagnostics/batch-events.jsonl`，普通 `diagnostics.json` 不嵌入完整计划。
doctor/overlap 请求仍保持 FireRedVAD 输出顺序。2026-08-21 在本机 GPU 0
以 420 秒、151 段 RAMC 录音校准 batch 1/2/4/8 各 3 次：batch 1/4 共享稳定
fingerprint，中位端到端 28.50 秒（batch 4），进程峰值 5,690 MiB；batch 2/8 的
fingerprint 与 batch 1 不同但各自三次重复稳定，batch 8 最快（中位 19.11 秒，峰值
5,872 MiB）。默认值现为 8；时长分桶在 809 段正式语料上的一次实测为 batch 8: 113.103 秒、
7,822 MiB 进程峰值，batch 12: 99.868 秒、8,692 MiB，batch 16: 93.709 秒、9,312 MiB。
这些结果与旧输入顺序 fingerprint 不同，故 fingerprint 仍是上线门槛，未自动修改默认 batch。
完整报告
位于 `outputs/measurements/20260821T060915Z-firered-batches/result.json`。官方结果只有句级
`asr_confidence`；稳定协议将句级值
附给所属 token，不虚构词级分数。标点附着到带时间戳 token 后，协议验证全文、句子、
token、VAD 与音频时长的一致性。

新的正式长音频基线固定为 `audio_test/刘洋洋.wav`（约 3600.6 秒）：3D-Speaker 模型
加载时间 28.039 秒不计，但模型加载完成后的音频读取、3D-Speaker VAD、embedding、
聚类、合并和 smoothing 均计入说话人分离耗时 28.252 秒。两个 FireRed AED worker 和独立 Punc owner 的模型加载
时间 36.631 秒和显式 warm-up 不计；worker 正式处理 809 个最终片段（batch 8）耗时
242.145 秒，其中说话人分离阶段的 FireRedVAD 8.604 秒、AED 208.678 秒、Punc 6.008 秒。两部分按
正式串行流程合计，模型加载和 warm-up 排除后的端到端基线为 270.397 秒。FireRed
进程 `nvidia-smi` 峰值为 6,882 MiB，PyTorch AED 实际分配峰值为 5,913 MiB，未发生
OOM。可复核记录位于
`outputs/measurements/20260821T-diarization-firered-vad-10s/formal-flow-batch8-summary.json`。

## overlap 的受控卸载路径

pyannote/ClearVoice 的 overlap 处理显存远高于常驻 Speaker，不能与热 FireRed 和
3D-Speaker 共驻。仅当当前请求使用 `--enable-overlap` 时，服务按下列顺序执行：

1. 先用热模型完成主 ASR 与 diarization，并由热 Speaker 计算说话人 centroid。
2. 两个 AED worker、独立 Punc owner（以及 worker-0 承载的 FireRedVAD helper）和 SpeakerDiarizer 关闭；
   服务以只读 GPU 采样记录热 owner 的释放。
3. 启动隔离 overlap worker，完成 pyannote 检测与 ClearVoice 分离；该 worker 退出后才继续。
4. 临时加载 3D-Speaker，对已分离轨道进行 embedding/centroid 匹配，然后关闭它。
5. 仅当有接受的轨道时，临时启动 FireRed 解码轨道，随后关闭该 worker。
6. 无论 overlap 成功、逐窗降级还是失败，服务都会重新热加载 FireRed 和 3D-Speaker，
   然后才消费下一个 FIFO 作业。

主 ASR 失败仍是整个作业失败；单一 overlap 窗口或轨道失败只会保留原 token 并留下 warning。
默认作业不进入该路径。

## GPU 安全与资源诊断

- 生产固定物理 GPU 0（`cuda:0`），拒绝 GPU 重映射或其他逻辑设备。
- 常驻服务在热加载前要求总显存至少 23 GiB、空闲至少 13,312 MiB；这是一项准入阈值，
  实际显存占用仍以 profiling 实测为准。
- `nvidia-smi` 只读。系统从不终止、暂停、重启、迁移或枚举/导出其他项目的进程。
- 两个 AED worker、独立 Punc owner、其承载的 FireRedVAD helper 与 SpeakerDiarizer 的释放状态会写入
  overlap 作业的 GPU stage。
  无法满足释放容差时不会继续 overlap 重型阶段。

启用 `resources` 后，作业的 `diagnostics.json` 额外写入：采样间隔和次数、service、speaker、
两个 AED、Punc owner 和临时 overlap worker 的峰值显存、服务 RSS 峰值、GPU 总占用峰值、初始
其他项目显存基线、净服务峰值，以及预热后 ASR 和 diarization 的开始/结束单调时间与耗时。`deep`
级别还会在 `diagnostics/` 写出完整 `scheduler.json` 和 `batch-events.jsonl`；普通摘要不会包含
完整 batch 计划。采样器只接受这些明确登记的本项目 owner 名称，不会写入其他 PID、命令行或显存行。

## 对齐与输出

生产路径中每个 token 按生成它的 diarization 段直接归属说话人，不做覆盖率对齐。
没有 overlap 修改时，最终轮次全文必须与 FireRed 全文在 NFKC 且忽略空白后完全一致；
不一致即失败。`diagnostics.json.asr` 保留 VAD 区间（生产作业为 diarization 段落）、
句级文字、时间范围、AED 置信度和阶段耗时，不包含语言识别结果或语言置信度。

每项成功作业原子发布：

```text
recording_transcript/
├── transcript.txt
├── transcript.json
├── diagnostics.json
└── diagnostics/               # 仅 deep 级别生成
    ├── scheduler.json
    └── batch-events.jsonl
```

## 主要实现位置

- `src/funasr_transcriber/service_protocol.py`：Unix socket 消息、状态文件与客户端。
- `src/funasr_transcriber/service.py`：常驻 owner、FIFO、普通/overlap 作业和服务入口。
- `src/funasr_transcriber/firered_worker.py`：隔离 AED/Punc worker、承载式 FireRedVAD helper
  和 JSONL 服务。
- `src/funasr_transcriber/firered_client.py`：持久 worker、超时、关联和关闭。
- `src/funasr_transcriber/speaker_diarization.py`：3D-Speaker 模型与显式释放。
- `src/funasr_transcriber/overlap.py`：检测/分离、轨道匹配与替换应用。
- `src/funasr_transcriber/resource_profile.py`：只观察本项目 PID 的可选采样器。
- `src/funasr_transcriber/cli.py`：自动启动、提交和 `service status|stop` 命令。

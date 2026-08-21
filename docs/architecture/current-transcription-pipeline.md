# 当前录音转录流程

生产转录由本机常驻服务执行。客户端通过 Unix socket 提交作业；服务只使用物理 GPU 0，
长期持有一个 FireRedWorkerClient 和一个 SpeakerDiarizer。除非收到 `service stop` 或
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
       ├─ FireRedWorkerClient.transcribe(normalized.wav)
       └─ SpeakerDiarizer.diarize(normalized.wav)
              ↑
       ASR 与 diarization 并行，两个分支都收束后才报告错误
              │
       token—speaker 对齐、轮次构建、原子导出 TXT / JSON / diagnostics
              │
       两个热模型继续常驻，供 FIFO 中的下一个作业复用
```

服务启动会先只读检查 GPU 0 的并行预算，然后并发加载 FireRed 和 3D-Speaker。
FireRed 的首个 `status` 请求是加载握手，不执行音频推理。FireRed 仍在独立的
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
作业不会创建 pyannote 或 ClearVoice worker。`--profile-resources` 仅为当前作业启用资源
采样；默认输出的 `resource_profile` 为 `{ "enabled": false }`。

## FireRed worker 边界

服务通过严格 JSON Lines 调用 `.venv-asr` 中的 FireRed worker。每个请求都有唯一 ID，
支持 `transcribe`、`verify`、`status` 和 `shutdown`。ID 不匹配、重复响应、非法 JSON、
超时或 worker 提前退出都会成为转录错误，不会切换旧 ASR。

worker 固定加载 FireRedASR2S 提交
`4e7d9aaf4482a47cec1724807026b9b151926eb5` 的 FireRedVAD、FireRedASR2-AED 和
FireRedPunc。语言识别模型不加载。VAD 先生成保持时间顺序的独立语音段；
`[asr].segment_batch_size` 同时传给官方 AED 与 Punc batch，默认 1，只有 GPU 0 校准
确认输出一致、无超预算后才会提高。官方结果只有句级 `asr_confidence`；稳定协议将句级值
附给所属 token，不虚构词级分数。标点附着到带时间戳 token 后，协议验证全文、句子、
token、VAD 与音频时长的一致性。

## overlap 的受控卸载路径

pyannote/ClearVoice 的 overlap 处理显存远高于常驻 Speaker，不能与热 FireRed 和
3D-Speaker 共驻。仅当当前请求使用 `--enable-overlap` 时，服务按下列顺序执行：

1. 先用热模型完成主 ASR 与 diarization，并由热 Speaker 计算说话人 centroid。
2. FireRed worker 关闭，SpeakerDiarizer 关闭；服务以只读 GPU 采样记录热 owner 的释放。
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
- FireRed worker 关闭与 SpeakerDiarizer 关闭的释放状态会写入 overlap 作业的 GPU stage。
  无法满足释放容差时不会继续 overlap 重型阶段。

启用 `--profile-resources` 后，作业的 `diagnostics.json` 额外写入：采样间隔和次数、
服务/FireRed/临时 overlap worker 的峰值显存、服务 RSS 峰值、FireRed 与 Speaker 的模型
inventory、以及 ASR 和 diarization 的开始/结束单调时间与耗时。采样器只接受这些明确登记
的本项目 owner 名称，不会写入其他 PID、命令行或显存行。

## 对齐与输出

没有 overlap 修改时，最终轮次全文必须与 FireRed 全文在 NFKC 且忽略空白后完全一致；
不一致即失败。`diagnostics.json.asr` 保留 VAD 区间、句级文字、时间范围、AED 置信度和
阶段耗时，不包含语言识别结果或语言置信度。

每项成功作业原子发布：

```text
recording_transcript/
├── transcript.txt
├── transcript.json
└── diagnostics.json
```

## 主要实现位置

- `src/funasr_transcriber/service_protocol.py`：Unix socket 消息、状态文件与客户端。
- `src/funasr_transcriber/service.py`：常驻 owner、FIFO、普通/overlap 作业和服务入口。
- `src/funasr_transcriber/firered_worker.py`：隔离三模块运行时和 JSONL 服务。
- `src/funasr_transcriber/firered_client.py`：持久 worker、超时、关联和关闭。
- `src/funasr_transcriber/speaker_diarization.py`：3D-Speaker 模型与显式释放。
- `src/funasr_transcriber/overlap.py`：检测/分离、轨道匹配与替换应用。
- `src/funasr_transcriber/resource_profile.py`：只观察本项目 PID 的可选采样器。
- `src/funasr_transcriber/cli.py`：自动启动、提交和 `service status|stop` 命令。

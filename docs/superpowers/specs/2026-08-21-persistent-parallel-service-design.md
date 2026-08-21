# 常驻并行转录服务设计

## 1. 目标

将单次 CLI 进程内、严格串行的 GPU 生命周期改为本机常驻转录服务。服务在物理
GPU 0 上长期持有 FireRedASR2S worker 和 3D-Speaker；对普通作业并行运行主 ASR
与说话人识别。客户端仍使用 `funasr-transcribe <audio>`，但服务不存在时自动
启动，并将多个作业按 FIFO 顺序排队。

重叠处理保留为可选能力，但默认不运行。每个转录请求可选择开启资源 profiling；
开启时将可复核的资源数据写入该作业的 `diagnostics.json`。

## 2. 已确认约束

- 仅使用物理 GPU 0 / `cuda:0`；绝不终止、暂停、迁移或以其他方式管理其他项目
  的进程。
- 服务只处理一个 GPU 作业；额外请求进入 FIFO 队列。
- 常规作业中 FireRed worker 与 3D-Speaker 都保持常驻，ASR 和 diarization 并行。
- `transcribe` 自动启动服务；提供显式 `service status` 与 `service stop`。
- `service stop` 停止接收新作业、等待当前作业完成、正常关闭模型和 FireRed 子进程，
  再退出。它只向通过本服务 socket 协议确认身份的本项目服务发请求。
- overlap 默认关闭但仍可显式启用。overlap 作业不能让 pyannote/ClearVoice 与常驻
  FireRed、3D-Speaker 共驻。开始 overlap 重型阶段前，服务自动关闭这两项常驻 owner
  并确认释放；全部 overlap 步骤完成后重新加载热模型。
- 本次范围只支持本机调用。输入、输出、模型目录和工作目录均须对客户端与服务进程
  可见；不提供网络 API、远程客户端或多 GPU 调度。

## 3. 方案选择

### 方案 A：本机 Unix domain socket 服务（采用）

服务状态文件、PID 文件和 Unix socket 位于配置的 `paths.work_dir` 下的服务专用
目录。客户端经一行一个 JSON 的本机协议提交作业、查询状态或请求停止。Unix socket
无需开放 TCP 端口、无需 HTTP 框架，也能以文件权限限定到同一主机用户。

### 方案 B：localhost HTTP 服务

HTTP 对第三方客户端更友好，但必须额外定义端口选择、身份校验和网络错误边界。当前
只有本机 CLI，收益不足，因此不采用。

### 方案 C：在每个 CLI 进程中保存全局模型对象

进程退出即释放模型，无法在下一次 CLI 调用复用，也无法提供可靠的 FIFO 和显式
`stop`，不满足常驻服务目标。

## 4. 命令与协议

### 4.1 CLI

保留现有转录用法，并新增：

```text
funasr-transcribe recording.wav [--enable-overlap] [--profile-resources]
funasr-transcribe service status [--config path]
funasr-transcribe service stop [--config path]
```

`[overlap].enabled` 的默认值改为 `false`。`--enable-overlap` 仅对当前作业启用
overlap；已有 `--disable-overlap` 保持兼容，且与 `--enable-overlap` 互斥。
`--profile-resources` 仅对当前作业启用采样，默认关闭。

客户端的 `transcribe` 流程为：验证物理 GPU 0 限制、连接服务；若 socket 不存在或
握手确认服务已失效，则在持有本项目启动锁时启动服务，等待其 ready，再提交作业。
启动锁只保护本项目 `work_dir` 中已知的路径。发现陈旧 socket/PID 时，客户端先验证
其不是存活且协议匹配的服务，才移除该服务自己的陈旧状态文件。

### 4.2 本机 JSON Lines 协议

请求和响应均是单行 JSON，包含随机 request ID。协议动作为：

- `health`：返回协议版本、服务 PID、队列深度、活动作业 ID、热模型状态和当前只读
  GPU 0 样本。
- `submit`：包含绝对输入路径、绝对输出路径、配置快照/指纹、`overwrite`、
  `keep_intermediate`、`overlap_enabled` 和 `profile_resources`。服务返回请求接受事件，
  后续返回唯一的终态成功或错误事件。
- `stop`：将服务置为 draining，拒绝后续 `submit`，当前作业和已接受队列完成后关闭。

连接中断不取消已接受作业；服务仍完成并发布输出。客户端重连同一 request ID 时可读取
终态结果，避免重复创建作业。错误使用既有领域错误类别及 CLI 退出码映射，不能将
内部 traceback 作为协议结果。

## 5. 服务生命周期和普通作业

```text
客户端提交
    │
    ├─ 服务不存在：启动服务，热加载 FireRed + 3D-Speaker
    ├─ FIFO 等待一个作业槽位
    ├─ 并行分支
    │    ├─ FireRedWorkerClient.transcribe(normalized.wav)
    │    └─ SpeakerDiarizer.diarize(normalized.wav)
    ├─ 等待两个分支；任一失败仍等待另一分支收束，不遗留并发调用
    ├─ token—speaker 对齐、轮次构建和导出
    └─ 两个模型继续常驻，服务处理下一个作业
```

服务启动时在两个受控并发任务中加载 FireRed 与 3D-Speaker。FireRed 的首个 `status`
请求作为加载完成握手；它不执行音频推理。每个作业使用最多两个工作线程：一个独占
FireRed JSONL client，另一个独占 `SpeakerDiarizer`。服务的 FIFO 单消费者保证没有
第二个作业同时触及这两个非线程安全 owner。

主进程绝不导入 FireRed 运行时；FireRed 仍驻留在 `.venv-asr` 子进程中。3D-Speaker
继续在服务的主 Python 环境中驻留。服务闲置时不会执行推理，直到收到 `stop` 或因
overlap 作业进入受控卸载路径才释放模型。

## 6. overlap 作业的受控降级

pyannote overlap 检测约需 11 GiB 显存，不能与 FireRed 和 Speaker 的常驻预算叠加。
为确保不会影响同卡其他项目，`overlap_enabled=true` 的作业采用以下序列：

1. 使用热 FireRed 和热 Speaker 并行完成主 ASR 和全局 diarization；在 Speaker 仍驻留
   时计算并保存该作业的说话人质心。
2. 正常关闭 FireRed 子进程和 Speaker，调用只读 GPU 0 采样确认释放在容差内。
3. 在 overlap 环境中运行 pyannote 检测和 ClearVoice 分离；该 worker 完成后退出并
   确认释放。
4. 临时加载 3D-Speaker，只为分离轨道提取 embedding 并与已保存质心匹配；随后关闭。
5. 临时加载一个 FireRed worker，串行识别接受的轨道并回填 token；随后关闭。
6. 导出结果后，服务重新并行热加载 FireRed 与 Speaker，才接收下一个队列作业。

任意释放确认失败、模型启动失败或轨道识别失败遵循当前的显式错误/逐窗口降级语义；
不会在显存不明的情况下强行继续。默认的非 overlap 作业不创建 overlap worker、也
不导入 pyannote 或 ClearVoice。

## 7. GPU 准入与资源预算

旧版每阶段 `12 GiB` 空闲门槛只适用于串行 owner。服务改为在热加载和普通作业前验证
并行预算：

```text
required free MiB = calibrated FireRed peak
                  + calibrated Speaker peak
                  + configured safety margin
```

初始校准候选值来自本机实测：FireRed 主 ASR峰值约 9,256 MiB，3D-Speaker 完整
diarization 峰值约 618 MiB。因此实现必须先通过可重复 profiling 记录真实并行峰值，
再将默认并行预算写入配置；不得把历史测量值当作跨机器的硬编码事实。GPU 总显存仍须
至少 23 GiB。监控、准入和释放检查只调用只读 `nvidia-smi`，不对其他 PID 发信号。

profile 关闭时仍执行低成本的准入与释放检查；profile 开启时才启动高频采样线程。

## 8. 可选资源 profiling

`--profile-resources` 开启后，服务在该作业的 `diagnostics.json` 写入
`resource_profile`。关闭时该字段只标明 `enabled: false`，不执行周期性采样，也不
写入伪精确的峰值。

开启时记录：

- 采样版本、采样间隔、采样次数、GPU 0 的总量/空闲量；
- 服务进程、FireRed worker 和 overlap worker 的 PID（仅本项目 PID）、逐 PID
  `nvidia-smi` 显存峰值，以及服务进程 CPU RSS 峰值；
- FireRed VAD、LID、AED、Punc 的参数量、参数/持久 buffer 字节数、dtype/device、
  CUDA allocated/reserved，以及 worker 加载时长；
- 3D-Speaker embedding 与 VAD owner 可发现的参数量、张量字节数、加载后的常驻量，
  以及 diarization 期间的 PyTorch 峰值；
- ASR、diarization、对齐、overlap 卸载/检测/分离/重载的开始时间、结束时间、耗时和
  各自的峰值样本；
- 资源采样的不确定性：采样间隔、是否观察到目标 PID、以及驱动 MiB 取整说明。

FireRed worker 扩展 `status` 动作，返回一次性模型 inventory 和当前 PyTorch 内存
快照。3D-Speaker 在服务内用同一套纯函数生成 inventory。采样器只观察明确登记的本
项目 PID 与 GPU 0；它不把其他项目的 PID、命令行或显存数写入 diagnostics。

## 9. 配置、状态文件和恢复

新增服务配置包含 socket 路径、服务状态路径、启动超时、队列限制、profiling 采样间隔、
并行预算和安全余量。客户端提交时携带配置指纹；与运行服务不兼容的模型目录、Python
路径、GPU 设备或预算配置必须明确拒绝，并要求先 `service stop` 再重启，不能悄悄
混用配置。

服务以原子方式写入自身的状态文件。正常关闭清理 socket 和状态；异常退出由下次客户
端安全判定为陈旧并恢复。服务自身日志写入 `work_dir`，每个作业仍使用独立 job 目录和
原子 staging 输出目录。

## 10. 测试与验收

单元测试覆盖协议关联、配置指纹、自动启动状态判定、陈旧状态恢复、FIFO、draining
stop、普通作业双分支并行与失败收束、默认不创建 overlap worker、overlap 的卸载重载
顺序，以及 profile 关闭不创建采样器。

集成测试使用假的 owner、socket 服务和临时状态目录，验证两次客户端调用只加载一次
热模型，作业按 FIFO 完成，`service stop` 正常释放。真实 GPU 验收在 GPU 0 上执行：

1. profile 模式运行短音频，确认 diagnostics 含 inventory 和带采样不确定性的峰值；
2. profile 模式运行长音频，确认 ASR 与 diarization 的实际重叠以及并行峰值未超过
   校准预算；
3. 运行默认作业，确认没有 overlap 进程、且不启用高频采样；
4. 显式 overlap 作业，确认 FireRed/Speaker 在 pyannote 前退出、各重型阶段串行，
   任务完成后热模型恢复；
5. 执行 `service stop`，确认 FireRed 子进程退出、socket 消失、GPU 0 恢复到测试前
   允许误差内的空闲量。

不在本设计中引入多用户网络服务、并发 GPU 作业、多 GPU、跨主机传输或隐式结束其他
项目的 GPU 进程。

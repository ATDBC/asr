# 生产式队列 RAMC 验收与专用输出设计

## 1. 目标

将 RAMC 20 条真实 GPU 验收从“提交一条、等待一条”的串行客户端行为改成与正式
生产调用相同的排队行为：客户端依次提交全部作业，服务立刻确认接收，单个 GPU
消费者按 FIFO 执行。每个作业一旦完成，客户端立即、原子地记录其结果；中途失败
或中断也保留已经完成的录音和可恢复的进度。

验收生成物不再写入 `artifacts/release_validation_corpus/`。该目录只保存输入语料、
校验清单和本机服务工作目录。所有验收输出写到项目根目录的专用、Git 忽略路径：

```text
outputs/release_validation/<run-id>/
```

每次运行保留独立的历史目录；不会覆盖其他运行结果。

## 2. 已确认约束

- 只使用物理 GPU 0 / `cuda:0`，绝不终止、暂停、迁移或管理其他项目进程。
- 不使用 git worktree 或子 agent；直接在当前仓库工作。
- FireRed worker 和 3D-Speaker 的常驻、普通作业内 ASR/diarization 并行，以及
  overlap 默认关闭的既有行为保持不变。
- 服务仍只消费一个 GPU 作业；这次变更扩展的是客户端和 socket 协议的排队语义，
  不是增加多 GPU 或并行 GPU 作业。
- RAMC 每条作业都开启既有的可选资源 profile，并保留每条 `diagnostics.json`。

## 3. 方案选择

### 方案 A：客户端保持 20 个阻塞连接

每个 `submit` 的 handler 持有连接直到作业结束。它能让请求先进入 daemon 队列，
但连接占用、断线语义和测试复杂度都会上升；此外协议仍无法表达“已接受但尚未完成”。
不采用。

### 方案 B：为 RAMC 工具单独调用 daemon 内部对象

RAMC 工具绕过本机协议并直接接触 `DaemonService`。这不能验证实际生产客户端路径，
还会使基准工具与正式 CLI 的行为分叉。不采用。

### 方案 C：两阶段本机协议（采用）

`submit` 只负责校验、去重和入队，立即返回终态为 `false` 的 `accepted` 消息，包含
稳定 `job_id` 和接收时可观察到的队列深度。新增 `wait`，按 `job_id` 等待该 future
并返回唯一的终态消息。正式 CLI 保留 `submit_and_wait()` 这个便捷方法，因此用户命令
的同步体验不变；RAMC 工具则先提交完整批次，再按提交顺序逐项等待和落盘。

## 4. 协议与作业状态

请求的 `action` 扩展为 `health`、`submit`、`wait`、`stop`。`wait` 只接受请求 ID 和
目标 `job_id`；`job_id` 即原 `submit` 的唯一标识符。

```text
submit(job-001) -> {id: job-001, ok: true, terminal: false,
                    status: "accepted", job_id: job-001, queue_depth: 1}
wait(job-001)   -> {id: wait-..., ok: true, terminal: true,
                    status: "completed", job_id: job-001, result: {...}}
```

同一个 `job_id` 再次 submit 不重复入队，并返回同一 accepted 信息；如果作业已经
完成，`wait` 立即返回已缓存的终态。未知 `job_id`、配置指纹不匹配、服务 draining、
队列满和任务失败均返回明确协议错误。客户端连接中断不会取消已接受的作业。

daemon 持续保存已接受 future，直到服务停止。这是有界于服务生命周期的状态，符合
既有单机队列限制；本次不引入跨服务重启的作业恢复或数据库。

## 5. RAMC 批处理与增量输出

RAMC 工具按 TSV 顺序准备 20 条输入及参考数据，生成全局唯一的 run ID，并把每条
发布输出目录设为：

```text
outputs/release_validation/<run-id>/ramc/<recording-stem>/
```

工具在开始推理前写入原子 `progress.json`，其中记录 `queued` 条目及开始时间。全部
提交得到 accepted 后，工具依 TSV 顺序调用 `wait`。某条返回成功时，它读取服务已经
原子发布的 `transcript.json` 和 `diagnostics.json`，计算该条指标，并立即原子重写：

- `progress.json`：每条 `queued` / `completed` / `failed` 状态、job ID、开始和完成
  时间、输出路径或错误文本；
- `ramc-summary.json`：截至当前完成集的逐条结果、可计算的聚合指标和待处理数。

最终 `ramc-summary.json` 的 schema 保持现有 release gate 的 `summary`、`gates`、
`passed`、`release_evidence` 和 `recording_results` 字段，以避免损坏发布检查。进行中
的汇总额外标记 `complete: false`；最后一条处理完成后变为 `complete: true`。若有
失败，工具继续等待已接受的其余作业，写下全部可用结果后以非零退出，并把失败记录
保留在 `progress.json` 与汇总中。

服务的 staging/publish 仍是每个录音输出目录内部的原子发布。基准工具绝不把未完成
目录当成功结果读取。

## 6. 专用输出目录与运行脚本

`outputs/` 加入 `.gitignore`。`scripts/run_validation_linux.sh` 把 `RESULTS_ROOT` 从
`$CORPUS_ROOT/validation_results` 改为：

```text
$PROJECT_ROOT/outputs/release_validation
```

脚本的 doctor、RAMC、overlap、输入矩阵、长音频、服务释放检查和最新汇总继续位于
同一 `<run-id>` 目录下。运行 script 时，RAMC 工具接收该 run 的 `ramc-summary.json`
作为输出参数，并直接把录音导出到 `<run-id>/ramc/<stem>/`，所以不会再创建临时的
`ramc-e2e-*` 输出目录。

语料目录的 runtime 忽略项只保留 `.service-work`；旧的 `validation_results` 被保留在
语料验证器的忽略集合中以兼容历史本地生成物，但任何新运行不得创建它。

## 7. 错误处理与可观测性

- 提交阶段失败时，工具立即写入该条失败状态；因为后续输入还未全部 accepted，工具
  不再提交新条目，但会等待此前已经 accepted 的所有作业并落盘。
- 等待阶段失败时，工具继续收集其余已接受作业。终态错误包含 job ID 和领域错误文本，
  不泄露 traceback。
- `stop` 仍将服务置为 draining。它会处理已接受队列，且现有等待方正常得到终态。
- `progress.json` 与 `ramc-summary.json` 都使用同目录临时文件 + replace 写入，以免
  电源中断留下半个 JSON 文件。
- run ID、job ID、队列深度、每条完成时间和 profile 输出路径使真实队列次序与资源
  结果可复核；不会记录其他项目的进程或 GPU 细节。

## 8. 测试与验收

单元测试覆盖：

1. `submit` 在阻塞作业执行时立即返回 accepted，而不是等待 future；
2. 多个 job 先全部 accepted、之后按 FIFO 运行，`wait` 获得正确终态；
3. `submit_and_wait()` 仍为正式 CLI 提供同步结果；
4. 重复 submit 不重复入队，未知 job 的 `wait` 明确失败；
5. RAMC 工具先提交完整批次才等待，且每成功一条就更新原子 `progress.json` 和
   `ramc-summary.json`；失败不会抹去先前结果；
6. runner 仅把输出写在 `$PROJECT_ROOT/outputs/release_validation/<run-id>`，并仍在
   正常/异常退出时停止本项目服务；
7. `.gitignore` 覆盖 `outputs/`，输入语料不再承载新输出。

真实 GPU 0 验收使用已验证的 20 条 RAMC 语料：启动服务一次，记录 20 个 accepted
响应和 FIFO job ID 序列，逐条观察 `progress.json` 变化，确认每条 output 及
`diagnostics.json` 位于该 run 的专用目录。完成后正常执行 `service stop`，确认只由
本项目启动的 FireRed worker 和服务退出，GPU 0 回到启动前允许误差内的可用显存。

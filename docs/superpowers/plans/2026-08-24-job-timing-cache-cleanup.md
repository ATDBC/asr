# Job Timing And AED Cache Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 只排除首次模型加载，在首任务计入预热，并在每个任务结束前并行清理两个 AED worker 的 CUDA cache。

**Architecture:** FireRed worker 提供独立的 `cleanup_cuda_cache` 协议操作，client 和双 worker owner 分层聚合结果。`HotModelHost` 将加载与预热拆开，`ServiceJobRunner` 在统一计时边界内执行首次预热和逐任务清理，再输出兼容且语义明确的诊断字段。

**Tech Stack:** Python 3.11、PyTorch CUDA、JSONL 子进程协议、pytest、现有常驻服务与 resource sampler。

---

### Task 1: 增加单 worker 缓存清理协议

**Files:**
- Modify: `src/funasr_transcriber/firered_worker.py`
- Modify: `src/funasr_transcriber/firered_client.py`
- Test: `tests/unit/test_firered_worker.py`
- Test: `tests/unit/test_firered_client.py`

- [ ] **Step 1: 写 worker 与 client 失败测试**

  worker 测试注入 GC、CUDA 同步、empty-cache 和显存读数，断言调用顺序及前后 allocated/reserved；client 测试使用假 JSONL worker，断言 `cleanup_cuda_cache()` 校验并返回结构化结果。

- [ ] **Step 2: 运行聚焦测试确认按预期失败**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_firered_worker.py tests/unit/test_firered_client.py -k 'cleanup_cuda_cache'`

  Expected: FAIL，因为 action 和 client 方法尚不存在。

- [ ] **Step 3: 实现最小协议**

  在 worker 中新增 `_cleanup_cuda_cache()`，严格执行 GC、同步、`empty_cache()`、同步并返回指标；在 `serve()` 中无需音频路径处理 action。在 client 中新增 `cleanup_cuda_cache()` 并拒绝缺失或非对象结果。

- [ ] **Step 4: 重跑聚焦测试**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_firered_worker.py tests/unit/test_firered_client.py -k 'cleanup_cuda_cache or warmup'`

  Expected: PASS。

### Task 2: 并行清理两个 AED worker

**Files:**
- Modify: `src/funasr_transcriber/firered_parallel.py`
- Test: `tests/unit/test_firered_parallel.py`

- [ ] **Step 1: 写并发和失败传播测试**

  用两个可阻塞 fake worker 证明两个 cleanup 同时开始，返回 `workers` 与 wall `seconds`；另一个测试证明任一 worker 失败会传播异常，且 Punc owner 不收到 cleanup。

- [ ] **Step 2: 运行测试确认失败**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_firered_parallel.py -k cleanup`

  Expected: FAIL，因为 parallel owner 尚无 cleanup API。

- [ ] **Step 3: 实现并行聚合**

  使用 `ThreadPoolExecutor(max_workers=2)` 并行调用 `worker0/worker1.cleanup_cuda_cache()`；返回宿主 wall time和两个 worker 结果，不操作独立 Punc owner。

- [ ] **Step 4: 重跑 parallel 测试**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_firered_parallel.py`

  Expected: PASS。

### Task 3: 调整首次预热和作业计时边界

**Files:**
- Modify: `src/funasr_transcriber/service.py`
- Modify: `tests/unit/test_service.py`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/architecture/current-transcription-pipeline.md`

- [ ] **Step 1: 写 HotModelHost 与 ServiceJobRunner 失败测试**

  断言 `start()` 只加载不预热、`warmup()` 只执行一次、首任务 diagnostics 计入 warmup、后续任务不预热、每个任务在端到端停止前 cleanup；清理失败令成功路径失败，主流程失败时仍尝试 cleanup。

- [ ] **Step 2: 运行 service 聚焦测试确认失败**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_service.py -k 'warmup or cleanup or timing'`

  Expected: FAIL，因为现有 `start()` 会预热且 job runner 不清理。

- [ ] **Step 3: 实现生命周期和诊断**

  从 `HotModelHost.start()` 移除 warmup，新增幂等 `warmup()`。在 `measurement_started` 与 sampler 启动后执行首次 warmup；成功输出前执行 cleanup，记录 `warmup`、`cache_cleanup`、`end_to_end_after_load` 和兼容别名，更新 runtime timing contract。

- [ ] **Step 4: 更新中文生产约定和文档**

  明确只排除首次加载、首任务预热计时、逐任务 AED cache 清理计时，以及 Punc owner 不参与该清理。

- [ ] **Step 5: 运行聚焦与完整测试并提交**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_firered_client.py tests/unit/test_firered_worker.py tests/unit/test_firered_parallel.py tests/unit/test_service.py`

  Run: `.venv/bin/python -m pytest -q`

  Run: `git diff --check`

  Expected: 全部通过；提交聚焦实现与文档。

### Task 4: 重跑 GPU0 验收

**Files:**
- Create ignored reports below: `outputs/measurements/20260824T-job-timing-cache-cleanup/`
- Create ignored reports below: `outputs/measurements/20260824T-three-jobs-cache-cleanup/`
- Create ignored reports below: `outputs/measurements/20260824T-aed-budget-matrix-cache-cleanup/`

- [ ] **Step 1: 只读确认 GPU0 与本项目服务状态**

  使用 `nvidia-smi -i 0` 和本项目 `service status`，不向外部 PID 发送信号；只在必要时通过本项目 `service stop` 关闭本项目已有服务。

- [ ] **Step 2: 跑一个 deep 单任务**

  使用生产 BF16 配置和 `audio_test/刘洋洋.wav`。确认首任务存在 warmup、每任务存在 cleanup、端到端包含二者，且 worker reserved 在清理后下降。

- [ ] **Step 3: 同一常驻服务连续跑三个任务**

  三项请求全部使用相同音频。确认只有第一项预热、三项均清理，并比较三项端到端/AED 时间与清理后显存。

- [ ] **Step 4: 核对 deep 显存时序**

  从第三步或独立 deep 任务读取 `aed_memory_trace`，报告 AED 峰值利用率与清理后低水位，不能把清理阶段低水位混入 AED 峰值定义。

- [ ] **Step 5: 重跑预算矩阵**

  Run: `.venv/bin/python tools/benchmark_aed_budget_matrix.py audio_test/刘洋洋.wav --config outputs/measurements/20260822T-memory-first-aed-liuyangyang/config.toml --output-root outputs/measurements/20260824T-aed-budget-matrix-cache-cleanup --fractions 0.60 0.70 0.80 0.90 0.95 --repeats 3 --diagnostics deep`

  Expected: manifest 逐项落盘；OOM、协议失败、安全余量不足和 fingerprint 不稳定仍作为失败。

- [ ] **Step 6: 汇总并提交测量说明**

  记录实际精度、各阶段耗时、显存峰值、cache 前后值、预算候选结果和命令时间；报告只引用本次新鲜输出。

# AED Memory Model Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 统一 AED 显存观测口径，覆盖生产形状预热，建立跨任务的安全成本校准，并在校准后优化双 AED 的批次与 makespan 分配。

**Architecture:** 保留现有双 AED + 单 Punc owner 和完整计划不变。先让资源采样和逐批诊断同时提供 allocated、reserved、driver-visible 三套证据，并按 PID 去重；再在 worker 生命周期内用已完成批次更新成本模型，后续任务使用校准模型生成完整计划。调度阶段使用多特征成本估计和 lane-aware makespan 分配，最终用同一常驻服务状态下的预算矩阵验证，不把未验证的参数直接替换为生产硬编码。

**Tech Stack:** Python 3.11、PyTorch CUDA/BF16、`nvidia-smi` 只读采样、现有 JSONL worker 协议、pytest。

---

### Task 1: 修正资源采样口径并增加 AED 预算利用率

**Files:**
- Modify: `src/funasr_transcriber/resource_profile.py:100-470`
- Modify: `src/funasr_transcriber/service.py:637-690`
- Test: `tests/unit/test_resource_profile.py`
- Test: `tests/unit/test_service.py`

- [ ] **Step 1: Write failing tests for PID deduplication and budget metrics**

  Add a sampler test where `service` and `speaker` resolve to the same PID; assert the trace sample’s project memory counts that PID once. Add a stage summary fixture with `stage_start_used_mib=12059`, `peak_used_mib=20881`, `effective_budget_mib=10980`, `safety_margin_mib=1536` and assert:

  ```python
  assert summary["aed_transient_increment_mib"] == 8822
  assert summary["budget_utilization_pct"] == 80.346
  assert summary["remaining_over_safety_mib"] == 2159
  ```

- [ ] **Step 2: Run the focused tests and verify they fail**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_resource_profile.py tests/unit/test_service.py -k 'dedup or budget'`

  Expected: FAIL because the sampler currently sums owner labels and does not expose AED budget metrics.

- [ ] **Step 3: Implement PID-normalized trace aggregation**

  Store both owner-to-PID and unique-PID memory maps in each trace sample. Compute project memory from unique PIDs only. Add `stage_start_used_mib`, `aed_transient_increment_mib`, `effective_budget_mib`, `budget_utilization_pct`, and `remaining_over_safety_mib` to the AED summary. Obtain the budget and safety values from the recorded `parallel.memory` fields, without querying or managing unrelated PIDs.

- [ ] **Step 4: Run the focused tests and the existing deep-trace tests**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_resource_profile.py tests/unit/test_service.py tests/unit/test_diagnostics.py`

  Expected: all tests pass and `resources` output remains backward-compatible apart from additive fields.

- [ ] **Step 5: Commit the measurement contract**

  ```bash
  git add src/funasr_transcriber/resource_profile.py src/funasr_transcriber/service.py tests/unit/test_resource_profile.py tests/unit/test_service.py
  git commit -m "fix: normalize AED memory utilization metrics"
  ```

### Task 2: Make worker warmup cover representative production shapes

**Files:**
- Modify: `src/funasr_transcriber/firered_worker.py:428-452`
- Modify: `src/funasr_transcriber/firered_client.py:519-528`
- Modify: `src/funasr_transcriber/service.py:90-140`
- Test: `tests/unit/test_firered_worker.py`
- Test: `tests/unit/test_firered_client.py`
- Test: `tests/unit/test_service.py`

- [ ] **Step 1: Write failing tests for representative warmup calls**

  Use a fake ASR system that records input batch sizes and durations. Assert that AED warmup executes short-single, short-batch, medium, and long representative shapes; assert all warmup responses still mark `excluded=True`. Keep Punc-only warmup unchanged.

- [ ] **Step 2: Run the focused warmup tests and verify they fail**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_firered_worker.py tests/unit/test_firered_client.py -k warmup`

  Expected: FAIL because the current worker sends exactly one short utterance to AED.

- [ ] **Step 3: Implement shape-only warmup**

  In `_warmup_process`, load the configured warmup audio once, derive bounded in-memory waveforms for short, medium, and long durations, and run a short multi-item batch plus one medium and one long item. Use unique synthetic utterance IDs, synchronize CUDA after each shape, and never write an output file. Return additive warmup shape metadata so diagnostics identify what was excluded from timing.

- [ ] **Step 4: Run warmup tests and verify service startup metadata**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_firered_worker.py tests/unit/test_firered_client.py tests/unit/test_service.py -k warmup`

  Expected: all warmup tests pass and the returned warmup record lists the representative shapes.

- [ ] **Step 5: Commit representative warmup**

  ```bash
  git add src/funasr_transcriber/firered_worker.py src/funasr_transcriber/firered_client.py src/funasr_transcriber/service.py tests/unit/test_firered_worker.py tests/unit/test_firered_client.py tests/unit/test_service.py
  git commit -m "feat: warm FireRed workers with representative shapes"
  ```

### Task 3: Separate cost metrics and calibrate the in-memory model

**Files:**
- Modify: `src/funasr_transcriber/firered_scheduler.py:20-120`
- Modify: `src/funasr_transcriber/firered_parallel.py:200-250,270-700`
- Modify: `tests/unit/test_firered_scheduler.py`
- Modify: `tests/unit/test_firered_parallel.py`

- [ ] **Step 1: Write failing tests for multi-metric observations**

  Add tests that estimate a batch with `total_frames`, `padded_frames`, `max_frames`, and `item_count`; observe actual seconds and reserved incremental MiB; assert `observations` increments and later estimates use updated coefficients. Assert allocated and reserved errors are both retained in prediction diagnostics.

- [ ] **Step 2: Run the focused scheduler tests and verify they fail**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_firered_scheduler.py tests/unit/test_firered_parallel.py -k 'observe or prediction or estimate'`

  Expected: FAIL because the current model only uses padded frames and `observe()` is not connected.

- [ ] **Step 3: Extend `BatchEstimate` and `BatchCostModel` features**

  Add max-frame and total-frame features to the estimate. Keep a non-negative conservative intercept and use robust exponential updates from actual completed batches. Make `observe()` accept actual incremental reserved memory and actual seconds; do not mutate the plan currently executing. Keep the model instance owned by the persistent `ParallelFireRedWorkerClient`, so updates affect only later jobs in the same service lifetime.

- [ ] **Step 4: Record allocated and reserved prediction accuracy separately**

  In `_prediction_accuracy`, compute `actual_allocated_incremental_mib` and `actual_reserved_incremental_mib` from each worker response’s resident baselines. Use reserved as the calibration signal, retain allocated as a diagnostic lower bound, and include the selected calibration metric explicitly in the response.

- [ ] **Step 5: Connect post-job observations to future plans**

  After a successful AED plan, feed only completed batch records into the persistent model. Never update before the current plan is complete and never let a failed/partial batch update coefficients. Add a model observation count and coefficients to deep scheduler diagnostics.

- [ ] **Step 6: Run focused tests and commit calibration**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_firered_scheduler.py tests/unit/test_firered_parallel.py`

  ```bash
  git add src/funasr_transcriber/firered_scheduler.py src/funasr_transcriber/firered_parallel.py tests/unit/test_firered_scheduler.py tests/unit/test_firered_parallel.py
  git commit -m "feat: calibrate AED cost model from completed batches"
  ```

### Task 4: Improve two-lane packing and makespan prediction

**Files:**
- Modify: `src/funasr_transcriber/firered_scheduler.py:129-220`
- Modify: `src/funasr_transcriber/firered_parallel.py:100-190,520-700`
- Test: `tests/unit/test_firered_scheduler.py`
- Test: `tests/unit/test_firered_parallel.py`

- [ ] **Step 1: Write failing tests for feature-aware estimates and lane balancing**

  Add a fixture with equal padded frames but different maximum segment durations; assert the long-segment batch receives a larger predicted time. Add an unequal-worker-resident fixture; assert lane budgets are derived from available shared budget and each worker’s resident baseline. Add a deterministic partition test asserting predicted lane loads differ by no more than the best candidate tolerance.

- [ ] **Step 2: Run the focused tests and verify they fail**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_firered_scheduler.py tests/unit/test_firered_parallel.py -k 'duration or lane or partition'`

  Expected: FAIL because current time estimates ignore maximum duration and lanes always receive half the budget.

- [ ] **Step 3: Implement feature-aware cost estimates**

  Use padded frames for encoder padding, total frames for aggregate work, max frames for decoder/long-utterance overhead, and item count for per-item overhead. Keep all coefficients non-negative and expose them in scheduler diagnostics.

- [ ] **Step 4: Implement deterministic lane-aware partitioning**

  Generate complete candidate batches first, then evaluate worker assignments using predicted makespan and each worker’s resident/transient capacity. Preserve batch indivisibility, deterministic tie-breaking, and serial fallback when the joint predicted capacity cannot fit.

- [ ] **Step 5: Run all scheduler and parallel tests and commit**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_firered_scheduler.py tests/unit/test_firered_parallel.py`

  ```bash
  git add src/funasr_transcriber/firered_scheduler.py src/funasr_transcriber/firered_parallel.py tests/unit/test_firered_scheduler.py tests/unit/test_firered_parallel.py
  git commit -m "feat: balance AED lanes with duration-aware costs"
  ```

### Task 5: Run controlled budget A/B validation

**Files:**
- Create: `tools/benchmark_aed_budget_matrix.py`
- Modify: `README.md`
- Test: `tests/unit/test_aed_budget_matrix.py`
- Output: `outputs/measurements/20260822T-aed-budget-matrix/`

- [ ] **Step 1: Write tests for the matrix manifest and safety checks**

  Assert the tool accepts one audio path, an existing config, explicit budget fractions, repeat count, and output directory; emits one manifest record per fraction/repeat; rejects non-GPU0 devices and never includes external PID rows.

- [ ] **Step 2: Run the tool tests and verify they fail**

  Run: `.venv/bin/python -m pytest -q tests/unit/test_aed_budget_matrix.py`

  Expected: FAIL because the matrix tool does not exist.

- [ ] **Step 3: Implement same-service-state matrix execution**

  Reuse the resident service, run each budget fraction in deterministic order, repeat each candidate three times, and write only under `outputs/measurements/`. Capture AED wall time, worker times, full-card and per-PID peaks, transient budget utilization, output fingerprint, diagnostics level, precision, and errors. Do not stop or inspect unrelated project processes.

- [ ] **Step 4: Run the matrix on GPU0 after representative warmup**

  Run the tool against `audio_test/刘洋洋.wav` with fractions `0.60 0.70 0.80 0.90 0.95`, three repeats, and `--diagnostics deep`. Treat OOM, protocol failures, fingerprint changes, or safety violations as failed candidates.

- [ ] **Step 5: Compare and record the selected operating point**

  Select the lowest median AED wall time whose output fingerprint is stable, no safety violation occurs, and the P99 remaining free memory remains at or above the configured safety margin. Update the architecture analysis with fresh evidence; do not change production defaults without this acceptance criterion.

- [ ] **Step 6: Run final verification and commit the matrix tool/docs**

  Run: `.venv/bin/python -m pytest -q`, `git diff --check`, and inspect `git status --short`.

  ```bash
  git add tools/benchmark_aed_budget_matrix.py README.md tests/unit/test_aed_budget_matrix.py docs/architecture/2026-08-22-aed-memory-utilization-analysis.md
  git commit -m "test: benchmark AED budget operating points"
  ```

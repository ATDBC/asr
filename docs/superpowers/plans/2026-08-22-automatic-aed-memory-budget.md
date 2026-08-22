# Automatic AED Memory Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive a shared two-AED transient-memory budget from GPU 0 after warm-up and record real per-batch time/memory prediction accuracy.

**Architecture:** Keep one immutable duration-aware plan, but derive its memory capacity from a read-only GPU snapshot after diarization/VAD. Execute the two fixed lanes concurrently when the second capacity check permits and serially otherwise; collect worker-side CUDA observations without allowing them to alter the plan.

**Tech Stack:** Python 3.11, dataclasses, JSONL worker protocol, `nvidia-smi`, PyTorch CUDA memory counters, pytest.

---

### Task 1: Add BF16 forced-alignment compatibility

**Files:**
- Create: `src/funasr_transcriber/firered_compat.py`
- Modify: `src/funasr_transcriber/firered_worker.py`
- Create: `tests/unit/test_firered_compat.py`

- [x] Write a failing test whose fake forced-align function receives FP32 for BF16 input and receives the original object for FP32 input.
- [x] Run `.venv/bin/python -m pytest -q tests/unit/test_firered_compat.py` and confirm the module is missing.
- [x] Implement an idempotent wrapper and install it before constructing the pinned FireRed system.
- [x] Re-run the focused test and the worker tests.

### Task 2: Model shared transient memory and fixed-plan capacity

**Files:**
- Modify: `src/funasr_transcriber/firered_scheduler.py`
- Modify: `src/funasr_transcriber/firered_parallel.py`
- Modify: `tests/unit/test_firered_scheduler.py`
- Modify: `tests/unit/test_firered_parallel.py`

- [x] Add failing tests for BF16-calibrated incremental estimates, `free - 1536` budget derivation, two-lane peak-fit planning, and deterministic behavior.
- [x] Run the focused scheduler/parallel tests and confirm each new assertion fails for the intended missing behavior.
- [x] Add predicted incremental memory to `BatchEstimate`, query only GPU 0 totals and owned PIDs, and constrain the immutable plan so the largest possible concurrent lane increments fit the shared budget.
- [x] Re-run focused tests.

### Task 3: Add dispatch gating and prediction observations

**Files:**
- Modify: `src/funasr_transcriber/firered_worker.py`
- Modify: `src/funasr_transcriber/firered_parallel.py`
- Modify: `tests/unit/test_firered_worker.py`
- Modify: `tests/unit/test_firered_parallel.py`

- [x] Write failing tests for per-batch elapsed/allocated/reserved metrics, concurrent dispatch, serial fallback, capacity failure, and estimate/actual error records.
- [x] Run the focused tests and verify the failures describe absent metrics and gating.
- [x] Return the warmed allocated baseline plus per-batch CUDA metrics from each fixed-plan request; sample capacity before planning and dispatch; preserve the plan while selecting concurrent or serial execution; reject the job if even one planned batch cannot fit; publish joined diagnostics.
- [x] Run the focused worker/parallel tests.

### Task 4: Expose and validate the safety margin

**Files:**
- Modify: `src/funasr_transcriber/config.py`
- Modify: `config/default.toml`
- Modify: `src/funasr_transcriber/service.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_service.py`
- Modify: `AGENT.md`
- Modify: `docs/architecture/current-transcription-pipeline.md`

- [x] Add failing tests for the `1536 MiB` default, non-positive rejection, and propagation into the parallel owner.
- [x] Run the config/service tests and verify the expected failures.
- [x] Add `gpu_safety_margin_mib`, retain the old per-worker limit only as an emergency ceiling, wire the owner, and document the post-warm-up budget.
- [x] Re-run config/service tests.

### Task 5: Verify and measure on GPU 0

**Files:**
- Create ignored report: `outputs/measurements/20260822T-auto-aed-budget-liuyangyang/result.json`

- [x] Run `.venv/bin/python -m pytest -q` and `git diff --check`.
- [x] Read GPU 0 state without changing any process and confirm enough capacity for the service.
- [x] Start only this repository's resident service, run `audio_test/刘洋洋.wav` with resource profiling after warm-up, then stop only this service.
- [x] Compute per-batch and aggregate elapsed-time and incremental-memory prediction errors from the generated diagnostics.
- [x] Confirm GPU 0 returns to the pre-test external baseline and commit all tracked changes.

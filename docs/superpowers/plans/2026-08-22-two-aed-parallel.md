# Two-AED Parallel Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one normal job through two fixed-plan BF16 AED workers and one shared Punc/VAD owner.

**Architecture:** Add a pure batch partition helper, extend the JSONL worker with explicit plans and VAD/Punc actions, and wrap two clients in a parallel owner that merges by original index. Keep the existing single client for compatibility paths.

**Tech Stack:** Python 3.11, dataclasses, JSONL subprocess protocol, pytest, existing `AdaptiveBatchScheduler`.

---

### Task 1: Add deterministic batch partitioning

**Files:**
- Create: `src/funasr_transcriber/firered_parallel.py`
- Test: `tests/unit/test_firered_parallel.py`

- [ ] Write tests for LPT assignment, deterministic tie-breaking, and no batch splitting.
- [ ] Run the focused tests and confirm they fail because the module is absent.
- [ ] Implement `PlannedBatch`, `partition_batches`, and `serialize_batch_plan` using predicted seconds and pairwise swap improvement.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Extend worker protocol for explicit AED plans and shared Punc

**Files:**
- Modify: `src/funasr_transcriber/firered_worker.py`
- Modify: `src/funasr_transcriber/firered_client.py`
- Test: `tests/unit/test_firered_worker.py`
- Test: `tests/unit/test_firered_client.py`

- [ ] Add failing tests for client command flags, explicit batch execution, VAD refinement, and punctuation request serialization.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Add worker flags for disabling VAD/Punc, `refine_vad`, `batch_plan`, per-request punctuation selection, and `punctuate` handling.
- [ ] Add client methods that validate and serialize these requests while retaining the existing API defaults.
- [ ] Run focused worker/client tests.

### Task 3: Implement the two-AED owner and result merge

**Files:**
- Modify: `src/funasr_transcriber/firered_parallel.py`
- Test: `tests/unit/test_firered_parallel.py`

- [ ] Add failing tests for parallel lifecycle, global-plan construction, original-index merge, and single-owner punctuation.
- [ ] Implement `ParallelFireRedWorkerClient` with concurrent AED calls, worker-0 global VAD/Punc, strict failure propagation, and diagnostic aggregation.
- [ ] Run focused tests and then the worker/client unit tests.

### Task 4: Integrate configuration, host lifecycle, and normal branches

**Files:**
- Modify: `src/funasr_transcriber/config.py`
- Modify: `config/default.toml`
- Modify: `src/funasr_transcriber/service.py`
- Modify: `src/funasr_transcriber/resource_profile.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_service.py`

- [ ] Add failing tests for `aed_worker_count`, host ownership of two clients, VAD refinement before ASR, and merged diagnostics.
- [ ] Add validated config field with default `2`, construct worker 0/1 with the correct VAD/Punc flags, and close both only through the owned host.
- [ ] Update normal branch execution to use refined VAD segments when the owner supports it and preserve speaker attribution on original diarization segments.
- [ ] Update resource profiling to register both owned AED PIDs only.
- [ ] Run focused service/config tests.

### Task 5: Documentation, verification, and commit

**Files:**
- Modify: `AGENT.md`
- Modify: `README.md`
- Modify: `docs/architecture/current-transcription-pipeline.md`

- [ ] Document the two-AED/one-Punc topology, capacity caveat, and fixed batch partition.
- [ ] Run `pytest -q`, `git diff --check`, and inspect Git status.
- [ ] Commit all implementation and documentation changes with `feat: add two-AED fixed-plan inference`.

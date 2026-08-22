# Memory-First AED Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the fixed AED batch/memory controls and make every AED batch a deterministic memory-first packing result from the post-warm-up GPU0 budget.

**Architecture:** Replace the old limit-driven scheduler with a pure greedy packer whose only resource input is the shared transient budget and each worker's warmed resident baseline. Build all batches before AED, assign complete batches to two workers, gate concurrent versus serial dispatch with a second GPU0 snapshot, and retain per-batch prediction diagnostics.

**Tech Stack:** Python 3.11, dataclasses, JSONL worker protocol, PyTorch CUDA counters, `nvidia-smi`, pytest.

---

### Task 1: Remove obsolete configuration and protocol controls

**Files:**
- Modify: `src/funasr_transcriber/config.py`
- Modify: `config/default.toml`
- Modify: `src/funasr_transcriber/firered_client.py`
- Modify: `src/funasr_transcriber/firered_worker.py`
- Modify: `src/funasr_transcriber/service.py`
- Modify: `src/funasr_transcriber/doctor.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_firered_client.py`
- Modify: `tests/unit/test_firered_worker.py`

- [ ] Write failing tests asserting `AsrConfig` has only `gpu_safety_margin_mib` for AED resource control, default TOML omits the four obsolete keys, and the worker command contains no obsolete flags.
- [ ] Run `.venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_firered_client.py tests/unit/test_firered_worker.py` and verify failures mention the obsolete fields still exist.
- [ ] Remove the four keys from defaults, dataclasses, validation, client constructor/command, worker parser/build function, service factories, and doctor factory; keep Punc's internal model batch setting independent and fixed.
- [ ] Run the focused tests and update all affected existing assertions to the new API.

### Task 2: Replace scheduler with memory-first greedy packing

**Files:**
- Modify: `src/funasr_transcriber/firered_scheduler.py`
- Modify: `src/funasr_transcriber/firered_parallel.py`
- Modify: `tests/unit/test_firered_scheduler.py`
- Modify: `tests/unit/test_firered_parallel.py`

- [ ] Write failing tests for: a budget of 1000 MiB producing different batch cardinalities as segment durations change; no fixed batch-size cap; exact original-index preservation; singleton capacity failure; and two-lane peak fitting.
- [ ] Run the focused scheduler/parallel tests and verify the new tests fail because the old scheduler still requires `max_batch_size`, frame budget, and static memory limits.
- [ ] Implement `MemoryFirstBatchPacker` with inputs `(segments, shared_budget_mib, worker_resident_mib, cost_model)`; greedily add duration-sorted segments while predicted padded-frame transient growth fits; close batches before overflow; raise a typed capacity error for an oversized singleton.
- [ ] Replace `build_global_plan()` with the packer, passing actual worker baselines and shared budget; preserve deterministic LPT lane assignment and no mid-run replanning.
- [ ] Run focused scheduler/parallel tests until green.

### Task 3: Wire actual worker baselines and dispatch diagnostics

**Files:**
- Modify: `src/funasr_transcriber/firered_parallel.py`
- Modify: `src/funasr_transcriber/firered_worker.py`
- Modify: `src/funasr_transcriber/service.py`
- Modify: `tests/unit/test_firered_parallel.py`
- Modify: `tests/unit/test_firered_worker.py`

- [ ] Write failing tests proving the planner subtracts worker-specific warmed allocations, emits exact batch membership, and reports predicted/actual time and allocated/reserved memory per batch.
- [ ] Run focused tests and verify failures identify the old hard-coded `5000 MiB` baseline or missing fields.
- [ ] Return warmed allocated/reserved baselines from worker warm-up/status, pass them into the packer, and expose `memory_first` diagnostics with budget, batch list, worker lanes, dispatch mode, and errors.
- [ ] Keep the second GPU snapshot as a gate only: concurrent if both lanes fit, serial if only one lane fits, fail before AED if a singleton cannot fit.
- [ ] Run focused tests and inspect serialized JSON protocol requests for absence of obsolete options.

### Task 4: Update documentation and remove stale references

**Files:**
- Modify: `README.md`
- Modify: `AGENT.md`
- Modify: `docs/architecture/current-transcription-pipeline.md`
- Modify: `docs/superpowers/specs/2026-08-22-automatic-aed-memory-budget-design.md`
- Modify: `docs/superpowers/plans/2026-08-22-automatic-aed-memory-budget.md`
- Modify: `tests/integration/test_firered_documentation.py`

- [ ] Add failing documentation assertions that the active pipeline describes memory-first packing and does not present the four removed settings as production controls.
- [ ] Run documentation tests and verify the stale references fail.
- [ ] Rewrite active documentation around `gpu_safety_margin_mib`, dynamic cardinality, actual worker baselines, and the dispatch gate; leave historical migration records untouched where they describe past designs.
- [ ] Run documentation tests and `rg` to distinguish only historical references from active configuration references.

### Task 5: Verify, benchmark, and commit

**Files:**
- Create ignored report: `outputs/measurements/20260822T-memory-first-aed-liuyangyang/result.json`

- [ ] Run `.venv/bin/python -m pytest -q` and `git diff --check`.
- [ ] Read GPU0 state without changing external processes; start the repository service with the new config and run `audio_test/刘洋洋.wav` using resource profiling.
- [ ] Verify the report includes dynamically chosen batch sizes, peak budget utilization, stage timing, prediction errors, and transcript fingerprint; compare against the previous 102-batch run.
- [ ] Stop only this service, confirm GPU0 returns to the external baseline, inspect `git status`, and commit all tracked changes.

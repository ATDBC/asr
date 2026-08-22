# Diagnostics Standardization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize production diagnostics into basic/resources/deep levels, report all resident FireRed owners correctly, separate summary metrics from large traces, expose Punc tuning in configuration, and provide Chinese repository agent instructions.

**Architecture:** Basic stage timing remains always-on with explicit wall/model/normalization names. Resource sampling becomes opt-in and observes named owners (`aed_worker_0`, `aed_worker_1`, `punc_worker`, service, speaker, overlap) without touching foreign processes. Deep traces are emitted as separate JSON/JSONL files, while `diagnostics.json` stays compact and stable.

**Tech Stack:** Python 3.11, argparse, TOML config, JSON/JSONL atomic output, pytest, read-only `nvidia-smi` sampling.

---

### Task 1: Add diagnostics levels and config validation

**Files:**
- Modify: `src/funasr_transcriber/config.py`, `config/default.toml`
- Modify: `src/funasr_transcriber/cli.py`, `src/funasr_transcriber/service_protocol.py`
- Test: `tests/unit/test_config.py`, `tests/unit/test_cli.py`, `tests/unit/test_service_protocol.py`

- [x] **Step 1: Write failing tests**

  Test default `diagnostics.level == "basic"`, accepted values `off|basic|resources|deep`, positive sampling interval, CLI `--diagnostics` propagation, and backward-compatible `--profile-resources` mapping to `resources`.

- [x] **Step 2: Run focused tests and confirm failure**

  Run `.venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_cli.py tests/unit/test_service_protocol.py`.

- [x] **Step 3: Implement configuration and request propagation**

  Add a diagnostics config section with `level`, `sampling_interval_ms`, `include_model_inventory`, and `write_batch_trace`. Add mutually compatible CLI handling where explicit `--diagnostics` wins and `--profile-resources` is an alias. Carry the resolved level in `ServiceRequest` without changing the existing output contract.

- [x] **Step 4: Run focused tests**

  Re-run the same command and verify all new and existing assertions pass.

- [x] **Step 5: Commit**

  ```bash
  git add src/funasr_transcriber/config.py config/default.toml src/funasr_transcriber/cli.py src/funasr_transcriber/service_protocol.py tests/unit/test_config.py tests/unit/test_cli.py tests/unit/test_service_protocol.py
  git commit -m "feat: add explicit diagnostics levels"
  ```

### Task 2: Correct named resource ownership and opt-in sampling

**Files:**
- Modify: `src/funasr_transcriber/service.py`, `src/funasr_transcriber/resource_profile.py`
- Modify: `src/funasr_transcriber/diagnostics.py`
- Test: `tests/unit/test_service.py`, `tests/unit/test_resource_profile.py`, `tests/unit/test_diagnostics.py`

- [x] **Step 1: Write failing owner-registration tests**

  Assert the sampler receives `service`, `speaker`, `aed_worker_0`, `aed_worker_1`, `punc_worker`, and `overlap_worker`, and that a one-worker configuration omits unavailable owners without inventing PIDs.

- [x] **Step 2: Run focused tests and confirm failure**

  Run `.venv/bin/python -m pytest -q tests/unit/test_service.py tests/unit/test_resource_profile.py tests/unit/test_diagnostics.py`.

- [x] **Step 3: Implement named owner PID getters and sampling metadata**

  Resolve the independent Punc PID from `ParallelFireRedWorkerClient`, identify the service-hosted Speaker owner separately, and keep all `nvidia-smi` rows filtered to registered PIDs. Record owner names, observed flags, baseline, and sampling interval in `resource_profile`.

- [x] **Step 4: Verify no foreign process interaction**

  Run resource sampler unit tests and inspect the generated query arguments to ensure only read-only GPU queries and registered PID filtering are used.

- [x] **Step 5: Commit**

  ```bash
  git add src/funasr_transcriber/service.py src/funasr_transcriber/resource_profile.py src/funasr_transcriber/diagnostics.py tests/unit/test_service.py tests/unit/test_resource_profile.py tests/unit/test_diagnostics.py
  git commit -m "fix: profile every resident FireRed owner"
  ```

### Task 3: Rename stage metrics and keep diagnostics.json compact

**Files:**
- Modify: `src/funasr_transcriber/firered_parallel.py`, `src/funasr_transcriber/service.py`, `src/funasr_transcriber/diagnostics.py`
- Create: `src/funasr_transcriber/diagnostic_traces.py`
- Test: `tests/unit/test_firered_parallel.py`, `tests/unit/test_diagnostics.py`, `tests/unit/test_service.py`

- [x] **Step 1: Write failing metric-schema tests**

  Assert explicit names for AED wall/worker, Punc queue/model/normalization, and that basic diagnostics omit full worker plans and per-batch records.

- [x] **Step 2: Run focused tests and confirm failure**

  Run `.venv/bin/python -m pytest -q tests/unit/test_firered_parallel.py tests/unit/test_diagnostics.py tests/unit/test_service.py`.

- [x] **Step 3: Implement metric separation**

  Keep low-cost stage summaries in `diagnostics.json`; emit complete scheduler plans to `diagnostics/scheduler.json` only for `deep`; emit batch/queue events as JSONL only for `deep`. Preserve wall times separately from sums of worker times and label overlap explicitly.

- [x] **Step 4: Verify atomic trace publication**

  Add atomic temporary-file replacement for trace files and ensure failed jobs remove staging traces with the existing job cleanup path.

- [x] **Step 5: Commit**

  ```bash
  git add src/funasr_transcriber/firered_parallel.py src/funasr_transcriber/service.py src/funasr_transcriber/diagnostics.py src/funasr_transcriber/diagnostic_traces.py tests/unit/test_firered_parallel.py tests/unit/test_diagnostics.py tests/unit/test_service.py
  git commit -m "feat: separate diagnostics summaries from deep traces"
  ```

### Task 4: Expose Punc microbatch controls

**Files:**
- Modify: `src/funasr_transcriber/config.py`, `config/default.toml`
- Modify: `src/funasr_transcriber/firered_client.py`, `src/funasr_transcriber/firered_worker.py`, `src/funasr_transcriber/firered_parallel.py`
- Test: `tests/unit/test_config.py`, `tests/unit/test_firered_client.py`, `tests/unit/test_punc_scheduler.py`

- [x] **Step 1: Write failing configuration/command tests**

  Assert configured Punc token budget and max item count are validated, passed to the Punc-only worker, and used by the stream planner.

- [x] **Step 2: Run focused tests and confirm failure**

  Run `.venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_firered_client.py tests/unit/test_punc_scheduler.py`.

- [x] **Step 3: Implement parameter propagation**

  Replace source constants at runtime with resolved config values, pass them as explicit worker arguments, and include the selected values in scheduler diagnostics.

- [x] **Step 4: Run focused and regression tests**

  Verify budgets 128/256/512 validate and produce deterministic bounded plans; reject zero or negative values.

- [x] **Step 5: Commit**

  ```bash
  git add src/funasr_transcriber/config.py config/default.toml src/funasr_transcriber/firered_client.py src/funasr_transcriber/firered_worker.py src/funasr_transcriber/firered_parallel.py tests/unit/test_config.py tests/unit/test_firered_client.py tests/unit/test_punc_scheduler.py
  git commit -m "feat: configure Punc microbatch budgets"
  ```

### Task 5: Chinese agent instructions and documentation consistency

**Files:**
- Rename: `AGENT.md` → `AGENTS.md`
- Modify: `README.md`, `docs/architecture/current-transcription-pipeline.md`
- Test: `tests/integration/test_firered_documentation.py`, `tests/unit/test_cli.py`

- [x] **Step 1: Write failing documentation assertions**

  Assert documentation describes two AED owners plus one independent Punc owner, diagnostics levels, named resource owners, and the Chinese `AGENTS.md` file.

- [x] **Step 2: Run documentation tests and confirm failure**

  Run `.venv/bin/python -m pytest -q tests/integration/test_firered_documentation.py tests/unit/test_cli.py`.

- [x] **Step 3: Rename and translate agent instructions**

  Preserve all safety, GPU0, no-worktree, no-subagent, ignored-corpus, verification, and commit rules while translating the full file to Chinese.

- [x] **Step 4: Update production documentation**

  Remove stale one-worker wording, document diagnostics defaults and flags, explain output/trace files, and mark benchmark scripts as non-production tools.

- [x] **Step 5: Commit**

  ```bash
  git add AGENT.md AGENTS.md README.md docs/architecture/current-transcription-pipeline.md tests/integration/test_firered_documentation.py tests/unit/test_cli.py
  git commit -m "docs: standardize diagnostics and Chinese agent guidance"
  ```

### Task 6: Full verification and GPU0 production smoke

**Files:**
- Create ignored report: `outputs/measurements/<timestamp>-diagnostics-standardization/result.json`

- [x] **Step 1: Run the complete test suite**

  Run `.venv/bin/python -m pytest -q` and `git diff --check`.

- [x] **Step 2: Run a real GPU0 basic-level job**

  Use `audio_test/刘洋洋.wav` with the existing BF16 production config and verify `diagnostics.json` contains only basic summaries and no resource trace.

- [x] **Step 3: Run resources and deep levels**

  Run the same audio with `--diagnostics resources` and `--diagnostics deep`; verify Punc owner peak is separately present, trace files are written only for deep, and no other GPU process is altered.

- [x] **Step 4: Verify cleanup and report**

  Stop only this project service, confirm GPU0 returns to the external baseline, and save timing, sample counts, owner peaks, output schema, and fingerprint status in the ignored report.

## Self-review

- All six requested work areas are covered in order: owner metrics, names, levels, traces, Punc configuration, and Chinese AGENTS instructions.
- Basic metrics remain cheap and always available; expensive sampling/tracing is opt-in.
- No task permits signaling or managing unrelated GPU processes.
- Output contract remains stable while optional deep artifacts are additive.

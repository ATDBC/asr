# Production Queue Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 20-recording RAMC release gate submit a real FIFO batch through the persistent service, write each completed result immediately, and store all new validation output under `outputs/release_validation/<run-id>`.

**Architecture:** The Unix-socket protocol is split into a non-terminal `submit` acknowledgement and a terminal `wait` action. `DaemonService` continues to own exactly one FIFO consumer and keeps a future per accepted job. The normal CLI composes the two actions synchronously; the RAMC tool submits the whole batch before waiting, then atomically updates run-level progress and summaries after every job.

**Tech Stack:** Python 3.11 stdlib Unix sockets, `Future`, JSON Lines, atomic `Path.replace`, pytest, existing FireRedASR2S/3D-Speaker persistent service.

**User constraints:** Work directly on the current branch (explicitly authorized; no worktree and no subagents), bind only physical GPU 0, and never signal, stop, pause, migrate, or otherwise affect other projects' processes.

---

## File structure

- Modify `src/funasr_transcriber/service_protocol.py`: validate the `wait` request and provide non-blocking submit plus terminal wait client methods.
- Modify `src/funasr_transcriber/service.py`: expose FIFO acceptance metadata and serve `submit` acknowledgement / `wait` terminal responses without holding a submit socket open.
- Modify `src/funasr_transcriber/cli.py`: retain auto-start and existing synchronous CLI behavior by composing `submit` and `wait`; expose a batch-safe submit/wait path for the RAMC tool.
- Modify `tools/benchmark_ramc_e2e.py`: prepare all recordings, enqueue them before waiting, export every recording below the requested run directory, and atomically maintain `progress.json` and `ramc-summary.json`.
- Modify `scripts/run_validation_linux.sh`: set the release run root below the repository `outputs/release_validation`, and pass the RAMC summary path in that run directory.
- Modify `.gitignore`: ignore the top-level `outputs/` tree.
- Modify `tests/unit/test_service_protocol.py`, `tests/unit/test_service.py`, and `tests/unit/test_cli.py`: prove immediate acceptance, FIFO wait results, duplicate submission semantics, and synchronous CLI compatibility.
- Modify `tests/unit/test_benchmarking.py`, `tests/benchmark/test_ramc_gate.py`, and `tests/integration/test_validation_corpus_cli.py`: prove batch submission, incremental run files, failure retention, and dedicated output roots.
- Preserve and include the existing uncommitted alignment-fallback and ASR-sentence text-retention fixes in final verification; they are required for the real corpus job that previously failed.

### Task 1: Define the two-stage protocol contract

**Files:**
- Modify: `tests/unit/test_service_protocol.py`
- Modify: `tests/unit/test_service.py`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Write failing request-validation tests**

Append tests that state the externally valid `wait` shape and reject invalid forms:

```python
def test_protocol_parses_wait_request_for_an_accepted_job():
    request = parse_request({"id": "wait-1", "action": "wait", "job_id": "job-1"})

    assert request.action == "wait"
    assert request.job_id == "job-1"


def test_protocol_rejects_wait_without_job_id():
    with pytest.raises(ServiceProtocolError, match="job_id"):
        parse_request({"id": "wait-1", "action": "wait"})
```

- [ ] **Step 2: Write a failing immediate-acceptance socket test**

Use a blocking `run_job` with `threading.Event`, then verify that `client.submit()` returns before the event is released and reports job metadata:

```python
accepted = client.submit(_request("job"))

assert accepted == {
    "id": "job", "ok": True, "terminal": False,
    "status": "accepted", "job_id": "job", "queue_depth": 1,
}
assert started.wait(timeout=1)
assert not released.is_set()
```

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_service_protocol.py tests/unit/test_service.py::test_socket_submit_acknowledges_before_blocking_job -v
```

Expected: FAIL because `wait` is unsupported and `submit_and_wait` is the only client method.

- [ ] **Step 3: Write failing FIFO, idempotency, wait, and CLI-composition tests**

Add a daemon test that accepts three requests before releasing the first, waits all three and asserts `completed == ["first", "second", "third"]`. Add an assertion that a second `submit` with `id="first"` does not queue a second copy. Add a test that `client.wait("missing")` raises `ServiceProtocolError` and update the CLI fake client expectation to record `submit` then `wait` while `submit_to_service()` still returns the terminal result.

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_service.py tests/unit/test_cli.py::test_transcribe_client_starts_service_only_when_health_is_unavailable -v
```

Expected: FAIL because there is no daemon lookup for a job future and no non-blocking client API.

### Task 2: Implement the two-stage FIFO protocol

**Files:**
- Modify: `src/funasr_transcriber/service_protocol.py`
- Modify: `src/funasr_transcriber/service.py`
- Modify: `src/funasr_transcriber/cli.py`
- Test: `tests/unit/test_service_protocol.py`
- Test: `tests/unit/test_service.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Extend typed requests and parser minimally**

Add `job_id: str | None = None` to `ServiceRequest`, accept `"wait"`, and require a non-empty string only for that action:

```python
if action == "wait":
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        raise ServiceProtocolError("wait job_id must not be empty")
    return ServiceRequest(identifier=identifier, action=action, job_id=job_id)
```

Keep `health` and `stop` payloads unchanged. Add a `_request_payload()` helper to serialize one submit request, `submit()` to require the accepted non-terminal response, and `wait(job_id, identifier)` to require a terminal response. Do not make `_exchange()` loop for a terminal response when requesting `submit`: it must read exactly its one acknowledgement.

- [ ] **Step 2: Make daemon acceptance queryable without changing its single consumer**

Keep `DaemonService.submit()` returning the existing future. Add locked helpers:

```python
def accepted_queue_depth(self) -> int:
    return self._queue.qsize()

def result_for(self, job_id: str) -> Future[dict[str, object]]:
    with self._lock:
        future = self._futures.get(job_id)
    if future is None:
        raise ServiceProtocolError(f"unknown service job: {job_id}")
    return future
```

In the request handler, replace `.submit(request).result()` with an immediate payload after `submit()`. For `wait`, call `result_for(request.job_id).result()` and return its normal result or the existing structured terminal error. The acknowledgement includes `id`, `ok`, `terminal: false`, `status: "accepted"`, `job_id`, and `queue_depth` from the daemon. A duplicate identifier reuses its future and returns the same non-terminal acknowledgement; it never calls `queue.put_nowait()` again.

- [ ] **Step 3: Preserve the public CLI’s synchronous behavior**

Extract the existing compatible auto-start sequence into a helper that returns `ServiceClient`. Change `submit_to_service()` to construct the request, call `client.submit(request)`, then `client.wait(request.identifier, uuid.uuid4().hex)`. It must still return the terminal response shape consumed by `_run_transcribe`. Provide a narrow `submit_service_job(args, config)` helper returning `(client, request, accepted)` for the RAMC tool; it uses the same startup, output validation, GPU restrictions, and request construction but does not wait.

- [ ] **Step 4: Run focused protocol tests and inspect formatting**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_service_protocol.py tests/unit/test_service.py tests/unit/test_cli.py -v
.venv/bin/python -m compileall -q src/funasr_transcriber
git diff --check
```

Expected: all selected tests pass, source compiles, and `git diff --check` has no output.

- [ ] **Step 5: Commit the protocol change**

```bash
git add src/funasr_transcriber/service_protocol.py src/funasr_transcriber/service.py src/funasr_transcriber/cli.py tests/unit/test_service_protocol.py tests/unit/test_service.py tests/unit/test_cli.py
git commit -m "feat: enqueue service jobs before waiting"
```

### Task 3: Specify RAMC incremental run files in tests

**Files:**
- Modify: `tests/unit/test_benchmarking.py`
- Modify: `tests/benchmark/test_ramc_gate.py`

- [ ] **Step 1: Write a failing batching test around injectable tool helpers**

Refactor the test module setup so it can inject a fake submit/wait pair. The test should prepare two WAV/TXT records, use a fake submit that appends job IDs and a fake wait that writes the matching transcript/diagnostics into the supplied output directory. Assert both `submit` calls occur before the first `wait` call and both job output directories are below `run_root / "ramc"`.

```python
assert events == ["submit:first", "submit:second", "wait:first", "wait:second"]
assert (run_root / "progress.json").is_file()
assert (run_root / "ramc-summary.json").is_file()
```

- [ ] **Step 2: Write a failing incremental-persistence test**

Make the fake second `wait` raise `ServiceProtocolError("failed")`. After `main()` returns non-zero, assert `progress.json` records the first item as `completed`, the second as `failed`, and `ramc-summary.json` has `complete: true`, one result, one failure, and `passed: false`. This test proves failure does not discard the first transcript, diagnostic, progress entry, or summary.

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_benchmarking.py tests/benchmark/test_ramc_gate.py::test_ramc_tool_emits_persistent_service_evidence -v
```

Expected: FAIL because the current tool waits inside its per-recording loop and only writes one output JSON at the end.

### Task 4: Implement RAMC batch queueing and atomic run output

**Files:**
- Modify: `tools/benchmark_ramc_e2e.py`
- Modify: `tests/unit/test_benchmarking.py`
- Modify: `tests/benchmark/test_ramc_gate.py`

- [ ] **Step 1: Prepare records before touching the service**

Introduce a small immutable or dictionary-based prepared record containing the source stem, clip path, reference intervals/text, and `output_dir`. `--output` remains the final `ramc-summary.json` path; define `run_root = args.output.parent` and every output path as `run_root / "ramc" / stem`. Keep `TemporaryDirectory` only for clips made from non-preclipped source WAVs, not for exported service outputs.

- [ ] **Step 2: Add atomic JSON and summary builders**

Add a local helper that writes `<name>.<pid>.tmp`, flushes, calls `os.fsync`, and replaces the target. Build the existing metrics and release evidence from completed results only; when no successful record exists, use empty/`None` aggregate values and set all release gates false rather than calling `statistics.mean([])`. Include these top-level fields on every write:

```python
{
    "run_id": run_root.name,
    "complete": pending_count == 0,
    "pending": pending_count,
    "failures": failures,
    "summary": summary,
    "gates": gates,
    "passed": complete and not failures and all(gates.values()),
    "release_evidence": release_evidence,
    "recording_results": completed_results,
}
```

`progress.json` contains ordered records with `name`, `job_id`, `status`, `output_dir`, `submitted_at`, `completed_at`, and optional `error`.

- [ ] **Step 3: Submit the full batch, then wait and persist after each terminal event**

Use `submit_service_job()` once per prepared record, append its accepted job ID to ordered progress, and atomically write `progress.json` after every acknowledgement. Do not call `wait` during this loop. In a second loop, wait by job ID in TSV order; on success read the published files, compute that one record’s metrics, append it, mark it completed, and atomically update both files. On `ServiceProtocolError` or invalid published export, mark that record failed, update both files, and continue with every already accepted job. Return non-zero if preparation/submission is incomplete, a terminal error occurred, or final gates fail.

- [ ] **Step 4: Expand static release-gate evidence checks**

Require the tool source to contain `"complete"`, `"pending"`, `"failures"`, `"progress.json"`, `"submit_service_job"`, and `"wait"`; update the optional live gate to assert its output path’s parents are `ramc/<stem>` below the test-selected run directory, `complete is True`, and no `artifacts/release_validation_corpus/validation_results` was created.

- [ ] **Step 5: Run all RAMC unit and gate tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_benchmarking.py tests/benchmark/test_ramc_gate.py -v
```

Expected: all non-optional tests pass and the live gate remains skipped unless its explicit environment variable is set.

- [ ] **Step 6: Commit the RAMC tool change**

```bash
git add tools/benchmark_ramc_e2e.py tests/unit/test_benchmarking.py tests/benchmark/test_ramc_gate.py
git commit -m "feat: persist RAMC queue progress per recording"
```

### Task 5: Move all release-validation output out of the corpus

**Files:**
- Modify: `.gitignore`
- Modify: `scripts/run_validation_linux.sh`
- Modify: `tests/integration/test_validation_corpus_cli.py`

- [ ] **Step 1: Write failing runner/ignore tests**

Add assertions that `.gitignore` contains `outputs/`, the runner assigns `RESULTS_ROOT="$PROJECT_ROOT/outputs/release_validation"`, and its RAMC invocation produces `$RESULTS/ramc-summary.json`. Assert the script contains no assignment to `$CORPUS_ROOT/validation_results`.

- [ ] **Step 2: Run the new tests to prove RED**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_validation_corpus_cli.py -v
```

Expected: FAIL because the script currently writes under the input corpus and the repository ignores only the old corpus-local path.

- [ ] **Step 3: Implement dedicated output wiring**

Add `outputs/` to `.gitignore`. Change the runner’s root assignment exactly to `RESULTS_ROOT="$PROJECT_ROOT/outputs/release_validation"`. Before the RAMC pytest call, export `FUNASR_TRANSCRIBER_RAMC_OUTPUT="$RESULTS/ramc-summary.json"`; make `tests/benchmark/test_ramc_gate.py` use this explicit path if set, otherwise retain its `tmp_path` default for isolated test execution. The script continues to pass all other stage outputs through `$RESULTS`.

- [ ] **Step 4: Re-run the integration runner tests**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_validation_corpus_cli.py -v
git diff --check
```

Expected: all tests pass and no whitespace errors are reported.

- [ ] **Step 5: Commit the output relocation**

```bash
git add .gitignore scripts/run_validation_linux.sh tests/integration/test_validation_corpus_cli.py tests/benchmark/test_ramc_gate.py
git commit -m "fix: keep validation output outside corpus artifacts"
```

### Task 6: Verify fixes and execute the real GPU 0 queue validation

**Files:**
- Modify: `src/funasr_transcriber/alignment.py`
- Modify: `tests/unit/test_alignment.py`
- Modify: `tools/benchmark_ramc_e2e.py`
- Modify: `tests/unit/test_benchmarking.py`

- [ ] **Step 1: Verify existing regression fixes before combining commits**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_alignment.py tests/unit/test_benchmarking.py -v
```

Expected: the distant-speaker fallback produces a low-confidence token with `nearest_speaker_fallback`, and text retention uses exported ASR sentences.

- [ ] **Step 2: Run the full non-model test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all normal tests pass; explicitly gated model and benchmark tests may skip.

- [ ] **Step 3: Commit the retained corpus correctness fixes**

```bash
git add src/funasr_transcriber/alignment.py tests/unit/test_alignment.py tools/benchmark_ramc_e2e.py tests/unit/test_benchmarking.py
git commit -m "fix: preserve text across diarization gaps"
```

Only stage files still uncommitted at this point; do not stage unrelated user files.

- [ ] **Step 4: Run live RAMC queue acceptance on GPU 0**

Before launch, record only `nvidia-smi --id=0 --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits` and the current process list filtered to this project. Run the tool against the 20 validated corpus entries with the local validation config and an output such as:

```bash
FUNASR_TRANSCRIBER_RUN_BENCHMARKS=1 \
FUNASR_TRANSCRIBER_RAMC_ROOT="$PWD/artifacts/release_validation_corpus/ramc_core" \
FUNASR_TRANSCRIBER_RAMC_TSV="$PWD/artifacts/release_validation_corpus/ramc_core/test.tsv" \
FUNASR_TRANSCRIBER_CONFIG="$PWD/artifacts/release_validation_corpus/.service-work/local-validation.toml" \
.venv/bin/python tools/benchmark_ramc_e2e.py \
  --dataset "$PWD/artifacts/release_validation_corpus/ramc_core" \
  --test-tsv "$PWD/artifacts/release_validation_corpus/ramc_core/test.tsv" \
  --config "$PWD/artifacts/release_validation_corpus/.service-work/local-validation.toml" \
  --output "$PWD/outputs/release_validation/<run-id>/ramc-summary.json" \
  --count 20 --preclipped
```

While it runs, inspect only its `progress.json`, its `service status`, GPU 0 aggregate memory, and command lines matching `funasr_transcriber.service|firered_worker|benchmark_ramc_e2e`. Do not signal any process. Confirm 20 accepted jobs precede the first completed record and every published recording directory stays below the run root.

- [ ] **Step 5: Stop only this project’s service normally and verify release**

Run the project CLI’s `service stop` with the same config, wait for that state/socket to disappear, then sample GPU 0. Confirm this project’s service and FireRed worker exited and report aggregate GPU memory against the pre-run baseline; do not inspect or affect unrelated PID details.

- [ ] **Step 6: Final evidence and worktree check**

Run:

```bash
git status --short
git log --oneline -5
```

Report the exact output run directory, summary/progress paths, test result counts, live queue evidence, and any gate failures rather than claiming success without this evidence.

# Weighted AED Device Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not use subagents for this repository. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add a resumable offline command that measures a pre-segmented corpus on physical GPU0, fits the two weighted-duration AED coefficients, writes a validated device artifact, and lets production load it safely.

**Architecture:** Reuse the existing explicit AED-plan worker protocol for GPU measurements. Put deterministic planning, fitting, grouped diagnostics, artifact validation, and fallback in one focused module; keep the CLI as an append-only orchestrator and pass one immutable weight selection into the production weighted owner.

**Tech Stack:** Python 3.11, dataclasses, hashlib/json/pathlib, NumPy 2.4.3, existing FireRed JSONL worker protocol, TOML, pytest.

**Spec:** docs/superpowers/specs/2026-09-17-weighted-aed-device-calibration-design.md

## Global Constraints

- Work only through ssh ai_server in /mnt/hdd/jiajun/project/standalone_transcriber on feature/server-weighted-aed-batching.
- Work in the current directory; do not create a worktree and do not use subagents.
- Preserve the unrelated uncommitted .gitignore modification and never include it in a task commit.
- Use only physical GPU0 as cuda:0; never remap another GPU with CUDA_VISIBLE_DEVICES.
- Start, inspect, clean, or stop only this repository's own workers, logs, and outputs.
- Calibration is solo AED only: no VAD, Punc, second AED worker, contention fit, or online learning.
- Add no free-memory budget or memory fallback. Keep 200 seconds and 32 items as fixed shape limits.
- Never read development-validation or blind-holdout records, train a candidate, publish a production manifest, or write models/cost_models/.
- Keep artifacts/ and artifacts/release_validation_corpus ignored; do not modify historical plans, manifests, records, audits, or benchmark outputs.
- Preserve built-in defaults exactly: longest_weight=0.12854663332839417 and padded_weight=0.008409640427755745.
- Use the repository Git identity. Do not push unless explicitly requested.
- On failure or OOM, preserve plan and records, report the exact error, and do not retry in a loop.

## File Map

- Create src/funasr_transcriber/weighted_aed_calibration.py: weights, source/case planning, checksums, fitting, grouped diagnostics, artifact validation, fallback.
- Create tools/calibrate_weighted_aed.py: CLI, create-only plan, one-worker collection, append-only resume, report/artifact writes.
- Create tests/unit/test_weighted_aed_calibration.py and tests/unit/test_weighted_aed_calibration_tool.py.
- Modify firered_worker.py, firered_parallel.py, config.py, service.py, config/default.toml, and their focused tests.
- Modify README.md for the operator workflow.
- Create docs/architecture/2026-09-17-weighted-aed-device-calibration-results.md only after fresh GPU evidence exists.

---

### Task 1: Parameterize the Weighted Scheduler Without Changing Defaults

**Files:**
- Create: src/funasr_transcriber/weighted_aed_calibration.py
- Modify: src/funasr_transcriber/firered_worker.py:37-170
- Modify: src/funasr_transcriber/firered_parallel.py:833-990
- Test: tests/unit/test_firered_worker.py
- Test: tests/unit/test_firered_parallel.py

**Interfaces:**
- Produces WeightedAedWeights(longest_weight: float, padded_weight: float) and DEFAULT_WEIGHTED_AED_WEIGHTS.
- Changes plan_aed_batches(segments, *, weights=DEFAULT_WEIGHTED_AED_WEIGHTS).
- Changes _aed_plan_summary(plan, *, weights=..., source="builtin", artifact_checksum=None, warnings=()).
- Changes WeightedAedPuncWorkerClient(..., weights=..., weight_source="builtin", weight_artifact_checksum=None, weight_warnings=()).
- Preserves AED_LONGEST_WEIGHT and AED_PADDED_WEIGHT aliases.

- [ ] **Step 1: Write failing planner and client tests**

~~~python
from funasr_transcriber.weighted_aed_calibration import WeightedAedWeights


def test_plan_aed_batches_uses_explicit_weights():
    segments = [(0.0, 46.0), (46.0, 100.0), (100.0, 148.0),
                (148.0, 203.0), (203.0, 234.0), (234.0, 278.0)]
    default = plan_aed_batches(segments)
    custom = plan_aed_batches(
        segments,
        weights=WeightedAedWeights(0.01, 1.0),
    )
    assert [[item.index for item in batch] for batch in custom] != [
        [item.index for item in batch] for batch in default
    ]


def test_weighted_owner_reports_explicit_weights(fake_aed, fake_punc):
    owner = WeightedAedPuncWorkerClient(
        fake_aed,
        fake_punc,
        weights=WeightedAedWeights(0.25, 0.015),
        weight_source="artifact",
        weight_artifact_checksum="a" * 64,
    )
    assert owner.cost_model.seconds_per_max_frame == pytest.approx(0.0025)
    assert owner.cost_model.seconds_per_padded_frame == pytest.approx(0.00015)
~~~

- [ ] **Step 2: Run the tests and verify RED**

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_firered_worker.py::test_plan_aed_batches_uses_explicit_weights \
  tests/unit/test_firered_parallel.py::test_weighted_owner_reports_explicit_weights
~~~

Expected: FAIL because the value type and weight parameters do not exist.

- [ ] **Step 3: Add the immutable weight value**

~~~python
@dataclass(frozen=True, slots=True)
class WeightedAedWeights:
    longest_weight: float
    padded_weight: float

    def __post_init__(self) -> None:
        values = (float(self.longest_weight), float(self.padded_weight))
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("weighted AED coefficients must be finite and positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "longest_weight": float(self.longest_weight),
            "padded_weight": float(self.padded_weight),
        }


DEFAULT_WEIGHTED_AED_WEIGHTS = WeightedAedWeights(
    0.12854663332839417,
    0.008409640427755745,
)
~~~

Import it in firered_worker.py and retain the two old constant names as aliases.

- [ ] **Step 4: Thread the value through planning, lane estimates, and diagnostics**

~~~python
def _batch_cost(
    batch: list[ScheduledItem],
    weights: WeightedAedWeights = DEFAULT_WEIGHTED_AED_WEIGHTS,
) -> float:
    longest = max(_item_duration_ms(item) for item in batch) / 1000
    return longest * (
        weights.longest_weight + weights.padded_weight * len(batch)
    )
~~~

Pass the same value through every relocation comparison, summary prediction, and BatchCostModel. Include weight_source, nullable artifact checksum, and warnings in scheduler diagnostics.

- [ ] **Step 5: Run scheduler tests and commit**

~~~bash
.venv/bin/python -m pytest -q tests/unit/test_firered_worker.py tests/unit/test_firered_parallel.py
git diff --check
git add src/funasr_transcriber/weighted_aed_calibration.py \
  src/funasr_transcriber/firered_worker.py \
  src/funasr_transcriber/firered_parallel.py \
  tests/unit/test_firered_worker.py tests/unit/test_firered_parallel.py
git commit -m "refactor: parameterize weighted AED scheduling"
~~~

Expected: all tests pass and the default plan remains unchanged.

---

### Task 2: Implement Deterministic Planning, Fitting, and Artifact Validation

**Files:**
- Modify: src/funasr_transcriber/weighted_aed_calibration.py
- Create: tests/unit/test_weighted_aed_calibration.py

**Interfaces:**
- Produces frozen CalibrationSource, CalibrationCase, and WeightSelection dataclasses.
- Produces canonical_sha256(payload), sha256_file(path).
- Produces load_calibration_sources(manifest_path, audio_root=None).
- Produces build_calibration_cases(sources), validate_calibration_coverage(cases).
- Produces assign_grouped_folds(groups, fold_count=5).
- Produces evaluate_calibration(records) -> tuple[WeightedAedWeights, dict[str, object]].
- Produces build_expected_runtime_identity(model_dir, gpu_name, driver_version, use_half).
- Produces build_ready_artifact(*, weights, report, runtime_identity, manifest_checksum, plan_checksum, records_checksum, gpu_start, gpu_end).
- Produces load_weight_artifact(path: Path, expected_runtime_identity: Mapping[str, object]) -> WeightSelection.

- [ ] **Step 1: Write failing source and case tests**

~~~python
def test_build_cases_uses_source_local_short_middle_long_windows(tmp_path: Path):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"audio")
    segments = [
        {"index": i, "start_ms": i * 20_000,
         "end_ms": i * 20_000 + (i + 1) * 100}
        for i in range(32)
    ]
    source = CalibrationSource(
        source_recording_group="source-a",
        audio_path=audio,
        source_checksum=sha256_file(audio),
        segments_checksum=canonical_sha256(segments),
        segments=tuple(segments),
    )
    cases = build_calibration_cases((source,))
    assert {case.item_count for case in cases} == {1, 2, 4, 8, 16, 32}
    assert all(case.source_recording_group == "source-a" for case in cases)
    assert all(case.total_duration_seconds <= 200.0 for case in cases)
~~~

Also cover duplicate groups, invalid bounds, checksum mismatch, over-200-second segments, deduplication, relative audio_path, and coverage failure without count 16 or greater.

- [ ] **Step 2: Run planning tests and verify RED**

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_weighted_aed_calibration.py -k "source or cases or coverage"
~~~

Expected: FAIL because the interfaces are absent.

- [ ] **Step 3: Implement validation and deterministic windows**

Use historical canonical checksum semantics:

~~~python
def canonical_sha256(payload: object) -> str:
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
~~~

For every source sorted by group, sort segments by (duration_ms, index). For counts (1, 2, 4, 8, 16, 32), use offsets {0, (n-count)//2, n-count}, deduplicate index tuples, skip totals above 200_000 ms, and hash source group plus selected segments into case_id. Require ten groups, four count levels, count 1, one level at least 16, and matrix rank two before GPU work.

- [ ] **Step 4: Write failing fit, fold, noise, and artifact tests**

~~~python
def test_evaluate_calibration_recovers_two_coefficients():
    records = []
    for group_index in range(10):
        for count in (1, 2, 4, 8, 16, 32):
            longest = 1.0 + group_index / 10
            actual = longest * (0.2 + 0.01 * count)
            records.append({
                "status": "complete",
                "source_recording_group": f"source-{group_index}",
                "item_count": count,
                "max_duration_seconds": longest,
                "repeat_seconds": [actual, actual, actual],
                "median_actual_seconds": actual,
            })
    weights, report = evaluate_calibration(records)
    assert weights.longest_weight == pytest.approx(0.2)
    assert weights.padded_weight == pytest.approx(0.01)
    assert report["repeat_noise"]["status"] == "stable"
    assert report["grouped_folds"] == 5
~~~

Add tests that groups never split across folds, repeat median deviation over 5% fails, repeat P95 over 20% fails, rank-one input fails, non-positive coefficients fail, strict runtime mismatch falls back, and driver/Torch patch differences only warn.

- [ ] **Step 5: Implement fit and grouped OOF diagnostics**

~~~python
matrix = np.asarray([
    [row["max_duration_seconds"],
     row["max_duration_seconds"] * row["item_count"]]
    for row in complete
], dtype=np.float64)
labels = np.asarray(
    [row["median_actual_seconds"] for row in complete], dtype=np.float64
)
if np.linalg.matrix_rank(matrix) != 2:
    raise ValueError("weighted AED calibration feature matrix must have rank 2")
coefficients, _, _, _ = np.linalg.lstsq(matrix, labels, rcond=None)
weights = WeightedAedWeights(*map(float, coefficients))
~~~

Assign groups by SHA-256 sort and round-robin fold. Report OOF MAE, median APE, P95 APE, item-count metrics, support, and repeat noise separately. Refit all rows only after coverage, fingerprint, failure, and repeat-noise gates pass.

- [ ] **Step 6: Implement artifact construction and fallback**

~~~python
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_KIND = "weighted_aed_device_calibration"
FORMULA_ID = "max_duration_seconds*(longest_weight+padded_weight*item_count)"

@dataclass(frozen=True, slots=True)
class WeightSelection:
    weights: WeightedAedWeights
    source: str
    artifact_checksum: str | None
    warnings: tuple[str, ...]
~~~

Treat schema, kind, ready status, formula, weights, cuda:0, dtype, GPU name, compute capability, FireRed commit, checkpoint SHA-256, and evidence fields as strict. Treat driver, CUDA, and Torch versions as warnings. Missing/invalid artifacts return built-ins with source="builtin_fallback" and the precise error.

- [ ] **Step 7: Run the pure suite and commit**

~~~bash
.venv/bin/python -m pytest -q tests/unit/test_weighted_aed_calibration.py
git diff --check
git add src/funasr_transcriber/weighted_aed_calibration.py \
  tests/unit/test_weighted_aed_calibration.py
git commit -m "feat: add weighted AED calibration model"
~~~

Expected: PASS with no whitespace errors.

---

### Task 3: Add the Resumable One-Command Collector

**Files:**
- Create: tools/calibrate_weighted_aed.py
- Create: tests/unit/test_weighted_aed_calibration_tool.py
- Modify: src/funasr_transcriber/firered_worker.py:783-819
- Modify: tests/unit/test_firered_worker.py:50-90

**Interfaces:**
- CLI: --manifest, optional --audio-root, --model-dir, --worker-python, --output, --repeats (3), --use-half/--no-use-half (BF16/--use-half by default), --resume.
- Produces create-only plan.json, append-only records.jsonl, atomic report, ready-only artifact.
- Uses one FireRedWorkerClient with VAD/Punc disabled on cuda:0.
- Exposes run(args, *, client_factory=FireRedWorkerClient, gpu_query=query_gpu) -> int for CPU tests.

- [ ] **Step 1: Write failing CPU-only command tests**

Load the tool with importlib.util.spec_from_file_location. Create ten temporary source files and inject this fake:

~~~python
class FakeClient:
    def __init__(self, **options):
        assert options["device"] == "cuda:0"
        assert options["enable_vad"] is False
        assert options["enable_punc"] is False

    def start(self):
        return self

    def status(self):
        return {"runtime": {
            "device": "cuda:0", "dtype": "torch.bfloat16",
            "compute_capability": "8.6", "torch_version": "2.test",
            "cuda_version": "12.test",
        }}

    def run_aed_plan(self, audio_path, batch_plan, vad_segments_ms=None):
        count = len(batch_plan[0])
        longest = max((item["end_ms"] - item["start_ms"]) / 1000
                      for item in batch_plan[0])
        actual = longest * (0.2 + 0.01 * count)
        return {
            "aed_results": [{"index": item["index"], "result": {"text": "ok"}}
                            for item in batch_plan[0]],
            "batch_metrics": [{"actual_seconds": actual,
                               "peak_allocated_mib": 100,
                               "peak_reserved_mib": 120}],
        }

    def close(self):
        pass
~~~

Assert a first run writes all four outputs, a second run without --resume fails, resume skips successful IDs, and manifest/plan mismatch fails before the fake client is constructed.

- [ ] **Step 2: Run tool tests and verify RED**

~~~bash
.venv/bin/python -m pytest -q tests/unit/test_weighted_aed_calibration_tool.py
~~~

Expected: FAIL because the tool is absent.

- [ ] **Step 3: Add actual runtime facts to worker status**

Extend firered_inventory() with device, AED parameter dtype, compute capability, Torch version, and CUDA version:

~~~python
device_index = torch.cuda.current_device()
major, minor = torch.cuda.get_device_capability(device_index)
parameter_groups = modules["asr"]["parameters"]
parameter_dtypes = {
    key.split(":", 1)[1]
    for key, values in parameter_groups.items()
    if int(values["elements"]) > 0
}
if len(parameter_dtypes) != 1:
    raise RuntimeError("FireRed AED parameter dtype is ambiguous")
runtime = {
    "device": f"cuda:{device_index}",
    "dtype": f"torch.{next(iter(parameter_dtypes))}",
    "compute_capability": f"{major}.{minor}",
    "torch_version": str(torch.__version__),
    "cuda_version": str(torch.version.cuda),
}
~~~

Require one AED parameter dtype and extend the inventory unit test with fake deterministic values.

- [ ] **Step 4: Implement create-only and append-only I/O**

Validate input/checksums/cases, query GPU0, compute source HEAD and checkpoint SHA-256, then create plan.json with Path.open("x"). Append only completed case records durably:

~~~python
def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False,
                                sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
~~~

Resume rejects malformed/duplicate records, checksum or identity mismatch, and any preserved failed record. It skips only successful matching case IDs.

- [ ] **Step 5: Implement collection and finalization**

Start one owned worker in try/finally; execute the first case once as excluded warm-up. Execute every planned case for three repeats as a single explicit batch, hash canonical aed_results, require equal fingerprints, and append median time plus repeat/peak values. On exception append one failed record, close only this worker, and exit nonzero.

After closing the owned worker, query and persist the ending GPU0 snapshot. After complete collection, atomically write calibration-report.json and call build_ready_artifact() with the manifest/plan/records checksums, runtime identity, and both GPU snapshots. Write weighted-aed-calibration.json only for ready status. Measurement instability returns exit 2 with the report and no artifact.

- [ ] **Step 6: Run focused tests and commit**

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_weighted_aed_calibration_tool.py \
  tests/unit/test_firered_worker.py tests/unit/test_firered_client.py
git diff --check
git add tools/calibrate_weighted_aed.py \
  src/funasr_transcriber/firered_worker.py \
  tests/unit/test_weighted_aed_calibration_tool.py \
  tests/unit/test_firered_worker.py
git commit -m "feat: calibrate weighted AED devices"
~~~

Expected: PASS without CUDA.

---

### Task 4: Load the Artifact in Production With Safe Fallback

**Files:**
- Modify: config/default.toml:11-20
- Modify: src/funasr_transcriber/config.py:26-35, 98-108, 255-268
- Modify: src/funasr_transcriber/service.py:1056-1137
- Modify: tests/unit/test_config.py:119-190
- Modify: tests/unit/test_service.py:234-282
- Modify: tests/unit/test_firered_parallel.py

**Interfaces:**
- Produces AsrConfig.aed_weight_artifact: Path | None.
- Calls load_weight_artifact(path, expected_runtime_identity) -> WeightSelection once per weighted owner construction.
- Passes the same selection to packing, lane estimates, and diagnostics.

- [ ] **Step 1: Write failing config and service tests**

~~~python
def test_aed_weight_artifact_is_optional_and_expands_user(tmp_path: Path):
    assert load_config().asr.aed_weight_artifact is None
    path = tmp_path / "config.toml"
    path.write_text('[asr]\naed_weight_artifact = "~/weights.json"\n',
                    encoding="utf-8")
    assert load_config(path).asr.aed_weight_artifact == Path(
        "~/weights.json"
    ).expanduser()
~~~

Monkeypatch service_module.load_weight_artifact to return custom and fallback selections. Assert the weighted owner receives exact weights, source, checksum, and warnings.

- [ ] **Step 2: Run tests and verify RED**

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_config.py::test_aed_weight_artifact_is_optional_and_expands_user \
  tests/unit/test_service.py -k weighted_service
~~~

Expected: FAIL because the field and loader call do not exist.

- [ ] **Step 3: Add the optional field**

~~~toml
[asr]
aed_weight_artifact = ""
~~~

Normalize empty string to None; otherwise expand it to Path. Do not require existence during generic config parsing.

- [ ] **Step 4: Select one value before building the weighted owner**

When configured, build expected identity from GPU0, configured precision, local compute capability, actual source HEAD, and AED checkpoint SHA-256. Load once, log warnings, and pass it:

~~~python
return WeightedAedPuncWorkerClient(
    worker0,
    punc_owner,
    worker1=worker1,
    weights=selection.weights,
    weight_source=selection.source,
    weight_artifact_checksum=selection.artifact_checksum,
    weight_warnings=selection.warnings,
    punc_token_budget=config.asr.punc_token_budget,
    punc_max_items=config.asr.punc_max_items,
)
~~~

When unset, use built-ins without hashing the checkpoint. Memory-first mode remains unchanged.

- [ ] **Step 5: Prove diagnostics and estimates agree**

Extend the parallel test to transcribe with custom weights. Assert scheduler weights/source/checksum and every predicted_seconds value agree with the owner's BatchCostModel. This prevents planner/lane divergence.

- [ ] **Step 6: Run integration tests and commit**

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_config.py tests/unit/test_service.py \
  tests/unit/test_firered_parallel.py
git diff --check
git add config/default.toml src/funasr_transcriber/config.py \
  src/funasr_transcriber/service.py src/funasr_transcriber/firered_parallel.py \
  tests/unit/test_config.py tests/unit/test_service.py \
  tests/unit/test_firered_parallel.py
git commit -m "feat: load weighted AED device artifacts"
~~~

Expected: PASS.

---

### Task 5: Document and Verify the Product Workflow

**Files:**
- Modify: README.md:270-310

**Interfaces:**
- Documents command, manifest contract, resume, artifact config, fallback, and capacity boundary.

- [ ] **Step 1: Add the operator section**

~~~bash
.venv/bin/python tools/calibrate_weighted_aed.py \
  --manifest /data/calibration/segments-plan.json \
  --audio-root /data/calibration/audio \
  --model-dir /opt/funasr-transcriber/models \
  --worker-python /opt/funasr-transcriber/.venv-asr/bin/python \
  --output /var/lib/funasr-transcriber/device-calibration
~~~

~~~toml
[asr]
aed_scheduler = "weighted_duration"
aed_weight_artifact = "/etc/funasr-transcriber/weighted-aed-calibration.json"
~~~

Explain metadata.source_inventory, --audio-root, append-only resume, built-in fallback, and that OOM means the fixed shape contract is unsupported rather than fixable by time weights.

- [ ] **Step 2: Run focused and complete checks**

~~~bash
.venv/bin/python -m pytest -q \
  tests/unit/test_weighted_aed_calibration.py \
  tests/unit/test_weighted_aed_calibration_tool.py \
  tests/unit/test_config.py tests/unit/test_firered_client.py \
  tests/unit/test_firered_worker.py tests/unit/test_firered_parallel.py \
  tests/unit/test_service.py
.venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_firered_client.py
.venv/bin/python -m pytest -q
git diff --check
~~~

Expected: all tests pass and whitespace check is silent.

- [ ] **Step 3: Review scope and commit documentation**

~~~bash
git status --short
git diff --stat HEAD~4..HEAD
git diff HEAD~4..HEAD -- src/funasr_transcriber tools/calibrate_weighted_aed.py \
  config/default.toml README.md tests/unit
git add README.md
git commit -m "docs: explain weighted AED device calibration"
~~~

Confirm no validation/blind path, memory cap, historical output, or unrelated .gitignore change entered a commit.

---

### Task 6: Run Fresh GPU0 Acceptance and Record Evidence

**Files:**
- Create ignored output: outputs/measurements/20260917T-weighted-aed-device-calibration/
- Create: docs/architecture/2026-09-17-weighted-aed-device-calibration-results.md

**Interfaces:**
- Reads only the V3 train plan's metadata.source_inventory.
- Produces a ready artifact and committed evidence report; does not deploy automatically.

- [ ] **Step 1: Check branch, input, and GPU0 read-only**

~~~bash
git status --short --branch
test -f outputs/measurements/20260827T-firered-cost-model-v3/full/train-plan.json
nvidia-smi --id=0 \
  --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
~~~

Expected: feature branch, known .gitignore change only, train plan present, GPU0 available. If GPU0 activity would invalidate timing or prevents worker loading, stop and report; do not inspect or manage other processes.

- [ ] **Step 2: Run one calibration**

~~~bash
.venv/bin/python tools/calibrate_weighted_aed.py \
  --manifest outputs/measurements/20260827T-firered-cost-model-v3/full/train-plan.json \
  --model-dir models \
  --worker-python .venv-asr/bin/python \
  --output outputs/measurements/20260917T-weighted-aed-device-calibration
~~~

Expected: exit 0; complete plan/records/report; ready artifact; no failure, OOM, safety, or fingerprint mismatch. If interrupted, run the same command once with --resume, never a retry loop.

- [ ] **Step 3: Validate checksums and agreement**

~~~bash
sha256sum outputs/measurements/20260917T-weighted-aed-device-calibration/{plan.json,records.jsonl,calibration-report.json,weighted-aed-calibration.json}
jq '{status,weights,runtime_identity,evidence,fit}' \
  outputs/measurements/20260917T-weighted-aed-device-calibration/weighted-aed-calibration.json
~~~

Expected: ready status plus checksums, counts, groups, folds, support, noise, snapshots, and zero-failure results.

- [ ] **Step 4: Verify loading across two owned service restarts**

Create an ignored acceptance TOML with apply_patch using the prior weighted-dual config values, changing work_dir, and adding:

~~~toml
[asr]
aed_scheduler = "weighted_duration"
aed_worker_count = 2
aed_weight_artifact = "/mnt/hdd/jiajun/project/standalone_transcriber/outputs/measurements/20260917T-weighted-aed-device-calibration/weighted-aed-calibration.json"
~~~

Run the existing smoke input twice and stop only this service between runs:

~~~bash
.venv/bin/funasr-transcribe \
  outputs/measurements/20260909T-server-weighted-single-aed-punc-overlap-liuyangyang/smoke.wav \
  --config outputs/measurements/20260917T-weighted-aed-device-calibration/acceptance.toml \
  --output-dir outputs/measurements/20260917T-weighted-aed-device-calibration/restart-1 \
  --diagnostics basic
.venv/bin/funasr-transcribe service stop \
  --config outputs/measurements/20260917T-weighted-aed-device-calibration/acceptance.toml
.venv/bin/funasr-transcribe \
  outputs/measurements/20260909T-server-weighted-single-aed-punc-overlap-liuyangyang/smoke.wav \
  --config outputs/measurements/20260917T-weighted-aed-device-calibration/acceptance.toml \
  --output-dir outputs/measurements/20260917T-weighted-aed-device-calibration/restart-2 \
  --diagnostics basic
.venv/bin/funasr-transcribe service stop \
  --config outputs/measurements/20260917T-weighted-aed-device-calibration/acceptance.toml
~~~

Extract and compare frozen fields and transcripts:

~~~bash
jq -S '.scheduler | {planner,plan,longest_weight,padded_weight,weight_source,weight_artifact_checksum}' \
  outputs/measurements/20260917T-weighted-aed-device-calibration/restart-1/diagnostics.json > /tmp/weighted-aed-restart-1.json
jq -S '.scheduler | {planner,plan,longest_weight,padded_weight,weight_source,weight_artifact_checksum}' \
  outputs/measurements/20260917T-weighted-aed-device-calibration/restart-2/diagnostics.json > /tmp/weighted-aed-restart-2.json
cmp /tmp/weighted-aed-restart-1.json /tmp/weighted-aed-restart-2.json
cmp outputs/measurements/20260917T-weighted-aed-device-calibration/restart-{1,2}/transcript.json
~~~

Expected: comparisons succeed, source is artifact, checksum and weights match, and no fallback warning appears.

- [ ] **Step 5: Write the evidence report**

Record manifest/plan/records/report/artifact checksums; source/segment/case/batch/repeat counts; weights; grouped OOF and repeat noise; runtime identity; GPU0 snapshots and project-worker peaks; failure/OOM/safety/fingerprint/readiness results; restart comparison; and explicit statements that no validation/blind labels, contention model, memory cap, or production-config mutation occurred.

- [ ] **Step 6: Run final verification and commit only the report**

~~~bash
.venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_firered_client.py
.venv/bin/python -m pytest -q
git diff --check
git add docs/architecture/2026-09-17-weighted-aed-device-calibration-results.md
git diff --cached --check
git commit -m "docs: record weighted AED device calibration"
git status --short --branch
~~~

Expected: tests pass, only the report is committed, outputs stay ignored, and the old .gitignore change remains unstaged.

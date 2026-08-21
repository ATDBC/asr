# FireRedLID Removal and VAD Segment Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove FireRedLID completely from the production contract and add safe official batch processing for independent VAD segments, calibrated on physical GPU 0.

**Architecture:** The isolated FireRed worker remains the sole owner of a single official `FireRedAsr2System`; it starts with `enable_lid=False` and receives a validated `segment_batch_size` command argument. The official system batches AED and Punc calls internally, while the stable JSONL protocol, schemas, diagnostics, model registry, release gates and documentation stop exposing language-identification facts. A standalone benchmark command uses the same worker construction to compare batches without touching other GPU processes.

**Tech Stack:** Python 3.11, pytest, TOML, FireRedASR2S, PyTorch CUDA on physical GPU 0, `nvidia-smi`, JSON.

---

## File structure

- `src/funasr_transcriber/config.py` and `config/default.toml`: replace the LID threshold with the validated FireRed segment batch setting.
- `src/funasr_transcriber/firered_worker.py` and `src/funasr_transcriber/firered_client.py`: disable LID construction and pass batch configuration over the process boundary.
- `src/funasr_transcriber/firered_contract.py`, `schemas.py`, and `pipeline.py`: remove LID fields while retaining sentence confidence, timestamps and text-integrity checks.
- `src/funasr_transcriber/model_registry.py`, `doctor.py`, and `service.py`: remove LID model requirements and use the configured batch size for every normal and temporary FireRed owner.
- `src/funasr_transcriber/firered_benchmark.py` (new) and `tools/benchmark_firered_batches.py` (new): reusable synchronized timing, response comparison, candidate selection, and the GPU 0 calibration CLI.
- `src/funasr_transcriber/release_validation.py`, `tools/*.py`, `scripts/run_validation_linux.sh`, and current-facing docs: remove LID evidence and describe the three FireRed modules plus batching.
- Corresponding unit, integration and benchmark tests: establish the no-LID contract and benchmark decision logic before implementation.

Historical migration specifications remain historical records and are not rewritten. The existing local `FireRedLID` directory is not deleted.

### Task 1: Remove FireRedLID from model acquisition and verification

**Files:**
- Modify: `src/funasr_transcriber/model_registry.py:18-112,254-387,401-449`
- Modify: `tests/unit/test_model_registry.py:13-77`
- Modify: `tests/unit/test_firered_model_registry.py:17-117`
- Modify: `tests/integration/test_corpus_builder.py:192-216`
- Modify: `tools/build_release_validation_corpus.py:16-35`
- Modify: `src/funasr_transcriber/corpus_builder.py:600-617`
- Modify: `src/funasr_transcriber/corpus_verifier.py:512-545`

- [ ] **Step 1: Write the failing no-LID model-layout tests.**

  In `tests/unit/test_model_registry.py`, make `_complete_layout()` create only `paths.vad`, `paths.aed`, and `paths.punc`, and add:

  ```python
  def test_verify_models_does_not_require_firered_lid(tmp_path: Path, monkeypatch):
      _complete_layout(tmp_path)
      monkeypatch.setattr(
          "funasr_transcriber.model_registry._source_head",
          lambda path, runner=None: FIRERED_COMMIT,
      )
      assert verify_models(tmp_path).ok is True
      assert not hasattr(resolve_firered_paths(tmp_path), "lid")
  ```

  In `tests/unit/test_firered_model_registry.py`, change each expected download list to exactly `("vad", "aed", "punc")`, and assert no requested ModelScope/Hugging Face ID contains `FireRedLID`.

- [ ] **Step 2: Run the focused tests and verify they fail because LID remains required.**

  Run:

  ```bash
  pytest tests/unit/test_model_registry.py tests/unit/test_firered_model_registry.py tests/integration/test_corpus_builder.py -v
  ```

  Expected: failures identify `paths.lid`, `firered_lid`, or an LID download expectation.

- [ ] **Step 3: Implement the three-model registry.**

  Make `FIRERED_MODELS` contain only `vad`, `aed`, and `punc`; remove the `lid` member from `FireRedPaths`; and change every asset iteration and missing-model check to the same ordered tuple:

  ```python
  FIRERED_MODULES = ("vad", "aed", "punc")
  ```

  Keep the VAD-specific `VAD/model.pth.tar` validation and the existing atomic-download safeguards unchanged. Import `FIRERED_MODULES` from `release_validation` only if doing so would not introduce an import cycle; otherwise define a local immutable `FIRERED_MODEL_NAMES` and keep the release constant equal to that tuple. Update corpus manifest generation and verification to emit the three names.

- [ ] **Step 4: Re-run the focused tests.**

  Run:

  ```bash
  pytest tests/unit/test_model_registry.py tests/unit/test_firered_model_registry.py tests/integration/test_corpus_builder.py -v
  ```

  Expected: PASS; a complete layout has no LID directory, and no downloader receives an LID model ID.

- [ ] **Step 5: Commit the model-registry change.**

  ```bash
  git add src/funasr_transcriber/model_registry.py src/funasr_transcriber/corpus_builder.py src/funasr_transcriber/corpus_verifier.py tools/build_release_validation_corpus.py tests/unit/test_model_registry.py tests/unit/test_firered_model_registry.py tests/integration/test_corpus_builder.py
  git commit -m "refactor: remove FireRed LID model dependency"
  ```

### Task 2: Replace the LID response contract with AED-only sentence metadata

**Files:**
- Modify: `src/funasr_transcriber/schemas.py:35-56`
- Modify: `src/funasr_transcriber/firered_contract.py:157-303`
- Modify: `src/funasr_transcriber/firered_client.py:23-49`
- Modify: `src/funasr_transcriber/pipeline.py:11-84,126-171`
- Modify: `tests/unit/test_firered_contract.py`
- Modify: `tests/unit/test_firered_client.py`
- Modify: `tests/unit/test_pipeline.py`
- Modify: `tests/unit/test_schemas.py`
- Modify: `tests/unit/test_diagnostics.py`
- Modify: `tests/integration/test_asr_smoke.py`
- Modify: `tests/integration/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing contract and pipeline tests with no language fields.**

  Remove `lang` and `lang_confidence` from the official fixtures. Change the stable-response assertion to:

  ```python
  assert response["sentences"][0] == {
      "text": "你好，世界。",
      "start_ms": 100,
      "end_ms": 900,
      "asr_confidence": 0.95,
  }
  assert "language" not in response["sentences"][0]
  ```

  Replace `test_language_summary_is_duration_weighted` and the LID-warning test with a staged-pipeline test asserting that `diagnostics.asr` has `duration_ms`, `vad_segments_ms`, sentence text/times/AED confidence, and no `languages` key or language-warning text.

- [ ] **Step 2: Run the contract, client, schema and pipeline tests and verify they fail.**

  Run:

  ```bash
  pytest tests/unit/test_firered_contract.py tests/unit/test_firered_client.py tests/unit/test_schemas.py tests/unit/test_pipeline.py tests/unit/test_diagnostics.py -v
  ```

  Expected: failures require `language`/`language_confidence` or still import `summarize_languages`.

- [ ] **Step 3: Implement the AED-only data contract.**

  Change `RecognizedSentence` to:

  ```python
  @dataclass(frozen=True, slots=True)
  class RecognizedSentence:
      text: str
      start_ms: int
      end_ms: int
      asr_confidence: float
  ```

  Keep nonempty text, range validation and the `0..1` AED-confidence check. In `normalize_official_result()` and `validate_stable_response()`, stop reading or requiring LID fields. In `parse_worker_response()`, construct the new four-field sentence. Delete `LanguageSummary`, `summarize_languages()`, LID warnings and the LID threshold lookup. Make `_record_asr_metadata()` write only the retained sentence fields.

- [ ] **Step 4: Update real-smoke and service-smoke assertions.**

  Retain sentence presence, AED confidence, monotonic token and punctuation assertions. Replace language assertions with:

  ```python
  assert all(0.0 <= sentence.asr_confidence <= 1.0 for sentence in result.sentences)
  assert "languages" not in diagnostics["asr"]
  ```

- [ ] **Step 5: Re-run the focused contract suite.**

  Run:

  ```bash
  pytest tests/unit/test_firered_contract.py tests/unit/test_firered_client.py tests/unit/test_schemas.py tests/unit/test_pipeline.py tests/unit/test_diagnostics.py tests/integration/test_asr_smoke.py tests/integration/test_pipeline_smoke.py -v
  ```

  Expected: PASS; neither fake nor real-worker protocol fixtures contain LID values.

- [ ] **Step 6: Commit the contract change.**

  ```bash
  git add src/funasr_transcriber/schemas.py src/funasr_transcriber/firered_contract.py src/funasr_transcriber/firered_client.py src/funasr_transcriber/pipeline.py tests/unit/test_firered_contract.py tests/unit/test_firered_client.py tests/unit/test_schemas.py tests/unit/test_pipeline.py tests/unit/test_diagnostics.py tests/integration/test_asr_smoke.py tests/integration/test_pipeline_smoke.py
  git commit -m "refactor: remove LID from FireRed response contract"
  ```

### Task 3: Configure official VAD-segment batching in every FireRed worker

**Files:**
- Modify: `src/funasr_transcriber/config.py:25-33,84-91,222-237,287-294`
- Modify: `config/default.toml:11-15`
- Modify: `src/funasr_transcriber/firered_worker.py:23-30,117-145,238-292,295-313`
- Modify: `src/funasr_transcriber/firered_client.py:63-105`
- Modify: `src/funasr_transcriber/service.py:567-582,806-827`
- Modify: `src/funasr_transcriber/doctor.py:42-56`
- Modify (ignored, if present): `artifacts/release_validation_corpus/.service-work/local-validation.toml`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_firered_worker.py`
- Modify: `tests/unit/test_firered_client.py`
- Modify: `tests/unit/test_service.py`
- Modify: `tests/unit/test_doctor.py`

- [ ] **Step 1: Add failing configuration and command-construction tests.**

  Replace LID threshold tests with:

  ```python
  def test_default_asr_worker_configuration():
      config = load_config()
      assert config.asr.segment_batch_size == 1

  @pytest.mark.parametrize("setting", ("segment_batch_size = 0", "segment_batch_size = -1"))
  def test_asr_config_rejects_non_positive_segment_batch_size(tmp_path, setting):
      (tmp_path / "config.toml").write_text(f"[asr]\\n{setting}\\n", encoding="utf-8")
      with pytest.raises(ValueError, match="segment_batch_size"):
          load_config(tmp_path / "config.toml")
  ```

  Extend the client fixture test so `_command()` contains exactly one `--segment-batch-size` followed by the configured integer. Add a fake-system inventory assertion that keys are `vad`, `asr`, and `punc`, never `lid`.

- [ ] **Step 2: Run the focused worker/config tests and verify they fail.**

  Run:

  ```bash
  pytest tests/unit/test_config.py tests/unit/test_firered_worker.py tests/unit/test_firered_client.py tests/unit/test_service.py tests/unit/test_doctor.py -v
  ```

  Expected: failures report absent `segment_batch_size`, absent command option, or LID still present in inventory construction.

- [ ] **Step 3: Implement validated batch plumbing and disable LID loading.**

  Use this configuration shape:

  ```python
  "asr": {
      "worker_python": "/opt/funasr-transcriber/.venv-asr/bin/python",
      "timeout_seconds": 1800,
      "segment_batch_size": 1,
  }

  @dataclass(frozen=True, slots=True)
  class AsrConfig:
      worker_python: Path
      timeout_seconds: int
      segment_batch_size: int
  ```

  Reject values below 1. Add `segment_batch_size` to `FireRedWorkerClient.__init__`, include it in `_command()`, expose a positive-int parser argument in the worker, and pass it to `build_firered_system()`.

  In the worker, remove `FireRedLidConfig`, `lid_model_dir`, `lid_config`, and the LID inventory owner. Configure the upstream system as:

  ```python
  asr_batch_size=segment_batch_size,
  punc_batch_size=segment_batch_size,
  enable_vad=True,
  enable_lid=False,
  enable_punc=True,
  ```

  Pass `config.asr.segment_batch_size` from both resident and temporary service factories and the doctor factory. Update the protocol docstring to `VAD, AED, and punctuation`.

  If the ignored local validation TOML exists, remove its obsolete
  `lid_warning_confidence` entry and add `segment_batch_size = 1` in `[asr]` so
  strict config parsing continues to load that user-provided validation setup. Do
  not add that ignored file to Git.

- [ ] **Step 4: Re-run the focused worker/config tests.**

  Run:

  ```bash
  pytest tests/unit/test_config.py tests/unit/test_firered_worker.py tests/unit/test_firered_client.py tests/unit/test_service.py tests/unit/test_doctor.py -v
  ```

  Expected: PASS; values are validated, the child gets the batch argument, and no worker inventory requires LID.

- [ ] **Step 5: Commit worker batching.**

  ```bash
  git add src/funasr_transcriber/config.py config/default.toml src/funasr_transcriber/firered_worker.py src/funasr_transcriber/firered_client.py src/funasr_transcriber/service.py src/funasr_transcriber/doctor.py tests/unit/test_config.py tests/unit/test_firered_worker.py tests/unit/test_firered_client.py tests/unit/test_service.py tests/unit/test_doctor.py
  git commit -m "feat: batch independent FireRed VAD segments"
  ```

### Task 4: Remove LID evidence from release validation and current documentation

**Files:**
- Modify: `src/funasr_transcriber/release_validation.py`
- Modify: `tools/benchmark_ramc_e2e.py:279-437`
- Modify: `tools/stress_long_audio.py:119-191`
- Modify: `scripts/run_validation_linux.sh:96-180,198-306`
- Modify: `tests/benchmark/test_ramc_gate.py`
- Modify: `tests/benchmark/test_long_audio.py`
- Modify: `tests/benchmark/test_input_matrix.py`
- Modify: `tests/benchmark/test_overlap_gate.py`
- Modify: `tests/integration/test_validation_corpus_cli.py`
- Modify: `tests/integration/test_firered_documentation.py`
- Modify: `README.md`
- Modify: `RELEASE_CHECKLIST.md`
- Modify: `docs/architecture/current-transcription-pipeline.md`

- [ ] **Step 1: Write failing three-module release-evidence tests.**

  Change complete evidence fixtures to use:

  ```python
  "firered_modules": {"vad": True, "aed": True, "punc": True}
  ```

  Remove `lid_coverage` from expected evidence fields and assert `missing_release_evidence()` rejects missing timestamp/GPU/hash evidence but does not require LID coverage. Update source-inspection tests to require `FireRedVAD / FireRedASR2-AED / FireRedPunc` and reject runtime mentions of `FireRedLID` in README, checklist, runner and current architecture documentation.

- [ ] **Step 2: Run the release/documentation tests and verify failures.**

  Run:

  ```bash
  pytest tests/benchmark/test_ramc_gate.py tests/benchmark/test_long_audio.py tests/benchmark/test_input_matrix.py tests/benchmark/test_overlap_gate.py tests/integration/test_validation_corpus_cli.py tests/integration/test_firered_documentation.py -v
  ```

  Expected: failures identify LID coverage and four-module assumptions.

- [ ] **Step 3: Implement the three-module evidence contract.**

  Set `FIRERED_MODULES = ("vad", "aed", "punc")`; remove `_valid_lid`, the `lid_coverage` evidence field and every computation of language coverage. Keep timestamp-integrity, resource-profile, GPU-release, hash and quality gates unchanged. Update the Linux runner model preflight to require only VAD/AED/Punc files and update doctor evidence to the same three booleans. Preserve RAMC transcript preservation checks by concatenating `diagnostics.asr.sentences[*].text`.

- [ ] **Step 4: Update current user-facing documentation.**

  Describe `VAD -> AED -> Punc`, say LID is not loaded, and document `[asr].segment_batch_size` as an official batching control whose default remains 1 until calibration. Do not alter dated historical migration specifications or plans.

- [ ] **Step 5: Re-run the release/documentation tests.**

  Run:

  ```bash
  pytest tests/benchmark/test_ramc_gate.py tests/benchmark/test_long_audio.py tests/benchmark/test_input_matrix.py tests/benchmark/test_overlap_gate.py tests/integration/test_validation_corpus_cli.py tests/integration/test_firered_documentation.py -v
  ```

  Expected: PASS; no current release gate or documentation test requires LID.

- [ ] **Step 6: Commit release and documentation updates.**

  ```bash
  git add src/funasr_transcriber/release_validation.py tools/benchmark_ramc_e2e.py tools/stress_long_audio.py scripts/run_validation_linux.sh tests/benchmark/test_ramc_gate.py tests/benchmark/test_long_audio.py tests/benchmark/test_input_matrix.py tests/benchmark/test_overlap_gate.py tests/integration/test_validation_corpus_cli.py tests/integration/test_firered_documentation.py README.md RELEASE_CHECKLIST.md docs/architecture/current-transcription-pipeline.md
  git commit -m "refactor: remove LID release validation evidence"
  ```

### Task 5: Add repeatable GPU 0 batch calibration and select the production default

**Files:**
- Create: `src/funasr_transcriber/firered_benchmark.py`
- Create: `tools/benchmark_firered_batches.py`
- Create: `tests/unit/test_firered_benchmark.py`
- Modify: `config/default.toml`
- Modify: `src/funasr_transcriber/config.py`
- Modify: `docs/architecture/current-transcription-pipeline.md`

- [ ] **Step 1: Write failing pure benchmark tests.**

  Add fixtures containing the required stable response keys and candidate records. Test these public helpers:

  ```python
  def stable_response_fingerprint(response: Mapping[str, object]) -> str: ...
  def summarize_stage_calls(calls: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float | int]]: ...
  def choose_batch_candidate(candidates: Sequence[Mapping[str, object]]) -> int: ...
  ```

  Require fingerprints to change when token timestamps change; summaries to return `calls`, `total_seconds`, `median_call_seconds`, and `p95_call_seconds`; and selection to choose the lowest median passing candidate, breaking equal medians in favour of the lower batch. Candidates marked failed, nonmatching baseline fingerprint, OOM, timeout or over-budget must never be selected. If none pass, return `1`.

- [ ] **Step 2: Run the new unit test and verify it fails because the module is absent.**

  Run:

  ```bash
  pytest tests/unit/test_firered_benchmark.py -v
  ```

  Expected: collection failure for `funasr_transcriber.firered_benchmark`.

- [ ] **Step 3: Implement the pure benchmark helpers.**

  Serialize the exact stable-result comparison payload with sorted JSON keys:

  ```python
  {
      "text": response["text"],
      "tokens": response["tokens"],
      "sentences": response["sentences"],
      "vad_segments_ms": response["vad_segments_ms"],
      "duration_ms": response["duration_ms"],
  }
  ```

  Hash its UTF-8 bytes with SHA-256. Compute p95 through sorted linear interpolation, not a dependency on NumPy. A passing candidate must have all repetitions successful, identical fingerprint to the batch-1 baseline, and `peak_process_mib <= maximum_process_mib`; selection sorts by `(median_end_to_end_seconds, batch_size)`.

- [ ] **Step 4: Implement the calibration CLI.**

  `tools/benchmark_firered_batches.py` accepts required `--audio`, `--model-dir`, `--output`, optional `--batches 1 2 4 8`, `--repeats 3`, `--warmup-audio`, `--maximum-process-mib`, and `--device cuda:0`. It must:

  1. reject non-`cuda:0` device and nonpositive/repeated batches;
  2. load one FireRed system per candidate with `enable_lid=False` and that candidate batch;
  3. run an unrecorded warm-up before resetting CUDA peak statistics;
  4. wrap VAD `detect`, AED `transcribe`, and Punc `process_with_timestamp` with `torch.cuda.synchronize(0)` before and after each call;
  5. normalize each official result with `normalize_official_result`, record its fingerprint, end-to-end time and the calling process's `nvidia-smi` MiB only;
  6. release only its own system (`del`, `gc.collect`, `torch.cuda.empty_cache`) between candidates; never signal or list another process in JSON;
  7. atomically write the JSON summary under the requested ignored `outputs/measurements/` path, including candidate failures and `recommended_segment_batch_size`.

  The CLI exits zero after a completed benchmark even if the recommendation is 1; it exits nonzero only for invalid arguments, unavailable GPU 0, unreadable input or inability to write the report.

- [ ] **Step 5: Re-run pure benchmark tests.**

  Run:

  ```bash
  pytest tests/unit/test_firered_benchmark.py -v
  ```

  Expected: PASS; candidate selection is deterministic and rejects nonmatching output.

- [ ] **Step 6: Run the complete CPU test suite.**

  Run:

  ```bash
  pytest -q
  ```

  Expected: all non-GPU tests pass; GPU-marked integration/benchmark tests remain skipped unless their documented environment variables are set.

- [ ] **Step 7: Run the real GPU 0 calibration without touching other processes.**

  Confirm that no standalone transcriber service is live, query GPU 0 read-only, then run:

  ```bash
  .venv/bin/python tools/benchmark_firered_batches.py \
      --audio artifacts/release_validation_corpus/ramc_core/WAV/CTS-CN-F2F-2019-11-15-1347.wav \
      --warmup-audio artifacts/release_validation_corpus/seeds/format_fixture.wav \
      --model-dir models \
      --output outputs/measurements/$(date -u +%Y%m%dT%H%M%SZ)-firered-batches/result.json \
      --batches 1 2 4 8 \
      --repeats 3 \
      --maximum-process-mib 12000 \
      --device cuda:0
  ```

  Before every candidate, require physical GPU 0 and record its free MiB. Stop only the calibration's own worker on exit; do not stop, pause, migrate or otherwise affect external GPU processes. Verify after completion that GPU 0 returns within the existing process's pre-run baseline tolerance.

- [ ] **Step 8: Set and verify the calibrated default.**

  Read `recommended_segment_batch_size` from the generated JSON. Set that exact positive integer in both `DEFAULTS["asr"]["segment_batch_size"]` and `config/default.toml`, retain 1 when the report recommends 1, and record the selected value and report path in the current architecture document. Run the real service smoke once with the same value and assert an output transcript, diagnostics without `languages`, and a worker inventory without `lid`.

  Set the same selected integer in the ignored local validation TOML when that
  file exists; leave it untracked.

- [ ] **Step 9: Commit the benchmark, calibrated default and verification evidence.**

  ```bash
  git add src/funasr_transcriber/firered_benchmark.py tools/benchmark_firered_batches.py tests/unit/test_firered_benchmark.py src/funasr_transcriber/config.py config/default.toml docs/architecture/current-transcription-pipeline.md
  git commit -m "perf: calibrate FireRed VAD segment batching"
  ```

### Task 6: Final verification and handoff

**Files:**
- Verify: all files modified in Tasks 1-5

- [ ] **Step 1: Check formatting and repository state.**

  Run:

  ```bash
  git diff --check
  git status --short
  ```

  Expected: no whitespace errors and no unintended tracked changes. Ignored measurement JSON is allowed only below `outputs/measurements/`.

- [ ] **Step 2: Run the full test suite after the calibrated default change.**

  Run:

  ```bash
  pytest -q
  ```

  Expected: all applicable tests pass.

- [ ] **Step 3: Inspect the calibration report and service output.**

  Verify that the chosen candidate has exactly the batch-1 response fingerprint, no LID model/inventory/diagnostic fields, valid timestamp evidence, and a process-memory peak at or below the configured calibration cap.

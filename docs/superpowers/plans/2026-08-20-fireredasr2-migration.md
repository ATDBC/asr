# FireRedASR2S Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the production FunASR recognition path with a persistent, isolated FireRedASR2S worker while preserving speaker alignment, overlap recovery, output formats, diagnostics, and release gates.

**Architecture:** The main Python 3.11 environment never imports FireRedASR2S. It starts a JSON Lines worker from `.venv-asr`, closes that worker before loading 3D-Speaker, prepares overlap tracks while the speaker model is resident, releases all speaker/overlap GPU state, and starts at most one second FireRed worker for every accepted track. Pure response normalization and lifecycle orchestration stay independently unit-testable without CUDA or model weights.

**Tech Stack:** Python 3.11, pytest, subprocess JSON Lines, FireRedASR2S pinned at `4e7d9aaf4482a47cec1724807026b9b151926eb5`, ModelScope/Hugging Face, PyTorch CUDA, 3D-Speaker, pyannote/ClearVoice, Bash.

**Resolved upstream detail:** At the pinned commit, `FireRedAsr2System` exposes word text and timestamps but only sentence/utterance-level `asr_confidence`. The stable worker contract still includes token `confidence`; each token receives the `asr_confidence` of the sentence that owns it. This is deterministic and does not invent a word-level score.

---

## File map

- Create `.gitignore`: keep virtual environments, models, caches, work output, and generated validation corpora out of Git.
- Create `requirements-asr.txt`: dependencies used only for source/model acquisition in `.venv-asr` in addition to the pinned upstream package.
- Create `src/funasr_transcriber/firered_contract.py`: normalize and validate official FireRed dictionaries without importing FireRed.
- Create `src/funasr_transcriber/firered_worker.py`: load the four official modules and serve the line protocol inside `.venv-asr`.
- Create `src/funasr_transcriber/firered_client.py`: persistent subprocess client, strict request/response correlation, timeout, shutdown, and conversion to `AsrResult`.
- Create `src/funasr_transcriber/gpu.py`: `nvidia-smi` preflight and per-stage/release measurements.
- Modify `src/funasr_transcriber/schemas.py`: carry FireRed confidence, sentence LID, VAD, and duration metadata across the main-process boundary.
- Modify `src/funasr_transcriber/config.py` and `config/default.toml`: add ASR worker/runtime paths, timeout, and LID warning threshold.
- Modify `src/funasr_transcriber/model_registry.py`: retain speaker assets, replace FunASR ASR assets with pinned FireRed source and four atomic model directories.
- Modify `src/funasr_transcriber/pipeline.py`: consume a completed `AsrResult`, aggregate LID diagnostics, and expose base/final stages for explicit model lifecycle control.
- Modify `src/funasr_transcriber/overlap.py`: split GPU-heavy overlap preparation from FireRed track decoding.
- Modify `src/funasr_transcriber/speaker_diarization.py`: explicit close/context-manager support.
- Modify `src/funasr_transcriber/diagnostics.py`: ASR/LID and GPU-stage diagnostics.
- Modify `src/funasr_transcriber/doctor.py` and `src/funasr_transcriber/cli.py`: verify the worker and orchestrate the three GPU phases.
- Modify `pyproject.toml`, `scripts/install_linux.sh`, `scripts/run_validation_linux.sh`, `README.md`, and `RELEASE_CHECKLIST.md`: isolated installation and FireRed release workflow.
- Delete `src/funasr_transcriber/asr.py` and obsolete FunASR adapter tests after all consumers move to the worker client.

## Task 1: Establish a reproducible Python 3.11 baseline

**Files:**
- Create: `.gitignore`
- Modify: `pyproject.toml`
- Test: `tests/integration/test_install_script.py`

- [ ] **Step 1: Install the already-configured Ubuntu Python 3.11 packages**

Run:

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

Expected: the final command prints `Python 3.11.x`. Do not proceed with Python 3.12.

- [ ] **Step 2: Write the repository hygiene test first**

Add to `tests/integration/test_install_script.py`:

```python
def test_local_runtime_artifacts_are_gitignored():
    ignored = (Path(__file__).parents[2] / ".gitignore").read_text("utf-8")
    for entry in (".venv/", ".venv-asr/", ".venv-overlap/", "models/", ".pytest_cache/", "artifacts/release_validation_corpus/"):
        assert entry in ignored
```

- [ ] **Step 3: Run the test and verify RED**

Run: `python3.11 -m pytest tests/integration/test_install_script.py::test_local_runtime_artifacts_are_gitignored -q`

Expected: FAIL because `.gitignore` does not exist.

- [ ] **Step 4: Add the minimal ignore file**

Create `.gitignore`:

```gitignore
.venv/
.venv-asr/
.venv-overlap/
models/
.pytest_cache/
__pycache__/
*.py[cod]
*.egg-info/
.coverage
htmlcov/
artifacts/release_validation_corpus/
validation_results/
```

- [ ] **Step 5: Create the core development environment and run the baseline suite**

Run:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip==25.2 setuptools==80.9.0 wheel==0.45.1
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

Expected: the ignore test passes and the pre-migration suite passes. Record any model-gated skips; do not treat those skips as failures.

- [ ] **Step 6: Create the baseline commit**

```bash
git add .gitignore README.md RELEASE_CHECKLIST.md config docs pyproject.toml requirements-overlap.txt scripts src tests tools
git commit -m "chore: establish standalone transcriber baseline"
```

## Task 2: Define FireRed-aware configuration and immutable schemas

**Files:**
- Modify: `src/funasr_transcriber/config.py`
- Modify: `config/default.toml`
- Modify: `src/funasr_transcriber/schemas.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_schemas.py`

- [ ] **Step 1: Write failing configuration and schema tests**

Add tests that assert this public shape:

```python
def test_default_asr_worker_configuration():
    config = load_config()
    assert config.asr.worker_python == Path("/opt/funasr-transcriber/.venv-asr/bin/python")
    assert config.asr.timeout_seconds == 1800
    assert config.asr.lid_warning_confidence == 0.80

def test_asr_result_carries_firered_metadata():
    sentence = RecognizedSentence("你好。", 100, 900, "zh mandarin", 0.99, 0.95)
    result = AsrResult(
        "你好。",
        (TimedToken("你好。", 100, 900, 0.95),),
        sentences=(sentence,),
        vad_segments_ms=((80, 920),),
        duration_ms=1000,
    )
    assert result.tokens[0].confidence == 0.95
    assert result.sentences == (sentence,)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/python -m pytest tests/unit/test_config.py tests/unit/test_schemas.py -q`

Expected: FAIL on missing `asr`, `RecognizedSentence`, and `TimedToken.confidence`.

- [ ] **Step 3: Add the exact schema types**

Implement in `schemas.py`:

```python
@dataclass(frozen=True, slots=True)
class RecognizedSentence:
    text: str
    start_ms: int
    end_ms: int
    language: str
    language_confidence: float
    asr_confidence: float

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text must not be empty")
        _validate_range(self.start_ms, self.end_ms)
        for name, value in (("language_confidence", self.language_confidence), ("asr_confidence", self.asr_confidence)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must stay within 0..1")

@dataclass(frozen=True, slots=True)
class TimedToken:
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None

@dataclass(frozen=True, slots=True)
class AsrResult:
    text: str
    tokens: tuple[TimedToken, ...]
    untimed_sentences: tuple[SentenceSpan, ...] = ()
    sentences: tuple[RecognizedSentence, ...] = ()
    vad_segments_ms: tuple[tuple[int, int], ...] = ()
    duration_ms: int = 0
```

Extend `TimedToken.__post_init__` to reject confidence outside `0..1`, and validate positive `duration_ms` plus all sentence/VAD/token upper bounds when duration is nonzero.

- [ ] **Step 4: Add `AsrConfig` and defaults**

Use these fields exactly:

```python
@dataclass(frozen=True, slots=True)
class AsrConfig:
    worker_python: Path
    timeout_seconds: int
    lid_warning_confidence: float
```

Add `[asr]` defaults to both configuration sources, build paths with `expanduser()`, and validate timeout `> 0` and LID threshold within `0..1`.

- [ ] **Step 5: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/unit/test_config.py tests/unit/test_schemas.py -q`

Expected: PASS.

```bash
git add config/default.toml src/funasr_transcriber/config.py src/funasr_transcriber/schemas.py tests/unit/test_config.py tests/unit/test_schemas.py
git commit -m "feat: define FireRed ASR configuration and metadata"
```

## Task 3: Normalize and validate official FireRed results

**Files:**
- Create: `src/funasr_transcriber/firered_contract.py`
- Create: `tests/unit/test_firered_contract.py`

- [ ] **Step 1: Write table-driven RED tests for the stable contract**

Use one canonical official result and assert punctuation attachment, sentence confidence propagation, and millisecond bounds:

```python
OFFICIAL = {
    "uttid": "req-1",
    "text": "你好，世界。Hello 2!",
    "sentences": [
        {"start_ms": 100, "end_ms": 900, "text": "你好，世界。", "asr_confidence": 0.95, "lang": "zh mandarin", "lang_confidence": 0.99},
        {"start_ms": 1000, "end_ms": 1800, "text": "Hello 2!", "asr_confidence": 0.91, "lang": "en", "lang_confidence": 0.98},
    ],
    "vad_segments_ms": [(80, 920), (980, 1820)],
    "dur_s": 2.0,
    "words": [
        {"start_ms": 120, "end_ms": 240, "text": "你"},
        {"start_ms": 240, "end_ms": 360, "text": "好"},
        {"start_ms": 400, "end_ms": 520, "text": "世"},
        {"start_ms": 520, "end_ms": 700, "text": "界"},
        {"start_ms": 1050, "end_ms": 1450, "text": "hello"},
        {"start_ms": 1450, "end_ms": 1700, "text": "2"},
    ],
}

def test_normalize_attaches_punctuation_and_confidence():
    response = normalize_official_result("req-1", OFFICIAL)
    assert [item["text"] for item in response["tokens"]] == ["你", "好，", "世", "界。", "hello", "2!"]
    assert [item["confidence"] for item in response["tokens"]] == [0.95] * 4 + [0.91] * 2
    assert response["duration_ms"] == 2000
```

Parametrize failures for empty VAD, text without words, word/text mismatch, out-of-bounds/zero-length/unsorted words, overlapping sentence ownership, sentence/text mismatch, and VAD segments longer than `60_000` ms. Include NFKC cases for full-width digits and whitespace.

- [ ] **Step 2: Run and verify RED**

Run: `.venv/bin/python -m pytest tests/unit/test_firered_contract.py -q`

Expected: collection FAIL because the module does not exist.

- [ ] **Step 3: Implement the pure normalizer**

Expose these concrete public call signatures; their behavior is fully specified by the rules immediately below and the table-driven tests from Step 1:

```python
normalize_text(text: str) -> str
attach_sentence_punctuation(words: list[dict[str, object]], sentence_text: str) -> list[str]
validate_stable_response(response: dict[str, object]) -> None
normalize_official_result(request_id: str, raw: dict[str, object]) -> dict[str, object]
```

Rules are exact: NFKC-normalize and ignore whitespace for comparison; recognize punctuation with Unicode category prefix `P`; attach in-sentence punctuation to the previous token and leading punctuation to the next token; assign words to exactly one sentence by midpoint; inherit that sentence's ASR confidence; reject speech segments over `60_000` ms before returning; require monotonically nondecreasing starts and ends and `0 <= start < end <= duration`; require normalized token, sentence, and full text equality. Raise `NoSpeechError` for empty VAD and `ContractError` for all other violations.

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/unit/test_firered_contract.py -q`

Expected: PASS.

```bash
git add src/funasr_transcriber/firered_contract.py tests/unit/test_firered_contract.py
git commit -m "feat: normalize FireRed worker results"
```

## Task 4: Implement the isolated FireRed worker

**Files:**
- Create: `src/funasr_transcriber/firered_worker.py`
- Create: `tests/unit/test_firered_worker.py`

- [ ] **Step 1: Write RED tests around an injected fake system**

Test `transcribe`, `verify`, `shutdown`, invalid JSON, unsupported actions, relative/non-WAV paths, 16 kHz mono PCM enforcement, and per-request exception serialization. Drive `serve(input_stream, protocol_output, system_factory, audio_info)` using `io.StringIO`; assert exactly one JSON object per non-shutdown request and no logs in protocol output.

```python
def test_worker_correlates_transcribe_response(tmp_path):
    wav = tmp_path / "normalized.wav"
    wav.write_bytes(b"fake")
    source = io.StringIO(json.dumps({"id": "abc", "action": "transcribe", "audio_path": str(wav.resolve())}) + "\n")
    target = io.StringIO()
    serve(source, target, lambda: FakeSystem(OFFICIAL), lambda path: AudioInfo(16000, 1, "PCM_16"))
    payload = json.loads(target.getvalue())
    assert payload["id"] == "abc"
    assert payload["ok"] is True
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/unit/test_firered_worker.py -q`

Expected: FAIL because `firered_worker` does not exist.

- [ ] **Step 3: Implement model construction and the VAD guard**

Build `FireRedAsr2SystemConfig` with `enable_vad/lid/punc=True`, `asr_type="aed"`, `FireRedAsr2Config(return_timestamp=True, use_gpu=True)`, and `FireRedVadConfig(use_gpu=False, max_speech_frame=2000)`. Wrap `system.vad` in `GuardedVad`; its `detect()` delegates, checks every returned duration is at most 60 seconds, then returns the unchanged pair. Resolve model directories as:

```text
<model_root>/fireredasr2s/FireRedVAD/VAD
<model_root>/fireredasr2s/FireRedLID
<model_root>/fireredasr2s/FireRedASR2-AED
<model_root>/fireredasr2s/FireRedPunc
```

- [ ] **Step 4: Implement the protocol loop**

Capture the original stdout before importing FireRed, redirect ordinary `sys.stdout` to stderr, and write protocol responses only through the captured stream. `verify` runs the same full four-module inference against its absolute `audio_path`. For a request ID of `abc`, `shutdown` writes `{"id": "abc", "ok": true, "shutdown": true}` and returns. A `ValueError("bad audio")` response is `{"id": "abc", "ok": false, "error": {"type": "ValueError", "message": "bad audio"}}`.

- [ ] **Step 5: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/unit/test_firered_worker.py -q`

Expected: PASS.

```bash
git add src/funasr_transcriber/firered_worker.py tests/unit/test_firered_worker.py
git commit -m "feat: add isolated FireRed JSONL worker"
```

## Task 5: Add the strict persistent worker client

**Files:**
- Create: `src/funasr_transcriber/firered_client.py`
- Create: `tests/unit/test_firered_client.py`

- [ ] **Step 1: Write RED protocol tests**

Create a temporary Python worker fixture that can echo, send a wrong/duplicate ID, print invalid JSON, sleep, or exit. Tests must cover all design failures plus clean shutdown and reuse across two transcriptions.

```python
def test_client_reuses_one_process_for_multiple_requests(fake_worker, tmp_path):
    with FireRedWorkerClient(sys.executable, fake_worker, tmp_path, 1.0, tmp_path / "worker.log") as client:
        first = client.transcribe((tmp_path / "one.wav").resolve())
        pid = client.pid
        second = client.transcribe((tmp_path / "two.wav").resolve())
        assert client.pid == pid
    assert first.text == "一。"
    assert second.text == "二。"
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/unit/test_firered_client.py -q`

Expected: FAIL because the client module is missing.

- [ ] **Step 3: Implement client lifecycle**

`FireRedWorkerClient` starts `[str(worker_python), '-m', 'funasr_transcriber.firered_worker', '--model-root', str(model_root.resolve()), '--device', device]` with text pipes, line buffering, a dedicated stderr log file, and a Linux selector around stdout. Every call generates `uuid.uuid4().hex`, writes one compact JSON line, waits no longer than `timeout_seconds`, parses exactly one object, verifies unseen matching ID, and raises `TranscriptionError` on any protocol or worker error. `close()` sends `shutdown`, waits 10 seconds, then `terminate()` and finally `kill()` if required; it is idempotent and closes all descriptors.

Convert successful dictionaries to the schema types from Task 2 in `parse_worker_response()`; never import FireRed in this module.

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/unit/test_firered_client.py -q`

Expected: PASS with no leaked subprocess warnings.

```bash
git add src/funasr_transcriber/firered_client.py tests/unit/test_firered_client.py
git commit -m "feat: add persistent FireRed worker client"
```

## Task 6: Replace the ASR model registry and make downloads atomic

**Files:**
- Modify: `src/funasr_transcriber/model_registry.py`
- Create: `tests/unit/test_firered_model_registry.py`
- Modify: `tests/unit/test_model_registry.py`

- [ ] **Step 1: Write RED registry tests**

Assert `FireRedPaths` resolves the four directories and pinned source, verification reports each missing component and a wrong source HEAD, an already-valid target is never overwritten, and download commands use a sibling temporary directory followed by `Path.replace()`. Mock subprocess/download execution; do not access the network.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/unit/test_model_registry.py tests/unit/test_firered_model_registry.py -q`

Expected: FAIL on missing FireRed paths.

- [ ] **Step 3: Implement pinned source and model acquisition**

Define:

```python
FIRERED_COMMIT = "4e7d9aaf4482a47cec1724807026b9b151926eb5"
FIRERED_MODELS = {
    "vad": ("xukaituo/FireRedVAD", "FireRedTeam/FireRedVAD", "FireRedVAD"),
    "lid": ("xukaituo/FireRedLID", "FireRedTeam/FireRedLID", "FireRedLID"),
    "aed": ("xukaituo/FireRedASR2-AED", "FireRedTeam/FireRedASR2-AED", "FireRedASR2-AED"),
    "punc": ("xukaituo/FireRedPunc", "FireRedTeam/FireRedPunc", "FireRedPunc"),
}
```

Clone source into a temporary sibling, checkout the exact commit, verify `rev-parse HEAD`, and atomically rename to `source`. Download each ModelScope ID into a temporary sibling, verify it contains files (and `FireRedVAD/VAD` exists for VAD), then rename. Accept Hugging Face only through an explicit `source="huggingface"` option. Never delete or replace an existing nonempty invalid directory; report it as `ModelError`.

Keep 3D-Speaker/FSMN/CAMPPlus verification and download logic intact, but remove Paraformer and FunASR punctuation assets plus `build_asr_model()`.

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/unit/test_model_registry.py tests/unit/test_firered_model_registry.py -q`

Expected: PASS.

```bash
git add src/funasr_transcriber/model_registry.py tests/unit/test_model_registry.py tests/unit/test_firered_model_registry.py
git commit -m "feat: manage pinned FireRed source and models"
```

## Task 7: Add enforceable GPU admission and release diagnostics

**Files:**
- Create: `src/funasr_transcriber/gpu.py`
- Create: `tests/unit/test_gpu.py`
- Modify: `src/funasr_transcriber/diagnostics.py`
- Modify: `tests/unit/test_diagnostics.py`

- [ ] **Step 1: Write RED GPU tests**

Feed representative `nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits` output. Assert a 24,576 MiB card passes total memory, 12,288 MiB free passes admission, 12,287 MiB fails, invalid `cuda:N` fails, and command/parse failure is fatal. Test release sampling and diagnostics JSON fields. The free-memory threshold was revised after real inference measured a 9,256 MiB process delta on a near-maximum speech segment.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/unit/test_gpu.py tests/unit/test_diagnostics.py -q`

Expected: FAIL because `gpu.py` and GPU diagnostics are absent.

- [ ] **Step 3: Implement exact thresholds and stage records**

Expose `query_gpu(device, runner=subprocess.run) -> GpuSample` and `require_gpu_capacity(device, minimum_total_mib=23 * 1024, minimum_free_mib=12 * 1024)`. Add `Diagnostics.gpu_stages: dict[str, dict[str, object]]`, recording before/after free MiB, worker-reported peak bytes, load seconds, inference seconds, and `released` (`after_free_mib >= before_free_mib - 512`). The release tolerance was revised after a real Speaker close left 261 MiB of CUDA context overhead while releasing all model memory. Failure to query is an environment error, never a warning.

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/unit/test_gpu.py tests/unit/test_diagnostics.py -q`

Expected: PASS.

```bash
git add src/funasr_transcriber/gpu.py src/funasr_transcriber/diagnostics.py tests/unit/test_gpu.py tests/unit/test_diagnostics.py
git commit -m "feat: enforce GPU memory admission and diagnostics"
```

## Task 8: Split overlap preparation from track ASR

**Files:**
- Modify: `src/funasr_transcriber/overlap.py`
- Modify: `tests/unit/test_overlap.py`
- Modify: `src/funasr_transcriber/speaker_diarization.py`
- Modify: `tests/unit/test_speaker_diarization.py`

- [ ] **Step 1: Write RED two-phase lifecycle tests**

Define immutable `PreparedTrack(path, speaker_id, score, margin, rms)` and `PreparedWindow(id, start_ms, end_ms, tracks)`. Assert `prepare()` performs detect/separate/centroid/embedding work without calling ASR; `apply()` transcribes every candidate through one supplied engine and retains the existing acceptance/degradation semantics. Assert `SpeakerDiarizer.close()` removes model references, calls CUDA cache cleanup, and is idempotent.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/unit/test_overlap.py tests/unit/test_speaker_diarization.py -q`

Expected: FAIL because the overlap implementation is still single-phase.

- [ ] **Step 3: Implement preparation**

Rename the current GPU preparation half to `SubprocessOverlapPreparer.prepare(audio_path, base_tokens, speaker_segments) -> OverlapPreparation`. It performs detection, separation, RMS filtering, centroid calculation, and speaker matching. It returns warnings and only tracks with non-null speaker IDs, scores, and margins; it does not decide text completeness and does not call ASR.

- [ ] **Step 4: Implement application**

Add `apply_prepared_overlap(preparation, base_tokens, asr_engine, min_score, min_margin)`. It transcribes all tracks sequentially through the one engine, builds assessments, applies `accept_separated_window`, translates accepted token times to the global timeline, and preserves original tokens plus a warning for every rejected/failed window. A track ASR error degrades only that overlap window.

- [ ] **Step 5: Add explicit speaker cleanup**

Make `SpeakerDiarizer` a context manager. `close()` drops `embedding_model`, `vad_model`, and other GPU-owned fields, runs `gc.collect()`, and calls `torch.cuda.empty_cache()` when CUDA is active.

- [ ] **Step 6: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/unit/test_overlap.py tests/unit/test_speaker_diarization.py -q`

Expected: PASS.

```bash
git add src/funasr_transcriber/overlap.py src/funasr_transcriber/speaker_diarization.py tests/unit/test_overlap.py tests/unit/test_speaker_diarization.py
git commit -m "refactor: split overlap preparation from FireRed decoding"
```

## Task 9: Refactor the pipeline into explicit lifecycle-safe stages

**Files:**
- Modify: `src/funasr_transcriber/pipeline.py`
- Modify: `tests/unit/test_pipeline.py`
- Delete: `tests/unit/test_local_redecode.py`

- [ ] **Step 1: Write RED stage and LID tests**

Test a pipeline API with `attribute_primary(asr_result, speaker_segments)`, `summarize_languages(sentences)`, and `finish(asr_result, attributed_tokens, overlap_enhancement, diagnostics)`. Assert language duration proportions, primary language, low-confidence/non-Mandarin warnings, 100% text integrity without local re-decode, and existing one-second turn merging.

```python
def test_language_summary_is_duration_weighted():
    summary = summarize_languages((
        RecognizedSentence("甲。", 0, 9000, "zh mandarin", 0.95, 0.9),
        RecognizedSentence("Hi.", 9000, 10000, "en", 0.99, 0.9),
    ))
    assert summary.primary_language == "zh mandarin"
    assert summary.duration_ratio["zh mandarin"] == pytest.approx(0.9)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/unit/test_pipeline.py -q`

Expected: FAIL on missing staged API.

- [ ] **Step 3: Implement pure stages and remove local redecode production use**

The FireRed contract forbids untimed success, so reject any `untimed_sentences`. `attribute_primary()` only aligns complete tokens. `finish()` applies prepared overlap output when present, builds turns, and checks integrity when overlap is unmodified. Add `Diagnostics.asr: dict[str, object] = field(default_factory=dict)` and store the LID summary under `diagnostics.asr["languages"]`; warn when primary language differs from `zh mandarin` or any duration-bearing sentence has confidence below `config.asr.lid_warning_confidence`.

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/unit/test_pipeline.py tests/unit/test_alignment.py tests/unit/test_turns.py -q`

Expected: PASS.

```bash
git add src/funasr_transcriber/pipeline.py tests/unit/test_pipeline.py tests/unit/test_local_redecode.py
git commit -m "refactor: expose lifecycle-safe transcription stages"
```

## Task 10: Wire the CLI to the three GPU phases

**Files:**
- Modify: `src/funasr_transcriber/cli.py`
- Create: `tests/unit/test_cli_lifecycle.py`
- Modify: `tests/unit/test_cli.py`
- Delete: `src/funasr_transcriber/asr.py`
- Delete: `tests/unit/test_asr.py`

- [ ] **Step 1: Write a RED ordered-event test**

Inject factories and capture these exact events:

```python
assert events == [
    "gpu-check:primary-asr", "asr:start", "asr:transcribe", "asr:close",
    "gpu-check:speaker", "speaker:start", "speaker:diarize",
    "overlap:prepare", "speaker:close",
    "gpu-check:overlap-asr", "asr:start", "asr:track-1", "asr:track-2", "asr:close",
    "export",
]
```

Add a no-overlap test proving FireRed starts only once, a base-ASR failure test proving there is no FunASR fallback, and a track failure test proving only overlap degrades.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/unit/test_cli.py tests/unit/test_cli_lifecycle.py -q`

Expected: FAIL because CLI still imports `FunAsrEngine` and co-loads models.

- [ ] **Step 3: Implement explicit orchestration**

After normalization, require GPU capacity, open one `FireRedWorkerClient`, transcribe primary audio, collect worker timing/peak metadata, and close. Require capacity again, open `SpeakerDiarizer`, diarize and prepare overlap, then close it. If and only if preparation has candidate tracks, require capacity and open one second client to apply every track. Build/export results after all GPU contexts close. Update progress messages to describe actual phases.

- [ ] **Step 4: Remove the legacy route and prove it statically**

Delete the FunASR adapter and add:

```python
def test_production_source_has_no_funasr_automodel_import():
    source = Path("src/funasr_transcriber")
    text = "\n".join(path.read_text("utf-8") for path in source.glob("*.py"))
    assert "from funasr import AutoModel" not in text
    assert "FunAsrEngine" not in text
```

- [ ] **Step 5: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/unit -q`

Expected: PASS.

```bash
git add src tests/unit
git commit -m "feat: orchestrate isolated FireRed GPU phases"
```

## Task 11: Make doctor verify the complete FireRed path

**Files:**
- Modify: `src/funasr_transcriber/doctor.py`
- Modify: `src/funasr_transcriber/cli.py`
- Modify: `tests/integration/test_doctor.py`

- [ ] **Step 1: Write RED doctor tests**

Assert doctor rejects missing `.venv-asr/bin/python`, wrong FireRed source commit, any missing module, total GPU below 23 GiB, free GPU below 12 GiB, and unavailable monitoring. With injected fakes, assert it sends `verify` with `<model_root>/fireredasr2s/source/assets/hello_zh.wav`, checks the full normalized response, closes the worker, samples released memory, then separately verifies speaker and overlap environments.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/integration/test_doctor.py -q`

Expected: FAIL because doctor still builds FunASR in-process.

- [ ] **Step 3: Implement doctor sequence**

Use the same factories and GPU checks as production. `--skip-cuda` may skip CUDA availability only for environment-oriented tests; it must not report the full model smoke as passed. Print explicit `firered_source`, `firered_models`, `firered_worker`, `speaker_models`, `overlap_models`, and `gpu_release` checks.

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/integration/test_doctor.py tests/unit/test_cli.py -q`

Expected: PASS.

```bash
git add src/funasr_transcriber/doctor.py src/funasr_transcriber/cli.py tests/integration/test_doctor.py
git commit -m "feat: verify complete FireRed runtime in doctor"
```

## Task 12: Build the three isolated environments

**Files:**
- Modify: `pyproject.toml`
- Create: `requirements-asr.txt`
- Modify: `scripts/install_linux.sh`
- Modify: `tests/integration/test_install_script.py`

- [ ] **Step 1: Write RED installation tests**

Assert core dependencies no longer contain `funasr`, the script creates `.venv-asr` with Python 3.11, validates upstream HEAD, installs pinned upstream metadata, installs this project with `--no-deps` into worker/overlap envs, and never uses `sudo pip`, system site packages, or deletes existing outputs/configuration.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/integration/test_install_script.py -q`

Expected: FAIL because the current script has no ASR environment.

- [ ] **Step 3: Update package dependencies**

Remove FunASR-only ASR packages from the core dependency list only when speaker code no longer needs them; retain `modelscope` for the FSMN/CAMPPlus speaker route. Create:

```text
modelscope==1.36.3
huggingface-hub==1.4.1
```

The FireRed package itself supplies its exact Torch/Transformers/Numpy dependencies from its pinned `pyproject.toml`.

- [ ] **Step 4: Update install flow**

Create `.venv`, `.venv-asr`, and `.venv-overlap` with `python3.11`. Install core and overlap as before. Clone/verify pinned FireRed source under the configured model root, then run `.venv-asr/bin/python -m pip install -e "$MODEL_ROOT/fireredasr2s/source"`, install `requirements-asr.txt`, and install this project with `--no-deps -e`. Verify CUDA independently in all three environments.

- [ ] **Step 5: Run static tests and a clean install smoke**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_install_script.py -q
bash -n scripts/install_linux.sh
```

Expected: PASS and no shell syntax errors. Run the full installer only after selecting an unoccupied 24 GiB GPU and confirming model download disk space.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements-asr.txt scripts/install_linux.sh tests/integration/test_install_script.py
git commit -m "build: isolate FireRedASR2S environment"
```

## Task 13: Replace integration tests and documentation

**Files:**
- Modify: `tests/integration/test_asr_smoke.py`
- Modify: `tests/integration/test_pipeline_smoke.py`
- Modify: `README.md`
- Modify: `RELEASE_CHECKLIST.md`
- Modify: `docs/architecture/current-transcription-pipeline.md`

- [ ] **Step 1: Convert model-gated tests to FireRed**

The ASR smoke must create `FireRedWorkerClient`, transcribe `short_mandarin.wav`, assert full tokens plus all four module metadata, and close the worker. The pipeline smoke must use explicit primary/speaker phases and assert FireRed LID diagnostics. Keep `FUNASR_TRANSCRIBER_RUN_MODEL_TESTS=1` for backward-compatible automation unless all runner scripts are changed in the same commit.

- [ ] **Step 2: Update operator documentation**

Document `.venv-asr`, pinned source/model layout, ModelScope default and explicit Hugging Face fallback, 23/12 GiB checks, no FunASR fallback, at-most-two FireRed loads, and GPU contention remediation. Replace stale statements that `.venv` contains the ASR model.

- [ ] **Step 3: Run doc/static checks and commit**

Run:

```bash
.venv/bin/python -m pytest tests/integration -q
rg -n "FunAsrEngine|from funasr import AutoModel|Paraformer" src tests/integration README.md docs/architecture
```

Expected: integration tests PASS/SKIP as gated; `rg` finds no production-route references.

```bash
git add tests/integration README.md RELEASE_CHECKLIST.md docs/architecture/current-transcription-pipeline.md
git commit -m "docs: document FireRed migration and smoke tests"
```

## Task 14: Update and rebuild release validation

**Files:**
- Modify: `scripts/run_validation_linux.sh`
- Modify: `tools/build_release_validation_corpus.py`
- Modify: `tools/verify_release_validation_corpus.py`
- Modify: `tests/integration/test_validation_corpus_cli.py`
- Modify: `tests/benchmark/test_ramc_gate.py`
- Modify: `tests/benchmark/test_long_audio.py`

- [ ] **Step 1: Write RED release-completeness assertions**

Require every summary to contain FireRed module verification, LID coverage, timestamp integrity, GPU samples/releases, and deterministic hashes. `passed` and `complete` must be false if any required field is absent, even when subprocess exit codes are zero.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/integration/test_validation_corpus_cli.py tests/benchmark/test_ramc_gate.py tests/benchmark/test_long_audio.py -q`

Expected: FAIL on the new required FireRed evidence.

- [ ] **Step 3: Update builder and Linux runner**

Package the updated runner, verify `.venv-asr/bin/python` and all four model directories before benchmarks, collect doctor/diagnostics evidence, and retain the existing 20 RAMC, 30 synthetic overlap, 10 observed real overlap, 10 change-negative, 48 input matrix, and 30/60/120-minute twice-run gates. Do not weaken existing metric thresholds.

- [ ] **Step 4: Rebuild the ignored corpus and run structural verification**

Run with the actual RAMC path:

```bash
PYTHONPATH=src .venv/bin/python tools/build_release_validation_corpus.py --source-root /path/to/MDT2021S003 --output-dir artifacts/release_validation_corpus --overwrite
PYTHONPATH=src .venv/bin/python tools/verify_release_validation_corpus.py artifacts/release_validation_corpus
```

Expected: verifier reports all counts/checksums valid. Confirm `git status --short` does not list the generated corpus.

- [ ] **Step 5: Run the full Linux/NVIDIA gate**

On a GPU with at least 12 GiB free:

```bash
artifacts/release_validation_corpus/run_validation_linux.sh --project-root "$PWD" --config /etc/funasr-transcriber.toml
```

Expected: all design-section-12 thresholds pass, both two-hour hashes match, and every GPU stage reports valid release evidence.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_validation_linux.sh tools tests/benchmark tests/integration/test_validation_corpus_cli.py
git commit -m "test: enforce FireRed release validation gates"
```

## Task 15: Final migration verification

**Files:**
- Modify only if a verification failure requires a TDD fix.

- [ ] **Step 1: Run the complete non-model suite**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS with only explicitly environment-gated model/benchmark skips.

- [ ] **Step 2: Run static route and repository checks**

Run:

```bash
rg -n "from funasr import AutoModel|FunAsrEngine|build_asr_model" src scripts
git status --short
git log --oneline --decorate -15
```

Expected: no legacy production ASR matches; only intentional working-tree changes remain; migration commits are visible.

- [ ] **Step 3: Run the real FireRed smoke and doctor**

Run:

```bash
FUNASR_TRANSCRIBER_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest tests/integration/test_asr_smoke.py tests/integration/test_pipeline_smoke.py -q
.venv/bin/funasr-transcribe doctor --config /etc/funasr-transcriber.toml
```

Expected: PASS; doctor names all four FireRed modules, speaker/overlap verification, and released GPU memory.

- [ ] **Step 4: Review the diff and commit any final TDD correction**

Run: `git diff --check && git status --short`

Expected: no whitespace errors and no accidental models, virtual environments, logs, or validation corpus files staged.

## Task 16: Calibrate the overlap PyTorch runtime

**Files:**
- Modify: `requirements-overlap.txt`
- Modify: `scripts/install_linux.sh`
- Modify: `tests/integration/test_install_script.py`
- Modify: `README.md`

- [ ] **Step 1: Add failing dependency compatibility assertions**

Extend `test_overlap_dependencies_use_torch_2_compatible_pyannote` to require
`torch==2.8.0`, `torchaudio==2.8.0`, and `torchvision==0.23.0` in
`requirements-overlap.txt`. Add
`test_install_script_pins_overlap_torch_runtime` to require the same versions in
the overlap install command while retaining 2.11 for the ASR environment.

- [ ] **Step 2: Verify RED against the installed 2.11 pins**

Run:

```bash
.venv/bin/python -m pytest \
  tests/integration/test_install_script.py::test_overlap_dependencies_use_torch_2_compatible_pyannote \
  tests/integration/test_install_script.py::test_install_script_pins_overlap_torch_runtime -q
```

Expected: FAIL because both files still select `torch/torchaudio==2.11.0`.

- [ ] **Step 3: Pin only the overlap runtime to PyTorch 2.8**

Change the first two lines of `requirements-overlap.txt` and the overlap install
command in `scripts/install_linux.sh` to `torch==2.8.0` and
`torchaudio==2.8.0`, and pin `torchvision==0.23.0` in the overlap requirements.
Keep the ASR environment pins unchanged. Document the split in `README.md`.

- [ ] **Step 4: Verify GREEN and rebuild the isolated environment**

Run the focused tests from Step 2, then install the two CUDA 12.8 wheels and
reapply `requirements-overlap.txt` in `.venv-overlap`.

- [ ] **Step 5: Verify runtime imports and package consistency**

Run:

```bash
.venv-overlap/bin/python -c \
  "import torch, torchaudio, pyannote.audio, clearvoice; print(torch.__version__, torchaudio.__version__)"
.venv-overlap/bin/python -m pip check
```

Expected: imports succeed, versions begin with `2.8.0`, and `pip check` reports
no broken requirements.

- [ ] **Step 6: Continue the overlap model gate**

With `HF_TOKEN` present, run model download/verification and the overlap-enabled
doctor on physical GPU 0. If the token is absent, report that external gate
without treating overlap as verified.

## Task 17: Add reproducible ModelScope overlap assets

**Files:**
- Modify: `src/funasr_transcriber/model_registry.py`
- Modify: `src/funasr_transcriber/cli.py`
- Modify: `src/funasr_transcriber/overlap_worker.py`
- Modify: `src/funasr_transcriber/overlap.py`
- Modify: `tests/unit/test_model_registry.py`
- Modify: `tests/unit/test_overlap_detection.py`
- Modify: `tests/integration/test_overlap_worker.py`
- Modify: `README.md`

- [ ] **Step 1: Add RED tests for local overlap acquisition**

Require `acquire_overlap_assets` to download the three pyannote repositories and
MossFormer checkpoint into fixed directories, reject incomplete existing assets,
and return verified paths.

- [ ] **Step 2: Add RED tests for local pipeline resolution**

Create minimal local pipeline, segmentation, embedding, and ClearVoice fixtures.
Assert that the worker rewrites the pyannote pipeline parameters to absolute local
checkpoint paths without reading `HF_TOKEN`, and rejects a missing ClearVoice file.

- [ ] **Step 3: Add a RED GPU visibility test**

Patch `subprocess.run` in `JsonOverlapWorkerClient` and assert that the child
environment contains `CUDA_VISIBLE_DEVICES=0` while retaining unrelated parent
environment variables.

- [ ] **Step 4: Implement atomic ModelScope acquisition**

Add verified overlap path types and a four-repository acquisition loop to
`model_registry.py`. Call it before the overlap worker's `download` action in the
CLI; verification remains offline and fails on incomplete local files.

- [ ] **Step 5: Implement local pipeline and checkpoint validation**

When all local assets exist, generate a temporary pyannote config whose
segmentation and embedding entries are local checkpoint mappings. Preserve the
token-required Hugging Face behavior only when local assets are absent. Validate
the ClearVoice pointer and target checkpoint before reporting success.

- [ ] **Step 6: Enforce physical GPU 0 in the child process**

Pass a copy of the current environment with `CUDA_VISIBLE_DEVICES=0` to every
overlap worker subprocess. Do not query, signal, stop, or otherwise manage other
GPU processes.

- [ ] **Step 7: Run focused and full verification**

Run the focused registry/worker/client tests, load the real local models on GPU 0,
then run the complete test suite and overlap-enabled doctor. Record `HF_TOKEN` as
optional when the complete ModelScope assets are present.

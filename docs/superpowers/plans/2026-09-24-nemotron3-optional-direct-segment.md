# Optional Nemotron 3 Direct-Segment Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking. The user explicitly chose native execution: no subagents and no worktree.

**Goal:** Add an explicitly selected Nemotron 3 speaker backend that preserves overlapping speaker intervals and transcribes each original-mixture crop independently with the existing resident FireRed AED/Punc owners.

**Architecture:** Keep the existing 3D-Speaker, single-path AED, and overlap-enhancement paths unchanged. A pinned offline Nemotron subprocess returns global SpeakerSegment values; a pure crop planner creates per-segment 16 kHz WAVs and local AED piece plans; a separate multi-crop method assigns whole crops to one or two AED workers, then uses the existing Punc owner. The service converts crop-local times to global times and attributes directly to the originating speaker.

**Tech Stack:** Python 3.11, pytest, PyTorch CUDA on physical GPU0, pinned Transformers audio-frame classifier, soundfile/standard-library WAV I/O, existing FireRed JSON-lines workers.

**Spec:** docs/superpowers/specs/2026-09-24-nemotron3-optional-direct-segment-design.md

**Model API reference:** https://huggingface.co/nvidia/Nemotron-3-Diarization (offline Transformers example; local pinned revisions above take precedence over the page's latest default).

## Global Constraints

- Work only through ssh ai_server in /mnt/hdd/jiajun/project/standalone_transcriber on feature/server-weighted-aed-batching; no worktree, subagents, or unrelated project processes.
- Keep 3D-Speaker as the default. Nemotron is selected only by diarization.backend = "nemotron3"; the selected config is fingerprinted.
- All GPU execution is physical cuda:0. Never remap another GPU with CUDA_VISIBLE_DEVICES. Resource sampling may inspect only this project's registered PIDs.
- Pin nvidia/Nemotron-3-Diarization at a435e9867d79e789e90053f9b6d6834053af564a and Transformers at f324707307757d9c0b8dac1c4462eceff911fa2f; use the tested 0.5 activity threshold and offline/chunked inference. The known model.safetensors SHA-256 is c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6.
- Install Nemotron only through an explicit optional step into .venv-nemotron. Default install and default model download must not fetch it. Runtime must use a locally verified snapshot with offline flags.
- Nemotron supports at most eight speaker channels. Explicit speaker_count > 8 is invalid; do not claim to detect an unobserved ninth speaker.
- For Nemotron, retain each overlapping interval as an independent original-mixture AED crop. Do not use ClearVoice, pyannote, speaker rematching, midpoint attribution, overlap deduplication, or legacy memory-first planning.
- Exclude segments below 100 ms with a warning. For intervals above 10 s, try FireRedVAD within the crop; on unavailable/unsafe refinement use deterministic pieces no longer than 10 s. Never send a piece over the AED 60 s hard limit.
- Reject Nemotron plus legacy overlap at config load, CLI request creation, and service request execution. Preserve the existing 3D-Speaker overlap path.
- Do not read validation labels at runtime or use them to tune thresholds. Do not train AED cost models, change production defaults, or endorse production accuracy from a smoke run.
- Keep artifacts/release_validation_corpus ignored and all measurements under outputs/measurements. Use repository Git identity; no automatic push.
- Baseline full pytest presently has one unrelated README assertion failure: tests/integration/test_firered_documentation.py expects "23 GiB". The README task below adds an accurate existing GPU requirement, then re-runs that test; never describe the earlier full suite as passing.

## Review Focus

- A worker response with NaN, negative, reversed, or out-of-audio timestamps must fail before any crop is written (Task 3 test).
- A 100 ms boundary crop or 190 ms no-speech crop must not disappear silently or cause other crops to be dropped (Tasks 5 and 6 tests).
- Two crossing/nested speaker intervals must preserve their speaker IDs and duplicated mixture samples even if their token timestamps coincide (Tasks 5 and 7 tests).
- One AED lane failing while the other succeeds must fail the whole job, retain the original error, and still perform project-owned cleanup (Tasks 6 and 7 tests).
- A caller enabling legacy overlap against a Nemotron daemon must be rejected even if the daemon's static config disables overlap (Task 1 and Task 7 tests).

---

### Task 1: Configuration and request conflicts

**Files:**
- Modify: config/default.toml; src/funasr_transcriber/config.py; src/funasr_transcriber/cli.py
- Test: tests/unit/test_config.py; tests/unit/test_cli.py; tests/unit/test_service_protocol.py

**Interfaces:**
- Produce: DiarizationConfig.backend: str and NemotronConfig(worker_python: Path, timeout_seconds: int, activity_threshold: float), exposed as AppConfig.nemotron.
- Preserve: config_fingerprint(config) in service_protocol.py already hashes dataclasses.asdict(config); no special-case serializer.

- [ ] **Step 1: Add failing configuration tests.**

~~~python
def test_nemotron_selection_rejects_legacy_overlap():
    with pytest.raises(ValueError, match="Nemotron.*overlap"):
        load_config(cli_overrides={
            "diarization.backend": "nemotron3",
            "overlap.enabled": True,
        })

def test_nemotron_rejects_ninth_configured_speaker():
    with pytest.raises(ValueError, match="eight"):
        load_config(cli_overrides={
            "diarization.backend": "nemotron3",
            "diarization.speaker_count": 9,
        })

def test_nemotron_rejects_memory_first():
    with pytest.raises(ValueError, match="memory_first"):
        load_config(cli_overrides={
            "diarization.backend": "nemotron3",
            "asr.aed_scheduler": "memory_first",
        })
~~~

Also assert the unchanged default is 3d_speaker, an unknown backend fails, a Nemotron-selected config has a different fingerprint, and _service_request rejects --enable-overlap under Nemotron.

- [ ] **Step 2: Verify red.** Run .venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_cli.py; expect new tests to fail for missing backend/config conflict.
- [ ] **Step 3: Add minimal configuration and request checks.**

~~~python
@dataclass(frozen=True, slots=True)
class NemotronConfig:
    worker_python: Path
    timeout_seconds: int
    activity_threshold: float

# In DEFAULTS:
# Add "backend": "3d_speaker" to the existing diarization section.
"nemotron": {
    "worker_python": "/opt/funasr-transcriber/.venv-nemotron/bin/python",
    "timeout_seconds": 1800,
    "activity_threshold": 0.5,
}

# In validate():
if config.diarization.backend == "nemotron3":
    if config.overlap.enabled:
        raise ValueError("Nemotron and legacy overlap cannot run together")
    if config.asr.aed_scheduler == "memory_first":
        raise ValueError("Nemotron does not support memory_first")
    if config.diarization.speaker_count is not None and config.diarization.speaker_count > 8:
        raise ValueError("Nemotron supports at most eight speakers")
~~~

Validate backend enum, threshold exactly 0.5 for this pinned design, and positive timeout. Do not apply the 3D-specific min_speakers/max_speakers values to Nemotron: its default max_speakers=10 must not reject Nemotron auto mode. In cli._service_request, reject an effective overlap_enabled value of true for Nemotron before submission.

- [ ] **Step 4: Verify green and regression.** Run .venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_cli.py tests/unit/test_service_protocol.py; expect 0 failures.
- [ ] **Step 5: Commit focused changes.** Stage only these files and commit "feat: configure optional Nemotron backend".

### Task 2: Explicit installation and pinned local model assets

**Files:**
- Create: requirements-nemotron.txt; scripts/install_nemotron_linux.sh; src/funasr_transcriber/nemotron_assets.py
- Modify: .gitignore; src/funasr_transcriber/model_registry.py; src/funasr_transcriber/cli.py
- Test: tests/unit/test_nemotron_assets.py; tests/unit/test_firered_model_registry.py; tests/integration/test_install_script.py

**Interfaces:**
- Produce: acquire_nemotron_snapshot(model_dir: Path, download: Callable[..., object] | None = None) -> Path and verify_nemotron_snapshot(model_dir: Path) -> Path.
- Extend download_models(model_dir: Path, speaker_backend: str = "3d_speaker") and verify_models(model_dir: Path, speaker_backend: str = "3d_speaker"). Both keep FireRed assets; 3D-Speaker assets are acquired/verified only for 3d_speaker, Nemotron assets only for nemotron3. The CLI passes config.diarization.backend, while one-argument callers retain the old default.
- Asset location: model_dir/nemotron3/a435e9867d79e789e90053f9b6d6834053af564a.

- [ ] **Step 1: Add red asset tests.**

~~~python
def test_default_models_command_never_calls_nemotron(monkeypatch):
    from argparse import Namespace
    import funasr_transcriber.cli as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("default path fetched Nemotron")
    monkeypatch.setattr(cli, "download_models", lambda model_dir, speaker_backend="3d_speaker": None)
    monkeypatch.setattr(cli, "acquire_nemotron_snapshot", forbidden)
    assert cli._run_models("download", Namespace(config=None)) == 0

def test_tampered_nemotron_checkpoint_fails(tmp_path):
    target = tmp_path / "nemotron3" / NEMOTRON_REVISION
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "processor_config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ModelError, match="checksum"):
        verify_nemotron_snapshot(tmp_path)

def test_selected_verifier_does_not_require_3d(monkeypatch, tmp_path):
    import funasr_transcriber.model_registry as registry
    monkeypatch.setattr(registry, "verify_nemotron_snapshot", lambda root: tmp_path)
    result = registry.verify_models(tmp_path, speaker_backend="nemotron3")
    assert "3d_speaker_repo" not in result.missing
    assert "speaker_vad" not in result.missing
    assert "diarization_embedding" not in result.missing
~~~

Add a mock download test checking repository ID, exact revision, staging sibling, required config.json/processor_config.json/model.safetensors, and refusal to overwrite a partial pre-existing target. Test that the separate install_nemotron_linux.sh exists and pins the revision, while install_linux.sh remains unchanged and never creates .venv-nemotron; do not install dependencies in unit tests.

- [ ] **Step 2: Verify red.** Run .venv/bin/python -m pytest -q tests/unit/test_nemotron_assets.py tests/integration/test_install_script.py; expect missing module/API failures.
- [ ] **Step 3: Implement atomic acquisition and offline verification.**

~~~python
NEMOTRON_REPO = "nvidia/Nemotron-3-Diarization"
NEMOTRON_REVISION = "a435e9867d79e789e90053f9b6d6834053af564a"
NEMOTRON_MODEL_SHA256 = "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"

def verify_nemotron_snapshot(model_dir: Path) -> Path:
    target = Path(model_dir) / "nemotron3" / NEMOTRON_REVISION
    for name in ("config.json", "processor_config.json", "model.safetensors"):
        if not (target / name).is_file():
            raise ModelError(f"missing Nemotron asset: {name}")
    if sha256_file(target / "model.safetensors") != NEMOTRON_MODEL_SHA256:
        raise ModelError("Nemotron checkpoint checksum mismatch")
    return target
~~~

Use snapshot_download(repo_id=NEMOTRON_REPO, revision=NEMOTRON_REVISION, local_dir=temporary), verify then atomically rename; never delete or replace an existing target. In model_registry, branch speaker asset acquisition and verification on speaker_backend; always keep FireRed checks. In cli._run_models, pass config.diarization.backend to those functions. In scripts/install_nemotron_linux.sh, require the same Python 3.11 path policy as install_linux.sh, create only .venv-nemotron, install torch==2.8.0 from the configured CUDA index and pinned Transformers Git commit plus huggingface-hub==1.32.0/tokenizers==0.23.1, then editable-install this project without default dependencies. Do not alter the default install script's behavior. Add .venv-nemotron/ to .gitignore.

- [ ] **Step 4: Verify green.** Run .venv/bin/python -m pytest -q tests/unit/test_nemotron_assets.py tests/unit/test_firered_model_registry.py tests/integration/test_install_script.py; expect 0 failures.
- [ ] **Step 5: Commit focused changes.** Stage only Task 2 files and commit "feat: provision pinned optional Nemotron assets".

### Task 3: Offline persistent Nemotron worker and validated segments

**Files:**
- Create: src/funasr_transcriber/nemotron_worker.py; src/funasr_transcriber/nemotron_client.py
- Test: tests/unit/test_nemotron_worker.py; tests/unit/test_nemotron_client.py

**Interfaces:**
- Produce: NemotronWorkerClient(worker_python: Path, model_dir: Path, timeout_seconds: int, stderr_log_path: Path) with start() -> self, diarize(path: Path) -> tuple[SpeakerSegment, ...], diarize_with_resource_metrics(path: Path) -> tuple[tuple[SpeakerSegment, ...], dict[str, object]], resource_inventory() -> dict[str, object], close() -> None, pid: int | None.
- The worker's newline-JSON request accepts only an absolute project-job WAV path and action in {"diarize", "status", "close"}; response carries id, segments, runtime identity, and inference metrics.

- [ ] **Step 1: Add red contract tests.**

~~~python
@pytest.mark.parametrize("raw", [
    {"Speaker": 0, "Start": float("nan"), "End": 1.0},
    {"Speaker": 0, "Start": -0.1, "End": 1.0},
    {"Speaker": 0, "Start": 1.0, "End": 0.5},
    {"Speaker": 0, "Start": 0.0, "End": 99.0},
])
def test_invalid_segment_fails_before_crop(raw):
    with pytest.raises(TranscriptionError):
        normalize_nemotron_segments([raw], duration_ms=8000)
~~~

Test two simultaneous speakers are preserved, zero-based model IDs become stable one-based IDs, >8 channels or duplicate/unsorted illegal response shapes fail, subprocess timeout retains the original error, and close acts only on its own Popen child.

- [ ] **Step 2: Verify red.** Run .venv/bin/python -m pytest -q tests/unit/test_nemotron_worker.py tests/unit/test_nemotron_client.py; expect missing API failures.
- [ ] **Step 3: Implement the pinned local runtime and protocol.**

~~~python
def normalize_nemotron_segments(
    raw: list[dict[str, object]], duration_ms: int,
) -> tuple[SpeakerSegment, ...]:
    if duration_ms <= 0:
        raise TranscriptionError("invalid normalized audio duration")
    segments = []
    for item in raw:
        start, end = float(item["Start"]), float(item["End"])
        speaker = int(item["Speaker"])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise TranscriptionError("non-finite Nemotron timestamp")
        start_ms, end_ms = round(start * 1000), round(end * 1000)
        if speaker not in range(8) or start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms:
            raise TranscriptionError("invalid Nemotron speaker segment")
        segments.append(SpeakerSegment(speaker + 1, start_ms, end_ms))
    return tuple(sorted(segments, key=lambda s: (s.start_ms, s.speaker_id, s.end_ms)))
~~~

At worker startup verify_nemotron_snapshot, set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1, and load AutoProcessor/AutoModelForAudioFrameClassification from the verified local directory with local_files_only=True and explicit device cuda:0 (never device_map="auto"). Read 16 kHz mono PCM into audio, call processor(audio, sampling_rate=16000).to(model.device, dtype=model.dtype), model(**inputs) under torch.inference_mode(), then processor.extract_speaker_dict(logits, inputs.attention_mask)[0]. Enforce tested 0.5 threshold in the pinned processor configuration. Reuse one loaded model across requests; record dtype, revision, checkpoint hash, PID, load/inference time. Use only this client's Popen PID for close/cleanup.

- [ ] **Step 4: Verify green.** Run .venv/bin/python -m pytest -q tests/unit/test_nemotron_worker.py tests/unit/test_nemotron_client.py; expect 0 failures.
- [ ] **Step 5: Commit focused changes.** Stage Task 3 files and commit "feat: add offline Nemotron worker".

### Task 4: Selected speaker ownership, doctor, and GPU admission

**Files:**
- Modify: src/funasr_transcriber/service.py; src/funasr_transcriber/doctor.py; src/funasr_transcriber/cli.py
- Test: tests/unit/test_service.py; tests/integration/test_doctor.py; tests/unit/test_gpu.py

**Interfaces:**
- Consume: NemotronWorkerClient and verify_nemotron_snapshot.
- Produce: build_hot_model_host selects the speaker factory by backend; ServiceJobRunner._owned_pid_getters returns the selected speaker worker PID when present.

- [ ] **Step 1: Add red selection/ownership tests.**

~~~python
def test_selected_nemotron_pid_is_profiled_not_service_pid():
    runner = ServiceJobRunner.__new__(ServiceJobRunner)
    runner.hot_host = type("Host", (), {
        "speaker_owner": type("Speaker", (), {"pid": 451})(),
        "asr_owner": None,
    })()
    runner._active_overlap_pid = None
    assert runner._owned_pid_getters()["speaker"]() == 451
~~~

Also test default speaker stays in-process, selected Nemotron worker is the only speaker loaded, missing snapshot/venv fails without 3D fallback, and admission failure calls neither owner factory. The doctor fake must call selected backend's smoke diarize on the existing hello_zh.wav and check only its own worker releases resources.

- [ ] **Step 2: Verify red.** Run .venv/bin/python -m pytest -q tests/unit/test_service.py tests/integration/test_doctor.py; expect new selection/ownership tests to fail.
- [ ] **Step 3: Route ownership by config.**

~~~python
def speaker_factory(config: AppConfig, paths: ServicePaths):
    if config.diarization.backend == "3d_speaker":
        return SpeakerDiarizer.from_config(config)
    return NemotronWorkerClient(
        worker_python=config.nemotron.worker_python,
        model_dir=config.paths.model_dir,
        timeout_seconds=config.nemotron.timeout_seconds,
        stderr_log_path=paths.directory / "nemotron-worker.log",
    )
~~~

Keep HotModelHost.start's admission before both owners. For Nemotron, require a GPU0 device snapshot and enough free memory for the selected resident combination before starting; call require_gpu_capacity("cuda:0", minimum_total_mib=config.service.minimum_total_mib, minimum_free_mib=config.service.minimum_free_mib), with current defaults 23552 MiB total and 13312 MiB free, report a clear admission error, and never manage another project's processes. In doctor, call verify_models(config.paths.model_dir, speaker_backend=config.diarization.backend), select the matching speaker owner, run the small inference, verify the snapshot and worker environment, then close/release only the project's speaker owner.

- [ ] **Step 4: Verify green.** Run .venv/bin/python -m pytest -q tests/unit/test_service.py tests/integration/test_doctor.py tests/unit/test_gpu.py; expect 0 failures.
- [ ] **Step 5: Commit focused changes.** Stage Task 4 files and commit "feat: host selected Nemotron speaker owner".

### Task 5: Deterministic original-mixture crop and long-interval planning

**Files:**
- Create: src/funasr_transcriber/speaker_crops.py
- Test: tests/unit/test_speaker_crops.py

**Interfaces:**
- Produce: SpeakerCrop(index: int, speaker_id: int, start_ms: int, end_ms: int, path: Path, sha256: str, pieces_ms: tuple[tuple[int, int], ...]); CropPlan(crops: tuple[SpeakerCrop, ...], duration_ms: int, short_count: int, subpiece_count: int, overlap_duration_ms: int, warnings: tuple[str, ...]).
- Produce: build_crop_plan(normalized: Path, segments: tuple[SpeakerSegment, ...], output_dir: Path, refine_vad: Callable[[Path, list[list[int]]], list[list[int]]] | None) -> CropPlan.
- All piece coordinates are crop-local milliseconds; crop start/end remain global milliseconds.

- [ ] **Step 1: Add red exact-sample tests.**

~~~python
def test_crossing_segments_copy_same_original_overlap(tmp_path):
    import numpy as np
    import soundfile as sf

    source = tmp_path / "normalized.wav"
    samples = np.arange(8000 * 16, dtype=np.int32).astype(np.int16)
    sf.write(source, samples, 16000, subtype="PCM_16")
    plan = build_crop_plan(source, (
        SpeakerSegment(1, 0, 6000),
        SpeakerSegment(2, 5000, 8000),
    ), tmp_path / "crops", refine_vad=None)
    assert len(plan.crops) == 2
    first, _ = sf.read(plan.crops[0].path, dtype="int16")
    second, _ = sf.read(plan.crops[1].path, dtype="int16")
    assert np.array_equal(first[5000 * 16:6000 * 16], second[:1000 * 16])
    assert (plan.crops[0].speaker_id, plan.crops[1].speaker_id) == (1, 2)
~~~

Test nested/crossing intervals, exact 10/100/190 ms boundaries, a >30 s segment, a >10 s valid VAD split and malformed/empty VAD fallback, no same-speaker merge, target-exists refusal, and SHA-256 of written WAV bytes.

- [ ] **Step 2: Verify red.** Run .venv/bin/python -m pytest -q tests/unit/test_speaker_crops.py; expect missing module/API.
- [ ] **Step 3: Implement create-only PCM crop planning.**

~~~python
def fixed_pieces(duration_ms: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, min(start + 10_000, duration_ms))
        for start in range(0, duration_ms, 10_000)
    )

# For 16 kHz PCM16 mono:
start_sample = segment.start_ms * 16
end_sample = segment.end_ms * 16
crop_samples = source_samples[start_sample:end_sample]
~~~

Read the normalized WAV once, verify its format and duration, write each crop with an exclusive-create path in the job's own intermediate directory, compute checksum after closing, and never modify the normalized source. Exclude <100 ms with warning. For >10 s call refine_vad(crop.path, [[0, duration_ms]]), accept only finite, in-bounds, non-overlapping pieces <=10 s; otherwise use fixed_pieces and warn. Compute overlap duration as union of time with two or more distinct speakers, not sum of pairwise intersections.

- [ ] **Step 4: Verify green.** Run .venv/bin/python -m pytest -q tests/unit/test_speaker_crops.py; expect 0 failures.
- [ ] **Step 5: Commit focused changes.** Stage Task 5 files and commit "feat: plan original-mixture speaker crops".

### Task 6: Separate one/two-lane multi-crop AED/Punc execution

**Files:**
- Modify: src/funasr_transcriber/firered_parallel.py
- Test: tests/unit/test_firered_parallel.py; tests/unit/test_firered_client.py

**Interfaces:**
- Consume: CropPlan and SpeakerCrop from speaker_crops.py.
- Produce: CropTranscript(crop: SpeakerCrop, result: AsrResult) and WeightedAedPuncWorkerClient.transcribe_crops(plan: CropPlan) -> tuple[CropTranscript, ...].
- Preserve existing transcribe(audio_path, vad_segments_ms) exactly for default backend. Never represent different crop paths as one aed_plan request.

- [ ] **Step 1: Add red multi-path tests.**

~~~python
def test_multicrop_uses_one_audio_path_per_aed_request(tmp_path):
    class PathWorker(_FakeWorker):
        def __init__(self):
            super().__init__()
            self.paths = []

        def run_aed_plan(self, path, batch_plan, vad):
            self.paths.append(path)
            return super().run_aed_plan(path, batch_plan, vad)

    aed = PathWorker()
    owner = WeightedAedPuncWorkerClient(aed, _FakeWorker())
    crops = (
        SpeakerCrop(0, 1, 0, 6000, tmp_path / "s1.wav", "a" * 64, ((0, 6000),)),
        SpeakerCrop(1, 2, 5000, 8000, tmp_path / "s2.wav", "b" * 64, ((0, 3000),)),
    )
    plan = CropPlan(crops, 8000, 0, 2, 1000, ())
    results = owner.transcribe_crops(plan)
    assert aed.paths == [crop.path for crop in crops]
    assert [item.crop.speaker_id for item in results] == [1, 2]
~~~

Test one and two AED workers, deterministic weighted assignment, each plan frozen before its first request, 190 ms no-speech crop reported while another succeeds, all-no-speech explicit failure, and a lane exception propagating after the other lane exits without losing the first error. Test that Punc is called only through the dedicated owner and sees the same crop-local audio path.

- [ ] **Step 2: Verify red.** Run .venv/bin/python -m pytest -q tests/unit/test_firered_parallel.py; expect missing transcribe_crops.
- [ ] **Step 3: Implement per-crop immutable plans and lanes.**

~~~python
def _plan_one_crop(self, crop: SpeakerCrop) -> list[list[dict[str, int]]]:
    batches = plan_aed_batches(
        [(start / 1000, end / 1000) for start, end in crop.pieces_ms],
        weights=self.weights,
    )
    return serialize_batch_plan(tuple(
        BatchDecision(
            items=tuple(batch),
            estimate=self.cost_model.estimate(batch),
            decision="weighted_duration",
        )
        for batch in batches
    ))
~~~

Construct every crop plan before submitting any AED request. Rank whole crops by sum of their predicted batch seconds, partition deterministically between the configured one/two AED workers, and run each lane serially with ThreadPoolExecutor. For each crop call run_aed_plan(crop.path, its_plan, local_pieces), then the single Punc owner on that crop's AED results and local duration. Serialize Punc calls to protect its worker protocol; retain no-speech counts and separate VAD/planning/AED/Punc/per-lane timing plus per-worker peaks in last_response. Do not route through _transcribe_streaming's single audio_path argument.

- [ ] **Step 4: Verify green.** Run .venv/bin/python -m pytest -q tests/unit/test_firered_parallel.py tests/unit/test_firered_client.py; expect 0 failures.
- [ ] **Step 5: Commit focused changes.** Stage Task 6 files and commit "feat: transcribe independent speaker crops".

### Task 7: Service branch, direct attribution, overlapping output, and diagnostics

**Files:**
- Modify: src/funasr_transcriber/service.py; src/funasr_transcriber/pipeline.py; src/funasr_transcriber/turns.py; src/funasr_transcriber/diagnostic_traces.py
- Create: src/funasr_transcriber/crop_attribution.py
- Test: tests/unit/test_service.py; tests/unit/test_pipeline.py; tests/unit/test_turns.py; tests/unit/test_exporters.py; tests/unit/test_crop_attribution.py

**Interfaces:**
- Consume: CropPlan, CropTranscript, and selected speaker_owner.diarize.
- Produce: attribute_crop_results(results: tuple[CropTranscript, ...], segments: tuple[SpeakerSegment, ...], duration_ms: int) -> tuple[AsrResult, tuple[AttributedToken, ...]].
- Nemotron service branch bypasses alignment.attribute_by_segment and _run_overlap_enhancement. Default branch retains both existing behaviors.

- [ ] **Step 1: Add red end-to-end fake-owner tests.**

~~~python
def test_overlapping_crop_results_keep_both_speakers(tmp_path):
    segments = (SpeakerSegment(1, 0, 6000), SpeakerSegment(2, 5000, 8000))
    first = SpeakerCrop(0, 1, 0, 6000, tmp_path / "s1.wav", "a" * 64, ((0, 6000),))
    second = SpeakerCrop(1, 2, 5000, 8000, tmp_path / "s2.wav", "b" * 64, ((0, 3000),))
    results = (
        CropTranscript(first, AsrResult("甲", (TimedToken("甲", 5500, 5800),), duration_ms=6000)),
        CropTranscript(second, AsrResult("乙", (TimedToken("乙", 500, 800),), duration_ms=3000)),
    )
    asr, attributed = attribute_crop_results(results, segments, 8000)
    assert [token.speaker_id for token in attributed] == [1, 2]
    assert all(token.overlap for token in attributed)
    assert [token.start_ms for token in attributed] == [5500, 5500]
    assert asr.text == "甲乙"
~~~

Test equal-start and nested intervals, crop-local times shifted to global and bounded to parent crop, deterministic JSON/TXT turn ordering, no nearest/midpoint reassignment, a caller's overlap_enabled=True rejection at ServiceJobRunner.run even with static overlap disabled, and cleanup after an AED lane error. Assert default 3D-Speaker service tests remain unchanged.

- [ ] **Step 2: Verify red.** Run .venv/bin/python -m pytest -q tests/unit/test_crop_attribution.py tests/unit/test_service.py tests/unit/test_turns.py; expect missing path/API failures.
- [ ] **Step 3: Wire the distinct service path.**

~~~python
if self.config.diarization.backend == "nemotron3":
    if request.overlap_enabled:
        raise TranscriptionError("Nemotron and legacy overlap cannot run together")
    segments, speaker_metrics = _run_speaker(speaker_owner, normalized, profile_enabled)
    crop_plan = build_crop_plan(
        normalized, segments, job_dir / "speaker-crops",
        asr_owner.worker0.refine_vad,
    )
    crop_results = asr_owner.transcribe_crops(crop_plan)
    asr_result, attributed = attribute_crop_results(
        crop_results, segments, crop_plan.duration_ms,
    )
else:
    branches = run_normal_branches(normalized, asr_owner, speaker_owner, profile_enabled)
    attributed = attribute_by_segment(branches.asr_result.tokens, branches.speaker_segments)
~~~

Place actual branch logic in a small helper rather than duplicating ServiceJobRunner.run's export/cleanup logic. Validate crop result timestamps before global shift; mark overlap only when a token overlaps another speaker's original interval, preserve duplicate words across speakers, and sort by (start_ms, speaker_id, crop.index, end_ms). Ensure AsrResult.text ordering matches build_turns so pipeline.finish's integrity check still proves the output. Populate diagnostics with backend/revision/threshold/checkpoint hash, crop/subpiece/short/no-speech/overlap counts, own-PID resource peak, per-owner times, warnings, and deep crop-to-lane trace; keep the full plan out of basic diagnostics. Existing transcript schema remains 1.0 and may contain crossing turns.

- [ ] **Step 4: Verify green and default regression.** Run .venv/bin/python -m pytest -q tests/unit/test_crop_attribution.py tests/unit/test_service.py tests/unit/test_pipeline.py tests/unit/test_turns.py tests/unit/test_exporters.py; expect 0 failures.
- [ ] **Step 5: Commit focused changes.** Stage only Task 7 files and commit "feat: route Nemotron crops through transcription service".

### Task 8: Operator docs, opt-in GPU0 acceptance, and full regression

**Files:**
- Modify: README.md; docs/architecture/current-transcription-pipeline.md
- Create: tests/integration/test_nemotron_gpu_smoke.py; tools/benchmark_nemotron_direct_crop.py
- Test: tests/integration/test_firered_documentation.py; tests/integration/test_nemotron_gpu_smoke.py

**Interfaces:**
- Benchmark uses the committed optional config and existing ignored inputs at artifacts/release_validation_corpus/overlap/audio/real-overlap-0001.wav and artifacts/concat_eval/CTS-298-431-1368-concat.wav. Source speaker text comes from artifacts/release_validation_corpus/overlap/manifest.json. It writes a new create-only directory under outputs/measurements and never modifies those inputs.
- No default backend switch or automatic download.

- [ ] **Step 1: Add red documentation and opt-in smoke tests.**

~~~python
def test_readme_documents_optional_nemotron():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in ("install_nemotron_linux.sh", "diarization.backend",
                     "nemotron3", "models download", "cuda:0",
                     "overlap", "eight speakers", "23 GiB"):
        assert required in text
~~~

Mark GPU test opt-in with an environment flag; without it, pytest must skip before model load. Its assertions require two retained overlapping speaker turns, separate AED crop paths, physical GPU0, finite per-stage times, recorded actual model dtypes, and no other-project PID introspection. The benchmark must compare to source speaker text as observation only, since real-overlap fine timing reference is unavailable.

- [ ] **Step 2: Verify red.** Run .venv/bin/python -m pytest -q tests/integration/test_firered_documentation.py tests/integration/test_nemotron_gpu_smoke.py; expect the new docs test to fail and GPU test to skip.
- [ ] **Step 3: Document and implement the explicit benchmark.**

~~~python
# Opt-in guard in the GPU test:
if os.environ.get("RUN_NEMOTRON_GPU_TEST") != "1":
    pytest.skip("explicit GPU0 acceptance only")
assert config.diarization.backend == "nemotron3"
assert config.general.device == "cuda:0"
~~~

README must give explicit optional install, pinned Hugging Face model acquisition via models download under Nemotron config, models verify, doctor, request command, overlap exclusion, eight-speaker limit, 100 ms drop, >10 s VAD fallback, output overlap semantics, and switching back to 3d_speaker. Add the already-existing 23 GiB minimum GPU requirement accurately, resolving the baseline README assertion. Benchmark records GPU0 admission/end device snapshots, project-owned peak only, load/prewarm/end-to-end/AED/Punc/cache-cleanup timing, failure/OOM status, actual precision, model/runtime hashes, and qualitative transcription comparison. Use a unique new output directory; do not overwrite prior spike reports.

- [ ] **Step 4: Verify static tests, then separately opt-in to GPU0.** Run .venv/bin/python -m pytest -q tests/integration/test_firered_documentation.py tests/integration/test_nemotron_gpu_smoke.py (GPU skipped); then .venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_firered_client.py; then .venv/bin/python -m pytest -q; then git diff --check. Before the opt-in GPU run, use only physical GPU0 device admission and this project's registered PIDs; if capacity is insufficient, record pending and stop without retries or managing other processes. When admitted, run RUN_NEMOTRON_GPU_TEST=1 .venv/bin/python -m pytest -q tests/integration/test_nemotron_gpu_smoke.py and the benchmark once. Any failure preserves its output and exact error; do not imply production acceptance.
- [ ] **Step 5: Commit focused docs/test/benchmark changes only after their applicable checks pass.** Stage only Task 8 files and commit "docs: explain and validate optional Nemotron backend". Report all test results, measurement path, and outstanding accuracy limitations. Do not push unless separately asked.

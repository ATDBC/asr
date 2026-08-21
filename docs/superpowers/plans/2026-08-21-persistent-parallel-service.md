# Persistent Parallel Transcription Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide an auto-started local FIFO transcription service that keeps FireRed and 3D-Speaker resident, runs normal ASR and diarization concurrently, preserves optional overlap through a safe unload/reload path, and writes opt-in resource profiles to diagnostics.

**Architecture:** A Unix-domain-socket daemon in the main environment owns 3D-Speaker and one isolated FireRed subprocess. The CLI becomes a local client: it starts the daemon only after acquiring a service-specific lock, submits one JSON-lines request, and waits for its terminal response. The daemon serializes jobs but runs the two normal model branches with a two-thread executor; an overlap job snapshots speaker centroids, releases both hot owners before overlap processing, and rehydrates the hot owners afterwards.

**Tech Stack:** Python 3.11 stdlib Unix sockets, `ThreadPoolExecutor`, subprocess JSON Lines, PyTorch CUDA, psutil, pytest, FireRedASR2S, 3D-Speaker, pyannote/ClearVoice.

**User constraints:** Work directly in the current repository (no worktree and no subagents); bind only physical GPU 0; do not signal, stop, or otherwise affect other projects' processes.

---

## File structure

- Create `src/funasr_transcriber/resource_profile.py`: pure model inventory, GPU-process parser, opt-in sampler, and JSON-safe profile payload helpers.
- Create `src/funasr_transcriber/service_protocol.py`: JSON-lines message validation, service paths, state-file fingerprinting, and client/server socket helpers.
- Create `src/funasr_transcriber/service.py`: daemon owner lifecycle, FIFO queue, hot-model host, normal and overlap job execution, and daemon module entry point.
- Modify `src/funasr_transcriber/config.py` and `config/default.toml`: default overlap-off setting and validated service/profiling/budget configuration.
- Modify `src/funasr_transcriber/diagnostics.py`: store and export `resource_profile` without exposing other projects' process details.
- Modify `src/funasr_transcriber/gpu.py`: parse read-only per-process memory for registered project PIDs and enforce the calibrated concurrent budget.
- Modify `src/funasr_transcriber/firered_worker.py` and `src/funasr_transcriber/firered_client.py`: add a non-inference `status` handshake and FireRed module inventory.
- Modify `src/funasr_transcriber/speaker_diarization.py`: produce CAMPPlus/FSMN inventory and expose per-call PyTorch peak snapshots without changing diarization results.
- Modify `src/funasr_transcriber/overlap.py`: split detect/separate from speaker-track matching so pyannote never co-resides with a 3D-Speaker owner.
- Modify `src/funasr_transcriber/cli.py`: add `service status|stop`, client auto-start, `--enable-overlap`, `--profile-resources`, and service submission routing.
- Modify `docs/architecture/current-transcription-pipeline.md`, `README.md`, `RELEASE_CHECKLIST.md`, and relevant release scripts: document service lifecycle, default overlap-off behavior, profiling, and service shutdown verification.
- Create `tests/unit/test_resource_profile.py`, `tests/unit/test_service_protocol.py`, and `tests/unit/test_service.py`; modify existing config, diagnostics, GPU, FireRed client/worker, CLI lifecycle, overlap, doctor, and integration smoke tests.

### Task 1: Define configuration and diagnostics contracts

**Files:**
- Modify: `src/funasr_transcriber/config.py`
- Modify: `config/default.toml`
- Modify: `src/funasr_transcriber/diagnostics.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_diagnostics.py`

- [ ] **Step 1: Write failing configuration and diagnostics tests**

```python
def test_default_service_configuration_disables_overlap_and_profiles():
    config = load_config()

    assert config.overlap.enabled is False
    assert config.service.queue_limit == 16
    assert config.service.startup_timeout_seconds == 180
    assert config.service.profile_sampling_interval_ms == 200
    assert config.service.minimum_total_mib == 23 * 1024
    assert config.service.minimum_free_mib == 13 * 1024


def test_config_rejects_invalid_service_limits(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[service]\\nqueue_limit = 0\\n", encoding="utf-8")

    with pytest.raises(ValueError, match="service queue_limit"):
        load_config(path)
```

```python
def test_export_diagnostics_marks_disabled_resource_profile(tmp_path: Path):
    diagnostics = Diagnostics(resource_profile={"enabled": False})

    export_diagnostics(diagnostics, tmp_path, overwrite=False)

    payload = json.loads((tmp_path / "diagnostics.json").read_text())
    assert payload["resource_profile"] == {"enabled": False}


def test_export_diagnostics_sanitizes_resource_profile(tmp_path: Path):
    diagnostics = Diagnostics(resource_profile={
        "enabled": True,
        "service": {"pid": 123, "HF_TOKEN": "secret"},
    })

    export_diagnostics(diagnostics, tmp_path, overwrite=False)

    payload = json.loads((tmp_path / "diagnostics.json").read_text())
    assert payload["resource_profile"]["service"]["HF_TOKEN"] == "<redacted>"
```

- [ ] **Step 2: Run the new tests and verify the expected RED failures**

Run:

```bash
pytest tests/unit/test_config.py::test_default_service_configuration_disables_overlap_and_profiles tests/unit/test_config.py::test_config_rejects_invalid_service_limits tests/unit/test_diagnostics.py::test_export_diagnostics_marks_disabled_resource_profile tests/unit/test_diagnostics.py::test_export_diagnostics_sanitizes_resource_profile -v
```

Expected: failures because `AppConfig.service` and `Diagnostics.resource_profile` do not exist and overlap is currently enabled by default.

- [ ] **Step 3: Add the validated contracts**

Add the following default section and immutable configuration type, then pass it through `_build` and `validate`:

```python
"service": {
    "queue_limit": 16,
    "startup_timeout_seconds": 180,
    "profile_sampling_interval_ms": 200,
    "minimum_total_mib": 23 * 1024,
    "minimum_free_mib": 13 * 1024,
},


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    queue_limit: int
    startup_timeout_seconds: int
    profile_sampling_interval_ms: int
    minimum_total_mib: int
    minimum_free_mib: int
```

Set `DEFAULTS["overlap"]["enabled"]` and `config/default.toml` to `false`. Require a positive queue limit, positive startup timeout, a sampling interval in `50..5000` ms, `minimum_total_mib >= 23 * 1024`, and a positive free-memory budget not greater than the total. Add `service: ServiceConfig` to `AppConfig`.

Extend `Diagnostics` and `export_diagnostics` as follows:

```python
resource_profile: dict[str, Any] = field(
    default_factory=lambda: {"enabled": False}
)

# export payload
"resource_profile": sanitize_mapping(diagnostics.resource_profile),
```

- [ ] **Step 4: Re-run the focused tests and verify GREEN**

Run the command from Step 2.

Expected: all four tests pass.

- [ ] **Step 5: Commit the contract change**

```bash
git add src/funasr_transcriber/config.py config/default.toml src/funasr_transcriber/diagnostics.py tests/unit/test_config.py tests/unit/test_diagnostics.py
git commit -m "feat: add service and profiling configuration"
```

### Task 2: Build safe, testable resource-profile primitives

**Files:**
- Create: `src/funasr_transcriber/resource_profile.py`
- Modify: `src/funasr_transcriber/gpu.py`
- Test: `tests/unit/test_resource_profile.py`
- Test: `tests/unit/test_gpu.py`

- [ ] **Step 1: Write failing tests for PID filtering, model inventory, and concurrent admission**

```python
def test_parse_process_memory_filters_to_registered_project_pids():
    rows = "\\n".join(("100, 900", "200, 1234", "300, 456"))

    assert parse_process_memory_mib(rows, {100, 300}) == {100: 900, 300: 456}


def test_module_inventory_counts_parameters_buffers_and_dtypes():
    module = torch.nn.BatchNorm1d(2)

    inventory = module_inventory("batchnorm", module)

    assert inventory["parameter_count"] == 4
    assert inventory["parameter_bytes"] == 16
    assert inventory["buffer_bytes"] == 24
    assert inventory["parameters"]["cpu:float32"]["tensors"] == 2


def test_sampler_only_emits_registered_project_processes(monkeypatch):
    sampler = ResourceSampler("cuda:0", interval_ms=200, pid_getters={
        "service": lambda: 100,
        "firered": lambda: 300,
    }, query=lambda: ({"total_mib": 24576, "free_mib": 10000}, {100: 1, 200: 2, 300: 3}))

    sampler.sample_once()

    assert sampler.snapshot()["process_peak_mib"] == {"service": 1, "firered": 3}
```

```python
def test_concurrent_capacity_requires_configured_total_budget():
    with pytest.raises(EnvironmentError, match="free memory"):
        require_concurrent_gpu_capacity(
            "cuda:0",
            runner=_runner("0, RTX 3090, 24576, 13311, 580.142"),
            minimum_total_mib=23 * 1024,
            required_free_mib=13 * 1024,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/unit/test_resource_profile.py tests/unit/test_gpu.py::test_concurrent_capacity_requires_configured_total_budget -v
```

Expected: import errors for the new profile module and a missing test helper/API mismatch.

- [ ] **Step 3: Implement pure inventory and read-only sampling**

Implement these public functions in `resource_profile.py`:

```python
def parse_process_memory_mib(output: str, allowed_pids: set[int]) -> dict[int, int]: ...

def module_inventory(name: str, module: Any) -> dict[str, object]: ...

class ResourceSampler:
    def __init__(self, device: str, interval_ms: int, pid_getters, query): ...
    def start(self) -> None: ...
    def sample_once(self) -> None: ...
    def stop(self) -> dict[str, object]: ...
    def snapshot(self) -> dict[str, object]: ...
```

`module_inventory` must only inspect `parameters()` and `buffers()`, group tensors by
`"{device.type}:{dtype}"`, and serialize counts/bytes as integers. `ResourceSampler` must
use a daemon thread only when `enabled=True`; no sampler is created for disabled profiles.
Its `query` must call read-only `nvidia-smi` and `/proc/<pid>/status`/psutil only for PIDs
obtained from its own owner getters. Do not retain, serialize, or expose unregistered rows.

Add `query_gpu_process_memory(device, allowed_pids, runner=...)` and
`require_concurrent_gpu_capacity(device, minimum_total_mib, required_free_mib, runner=...)`
helpers in `gpu.py`, and reuse `parse_process_memory_mib`; retain the existing `query_gpu` and
`require_gpu_capacity` behavior and tests.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2.

Expected: all profile parser, inventory, sampler, and admission tests pass without CUDA.

- [ ] **Step 5: Commit the resource primitives**

```bash
git add src/funasr_transcriber/resource_profile.py src/funasr_transcriber/gpu.py tests/unit/test_resource_profile.py tests/unit/test_gpu.py
git commit -m "feat: add opt-in resource profile primitives"
```

### Task 3: Add FireRed and Speaker inventory/status APIs

**Files:**
- Modify: `src/funasr_transcriber/firered_worker.py`
- Modify: `src/funasr_transcriber/firered_client.py`
- Modify: `src/funasr_transcriber/speaker_diarization.py`
- Test: `tests/unit/test_firered_worker.py`
- Test: `tests/unit/test_firered_client.py`
- Test: `tests/unit/test_resource_profile.py`

- [ ] **Step 1: Write failing worker/client and Speaker inventory tests**

```python
def test_worker_status_returns_loaded_inventory_without_audio():
    system = FakeSystem({})
    target = StringIO()

    serve(
        StringIO(_request("status", "status")), target,
        lambda: system, inventory_provider=lambda _: {"models": ["vad"]},
    )

    assert json.loads(target.getvalue()) == {
        "id": "status", "ok": True,
        "status": {"models": ["vad"]},
        "metrics": {"load_seconds": pytest.approx(0.0), "peak_gpu_bytes": None},
    }
    assert system.calls == []
```

```python
def test_client_status_reuses_existing_worker(tmp_path: Path):
    with _client(tmp_path, mode="status") as client:
        first_pid = client.pid
        status = client.status()

    assert client.pid is None
    assert status["models"] == ["vad"]
    assert first_pid is not None


def test_speaker_inventory_reports_embedding_and_vad_without_gpu(monkeypatch):
    diarizer = object.__new__(SpeakerDiarizer)
    diarizer.embedding_model = torch.nn.Linear(2, 1)
    diarizer.vad_model = object()

    inventory = diarizer.resource_inventory()

    assert inventory["embedding"]["parameter_count"] == 3
    assert inventory["vad"]["available"] is False
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest tests/unit/test_firered_worker.py::test_worker_status_returns_loaded_inventory_without_audio tests/unit/test_firered_client.py::test_client_status_reuses_existing_worker tests/unit/test_resource_profile.py::test_speaker_inventory_reports_embedding_and_vad_without_gpu -v
```

Expected: failures because `status` and `resource_inventory` are unsupported.

- [ ] **Step 3: Implement the status handshake and inventory providers**

Extend `serve` with an injectable `inventory_provider` and accept `status` before audio-path
validation:

```python
if action == "status":
    _write_response(protocol_output, {
        "id": identifier,
        "ok": True,
        "status": inventory_provider(system),
        "metrics": {
            "load_seconds": load_seconds,
            "peak_gpu_bytes": peak_sampler(),
        },
    })
    continue
```

Implement `firered_inventory(system)` with explicit owner mappings: `vad.vad_model`,
`lid.model`, `asr.model`, and `punc.model`. It must call `module_inventory` lazily after
FireRed is loaded and include current `torch.cuda.memory_allocated/reserved` when CUDA exists.

Add `FireRedWorkerClient.status() -> dict[str, object]`; it sends the normal correlated
request, updates `last_response`, and returns the `status` mapping. Update the fake worker
fixture to answer `status` without assuming an `audio_path`.

Add `SpeakerDiarizer.resource_inventory() -> dict[str, object]`, returning an inventory for
the embedding model and either a concrete VAD inventory or `{"available": False}` when the
ModelScope pipeline has no inspectable `torch.nn.Module`. Add
`SpeakerDiarizer.diarize_with_resource_metrics(path)` that returns the unchanged segments
and a snapshot containing allocated/reserved before/after plus PyTorch peak values; it must
reset peak statistics only when called by profiling mode.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2, then run:

```bash
pytest tests/unit/test_firered_worker.py tests/unit/test_firered_client.py -v
```

Expected: status requests never invoke `process`, clients reuse one worker, and all existing
worker protocol tests remain green.

- [ ] **Step 5: Commit model status support**

```bash
git add src/funasr_transcriber/firered_worker.py src/funasr_transcriber/firered_client.py src/funasr_transcriber/speaker_diarization.py tests/unit/test_firered_worker.py tests/unit/test_firered_client.py tests/unit/test_resource_profile.py
git commit -m "feat: expose model resource inventories"
```

### Task 4: Implement the local service protocol and persistent state

**Files:**
- Create: `src/funasr_transcriber/service_protocol.py`
- Test: `tests/unit/test_service_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

```python
def test_service_paths_are_scoped_to_work_dir(tmp_path: Path):
    paths = ServicePaths.from_work_dir(tmp_path)

    assert paths.directory == tmp_path / "service"
    assert paths.socket == tmp_path / "service" / "transcriber.sock"
    assert paths.state == tmp_path / "service" / "state.json"


def test_state_store_rejects_live_incompatible_service(tmp_path: Path):
    paths = ServicePaths.from_work_dir(tmp_path)
    store = ServiceStateStore(paths, pid_alive=lambda pid: pid == 99)
    store.write(ServiceState(pid=99, protocol_version=1, config_fingerprint="old"))

    with pytest.raises(ServiceProtocolError, match="configuration"):
        store.require_compatible("new")


def test_protocol_rejects_relative_submit_paths():
    with pytest.raises(ServiceProtocolError, match="absolute"):
        parse_request({"id": "x", "action": "submit", "input_audio": "a.wav"})
```

- [ ] **Step 2: Run the protocol tests and verify RED**

Run:

```bash
pytest tests/unit/test_service_protocol.py -v
```

Expected: module import failure.

- [ ] **Step 3: Implement JSON-lines messages, state, and socket client helpers**

Implement immutable request/state dataclasses and these APIs:

```python
PROTOCOL_VERSION = 1

@dataclass(frozen=True, slots=True)
class ServicePaths: ...

@dataclass(frozen=True, slots=True)
class ServiceState: ...

def config_fingerprint(config: AppConfig) -> str: ...
def parse_request(payload: Mapping[str, Any]) -> ServiceRequest: ...
def encode_message(payload: Mapping[str, Any]) -> bytes: ...
def read_message(stream: IO[bytes]) -> dict[str, Any]: ...
```

Validate nonempty request IDs, known action names, absolute input/output paths for `submit`,
and boolean overlap/profile flags. `ServiceStateStore.write` must use a temporary file plus
`os.fsync` and atomic replace. It may remove a stale state/socket only after `pid_alive` says
the recorded PID is not alive; it must never send signals. Include `ServiceClient.health()`,
`submit_and_wait()`, and `stop_and_wait()` methods that preserve request IDs and require one
terminal event.

- [ ] **Step 4: Run the protocol tests and verify GREEN**

Run the command from Step 2.

Expected: all validation and state compatibility tests pass on CPU-only pytest.

- [ ] **Step 5: Commit the service protocol**

```bash
git add src/funasr_transcriber/service_protocol.py tests/unit/test_service_protocol.py
git commit -m "feat: add local service protocol"
```

### Task 5: Build the hot-model host and normal parallel job executor

**Files:**
- Create: `src/funasr_transcriber/service.py`
- Modify: `src/funasr_transcriber/pipeline.py`
- Modify: `src/funasr_transcriber/gpu.py`
- Test: `tests/unit/test_service.py`
- Modify: `tests/unit/test_cli_lifecycle.py`

- [ ] **Step 1: Write failing owner and parallelism tests**

```python
def test_hot_model_host_loads_once_and_closes_only_owned_models():
    events: list[str] = []
    host = HotModelHost(
        asr_factory=lambda: FakeAsr(events),
        diarizer_factory=lambda: FakeDiarizer(events),
        gpu_check=lambda: events.append("gpu-check"),
    )

    host.start()
    host.start()
    host.close()

    assert events[0] == "gpu-check"
    assert events.count("asr:start") == 1
    assert events.count("speaker:start") == 1
    assert events.count("asr:close") == 1
    assert events.count("speaker:close") == 1


def test_normal_job_starts_asr_and_diarization_before_waiting_for_results(tmp_path: Path):
    gate = threading.Barrier(2)
    result = run_normal_branches(
        tmp_path / "normalized.wav", CoordinatedAsr(gate), CoordinatedDiarizer(gate),
    )

    assert result.asr_result.text == "原始。"
    assert result.speaker_segments
    assert gate.broken is False


def test_normal_branch_waits_for_the_other_owner_before_raising(tmp_path: Path):
    asr = FailingAsr()
    diarizer = BlockingDiarizer()

    with pytest.raises(TranscriptionError, match="asr failed"):
        run_normal_branches(tmp_path / "normalized.wav", asr, diarizer)

    assert diarizer.finished is True
```

- [ ] **Step 2: Run the owner/parallelism tests and verify RED**

Run:

```bash
pytest tests/unit/test_service.py::test_hot_model_host_loads_once_and_closes_only_owned_models tests/unit/test_service.py::test_normal_job_starts_asr_and_diarization_before_waiting_for_results tests/unit/test_service.py::test_normal_branch_waits_for_the_other_owner_before_raising -v
```

Expected: import errors for `HotModelHost` and `run_normal_branches`.

- [ ] **Step 3: Implement the normal service owner lifecycle**

Implement `HotModelHost` as the only owner of one `FireRedWorkerClient` and one
`SpeakerDiarizer`. `start()` must first call `require_concurrent_gpu_capacity` with the
configured parallel budget, then call concurrent owner starts, wait for FireRed `status`, and
capture inventories. It must close any owner that did start if its peer fails. `close()` is
idempotent and only calls each owned context manager/`close` method once.

Implement the following result boundary in `service.py`:

```python
@dataclass(frozen=True, slots=True)
class NormalBranchResult:
    asr_result: AsrResult
    speaker_segments: tuple[SpeakerSegment, ...]
    speaker_metrics: dict[str, object]

def run_normal_branches(
    normalized: Path,
    asr_owner: Any,
    speaker_owner: Any,
    profile_enabled: bool = False,
) -> NormalBranchResult: ...
```

Use a `ThreadPoolExecutor(max_workers=2)`, start both futures before calling `.result()`, and
always wait for both futures. When both fail, raise the ASR error first because primary ASR is
the hard task failure; otherwise re-raise the single failing branch error. Move only the pure
alignment/finalization logic into a reusable `finish_primary_result` helper in `pipeline.py`.
Do not import FireRed in the service process.

Implement `ServiceJobRunner.run_normal_job(request)` around that branch helper: probe and
normalize the source in the request-specific job directory, invoke `run_normal_branches`, reject
empty speaker segments, call `attribute_primary` and `TranscriptionPipeline.finish`, then use
the existing atomic transcript/diagnostics exporters and staging-directory publish function.
The returned protocol terminal event contains the published output directory and serialized
domain-error kind, not an in-memory transcript.

- [ ] **Step 4: Run the focused tests and existing lifecycle tests**

Run:

```bash
pytest tests/unit/test_service.py tests/unit/test_cli_lifecycle.py -v
```

Expected: concurrent events are observed, owners are reused instead of serially reconstructed,
and all non-overlap transcript integrity tests pass.

- [ ] **Step 5: Commit the hot normal executor**

```bash
git add src/funasr_transcriber/service.py src/funasr_transcriber/pipeline.py src/funasr_transcriber/gpu.py tests/unit/test_service.py tests/unit/test_cli_lifecycle.py
git commit -m "feat: run hot ASR and speaker branches in parallel"
```

### Task 6: Add daemon FIFO, auto-start, health, and graceful stop

**Files:**
- Modify: `src/funasr_transcriber/service.py`
- Modify: `src/funasr_transcriber/locking.py`
- Modify: `src/funasr_transcriber/cli.py`
- Test: `tests/unit/test_service.py`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Write failing daemon lifecycle tests**

```python
def test_daemon_processes_submissions_in_fifo_order(tmp_path: Path):
    daemon = DaemonService.for_test(tmp_path, run_job=lambda request: request.id)
    first = daemon.submit(_request("first"))
    second = daemon.submit(_request("second"))

    daemon.run_until_idle()

    assert [first.result(), second.result()] == ["first", "second"]
    assert daemon.completed_ids == ["first", "second"]


def test_stop_drains_accepted_jobs_and_rejects_new_submissions(tmp_path: Path):
    daemon = DaemonService.for_test(tmp_path, run_job=lambda request: request.id)
    accepted = daemon.submit(_request("accepted"))
    daemon.request_stop()

    with pytest.raises(ServiceBusyError, match="draining"):
        daemon.submit(_request("new"))

    daemon.run_until_idle()
    assert accepted.result() == "accepted"
    assert daemon.stopped is True
```

```python
def test_transcribe_client_starts_service_only_when_health_is_unavailable(monkeypatch, tmp_path: Path):
    launches: list[Path] = []
    monkeypatch.setattr(cli, "ServiceClient", FakeUnavailableThenReadyClient)
    monkeypatch.setattr(cli, "launch_service", lambda config, paths: launches.append(paths.socket))

    cli.submit_to_service(_args(tmp_path), load_config())

    assert launches == [tmp_path / "service" / "transcriber.sock"]
```

- [ ] **Step 2: Run daemon tests and verify RED**

Run:

```bash
pytest tests/unit/test_service.py::test_daemon_processes_submissions_in_fifo_order tests/unit/test_service.py::test_stop_drains_accepted_jobs_and_rejects_new_submissions tests/unit/test_cli.py::test_transcribe_client_starts_service_only_when_health_is_unavailable -v
```

Expected: failures because the daemon, startup lock, and service client route do not exist.

- [ ] **Step 3: Implement daemon and CLI routing**

Add a `ServiceStartupLock` to `locking.py`, scoped to `ServicePaths.directory`, with the same
atomic/stale-PID safety rules as `TaskLock` but no input-audio ownership. Implement
`DaemonService` with `queue.Queue(maxsize=config.service.queue_limit)`, one consumer thread,
and a Unix `socketserver.ThreadingUnixStreamServer`. Socket handler threads may accept and
wait for jobs, but only the consumer may call `run_job`.

Implement a module entry point:

```bash
python -m funasr_transcriber.service --config /absolute/config.toml
```

It writes compatible `ServiceState` only after `HotModelHost.start()` succeeds, then serves
`health`, `submit`, and `stop`. `stop` must set draining before acknowledging it and delete
the socket/state only after the consumer and hot owners have closed.

Refactor `_run_transcribe` in `cli.py` into request construction plus
`submit_to_service(args, config)`. Add a dedicated `service` parser for `status` and `stop`.
`submit_to_service` must health-check first, acquire `ServiceStartupLock` only when needed,
launch the daemon with `CUDA_VISIBLE_DEVICES=0`, re-check health until the configured timeout,
then submit the original request ID. Replace the old per-invocation `TaskLock` use on the new
service route; keep `TaskLock` only where a non-service maintenance command still needs it.

- [ ] **Step 4: Run daemon/CLI tests and the full unit suite**

Run:

```bash
pytest tests/unit/test_service.py tests/unit/test_service_protocol.py tests/unit/test_cli.py tests/unit/test_cli_lifecycle.py -v
pytest tests/unit -q
```

Expected: FIFO, draining, auto-start and CLI exit mapping pass; no unit regressions.

- [ ] **Step 5: Commit daemon lifecycle support**

```bash
git add src/funasr_transcriber/service.py src/funasr_transcriber/service_protocol.py src/funasr_transcriber/locking.py src/funasr_transcriber/cli.py tests/unit/test_service.py tests/unit/test_cli.py
git commit -m "feat: add persistent local transcription service"
```

### Task 7: Split overlap preparation and implement safe hot-owner eviction

**Files:**
- Modify: `src/funasr_transcriber/overlap.py`
- Modify: `src/funasr_transcriber/service.py`
- Modify: `src/funasr_transcriber/cli.py`
- Test: `tests/unit/test_service.py`
- Test: `tests/unit/test_cli_lifecycle.py`
- Test: `tests/integration/test_overlap_worker.py`

- [ ] **Step 1: Write failing overlap sequencing tests**

```python
def test_overlap_job_evicts_hot_owners_before_pyannote_and_rehydrates_afterwards(tmp_path: Path):
    events: list[str] = []
    service = ServiceJobRunner.for_test(
        hot_host=FakeHotHost(events),
        overlap_factory=lambda: FakeOverlap(events),
    )

    service.run(_request("overlap", overlap_enabled=True))

    assert events.index("hot:close") < events.index("overlap:detect")
    assert events.index("overlap:close") < events.index("temporary:speaker")
    assert events.index("temporary:speaker-close") < events.index("temporary:asr")
    assert events[-1] == "hot:start"


def test_default_job_never_constructs_overlap_worker(tmp_path: Path):
    service = ServiceJobRunner.for_test(
        hot_host=FakeHotHost([]),
        overlap_factory=lambda: pytest.fail("overlap must stay disabled"),
    )

    service.run(_request("normal", overlap_enabled=False))
```

- [ ] **Step 2: Run overlap sequencing tests and verify RED**

Run:

```bash
pytest tests/unit/test_service.py::test_overlap_job_evicts_hot_owners_before_pyannote_and_rehydrates_afterwards tests/unit/test_service.py::test_default_job_never_constructs_overlap_worker -v
```

Expected: failures because overlap is still coupled to an active diarizer and no service job
runner exists.

- [ ] **Step 3: Refactor overlap into isolated stages**

Split `SubprocessOverlapPreparer.prepare` into two boundaries:

```python
def detect_and_separate(self, audio_path: Path) -> SeparatedOverlap: ...

def match_tracks(
    separated: SeparatedOverlap,
    centroids: dict[int, np.ndarray],
    track_diarizer: TrackDiarizer,
) -> OverlapPreparation: ...
```

`detect_and_separate` owns only the `.venv-overlap` subprocess calls and returns validated
window/track paths; it must not hold a diarizer reference. `match_tracks` owns only the
temporary 3D-Speaker embedding calls and the existing score/margin/RMS decisions.

Refactor `JsonOverlapWorkerClient.call` from opaque `subprocess.run` to an equivalent
`subprocess.Popen`/`communicate` implementation with optional `on_started(pid)` and
`on_finished(pid)` callbacks. Keep its timeout, JSON result-file validation, CUDA-visible-device
setting, and errors unchanged. The service registers that PID with `ResourceSampler` only while
the overlap subprocess is alive.

In `ServiceJobRunner.run`, execute normal hot branches first for an overlap request, call
`speaker_centroids`, close `HotModelHost`, and use `build_gpu_stage_record`/configured release
tolerance to verify release. Run detect/separate, close the overlap subprocess, load a
temporary diarizer for matching, close it, then use a temporary FireRed client for accepted
track transcription. Call `HotModelHost.start()` in `finally` before the daemon accepts the
next queue item. Preserve the existing per-window fallback warnings and never rehydrate early.

Add mutually-exclusive CLI flags:

```python
overlap_group = parser.add_mutually_exclusive_group()
overlap_group.add_argument("--enable-overlap", action="store_true")
overlap_group.add_argument("--disable-overlap", action="store_true")
```

Resolve the request value as `enable` → `True`, `disable` → `False`, otherwise the config
default (`False`).

- [ ] **Step 4: Run overlap tests and existing overlap integration tests**

Run:

```bash
pytest tests/unit/test_service.py tests/unit/test_cli_lifecycle.py tests/integration/test_overlap_worker.py -v
```

Expected: normal jobs avoid overlap imports/workers; overlap event order proves hot owners are
gone before pyannote; existing overlap protocol tests remain green.

- [ ] **Step 5: Commit safe overlap orchestration**

```bash
git add src/funasr_transcriber/overlap.py src/funasr_transcriber/service.py src/funasr_transcriber/cli.py tests/unit/test_service.py tests/unit/test_cli_lifecycle.py tests/integration/test_overlap_worker.py
git commit -m "feat: evict hot models for overlap jobs"
```

### Task 8: Attach opt-in profiling to service jobs and diagnostics

**Files:**
- Modify: `src/funasr_transcriber/service.py`
- Modify: `src/funasr_transcriber/diagnostics.py`
- Test: `tests/unit/test_service.py`
- Test: `tests/unit/test_diagnostics.py`

- [ ] **Step 1: Write failing profile behavior tests**

```python
def test_disabled_profile_does_not_create_sampler(tmp_path: Path):
    created: list[object] = []
    runner = ServiceJobRunner.for_test(
        sampler_factory=lambda *args: created.append(args),
    )

    result = runner.run(_request("normal", profile_resources=False))

    assert created == []
    assert result.diagnostics.resource_profile == {"enabled": False}


def test_enabled_profile_records_only_service_owned_pids_and_branch_metrics(tmp_path: Path):
    runner = ServiceJobRunner.for_test(
        sampler_factory=FakeSampler.with_samples({"service": 514, "firered": 9256}),
    )

    result = runner.run(_request("profiled", profile_resources=True))

    profile = result.diagnostics.resource_profile
    assert profile["enabled"] is True
    assert profile["process_peak_mib"] == {"service": 514, "firered": 9256}
    assert profile["branches"]["asr"]["seconds"] >= 0
    assert profile["branches"]["diarization"]["seconds"] >= 0
    assert "other_processes" not in profile
```

- [ ] **Step 2: Run profile tests and verify RED**

Run:

```bash
pytest tests/unit/test_service.py::test_disabled_profile_does_not_create_sampler tests/unit/test_service.py::test_enabled_profile_records_only_service_owned_pids_and_branch_metrics -v
```

Expected: failures because no per-request sampler is attached to the service job.

- [ ] **Step 3: Implement per-request profiling scope**

Use a context manager in `service.py` that constructs `ResourceSampler` only when
`request.profile_resources` is true. Register the daemon PID, current FireRed worker PID, and
only the temporary overlap PID while that owner exists. At scope exit, merge these exact keys:

```python
{
    "enabled": True,
    "sampling_interval_ms": config.service.profile_sampling_interval_ms,
    "sample_count": ...,
    "process_peak_mib": {"service": ..., "firered": ..., "overlap": ...},
    "cpu_rss_peak_kib": {"service": ...},
    "inventories": {"firered": ..., "speaker": ...},
    "branches": {"asr": ..., "diarization": ..., "alignment": ...},
    "uncertainty": {"driver_memory_unit": "MiB", "pid_observed": ...},
}
```

All PID values in diagnostics must be the service's own PID names, never raw system process
tables or other commands. For disabled requests set exactly `{"enabled": False}`. Record
parallel branch wall spans (`started_at_monotonic`, `ended_at_monotonic`, `seconds`) so output
can prove overlap in time without relying on thread scheduling order.

- [ ] **Step 4: Run profile tests and diagnostics tests**

Run:

```bash
pytest tests/unit/test_service.py tests/unit/test_diagnostics.py -v
```

Expected: disabled mode creates no sampler; enabled mode exports inventories, bounded PID data,
and parallel branch spans.

- [ ] **Step 5: Commit profiling integration**

```bash
git add src/funasr_transcriber/service.py src/funasr_transcriber/diagnostics.py tests/unit/test_service.py tests/unit/test_diagnostics.py
git commit -m "feat: write optional service resource profiles"
```

### Task 9: Update operator documentation, release gates, and integration tests

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/current-transcription-pipeline.md`
- Modify: `RELEASE_CHECKLIST.md`
- Modify: `scripts/run_validation_linux.sh`
- Modify: `tests/integration/test_pipeline_smoke.py`
- Modify: `tests/integration/test_firered_documentation.py`
- Modify: `tests/integration/test_doctor.py`
- Modify: `tests/integration/test_validation_corpus_cli.py`

- [ ] **Step 1: Write failing integration/documentation assertions**

```python
def test_architecture_documents_hot_parallel_service():
    text = Path("docs/architecture/current-transcription-pipeline.md").read_text()

    assert "Unix socket" in text
    assert "ASR 与 diarization 并行" in text
    assert "--profile-resources" in text
    assert "--enable-overlap" in text


def test_service_smoke_reuses_hot_owners_for_two_requests(tmp_path: Path):
    service = launch_fake_service(tmp_path)

    first = service.submit_and_wait(fake_submit("one"))
    second = service.submit_and_wait(fake_submit("two"))

    assert first["ok"] is True
    assert second["ok"] is True
    assert service.hot_load_count == 1
```

- [ ] **Step 2: Run the focused integration/documentation tests and verify RED**

Run:

```bash
pytest tests/integration/test_firered_documentation.py::test_architecture_documents_hot_parallel_service tests/integration/test_pipeline_smoke.py::test_service_smoke_reuses_hot_owners_for_two_requests -v
```

Expected: documentation assertion and fake service smoke fail until the new workflow is
documented and wired.

- [ ] **Step 3: Update docs and Linux validation**

Document `service status`, graceful `service stop`, automatic daemon startup, default overlap
behavior, `--enable-overlap`, `--profile-resources`, FIFO semantics, physical GPU 0-only
behavior, and the fact that `service stop` is the required model-release action before manual
GPU maintenance.

Update `scripts/run_validation_linux.sh` to run a service profile job against the release
corpus, assert `diagnostics.json.resource_profile.enabled == true`, assert the profile has only
the named project owner keys, run a second job and assert the service reports hot reuse, then
run `service stop` and verify no FireRed PID remains. Gate overlap validation behind an explicit
`--enable-overlap` invocation so the default validation path never imports overlap runtime.

- [ ] **Step 4: Run full non-GPU verification**

Run:

```bash
pytest -q
python -m compileall -q src
git diff --check
```

Expected: all unit/integration tests pass (model tests remain explicitly skipped unless their
environment variable is enabled), compilation succeeds, and the diff has no whitespace errors.

- [ ] **Step 5: Run real GPU 0 acceptance without affecting other projects**

Run only when GPU 0 meets the configured concurrent admission budget:

```bash
CUDA_VISIBLE_DEVICES=0 FUNASR_TRANSCRIBER_RUN_MODEL_TESTS=1 \
  FUNASR_TRANSCRIBER_MODEL_DIR="$PWD/models" \
  FUNASR_TRANSCRIBER_ASR_WORKER_PYTHON="$PWD/.venv-asr/bin/python" \
  pytest tests/integration/test_pipeline_smoke.py -v
```

Then submit one short profile-enabled job and one default job through the installed CLI, inspect
the two `diagnostics.json` files, run `funasr-transcribe service status`, and run
`funasr-transcribe service stop`. Query GPU 0 before and after with read-only `nvidia-smi` and
confirm no project service/FireRed process remains. Do not signal or otherwise interact with
any unrelated PID.

- [ ] **Step 6: Commit docs and release validation**

```bash
git add README.md docs/architecture/current-transcription-pipeline.md RELEASE_CHECKLIST.md scripts/run_validation_linux.sh tests/integration/test_pipeline_smoke.py tests/integration/test_firered_documentation.py tests/integration/test_doctor.py tests/integration/test_validation_corpus_cli.py
git commit -m "docs: document persistent parallel service"
```

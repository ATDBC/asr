from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading

import pytest

from funasr_transcriber.audio import AudioInfo
from funasr_transcriber.config import load_config
from funasr_transcriber.errors import TranscriptionError
from funasr_transcriber.schemas import (
    AsrResult,
    RecognizedSentence,
    SpeakerSegment,
    TimedToken,
)
from funasr_transcriber.service import (
    build_hot_model_host,
    DaemonService,
    HotModelHost,
    ServiceJobRunner,
    ServiceSocketServer,
    run_normal_branches,
)
import funasr_transcriber.service as service_module
from funasr_transcriber.service_protocol import (
    ServiceBusyError,
    ServiceClient,
    ServicePaths,
    ServiceProtocolError,
    ServiceRequest,
)
from funasr_transcriber.overlap import (
    OverlapPreparation,
    PreparedTrack,
    PreparedWindow,
    SeparatedOverlap,
)


def _result() -> AsrResult:
    return AsrResult(
        text="原始。",
        tokens=(TimedToken("原始。", 0, 500, 0.9),),
        sentences=(
            RecognizedSentence("原始。", 0, 500, 0.9),
        ),
        vad_segments_ms=((0, 500),),
        duration_ms=600,
    )


def test_owned_pid_getters_name_all_resident_owners():
    runner = ServiceJobRunner.__new__(ServiceJobRunner)
    runner.hot_host = type(
        "Host",
        (),
        {
            "asr_owner": type(
                "Asr",
                (),
                {
                    "pid": 101,
                    "pids": (101, 102),
                    "punc_owner": type("Punc", (), {"pid": 103})(),
                },
            )(),
        },
    )()
    runner._active_overlap_pid = 104

    getters = runner._owned_pid_getters()

    assert set(getters) == {
        "service",
        "speaker",
        "aed_worker_0",
        "aed_worker_1",
        "punc_worker",
        "overlap_worker",
    }
    assert getters["aed_worker_0"]() == 101
    assert getters["aed_worker_1"]() == 102
    assert getters["punc_worker"]() == 103
    assert getters["overlap_worker"]() == 104


class FakeAsr:
    def __init__(self, events: list[str]):
        self.events = events

    def start(self):
        self.events.append("asr:start")
        return self

    def status(self):
        return {"modules": {}}

    def warmup(self):
        self.events.append("asr:warmup")
        return {
            "seconds": 0.25,
            "excluded": True,
            "workers": [],
        }

    def cleanup_cuda_cache(self):
        self.events.append("asr:cleanup")
        return {
            "seconds": 0.125,
            "workers": [
                {
                    "worker_index": 0,
                    "reserved_before_mib": 7_104,
                    "reserved_after_mib": 2_468,
                },
                {
                    "worker_index": 1,
                    "reserved_before_mib": 7_104,
                    "reserved_after_mib": 2_468,
                },
            ],
        }

    def close(self):
        self.events.append("asr:close")

    def transcribe(self, path: Path, vad_segments_ms=None):
        self.events.append("asr:transcribe")
        return _result()


class FakeDiarizer:
    def __init__(self, events: list[str]):
        self.events = events

    def start(self):
        self.events.append("speaker:start")
        return self

    def close(self):
        self.events.append("speaker:close")

    def diarize(self, path: Path):
        self.events.append("speaker:diarize")
        return (SpeakerSegment(1, 0, 600),)

    def resource_inventory(self):
        return {"embedding": {}}


def test_hot_model_host_loads_once_and_closes_only_owned_models():
    events: list[str] = []
    host = HotModelHost(
        asr_factory=lambda: FakeAsr(events),
        diarizer_factory=lambda: FakeDiarizer(events),
        gpu_check=lambda: events.append("gpu-check"),
    )

    host.start()
    host.start()
    assert "asr:warmup" not in events
    first_warmup = host.warmup()
    second_warmup = host.warmup()
    host.close()

    assert events[0] == "gpu-check"
    assert events.count("asr:start") == 1
    assert events.count("speaker:start") == 1
    assert events.count("asr:close") == 1
    assert events.count("speaker:close") == 1
    assert events.count("asr:warmup") == 1
    assert first_warmup["performed"] is True
    assert first_warmup["excluded_from_job_timing"] is False
    assert second_warmup == {
        "performed": False,
        "excluded_from_job_timing": False,
        "reason": "already_performed",
    }


@pytest.mark.parametrize(
    ("worker_count", "expected_free_mib"),
    ((1, 0), (2, 13 * 1024)),
)
def test_only_single_worker_admission_omits_free_memory_gate(
    tmp_path: Path,
    monkeypatch,
    worker_count: int,
    expected_free_mib: int,
):
    calls = []
    monkeypatch.setattr(
        service_module,
        "require_gpu_capacity",
        lambda device, **kwargs: calls.append((device, kwargs)),
        raising=False,
    )

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"[asr]\naed_worker_count = {worker_count}\n",
        encoding="utf-8",
    )
    host = build_hot_model_host(
        load_config(config_path),
        ServicePaths.from_work_dir(tmp_path),
    )
    host._gpu_check()

    assert calls == [
        (
            "cuda:0",
            {
                "minimum_total_mib": 23 * 1024,
                "minimum_free_mib": expected_free_mib,
            },
        )
    ]


class RecordingAsr:
    def __init__(self):
        self.received_segments = None
        self.events: list[str] = []

    def transcribe(self, path: Path, vad_segments_ms=None):
        self.events.append("asr:transcribe")
        self.received_segments = vad_segments_ms
        return _result()


class RecordingDiarizer:
    def __init__(self, events: list[str]):
        self.events = events

    def diarize(self, path: Path):
        self.events.append("speaker:diarize")
        return (SpeakerSegment(1, 0, 600), SpeakerSegment(2, 700, 1200))


def test_normal_job_runs_diarization_before_asr_with_segments(tmp_path: Path):
    asr = RecordingAsr()
    diarizer = RecordingDiarizer(asr.events)

    result = run_normal_branches(
        tmp_path / "normalized.wav",
        asr,
        diarizer,
    )

    assert asr.events == ["speaker:diarize", "asr:transcribe"]
    assert asr.received_segments == [[0, 600], [700, 1200]]
    assert result.asr_result.text == "原始。"
    assert result.speaker_segments == (
        SpeakerSegment(1, 0, 600),
        SpeakerSegment(2, 700, 1200),
    )


class FailingDiarizer:
    def diarize(self, path: Path):
        raise TranscriptionError("diarization failed")


def test_diarization_failure_skips_asr(tmp_path: Path):
    asr = RecordingAsr()

    with pytest.raises(TranscriptionError, match="diarization failed"):
        run_normal_branches(tmp_path / "normalized.wav", asr, FailingDiarizer())

    assert asr.events == []


def _request(identifier: str) -> ServiceRequest:
    return ServiceRequest(identifier=identifier, action="submit")


def _socket_request(tmp_path: Path, identifier: str) -> ServiceRequest:
    return ServiceRequest(
        identifier=identifier,
        action="submit",
        input_audio=(tmp_path / f"{identifier}.wav").resolve(),
        output_dir=(tmp_path / identifier).resolve(),
        config_fingerprint="test-fingerprint",
    )


def test_daemon_processes_submissions_in_fifo_order():
    completed: list[str] = []
    daemon = DaemonService(
        queue_limit=2,
        run_job=lambda request: completed.append(request.identifier)
        or {"output_dir": request.identifier},
    )
    daemon.start()
    try:
        first = daemon.submit(_request("first"))
        second = daemon.submit(_request("second"))

        assert first.result(timeout=1) == {"output_dir": "first"}
        assert second.result(timeout=1) == {"output_dir": "second"}
        assert completed == ["first", "second"]
    finally:
        daemon.request_stop()
        assert daemon.wait_stopped(timeout=1)


def test_stop_drains_accepted_jobs_and_rejects_new_submissions():
    release = threading.Event()
    started = threading.Event()

    def run_job(request):
        started.set()
        release.wait(timeout=1)
        return {"output_dir": request.identifier}

    daemon = DaemonService(queue_limit=2, run_job=run_job)
    daemon.start()
    accepted = daemon.submit(_request("accepted"))
    assert started.wait(timeout=1)
    daemon.request_stop()

    with pytest.raises(ServiceBusyError, match="draining"):
        daemon.submit(_request("new"))

    release.set()
    assert accepted.result(timeout=1) == {"output_dir": "accepted"}
    assert daemon.wait_stopped(timeout=1)


def test_socket_server_serves_health_submission_and_graceful_stop(
    tmp_path: Path,
):
    paths = ServicePaths.from_work_dir(tmp_path)
    daemon = DaemonService(
        queue_limit=1,
        run_job=lambda request: {"output_dir": request.identifier},
    )
    server = ServiceSocketServer(
        paths,
        daemon,
        config_fingerprint="test-fingerprint",
    )
    server.start()
    client = ServiceClient(paths, timeout_seconds=1)
    try:
        assert client.health("health")["status"] == "ready"
        result = client.submit_and_wait(
            ServiceRequest(
                identifier="job",
                action="submit",
                input_audio=tmp_path / "input.wav",
                output_dir=tmp_path / "output",
                config_fingerprint="test-fingerprint",
            )
        )
        assert result["result"] == {"output_dir": "job"}
        assert client.stop_and_wait("stop")["status"] == "draining"
        assert daemon.wait_stopped(timeout=1)
    finally:
        server.close()


def test_socket_submit_acknowledges_before_blocking_job(tmp_path: Path):
    started = threading.Event()
    release = threading.Event()

    def run_job(request):
        started.set()
        assert release.wait(timeout=1)
        return {"output_dir": request.identifier}

    paths = ServicePaths.from_work_dir(tmp_path)
    daemon = DaemonService(queue_limit=2, run_job=run_job)
    server = ServiceSocketServer(
        paths,
        daemon,
        config_fingerprint="test-fingerprint",
    )
    server.start()
    client = ServiceClient(paths, timeout_seconds=1)
    try:
        accepted = client.submit(_socket_request(tmp_path, "job"))

        assert accepted == {
            "id": "job",
            "ok": True,
            "terminal": False,
            "status": "accepted",
            "job_id": "job",
            "queue_depth": 1,
        }
        assert started.wait(timeout=1)
        assert not release.is_set()

        release.set()
        assert client.wait("job", "wait-job")["result"] == {
            "output_dir": "job"
        }
    finally:
        release.set()
        server.close()


def test_socket_accepts_full_fifo_batch_before_waiting(tmp_path: Path):
    started = threading.Event()
    release = threading.Event()
    completed: list[str] = []

    def run_job(request):
        completed.append(request.identifier)
        if request.identifier == "first":
            started.set()
            assert release.wait(timeout=1)
        return {"output_dir": request.identifier}

    paths = ServicePaths.from_work_dir(tmp_path)
    daemon = DaemonService(queue_limit=3, run_job=run_job)
    server = ServiceSocketServer(
        paths,
        daemon,
        config_fingerprint="test-fingerprint",
    )
    server.start()
    client = ServiceClient(paths, timeout_seconds=1)
    try:
        first = _socket_request(tmp_path, "first")
        assert client.submit(first)["job_id"] == "first"
        assert started.wait(timeout=1)
        assert client.submit(_socket_request(tmp_path, "second"))["job_id"] == "second"
        assert client.submit(_socket_request(tmp_path, "third"))["job_id"] == "third"
        assert client.submit(first)["job_id"] == "first"
        assert completed == ["first"]

        release.set()

        assert client.wait("first", "wait-first")["result"] == {
            "output_dir": "first"
        }
        assert client.wait("second", "wait-second")["result"] == {
            "output_dir": "second"
        }
        assert client.wait("third", "wait-third")["result"] == {
            "output_dir": "third"
        }
        assert completed == ["first", "second", "third"]
        with pytest.raises(ServiceProtocolError, match="unknown service job"):
            client.wait("missing", "wait-missing")
    finally:
        release.set()
        server.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="abstract Unix sockets are a Linux transport fallback",
)
def test_socket_server_serves_health_from_a_long_work_directory(tmp_path: Path):
    paths = ServicePaths.from_work_dir(tmp_path / ("x" * 100))
    assert len(os.fsencode(str(paths.socket))) > 108
    daemon = DaemonService(queue_limit=1, run_job=lambda request: {})
    server = ServiceSocketServer(
        paths,
        daemon,
        config_fingerprint="test-fingerprint",
    )
    try:
        server.start()
        assert ServiceClient(paths, timeout_seconds=1).health("health")[
            "status"
        ] == "ready"
    finally:
        server.close()


def test_service_job_runner_exports_normal_job_from_hot_owners(
    tmp_path: Path,
):
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    config = load_config(
        cli_overrides={"paths.work_dir": str(tmp_path / "work")},
    )

    class HotHost:
        started = True
        asr_owner = FakeAsr([])
        speaker_owner = FakeDiarizer([])

    def normalize(source_path, destination, max_seconds):
        destination.write_bytes(b"normalized")
        return AudioInfo(
            path=Path(destination),
            duration_ms=600,
            sample_rate=16000,
            channels=1,
            codec="pcm_s16le",
        )

    runner = ServiceJobRunner(
        config,
        hot_host=HotHost(),
        probe=lambda path, max_seconds: AudioInfo(
            path=Path(path),
            duration_ms=600,
            sample_rate=16000,
            channels=1,
            codec="wav",
        ),
        normalize=normalize,
    )
    request = ServiceRequest(
        identifier="normal",
        action="submit",
        input_audio=source,
        output_dir=tmp_path / "output",
        config_fingerprint="test-fingerprint",
    )

    result = runner.run(request)

    output_dir = Path(result["output_dir"])
    assert output_dir == tmp_path / "output"
    assert (output_dir / "transcript.txt").read_text(encoding="utf-8")
    assert (output_dir / "diagnostics.json").is_file()
    assert not (tmp_path / "work" / "job-normal").exists()


def test_service_first_job_times_warmup_and_every_job_times_cleanup(
    tmp_path: Path,
):
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    config = load_config(
        cli_overrides={"paths.work_dir": str(tmp_path / "work")},
    )
    events: list[str] = []
    host = HotModelHost(
        asr_factory=lambda: FakeAsr(events),
        diarizer_factory=lambda: FakeDiarizer(events),
        gpu_check=lambda: events.append("gpu-check"),
    )

    def normalize(source_path, destination, max_seconds):
        destination.write_bytes(b"normalized")

    runner = ServiceJobRunner(
        config,
        hot_host=host,
        probe=lambda path, max_seconds: AudioInfo(
            Path(path), 600, 16000, 1, "wav"
        ),
        normalize=normalize,
        gpu_sampler=None,
    )

    try:
        for identifier in ("first", "second"):
            runner.run(
                ServiceRequest(
                    identifier=identifier,
                    action="submit",
                    input_audio=source,
                    output_dir=tmp_path / identifier,
                    config_fingerprint="test-fingerprint",
                )
            )
    finally:
        host.close()

    first = json.loads(
        (tmp_path / "first" / "diagnostics.json").read_text(encoding="utf-8")
    )
    second = json.loads(
        (tmp_path / "second" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert events.count("asr:warmup") == 1
    assert events.count("asr:cleanup") == 2
    assert events.index("asr:warmup") < events.index("speaker:diarize")
    assert first["runtime"]["warmup"]["performed"] is True
    assert first["runtime"]["warmup"]["excluded_from_job_timing"] is False
    assert second["runtime"]["warmup"] == {
        "performed": False,
        "excluded_from_job_timing": False,
        "reason": "already_performed",
    }
    for payload in (first, second):
        cleanup = payload["runtime"]["cache_cleanup"]
        assert cleanup["workers"][0]["reserved_after_mib"] == 2_468
        assert payload["stage_seconds"]["cache_cleanup"] >= 0
        assert payload["stage_seconds"]["output"] >= 0
        assert payload["stage_seconds"]["end_to_end_after_load"] >= payload[
            "stage_seconds"
        ]["cache_cleanup"] + payload["stage_seconds"]["output"]
        assert payload["stage_seconds"]["end_to_end_after_warmup"] == payload[
            "stage_seconds"
        ]["end_to_end_after_load"]
    assert first["stage_seconds"]["warmup"] >= 0
    assert "warmup" not in second["stage_seconds"]


def test_service_attempts_aed_cache_cleanup_when_job_fails(tmp_path: Path):
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    events: list[str] = []
    asr = FakeAsr(events)

    class Host:
        started = True
        asr_owner = asr
        speaker_owner = FailingDiarizer()

        def warmup(self):
            return {
                "performed": False,
                "excluded_from_job_timing": False,
                "reason": "already_performed",
            }

    runner = ServiceJobRunner(
        load_config(
            cli_overrides={"paths.work_dir": str(tmp_path / "work")},
        ),
        hot_host=Host(),
        probe=lambda path, max_seconds: AudioInfo(
            Path(path), 600, 16000, 1, "wav"
        ),
        normalize=lambda source_path, destination, max_seconds: (
            destination.write_bytes(b"normalized")
        ),
        gpu_sampler=None,
    )

    with pytest.raises(TranscriptionError, match="diarization failed"):
        runner.run(
            ServiceRequest(
                identifier="failed",
                action="submit",
                input_audio=source,
                output_dir=tmp_path / "failed",
                config_fingerprint="test-fingerprint",
            )
        )

    assert events.count("asr:cleanup") == 1


def test_service_fails_successful_job_when_aed_cache_cleanup_fails(
    tmp_path: Path,
):
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    events: list[str] = []

    class CleanupFailAsr(FakeAsr):
        def cleanup_cuda_cache(self):
            events.append("asr:cleanup")
            raise RuntimeError("cleanup failed")

    host = HotModelHost(
        asr_factory=lambda: CleanupFailAsr(events),
        diarizer_factory=lambda: FakeDiarizer(events),
        gpu_check=lambda: None,
    )
    runner = ServiceJobRunner(
        load_config(
            cli_overrides={"paths.work_dir": str(tmp_path / "work")},
        ),
        hot_host=host,
        probe=lambda path, max_seconds: AudioInfo(
            Path(path), 600, 16000, 1, "wav"
        ),
        normalize=lambda source_path, destination, max_seconds: (
            destination.write_bytes(b"normalized")
        ),
        gpu_sampler=None,
    )

    try:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            runner.run(
                ServiceRequest(
                    identifier="cleanup-failed",
                    action="submit",
                    input_audio=source,
                    output_dir=tmp_path / "output",
                    config_fingerprint="test-fingerprint",
                )
            )
    finally:
        host.close()

    assert events.count("asr:cleanup") == 1
    assert not (tmp_path / "output").exists()


def test_daemon_module_loads_config_and_delegates_to_runner(
    monkeypatch,
    tmp_path: Path,
):
    config = load_config(
        cli_overrides={"paths.work_dir": str(tmp_path / "work")},
    )
    received = []
    monkeypatch.setattr(service_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        service_module,
        "run_daemon",
        lambda received_config: received.append(received_config),
    )

    assert service_module.main(["--config", str(tmp_path / "config.toml")]) == 0
    assert received == [config]


def test_overlap_job_evicts_hot_owners_before_temporary_models(
    tmp_path: Path,
):
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    track_one = tmp_path / "track-one.wav"
    track_two = tmp_path / "track-two.wav"
    track_one.write_bytes(b"one")
    track_two.write_bytes(b"two")
    config = load_config(
        cli_overrides={"paths.work_dir": str(tmp_path / "work")},
    )
    events: list[str] = []

    class ResidentAsr:
        def transcribe(self, path, vad_segments_ms=None):
            return _result()

        def cleanup_cuda_cache(self):
            events.append("hot:cleanup")
            return {"seconds": 0.0, "workers": []}

    class ResidentSpeaker:
        def diarize(self, path):
            return (SpeakerSegment(1, 0, 600),)

        def speaker_centroids(self, path, segments):
            events.append("hot:centroids")
            return {1: object(), 2: object()}

    class HotHost:
        def __init__(self):
            self.started = True
            self.asr_owner = ResidentAsr()
            self.speaker_owner = ResidentSpeaker()

        def close(self):
            events.append("hot:close")
            self.started = False
            self.asr_owner = None
            self.speaker_owner = None

        def start(self):
            events.append("hot:start")
            self.started = True
            self.asr_owner = ResidentAsr()
            self.speaker_owner = ResidentSpeaker()
            return self

    class Preparer:
        def detect_and_separate(self, path):
            events.append("overlap:detect")
            return SeparatedOverlap(({},), ())

        def match_tracks(self, separated, centroids, diarizer):
            events.append("overlap:match")
            return OverlapPreparation(
                (
                    PreparedWindow(
                        "w1",
                        0,
                        600,
                        (
                            PreparedTrack(track_one, 1, 0.9, 0.2, 0.1),
                            PreparedTrack(track_two, 2, 0.9, 0.2, 0.1),
                        ),
                    ),
                ),
                (),
            )

    class TemporarySpeaker:
        def __init__(self):
            events.append("temporary:speaker")

        def close(self):
            events.append("temporary:speaker-close")

    class TemporaryAsr:
        def __init__(self):
            events.append("temporary:asr")

        def transcribe(self, path, vad_segments_ms=None):
            text = "甲。" if Path(path) == track_one else "乙。"
            return AsrResult(
                text,
                (TimedToken(text, 0, 500, 0.9),),
                sentences=(
                    RecognizedSentence(text, 0, 500, 0.9),
                ),
                vad_segments_ms=((0, 500),),
                duration_ms=600,
            )

        def close(self):
            events.append("temporary:asr-close")

    def normalize(source_path, destination, max_seconds):
        destination.write_bytes(b"normalized")

    runner = ServiceJobRunner(
        config,
        hot_host=HotHost(),
        probe=lambda path, max_seconds: AudioInfo(
            Path(path), 600, 16000, 1, "wav"
        ),
        normalize=normalize,
        overlap_factory=lambda job_dir: Preparer(),
        temporary_diarizer_factory=TemporarySpeaker,
        temporary_asr_factory=lambda job_dir: TemporaryAsr(),
        gpu_sampler=None,
    )

    runner.run(
        ServiceRequest(
            identifier="overlap",
            action="submit",
            input_audio=source,
            output_dir=tmp_path / "output",
            config_fingerprint="test-fingerprint",
            overlap_enabled=True,
        )
    )

    assert events.index("hot:close") < events.index("overlap:detect")
    assert events.index("overlap:detect") < events.index("temporary:speaker")
    assert events.index("temporary:speaker-close") < events.index("temporary:asr")
    assert events.index("hot:start") < events.index("hot:cleanup")
    assert events[-1] == "hot:cleanup"


def _profiled_runner(tmp_path: Path, sampler_factory):
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    config = load_config(
        cli_overrides={"paths.work_dir": str(tmp_path / "work")},
    )

    class HotHost:
        started = True
        asr_owner = FakeAsr([])
        speaker_owner = FakeDiarizer([])
        inventories = {"firered": {"modules": {}}, "speaker": {"embedding": {}}}

    def normalize(source_path, destination, max_seconds):
        destination.write_bytes(b"normalized")

    return (
        ServiceJobRunner(
            config,
            hot_host=HotHost(),
            probe=lambda path, max_seconds: AudioInfo(
                Path(path), 600, 16000, 1, "wav"
            ),
            normalize=normalize,
            sampler_factory=sampler_factory,
        ),
        source,
    )


def test_disabled_profile_does_not_create_sampler(tmp_path: Path):
    created = []
    runner, source = _profiled_runner(
        tmp_path,
        sampler_factory=lambda *args, **kwargs: created.append((args, kwargs)),
    )

    runner.run(
        ServiceRequest(
            identifier="normal",
            action="submit",
            input_audio=source,
            output_dir=tmp_path / "output",
            config_fingerprint="test-fingerprint",
            profile_resources=False,
        )
    )

    payload = __import__("json").loads(
        (tmp_path / "output" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert created == []
    assert payload["resource_profile"] == {"enabled": False}


def test_enabled_profile_records_owned_pids_and_branch_metrics(tmp_path: Path):
    class FakeSampler:
        def start(self):
            return None

        def stop(self):
            return {
                "sampling_interval_ms": 200,
                "sample_count": 2,
                "process_peak_mib": {
                    "service": 514,
                    "aed_worker_0": 9256,
                    "punc_worker": 500,
                },
                "cpu_rss_peak_kib": {"service": 1000},
                "pid_observed": {
                    "service": True,
                    "aed_worker_0": True,
                    "punc_worker": True,
                },
            }

    runner, source = _profiled_runner(
        tmp_path,
        sampler_factory=lambda *args, **kwargs: FakeSampler(),
    )

    runner.run(
        ServiceRequest(
            identifier="profiled",
            action="submit",
            input_audio=source,
            output_dir=tmp_path / "output",
            config_fingerprint="test-fingerprint",
            profile_resources=True,
        )
    )

    payload = __import__("json").loads(
        (tmp_path / "output" / "diagnostics.json").read_text(encoding="utf-8")
    )
    profile = payload["resource_profile"]
    assert profile["enabled"] is True
    assert profile["process_peak_mib"] == {
        "service": 514,
        "aed_worker_0": 9256,
        "punc_worker": 500,
    }
    assert profile["branches"]["asr"]["seconds"] >= 0
    assert profile["branches"]["diarization"]["seconds"] >= 0
    assert "other_processes" not in profile


def test_deep_profile_includes_aed_memory_trace(tmp_path: Path):
    created: dict[str, object] = {}

    class DeepAsr(FakeAsr):
        def __init__(self):
            super().__init__([])
            self.last_response = {
                "metrics": {
                    "parallel": {
                        "aed_started_at_monotonic": 10.0,
                        "aed_ended_at_monotonic": 20.0,
                    }
                }
            }

    class TraceSampler:
        def start(self):
            return None

        def stop(self):
            return {"sample_count": 1, "pid_observed": {}}

        def trace_stage(self, started, ended, **kwargs):
            assert (started, ended) == (10.0, 20.0)
            assert kwargs["effective_budget_mib"] is None
            return {"sample_count": 1, "samples": [], "summary": {}}

    runner, source = _profiled_runner(
        tmp_path,
        sampler_factory=lambda *args, **kwargs: (
            created.update(kwargs) or TraceSampler()
        ),
    )
    runner.hot_host.asr_owner = DeepAsr()

    runner.run(
        ServiceRequest(
            identifier="deep-profiled",
            action="submit",
            input_audio=source,
            output_dir=tmp_path / "output",
            config_fingerprint="test-fingerprint",
            diagnostics_level="deep",
        )
    )

    payload = __import__("json").loads(
        (tmp_path / "output" / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert created["trace_enabled"] is True
    assert payload["resource_profile"]["aed_memory_trace"]["sample_count"] == 1

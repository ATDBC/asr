from __future__ import annotations

from pathlib import Path
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
    ServiceRequest,
)


def _result() -> AsrResult:
    return AsrResult(
        text="原始。",
        tokens=(TimedToken("原始。", 0, 500, 0.9),),
        sentences=(
            RecognizedSentence("原始。", 0, 500, "zh mandarin", 0.99, 0.9),
        ),
        vad_segments_ms=((0, 500),),
        duration_ms=600,
    )


class FakeAsr:
    def __init__(self, events: list[str]):
        self.events = events

    def start(self):
        self.events.append("asr:start")
        return self

    def status(self):
        return {"modules": {}}

    def close(self):
        self.events.append("asr:close")

    def transcribe(self, path: Path):
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
    host.close()

    assert events[0] == "gpu-check"
    assert events.count("asr:start") == 1
    assert events.count("speaker:start") == 1
    assert events.count("asr:close") == 1
    assert events.count("speaker:close") == 1


class CoordinatedAsr:
    def __init__(self, gate: threading.Barrier):
        self.gate = gate

    def transcribe(self, path: Path):
        self.gate.wait()
        return _result()


class CoordinatedDiarizer:
    def __init__(self, gate: threading.Barrier):
        self.gate = gate

    def diarize(self, path: Path):
        self.gate.wait()
        return (SpeakerSegment(1, 0, 600),)


def test_normal_job_starts_asr_and_diarization_before_waiting_for_results(
    tmp_path: Path,
):
    gate = threading.Barrier(2, timeout=1)

    result = run_normal_branches(
        tmp_path / "normalized.wav",
        CoordinatedAsr(gate),
        CoordinatedDiarizer(gate),
    )

    assert result.asr_result.text == "原始。"
    assert result.speaker_segments == (SpeakerSegment(1, 0, 600),)
    assert gate.broken is False


class FailingAsr:
    def transcribe(self, path: Path):
        raise TranscriptionError("asr failed")


class BlockingDiarizer:
    def __init__(self):
        self.finished = False

    def diarize(self, path: Path):
        self.finished = True
        return (SpeakerSegment(1, 0, 600),)


def test_normal_branch_waits_for_the_other_owner_before_raising(tmp_path: Path):
    diarizer = BlockingDiarizer()

    with pytest.raises(TranscriptionError, match="asr failed"):
        run_normal_branches(tmp_path / "normalized.wav", FailingAsr(), diarizer)

    assert diarizer.finished is True


def _request(identifier: str) -> ServiceRequest:
    return ServiceRequest(identifier=identifier, action="submit")


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

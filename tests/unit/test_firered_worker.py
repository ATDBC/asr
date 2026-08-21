from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from funasr_transcriber.firered_worker import (
    AudioInfo,
    GuardedVad,
    firered_inventory,
    serve,
)


def _official(path: Path) -> dict[str, object]:
    return {
        "uttid": "official-id",
        "text": "你好。",
        "sentences": [
            {
                "start_ms": 100,
                "end_ms": 900,
                "text": "你好。",
                "asr_confidence": 0.95,
            }
        ],
        "vad_segments_ms": [(80, 920)],
        "dur_s": 1.0,
        "words": [
            {"start_ms": 120, "end_ms": 300, "text": "你"},
            {"start_ms": 300, "end_ms": 700, "text": "好"},
        ],
        "wav_path": str(path),
    }


class FakeSystem:
    def __init__(self, result: dict[str, object]):
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def process(self, path: str, uttid: str):
        self.calls.append((path, uttid))
        return self.result


class EmptyModule:
    def parameters(self):
        return ()

    def buffers(self):
        return ()


def test_firered_inventory_excludes_lid_module():
    module = EmptyModule()
    system = SimpleNamespace(
        vad=SimpleNamespace(vad_model=module),
        asr=SimpleNamespace(model=module),
        punc=SimpleNamespace(model=module),
    )

    inventory = firered_inventory(system)

    assert set(inventory["modules"]) == {"vad", "asr", "punc"}


def _request(identifier: str, action: str, path: Path | None = None) -> str:
    payload = {"id": identifier, "action": action}
    if path is not None:
        payload["audio_path"] = str(path)
    return json.dumps(payload) + "\n"


def _valid_info(path: Path) -> AudioInfo:
    return AudioInfo(sample_rate=16000, channels=1, subtype="PCM_16")


def test_worker_correlates_transcribe_response(tmp_path: Path):
    wav = (tmp_path / "normalized.wav").resolve()
    wav.write_bytes(b"fake")
    system = FakeSystem(_official(wav))
    target = StringIO()

    serve(
        StringIO(_request("abc", "transcribe", wav)),
        target,
        lambda: system,
        _valid_info,
    )

    payload = json.loads(target.getvalue())
    assert payload["id"] == "abc"
    assert payload["ok"] is True
    assert payload["text"] == "你好。"
    assert system.calls == [(str(wav), "abc")]


def test_worker_loads_once_and_acknowledges_shutdown(tmp_path: Path):
    wav = (tmp_path / "normalized.wav").resolve()
    wav.write_bytes(b"fake")
    system = FakeSystem(_official(wav))
    loads = []

    source = StringIO(
        _request("one", "transcribe", wav)
        + _request("two", "transcribe", wav)
        + _request("stop", "shutdown")
        + _request("ignored", "transcribe", wav)
    )
    target = StringIO()
    serve(source, target, lambda: loads.append(True) or system, _valid_info)

    responses = [json.loads(line) for line in target.getvalue().splitlines()]
    assert loads == [True]
    assert [item["id"] for item in responses] == ["one", "two", "stop"]
    assert responses[-1] == {"id": "stop", "ok": True, "shutdown": True}


def test_worker_reports_load_cumulative_inference_and_peak_metrics(
    tmp_path: Path,
):
    wav = (tmp_path / "normalized.wav").resolve()
    wav.write_bytes(b"fake")
    system = FakeSystem(_official(wav))
    target = StringIO()
    times = iter((1.0, 3.5, 10.0, 12.0, 20.0, 23.0))

    serve(
        StringIO(
            _request("one", "transcribe", wav)
            + _request("two", "transcribe", wav)
        ),
        target,
        lambda: system,
        _valid_info,
        clock=lambda: next(times),
        peak_sampler=lambda: 456,
    )

    responses = [json.loads(line) for line in target.getvalue().splitlines()]
    assert responses[0]["metrics"] == {
        "load_seconds": 2.5,
        "inference_seconds": 2.0,
        "peak_gpu_bytes": 456,
    }
    assert responses[1]["metrics"] == {
        "load_seconds": 2.5,
        "inference_seconds": 5.0,
        "peak_gpu_bytes": 456,
    }


def test_verify_runs_the_full_system(tmp_path: Path):
    wav = (tmp_path / "doctor.wav").resolve()
    wav.write_bytes(b"fake")
    system = FakeSystem(_official(wav))
    target = StringIO()

    serve(
        StringIO(_request("doctor", "verify", wav)),
        target,
        lambda: system,
        _valid_info,
    )

    response = json.loads(target.getvalue())
    assert response["verified"] is True
    assert response["tokens"]
    assert system.calls == [(str(wav), "doctor")]


def test_worker_status_returns_loaded_inventory_without_audio():
    system = FakeSystem({})
    target = StringIO()
    times = iter((1.0, 3.5))

    serve(
        StringIO(_request("status", "status")),
        target,
        lambda: system,
        clock=lambda: next(times),
        peak_sampler=lambda: 456,
        inventory_provider=lambda _: {"models": ["vad"]},
    )

    assert json.loads(target.getvalue()) == {
        "id": "status",
        "ok": True,
        "status": {"models": ["vad"]},
        "metrics": {"load_seconds": 2.5, "peak_gpu_bytes": 456},
    }
    assert system.calls == []


def test_invalid_json_does_not_pollute_following_response(tmp_path: Path):
    wav = (tmp_path / "normalized.wav").resolve()
    wav.write_bytes(b"fake")
    target = StringIO()

    serve(
        StringIO("not-json\n" + _request("ok", "transcribe", wav)),
        target,
        lambda: FakeSystem(_official(wav)),
        _valid_info,
    )

    responses = [json.loads(line) for line in target.getvalue().splitlines()]
    assert responses[0]["id"] is None
    assert responses[0]["ok"] is False
    assert responses[0]["error"]["type"] == "JSONDecodeError"
    assert responses[1]["id"] == "ok"
    assert responses[1]["ok"] is True


def test_worker_rejects_duplicate_request_id(tmp_path: Path):
    wav = (tmp_path / "normalized.wav").resolve()
    wav.write_bytes(b"fake")
    target = StringIO()

    serve(
        StringIO(
            _request("same", "transcribe", wav)
            + _request("same", "transcribe", wav)
        ),
        target,
        lambda: FakeSystem(_official(wav)),
        _valid_info,
    )

    responses = [json.loads(line) for line in target.getvalue().splitlines()]
    assert responses[0]["ok"] is True
    assert responses[1]["ok"] is False
    assert "duplicate" in responses[1]["error"]["message"]


def test_worker_rejects_unsupported_action():
    target = StringIO()

    serve(
        StringIO(_request("bad", "download")),
        target,
        lambda: FakeSystem({}),
        _valid_info,
    )

    response = json.loads(target.getvalue())
    assert response["ok"] is False
    assert "unsupported" in response["error"]["message"]


def test_worker_rejects_relative_audio_path(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("relative.wav").write_bytes(b"fake")
    target = StringIO()

    serve(
        StringIO(_request("bad", "transcribe", Path("relative.wav"))),
        target,
        lambda: FakeSystem({}),
        _valid_info,
    )

    response = json.loads(target.getvalue())
    assert response["ok"] is False
    assert "absolute" in response["error"]["message"]


def test_worker_rejects_non_normalized_audio(tmp_path: Path):
    wav = (tmp_path / "stereo.wav").resolve()
    wav.write_bytes(b"fake")
    target = StringIO()

    serve(
        StringIO(_request("bad", "transcribe", wav)),
        target,
        lambda: FakeSystem({}),
        lambda path: AudioInfo(48000, 2, "FLOAT"),
    )

    response = json.loads(target.getvalue())
    assert response["ok"] is False
    assert "16 kHz mono PCM_16" in response["error"]["message"]


def test_worker_serializes_per_request_model_failure(tmp_path: Path):
    wav = (tmp_path / "normalized.wav").resolve()
    wav.write_bytes(b"fake")

    class BrokenSystem:
        def process(self, path: str, uttid: str):
            raise RuntimeError("inference failed")

    target = StringIO()
    serve(
        StringIO(_request("broken", "transcribe", wav)),
        target,
        BrokenSystem,
        _valid_info,
    )

    response = json.loads(target.getvalue())
    assert response == {
        "id": "broken",
        "ok": False,
        "error": {"type": "RuntimeError", "message": "inference failed"},
    }


def test_guarded_vad_rejects_unsafe_segment_before_aed():
    class UnsafeVad:
        def detect(self, path: str):
            return {"timestamps": [(1.0, 61.001)]}, [0.9]

    with pytest.raises(ValueError, match="60 second"):
        GuardedVad(UnsafeVad()).detect("normalized.wav")


class ScriptedVadSystem:
    def __init__(self, result: dict[str, object]):
        self.result = result
        self.seen_timestamps: list[tuple[float, float]] | None = None

    def process(self, path: str, uttid: str):
        vad_result, _ = self.vad.detect(path)
        self.seen_timestamps = list(vad_result["timestamps"])
        return self.result


def _segment_request(identifier: str, path: Path, segments_ms) -> str:
    payload = {
        "id": identifier,
        "action": "transcribe",
        "audio_path": str(path),
        "vad_segments_ms": segments_ms,
    }
    return json.dumps(payload) + "\n"


def test_worker_scripts_vad_segments_for_one_request(tmp_path: Path):
    wav = (tmp_path / "normalized.wav").resolve()
    wav.write_bytes(b"fake")
    system = ScriptedVadSystem(_official(wav))
    target = StringIO()

    serve(
        StringIO(_segment_request("seg", wav, [[0, 500], [700, 1200]])),
        target,
        lambda: system,
        _valid_info,
    )

    payload = json.loads(target.getvalue())
    assert payload["ok"] is True
    assert system.seen_timestamps == [(0.0, 0.5), (0.7, 1.2)]
    assert system.vad is None


def test_worker_subdivides_long_scripted_segment_with_neural_vad(
    tmp_path: Path,
):
    wav = (tmp_path / "normalized.wav").resolve()
    wav.write_bytes(b"fake")

    class NeuralVad:
        def detect(self, path: str):
            return {"timestamps": [(0.0, 20.0), (25.0, 70.0)]}, None

    neural = NeuralVad()
    system = ScriptedVadSystem(_official(wav))
    system.vad = neural
    target = StringIO()

    serve(
        StringIO(
            _segment_request("long", wav, [[0, 5_000], [10_000, 80_000]])
        ),
        target,
        lambda: system,
        _valid_info,
    )

    payload = json.loads(target.getvalue())
    assert payload["ok"] is True
    assert system.seen_timestamps == [(0.0, 5.0), (10.0, 20.0), (25.0, 70.0)]
    assert system.vad is neural


def test_worker_hard_splits_long_segment_without_fallback_vad(tmp_path: Path):
    wav = (tmp_path / "normalized.wav").resolve()
    wav.write_bytes(b"fake")
    system = ScriptedVadSystem(_official(wav))
    target = StringIO()

    serve(
        StringIO(_segment_request("long", wav, [[0, 121_000]])),
        target,
        lambda: system,
        _valid_info,
    )

    payload = json.loads(target.getvalue())
    assert payload["ok"] is True
    assert system.seen_timestamps == [(0.0, 60.0), (60.0, 120.0), (120.0, 121.0)]
    assert system.vad is None


def test_worker_rejects_non_monotonic_scripted_segments(tmp_path: Path):
    wav = (tmp_path / "normalized.wav").resolve()
    wav.write_bytes(b"fake")
    system = ScriptedVadSystem(_official(wav))
    target = StringIO()

    serve(
        StringIO(_segment_request("unordered", wav, [[700, 1200], [0, 500]])),
        target,
        lambda: system,
        _valid_info,
    )

    response = json.loads(target.getvalue())
    assert response["ok"] is False
    assert "monotonic" in response["error"]["message"]
    assert system.seen_timestamps is None

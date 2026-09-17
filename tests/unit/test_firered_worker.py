from __future__ import annotations

import gc
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from funasr_transcriber.firered_worker import (
    AudioInfo,
    GuardedVad,
    _explicit_aed_process,
    _scheduled_process,
    firered_inventory,
    serve,
    _parser,
    _warmup_process,
)
import funasr_transcriber.firered_worker as worker_module


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


def test_worker_half_precision_switch_defaults_off_and_can_be_enabled(tmp_path: Path):
    parser = _parser()

    assert parser.parse_args(["--model-root", str(tmp_path)]).use_half is False
    assert parser.parse_args(
        ["--model-root", str(tmp_path), "--use-half"]
    ).use_half is True


def test_warmup_process_exercises_representative_aed_shapes(tmp_path: Path, monkeypatch):
    wav = tmp_path / "warmup.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.int16), 16_000, subtype="PCM_16")

    class Asr:
        def __init__(self):
            self.calls = []

        def transcribe(self, uttids, wavs):
            self.calls.append((list(uttids), [len(item[1]) for item in wavs]))
            return [
                {"uttid": uttid, "timestamp": [], "confidence": 0.9}
                for uttid in uttids
            ]

    asr = Asr()
    monkeypatch.setattr(worker_module, "_synchronize_cuda", lambda: None)
    result = _warmup_process(SimpleNamespace(asr=asr, punc=None), str(wav))

    assert [len(call[0]) for call in asr.calls] == [1, 4, 1, 1]
    assert [call[1][0] for call in asr.calls] == [16000, 16000, 160000, 480000]
    assert result["shapes"] == ["short_single", "short_batch", "medium", "long"]


def test_cleanup_cuda_cache_collects_then_synchronizes_and_empties(monkeypatch):
    events: list[str] = []
    allocated = iter((2_246, 2_246))
    reserved = iter((7_104, 2_468))
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: events.append("empty_cache"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(gc, "collect", lambda: events.append("gc") or 17)
    monkeypatch.setattr(
        worker_module,
        "_synchronize_cuda",
        lambda: events.append("synchronize"),
    )
    monkeypatch.setattr(
        worker_module,
        "_cuda_memory_allocated_mib",
        lambda: next(allocated),
    )
    monkeypatch.setattr(
        worker_module,
        "_cuda_reserved_memory_mib",
        lambda: next(reserved),
    )

    result = getattr(worker_module, "_cleanup_cuda_cache")()

    assert events == ["gc", "synchronize", "empty_cache", "synchronize"]
    assert result == {
        "gc_collected": 17,
        "allocated_before_mib": 2_246,
        "reserved_before_mib": 7_104,
        "allocated_after_mib": 2_246,
        "reserved_after_mib": 2_468,
    }


def test_worker_serves_cleanup_cuda_cache_without_audio(monkeypatch):
    monkeypatch.setattr(
        worker_module,
        "_cleanup_cuda_cache",
        lambda: {
            "gc_collected": 3,
            "allocated_before_mib": 2_246,
            "reserved_before_mib": 6_944,
            "allocated_after_mib": 2_246,
            "reserved_after_mib": 2_468,
        },
        raising=False,
    )
    protocol_output = StringIO()
    times = iter((0.0, 1.0, 10.0, 10.25))

    serve(
        StringIO(
            '{"id":"cleanup","action":"cleanup_cuda_cache"}\n'
            '{"id":"stop","action":"shutdown"}\n'
        ),
        protocol_output,
        system_factory=lambda: SimpleNamespace(),
        clock=lambda: next(times),
    )

    responses = [json.loads(line) for line in protocol_output.getvalue().splitlines()]
    assert responses[0] == {
        "id": "cleanup",
        "ok": True,
        "cache_cleanup": {
            "gc_collected": 3,
            "allocated_before_mib": 2_246,
            "reserved_before_mib": 6_944,
            "allocated_after_mib": 2_246,
            "reserved_after_mib": 2_468,
            "seconds": 0.25,
        },
    }
    assert responses[1] == {"id": "stop", "ok": True, "shutdown": True}


def test_worker_reports_punc_chunk_cuda_peaks(tmp_path: Path, monkeypatch):
    wav = tmp_path / "normalized.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.int16), 16_000, subtype="PCM_16")

    class Punc:
        def process_with_timestamp(self, timestamps, uttids):
            return [
                {
                    "uttid": uttids[0],
                    "punc_sentences": [
                        {"punc_text": "测试。", "start_s": 0, "end_s": 1}
                    ],
                }
            ]

    monkeypatch.setattr(worker_module, "_reset_cuda_peak_memory", lambda: None)
    monkeypatch.setattr(worker_module, "_synchronize_cuda", lambda: None)
    monkeypatch.setattr(worker_module, "_cuda_peak_memory_mib", lambda: 900)
    monkeypatch.setattr(worker_module, "_cuda_peak_reserved_mib", lambda: 1_000)
    monkeypatch.setattr(worker_module, "_cuda_memory_allocated_mib", lambda: 800)
    monkeypatch.setattr(worker_module, "_cuda_reserved_memory_mib", lambda: 850)
    request = {
        "id": "punc",
        "action": "punctuate_chunk",
        "audio_path": str(wav),
        "aed_results": [
            {
                "index": 0,
                "result": {
                    "uttid": "job_s0_e1000",
                    "timestamp": [["测试", 0.0, 1.0]],
                    "confidence": 0.9,
                },
            }
        ],
    }
    protocol_output = StringIO()
    times = iter((0.0, 1.0, 10.0, 10.25))

    serve(
        StringIO(json.dumps(request) + "\n" + _request("stop", "shutdown")),
        protocol_output,
        system_factory=lambda: SimpleNamespace(
            punc=Punc(),
            punc_token_budget=256,
            punc_max_items=16,
        ),
        clock=lambda: next(times),
    )

    response = json.loads(protocol_output.getvalue().splitlines()[0])
    assert response["punc_metrics"] == {
        "actual_seconds": 0.25,
        "peak_allocated_mib": 900,
        "peak_reserved_mib": 1_000,
        "resident_allocated_mib": 800,
        "resident_reserved_mib": 850,
    }


def test_explicit_aed_process_reports_per_batch_cuda_metrics(tmp_path: Path, monkeypatch):
    wav = tmp_path / "normalized.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.int16), 16_000, subtype="PCM_16")

    class Asr:
        def transcribe(self, uttids, wavs):
            return [
                {
                    "uttid": uttids[0],
                    "text": "测试",
                    "confidence": 0.9,
                    "timestamp": [("测试", 0.0, 1.0)],
                }
            ]

    monkeypatch.setattr(worker_module, "_reset_cuda_peak_memory", lambda: None)
    monkeypatch.setattr(worker_module, "_synchronize_cuda", lambda: None)
    monkeypatch.setattr(worker_module, "_cuda_peak_memory_mib", lambda: 3_000)
    monkeypatch.setattr(worker_module, "_cuda_peak_reserved_mib", lambda: 3_200)
    monkeypatch.setattr(worker_module, "_cuda_memory_allocated_mib", lambda: 2_900)
    times = iter((10.0, 10.5))

    response = _explicit_aed_process(
        SimpleNamespace(asr=Asr()),
        str(wav),
        "job",
        [[{"index": 0, "start_ms": 0, "end_ms": 1000}]],
        [(0.0, 1.0)],
        clock=lambda: next(times),
    )

    assert response["batch_metrics"] == [
        {
            "batch_index": 0,
            "item_count": 1,
            "indices": [0],
            "actual_seconds": 0.5,
            "peak_allocated_mib": 3_000,
                "peak_reserved_mib": 3_200,
                "resident_allocated_mib": 2_900,
                "resident_reserved_mib": 0,
        }
    ]


def test_explicit_aed_process_emits_completed_batch_event(tmp_path: Path, monkeypatch):
    wav = tmp_path / "normalized.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.int16), 16_000, subtype="PCM_16")

    class Asr:
        def transcribe(self, uttids, wavs):
            return [{
                "uttid": uttids[0],
                "text": "测试",
                "confidence": 0.9,
                "timestamp": [("测试", 0.0, 1.0)],
            }]

    monkeypatch.setattr(worker_module, "_reset_cuda_peak_memory", lambda: None)
    monkeypatch.setattr(worker_module, "_synchronize_cuda", lambda: None)
    monkeypatch.setattr(worker_module, "_cuda_peak_memory_mib", lambda: 3_000)
    monkeypatch.setattr(worker_module, "_cuda_peak_reserved_mib", lambda: 3_200)
    monkeypatch.setattr(worker_module, "_cuda_memory_allocated_mib", lambda: 2_900)

    events = []
    _explicit_aed_process(
        SimpleNamespace(asr=Asr()),
        str(wav),
        "job",
        [[{"index": 0, "start_ms": 0, "end_ms": 1000}]],
        [(0.0, 1.0)],
        on_batch=events.append,
    )

    assert events[0]["batch_index"] == 0
    assert events[0]["aed_results"][0]["index"] == 0


def test_plan_aed_batches_uses_server_weighted_relocation():
    durations_s = [46.0, 54.0, 48.0, 55.0, 31.0, 44.0]
    cursor = 0.0
    segments = []
    for duration_s in durations_s:
        segments.append((cursor, cursor + duration_s))
        cursor += duration_s

    batches = worker_module.plan_aed_batches(segments)

    assert [[item.index for item in batch] for batch in batches] == [
        [0, 5, 4],
        [3, 1, 2],
    ]


def test_plan_aed_batches_uses_explicit_weights():
    durations_s = [46.0, 54.0, 48.0, 55.0, 31.0, 44.0]
    cursor = 0.0
    segments = []
    for duration_s in durations_s:
        segments.append((cursor, cursor + duration_s))
        cursor += duration_s

    batches = worker_module.plan_aed_batches(
        segments,
        weights=SimpleNamespace(longest_weight=0.01, padded_weight=1.0),
    )

    assert [[item.index for item in batch] for batch in batches] == [
        [0, 5, 4, 2],
        [3, 1],
    ]


def test_plan_aed_batches_limits_size_without_losing_segments():
    segments = [(float(index), float(index + 1)) for index in range(33)]

    first = worker_module.plan_aed_batches(segments)
    second = worker_module.plan_aed_batches(segments)

    assert first == second
    assert [len(batch) for batch in first] == [1, 32]
    assert sorted(item.index for batch in first for item in batch) == list(range(33))


def test_plan_aed_batches_uses_exact_millisecond_capacity():
    segments = [(876.084, 970.734), (1148.031, 1253.381)]

    batches = worker_module.plan_aed_batches(segments)

    assert [[item.index for item in batch] for batch in batches] == [[1, 0]]


def test_scheduled_process_uses_weighted_plan_without_gpu_memory_budget(
    tmp_path: Path,
    monkeypatch,
):
    wav = tmp_path / "normalized.wav"
    sf.write(wav, np.zeros(7 * 16_000, dtype=np.int16), 16_000, subtype="PCM_16")

    class Asr:
        def __init__(self):
            self.batches: list[list[str]] = []

        def transcribe(self, uttids, wavs):
            self.batches.append(list(uttids))
            results = []
            for uttid in uttids:
                start_ms = int(uttid.split("_s", 1)[1].split("_e", 1)[0])
                end_ms = int(uttid.rsplit("_e", 1)[1])
                results.append(
                    {
                        "uttid": uttid,
                        "text": str(start_ms),
                        "confidence": 0.9,
                        "timestamp": [(str(start_ms), 0.0, (end_ms - start_ms) / 1000)],
                    }
                )
            return results

    class Punc:
        def process_with_timestamp(self, timestamps, uttids):
            return [
                {
                    "uttid": uttid,
                    "punc_sentences": [
                        {"punc_text": uttid.split("_s", 1)[1], "start_s": 0, "end_s": 1}
                    ],
                }
                for uttid in uttids
            ]

    asr = Asr()
    system = SimpleNamespace(
        asr=asr,
        punc=Punc(),
        config=SimpleNamespace(
            enable_lid=False,
            enable_punc=True,
            asr_config=SimpleNamespace(return_timestamp=True),
        ),
    )
    monkeypatch.setattr(worker_module, "_cuda_free_memory_mib", lambda: None)
    raw = _scheduled_process(
        system,
        str(wav),
        "job",
        [(0.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 7.0)],
    )

    assert [batch[0].split("_s", 1)[1].split("_e", 1)[0] for batch in asr.batches] == ["0"]
    assert raw["_scheduler"]["planner"] == "weighted_duration_single_worker"
    assert "shared_budget_mib" not in raw["_scheduler"]
    assert [sentence["start_ms"] for sentence in raw["sentences"]] == [
        0,
        3000,
        4000,
        6000,
    ]


def test_scheduled_process_matches_dropped_aed_items_by_uttid(tmp_path: Path):
    wav = tmp_path / "normalized.wav"
    sf.write(wav, np.zeros(4 * 16_000, dtype=np.int16), 16_000, subtype="PCM_16")

    class Asr:
        def transcribe(self, uttids, wavs):
            # Simulate the pinned feature extractor returning one non-positional item.
            uttid = next(item for item in uttids if "_s1000_e5000" in item)
            start_ms = int(uttid.split("_s", 1)[1].split("_e", 1)[0])
            end_ms = int(uttid.rsplit("_e", 1)[1])
            return [
                {
                    "uttid": uttid,
                    "text": "保留",
                    "confidence": 0.9,
                    "timestamp": [("保", 0.0, (end_ms - start_ms) / 1000)],
                }
            ]

    class Punc:
        def process_with_timestamp(self, timestamps, uttids):
            return [
                {
                    "uttid": uttid,
                    "punc_sentences": [
                        {"punc_text": "保留。", "start_s": 0, "end_s": 1}
                    ],
                }
                for uttid in uttids
            ]

    system = SimpleNamespace(
        asr=Asr(),
        punc=Punc(),
        config=SimpleNamespace(
            enable_lid=False,
            enable_punc=True,
            asr_config=SimpleNamespace(return_timestamp=True),
        ),
    )
    raw = _scheduled_process(
        system,
        str(wav),
        "job",
        [(0.0, 1.0), (1.0, 5.0), (5.0, 6.0)],
    )

    assert [sentence["start_ms"] for sentence in raw["sentences"]] == [1000]


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
    sf.write(wav, np.zeros(30 * 16_000, dtype=np.int16), 16_000, subtype="PCM_16")

    class NeuralVad:
        def detect(self, audio):
            assert audio == str(wav)
            return {"timestamps": [(0.0, 10.0), (15.0, 20.0)]}, None

    neural = NeuralVad()
    system = ScriptedVadSystem(_official(wav))
    system.vad = neural
    target = StringIO()

    serve(
        StringIO(
            _segment_request("long", wav, [[0, 5_000], [10_000, 30_000]])
        ),
        target,
        lambda: system,
        _valid_info,
    )

    payload = json.loads(target.getvalue())
    assert payload["ok"] is True
    assert system.seen_timestamps == [(0.0, 5.0), (15.0, 20.0)]
    assert system.vad is neural


def test_worker_runs_neural_vad_once_and_intersects_long_speaker_segments(
    tmp_path: Path,
):
    wav = tmp_path / "normalized.wav"
    sf.write(wav, np.zeros(35 * 16_000, dtype=np.int16), 16_000, subtype="PCM_16")

    class NeuralVad:
        def __init__(self):
            self.calls = []

        def detect(self, audio):
            self.calls.append(audio)
            assert audio == str(wav)
            return {"timestamps": [(2.0, 14.0), (18.0, 32.0)]}, None

    neural = NeuralVad()
    system = ScriptedVadSystem(_official(wav))
    system.vad = neural
    target = StringIO()

    serve(
        StringIO(
            _segment_request(
                "global-vad",
                wav,
                [[0, 5_000], [10_000, 21_000], [25_000, 36_000]],
            )
        ),
        target,
        lambda: system,
        _valid_info,
    )

    payload = json.loads(target.getvalue())
    assert payload["ok"] is True
    assert len(neural.calls) == 1
    assert system.seen_timestamps == [
        (0.0, 5.0),
        (10.0, 14.0),
        (18.0, 21.0),
        (25.0, 32.0),
    ]


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
    assert system.seen_timestamps == [
        (float(start), float(min(start + 10, 121)))
        for start in range(0, 121, 10)
    ]
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

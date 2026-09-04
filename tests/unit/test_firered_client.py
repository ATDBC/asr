from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import pytest

from funasr_transcriber.errors import TranscriptionError
from funasr_transcriber.firered_client import FireRedWorkerClient


def _fake_worker(tmp_path: Path, mode: str = "normal") -> Path:
    script = tmp_path / f"worker_{mode}.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            import sys
            import time

            MODE = {mode!r}
            for line in sys.stdin:
                request = json.loads(line)
                if request["action"] == "shutdown":
                    print(json.dumps({{"id": request["id"], "ok": True, "shutdown": True}}), flush=True)
                    break
                if request["action"] == "status":
                    print(json.dumps({{
                        "id": request["id"], "ok": True,
                        "status": {{"models": ["vad"]}},
                        "metrics": {{"load_seconds": 1.0, "peak_gpu_bytes": 2}}
                    }}), flush=True)
                    continue
                if MODE == "crash":
                    raise SystemExit(7)
                if MODE == "timeout":
                    time.sleep(2)
                if MODE == "invalid":
                    print("not-json", flush=True)
                    continue
                if MODE == "error":
                    print(json.dumps({{"id": request["id"], "ok": False, "error": {{"type": "RuntimeError", "message": "model failed"}}}}), flush=True)
                    continue
                identifier = "wrong-id" if MODE == "wrong_id" else request["id"]
                stem = Path(request["audio_path"]).stem
                text = "一。" if stem == "one" else "二。"
                vad_segments = [[0, 500]]
                if MODE == "echo":
                    vad_segments = request.get("vad_segments_ms") or [[0, 600]]
                response = {{
                    "id": identifier,
                    "ok": True,
                    "text": text,
                    "tokens": [{{"text": text, "start_ms": 0, "end_ms": 500, "confidence": 0.9}}],
                    "sentences": [{{
                        "text": text,
                        "start_ms": 0,
                        "end_ms": 500,
                        "asr_confidence": 0.9
                    }}],
                    "vad_segments_ms": vad_segments,
                    "duration_ms": 600
                }}
                if MODE == "stream" and request["action"] == "aed_plan":
                    print(json.dumps({{
                        "id": request["id"], "ok": True,
                        "event": "aed_batch",
                        "batch_index": 0,
                        "aed_results": [{{"index": 0, "result": {{"uttid": "job_s0_e500", "timestamp": [["一", 0.0, 0.5]], "confidence": 0.9}}}}]
                    }}), flush=True)
                print(json.dumps(response, ensure_ascii=False), flush=True)
                if MODE == "duplicate":
                    print(json.dumps(response, ensure_ascii=False), flush=True)
            """
        ),
        encoding="utf-8",
    )
    return script


def _client(tmp_path: Path, mode: str = "normal", timeout: float = 1.0):
    return FireRedWorkerClient(
        worker_python=Path(sys.executable),
        model_root=tmp_path / "models",
        device="cuda:0",
        timeout_seconds=timeout,
        stderr_log_path=tmp_path / f"{mode}.log",
        worker_script=_fake_worker(tmp_path, mode),
    )


def _audio(tmp_path: Path, name: str) -> Path:
    path = (tmp_path / f"{name}.wav").resolve()
    path.write_bytes(b"fake")
    return path


def test_client_reuses_one_process_for_multiple_requests(tmp_path: Path):
    with _client(tmp_path) as client:
        first = client.transcribe(_audio(tmp_path, "one"))
        pid = client.pid
        second = client.transcribe(_audio(tmp_path, "two"))
        assert client.pid == pid

    assert first.text == "一。"
    assert second.text == "二。"
    assert first.sentences[0].asr_confidence == 0.9
    assert not hasattr(first.sentences[0], "language")


def test_client_warmup_returns_representative_shapes(tmp_path: Path):
    script = tmp_path / "warmup_worker.py"
    script.write_text(
        """
import json
import sys
for line in sys.stdin:
    request = json.loads(line)
    if request['action'] == 'warmup':
        print(json.dumps({'id': request['id'], 'ok': True,
                          'warmup_seconds': 1.0, 'warmup_excluded': True,
                          'warmup_shapes': ['short_single', 'short_batch'],
                          'resident_allocated_mib': 10,
                          'resident_reserved_mib': 20}), flush=True)
    elif request['action'] == 'shutdown':
        print(json.dumps({'id': request['id'], 'ok': True, 'shutdown': True}), flush=True)
""",
        encoding="utf-8",
    )
    client = FireRedWorkerClient(
        worker_python=Path(sys.executable),
        model_root=tmp_path / "models",
        device="cuda:0",
        timeout_seconds=1.0,
        stderr_log_path=tmp_path / "warmup.log",
        worker_script=script,
    )
    audio = _audio(tmp_path, "warmup")
    with client:
        result = client.warmup(audio)
    assert result["excluded"] is True
    assert result["shapes"] == ["short_single", "short_batch"]


def test_client_cleanup_cuda_cache_returns_worker_metrics(tmp_path: Path):
    script = tmp_path / "cleanup_worker.py"
    script.write_text(
        """
import json
import sys
for line in sys.stdin:
    request = json.loads(line)
    if request['action'] == 'cleanup_cuda_cache':
        print(json.dumps({
            'id': request['id'], 'ok': True,
            'cache_cleanup': {
                'seconds': 0.25,
                'gc_collected': 3,
                'allocated_before_mib': 2246,
                'reserved_before_mib': 6944,
                'allocated_after_mib': 2246,
                'reserved_after_mib': 2468,
            },
        }), flush=True)
    elif request['action'] == 'shutdown':
        print(json.dumps({'id': request['id'], 'ok': True, 'shutdown': True}), flush=True)
""",
        encoding="utf-8",
    )
    client = FireRedWorkerClient(
        worker_python=Path(sys.executable),
        model_root=tmp_path / "models",
        device="cuda:0",
        timeout_seconds=1.0,
        stderr_log_path=tmp_path / "cleanup.log",
        worker_script=script,
    )

    with client:
        result = client.cleanup_cuda_cache()

    assert result == {
        "seconds": 0.25,
        "gc_collected": 3,
        "allocated_before_mib": 2_246,
        "reserved_before_mib": 6_944,
        "allocated_after_mib": 2_246,
        "reserved_after_mib": 2_468,
    }


def test_client_verify_uses_same_stable_result(tmp_path: Path):
    with _client(tmp_path) as client:
        result = client.verify(_audio(tmp_path, "one"))

    assert result.text == "一。"
    assert client.last_response["text"] == "一。"


def test_client_status_reuses_existing_worker(tmp_path: Path):
    with _client(tmp_path) as client:
        first_pid = client.pid
        status = client.status()

        assert client.pid == first_pid

    assert first_pid is not None
    assert status == {"models": ["vad"]}


def test_client_does_not_pass_obsolete_batch_or_memory_flags_to_worker(tmp_path: Path):
    client = FireRedWorkerClient(
        worker_python=Path(sys.executable),
        model_root=tmp_path / "models",
        device="cuda:0",
        timeout_seconds=1.0,
        stderr_log_path=tmp_path / "worker.log",
        use_half=True,
    )

    command = client._command()
    assert "--use-half" in command
    assert not any("segment-batch" in item for item in command)
    assert not any("segment-frame" in item for item in command)
    assert not any("segment-memory" in item for item in command)


def test_client_can_start_a_punc_only_worker(tmp_path: Path):
    client = FireRedWorkerClient(
        worker_python=Path(sys.executable),
        model_root=tmp_path / "models",
        device="cuda:0",
        timeout_seconds=1.0,
        stderr_log_path=tmp_path / "punc.log",
        punc_only=True,
    )

    command = client._command()

    assert "--punc-only" in command
    assert "--disable-punc" not in command


def test_client_passes_punc_microbatch_controls(tmp_path: Path):
    client = FireRedWorkerClient(
        worker_python=Path(sys.executable),
        model_root=tmp_path / "models",
        device="cuda:0",
        timeout_seconds=1.0,
        stderr_log_path=tmp_path / "punc.log",
        punc_only=True,
        punc_token_budget=512,
        punc_max_items=32,
    )

    command = client._command()

    assert "--punc-token-budget" in command
    assert "512" in command
    assert "--punc-max-items" in command
    assert "32" in command


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("wrong_id", "response id"),
        ("invalid", "invalid JSON"),
        ("crash", "exited"),
        ("error", "model failed"),
    ),
)
def test_client_rejects_protocol_and_worker_failures(
    tmp_path: Path,
    mode: str,
    message: str,
):
    with _client(tmp_path, mode) as client:
        with pytest.raises(TranscriptionError, match=message):
            client.transcribe(_audio(tmp_path, "one"))


def test_client_times_out_and_stops_only_its_child(tmp_path: Path):
    client = _client(tmp_path, "timeout", timeout=0.05)
    client.start()
    child_pid = client.pid

    with pytest.raises(TranscriptionError, match="timed out"):
        client.transcribe(_audio(tmp_path, "one"))
    client.close()

    assert child_pid is not None
    assert client.pid is None


def test_client_detects_duplicate_response_on_following_request(tmp_path: Path):
    with _client(tmp_path, "duplicate") as client:
        client.transcribe(_audio(tmp_path, "one"))
        with pytest.raises(TranscriptionError, match="duplicate response"):
            client.transcribe(_audio(tmp_path, "two"))


def test_client_close_is_idempotent(tmp_path: Path):
    client = _client(tmp_path)
    client.start()

    client.close()
    client.close()

    assert client.pid is None


def test_client_sends_vad_segments_when_provided(tmp_path: Path):
    with _client(tmp_path, "echo") as client:
        result = client.transcribe(
            _audio(tmp_path, "one"),
            vad_segments_ms=[[0, 500], [500, 600]],
        )

    assert result.vad_segments_ms == ((0, 500), (500, 600))


def test_client_omits_vad_segments_by_default(tmp_path: Path):
    with _client(tmp_path, "echo") as client:
        result = client.transcribe(_audio(tmp_path, "one"))

    assert result.vad_segments_ms == ((0, 600),)


def test_client_streams_aed_batch_events_before_final_response(tmp_path: Path):
    events = []
    with _client(tmp_path, "stream") as client:
        response = client.run_aed_plan_stream(
            _audio(tmp_path, "one"),
            [[{"index": 0, "start_ms": 0, "end_ms": 500}]],
            [[0, 500]],
            events.append,
        )

    assert events == [{
        "id": events[0]["id"],
        "ok": True,
        "event": "aed_batch",
        "batch_index": 0,
        "aed_results": [{
            "index": 0,
            "result": {
                "uttid": "job_s0_e500",
                "timestamp": [["一", 0.0, 0.5]],
                "confidence": 0.9,
            },
        }],
    }]
    assert response["ok"] is True


def test_client_reuses_stream_worker_for_multiple_requests(tmp_path: Path):
    """A completed stream must not leave a competing stdout reader behind."""

    with _client(tmp_path, "stream", timeout=0.5) as client:
        first_events: list[dict[str, object]] = []
        second_events: list[dict[str, object]] = []
        plan = [[{"index": 0, "start_ms": 0, "end_ms": 500}]]
        vad = [[0, 500]]

        first = client.run_aed_plan_stream(
            _audio(tmp_path, "one"), plan, vad, first_events.append
        )
        second = client.run_aed_plan_stream(
            _audio(tmp_path, "two"), plan, vad, second_events.append
        )

    assert first["ok"] is True
    assert second["ok"] is True
    assert len(first_events) == 1
    assert len(second_events) == 1


def test_client_serializes_parallel_actions_and_optional_modules(tmp_path: Path):
    script = tmp_path / "parallel_worker.py"
    script.write_text(
        """
import json
for line in __import__('sys').stdin:
    request = json.loads(line)
    action = request['action']
    if action == 'shutdown':
        print(json.dumps({'id': request['id'], 'ok': True, 'shutdown': True}), flush=True)
    elif action == 'refine_vad':
        print(json.dumps({'id': request['id'], 'ok': True, 'vad_segments_ms': [[0, 500]]}), flush=True)
    elif action == 'aed_plan':
        print(json.dumps({'id': request['id'], 'ok': True, 'aed_results': [], 'vad_segments_ms': [[0, 500]], 'duration_ms': 500}), flush=True)
    elif action == 'punctuate':
        print(json.dumps({'id': request['id'], 'ok': True, 'text': '一。', 'tokens': [{'text': '一。', 'start_ms': 0, 'end_ms': 500, 'confidence': 0.9}], 'sentences': [{'text': '一。', 'start_ms': 0, 'end_ms': 500, 'asr_confidence': 0.9}], 'vad_segments_ms': [[0, 500]], 'duration_ms': 500}), flush=True)
""",
        encoding="utf-8",
    )
    client = FireRedWorkerClient(
        worker_python=Path(sys.executable), model_root=tmp_path / "models",
        device="cuda:0", timeout_seconds=1.0, stderr_log_path=tmp_path / "p.log",
        worker_script=script, enable_vad=False, enable_punc=False,
    )
    wav = _audio(tmp_path, "one")
    with client:
        assert client.refine_vad(wav, [[0, 600]]) == [[0, 500]]
        assert client.run_aed_plan(wav, [[{"index": 0, "start_ms": 0, "end_ms": 500}]])["aed_results"] == []
        assert client.punctuate(wav, [], 500, [[0, 500]]).text == "一。"
    command_client = FireRedWorkerClient(
        worker_python=Path(sys.executable), model_root=tmp_path / "models",
        device="cuda:0", timeout_seconds=1.0, stderr_log_path=tmp_path / "q.log",
        enable_vad=False, enable_punc=False,
    )
    assert "--disable-vad" in command_client._command()
    assert "--disable-punc" in command_client._command()

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from funasr_transcriber.overlap import JsonOverlapWorkerClient, OverlapWorkerError


def write_fake_worker(path: Path, sleep_seconds=0, include_environment=False):
    result_payload = "{'ok': True, 'action': request['action']}"
    if include_environment:
        result_payload = (
            "{'ok': True, 'action': request['action'], "
            "'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), "
            "'marker': os.environ.get('OVERLAP_TEST_MARKER')}"
        )
    path.write_text(
        "\n".join(
            [
                "import json, os, pathlib, sys, time",
                f"time.sleep({sleep_seconds})",
                "request_path = pathlib.Path(sys.argv[1])",
                "request = json.loads(request_path.read_text(encoding='utf-8'))",
                "result = pathlib.Path(request['result_path'])",
                f"result.write_text(json.dumps({result_payload}), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )


def test_worker_client_round_trips_json(tmp_path: Path):
    worker = tmp_path / "worker.py"
    write_fake_worker(worker)
    client = JsonOverlapWorkerClient(
        worker_python=Path(sys.executable),
        worker_script=worker,
        work_dir=tmp_path / "requests",
        timeout_seconds=2,
    )

    result = client.call("detect", {"audio_path": "sample.wav"})

    assert result == {"ok": True, "action": "detect"}


def test_worker_client_supports_model_verify(tmp_path: Path):
    worker = tmp_path / "worker.py"
    write_fake_worker(worker)
    client = JsonOverlapWorkerClient(
        worker_python=Path(sys.executable),
        worker_script=worker,
        work_dir=tmp_path / "requests",
        timeout_seconds=2,
    )

    result = client.call("verify", {})

    assert result == {"ok": True, "action": "verify"}


def test_worker_client_reports_timeout(tmp_path: Path):
    worker = tmp_path / "worker.py"
    write_fake_worker(worker, sleep_seconds=1)
    client = JsonOverlapWorkerClient(
        worker_python=Path(sys.executable),
        worker_script=worker,
        work_dir=tmp_path / "requests",
        timeout_seconds=0.05,
    )

    with pytest.raises(OverlapWorkerError, match="timed out"):
        client.call("detect", {"audio_path": "sample.wav"})


def test_worker_client_restricts_child_to_physical_gpu_zero(
    monkeypatch,
    tmp_path,
):
    worker = tmp_path / "worker.py"
    write_fake_worker(worker, include_environment=True)
    monkeypatch.setenv("OVERLAP_TEST_MARKER", "retained")
    client = JsonOverlapWorkerClient(
        worker_python=Path(sys.executable),
        worker_script=worker,
        work_dir=tmp_path / "requests",
        timeout_seconds=2,
    )

    result = client.call("verify", {})

    assert result["cuda_visible_devices"] == "0"
    assert result["marker"] == "retained"


def test_worker_client_reports_only_its_child_pid_lifetime(tmp_path: Path):
    worker = tmp_path / "worker.py"
    write_fake_worker(worker)
    client = JsonOverlapWorkerClient(
        worker_python=Path(sys.executable),
        worker_script=worker,
        work_dir=tmp_path / "requests",
        timeout_seconds=2,
    )
    started = []
    finished = []

    result = client.call(
        "detect",
        {"audio_path": "sample.wav"},
        on_started=started.append,
        on_finished=finished.append,
    )

    assert result["ok"] is True
    assert len(started) == 1
    assert finished == started

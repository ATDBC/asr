import json
from pathlib import Path
import sys

import pytest

from funasr_transcriber.overlap import JsonOverlapWorkerClient, OverlapWorkerError


def write_fake_worker(path: Path, sleep_seconds=0):
    path.write_text(
        "\n".join(
            [
                "import json, pathlib, sys, time",
                f"time.sleep({sleep_seconds})",
                "request_path = pathlib.Path(sys.argv[1])",
                "request = json.loads(request_path.read_text(encoding='utf-8'))",
                "result = pathlib.Path(request['result_path'])",
                "result.write_text(json.dumps({'ok': True, 'action': request['action']}), encoding='utf-8')",
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

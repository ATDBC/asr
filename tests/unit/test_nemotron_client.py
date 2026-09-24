from pathlib import Path

import pytest

from funasr_transcriber.nemotron_client import NemotronWorkerClient


class FakeChild:
    pid = 451
    stdin = None
    stdout = None

    def __init__(self):
        self.actions = []

    def poll(self):
        return None

    def terminate(self):
        self.actions.append("terminate")

    def wait(self, timeout=None):
        self.actions.append("wait")
        return 0


def test_close_only_terminates_owned_child(tmp_path: Path):
    client = NemotronWorkerClient(
        worker_python=tmp_path / "python",
        model_dir=tmp_path / "models",
        timeout_seconds=1,
        stderr_log_path=tmp_path / "worker.log",
    )
    child = FakeChild()
    client._process = child

    client.close()

    assert child.actions == ["terminate", "wait"]
    assert client.pid is None


def test_client_rejects_relative_audio_path(tmp_path: Path):
    client = NemotronWorkerClient(
        worker_python=tmp_path / "python",
        model_dir=tmp_path / "models",
        timeout_seconds=1,
        stderr_log_path=tmp_path / "worker.log",
    )
    with pytest.raises(ValueError, match="absolute"):
        client.diarize(Path("relative.wav"))

def test_json_exchange_with_owned_child(tmp_path: Path):
    import subprocess
    import sys

    from funasr_transcriber.errors import TranscriptionError

    code = (
        "import json,sys; "
        "line=sys.stdin.readline(); "
        "request=json.loads(line); "
        "print(json.dumps({'id':request['id'],'ok':True,"
        "'runtime':{'device':'cuda:0'}}), flush=True)"
    )
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    client = NemotronWorkerClient(
        worker_python=tmp_path / "python",
        model_dir=tmp_path / "models",
        timeout_seconds=2,
        stderr_log_path=tmp_path / "worker.log",
    )
    client._process = child
    try:
        assert client._exchange("status")["runtime"]["device"] == "cuda:0"
        child.wait(timeout=2)
        with pytest.raises(TranscriptionError, match="not running"):
            client._exchange("status")
    finally:
        client.close()


def test_exchange_timeout_terminates_only_owned_child(tmp_path: Path):
    import subprocess
    import sys

    from funasr_transcriber.errors import TranscriptionError

    child = subprocess.Popen(
        [sys.executable, "-u", "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    client = NemotronWorkerClient(
        worker_python=tmp_path / "python",
        model_dir=tmp_path / "models",
        timeout_seconds=0.1,
        stderr_log_path=tmp_path / "worker.log",
    )
    client._process = child

    with pytest.raises(TranscriptionError, match="timed out"):
        client._exchange("status")

    assert child.poll() is not None
    assert client.pid is None

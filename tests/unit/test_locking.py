import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import funasr_transcriber.locking as locking
from funasr_transcriber.locking import BusyError, TaskLock


def test_second_lock_is_rejected(tmp_path: Path):
    first = TaskLock(tmp_path, Path("/data/first.wav"))
    second = TaskLock(tmp_path, Path("/data/second.wav"))

    with first:
        with pytest.raises(BusyError, match="already running"):
            second.acquire()


def test_stale_lock_is_reclaimed(tmp_path: Path):
    lock_path = tmp_path / "transcriber.lock"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 999_999_999,
                "input": "/data/old.wav",
                "started_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with TaskLock(tmp_path, Path("/data/new.wav")):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["input"] == "/data/new.wav"

    with TaskLock(tmp_path, Path("/data/next.wav")):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["input"] == "/data/next.wav"


def test_windows_pid_probe_does_not_call_os_kill(monkeypatch):
    monkeypatch.setattr(locking.sys, "platform", "win32")
    monkeypatch.setattr(
        locking.os,
        "kill",
        lambda pid, signal: (_ for _ in ()).throw(
            AssertionError("os.kill must not be used on Windows")
        ),
    )
    monkeypatch.setattr(
        locking,
        "_windows_pid_is_alive",
        lambda pid: pid == 123,
        raising=False,
    )

    assert locking._pid_is_alive(123) is True
    assert locking._pid_is_alive(124) is False


def test_linux_lock_uses_kernel_flock(monkeypatch, tmp_path: Path):
    calls = []
    fake_fcntl = SimpleNamespace(
        LOCK_EX=1,
        LOCK_NB=2,
        LOCK_UN=4,
        flock=lambda descriptor, operation: calls.append(operation),
    )
    monkeypatch.setattr(locking.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "fcntl", fake_fcntl)

    with TaskLock(tmp_path, Path("/data/new.wav")):
        assert (tmp_path / "transcriber.lock").exists()

    assert calls == [fake_fcntl.LOCK_EX | fake_fcntl.LOCK_NB, fake_fcntl.LOCK_UN]

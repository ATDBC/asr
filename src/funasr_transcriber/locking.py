"""Single-task process lock with stale PID recovery."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from .errors import EnvironmentError as TranscriberEnvironmentError


class BusyError(TranscriberEnvironmentError):
    """Raised when another live transcription owns the lock."""


def _windows_pid_is_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        pid,
    )
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class TaskLock:
    def __init__(self, directory: Path, input_path: Path):
        self.directory = Path(directory)
        self.input_path = Path(input_path)
        self.path = self.directory / "transcriber.lock"
        self.acquired = False
        self._stream = None

    def _existing_payload(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BusyError(
                f"transcription lock is unreadable: {self.path}"
            ) from exc

    def acquire(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if sys.platform != "win32":
            self._acquire_posix()
            return
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                existing = self._existing_payload()
                pid = int(existing.get("pid") or 0)
                if _pid_is_alive(pid):
                    raise BusyError(
                        "transcription is already running: "
                        f"pid={pid}, input={existing.get('input')}"
                    )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            payload = {
                "pid": os.getpid(),
                "input": self.input_path.as_posix(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            self.acquired = True
            return
        raise BusyError("unable to acquire transcription task lock")

    def _acquire_posix(self) -> None:
        import fcntl

        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            try:
                existing = json.load(stream)
            except (OSError, json.JSONDecodeError):
                existing = {}
            stream.close()
            raise BusyError(
                "transcription is already running: "
                f"pid={existing.get('pid')}, input={existing.get('input')}"
            ) from exc
        payload = {
            "pid": os.getpid(),
            "input": self.input_path.as_posix(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        stream.seek(0)
        stream.truncate()
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
        self._stream = stream
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        if self._stream is not None:
            import fcntl

            try:
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            finally:
                self._stream.close()
                self._stream = None
                self.acquired = False
            return
        try:
            payload = self._existing_payload()
            if int(payload.get("pid") or 0) == os.getpid():
                self.path.unlink(missing_ok=True)
        finally:
            self.acquired = False

    def __enter__(self) -> "TaskLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class ServiceStartupLock:
    """Serialize service startup without claiming any input-audio path."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path = self.directory / "startup.lock"
        self.acquired = False
        self._stream = None

    def _existing_payload(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BusyError(
                f"service startup lock is unreadable: {self.path}"
            ) from exc

    def acquire(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if sys.platform != "win32":
            self._acquire_posix()
            return
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                existing = self._existing_payload()
                pid = int(existing.get("pid") or 0)
                if _pid_is_alive(pid):
                    raise BusyError(
                        f"service startup is already running: pid={pid}"
                    )
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    },
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                stream.flush()
                os.fsync(stream.fileno())
            self.acquired = True
            return
        raise BusyError("unable to acquire service startup lock")

    def _acquire_posix(self) -> None:
        import fcntl

        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise BusyError("service startup is already running") from exc
        stream.seek(0)
        stream.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            stream,
            ensure_ascii=False,
            sort_keys=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
        self._stream = stream
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        if self._stream is not None:
            import fcntl

            try:
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            finally:
                self._stream.close()
                self._stream = None
                self.acquired = False
            return
        try:
            payload = self._existing_payload()
            if int(payload.get("pid") or 0) == os.getpid():
                self.path.unlink(missing_ok=True)
        finally:
            self.acquired = False

    def __enter__(self) -> "ServiceStartupLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

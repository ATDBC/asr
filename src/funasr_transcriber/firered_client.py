"""Strict persistent client for the isolated FireRed JSON Lines worker."""

from __future__ import annotations

import json
from pathlib import Path
import selectors
import subprocess
from typing import IO, Any
import uuid

from .errors import TranscriptionError
from .firered_contract import (
    FireRedContractError,
    validate_stable_response,
)
from .schemas import AsrResult, RecognizedSentence, TimedToken


def parse_worker_response(response: dict[str, object]) -> AsrResult:
    """Convert one validated stable response to main-process schemas."""

    try:
        validate_stable_response(response)
        tokens = tuple(
            TimedToken(
                text=str(item["text"]),
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                confidence=float(item["confidence"]),
            )
            for item in response["tokens"]
        )
        sentences = tuple(
            RecognizedSentence(
                text=str(item["text"]),
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                asr_confidence=float(item["asr_confidence"]),
            )
            for item in response["sentences"]
        )
        vad_segments = tuple(
            (int(segment[0]), int(segment[1]))
            for segment in response["vad_segments_ms"]
        )
        return AsrResult(
            text=str(response["text"]),
            tokens=tokens,
            sentences=sentences,
            vad_segments_ms=vad_segments,
            duration_ms=int(response["duration_ms"]),
        )
    except (FireRedContractError, KeyError, TypeError, ValueError) as exc:
        raise TranscriptionError(
            f"invalid FireRed worker response: {exc}"
        ) from exc


class FireRedWorkerClient:
    """Own exactly one FireRed subprocess and correlate all its responses."""

    def __init__(
        self,
        worker_python: Path,
        model_root: Path,
        device: str,
        timeout_seconds: float,
        stderr_log_path: Path,
        worker_script: Path | None = None,
    ):
        self.worker_python = Path(worker_python)
        self.model_root = Path(model_root)
        self.device = str(device)
        self.timeout_seconds = float(timeout_seconds)
        self.stderr_log_path = Path(stderr_log_path)
        self.worker_script = (
            Path(worker_script) if worker_script is not None else None
        )
        self._process: subprocess.Popen[str] | None = None
        self._selector: selectors.BaseSelector | None = None
        self._stderr: IO[str] | None = None
        self._response_ids: set[str] = set()
        self.last_response: dict[str, object] = {}

    @property
    def pid(self) -> int | None:
        if self._process is None:
            return None
        return self._process.pid

    def _command(self) -> list[str]:
        if self.worker_script is not None:
            return [str(self.worker_python), str(self.worker_script)]
        return [
            str(self.worker_python),
            "-m",
            "funasr_transcriber.firered_worker",
            "--model-root",
            str(self.model_root.expanduser().resolve()),
            "--device",
            self.device,
        ]

    def start(self) -> "FireRedWorkerClient":
        if self._process is not None:
            return self
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = self.stderr_log_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )
        try:
            self._process = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
            )
        except Exception:
            self._stderr.close()
            self._stderr = None
            raise
        if self._process.stdout is None or self._process.stdin is None:
            self.close()
            raise TranscriptionError("FireRed worker pipes are unavailable")
        self._selector = selectors.DefaultSelector()
        self._selector.register(
            self._process.stdout,
            selectors.EVENT_READ,
        )
        return self

    def _read_response(self, expected_id: str) -> dict[str, object]:
        process = self._process
        selector = self._selector
        if process is None or selector is None or process.stdout is None:
            raise TranscriptionError("FireRed worker is not running")
        events = selector.select(self.timeout_seconds)
        if not events:
            return_code = process.poll()
            if return_code is not None:
                raise TranscriptionError(
                    f"FireRed worker exited with code {return_code}"
                )
            raise TranscriptionError(
                "FireRed worker timed out after "
                f"{self.timeout_seconds} seconds"
            )
        line = process.stdout.readline()
        if not line:
            try:
                return_code = process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                return_code = process.poll()
            raise TranscriptionError(
                f"FireRed worker exited with code {return_code}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TranscriptionError(
                "FireRed worker returned invalid JSON"
            ) from exc
        if not isinstance(response, dict):
            raise TranscriptionError(
                "FireRed worker response must be a JSON object"
            )
        response_id = str(response.get("id") or "")
        if response_id in self._response_ids:
            raise TranscriptionError(
                f"FireRed worker returned duplicate response id: {response_id}"
            )
        self._response_ids.add(response_id)
        if response_id != expected_id:
            raise TranscriptionError(
                "FireRed worker response id mismatch: "
                f"expected {expected_id}, got {response_id or '<empty>'}"
            )
        if response.get("ok") is not True:
            error = response.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or "worker failed")
            else:
                message = "worker failed"
            raise TranscriptionError(f"FireRed worker failed: {message}")
        return response

    def _request(
        self,
        action: str,
        audio_path: Path | None = None,
    ) -> dict[str, object]:
        self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise TranscriptionError("FireRed worker is not running")
        identifier = uuid.uuid4().hex
        request: dict[str, Any] = {"id": identifier, "action": action}
        if audio_path is not None:
            path = Path(audio_path)
            if not path.is_absolute():
                raise ValueError("FireRed audio path must be absolute")
            request["audio_path"] = str(path)
        try:
            process.stdin.write(
                json.dumps(
                    request,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            return_code = process.poll()
            raise TranscriptionError(
                f"FireRed worker exited before request: {return_code}"
            ) from exc
        response = self._read_response(identifier)
        if action != "shutdown":
            self.last_response = response
        return response

    def transcribe(self, audio_path: Path) -> AsrResult:
        return parse_worker_response(
            self._request("transcribe", Path(audio_path))
        )

    def verify(self, audio_path: Path) -> AsrResult:
        return parse_worker_response(
            self._request("verify", Path(audio_path))
        )

    def status(self) -> dict[str, object]:
        response = self._request("status")
        status = response.get("status")
        if not isinstance(status, dict):
            raise TranscriptionError("FireRed worker returned invalid status")
        return status

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    self._request("shutdown")
                except (TranscriptionError, OSError, ValueError):
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if self._selector is not None:
                self._selector.close()
                self._selector = None
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None
            self._process = None

    def __enter__(self) -> "FireRedWorkerClient":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

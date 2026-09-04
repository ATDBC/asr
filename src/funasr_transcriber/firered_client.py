"""Strict persistent client for the isolated FireRed JSON Lines worker."""

from __future__ import annotations

import json
from pathlib import Path
from queue import Empty, Queue
import selectors
import subprocess
import threading
from typing import IO, Any, Callable
import uuid

from .errors import TranscriptionError
from .punc_scheduler import DEFAULT_PUNC_MAX_ITEMS, DEFAULT_PUNC_TOKEN_BUDGET
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
    """Own one isolated FireRed AED/Punc subprocess and correlate responses.

    The subprocess also hosts the FireRedVAD helper used by the speaker-
    separation stage; that helper is not part of the worker's functional ASR
    role.
    """

    def __init__(
        self,
        worker_python: Path,
        model_root: Path,
        device: str,
        timeout_seconds: float,
        stderr_log_path: Path,
        worker_script: Path | None = None,
        # FireRed's upstream use_half flag selects BF16 when the GPU supports it.
        use_half: bool = False,
        enable_vad: bool = True,
        enable_punc: bool = True,
        punc_only: bool = False,
        punc_token_budget: int = DEFAULT_PUNC_TOKEN_BUDGET,
        punc_max_items: int = DEFAULT_PUNC_MAX_ITEMS,
        warmup_audio: Path | None = None,
    ):
        self.worker_python = Path(worker_python)
        self.model_root = Path(model_root)
        self.device = str(device)
        self.timeout_seconds = float(timeout_seconds)
        self.stderr_log_path = Path(stderr_log_path)
        self.worker_script = (
            Path(worker_script) if worker_script is not None else None
        )
        self.use_half = bool(use_half)
        self.enable_vad = bool(enable_vad)
        self.enable_punc = bool(enable_punc)
        self.punc_only = bool(punc_only)
        self.punc_token_budget = int(punc_token_budget)
        self.punc_max_items = int(punc_max_items)
        self.warmup_audio = (
            None if warmup_audio is None else Path(warmup_audio)
        )
        self._process: subprocess.Popen[str] | None = None
        self._selector: selectors.BaseSelector | None = None
        self._stderr: IO[str] | None = None
        self._response_ids: set[str] = set()
        self.last_response: dict[str, object] = {}
        self._stdout_lines: Queue[str | None] | None = None
        self._stdout_reader: threading.Thread | None = None

    @property
    def pid(self) -> int | None:
        if self._process is None:
            return None
        return self._process.pid

    def _command(self) -> list[str]:
        if self.worker_script is not None:
            return [str(self.worker_python), str(self.worker_script)]
        command = [
            str(self.worker_python),
            "-m",
            "funasr_transcriber.firered_worker",
            "--model-root",
            str(self.model_root.expanduser().resolve()),
            "--device",
            self.device,
        ]
        if self.use_half:
            command.append("--use-half")
        if self.punc_only:
            command.extend(["--punc-token-budget", str(self.punc_token_budget)])
            command.extend(["--punc-max-items", str(self.punc_max_items)])
            command.append("--punc-only")
            return command
        if not self.enable_vad:
            command.append("--disable-vad")
        if not self.enable_punc:
            command.append("--disable-punc")
        return command

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

    def _start_stdout_reader(self) -> Queue[str | None]:
        """Start one persistent reader when stream responses are needed."""

        if self._stdout_lines is not None:
            return self._stdout_lines
        process = self._process
        if process is None or process.stdout is None:
            raise TranscriptionError("FireRed worker is not running")
        lines: Queue[str | None] = Queue()

        def reader() -> None:
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        self._stdout_lines = lines
        self._stdout_reader = threading.Thread(
            target=reader,
            name=f"firered-stdout-{process.pid}",
            daemon=True,
        )
        self._stdout_reader.start()
        return lines

    def _next_stdout_line(self) -> str | None:
        lines = self._stdout_lines
        if lines is not None:
            try:
                return lines.get(timeout=self.timeout_seconds)
            except Empty:
                process = self._process
                return_code = None if process is None else process.poll()
                if return_code is not None:
                    raise TranscriptionError(
                        f"FireRed worker exited with code {return_code}"
                    )
                raise TranscriptionError(
                    "FireRed worker timed out after "
                    f"{self.timeout_seconds} seconds"
                )

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
        return process.stdout.readline() or None

    def _read_response(self, expected_id: str) -> dict[str, object]:
        process = self._process
        if process is None or process.stdout is None:
            raise TranscriptionError("FireRed worker is not running")
        line = self._next_stdout_line()
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

    def _read_stream_response(
        self,
        expected_id: str,
        on_event: Callable[[dict[str, object]], None],
    ) -> dict[str, object]:
        """Read event frames until the one final response for a request."""

        process = self._process
        if process is None or process.stdout is None:
            raise TranscriptionError("FireRed worker is not running")
        lines = self._start_stdout_reader()
        while True:
            try:
                line = lines.get(timeout=self.timeout_seconds)
            except Empty:
                return_code = process.poll()
                if return_code is not None:
                    raise TranscriptionError(
                        f"FireRed worker exited with code {return_code}"
                    )
                raise TranscriptionError(
                    "FireRed worker timed out after "
                    f"{self.timeout_seconds} seconds"
                )
            if line is None:
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
            if response_id != expected_id:
                raise TranscriptionError(
                    "FireRed worker response id mismatch: "
                    f"expected {expected_id}, got {response_id or '<empty>'}"
                )
            if response.get("event"):
                on_event(response)
                continue
            if response_id in self._response_ids:
                raise TranscriptionError(
                    f"FireRed worker returned duplicate response id: {response_id}"
                )
            self._response_ids.add(response_id)
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
        vad_segments_ms: list[list[int]] | None = None,
        extra: dict[str, Any] | None = None,
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
        if vad_segments_ms is not None:
            request["vad_segments_ms"] = [
                [int(start_ms), int(end_ms)]
                for start_ms, end_ms in vad_segments_ms
            ]
        if extra:
            request.update(extra)
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

    def refine_vad(
        self,
        audio_path: Path,
        vad_segments_ms: list[list[int]],
    ) -> list[list[int]]:
        response = self._request(
            "refine_vad",
            Path(audio_path),
            vad_segments_ms,
        )
        segments = response.get("vad_segments_ms")
        if not isinstance(segments, list):
            raise TranscriptionError("FireRed worker returned invalid refined VAD")
        return [
            [int(item[0]), int(item[1])]
            for item in segments
            if isinstance(item, (list, tuple)) and len(item) == 2
        ]

    def run_aed_plan(
        self,
        audio_path: Path,
        batch_plan: list[list[dict[str, int]]],
        vad_segments_ms: list[list[int]] | None = None,
    ) -> dict[str, object]:
        return self._request(
            "aed_plan",
            Path(audio_path),
            vad_segments_ms,
            extra={"batch_plan": batch_plan},
        )

    def run_aed_plan_stream(
        self,
        audio_path: Path,
        batch_plan: list[list[dict[str, int]]],
        vad_segments_ms: list[list[int]] | None,
        on_event: Callable[[dict[str, object]], None],
    ) -> dict[str, object]:
        """Run one immutable AED plan while delivering batch events."""

        self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise TranscriptionError("FireRed worker is not running")
        identifier = uuid.uuid4().hex
        request: dict[str, Any] = {
            "id": identifier,
            "action": "aed_plan",
            "audio_path": str(Path(audio_path).resolve()),
            "batch_plan": batch_plan,
            "stream": True,
        }
        if vad_segments_ms is not None:
            request["vad_segments_ms"] = [
                [int(start_ms), int(end_ms)]
                for start_ms, end_ms in vad_segments_ms
            ]
        try:
            process.stdin.write(
                json.dumps(request, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise TranscriptionError(
                f"FireRed worker exited before request: {process.poll()}"
            ) from exc
        response = self._read_stream_response(identifier, on_event)
        self.last_response = response
        return response

    def punctuate(
        self,
        audio_path: Path,
        aed_results: list[dict[str, object]],
        duration_ms: int,
        vad_segments_ms: list[list[int]],
    ) -> AsrResult:
        response = self._request(
            "punctuate",
            Path(audio_path),
            vad_segments_ms,
            extra={
                "aed_results": aed_results,
                "duration_ms": int(duration_ms),
            },
        )
        return parse_worker_response(response)

    def punctuate_chunk(
        self,
        audio_path: Path,
        aed_results: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        response = self._request(
            "punctuate_chunk",
            Path(audio_path),
            extra={"aed_results": aed_results},
        )
        raw = response.get("punc_results")
        if not isinstance(raw, list):
            raise TranscriptionError("Punc worker returned invalid chunk")
        return [item for item in raw if isinstance(item, dict)]

    def transcribe(
        self,
        audio_path: Path,
        vad_segments_ms: list[list[int]] | None = None,
    ) -> AsrResult:
        return parse_worker_response(
            self._request("transcribe", Path(audio_path), vad_segments_ms)
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

    def warmup(self, audio_path: Path | None = None) -> dict[str, object]:
        path = Path(audio_path) if audio_path is not None else self.warmup_audio
        if path is None:
            raise ValueError("warmup audio path is required")
        response = self._request("warmup", path)
        return {
            "seconds": float(response.get("warmup_seconds") or 0.0),
            "excluded": bool(response.get("warmup_excluded")),
            "shapes": list(response.get("warmup_shapes") or []),
            "resident_allocated_mib": response.get("resident_allocated_mib"),
            "resident_reserved_mib": response.get("resident_reserved_mib"),
        }

    def cleanup_cuda_cache(self) -> dict[str, object]:
        response = self._request("cleanup_cuda_cache")
        cleanup = response.get("cache_cleanup")
        if not isinstance(cleanup, dict):
            raise TranscriptionError(
                "FireRed worker returned invalid CUDA cache cleanup metrics"
            )
        return dict(cleanup)

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
            self._stdout_lines = None
            self._stdout_reader = None

    def __enter__(self) -> "FireRedWorkerClient":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

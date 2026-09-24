"""Client for one project-owned persistent Nemotron worker process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
from typing import Any
import uuid

from .errors import TranscriptionError
from .schemas import SpeakerSegment


class NemotronWorkerClient:
    def __init__(
        self,
        worker_python: Path,
        model_dir: Path,
        timeout_seconds: int,
        stderr_log_path: Path,
    ):
        self.worker_python = Path(worker_python)
        self.model_dir = Path(model_dir)
        self.timeout_seconds = timeout_seconds
        self.stderr_log_path = Path(stderr_log_path)
        self._process: subprocess.Popen[str] | None = None
        self._runtime: dict[str, object] = {}

    @property
    def pid(self) -> int | None:
        process = self._process
        if process is None or process.poll() is not None:
            return None
        return process.pid

    def start(self) -> "NemotronWorkerClient":
        if self.pid is not None:
            return self
        if not self.worker_python.is_file():
            raise TranscriptionError(
                f"Nemotron worker Python is missing: {self.worker_python}"
            )
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible not in {None, "0", "0,1"}:
            raise TranscriptionError(
                "Nemotron requires physical GPU0 as cuda:0"
            )
        self.stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        from .nemotron_assets import verify_nemotron_snapshot

        verify_nemotron_snapshot(self.model_dir)
        allowed_root = self.stderr_log_path.parent.parent
        with self.stderr_log_path.open("a", encoding="utf-8") as log:
            self._process = subprocess.Popen(
                [
                    str(self.worker_python),
                    "-m",
                    "funasr_transcriber.nemotron_worker",
                    "--model-dir",
                    str(self.model_dir),
                    "--allowed-root",
                    str(allowed_root),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "0",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            )
        try:
            status = self._exchange("status")
            runtime = status.get("runtime")
            if not isinstance(runtime, dict):
                raise TranscriptionError(
                    "Nemotron worker omitted runtime identity"
                )
            self._runtime = runtime
        except Exception:
            self.close()
            raise
        return self

    def _exchange(self, action: str, **payload: object) -> dict[str, Any]:
        process = self._process
        if process is None or process.poll() is not None:
            raise TranscriptionError("Nemotron worker is not running")
        if process.stdin is None or process.stdout is None:
            raise TranscriptionError("Nemotron worker pipes are unavailable")
        request_id = uuid.uuid4().hex
        try:
            process.stdin.write(
                json.dumps({"id": request_id, "action": action, **payload})
                + "\n"
            )
            process.stdin.flush()
            ready, _, _ = select.select(
                [process.stdout], [], [], self.timeout_seconds
            )
            if not ready:
                raise TimeoutError(
                    f"Nemotron worker timed out after "
                    f"{self.timeout_seconds} seconds"
                )
            line = process.stdout.readline()
        except (OSError, TimeoutError) as exc:
            self.close()
            raise TranscriptionError(str(exc)) from exc
        if not line:
            raise TranscriptionError(
                f"Nemotron worker exited unexpectedly; see "
                f"{self.stderr_log_path}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TranscriptionError(
                "Nemotron worker returned invalid JSON"
            ) from exc
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise TranscriptionError(
                "Nemotron worker response ID mismatch"
            )
        if not response.get("ok"):
            raise TranscriptionError(
                str(response.get("error") or "Nemotron worker failed")
            )
        return response

    def diarize_with_resource_metrics(
        self,
        path: Path,
    ) -> tuple[tuple[SpeakerSegment, ...], dict[str, object]]:
        path = Path(path)
        if not path.is_absolute():
            raise ValueError("Nemotron audio path must be absolute")
        if self.pid is None:
            self.start()
        response = self._exchange("diarize", audio_path=str(path))
        raw = response.get("segments")
        metrics = response.get("metrics")
        if not isinstance(raw, list) or not isinstance(metrics, dict):
            raise TranscriptionError("Nemotron worker response is incomplete")
        try:
            segments = tuple(
                SpeakerSegment(
                    speaker_id=int(item["speaker_id"]),
                    start_ms=int(item["start_ms"]),
                    end_ms=int(item["end_ms"]),
                )
                for item in raw
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError(
                "Nemotron worker returned invalid segments"
            ) from exc
        if any(segment.speaker_id > 8 for segment in segments):
            raise TranscriptionError(
                "Nemotron returned more than eight speaker channels"
            )
        return segments, dict(metrics)

    def diarize(self, path: Path) -> tuple[SpeakerSegment, ...]:
        segments, _ = self.diarize_with_resource_metrics(path)
        return segments

    def resource_inventory(self) -> dict[str, object]:
        if self.pid is None:
            self.start()
        status = self._exchange("status")
        return dict(status.get("runtime") or {})

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

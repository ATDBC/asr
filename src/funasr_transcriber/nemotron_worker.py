"""Offline, persistent Nemotron 3 speaker-activity worker."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from .errors import TranscriptionError
from .nemotron_assets import (
    NEMOTRON_MODEL_SHA256,
    NEMOTRON_REVISION,
    verify_nemotron_snapshot,
)
from .schemas import SpeakerSegment


def normalize_nemotron_segments(
    raw: list[dict[str, object]],
    duration_ms: int,
) -> tuple[SpeakerSegment, ...]:
    if not isinstance(duration_ms, int) or duration_ms <= 0:
        raise TranscriptionError("invalid normalized audio duration")
    if not isinstance(raw, list):
        raise TranscriptionError("invalid Nemotron segment response")
    segments: list[SpeakerSegment] = []
    seen: set[tuple[int, int, int]] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise TranscriptionError("invalid Nemotron segment object")
        try:
            speaker = item["Speaker"]
            start = item["Start"]
            end = item["End"]
            if isinstance(speaker, bool) or not isinstance(speaker, int):
                raise TypeError("speaker must be an integer")
            if isinstance(start, bool) or isinstance(end, bool):
                raise TypeError("timestamp must be numeric")
            start_float = float(start)
            end_float = float(end)
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError(
                "invalid Nemotron segment fields"
            ) from exc
        if not (math.isfinite(start_float) and math.isfinite(end_float)):
            raise TranscriptionError("non-finite Nemotron timestamp")
        start_ms = round(start_float * 1000)
        end_ms = round(end_float * 1000)
        if (
            speaker not in range(8)
            or start_ms < 0
            or end_ms <= start_ms
            or end_ms > duration_ms
        ):
            raise TranscriptionError("invalid Nemotron speaker segment")
        key = (speaker, start_ms, end_ms)
        if key in seen:
            raise TranscriptionError("duplicate Nemotron speaker segment")
        seen.add(key)
        segments.append(
            SpeakerSegment(
                speaker_id=speaker + 1,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    return tuple(
        sorted(
            segments,
            key=lambda segment: (
                segment.start_ms,
                segment.speaker_id,
                segment.end_ms,
            ),
        )
    )


def validate_job_wav_path(path: Path, allowed_root: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.suffix.lower() != ".wav":
        raise TranscriptionError("Nemotron audio path must be absolute WAV")
    try:
        resolved = path.resolve(strict=True)
        root = Path(allowed_root).resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise TranscriptionError(
            "Nemotron audio path is outside this project's work root"
        ) from exc
    if (
        len(relative.parts) != 2
        or not relative.parts[0].startswith("job-")
        or relative.parts[1] != "normalized.wav"
        or not resolved.is_file()
    ):
        raise TranscriptionError(
            "Nemotron audio path must be a normalized project-job WAV"
        )
    return resolved


class NemotronRuntime:
    def __init__(self, model_dir: Path):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible not in {None, "0", "0,1"}:
            raise TranscriptionError(
                "Nemotron requires physical GPU0 as cuda:0"
            )
        import torch
        from transformers import (
            AutoModelForAudioFrameClassification,
            AutoProcessor,
        )

        started = perf_counter()
        snapshot = verify_nemotron_snapshot(model_dir)
        if not torch.cuda.is_available():
            raise TranscriptionError("Nemotron cuda:0 is unavailable")
        torch.cuda.set_device(0)
        self.processor = AutoProcessor.from_pretrained(
            str(snapshot), local_files_only=True
        )
        self.model = AutoModelForAudioFrameClassification.from_pretrained(
            str(snapshot), local_files_only=True
        ).to("cuda:0")
        self.model.eval()
        self.torch = torch
        self.identity = {
            "device": "cuda:0",
            "dtype": str(self.model.dtype),
            "model_revision": NEMOTRON_REVISION,
            "checkpoint_sha256": NEMOTRON_MODEL_SHA256,
            "threshold": 0.5,
            "mode": "offline",
            "pid": os.getpid(),
            "load_seconds": perf_counter() - started,
        }

    def diarize(
        self,
        audio_path: Path,
        allowed_root: Path,
    ) -> tuple[tuple[SpeakerSegment, ...], dict[str, object]]:
        import soundfile as sf

        path = validate_job_wav_path(audio_path, allowed_root)
        audio, sample_rate = sf.read(
            str(path), dtype="float32", always_2d=True
        )
        if sample_rate != 16000 or audio.shape[1] != 1:
            raise TranscriptionError(
                "Nemotron requires 16 kHz mono normalized WAV"
            )
        duration_ms = round(audio.shape[0] * 1000 / sample_rate)
        if duration_ms <= 0:
            raise TranscriptionError("empty normalized WAV")
        started = perf_counter()
        torch = self.torch
        torch.cuda.reset_peak_memory_stats(0)
        inputs = self.processor(
            audio[:, 0], sampling_rate=sample_rate
        ).to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        raw = self.processor.extract_speaker_dict(
            logits, inputs.attention_mask, threshold=0.5
        )[0]
        segments = normalize_nemotron_segments(raw, duration_ms)
        metrics: dict[str, object] = {
            "inference_seconds": perf_counter() - started,
            "gpu_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(0) / 1024**2
            ),
            "segment_count": len(segments),
            "speaker_count": len({s.speaker_id for s in segments}),
        }
        return segments, metrics


def serve(model_dir: Path, allowed_root: Path) -> int:
    runtime = NemotronRuntime(model_dir)
    for line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            request_id = request.get("id")
            action = request.get("action")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("missing request id")
            if action == "status":
                result = {
                    "id": request_id,
                    "ok": True,
                    "runtime": runtime.identity,
                }
            elif action == "diarize":
                path = Path(str(request.get("audio_path", "")))
                segments, metrics = runtime.diarize(path, allowed_root)
                result = {
                    "id": request_id,
                    "ok": True,
                    "runtime": runtime.identity,
                    "metrics": metrics,
                    "segments": [
                        {
                            "speaker_id": s.speaker_id,
                            "start_ms": s.start_ms,
                            "end_ms": s.end_ms,
                        }
                        for s in segments
                    ],
                }
            elif action == "close":
                result = {"id": request_id, "ok": True}
            else:
                raise ValueError(f"unsupported Nemotron action: {action}")
        except Exception as exc:
            result = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if isinstance(request, dict) and request.get("action") == "close":
            return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    args = parser.parse_args()
    return serve(args.model_dir, args.allowed_root)


if __name__ == "__main__":
    raise SystemExit(main())

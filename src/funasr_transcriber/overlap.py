"""Best-effort overlap worker protocol and result acceptance."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Protocol
import uuid

import numpy as np

from .schemas import (
    AlignmentConfidence,
    AsrResult,
    AttributedToken,
    SpeakerSegment,
)


class OverlapWorkerError(RuntimeError):
    """An isolated overlap worker could not produce a valid result."""


@dataclass(frozen=True, slots=True)
class SeparationDecision:
    accepted: bool
    reason: str


def _normalized_track_text(text: str) -> str:
    return "".join(str(text).split())


def accept_separated_window(
    tracks: list[dict[str, Any]],
    min_score: float,
    min_margin: float,
) -> SeparationDecision:
    if len(tracks) != 2:
        return SeparationDecision(False, "track_count_not_two")
    texts = [_normalized_track_text(track.get("text", "")) for track in tracks]
    if any(not text for text in texts):
        return SeparationDecision(False, "empty_track_text")
    if texts[0] == texts[1]:
        return SeparationDecision(False, "duplicate_track_text")
    if any(
        "timed_token_count" in track
        and int(track["timed_token_count"]) <= 0
        for track in tracks
    ):
        return SeparationDecision(False, "missing_timed_tokens")
    if any(
        "timed_text_complete" in track
        and not bool(track["timed_text_complete"])
        for track in tracks
    ):
        return SeparationDecision(False, "incomplete_timed_text")
    speakers = [track.get("speaker_id") for track in tracks]
    if None in speakers or speakers[0] == speakers[1]:
        return SeparationDecision(False, "tracks_not_distinct")
    if any(float(track.get("score", -1.0)) < min_score for track in tracks):
        return SeparationDecision(False, "speaker_score_too_low")
    if any(float(track.get("margin", -1.0)) < min_margin for track in tracks):
        return SeparationDecision(False, "speaker_margin_too_low")
    return SeparationDecision(True, "accepted")


class JsonOverlapWorkerClient:
    def __init__(
        self,
        worker_python: Path,
        worker_script: Path,
        work_dir: Path,
        timeout_seconds: float,
    ):
        self.worker_python = Path(worker_python)
        self.worker_script = Path(worker_script)
        self.work_dir = Path(work_dir)
        self.timeout_seconds = timeout_seconds

    def call(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if action not in {"detect", "separate", "download", "verify"}:
            raise ValueError(f"unsupported overlap action: {action}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex
        request_path = self.work_dir / f"{request_id}.request.json"
        result_path = self.work_dir / f"{request_id}.result.json"
        request = {
            **payload,
            "action": action,
            "result_path": str(result_path.resolve()),
        }
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        command = [
            str(self.worker_python),
            str(self.worker_script),
            str(request_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OverlapWorkerError(
                f"overlap worker timed out after {self.timeout_seconds} seconds"
            ) from exc
        if not result_path.is_file():
            raise OverlapWorkerError(
                "overlap worker did not write a result file: "
                f"exit={completed.returncode}, stderr={completed.stderr.strip()}"
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OverlapWorkerError(
                "overlap worker wrote invalid JSON"
            ) from exc
        if completed.returncode != 0 or not result.get("ok"):
            raise OverlapWorkerError(
                str(result.get("error") or "overlap worker failed")
            )
        return result


@dataclass(frozen=True, slots=True)
class OverlapEnhancement:
    tokens: tuple[AttributedToken, ...]
    warnings: tuple[str, ...]
    modified: bool


class WorkerClient(Protocol):
    def call(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run an isolated worker action."""


class TrackAsrEngine(Protocol):
    def transcribe(self, path: Path) -> AsrResult:
        """Transcribe one separated track."""


class TrackDiarizer(Protocol):
    def speaker_centroids(
        self,
        path: Path,
        segments: tuple[SpeakerSegment, ...],
    ) -> dict[int, np.ndarray]:
        """Return global speaker centroids."""

    def embedding_for_audio(self, path: Path) -> np.ndarray:
        """Return one normalized embedding for a separated track."""


@dataclass(frozen=True, slots=True)
class PreparedTrack:
    path: Path
    speaker_id: int
    score: float
    margin: float
    rms: float


@dataclass(frozen=True, slots=True)
class PreparedWindow:
    id: str
    start_ms: int
    end_ms: int
    tracks: tuple[PreparedTrack, ...]


@dataclass(frozen=True, slots=True)
class OverlapPreparation:
    windows: tuple[PreparedWindow, ...]
    warnings: tuple[str, ...]


class SubprocessOverlapPreparer:
    def __init__(
        self,
        worker: WorkerClient,
        diarizer: TrackDiarizer,
        min_score: float,
        min_margin: float,
        min_rms: float,
        detector_model: str = "pyannote/speaker-diarization-3.1",
        hf_token_env: str = "HF_TOKEN",
        device: str = "cuda:0",
        output_dir: Path | None = None,
        cache_dir: Path | None = None,
    ):
        self.worker = worker
        self.diarizer = diarizer
        self.min_score = min_score
        self.min_margin = min_margin
        self.min_rms = min_rms
        self.detector_model = detector_model
        self.hf_token_env = hf_token_env
        self.device = device
        self.output_dir = output_dir or Path("overlap-output")
        self.cache_dir = cache_dir

    @staticmethod
    def _match_speaker(
        embedding: np.ndarray,
        centroids: dict[int, np.ndarray],
    ) -> tuple[int | None, float, float]:
        if not centroids:
            return None, -1.0, -1.0
        normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
        scores = sorted(
            (
                (
                    speaker,
                    float(
                        np.dot(
                            normalized,
                            centroid
                            / (np.linalg.norm(centroid) + 1e-8),
                        )
                    ),
                )
                for speaker, centroid in centroids.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        best_speaker, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else -1.0
        return best_speaker, best_score, best_score - second_score

    def prepare(
        self,
        audio_path: Path,
        speaker_segments: tuple[SpeakerSegment, ...],
    ) -> OverlapPreparation:
        try:
            detected = self.worker.call(
                "detect",
                {
                    "audio_path": str(Path(audio_path).resolve()),
                    "model_id": self.detector_model,
                    "hf_token_env": self.hf_token_env,
                    "device": self.device,
                    "cache_dir": (
                        str(self.cache_dir.resolve())
                        if self.cache_dir is not None
                        else None
                    ),
                },
            )
        except Exception as exc:
            return OverlapPreparation(
                (),
                (f"overlap detection unavailable: {exc}",),
            )
        windows = detected.get("windows") or []
        if not windows:
            return OverlapPreparation((), ())
        try:
            separated = self.worker.call(
                "separate",
                {
                    "audio_path": str(Path(audio_path).resolve()),
                    "output_dir": str(self.output_dir.resolve()),
                    "windows": windows,
                    "model": "MossFormer2_SS_16K",
                    "cache_dir": (
                        str(self.cache_dir.resolve())
                        if self.cache_dir is not None
                        else None
                    ),
                },
            )
            centroids = self.diarizer.speaker_centroids(
                audio_path,
                speaker_segments,
            )
        except Exception as exc:
            return OverlapPreparation(
                (),
                (f"overlap separation unavailable: {exc}",),
            )

        warnings: list[str] = []
        prepared_windows: list[PreparedWindow] = []
        for window in separated.get("windows") or []:
            start_ms = int(window["start_ms"])
            end_ms = int(window["end_ms"])
            prepared_tracks: list[PreparedTrack] = []
            for track in window.get("tracks") or []:
                track_path = Path(track["path"])
                rms = float(track.get("rms", 0.0))
                if rms < self.min_rms:
                    continue
                try:
                    embedding = self.diarizer.embedding_for_audio(track_path)
                    speaker, score, margin = self._match_speaker(
                        embedding,
                        centroids,
                    )
                except Exception:
                    continue
                if speaker is None:
                    continue
                prepared_tracks.append(
                    PreparedTrack(
                        path=track_path,
                        speaker_id=speaker,
                        score=score,
                        margin=margin,
                        rms=rms,
                    )
                )
            prepared_windows.append(
                PreparedWindow(
                    id=str(window.get("id") or ""),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    tracks=tuple(prepared_tracks),
                )
            )
        return OverlapPreparation(
            windows=tuple(prepared_windows),
            warnings=tuple(warnings),
        )


def apply_prepared_overlap(
    preparation: OverlapPreparation,
    base_tokens: tuple[AttributedToken, ...],
    asr_engine: TrackAsrEngine,
    min_score: float,
    min_margin: float,
) -> OverlapEnhancement:
    output = list(base_tokens)
    warnings = list(preparation.warnings)
    modified = False
    for window in preparation.windows:
        assessments: list[dict[str, Any]] = []
        candidate_tokens: list[AttributedToken] = []
        for track in window.tracks:
            try:
                asr_result = asr_engine.transcribe(track.path)
            except Exception:
                continue
            assessments.append(
                {
                    "speaker_id": track.speaker_id,
                    "score": track.score,
                    "margin": track.margin,
                    "text": asr_result.text,
                    "timed_token_count": len(asr_result.tokens),
                    "timed_text_complete": (
                        not asr_result.untimed_sentences
                        and _normalized_track_text(
                            "".join(token.text for token in asr_result.tokens)
                        )
                        == _normalized_track_text(asr_result.text)
                    ),
                }
            )
            for token in asr_result.tokens:
                global_start = min(
                    window.end_ms,
                    window.start_ms + token.start_ms,
                )
                global_end = min(
                    window.end_ms,
                    window.start_ms + token.end_ms,
                )
                if global_end <= global_start:
                    continue
                candidate_tokens.append(
                    AttributedToken(
                        text=token.text,
                        start_ms=global_start,
                        end_ms=global_end,
                        speaker_id=track.speaker_id,
                        confidence=AlignmentConfidence.HIGH,
                        source="separated",
                        overlap=True,
                    )
                )
        decision = accept_separated_window(
            assessments,
            min_score,
            min_margin,
        )
        if not decision.accepted:
            warnings.append(
                f"overlap window {window.id} kept original: "
                f"{decision.reason}"
            )
            continue
        output = [
            token
            for token in output
            if not (
                window.start_ms
                <= (token.start_ms + token.end_ms) / 2
                < window.end_ms
            )
        ]
        output.extend(candidate_tokens)
        modified = True
    output.sort(
        key=lambda token: (
            token.start_ms,
            token.speaker_id,
            token.end_ms,
        )
    )
    return OverlapEnhancement(tuple(output), tuple(warnings), modified)


class SubprocessOverlapEnhancer:
    """Compatibility facade until the CLI adopts explicit two-phase lifetime."""

    def __init__(
        self,
        worker: WorkerClient,
        asr_engine: TrackAsrEngine,
        diarizer: TrackDiarizer,
        min_score: float,
        min_margin: float,
        min_rms: float,
        detector_model: str = "pyannote/speaker-diarization-3.1",
        hf_token_env: str = "HF_TOKEN",
        device: str = "cuda:0",
        output_dir: Path | None = None,
        cache_dir: Path | None = None,
    ):
        self.asr_engine = asr_engine
        self.min_score = min_score
        self.min_margin = min_margin
        self.preparer = SubprocessOverlapPreparer(
            worker=worker,
            diarizer=diarizer,
            min_score=min_score,
            min_margin=min_margin,
            min_rms=min_rms,
            detector_model=detector_model,
            hf_token_env=hf_token_env,
            device=device,
            output_dir=output_dir,
            cache_dir=cache_dir,
        )

    def enhance(
        self,
        audio_path: Path,
        base_tokens: tuple[AttributedToken, ...],
        speaker_segments: tuple[SpeakerSegment, ...],
    ) -> OverlapEnhancement:
        preparation = self.preparer.prepare(audio_path, speaker_segments)
        return apply_prepared_overlap(
            preparation,
            base_tokens,
            self.asr_engine,
            self.min_score,
            self.min_margin,
        )

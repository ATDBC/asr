"""Independent TOML configuration with explicit override precedence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any, Mapping


ENV_PREFIX = "FUNASR_TRANSCRIBER_"

DEFAULTS: dict[str, dict[str, Any]] = {
    "general": {
        "language": "zh",
        "device": "cuda:0",
        "max_audio_seconds": 7200,
        "random_seed": 0,
    },
    "paths": {
        "model_dir": "/opt/funasr-transcriber/models",
        "work_dir": "/var/tmp/funasr-transcriber",
    },
    "asr": {
        "worker_python": "/opt/funasr-transcriber/.venv-asr/bin/python",
        "timeout_seconds": 1800,
        "use_half": True,
        "aed_worker_count": 2,
        "segment_batch_size": 8,
        "segment_frame_budget": 8000,
        "segment_memory_limit_mib": 10000,
        "segment_memory_safety_mib": 512,
        "gpu_safety_margin_mib": 1536,
    },
    "diarization": {
        "speaker_count": "auto",
        "min_speakers": 1,
        "max_speakers": 10,
        "chunk_seconds": 1.0,
        "chunk_step_seconds": 0.5,
        "merge_cosine": 0.23,
        "merge_min_fraction": 0.6,
        "merge_relative_ratio": 0.55,
    },
    "vad": {
        "speech_noise_thres": 0.45,
        "sil_to_speech_time_thres": 120,
        "speech_to_sil_time_thres": 180,
        "max_end_silence_time": 1000,
    },
    "alignment": {
        "high_coverage_ratio": 0.70,
        "medium_coverage_ratio": 0.50,
        "nearest_gap_ms": 300,
    },
    "turns": {"merge_gap_seconds": 1.0},
    "service": {
        "queue_limit": 16,
        "startup_timeout_seconds": 180,
        "profile_sampling_interval_ms": 200,
        "minimum_total_mib": 23 * 1024,
        "minimum_free_mib": 13 * 1024,
    },
    "overlap": {
        "enabled": False,
        "detector_model": "pyannote/speaker-diarization-3.1",
        "hf_token_env": "HF_TOKEN",
        "worker_python": "/opt/funasr-transcriber/.venv-overlap/bin/python",
        "timeout_seconds": 1200,
        "speaker_match_min_score": 0.05,
        "speaker_match_min_margin": 0.03,
        "track_min_rms": 0.0005,
    },
}


@dataclass(frozen=True, slots=True)
class GeneralConfig:
    language: str
    device: str
    max_audio_seconds: int
    random_seed: int


@dataclass(frozen=True, slots=True)
class PathConfig:
    model_dir: Path
    work_dir: Path


@dataclass(frozen=True, slots=True)
class AsrConfig:
    worker_python: Path
    timeout_seconds: int
    use_half: bool
    aed_worker_count: int
    segment_batch_size: int
    segment_frame_budget: int
    segment_memory_limit_mib: int
    segment_memory_safety_mib: int
    gpu_safety_margin_mib: int


@dataclass(frozen=True, slots=True)
class DiarizationConfig:
    speaker_count: int | None
    min_speakers: int
    max_speakers: int
    chunk_seconds: float
    chunk_step_seconds: float
    merge_cosine: float
    merge_min_fraction: float
    merge_relative_ratio: float


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    high_coverage_ratio: float
    medium_coverage_ratio: float
    nearest_gap_ms: int


@dataclass(frozen=True, slots=True)
class VadConfig:
    speech_noise_thres: float
    sil_to_speech_time_thres: int
    speech_to_sil_time_thres: int
    max_end_silence_time: int


@dataclass(frozen=True, slots=True)
class TurnConfig:
    merge_gap_seconds: float


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    queue_limit: int
    startup_timeout_seconds: int
    profile_sampling_interval_ms: int
    minimum_total_mib: int
    minimum_free_mib: int


@dataclass(frozen=True, slots=True)
class OverlapConfig:
    enabled: bool
    detector_model: str
    hf_token_env: str
    worker_python: Path
    timeout_seconds: int
    speaker_match_min_score: float
    speaker_match_min_margin: float
    track_min_rms: float


@dataclass(frozen=True, slots=True)
class AppConfig:
    general: GeneralConfig
    paths: PathConfig
    asr: AsrConfig
    diarization: DiarizationConfig
    vad: VadConfig
    alignment: AlignmentConfig
    turns: TurnConfig
    service: ServiceConfig
    overlap: OverlapConfig


def _merge_known(target: dict[str, dict[str, Any]], incoming: Mapping[str, Any]) -> None:
    for section, values in incoming.items():
        if section not in target or not isinstance(values, Mapping):
            raise ValueError(f"unknown configuration section: {section}")
        for key, value in values.items():
            if key not in target[section]:
                raise ValueError(f"unknown configuration key: {section}.{key}")
            target[section][key] = value


def _coerce(raw: Any, sample: Any, dotted_key: str) -> Any:
    if dotted_key == "diarization.speaker_count":
        if raw is None or str(raw).strip().lower() in {"auto", "none", "null", ""}:
            return "auto"
        return int(raw)
    if not isinstance(raw, str):
        return raw
    if isinstance(sample, bool):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{dotted_key} must be a boolean")
    if isinstance(sample, int) and not isinstance(sample, bool):
        return int(raw)
    if isinstance(sample, float):
        return float(raw)
    return raw


def _apply_dotted(
    values: dict[str, dict[str, Any]],
    dotted_key: str,
    raw: Any,
) -> None:
    try:
        section, key = dotted_key.lower().split(".", 1)
        sample = values[section][key]
    except (ValueError, KeyError) as exc:
        raise ValueError(f"unknown configuration key: {dotted_key}") from exc
    values[section][key] = _coerce(raw, sample, f"{section}.{key}")


def _environment_overrides() -> dict[str, str]:
    overrides: dict[str, str] = {}
    for name, value in os.environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        suffix = name[len(ENV_PREFIX):]
        if "__" not in suffix:
            continue
        section, key = suffix.split("__", 1)
        overrides[f"{section.lower()}.{key.lower()}"] = value
    return overrides


def _build(values: dict[str, dict[str, Any]]) -> AppConfig:
    speaker_count_raw = values["diarization"]["speaker_count"]
    speaker_count = (
        None
        if str(speaker_count_raw).strip().lower() == "auto"
        else int(speaker_count_raw)
    )
    config = AppConfig(
        general=GeneralConfig(**values["general"]),
        paths=PathConfig(
            model_dir=Path(values["paths"]["model_dir"]).expanduser(),
            work_dir=Path(values["paths"]["work_dir"]).expanduser(),
        ),
        asr=AsrConfig(
            worker_python=Path(
                values["asr"]["worker_python"]
            ).expanduser(),
            timeout_seconds=int(values["asr"]["timeout_seconds"]),
            use_half=bool(values["asr"]["use_half"]),
            aed_worker_count=int(values["asr"]["aed_worker_count"]),
            segment_batch_size=int(values["asr"]["segment_batch_size"]),
            segment_frame_budget=int(values["asr"]["segment_frame_budget"]),
            segment_memory_limit_mib=int(
                values["asr"]["segment_memory_limit_mib"]
            ),
            segment_memory_safety_mib=int(
                values["asr"]["segment_memory_safety_mib"]
            ),
            gpu_safety_margin_mib=int(values["asr"]["gpu_safety_margin_mib"]),
        ),
        diarization=DiarizationConfig(
            speaker_count=speaker_count,
            min_speakers=int(values["diarization"]["min_speakers"]),
            max_speakers=int(values["diarization"]["max_speakers"]),
            chunk_seconds=float(values["diarization"]["chunk_seconds"]),
            chunk_step_seconds=float(values["diarization"]["chunk_step_seconds"]),
            merge_cosine=float(values["diarization"]["merge_cosine"]),
            merge_min_fraction=float(values["diarization"]["merge_min_fraction"]),
            merge_relative_ratio=float(
                values["diarization"]["merge_relative_ratio"]
            ),
        ),
        vad=VadConfig(
            speech_noise_thres=float(values["vad"]["speech_noise_thres"]),
            sil_to_speech_time_thres=int(
                values["vad"]["sil_to_speech_time_thres"]
            ),
            speech_to_sil_time_thres=int(
                values["vad"]["speech_to_sil_time_thres"]
            ),
            max_end_silence_time=int(
                values["vad"]["max_end_silence_time"]
            ),
        ),
        alignment=AlignmentConfig(**values["alignment"]),
        turns=TurnConfig(**values["turns"]),
        service=ServiceConfig(
            queue_limit=int(values["service"]["queue_limit"]),
            startup_timeout_seconds=int(
                values["service"]["startup_timeout_seconds"]
            ),
            profile_sampling_interval_ms=int(
                values["service"]["profile_sampling_interval_ms"]
            ),
            minimum_total_mib=int(values["service"]["minimum_total_mib"]),
            minimum_free_mib=int(values["service"]["minimum_free_mib"]),
        ),
        overlap=OverlapConfig(
            enabled=bool(values["overlap"]["enabled"]),
            detector_model=str(values["overlap"]["detector_model"]),
            hf_token_env=str(values["overlap"]["hf_token_env"]),
            worker_python=Path(values["overlap"]["worker_python"]).expanduser(),
            timeout_seconds=int(values["overlap"]["timeout_seconds"]),
            speaker_match_min_score=float(
                values["overlap"]["speaker_match_min_score"]
            ),
            speaker_match_min_margin=float(
                values["overlap"]["speaker_match_min_margin"]
            ),
            track_min_rms=float(values["overlap"]["track_min_rms"]),
        ),
    )
    return validate(config)


def validate(config: AppConfig) -> AppConfig:
    if config.general.max_audio_seconds != 7200:
        raise ValueError("max_audio_seconds must be 7200 in version 1")
    if config.asr.timeout_seconds <= 0:
        raise ValueError("ASR timeout_seconds must be positive")
    if config.asr.segment_batch_size <= 0:
        raise ValueError("ASR segment_batch_size must be positive")
    if config.asr.aed_worker_count not in (1, 2):
        raise ValueError("ASR aed_worker_count must be 1 or 2")
    if config.asr.segment_frame_budget <= 0:
        raise ValueError("ASR segment_frame_budget must be positive")
    if config.asr.segment_memory_limit_mib <= 0:
        raise ValueError("ASR segment_memory_limit_mib must be positive")
    if config.asr.segment_memory_safety_mib < 0:
        raise ValueError("ASR segment_memory_safety_mib must not be negative")
    if config.asr.gpu_safety_margin_mib <= 0:
        raise ValueError("ASR gpu_safety_margin_mib must be positive")
    diarization = config.diarization
    if not 1 <= diarization.min_speakers <= diarization.max_speakers <= 10:
        raise ValueError("min_speakers/max_speakers must stay within 1..10")
    if diarization.speaker_count is not None and not (
        diarization.min_speakers
        <= diarization.speaker_count
        <= diarization.max_speakers
    ):
        raise ValueError("speaker_count must stay within min_speakers/max_speakers")
    if not 0.0 < diarization.merge_cosine <= 1.0:
        raise ValueError("merge_cosine must stay within 0..1")
    if not 0.0 < diarization.merge_min_fraction <= 1.0:
        raise ValueError("merge_min_fraction must stay within 0..1")
    if not 0.0 < diarization.merge_relative_ratio <= 1.0:
        raise ValueError("merge_relative_ratio must stay within 0..1")
    if config.turns.merge_gap_seconds != 1.0:
        raise ValueError("merge_gap_seconds must be 1.0 in version 1")
    service = config.service
    if service.queue_limit <= 0:
        raise ValueError("service queue_limit must be positive")
    if service.startup_timeout_seconds <= 0:
        raise ValueError("service startup_timeout_seconds must be positive")
    if not 50 <= service.profile_sampling_interval_ms <= 5000:
        raise ValueError(
            "service profile_sampling_interval_ms must stay within 50..5000"
        )
    if service.minimum_total_mib < 23 * 1024:
        raise ValueError("service minimum_total_mib must be at least 23552")
    if not 0 < service.minimum_free_mib <= service.minimum_total_mib:
        raise ValueError("service minimum_free_mib is invalid")
    if not (
        0.0
        < config.alignment.medium_coverage_ratio
        <= config.alignment.high_coverage_ratio
        <= 1.0
    ):
        raise ValueError("alignment coverage ratios are invalid")
    return config


def load_config(
    path: Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    values = deepcopy(DEFAULTS)
    if path is not None:
        with Path(path).open("rb") as stream:
            _merge_known(values, tomllib.load(stream))
    for dotted_key, raw in _environment_overrides().items():
        _apply_dotted(values, dotted_key, raw)
    for dotted_key, raw in (cli_overrides or {}).items():
        _apply_dotted(values, dotted_key, raw)
    return _build(values)

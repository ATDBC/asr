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
        "aed_scheduler": "weighted_duration",
        "aed_worker_count": 1,
        "aed_weight_artifact": "",
        "gpu_safety_margin_mib": 1536,
        "punc_token_budget": 256,
        "punc_max_items": 16,
    },
    "diarization": {
        "backend": "3d_speaker",
        "speaker_count": "auto",
        "min_speakers": 1,
        "max_speakers": 10,
        "chunk_seconds": 1.5,
        "chunk_step_seconds": 0.75,
        "merge_cosine": 0.23,
        "merge_min_fraction": 0.6,
        "merge_relative_ratio": 0.55,
        "major_cluster_min_fraction": 0.10,
        "major_cluster_merge_threshold": 0.88,
    },
    "nemotron": {
        "worker_python": "/opt/funasr-transcriber/.venv-nemotron/bin/python",
        "timeout_seconds": 1800,
        "activity_threshold": 0.5,
        "clause_reassembly": False,
        "min_interruption_ratio": 0.50,
        "min_clause_ms": 400,
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
    "diagnostics": {
        "level": "basic",
        "sampling_interval_ms": 200,
        "include_model_inventory": False,
        "write_batch_trace": False,
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
    "llm": {
        "enabled": False,
        "provider": "deepseek",
        "api_key": "",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4.1-flash",
        "enable_thinking": False,
        "temperature": 0.1,
        "timeout_seconds": 60,
        "pinyin_gate_threshold": 0.5,
        # Compatibility-only: accepted from older TOML files but ignored.
        "max_context_turns": None,
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
    aed_scheduler: str
    aed_worker_count: int
    aed_weight_artifact: Path | None
    gpu_safety_margin_mib: int
    punc_token_budget: int
    punc_max_items: int


@dataclass(frozen=True, slots=True)
class DiarizationConfig:
    backend: str
    speaker_count: int | None
    min_speakers: int
    max_speakers: int
    chunk_seconds: float
    chunk_step_seconds: float
    merge_cosine: float
    merge_min_fraction: float
    merge_relative_ratio: float
    major_cluster_min_fraction: float
    major_cluster_merge_threshold: float


@dataclass(frozen=True, slots=True)
class NemotronConfig:
    worker_python: Path
    timeout_seconds: int
    activity_threshold: float
    clause_reassembly: bool = False
    min_interruption_ratio: float = 0.50
    min_clause_ms: int = 400


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
class DiagnosticsConfig:
    level: str
    sampling_interval_ms: int
    include_model_inventory: bool
    write_batch_trace: bool


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
class LLMConfig:
    enabled: bool
    provider: str
    api_key: str
    api_key_env: str
    base_url: str
    model: str
    enable_thinking: bool
    temperature: float
    timeout_seconds: int
    pinyin_gate_threshold: float

    def resolve_api_key(self) -> str:
        """Resolve API key from direct config or environment variable."""
        if self.api_key.strip():
            return self.api_key.strip()
        env_name = self.api_key_env.strip()
        is_mimo = self.provider == "mimo" or "mimo" in self.model.lower()
        is_dashscope = self.provider in {"dashscope", "qwen"} or "qwen" in self.model.lower()
        is_deepseek = self.provider == "deepseek" or "deepseek" in self.model.lower()

        standard_env_vars = {
            "DEEPSEEK_API_KEY",
            "MIMO_API_KEY",
            "DASHSCOPE_API_KEY",
            "QWEN_API_KEY",
            "LLM_API_KEY",
        }

        # If user explicitly configured an env_name that is not a standard default,
        # strictly respect that custom env var without falling back.
        if env_name and env_name not in standard_env_vars:
            return os.environ.get(env_name, "").strip()

        # If standard env var is configured and present in environment:
        if env_name and env_name in os.environ and os.environ[env_name].strip():
            # If provider does not match the default env var name, prefer the matching provider key if available
            if is_mimo and env_name != "MIMO_API_KEY" and "MIMO_API_KEY" in os.environ and os.environ["MIMO_API_KEY"].strip():
                return os.environ["MIMO_API_KEY"].strip()
            if is_dashscope and env_name not in {"DASHSCOPE_API_KEY", "QWEN_API_KEY"}:
                for k in ("DASHSCOPE_API_KEY", "QWEN_API_KEY"):
                    if k in os.environ and os.environ[k].strip():
                        return os.environ[k].strip()
            return os.environ[env_name].strip()

        # Fallback to provider-specific default env vars
        if is_mimo:
            if "MIMO_API_KEY" in os.environ and os.environ["MIMO_API_KEY"].strip():
                return os.environ["MIMO_API_KEY"].strip()
        if is_dashscope:
            for k in ("DASHSCOPE_API_KEY", "QWEN_API_KEY"):
                if k in os.environ and os.environ[k].strip():
                    return os.environ[k].strip()
        if is_deepseek:
            if "DEEPSEEK_API_KEY" in os.environ and os.environ["DEEPSEEK_API_KEY"].strip():
                return os.environ["DEEPSEEK_API_KEY"].strip()
        if "LLM_API_KEY" in os.environ and os.environ["LLM_API_KEY"].strip():
            return os.environ["LLM_API_KEY"].strip()
        return ""


@dataclass(frozen=True, slots=True)
class AppConfig:
    general: GeneralConfig
    paths: PathConfig
    asr: AsrConfig
    diarization: DiarizationConfig
    nemotron: NemotronConfig
    vad: VadConfig
    alignment: AlignmentConfig
    turns: TurnConfig
    service: ServiceConfig
    diagnostics: DiagnosticsConfig
    overlap: OverlapConfig
    llm: LLMConfig


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
            aed_scheduler=str(
                values["asr"]["aed_scheduler"]
            ).strip().lower(),
            aed_worker_count=int(values["asr"]["aed_worker_count"]),
            aed_weight_artifact=(
                Path(str(values["asr"]["aed_weight_artifact"])).expanduser()
                if str(values["asr"]["aed_weight_artifact"]).strip()
                else None
            ),
            gpu_safety_margin_mib=int(values["asr"]["gpu_safety_margin_mib"]),
            punc_token_budget=int(values["asr"]["punc_token_budget"]),
            punc_max_items=int(values["asr"]["punc_max_items"]),
        ),
        diarization=DiarizationConfig(
            backend=str(values["diarization"]["backend"]).strip().lower(),
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
            major_cluster_min_fraction=float(
                values["diarization"]["major_cluster_min_fraction"]
            ),
            major_cluster_merge_threshold=float(
                values["diarization"]["major_cluster_merge_threshold"]
            ),
        ),
        nemotron=NemotronConfig(
            worker_python=Path(values["nemotron"]["worker_python"]).expanduser(),
            timeout_seconds=int(values["nemotron"]["timeout_seconds"]),
            activity_threshold=float(values["nemotron"]["activity_threshold"]),
            clause_reassembly=bool(values["nemotron"].get("clause_reassembly", False)),
            min_interruption_ratio=float(
                values["nemotron"].get("min_interruption_ratio", 0.50)
            ),
            min_clause_ms=int(values["nemotron"].get("min_clause_ms", 400)),
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
        diagnostics=DiagnosticsConfig(
            level=str(values["diagnostics"]["level"]).lower(),
            sampling_interval_ms=int(
                values["diagnostics"]["sampling_interval_ms"]
            ),
            include_model_inventory=bool(
                values["diagnostics"]["include_model_inventory"]
            ),
            write_batch_trace=bool(values["diagnostics"]["write_batch_trace"]),
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
        llm=LLMConfig(
            enabled=bool(values["llm"]["enabled"]),
            provider=str(values["llm"]["provider"]).strip().lower(),
            api_key=str(values["llm"]["api_key"]).strip(),
            api_key_env=str(values["llm"]["api_key_env"]).strip(),
            base_url=str(values["llm"]["base_url"]).strip(),
            model=str(values["llm"]["model"]).strip(),
            enable_thinking=bool(values["llm"]["enable_thinking"]),
            temperature=float(values["llm"]["temperature"]),
            timeout_seconds=int(values["llm"]["timeout_seconds"]),
            pinyin_gate_threshold=float(values["llm"]["pinyin_gate_threshold"]),
        ),
    )
    return validate(config)


def validate(config: AppConfig) -> AppConfig:
    if config.general.max_audio_seconds != 7200:
        raise ValueError("max_audio_seconds must be 7200 in version 1")
    if config.asr.timeout_seconds <= 0:
        raise ValueError("ASR timeout_seconds must be positive")
    if config.asr.aed_scheduler not in {
        "weighted_duration",
        "memory_first",
    }:
        raise ValueError(
            "ASR aed_scheduler must be weighted_duration or memory_first"
        )
    if config.asr.aed_worker_count not in (1, 2):
        raise ValueError("ASR aed_worker_count must be 1 or 2")
    if (
        config.asr.aed_scheduler == "memory_first"
        and config.asr.aed_worker_count != 2
    ):
        raise ValueError("ASR memory_first aed_scheduler requires 2 AED workers")
    if config.asr.gpu_safety_margin_mib <= 0:
        raise ValueError("ASR gpu_safety_margin_mib must be positive")
    if config.asr.punc_token_budget <= 0:
        raise ValueError("ASR punc_token_budget must be positive")
    if config.asr.punc_max_items <= 0:
        raise ValueError("ASR punc_max_items must be positive")
    diarization = config.diarization
    if diarization.backend not in {"3d_speaker", "nemotron3"}:
        raise ValueError("diarization backend must be 3d_speaker or nemotron3")
    if config.nemotron.timeout_seconds <= 0:
        raise ValueError("Nemotron timeout_seconds must be positive")
    if config.nemotron.activity_threshold != 0.5:
        raise ValueError("Nemotron activity_threshold must be 0.5")
    if not (0.0 < config.nemotron.min_interruption_ratio <= 1.0):
        raise ValueError("Nemotron min_interruption_ratio must stay within 0..1")
    if config.nemotron.min_clause_ms <= 0:
        raise ValueError("Nemotron min_clause_ms must be positive")
    if diarization.backend == "3d_speaker":
        if not 1 <= diarization.min_speakers <= diarization.max_speakers <= 10:
            raise ValueError("min_speakers/max_speakers must stay within 1..10")
        if diarization.speaker_count is not None and not (
            diarization.min_speakers
            <= diarization.speaker_count
            <= diarization.max_speakers
        ):
            raise ValueError("speaker_count must stay within min_speakers/max_speakers")
    else:
        if diarization.speaker_count is not None and not 1 <= diarization.speaker_count <= 8:
            raise ValueError("Nemotron supports one to eight configured speakers")

        if config.asr.aed_scheduler == "memory_first":
            raise ValueError("Nemotron does not support memory_first")
    if not 0.0 < diarization.merge_cosine <= 1.0:
        raise ValueError("merge_cosine must stay within 0..1")
    if not 0.0 < diarization.merge_min_fraction <= 1.0:
        raise ValueError("merge_min_fraction must stay within 0..1")
    if not 0.0 < diarization.merge_relative_ratio <= 1.0:
        raise ValueError("merge_relative_ratio must stay within 0..1")
    if not 0.0 < diarization.major_cluster_min_fraction <= 1.0:
        raise ValueError("major_cluster_min_fraction must stay within 0..1")
    if not 0.0 < diarization.major_cluster_merge_threshold <= 1.0:
        raise ValueError("major_cluster_merge_threshold must stay within 0..1")
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
    diagnostics = config.diagnostics
    if diagnostics.level not in {"off", "basic", "resources", "deep"}:
        raise ValueError("diagnostics level must be off, basic, resources, or deep")
    if not 50 <= diagnostics.sampling_interval_ms <= 5000:
        raise ValueError(
            "diagnostics sampling_interval_ms must stay within 50..5000"
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
    llm = config.llm
    if llm.provider not in {"deepseek", "mimo", "dashscope", "qwen", "openai_compatible"}:
        raise ValueError("llm provider must be deepseek, mimo, dashscope, qwen, or openai_compatible")
    if not (0.0 <= llm.temperature <= 2.0):
        raise ValueError("llm temperature must stay within 0..2")
    if llm.timeout_seconds <= 0:
        raise ValueError("llm timeout_seconds must be positive")
    if not (0.0 <= llm.pinyin_gate_threshold <= 1.0):
        raise ValueError("llm pinyin_gate_threshold must stay within 0..1")
    return config


def _load_dotenv(start_path: Path | None = None) -> None:
    search_dirs: list[Path] = []
    if start_path is not None:
        target = Path(start_path).resolve()
        search_dirs.append(target if target.is_dir() else target.parent)
    search_dirs.append(Path.cwd().resolve())
    search_dirs.append(Path(__file__).resolve().parent.parent.parent)

    seen: set[Path] = set()
    for directory in search_dirs:
        if directory in seen:
            continue
        seen.add(directory)
        env_file = directory / ".env"
        if env_file.is_file():
            try:
                with env_file.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
                break
            except Exception:
                pass


def load_config(
    path: Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    _load_dotenv(path)
    values = deepcopy(DEFAULTS)
    if path is not None:
        with Path(path).open("rb") as stream:
            _merge_known(values, tomllib.load(stream))
    for dotted_key, raw in _environment_overrides().items():
        _apply_dotted(values, dotted_key, raw)
    for dotted_key, raw in (cli_overrides or {}).items():
        _apply_dotted(values, dotted_key, raw)
    return _build(values)

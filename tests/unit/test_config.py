from pathlib import Path

import pytest

from funasr_transcriber.config import load_config


def test_environment_overrides_toml(monkeypatch, tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        "\n".join(
            [
                "[general]",
                'device = "cpu"',
                "[diarization]",
                'speaker_count = "auto"',
                "max_speakers = 8",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FUNASR_TRANSCRIBER_GENERAL__DEVICE", "cuda:1")
    monkeypatch.setenv("FUNASR_TRANSCRIBER_DIARIZATION__MAX_SPEAKERS", "10")

    config = load_config(path)

    assert config.general.device == "cuda:1"
    assert config.diarization.speaker_count is None
    assert config.diarization.max_speakers == 10


def test_config_rejects_more_than_ten_speakers(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        "\n".join(
            [
                "[diarization]",
                'speaker_count = "auto"',
                "min_speakers = 1",
                "max_speakers = 11",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_speakers"):
        load_config(path)


def test_cli_overrides_environment(monkeypatch, tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text('[general]\ndevice = "cpu"\n', encoding="utf-8")
    monkeypatch.setenv("FUNASR_TRANSCRIBER_GENERAL__DEVICE", "cuda:1")

    config = load_config(path, cli_overrides={"general.device": "cuda:2"})

    assert config.general.device == "cuda:2"


def test_default_config_carries_validated_vad_values():
    config = load_config()

    assert config.vad.speech_noise_thres == 0.45
    assert config.vad.sil_to_speech_time_thres == 120
    assert config.vad.speech_to_sil_time_thres == 180
    assert config.vad.max_end_silence_time == 1000


def test_default_overlap_detector_uses_pyannote_3_pipeline():
    config = load_config()

    assert config.overlap.detector_model == "pyannote/speaker-diarization-3.1"


def test_default_service_configuration_disables_overlap_and_profiles():
    config = load_config()

    assert config.overlap.enabled is False
    assert config.service.queue_limit == 16
    assert config.service.startup_timeout_seconds == 180
    assert config.service.profile_sampling_interval_ms == 200
    assert config.service.minimum_total_mib == 23 * 1024
    assert config.service.minimum_free_mib == 13 * 1024


def test_default_diagnostics_level_is_basic():
    config = load_config()

    assert config.diagnostics.level == "basic"
    assert config.diagnostics.sampling_interval_ms == 200
    assert config.diagnostics.include_model_inventory is False
    assert config.diagnostics.write_batch_trace is False


@pytest.mark.parametrize("level", ["off", "basic", "resources", "deep"])
def test_diagnostics_accepts_supported_levels(tmp_path: Path, level: str):
    path = tmp_path / "config.toml"
    path.write_text(f"[diagnostics]\nlevel = \"{level}\"\n", encoding="utf-8")

    assert load_config(path).diagnostics.level == level


def test_diagnostics_rejects_unknown_level(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text('[diagnostics]\nlevel = "verbose"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="diagnostics level"):
        load_config(path)


def test_config_rejects_invalid_service_limits(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[service]\nqueue_limit = 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="service queue_limit"):
        load_config(path)


def test_default_asr_worker_configuration():
    config = load_config()

    assert config.asr.worker_python == Path(
        "/opt/funasr-transcriber/.venv-asr/bin/python"
    )
    assert config.asr.timeout_seconds == 1800
    assert config.asr.use_half is True
    assert config.asr.aed_scheduler == "weighted_duration"
    assert config.asr.aed_worker_count == 1
    assert config.asr.gpu_safety_margin_mib == 1536
    assert config.asr.punc_token_budget == 256
    assert config.asr.punc_max_items == 16
    assert not hasattr(config.asr, "segment_batch_size")
    assert not hasattr(config.asr, "segment_frame_budget")
    assert not hasattr(config.asr, "segment_memory_limit_mib")
    assert not hasattr(config.asr, "segment_memory_safety_mib")


@pytest.mark.parametrize(
    ("scheduler", "worker_count"),
    (
        ("weighted_duration", 1),
        ("weighted_duration", 2),
        ("memory_first", 2),
    ),
)
def test_aed_scheduler_accepts_supported_worker_counts(
    tmp_path: Path,
    scheduler: str,
    worker_count: int,
):
    path = tmp_path / "config.toml"
    path.write_text(
        (
            "[asr]\n"
            f'aed_scheduler = "{scheduler}"\n'
            f"aed_worker_count = {worker_count}\n"
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.asr.aed_scheduler == scheduler
    assert config.asr.aed_worker_count == worker_count


@pytest.mark.parametrize(
    "body",
    (
        '[asr]\naed_scheduler = "unknown"\n',
        '[asr]\naed_scheduler = "memory_first"\naed_worker_count = 1\n',
    ),
)
def test_aed_scheduler_rejects_unsupported_configuration(
    tmp_path: Path,
    body: str,
):
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match="aed_scheduler"):
        load_config(path)


def test_asr_half_precision_can_be_disabled(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[asr]\nuse_half = false\n", encoding="utf-8")

    assert load_config(path).asr.use_half is False


@pytest.mark.parametrize(
    ("setting", "message"),
    (
        ("timeout_seconds = 0", "ASR timeout_seconds"),
        ("aed_worker_count = 0", "ASR aed_worker_count"),
        ("aed_worker_count = 3", "ASR aed_worker_count"),
        ("gpu_safety_margin_mib = 0", "ASR gpu_safety_margin_mib"),
        ("gpu_safety_margin_mib = -1", "ASR gpu_safety_margin_mib"),
        ("punc_token_budget = 0", "ASR punc_token_budget"),
        ("punc_max_items = 0", "ASR punc_max_items"),
    ),
)
def test_asr_config_rejects_invalid_limits(
    tmp_path: Path,
    setting: str,
    message: str,
):
    path = tmp_path / "config.toml"
    path.write_text(f"[asr]\n{setting}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(path)

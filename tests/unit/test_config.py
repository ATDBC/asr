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
    assert config.asr.aed_weight_artifact is None
    assert not hasattr(config.asr, "segment_batch_size")
    assert not hasattr(config.asr, "segment_frame_budget")
    assert not hasattr(config.asr, "segment_memory_limit_mib")
    assert not hasattr(config.asr, "segment_memory_safety_mib")


def test_aed_weight_artifact_is_optional_and_expands_user(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[asr]\naed_weight_artifact = "~/weights.json"\n',
        encoding="utf-8",
    )

    assert load_config(path).asr.aed_weight_artifact == Path(
        "~/weights.json"
    ).expanduser()


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


def test_default_diarization_configuration_includes_major_cluster_guard():
    config = load_config()

    assert config.diarization.chunk_seconds == 1.5
    assert config.diarization.chunk_step_seconds == 0.75
    assert config.diarization.major_cluster_min_fraction == 0.10
    assert config.diarization.major_cluster_merge_threshold == 0.88


def test_config_rejects_invalid_major_cluster_guard_params(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[diarization]\nmajor_cluster_min_fraction = 0.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="major_cluster_min_fraction"):
        load_config(path)

    path.write_text("[diarization]\nmajor_cluster_merge_threshold = 1.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="major_cluster_merge_threshold"):
        load_config(path)

def test_nemotron_auto_mode_keeps_3d_speaker_default_unchanged():
    normal = load_config()
    selected = load_config(cli_overrides={"diarization.backend": "nemotron3"})
    assert normal.diarization.backend == "3d_speaker"
    assert selected.diarization.backend == "nemotron3"
    assert selected.diarization.speaker_count is None
    assert selected.diarization.max_speakers == 10
    assert selected.nemotron.activity_threshold == 0.5


def test_nemotron_selection_supports_overlap():
    config = load_config(cli_overrides={
        "diarization.backend": "nemotron3",
        "overlap.enabled": True,
    })
    assert config.diarization.backend == "nemotron3"
    assert config.overlap.enabled is True


def test_nemotron_rejects_ninth_configured_speaker():
    with pytest.raises(ValueError, match="eight"):
        load_config(cli_overrides={
            "diarization.backend": "nemotron3",
            "diarization.speaker_count": 9,
        })


def test_nemotron_rejects_memory_first():
    with pytest.raises(ValueError, match="memory_first"):
        load_config(cli_overrides={
            "diarization.backend": "nemotron3",
            "asr.aed_scheduler": "memory_first",
            "asr.aed_worker_count": 2,
        })


@pytest.mark.parametrize("backend", ["unknown", "", "Nemotron-3"])
def test_unknown_speaker_backend_is_rejected(backend: str):
    with pytest.raises(ValueError, match="diarization backend must be"):
        load_config(cli_overrides={"diarization.backend": backend})


def test_nemotron_threshold_is_locked_to_tested_value():
    with pytest.raises(ValueError, match="activity_threshold"):
        load_config(cli_overrides={
            "diarization.backend": "nemotron3",
            "nemotron.activity_threshold": 0.6,
        })


def test_nemotron_timeout_must_be_positive():
    with pytest.raises(ValueError, match="Nemotron.*timeout"):
        load_config(cli_overrides={
            "diarization.backend": "nemotron3",
            "nemotron.timeout_seconds": 0,
        })


def test_default_llm_configuration():
    config = load_config()

    assert config.llm.enabled is False
    assert config.llm.provider == "deepseek"
    assert config.llm.model == "deepseek-v4.1-flash"
    assert config.llm.enable_thinking is False
    assert config.llm.temperature == 0.1
    assert config.llm.timeout_seconds == 60
    assert config.llm.pinyin_gate_threshold == 0.5
    assert config.llm.max_context_turns == 100


def test_llm_configuration_can_be_customized(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[llm]\n"
        "enabled = true\n"
        'model = "deepseek-v4.1-flash"\n'
        "enable_thinking = false\n"
        "temperature = 0.2\n"
        "pinyin_gate_threshold = 0.6\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.llm.enabled is True
    assert config.llm.model == "deepseek-v4.1-flash"
    assert config.llm.enable_thinking is False
    assert config.llm.temperature == 0.2
    assert config.llm.pinyin_gate_threshold == 0.6


@pytest.mark.parametrize(
    ("setting", "match"),
    (
        ('provider = "unknown"', "llm provider"),
        ("temperature = -0.1", "llm temperature"),
        ("temperature = 2.5", "llm temperature"),
        ("timeout_seconds = 0", "llm timeout_seconds"),
        ("pinyin_gate_threshold = -0.1", "llm pinyin_gate_threshold"),
        ("pinyin_gate_threshold = 1.5", "llm pinyin_gate_threshold"),
        ("max_context_turns = 0", "llm max_context_turns"),
    ),
)
def test_llm_configuration_rejects_invalid_values(
    tmp_path: Path,
    setting: str,
    match: str,
):
    path = tmp_path / "config.toml"
    path.write_text(f"[llm]\n{setting}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_load_dotenv_reads_env_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=test_env_key_abc\n", encoding="utf-8")
    config_file = tmp_path / "config.toml"
    config_file.write_text("", encoding="utf-8")

    config = load_config(path=config_file)
    assert config.llm.resolve_api_key() == "test_env_key_abc"

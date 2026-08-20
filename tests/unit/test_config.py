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

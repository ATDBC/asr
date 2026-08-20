from pathlib import Path

import pytest

from funasr_transcriber.config import load_config
from funasr_transcriber.doctor import run_doctor
from funasr_transcriber.errors import EnvironmentError
from funasr_transcriber.model_registry import ModelVerification


def test_doctor_fails_when_ffmpeg_is_missing(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[paths]",
                f'model_dir = "{(tmp_path / "models").as_posix()}"',
                f'work_dir = "{(tmp_path / "work").as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "funasr_transcriber.doctor.shutil.which",
        lambda name: None if name == "ffmpeg" else f"/usr/bin/{name}",
    )

    with pytest.raises(EnvironmentError, match="ffmpeg"):
        run_doctor(load_config(config_path), check_cuda=False)


def test_doctor_checks_overlap_virtual_environment(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.toml"
    missing_python = tmp_path / "overlap" / "bin" / "python"
    config_path.write_text(
        "\n".join(
            [
                "[paths]",
                f'model_dir = "{(tmp_path / "models").as_posix()}"',
                f'work_dir = "{(tmp_path / "work").as_posix()}"',
                "[overlap]",
                f'worker_python = "{missing_python.as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "funasr_transcriber.doctor.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        "funasr_transcriber.doctor.verify_models",
        lambda path: ModelVerification(True, ()),
    )

    with pytest.raises(EnvironmentError, match="overlap worker Python"):
        run_doctor(load_config(config_path), check_cuda=False)

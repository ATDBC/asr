from pathlib import Path
from types import SimpleNamespace

import pytest

from funasr_transcriber.config import load_config
from funasr_transcriber.doctor import run_doctor
from funasr_transcriber.errors import EnvironmentError
from funasr_transcriber.gpu import GpuSample
from funasr_transcriber.model_registry import ModelVerification
from funasr_transcriber.schemas import (
    AsrResult,
    RecognizedSentence,
    SpeakerSegment,
    TimedToken,
)
import funasr_transcriber.doctor as doctor_module


def _config(tmp_path: Path, *, create_nemotron_python: bool):
    asr_python = tmp_path / "asr" / "bin" / "python"
    asr_python.parent.mkdir(parents=True)
    asr_python.write_bytes(b"")
    nemotron_python = tmp_path / "nemotron" / "bin" / "python"
    if create_nemotron_python:
        nemotron_python.parent.mkdir(parents=True)
        nemotron_python.write_bytes(b"")
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "\n".join(
            (
                "[paths]",
                f'model_dir = "{tmp_path / "models"}"',
                f'work_dir = "{tmp_path / "work"}"',
                "[asr]",
                f'worker_python = "{asr_python}"',
                "[diarization]",
                'backend = "nemotron3"',
                "[nemotron]",
                f'worker_python = "{nemotron_python}"',
            )
        ),
        encoding="utf-8",
    )
    return load_config(config_file)


def _patch(monkeypatch):
    monkeypatch.setattr(
        doctor_module.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(
        doctor_module,
        "verify_models",
        lambda path, speaker_backend: ModelVerification(True, ()),
    )


def test_selected_doctor_requires_nemotron_runtime(monkeypatch, tmp_path):
    config = _config(tmp_path, create_nemotron_python=False)
    _patch(monkeypatch)

    with pytest.raises(EnvironmentError, match="Nemotron worker Python"):
        run_doctor(config, check_cuda=False)


def test_selected_doctor_verifies_backend_and_smoke_diarizes(
    monkeypatch, tmp_path
):
    config = _config(tmp_path, create_nemotron_python=True)
    _patch(monkeypatch)
    backend_calls = []
    monkeypatch.setattr(
        doctor_module,
        "verify_models",
        lambda path, speaker_backend: backend_calls.append(speaker_backend)
        or ModelVerification(True, ()),
    )
    hello = (
        config.paths.model_dir
        / "fireredasr2s"
        / "source"
        / "assets"
        / "hello_zh.wav"
    )
    hello.parent.mkdir(parents=True)
    hello.write_bytes(b"source")
    monkeypatch.setattr(
        doctor_module,
        "normalize_audio",
        lambda source, destination: Path(destination).write_bytes(b"wav"),
        raising=False,
    )
    events = []

    class Asr:
        def __enter__(self):
            return self

        def verify(self, path):
            return AsrResult(
                "你好。",
                (TimedToken("你好。", 0, 500, 0.9),),
                sentences=(RecognizedSentence("你好。", 0, 500, 0.9),),
                vad_segments_ms=((0, 500),),
                duration_ms=600,
            )

        def __exit__(self, *args):
            pass

    class Speaker:
        def __enter__(self):
            events.append("start")
            return self

        def diarize(self, path):
            events.append(("diarize", Path(path).name))
            return (SpeakerSegment(1, 0, 500),)

        def __exit__(self, *args):
            events.append("close")

    sample = GpuSample(0, "GPU", 24_576, 22_000, "600.0")
    checks = run_doctor(
        config,
        asr_client_factory=Asr,
        speaker_factory=Speaker,
        gpu_capacity_check=lambda device: sample,
        gpu_sampler=lambda device: sample,
    )

    assert backend_calls == ["nemotron3"]
    assert events == ["start", ("diarize", "normalized.wav"), "close"]
    assert "speaker_models: verified" in checks

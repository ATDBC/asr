from pathlib import Path
from types import SimpleNamespace

import pytest

from funasr_transcriber.config import load_config
from funasr_transcriber.doctor import run_doctor
from funasr_transcriber.errors import EnvironmentError, ModelError
from funasr_transcriber.gpu import GpuSample
from funasr_transcriber.model_registry import ModelVerification
from funasr_transcriber.schemas import (
    AsrResult,
    RecognizedSentence,
    TimedToken,
)


def _write_config(
    tmp_path: Path,
    *,
    create_asr_python: bool = True,
    create_overlap_python: bool = True,
    overlap_enabled: bool = False,
):
    asr_python = tmp_path / "asr" / "bin" / "python"
    overlap_python = tmp_path / "overlap" / "bin" / "python"
    for path, create in (
        (asr_python, create_asr_python),
        (overlap_python, create_overlap_python),
    ):
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[paths]",
                f'model_dir = "{(tmp_path / "models").as_posix()}"',
                f'work_dir = "{(tmp_path / "work").as_posix()}"',
                "[asr]",
                f'worker_python = "{asr_python.as_posix()}"',
                "[overlap]",
                f'worker_python = "{overlap_python.as_posix()}"',
                f"enabled = {str(overlap_enabled).lower()}",
            ]
        ),
        encoding="utf-8",
    )
    return load_config(config_path)


def _patch_environment(monkeypatch):
    monkeypatch.setattr(
        "funasr_transcriber.doctor.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        "funasr_transcriber.doctor.verify_models",
        lambda path: ModelVerification(True, ()),
    )


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
    config = _write_config(
        tmp_path,
        create_overlap_python=False,
        overlap_enabled=True,
    )
    _patch_environment(monkeypatch)

    with pytest.raises(EnvironmentError, match="overlap worker Python"):
        run_doctor(config, check_cuda=False)


def test_doctor_checks_asr_virtual_environment(monkeypatch, tmp_path: Path):
    config = _write_config(tmp_path, create_asr_python=False)
    _patch_environment(monkeypatch)

    with pytest.raises(EnvironmentError, match="ASR worker Python"):
        run_doctor(config, check_cuda=False)


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        (("firered_source_commit",), "firered_source_commit"),
        (("firered_lid",), "firered_lid"),
    ],
)
def test_doctor_rejects_invalid_firered_assets(
    monkeypatch,
    tmp_path: Path,
    missing,
    message,
):
    config = _write_config(tmp_path, overlap_enabled=True)
    _patch_environment(monkeypatch)
    monkeypatch.setattr(
        "funasr_transcriber.doctor.verify_models",
        lambda path: ModelVerification(False, missing),
    )

    with pytest.raises(ModelError, match=message):
        run_doctor(config, check_cuda=False)


@pytest.mark.parametrize(
    "failure",
    [
        EnvironmentError("GPU 0 total memory is below 23 GiB"),
        EnvironmentError("GPU 0 free memory is below 12 GiB"),
        EnvironmentError("nvidia-smi GPU monitoring failed"),
    ],
)
def test_doctor_propagates_gpu_admission_failures(
    monkeypatch,
    tmp_path: Path,
    failure,
):
    config = _write_config(tmp_path)
    _patch_environment(monkeypatch)

    def reject_gpu(device):
        assert device == "cuda:0"
        raise failure

    with pytest.raises(EnvironmentError, match=str(failure)):
        run_doctor(config, gpu_capacity_check=reject_gpu)


def test_doctor_runs_complete_firered_then_speaker_then_overlap_sequence(
    monkeypatch,
    tmp_path: Path,
):
    config = _write_config(tmp_path, overlap_enabled=True)
    _patch_environment(monkeypatch)
    events: list[str] = []
    before = GpuSample(0, "GPU", 24_576, 22_000, "600.0")
    after = GpuSample(0, "GPU", 24_576, 21_900, "600.0")
    expected_audio = (
        config.paths.model_dir
        / "fireredasr2s"
        / "source"
        / "assets"
        / "hello_zh.wav"
    ).resolve()

    class AsrClient:
        def __enter__(self):
            events.append("firered:start")
            return self

        def verify(self, audio_path):
            assert Path(audio_path) == expected_audio
            events.append("firered:verify")
            return AsrResult(
                "你好。",
                (TimedToken("你好。", 0, 500, 0.9),),
                sentences=(
                    RecognizedSentence(
                        "你好。", 0, 500, "zh mandarin", 0.99, 0.9
                    ),
                ),
                vad_segments_ms=((0, 500),),
                duration_ms=600,
            )

        def __exit__(self, *args):
            events.append("firered:close")

    class Speaker:
        def __enter__(self):
            events.append("speaker:start")
            return self

        def __exit__(self, *args):
            events.append("speaker:close")

    samples = iter((after, after, after))

    def gpu_check(device):
        assert device == "cuda:0"
        events.append("gpu:check")
        return before

    def gpu_sample(device):
        assert device == "cuda:0"
        events.append("gpu:sample")
        return next(samples)

    def verify_overlap():
        events.append("overlap:verify")

    checks = run_doctor(
        config,
        asr_client_factory=AsrClient,
        speaker_factory=Speaker,
        overlap_verifier=verify_overlap,
        gpu_capacity_check=gpu_check,
        gpu_sampler=gpu_sample,
    )

    assert events == [
        "gpu:check",
        "firered:start",
        "firered:verify",
        "firered:close",
        "gpu:sample",
        "gpu:check",
        "speaker:start",
        "speaker:close",
        "gpu:sample",
        "gpu:check",
        "overlap:verify",
        "gpu:sample",
    ]
    assert "firered_source: verified" in checks
    assert "firered_models: verified" in checks
    assert "firered_worker: verified" in checks
    assert "speaker_models: verified" in checks
    assert "overlap_models: verified" in checks
    assert "gpu_release: verified" in checks


def test_doctor_rejects_incomplete_firered_verify_response(
    monkeypatch,
    tmp_path: Path,
):
    config = _write_config(tmp_path)
    _patch_environment(monkeypatch)

    class IncompleteAsrClient:
        def __enter__(self):
            return self

        def verify(self, audio_path):
            return SimpleNamespace(
                text="",
                tokens=(),
                sentences=(),
                vad_segments_ms=(),
                duration_ms=0,
            )

        def __exit__(self, *args):
            return None

    with pytest.raises(ModelError, match="incomplete response"):
        run_doctor(
            config,
            asr_client_factory=IncompleteAsrClient,
            gpu_capacity_check=lambda device: GpuSample(
                0, "GPU", 24_576, 22_000, "600.0"
            ),
        )


def test_skip_cuda_does_not_report_full_model_smoke(monkeypatch, tmp_path: Path):
    config = _write_config(tmp_path)
    _patch_environment(monkeypatch)

    checks = run_doctor(config, check_cuda=False)

    assert "firered_source: verified" in checks
    assert "firered_models: verified" in checks
    assert "model_smoke: skipped (--skip-cuda)" in checks
    assert "firered_worker: verified" not in checks
    assert "gpu_release: verified" not in checks

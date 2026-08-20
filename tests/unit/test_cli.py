from pathlib import Path
from types import SimpleNamespace

import pytest

from funasr_transcriber import cli
from funasr_transcriber.cli import (
    default_output_dir,
    main,
    require_physical_gpu_zero,
    staging_output_dir,
)
from funasr_transcriber.errors import (
    EnvironmentError,
    InputAudioError,
    ModelError,
    OutputError,
    TranscriptionError,
)


def test_default_output_is_next_to_input():
    assert default_output_dir(Path("/data/录音.mp3")) == Path(
        "/data/录音_transcript"
    )


def test_staging_output_is_hidden_sibling_of_final_directory():
    target = Path("/data/results/call")

    assert staging_output_dir(target, "abc") == Path(
        "/data/results/.call.staging-abc"
    )


@pytest.mark.parametrize("visible", [None, "0", "0,1"])
def test_physical_gpu_zero_accepts_compatible_visibility(monkeypatch, visible):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    require_physical_gpu_zero("cuda:0")


@pytest.mark.parametrize("visible", ["", "1", "1,0", "GPU-abcd"])
def test_physical_gpu_zero_rejects_cuda_remapping(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    with pytest.raises(EnvironmentError, match="physical GPU 0"):
        require_physical_gpu_zero("cuda:0")


def test_physical_gpu_zero_rejects_other_logical_device(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(EnvironmentError, match="physical GPU 0"):
        require_physical_gpu_zero("cuda:1")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (InputAudioError("bad input"), 3),
        (EnvironmentError("bad environment"), 4),
        (ModelError("bad model"), 4),
        (TranscriptionError("bad transcript"), 5),
        (OutputError("bad output"), 6),
    ],
)
def test_main_maps_domain_errors(error, expected, capsys):
    def runner(args):
        raise error

    code = main(["sample.wav"], transcribe_runner=runner)

    assert code == expected
    assert str(error) in capsys.readouterr().err


def test_main_returns_two_for_missing_input(capsys):
    assert main([], transcribe_runner=lambda args: 0) == 2
    assert "usage:" in capsys.readouterr().err.lower()


def test_main_maps_configuration_value_error_to_two(capsys):
    code = main(
        ["sample.wav"],
        transcribe_runner=lambda args: (_ for _ in ()).throw(
            ValueError("bad configuration")
        ),
    )

    assert code == 2
    assert "bad configuration" in capsys.readouterr().err


def test_main_passes_transcribe_arguments():
    received = {}

    def runner(args):
        received["args"] = args
        return 0

    code = main(
        [
            "sample.wav",
            "--output-dir",
            "result",
            "--device",
            "cuda:1",
            "--keep-intermediate",
        ],
        transcribe_runner=runner,
    )

    assert code == 0
    assert received["args"].input_audio == Path("sample.wav")
    assert received["args"].output_dir == Path("result")
    assert received["args"].device == "cuda:1"
    assert received["args"].keep_intermediate is True


def test_models_verify_dispatches_to_model_runner():
    received = {}

    def runner(action, args):
        received["action"] = action
        return 0

    assert main(["models", "verify"], model_runner=runner) == 0
    assert received["action"] == "verify"


def test_doctor_dispatches_to_doctor_runner():
    assert main(["doctor"], doctor_runner=lambda args: 0) == 0


def test_doctor_route_delegates_complete_model_sequence_once(
    monkeypatch,
    capsys,
):
    config = SimpleNamespace(
        general=SimpleNamespace(device="cuda:0"),
        overlap=SimpleNamespace(enabled=True),
    )
    monkeypatch.setattr(cli, "load_config", lambda path: config)

    def fake_run_doctor(received, check_cuda, smoke_test):
        assert received is config
        assert check_cuda is False
        assert smoke_test is None
        return ("doctor: ok",)

    monkeypatch.setattr(cli, "run_doctor", fake_run_doctor)
    monkeypatch.setattr(
        cli,
        "JsonOverlapWorkerClient",
        lambda *args, **kwargs: pytest.fail(
            "CLI must not repeat overlap verification"
        ),
    )

    assert cli._run_doctor(SimpleNamespace(config=None, skip_cuda=True)) == 0
    assert capsys.readouterr().out == "doctor: ok\n"


def test_doctor_enforces_physical_gpu_zero_before_full_smoke(monkeypatch):
    config = SimpleNamespace(
        general=SimpleNamespace(device="cuda:0"),
        overlap=SimpleNamespace(enabled=False),
    )
    events = []
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "require_physical_gpu_zero",
        lambda device: events.append(f"physical:{device}"),
    )
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda received, check_cuda, smoke_test: events.append("doctor") or (),
    )

    assert cli._run_doctor(SimpleNamespace(config=None, skip_cuda=False)) == 0
    assert events == ["physical:cuda:0", "doctor"]

from pathlib import Path

import pytest

from funasr_transcriber.cli import default_output_dir, main, staging_output_dir
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

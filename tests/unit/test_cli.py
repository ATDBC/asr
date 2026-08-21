from pathlib import Path
from types import SimpleNamespace

import pytest

from funasr_transcriber import cli
from funasr_transcriber.config import load_config
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
from funasr_transcriber.service_protocol import ServiceProtocolError


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


def test_transcribe_client_starts_service_only_when_health_is_unavailable(
    monkeypatch,
    tmp_path: Path,
):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    config = load_config(
        cli_overrides={"paths.work_dir": str(tmp_path / "work")},
    )
    calls: list[str] = []
    launched = False

    class FakeClient:
        health_calls = 0

        def __init__(self, paths, timeout_seconds):
            self.paths = paths

        def health(self, identifier):
            type(self).health_calls += 1
            calls.append(f"health:{identifier}")
            if not launched:
                raise ServiceProtocolError("not running")
            return {"status": "ready"}

        def submit_and_wait(self, request):
            calls.append(f"submit:{request.identifier}")
            return {"ok": True, "result": {"output_dir": str(request.output_dir)}}

    class FakeStartupLock:
        def __init__(self, directory):
            assert directory == tmp_path / "work" / "service"

        def __enter__(self):
            calls.append("lock")
            return self

        def __exit__(self, *args):
            calls.append("unlock")

    monkeypatch.setattr(cli, "ServiceClient", FakeClient)
    monkeypatch.setattr(cli, "ServiceStartupLock", FakeStartupLock)

    def fake_launch(received_config, paths, config_path):
        nonlocal launched
        launched = True
        calls.append(f"launch:{paths.socket}")

    monkeypatch.setattr(cli, "launch_service", fake_launch)
    args = SimpleNamespace(
        input_audio=audio,
        output_dir=tmp_path / "output",
        overwrite=False,
        keep_intermediate=False,
        enable_overlap=False,
        disable_overlap=False,
        profile_resources=False,
        config=None,
    )

    result = cli.submit_to_service(args, config)

    assert result["result"]["output_dir"] == str(tmp_path / "output")
    assert calls[0].startswith("health:")
    assert calls.count("lock") == 1
    assert calls.count("unlock") == 1
    assert any(item.startswith("launch:") for item in calls)
    assert any(item.startswith("submit:") for item in calls)


def test_transcribe_route_submits_to_persistent_service(monkeypatch, tmp_path: Path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio")
    config = load_config(
        cli_overrides={"paths.work_dir": str(tmp_path / "work")},
    )
    received = []
    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        cli,
        "submit_to_service",
        lambda args, received_config: received.append((args, received_config))
        or {"ok": True, "result": {"output_dir": str(tmp_path / "output")}},
    )
    args = SimpleNamespace(
        input_audio=audio,
        output_dir=tmp_path / "output",
        device=None,
        overwrite=False,
        keep_intermediate=False,
        disable_overlap=False,
        enable_overlap=False,
        profile_resources=False,
        config=None,
    )

    assert cli._run_transcribe(args) == 0
    assert received == [(args, config)]


def test_service_status_uses_local_client(monkeypatch, tmp_path: Path, capsys):
    config = load_config(
        cli_overrides={"paths.work_dir": str(tmp_path / "work")},
    )

    class FakeClient:
        def __init__(self, paths, timeout_seconds):
            assert paths == cli.ServicePaths.from_work_dir(config.paths.work_dir)

        def health(self, identifier):
            return {"ok": True, "status": "ready", "terminal": True}

    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "ServiceClient", FakeClient)

    assert cli.main(["service", "status"]) == 0
    assert '"status": "ready"' in capsys.readouterr().out

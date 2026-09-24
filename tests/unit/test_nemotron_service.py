from pathlib import Path

import pytest

from funasr_transcriber.config import load_config
from funasr_transcriber.service import (
    HotModelHost,
    ServiceJobRunner,
    build_hot_model_host,
)
from funasr_transcriber.service_protocol import ServicePaths
import funasr_transcriber.service as service_module


def test_selected_nemotron_pid_is_profiled():
    runner = ServiceJobRunner.__new__(ServiceJobRunner)
    runner.hot_host = type(
        "Host",
        (),
        {
            "speaker_owner": type("Speaker", (), {"pid": 451})(),
            "asr_owner": None,
        },
    )()
    runner._active_overlap_pid = None

    assert runner._owned_pid_getters()["speaker"]() == 451


def test_selected_nemotron_factory_and_gpu_admission(
    tmp_path: Path, monkeypatch
):
    import funasr_transcriber.nemotron_client as nemotron_module

    calls = []
    marker = object()
    monkeypatch.setattr(
        nemotron_module,
        "NemotronWorkerClient",
        lambda **kwargs: calls.append(kwargs) or marker,
    )
    monkeypatch.setattr(
        service_module,
        "require_gpu_capacity",
        lambda device, **kwargs: calls.append((device, kwargs)),
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[diarization]\nbackend = "nemotron3"\n',
        encoding="utf-8",
    )
    config = load_config(config_path)
    host = build_hot_model_host(
        config, ServicePaths.from_work_dir(tmp_path)
    )

    host._gpu_check()
    assert host._diarizer_factory() is marker
    assert calls[0] == (
        "cuda:0",
        {
            "minimum_total_mib": config.service.minimum_total_mib,
            "minimum_free_mib": config.service.minimum_free_mib,
        },
    )
    assert calls[1]["worker_python"] == config.nemotron.worker_python
    assert calls[1]["model_dir"] == config.paths.model_dir


def test_gpu_admission_failure_starts_no_owner():
    events = []
    host = HotModelHost(
        asr_factory=lambda: events.append("asr"),
        diarizer_factory=lambda: events.append("speaker"),
        gpu_check=lambda: (_ for _ in ()).throw(
            RuntimeError("insufficient GPU0 memory")
        ),
    )
    with pytest.raises(RuntimeError, match="GPU0"):
        host.start()
    assert events == []

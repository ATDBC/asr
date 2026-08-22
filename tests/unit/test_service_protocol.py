from __future__ import annotations

from pathlib import Path

import pytest

from funasr_transcriber.service_protocol import (
    PROTOCOL_VERSION,
    ServicePaths,
    ServiceProtocolError,
    ServiceState,
    ServiceStateStore,
    parse_request,
)


def test_service_paths_are_scoped_to_work_dir(tmp_path: Path):
    paths = ServicePaths.from_work_dir(tmp_path)

    assert paths.directory == tmp_path / "service"
    assert paths.socket == tmp_path / "service" / "transcriber.sock"
    assert paths.state == tmp_path / "service" / "state.json"


def test_state_store_rejects_live_incompatible_service(tmp_path: Path):
    paths = ServicePaths.from_work_dir(tmp_path)
    store = ServiceStateStore(paths, pid_alive=lambda pid: pid == 99)
    store.write(
        ServiceState(
            pid=99,
            protocol_version=PROTOCOL_VERSION,
            config_fingerprint="old",
        )
    )

    with pytest.raises(ServiceProtocolError, match="configuration"):
        store.require_compatible("new")


def test_protocol_rejects_relative_submit_paths():
    with pytest.raises(ServiceProtocolError, match="absolute"):
        parse_request(
            {
                "id": "x",
                "action": "submit",
                "input_audio": "a.wav",
                "output_dir": "/tmp/output",
                "config_fingerprint": "fingerprint",
            }
        )


def test_protocol_parses_wait_request_for_an_accepted_job():
    request = parse_request(
        {"id": "wait-1", "action": "wait", "job_id": "job-1"}
    )

    assert request.action == "wait"
    assert request.job_id == "job-1"


def test_protocol_rejects_wait_without_job_id():
    with pytest.raises(ServiceProtocolError, match="job_id"):
        parse_request({"id": "wait-1", "action": "wait"})


def test_protocol_parses_diagnostics_level():
    request = parse_request(
        {
            "id": "job-1",
            "action": "submit",
            "input_audio": "/tmp/input.wav",
            "output_dir": "/tmp/output",
            "config_fingerprint": "fingerprint",
            "diagnostics_level": "deep",
        }
    )

    assert request.diagnostics_level == "deep"

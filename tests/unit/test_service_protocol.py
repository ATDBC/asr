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

import json
from pathlib import Path

from funasr_transcriber.diagnostics import (
    Diagnostics,
    export_diagnostics,
    sanitize_mapping,
)


def test_sanitize_mapping_redacts_secret_keys():
    sanitized = sanitize_mapping(
        {
            "HF_TOKEN": "secret",
            "nested": {
                "password": "hidden",
                "device": "cuda:0",
            },
        }
    )

    assert sanitized["HF_TOKEN"] == "<redacted>"
    assert sanitized["nested"]["password"] == "<redacted>"
    assert sanitized["nested"]["device"] == "cuda:0"


def test_sanitize_mapping_converts_paths_to_strings(tmp_path: Path):
    sanitized = sanitize_mapping({"work_dir": tmp_path})

    assert sanitized["work_dir"] == str(tmp_path)


def test_export_diagnostics_includes_runtime_and_config(tmp_path: Path):
    diagnostics = Diagnostics()
    diagnostics.runtime = {"python": "3.11"}
    diagnostics.config = {"HF_TOKEN": "secret", "device": "cuda:0"}
    diagnostics.speaker_count = 2

    export_diagnostics(diagnostics, tmp_path, overwrite=False)
    payload = json.loads(
        (tmp_path / "diagnostics.json").read_text(encoding="utf-8")
    )

    assert payload["runtime"]["python"] == "3.11"
    assert payload["config"]["HF_TOKEN"] == "<redacted>"
    assert payload["speaker_count"] == 2

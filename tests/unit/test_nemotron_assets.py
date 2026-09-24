from pathlib import Path

import pytest

from funasr_transcriber.errors import ModelError
from funasr_transcriber.nemotron_assets import (
    NEMOTRON_MODEL_SHA256,
    NEMOTRON_REPO,
    NEMOTRON_REVISION,
    acquire_nemotron_snapshot,
    verify_nemotron_snapshot,
)


def test_tampered_checkpoint_is_rejected(tmp_path: Path):
    target = tmp_path / "nemotron3" / NEMOTRON_REVISION
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "processor_config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(ModelError, match="checksum"):
        verify_nemotron_snapshot(tmp_path)


def test_acquisition_uses_pinned_revision_and_atomic_sibling(
    monkeypatch, tmp_path: Path
):
    calls = []
    monkeypatch.setattr(
        "funasr_transcriber.nemotron_assets._sha256_file",
        lambda path: NEMOTRON_MODEL_SHA256,
    )

    def download(*, repo_id, revision, local_dir):
        target = Path(local_dir)
        calls.append((repo_id, revision, target))
        target.mkdir()
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "processor_config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"test")

    result = acquire_nemotron_snapshot(tmp_path, download=download)

    assert result == tmp_path / "nemotron3" / NEMOTRON_REVISION
    assert calls[0][:2] == (NEMOTRON_REPO, NEMOTRON_REVISION)
    assert calls[0][2].parent == result.parent
    assert ".download-" in calls[0][2].name
    assert result.is_dir()


def test_partial_existing_snapshot_is_preserved(tmp_path: Path):
    target = tmp_path / "nemotron3" / NEMOTRON_REVISION
    target.mkdir(parents=True)
    (target / "partial.bin").write_bytes(b"keep")

    with pytest.raises(ModelError, match="missing Nemotron asset"):
        acquire_nemotron_snapshot(
            tmp_path,
            download=lambda **kwargs: pytest.fail("must not download"),
        )

    assert (target / "partial.bin").read_bytes() == b"keep"

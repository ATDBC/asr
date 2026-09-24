"""Pinned, offline-verifiable Nemotron 3 Diarization snapshot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable
import uuid

from .errors import ModelError


NEMOTRON_REPO = "nvidia/Nemotron-3-Diarization"
NEMOTRON_REVISION = "a435e9867d79e789e90053f9b6d6834053af564a"
NEMOTRON_MODEL_SHA256 = (
    "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"
)
_REQUIRED_FILES = ("config.json", "processor_config.json", "model.safetensors")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_target(target: Path) -> Path:
    for name in _REQUIRED_FILES:
        path = target / name
        if not path.is_file() or path.is_symlink():
            raise ModelError(f"missing Nemotron asset: {path}")
    if _sha256_file(target / "model.safetensors") != NEMOTRON_MODEL_SHA256:
        raise ModelError("Nemotron checkpoint checksum mismatch")
    return target


def verify_nemotron_snapshot(model_dir: Path) -> Path:
    """Verify a local snapshot without network access."""
    target = (
        Path(model_dir).expanduser().resolve()
        / "nemotron3"
        / NEMOTRON_REVISION
    )
    return _verify_target(target)


def acquire_nemotron_snapshot(
    model_dir: Path,
    download: Callable[..., Any] | None = None,
) -> Path:
    """Fetch exactly the pinned revision, staging beside the final path."""
    target = (
        Path(model_dir).expanduser().resolve()
        / "nemotron3"
        / NEMOTRON_REVISION
    )
    if target.exists() or target.is_symlink():
        return _verify_target(target)

    if download is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ModelError(
                f"unable to import huggingface_hub: {exc}"
            ) from exc
        download = snapshot_download

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.download-{uuid.uuid4().hex}"
    download(
        repo_id=NEMOTRON_REPO,
        revision=NEMOTRON_REVISION,
        local_dir=str(temporary),
    )
    _verify_target(temporary)
    if target.exists() or target.is_symlink():
        raise ModelError(f"Nemotron asset appeared during download: {target}")
    temporary.rename(target)
    return target

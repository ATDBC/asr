from pathlib import Path

import pytest

from funasr_transcriber.errors import ModelError
from funasr_transcriber.model_registry import (
    FIRERED_COMMIT,
    FIRERED_MODELS,
    acquire_firered_assets,
    resolve_firered_paths,
)


class GitResult:
    def __init__(self, stdout: str = ""):
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


def test_acquisition_uses_modelscope_and_atomic_sibling_directories(
    tmp_path: Path,
):
    downloads: list[tuple[str, Path]] = []
    commands: list[tuple[tuple[str, ...], Path | None]] = []

    def snapshot_download(model_id: str, local_dir: str):
        target = Path(local_dir)
        downloads.append((model_id, target))
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.pth.tar").write_bytes(b"model")
        if model_id == FIRERED_MODELS["vad"][0]:
            (target / "VAD").mkdir()
            (target / "VAD" / "model.pth.tar").write_bytes(b"vad")

    def git_runner(arguments: list[str], cwd: Path | None = None):
        commands.append((tuple(arguments), cwd))
        if arguments[0] == "clone":
            target = Path(arguments[-1])
            target.mkdir(parents=True)
            (target / "pyproject.toml").write_text("[project]\n", "utf-8")
        if arguments[:2] == ["rev-parse", "HEAD"]:
            return GitResult(FIRERED_COMMIT + "\n")
        return GitResult()

    paths = acquire_firered_assets(
        tmp_path,
        snapshot_download=snapshot_download,
        git_runner=git_runner,
    )

    assert paths == resolve_firered_paths(tmp_path, require_existing=True)
    assert [item[0] for item in downloads] == [
        FIRERED_MODELS[key][0] for key in ("vad", "lid", "aed", "punc")
    ]
    assert all(".download-" in target.name for _, target in downloads)
    assert any(command[0] == "clone" for command, _ in commands)
    assert any(
        command == ("checkout", "--detach", FIRERED_COMMIT)
        for command, _ in commands
    )


def test_acquisition_refuses_existing_invalid_target(tmp_path: Path):
    paths = resolve_firered_paths(tmp_path, require_existing=False)
    paths.aed.mkdir(parents=True)
    (paths.aed / "partial.bin").write_bytes(b"partial")

    with pytest.raises(ModelError, match="existing FireRed asset"):
        acquire_firered_assets(
            tmp_path,
            snapshot_download=lambda model_id, local_dir: None,
            git_runner=lambda arguments, cwd=None: GitResult(FIRERED_COMMIT),
        )

    assert (paths.aed / "partial.bin").read_bytes() == b"partial"


def test_huggingface_is_only_used_when_explicit(tmp_path: Path):
    downloaded: list[str] = []

    def hf_download(repo_id: str, local_dir: str):
        downloaded.append(repo_id)
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.pth.tar").write_bytes(b"model")
        if repo_id == FIRERED_MODELS["vad"][1]:
            (target / "VAD").mkdir()
            (target / "VAD" / "model.pth.tar").write_bytes(b"vad")

    def git_runner(arguments: list[str], cwd: Path | None = None):
        if arguments[0] == "clone":
            target = Path(arguments[-1])
            target.mkdir(parents=True)
            (target / "pyproject.toml").write_text("[project]\n", "utf-8")
        if arguments[:2] == ["rev-parse", "HEAD"]:
            return GitResult(FIRERED_COMMIT)
        return GitResult()

    acquire_firered_assets(
        tmp_path,
        source="huggingface",
        hf_download=hf_download,
        git_runner=git_runner,
    )

    assert downloaded == [
        FIRERED_MODELS[key][1] for key in ("vad", "lid", "aed", "punc")
    ]

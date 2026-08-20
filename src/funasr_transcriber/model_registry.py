"""Model paths and lazy model construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

from .errors import ModelError


MODEL_DIRS = {
    "asr": "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "vad": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc": "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "spk": "speech_campplus_sv_zh-cn_16k-common",
}

THREED_SPEAKER_COMMIT = "065629c313eaf1a01c65c640c46d77e61e9607b4"
DIARIZATION_EMBEDDING_DIR = (
    "speech_campplus_sv_zh_en_16k-common_advanced"
)


@dataclass(frozen=True, slots=True)
class ModelPaths:
    asr: Path
    vad: Path
    punc: Path
    spk: Path


@dataclass(frozen=True, slots=True)
class ModelVerification:
    ok: bool
    missing: tuple[str, ...]


def resolve_model_paths(model_dir: Path, require_existing: bool = True) -> ModelPaths:
    root = (
        Path(model_dir).expanduser().resolve() / "modelscope" / "iic"
    )
    paths = ModelPaths(**{key: root / value for key, value in MODEL_DIRS.items()})
    if require_existing:
        missing = [
            f"{name}: {path}"
            for name, path in (
                ("asr", paths.asr),
                ("vad", paths.vad),
                ("punc", paths.punc),
            )
            if not path.is_dir()
        ]
        if missing:
            raise ModelError("missing required models: " + "; ".join(missing))
    return paths


def _directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def verify_models(model_dir: Path) -> ModelVerification:
    root = Path(model_dir).expanduser().resolve()
    paths = resolve_model_paths(root, require_existing=False)
    missing: list[str] = []
    for name, path in (
        ("asr", paths.asr),
        ("vad", paths.vad),
        ("punc", paths.punc),
    ):
        if not _directory_has_files(path):
            missing.append(name)
    embedding_checkpoint = (
        root
        / "modelscope"
        / "iic"
        / DIARIZATION_EMBEDDING_DIR
        / "campplus_cn_en_common.pt"
    )
    if not embedding_checkpoint.is_file():
        missing.append("diarization_embedding")
    speakerlab = root / "3d-speaker" / "3D-Speaker" / "speakerlab"
    if not speakerlab.is_dir():
        missing.append("3d_speaker_repo")
    return ModelVerification(not missing, tuple(missing))


def _run_git(arguments: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise ModelError(
            f"git command failed: {' '.join(arguments)}: "
            f"{completed.stderr.strip()}"
        )


def download_models(model_dir: Path) -> ModelVerification:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ModelError(f"unable to import ModelScope: {exc}") from exc

    root = Path(model_dir).expanduser().resolve()
    iic_root = root / "modelscope" / "iic"
    iic_root.mkdir(parents=True, exist_ok=True)
    for key in ("asr", "vad", "punc"):
        snapshot_download(
            f"iic/{MODEL_DIRS[key]}",
            local_dir=str(iic_root / MODEL_DIRS[key]),
        )
    snapshot_download(
        f"iic/{DIARIZATION_EMBEDDING_DIR}",
        revision="v1.0.0",
        local_dir=str(iic_root / DIARIZATION_EMBEDDING_DIR),
    )

    repository = root / "3d-speaker" / "3D-Speaker"
    if not repository.is_dir():
        repository.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            [
                "clone",
                "https://github.com/modelscope/3D-Speaker.git",
                str(repository),
            ]
        )
    _run_git(["checkout", THREED_SPEAKER_COMMIT], cwd=repository)
    result = verify_models(root)
    if not result.ok:
        raise ModelError(
            "model download completed with missing components: "
            + ", ".join(result.missing)
        )
    return result


def build_asr_model(model_dir: Path, device: str) -> Any:
    try:
        import torch
        from funasr import AutoModel
    except ImportError as exc:
        raise ModelError(f"unable to import FunASR runtime: {exc}") from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ModelError(f"CUDA device requested but unavailable: {device}")
    paths = resolve_model_paths(model_dir)
    try:
        return AutoModel(
            model=str(paths.asr),
            vad_model=str(paths.vad),
            punc_model=str(paths.punc),
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
            device=device,
            ngpu=1 if device.startswith("cuda") else 0,
        )
    except Exception as exc:
        raise ModelError(f"unable to load FunASR models: {exc}") from exc

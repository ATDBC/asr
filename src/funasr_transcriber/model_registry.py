"""Model paths and lazy model construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable
import uuid

from .errors import ModelError


MODEL_DIRS = {
    "vad": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
}

THREED_SPEAKER_COMMIT = "065629c313eaf1a01c65c640c46d77e61e9607b4"
FIRERED_COMMIT = "4e7d9aaf4482a47cec1724807026b9b151926eb5"
FIRERED_REPOSITORY = "https://github.com/FireRedTeam/FireRedASR2S.git"
FIRERED_MODELS = {
    "vad": ("xukaituo/FireRedVAD", "FireRedTeam/FireRedVAD", "FireRedVAD"),
    "aed": (
        "xukaituo/FireRedASR2-AED",
        "FireRedTeam/FireRedASR2-AED",
        "FireRedASR2-AED",
    ),
    "punc": (
        "xukaituo/FireRedPunc",
        "FireRedTeam/FireRedPunc",
        "FireRedPunc",
    ),
}
FIRERED_MODEL_NAMES = ("vad", "aed", "punc")
OVERLAP_MODELS = (
    (
        "pyannote/speaker-diarization-3.1",
        "pyannote/speaker-diarization-3.1",
        "config.yaml",
    ),
    (
        "pyannote/segmentation-3.0",
        "pyannote/segmentation-3.0",
        "pytorch_model.bin",
    ),
    (
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        "pytorch_model.bin",
    ),
    (
        "alibabasglab/MossFormer2_SS_16K",
        "checkpoints/MossFormer2_SS_16K",
        "last_best_checkpoint",
    ),
)
SPEAKER_VAD_DIR = MODEL_DIRS["vad"]
DIARIZATION_EMBEDDING_DIR = (
    "speech_campplus_sv_zh_en_16k-common_advanced"
)


@dataclass(frozen=True, slots=True)
class ModelVerification:
    ok: bool
    missing: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FireRedPaths:
    source: Path
    vad: Path
    aed: Path
    punc: Path


@dataclass(frozen=True, slots=True)
class OverlapPaths:
    pipeline: Path
    pipeline_config: Path
    segmentation_checkpoint: Path
    embedding_checkpoint: Path
    clearvoice: Path
    clearvoice_checkpoint: Path


def resolve_firered_paths(
    model_dir: Path,
    require_existing: bool = True,
) -> FireRedPaths:
    root = Path(model_dir).expanduser().resolve() / "fireredasr2s"
    paths = FireRedPaths(
        source=root / "source",
        vad=root / FIRERED_MODELS["vad"][2],
        aed=root / FIRERED_MODELS["aed"][2],
        punc=root / FIRERED_MODELS["punc"][2],
    )
    if require_existing:
        missing = [
            name
            for name, path in (
                ("firered_source", paths.source),
                ("firered_vad", paths.vad / "VAD"),
                ("firered_aed", paths.aed),
                ("firered_punc", paths.punc),
            )
            if not _directory_has_files(path)
        ]
        if missing:
            raise ModelError(
                "missing required models: " + ", ".join(missing)
            )
    return paths


def _clearvoice_checkpoint(directory: Path) -> Path:
    pointer = directory / "last_best_checkpoint"
    name = "last_best_checkpoint.pt"
    if pointer.is_file():
        name = pointer.read_text(encoding="utf-8").strip()
        if not name or Path(name).name != name:
            raise ModelError(
                f"invalid ClearVoice checkpoint pointer: {pointer}"
            )
    return directory / name


def resolve_overlap_paths(
    model_dir: Path,
    require_existing: bool = True,
) -> OverlapPaths:
    root = Path(model_dir).expanduser().resolve() / "huggingface"
    pipeline = root / "pyannote" / "speaker-diarization-3.1"
    segmentation = (
        root / "pyannote" / "segmentation-3.0" / "pytorch_model.bin"
    )
    embedding = (
        root
        / "pyannote"
        / "wespeaker-voxceleb-resnet34-LM"
        / "pytorch_model.bin"
    )
    clearvoice = root / "checkpoints" / "MossFormer2_SS_16K"
    paths = OverlapPaths(
        pipeline=pipeline,
        pipeline_config=pipeline / "config.yaml",
        segmentation_checkpoint=segmentation,
        embedding_checkpoint=embedding,
        clearvoice=clearvoice,
        clearvoice_checkpoint=_clearvoice_checkpoint(clearvoice),
    )
    if require_existing:
        missing = [
            name
            for name, path in (
                ("pipeline", paths.pipeline_config),
                ("segmentation", paths.segmentation_checkpoint),
                ("embedding", paths.embedding_checkpoint),
                ("clearvoice_pointer", clearvoice / "last_best_checkpoint"),
                ("clearvoice_checkpoint", paths.clearvoice_checkpoint),
            )
            if not path.is_file()
        ]
        if missing:
            raise ModelError(
                "missing overlap assets: " + ", ".join(missing)
            )
    return paths


def acquire_overlap_assets(
    model_dir: Path,
    snapshot_download: Callable[..., Any] | None = None,
) -> OverlapPaths:
    if snapshot_download is None:
        try:
            from modelscope import snapshot_download as modelscope_download
        except ImportError as exc:
            raise ModelError(f"unable to import ModelScope: {exc}") from exc
        snapshot_download = modelscope_download

    root = Path(model_dir).expanduser().resolve() / "huggingface"
    for model_id, relative, marker in OVERLAP_MODELS:
        target = root / relative
        if target.exists():
            marker_path = target / marker
            valid = marker_path.is_file()
            if marker == "last_best_checkpoint" and valid:
                valid = _clearvoice_checkpoint(target).is_file()
            if not valid:
                raise ModelError(f"incomplete overlap asset: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / (
            f".{target.name}.download-{uuid.uuid4().hex}"
        )
        try:
            snapshot_download(model_id, local_dir=str(temporary))
            marker_path = temporary / marker
            valid = marker_path.is_file()
            if marker == "last_best_checkpoint" and valid:
                valid = _clearvoice_checkpoint(temporary).is_file()
            if not valid:
                raise ModelError(
                    f"downloaded overlap asset is incomplete: {model_id}"
                )
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    return resolve_overlap_paths(model_dir)


def _directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _default_git_runner(
    arguments: list[str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _checked_git(
    arguments: list[str],
    cwd: Path | None = None,
    runner: Callable[..., Any] | None = None,
) -> Any:
    completed = (runner or _default_git_runner)(arguments, cwd=cwd)
    if completed.returncode != 0:
        raise ModelError(
            f"git command failed: {' '.join(arguments)}: "
            f"{str(completed.stderr).strip()}"
        )
    return completed


def _source_head(
    path: Path,
    runner: Callable[..., Any] | None = None,
) -> str:
    completed = _checked_git(["rev-parse", "HEAD"], cwd=path, runner=runner)
    return str(completed.stdout).strip()


def _valid_firered_asset(name: str, path: Path) -> bool:
    if name == "vad":
        return (path / "VAD" / "model.pth.tar").is_file()
    return (path / "model.pth.tar").is_file()


def verify_models(model_dir: Path) -> ModelVerification:
    root = Path(model_dir).expanduser().resolve()
    paths = resolve_firered_paths(root, require_existing=False)
    missing: list[str] = []
    if not _directory_has_files(paths.source):
        missing.append("firered_source")
    else:
        try:
            if _source_head(paths.source) != FIRERED_COMMIT:
                missing.append("firered_source_commit")
        except ModelError:
            missing.append("firered_source_commit")
    for name in FIRERED_MODEL_NAMES:
        if not _valid_firered_asset(name, getattr(paths, name)):
            missing.append(f"firered_{name}")
    speaker_vad = root / "modelscope" / "iic" / SPEAKER_VAD_DIR
    if not _directory_has_files(speaker_vad):
        missing.append("speaker_vad")
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
    _checked_git(arguments, cwd=cwd)


def acquire_firered_assets(
    model_dir: Path,
    source: str = "modelscope",
    snapshot_download: Callable[..., Any] | None = None,
    hf_download: Callable[..., Any] | None = None,
    git_runner: Callable[..., Any] | None = None,
) -> FireRedPaths:
    """Acquire pinned source and three model directories using atomic renames."""

    if source not in {"modelscope", "huggingface"}:
        raise ValueError("source must be modelscope or huggingface")
    root = Path(model_dir).expanduser().resolve()
    paths = resolve_firered_paths(root, require_existing=False)
    firered_root = paths.source.parent
    firered_root.mkdir(parents=True, exist_ok=True)

    for name, target in (
        ("source", paths.source),
        ("vad", paths.vad),
        ("aed", paths.aed),
        ("punc", paths.punc),
    ):
        if not target.exists():
            continue
        valid = (
            _directory_has_files(target)
            if name == "source"
            else _valid_firered_asset(name, target)
        )
        if not valid:
            raise ModelError(
                f"existing FireRed asset is incomplete: {target}"
            )
        if name == "source" and _source_head(
            target,
            runner=git_runner,
        ) != FIRERED_COMMIT:
            raise ModelError(
                f"existing FireRed source has wrong commit: {target}"
            )

    if not paths.source.exists():
        temporary = firered_root / f".source.download-{uuid.uuid4().hex}"
        try:
            _checked_git(
                ["clone", FIRERED_REPOSITORY, str(temporary)],
                runner=git_runner,
            )
            _checked_git(
                ["checkout", "--detach", FIRERED_COMMIT],
                cwd=temporary,
                runner=git_runner,
            )
            if _source_head(temporary, runner=git_runner) != FIRERED_COMMIT:
                raise ModelError("downloaded FireRed source commit mismatch")
            if not (temporary / "pyproject.toml").is_file():
                raise ModelError("downloaded FireRed source is incomplete")
            temporary.replace(paths.source)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    if source == "modelscope" and snapshot_download is None:
        try:
            from modelscope import snapshot_download as modelscope_download
        except ImportError as exc:
            raise ModelError(f"unable to import ModelScope: {exc}") from exc
        snapshot_download = modelscope_download
    if source == "huggingface" and hf_download is None:
        try:
            from huggingface_hub import snapshot_download as huggingface_download
        except ImportError as exc:
            raise ModelError(f"unable to import Hugging Face Hub: {exc}") from exc
        hf_download = huggingface_download

    for name in FIRERED_MODEL_NAMES:
        target = getattr(paths, name)
        if target.exists():
            continue
        temporary = firered_root / (
            f".{target.name}.download-{uuid.uuid4().hex}"
        )
        try:
            if source == "modelscope":
                assert snapshot_download is not None
                snapshot_download(
                    FIRERED_MODELS[name][0],
                    local_dir=str(temporary),
                )
            else:
                assert hf_download is not None
                hf_download(
                    FIRERED_MODELS[name][1],
                    local_dir=str(temporary),
                )
            if not _valid_firered_asset(name, temporary):
                raise ModelError(
                    f"downloaded FireRed {name} asset is incomplete"
                )
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    return resolve_firered_paths(root, require_existing=True)


def download_models(model_dir: Path) -> ModelVerification:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ModelError(f"unable to import ModelScope: {exc}") from exc

    root = Path(model_dir).expanduser().resolve()
    acquire_firered_assets(root, snapshot_download=snapshot_download)
    iic_root = root / "modelscope" / "iic"
    iic_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        f"iic/{SPEAKER_VAD_DIR}",
        local_dir=str(iic_root / SPEAKER_VAD_DIR),
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

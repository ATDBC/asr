from pathlib import Path

import pytest

from funasr_transcriber.errors import ModelError
from funasr_transcriber.model_registry import (
    DIARIZATION_EMBEDDING_DIR,
    FIRERED_COMMIT,
    SPEAKER_VAD_DIR,
    acquire_overlap_assets,
    resolve_firered_paths,
    resolve_overlap_paths,
    verify_models,
)


def _complete_layout(tmp_path: Path) -> None:
    paths = resolve_firered_paths(tmp_path, require_existing=False)
    for path in (paths.vad, paths.aed, paths.punc):
        path.mkdir(parents=True)
        (path / "model.pth.tar").write_bytes(b"model")
    (paths.vad / "VAD").mkdir()
    (paths.vad / "VAD" / "model.pth.tar").write_bytes(b"model")
    speaker_vad = tmp_path / "modelscope" / "iic" / SPEAKER_VAD_DIR
    speaker_vad.mkdir(parents=True)
    (speaker_vad / "model.bin").write_bytes(b"model")
    embedding = tmp_path / "modelscope" / "iic" / DIARIZATION_EMBEDDING_DIR
    embedding.mkdir(parents=True)
    (embedding / "campplus_cn_en_common.pt").write_bytes(b"model")
    (tmp_path / "3d-speaker" / "3D-Speaker" / "speakerlab").mkdir(
        parents=True
    )
    paths.source.mkdir(parents=True)
    (paths.source / "pyproject.toml").write_text("[project]\n", "utf-8")


def test_firered_paths_use_isolated_layout(tmp_path: Path):
    paths = resolve_firered_paths(tmp_path, require_existing=False)

    assert paths.source == (tmp_path / "fireredasr2s" / "source").resolve()
    assert paths.vad == (tmp_path / "fireredasr2s" / "FireRedVAD").resolve()
    assert paths.aed == (
        tmp_path / "fireredasr2s" / "FireRedASR2-AED"
    ).resolve()
    assert not hasattr(paths, "lid")


def test_verify_models_lists_missing_components(tmp_path: Path):
    result = verify_models(tmp_path)

    assert result.ok is False
    assert "firered_source" in result.missing
    assert "firered_aed" in result.missing
    assert "speaker_vad" in result.missing
    assert "3d_speaker_repo" in result.missing
    assert "diarization_embedding" in result.missing


def test_verify_models_accepts_complete_layout(monkeypatch, tmp_path: Path):
    _complete_layout(tmp_path)
    monkeypatch.setattr(
        "funasr_transcriber.model_registry._source_head",
        lambda path, runner=None: FIRERED_COMMIT,
    )

    result = verify_models(tmp_path)

    assert result.ok is True
    assert result.missing == ()


def test_verify_models_does_not_require_firered_lid(
    monkeypatch,
    tmp_path: Path,
):
    _complete_layout(tmp_path)
    monkeypatch.setattr(
        "funasr_transcriber.model_registry._source_head",
        lambda path, runner=None: FIRERED_COMMIT,
    )

    result = verify_models(tmp_path)

    assert result.ok is True
    assert "firered_lid" not in result.missing


def test_verify_models_rejects_wrong_firered_commit(monkeypatch, tmp_path: Path):
    _complete_layout(tmp_path)
    monkeypatch.setattr(
        "funasr_transcriber.model_registry._source_head",
        lambda path, runner=None: "wrong-commit",
    )

    result = verify_models(tmp_path)

    assert result.ok is False
    assert "firered_source_commit" in result.missing


def test_resolve_firered_models_raises_with_named_missing_models(
    tmp_path: Path,
):
    with pytest.raises(ModelError, match="firered_vad"):
        resolve_firered_paths(tmp_path, require_existing=True)


def _write_overlap_snapshot(model_id: str, local_dir: str) -> None:
    target = Path(local_dir)
    target.mkdir(parents=True)
    if model_id == "pyannote/speaker-diarization-3.1":
        (target / "config.yaml").write_text(
            "pipeline:\n  params: {}\n", encoding="utf-8"
        )
    elif model_id in {
        "pyannote/segmentation-3.0",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
    }:
        (target / "pytorch_model.bin").write_bytes(b"model")
    elif model_id == "alibabasglab/MossFormer2_SS_16K":
        (target / "last_best_checkpoint").write_text(
            "last_best_checkpoint.pt\n", encoding="utf-8"
        )
        (target / "last_best_checkpoint.pt").write_bytes(b"model")


def test_acquire_overlap_assets_uses_modelscope_mirrors(tmp_path: Path):
    calls = []

    def download(model_id, local_dir):
        calls.append(model_id)
        _write_overlap_snapshot(model_id, local_dir)

    paths = acquire_overlap_assets(tmp_path, snapshot_download=download)

    assert calls == [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        "alibabasglab/MossFormer2_SS_16K",
    ]
    assert paths.pipeline_config.is_file()
    assert paths.segmentation_checkpoint.is_file()
    assert paths.embedding_checkpoint.is_file()
    assert paths.clearvoice_checkpoint.is_file()
    assert resolve_overlap_paths(tmp_path) == paths


def test_acquire_overlap_assets_rejects_incomplete_existing_directory(
    tmp_path: Path,
):
    paths = resolve_overlap_paths(tmp_path, require_existing=False)
    paths.pipeline.mkdir(parents=True)

    with pytest.raises(ModelError, match="incomplete overlap asset"):
        acquire_overlap_assets(tmp_path, snapshot_download=lambda *args: None)

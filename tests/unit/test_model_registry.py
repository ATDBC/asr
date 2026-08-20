from pathlib import Path

import pytest

from funasr_transcriber.errors import ModelError
from funasr_transcriber.model_registry import (
    DIARIZATION_EMBEDDING_DIR,
    FIRERED_COMMIT,
    SPEAKER_VAD_DIR,
    resolve_firered_paths,
    verify_models,
)


def _complete_layout(tmp_path: Path) -> None:
    paths = resolve_firered_paths(tmp_path, require_existing=False)
    for path in (paths.vad, paths.lid, paths.aed, paths.punc):
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

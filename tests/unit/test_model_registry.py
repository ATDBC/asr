from pathlib import Path

import pytest

from funasr_transcriber.errors import ModelError
from funasr_transcriber.model_registry import (
    MODEL_DIRS,
    resolve_model_paths,
    verify_models,
)


def test_model_paths_use_independent_modelscope_layout(tmp_path: Path):
    paths = resolve_model_paths(tmp_path, require_existing=False)

    assert paths.asr == (
        tmp_path / "modelscope" / "iic" / MODEL_DIRS["asr"]
    ).resolve()


def test_verify_models_lists_missing_components(tmp_path: Path):
    result = verify_models(tmp_path)

    assert result.ok is False
    assert "asr" in result.missing
    assert "3d_speaker_repo" in result.missing
    assert "diarization_embedding" in result.missing


def test_verify_models_accepts_complete_layout(tmp_path: Path):
    paths = resolve_model_paths(tmp_path, require_existing=False)
    for path in (paths.asr, paths.vad, paths.punc):
        path.mkdir(parents=True)
        (path / "model.bin").write_bytes(b"model")
    embedding = (
        tmp_path
        / "modelscope"
        / "iic"
        / "speech_campplus_sv_zh_en_16k-common_advanced"
    )
    embedding.mkdir(parents=True)
    (embedding / "campplus_cn_en_common.pt").write_bytes(b"model")
    repo = tmp_path / "3d-speaker" / "3D-Speaker" / "speakerlab"
    repo.mkdir(parents=True)

    result = verify_models(tmp_path)

    assert result.ok is True
    assert result.missing == ()


def test_resolve_models_raises_with_named_missing_models(tmp_path: Path):
    with pytest.raises(ModelError, match="asr"):
        resolve_model_paths(tmp_path, require_existing=True)


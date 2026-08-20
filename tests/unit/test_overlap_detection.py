from funasr_transcriber import overlap_worker
import pytest


def test_detect_uses_diarization_overlap_timeline(monkeypatch):
    class Segment:
        start = 1.25
        end = 2.75

    class Timeline:
        def support(self):
            return [Segment()]

    class Annotation:
        def get_overlap(self):
            return Timeline()

    monkeypatch.setattr(
        overlap_worker,
        "_load_detector",
        lambda request: (lambda path: Annotation(), "detector"),
    )

    result = overlap_worker._detect({"audio_path": "recording.wav"})

    assert result["windows"] == [
        {
            "id": "overlap-0001",
            "start_ms": 1250,
            "end_ms": 2750,
        }
    ]


def test_worker_configures_shared_model_cache_before_imports(
    monkeypatch,
    tmp_path,
):
    changed_to = []
    monkeypatch.setattr(overlap_worker.os, "chdir", changed_to.append)

    overlap_worker._configure_cache({"cache_dir": str(tmp_path)})

    assert overlap_worker.os.environ["HF_HOME"] == str(tmp_path.resolve())
    assert overlap_worker.os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert changed_to == [tmp_path.resolve()]


def test_local_detector_config_rewrites_modelscope_checkpoints(
    monkeypatch,
    tmp_path,
):
    pipeline = tmp_path / "pyannote" / "speaker-diarization-3.1"
    segmentation = tmp_path / "pyannote" / "segmentation-3.0"
    embedding = (
        tmp_path / "pyannote" / "wespeaker-voxceleb-resnet34-LM"
    )
    pipeline.mkdir(parents=True)
    segmentation.mkdir(parents=True)
    embedding.mkdir(parents=True)
    (pipeline / "config.yaml").write_text(
        "pipeline:\n"
        "  name: pyannote.audio.pipelines.SpeakerDiarization\n"
        "  params:\n"
        "    segmentation: pyannote/segmentation-3.0\n"
        "    embedding: pyannote/wespeaker-voxceleb-resnet34-LM\n",
        encoding="utf-8",
    )
    (segmentation / "pytorch_model.bin").write_bytes(b"segmentation")
    (embedding / "pytorch_model.bin").write_bytes(b"embedding")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    config = overlap_worker._local_detector_config(
        {
            "cache_dir": str(tmp_path),
            "model_id": "pyannote/speaker-diarization-3.1",
        }
    )

    assert config["pipeline"]["params"]["segmentation"] == {
        "checkpoint": str(
            (segmentation / "pytorch_model.bin").resolve()
        ),
        "strict": False,
    }
    assert config["pipeline"]["params"]["embedding"] == {
        "checkpoint": str((embedding / "pytorch_model.bin").resolve()),
        "strict": False,
    }


def test_clearvoice_checkpoint_validation_rejects_missing_target(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints" / "MossFormer2_SS_16K"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "last_best_checkpoint").write_text(
        "missing.pt\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="ClearVoice checkpoint is missing"):
        overlap_worker._require_clearvoice_checkpoint(
            tmp_path, "MossFormer2_SS_16K"
        )

from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_readme_documents_isolated_firered_operation():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for required in (
        ".venv-asr",
        "4e7d9aaf4482a47cec1724807026b9b151926eb5",
        "FireRedVAD",
        "FireRedASR2-AED",
        "FireRedPunc",
        "23 GiB",
        "12 GiB",
        "9.04 GiB",
        "ModelScope",
        "Hugging Face",
        "cuda:0",
    ):
        assert required in readme
    assert "不会尝试终止或清理其他项目的进程" in readme
    assert "FireRedLID" not in readme
    assert "最多加载两次" in readme
    assert ".venv：FunASR" not in readme


def test_architecture_describes_current_worker_lifecycle():
    architecture = (
        ROOT / "docs" / "architecture" / "current-transcription-pipeline.md"
    ).read_text(encoding="utf-8")

    assert "FireRedWorkerClient" in architecture
    assert "FireRed worker 关闭" in architecture
    assert "SpeakerDiarizer 关闭" in architecture
    assert "segment_batch_size" in architecture
    assert "FireRedLID" not in architecture
    assert "GPU 0" in architecture
    for legacy_name in ("FunASR AutoModel", "Para" + "former"):
        assert legacy_name not in architecture


def test_architecture_documents_hot_parallel_service():
    architecture = (
        ROOT / "docs" / "architecture" / "current-transcription-pipeline.md"
    ).read_text(encoding="utf-8")

    for required in (
        "Unix socket",
        "ASR 与 diarization 并行",
        "--profile-resources",
        "--enable-overlap",
        "service stop",
    ):
        assert required in architecture


def test_release_checklist_requires_firered_and_gpu_evidence():
    checklist = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    for required in (
        "FireRedASR2S source commit",
        "FireRedVAD / FireRedASR2-AED / FireRedPunc",
        "GPU 0",
        "23 GiB total",
        "12 GiB free",
        "GPU release evidence",
    ):
        assert required in checklist

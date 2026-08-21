from pathlib import Path


def test_production_source_has_no_legacy_asr_route_or_single_job_lifecycle():
    source = Path("src/funasr_transcriber")
    text = "\n".join(
        path.read_text("utf-8")
        for path in source.glob("*.py")
    )

    assert "from funasr import AutoModel" not in text
    assert "FunAsrEngine" not in text
    assert "build_asr_model" not in text
    assert "def execute_model_phases" not in text

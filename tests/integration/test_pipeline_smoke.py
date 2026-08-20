import os
from pathlib import Path

import pytest

from funasr_transcriber.config import load_config
from funasr_transcriber.cli import execute_model_phases
from funasr_transcriber.firered_client import FireRedWorkerClient
from funasr_transcriber.gpu import require_gpu_capacity
from funasr_transcriber.speaker_diarization import SpeakerDiarizer


@pytest.mark.skipif(
    os.getenv("FUNASR_TRANSCRIBER_RUN_MODEL_TESTS") != "1",
    reason="model integration tests are disabled",
)
def test_real_non_overlap_pipeline_returns_turns(tmp_path: Path):
    fixture = Path(__file__).parents[1] / "fixtures" / "short_mandarin.wav"
    if not fixture.is_file():
        pytest.skip("short_mandarin.wav fixture is not installed")
    model_dir = Path(os.environ["FUNASR_TRANSCRIBER_MODEL_DIR"])
    worker_python = Path(
        os.environ.get(
            "FUNASR_TRANSCRIBER_ASR_WORKER_PYTHON",
            Path(__file__).parents[2] / ".venv-asr" / "bin" / "python",
        )
    )
    config = load_config(
        cli_overrides={
            "paths.model_dir": str(model_dir),
            "asr.worker_python": str(worker_python),
            "general.device": "cuda:0",
        }
    )
    result = execute_model_phases(
        fixture.resolve(),
        config,
        overlap_enabled=False,
        asr_client_factory=lambda stage: FireRedWorkerClient(
            worker_python=worker_python,
            model_root=model_dir,
            device="cuda:0",
            timeout_seconds=1800,
            stderr_log_path=tmp_path / f"{stage}.log",
        ),
        diarizer_factory=lambda: SpeakerDiarizer.from_config(config),
        overlap_preparer_factory=lambda diarizer: None,
        gpu_check=lambda stage: require_gpu_capacity("cuda:0"),
    )

    assert result.transcript.turns
    assert 1 <= result.transcript.speaker_count <= 10
    assert result.transcript.full_text
    assert result.diagnostics.asr["languages"]["primary"]

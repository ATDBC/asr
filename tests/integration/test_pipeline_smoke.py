import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from funasr_transcriber.cli import submit_to_service
from funasr_transcriber.config import load_config
from funasr_transcriber.service_protocol import (
    ServiceClient,
    ServicePaths,
    ServiceProtocolError,
)


@pytest.mark.skipif(
    os.getenv("FUNASR_TRANSCRIBER_RUN_MODEL_TESTS") != "1",
    reason="model integration tests are disabled",
)
def test_real_non_overlap_pipeline_uses_resident_service(tmp_path: Path):
    fixture = Path(__file__).parents[1] / "fixtures" / "short_mandarin.wav"
    if not fixture.is_file():
        pytest.skip("short_mandarin.wav fixture is not installed")
    model_dir = Path(os.environ["FUNASR_TRANSCRIBER_MODEL_DIR"]).resolve()
    worker_python = Path(
        os.environ.get(
            "FUNASR_TRANSCRIBER_ASR_WORKER_PYTHON",
            Path(__file__).parents[2] / ".venv-asr" / "bin" / "python",
        )
    ).resolve()
    work_dir = tmp_path / "work"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            (
                "[general]",
                'device = "cuda:0"',
                "[paths]",
                f'model_dir = "{model_dir}"',
                f'work_dir = "{work_dir}"',
                "[asr]",
                f'worker_python = "{worker_python}"',
                "timeout_seconds = 1800",
                "[service]",
                "startup_timeout_seconds = 1800",
                "",
            )
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    paths = ServicePaths.from_work_dir(work_dir)
    output_dir = tmp_path / "output"
    try:
        response = submit_to_service(
            SimpleNamespace(
                input_audio=fixture.resolve(),
                output_dir=output_dir,
                overwrite=False,
                keep_intermediate=False,
                enable_overlap=False,
                disable_overlap=True,
                profile_resources=True,
                config=config_path,
            ),
            config,
        )
        result = response["result"]
        assert isinstance(result, dict)
        assert result["output_dir"] == str(output_dir)
        assert ServiceClient(paths, timeout_seconds=10).health(uuid.uuid4().hex)[
            "status"
        ] == "ready"

        transcript = json.loads(
            (output_dir / "transcript.json").read_text(encoding="utf-8")
        )
        diagnostics = json.loads(
            (output_dir / "diagnostics.json").read_text(encoding="utf-8")
        )
        assert transcript["turns"]
        assert 1 <= transcript["speaker_count"] <= 10
        assert transcript["full_text"]
        assert "languages" not in diagnostics["asr"]
        assert diagnostics["resource_profile"]["enabled"] is True
    finally:
        try:
            ServiceClient(paths, timeout_seconds=1800).stop_and_wait(uuid.uuid4().hex)
        except ServiceProtocolError:
            pass

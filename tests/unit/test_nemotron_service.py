from pathlib import Path

import pytest

from funasr_transcriber.config import load_config
from funasr_transcriber.service import (
    HotModelHost,
    ServiceJobRunner,
    build_hot_model_host,
)
from funasr_transcriber.service_protocol import ServicePaths
import funasr_transcriber.service as service_module


def test_selected_nemotron_pid_is_profiled():
    runner = ServiceJobRunner.__new__(ServiceJobRunner)
    runner.hot_host = type(
        "Host",
        (),
        {
            "speaker_owner": type("Speaker", (), {"pid": 451})(),
            "asr_owner": None,
        },
    )()
    runner._active_overlap_pid = None

    assert runner._owned_pid_getters()["speaker"]() == 451


def test_selected_nemotron_factory_and_gpu_admission(
    tmp_path: Path, monkeypatch
):
    import funasr_transcriber.nemotron_client as nemotron_module

    calls = []
    marker = object()
    monkeypatch.setattr(
        nemotron_module,
        "NemotronWorkerClient",
        lambda **kwargs: calls.append(kwargs) or marker,
    )
    monkeypatch.setattr(
        service_module,
        "require_gpu_capacity",
        lambda device, **kwargs: calls.append((device, kwargs)),
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[diarization]\nbackend = "nemotron3"\n',
        encoding="utf-8",
    )
    config = load_config(config_path)
    host = build_hot_model_host(
        config, ServicePaths.from_work_dir(tmp_path)
    )

    host._gpu_check()
    assert host._diarizer_factory() is marker
    assert calls[0] == (
        "cuda:0",
        {
            "minimum_total_mib": config.service.minimum_total_mib,
            "minimum_free_mib": config.service.minimum_free_mib,
        },
    )
    assert calls[1]["worker_python"] == config.nemotron.worker_python
    assert calls[1]["model_dir"] == config.paths.model_dir


def test_gpu_admission_failure_starts_no_owner():
    events = []
    host = HotModelHost(
        asr_factory=lambda: events.append("asr"),
        diarizer_factory=lambda: events.append("speaker"),
        gpu_check=lambda: (_ for _ in ()).throw(
            RuntimeError("insufficient GPU0 memory")
        ),
    )
    with pytest.raises(RuntimeError, match="GPU0"):
        host.start()
    assert events == []

def test_selected_branch_transcribes_original_overlap_crops(tmp_path: Path):
    import numpy as np
    import soundfile as sf
    from funasr_transcriber.firered_parallel import CropTranscript
    from funasr_transcriber.schemas import (
        AsrResult,
        RecognizedSentence,
        SpeakerSegment,
        TimedToken,
    )
    from funasr_transcriber.service import run_nemotron_branches

    normalized = tmp_path / "normalized.wav"
    sf.write(
        normalized,
        np.zeros(8000 * 16, dtype="int16"),
        16000,
        subtype="PCM_16",
    )

    class Speaker:
        def diarize(self, path):
            return (
                SpeakerSegment(1, 0, 6000),
                SpeakerSegment(2, 5000, 8000),
            )

    class Aed:
        def refine_vad(self, path, ranges):
            return ranges

    class Asr:
        worker0 = Aed()
        last_response = {"metrics": {"scheduler": {}, "parallel": {}}}

        def transcribe_crops(self, plan):
            assert plan.overlap_duration_ms == 1000
            return tuple(
                CropTranscript(
                    crop,
                    AsrResult(
                        "甲" if crop.speaker_id == 1 else "乙",
                        (
                            TimedToken(
                                "甲" if crop.speaker_id == 1 else "乙",
                                5500 if crop.speaker_id == 1 else 500,
                                5800 if crop.speaker_id == 1 else 800,
                            ),
                        ),
                        sentences=(
                            RecognizedSentence(
                                "甲" if crop.speaker_id == 1 else "乙",
                                5500 if crop.speaker_id == 1 else 500,
                                5800 if crop.speaker_id == 1 else 800,
                                0.9,
                            ),
                        ),
                        duration_ms=crop.end_ms - crop.start_ms,
                    ),
                )
                for crop in plan.crops
            )

    branches, attributed, crop_plan = run_nemotron_branches(
        normalized, Asr(), Speaker(), tmp_path / "crops", False
    )
    assert branches.asr_result.text == "甲乙"
    assert [token.speaker_id for token in attributed] == [1, 2]
    assert all(token.overlap for token in attributed)
    assert crop_plan.overlap_duration_ms == 1000

def test_selected_service_rejects_request_overlap_override_before_job(
    tmp_path: Path
):
    from funasr_transcriber.errors import TranscriptionError
    from funasr_transcriber.service_protocol import ServiceRequest

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "\n".join(
            (
                "[paths]",
                f'work_dir = "{tmp_path / "work"}"',
                "[diarization]",
                'backend = "nemotron3"',
            )
        ),
        encoding="utf-8",
    )
    runner = ServiceJobRunner(load_config(config_file), hot_host=object())
    output = tmp_path / "output"
    request = ServiceRequest(
        identifier="conflict",
        action="submit",
        input_audio=tmp_path / "input.wav",
        output_dir=output,
        overlap_enabled=True,
    )

    with pytest.raises(TranscriptionError, match="Nemotron.*overlap"):
        runner.run(request)
    assert not output.exists()
    assert not (runner.config.paths.work_dir / "job-conflict").exists()

def test_selected_service_publishes_two_overlapping_turns(
    tmp_path: Path
):
    import json

    import numpy as np
    import soundfile as sf

    from funasr_transcriber.audio import AudioInfo
    from funasr_transcriber.firered_parallel import CropTranscript
    from funasr_transcriber.schemas import (
        AsrResult,
        RecognizedSentence,
        SpeakerSegment,
        TimedToken,
    )
    from funasr_transcriber.service_protocol import ServiceRequest

    config = load_config(
        cli_overrides={
            "paths.work_dir": str(tmp_path / "work"),
            "diarization.backend": "nemotron3",
        }
    )
    source = tmp_path / "input.wav"
    source.write_bytes(b"source")

    class Speaker:
        pid = 451

        def diarize(self, path):
            return (
                SpeakerSegment(1, 0, 6000),
                SpeakerSegment(2, 5000, 8000),
            )

    class AedWorker:
        def refine_vad(self, path, ranges):
            return ranges

    class AsrOwner:
        worker0 = AedWorker()
        last_response = {}

        def transcribe_crops(self, plan):
            self.last_response = {
                "metrics": {
                    "scheduler": {
                        "crop_plans": [
                            {"crop_index": crop.index, "path": str(crop.path)}
                            for crop in plan.crops
                        ],
                        "worker_crop_indexes": [[0, 1]],
                    },
                    "parallel": {"no_speech_crop_count": 0},
                }
            }
            return tuple(
                CropTranscript(
                    crop,
                    AsrResult(
                        "甲" if crop.speaker_id == 1 else "乙",
                        (
                            TimedToken(
                                "甲" if crop.speaker_id == 1 else "乙",
                                5500 if crop.speaker_id == 1 else 500,
                                5800 if crop.speaker_id == 1 else 800,
                            ),
                        ),
                        sentences=(
                            RecognizedSentence(
                                "甲" if crop.speaker_id == 1 else "乙",
                                5500 if crop.speaker_id == 1 else 500,
                                5800 if crop.speaker_id == 1 else 800,
                                0.9,
                            ),
                        ),
                        duration_ms=crop.end_ms - crop.start_ms,
                    ),
                )
                for crop in plan.crops
            )

        def cleanup_cuda_cache(self):
            return {"seconds": 0.0, "workers": []}

    asr = AsrOwner()

    class Host:
        started = True
        asr_owner = asr
        speaker_owner = Speaker()
        inventories = {
            "speaker": {
                "device": "cuda:0",
                "dtype": "torch.float32",
                "model_revision": "pinned",
                "checkpoint_sha256": "a" * 64,
                "threshold": 0.5,
            }
        }

        def warmup(self):
            return {"performed": False}

    def normalize(source_path, destination, max_seconds):
        sf.write(
            destination,
            np.zeros(8000 * 16, dtype="int16"),
            16000,
            subtype="PCM_16",
        )

    runner = ServiceJobRunner(
        config,
        hot_host=Host(),
        probe=lambda path, max_seconds: AudioInfo(
            Path(path), 8000, 16000, 1, "wav"
        ),
        normalize=normalize,
        gpu_sampler=None,
    )
    output = tmp_path / "result"
    runner.run(
        ServiceRequest(
            identifier="nemotron-job",
            action="submit",
            input_audio=source,
            output_dir=output,
            config_fingerprint="test",
        )
    )

    transcript = json.loads(
        (output / "transcript.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (output / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert transcript["schema_version"] == "1.0"
    assert transcript["full_text"] == "甲乙"
    assert [turn["speaker"] for turn in transcript["turns"]] == [1, 2]
    assert all(turn["overlap"] for turn in transcript["turns"])
    assert "说话人1：甲" in (output / "transcript.txt").read_text()
    assert "说话人2：乙" in (output / "transcript.txt").read_text()
    assert diagnostics["runtime"]["nemotron"]["device"] == "cuda:0"
    assert "crop_plans" not in diagnostics["scheduler"]

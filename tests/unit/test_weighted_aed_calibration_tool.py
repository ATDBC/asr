from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import funasr_transcriber.weighted_aed_calibration as calibration


_PATH = Path(__file__).parents[2] / "tools" / "calibrate_weighted_aed.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("weighted_aed_tool", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(tmp_path: Path) -> Path:
    inventory = []
    for source_index in range(10):
        audio = tmp_path / f"source-{source_index}.wav"
        audio.write_bytes(f"audio-{source_index}".encode())
        segments = [
            {
                "index": index,
                "start_ms": index * 10_000,
                "end_ms": index * 10_000 + 1_000 + index * 10,
            }
            for index in range(32)
        ]
        inventory.append(
            {
                "audio_path": str(audio),
                "source_recording_group": f"source-{source_index}",
                "source_checksum": calibration.sha256_file(audio),
                "segments_checksum": calibration.canonical_sha256(segments),
                "segments": segments,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"metadata": {"source_inventory": inventory}}),
        encoding="utf-8",
    )
    return manifest


class FakeClient:
    constructed = 0
    calls = 0
    worker_pythons = []

    def __init__(self, **options):
        type(self).constructed += 1
        type(self).worker_pythons.append(options["worker_python"])
        assert options["device"] == "cuda:0"
        assert options["enable_vad"] is False
        assert options["enable_punc"] is False

    def start(self):
        return self

    def status(self):
        return {
            "runtime": {
                "device": "cuda:0",
                "dtype": "torch.bfloat16",
                "compute_capability": "8.6",
                "torch_version": "2.test",
                "cuda_version": "12.test",
            }
        }

    def run_aed_plan(self, audio_path, batch_plan, vad_segments_ms=None):
        type(self).calls += 1
        count = len(batch_plan[0])
        longest = max(
            (item["end_ms"] - item["start_ms"]) / 1000
            for item in batch_plan[0]
        )
        actual = longest * (0.2 + 0.01 * count)
        return {
            "aed_results": [
                {"index": item["index"], "result": {"text": "ok"}}
                for item in batch_plan[0]
            ],
            "batch_metrics": [
                {
                    "actual_seconds": actual,
                    "peak_allocated_mib": 100,
                    "peak_reserved_mib": 120,
                }
            ],
        }

    def close(self):
        pass


def _gpu():
    return SimpleNamespace(
        index=0,
        name="NVIDIA Test GPU",
        total_mib=24_576,
        free_mib=20_000,
        driver_version="580.1",
    )


def test_aed_output_fingerprint_ignores_request_and_timing_metadata():
    tool = _load_tool()
    first = [{
        "index": 0,
        "result": {
            "uttid": "request-a_s0_e1000",
            "rtf": "0.1",
            "text": "相同",
            "confidence": 0.9,
            "timestamp": [["相", 0.0, 0.5], ["同", 0.5, 1.0]],
        },
    }]
    second = json.loads(json.dumps(first, ensure_ascii=False))
    second[0]["result"]["uttid"] = "request-b_s0_e1000"
    second[0]["result"]["rtf"] = "0.2"

    assert tool._aed_output_fingerprint(first) == tool._aed_output_fingerprint(second)
    second[0]["result"]["text"] = "不同"
    assert tool._aed_output_fingerprint(first) != tool._aed_output_fingerprint(second)


def test_measure_case_sends_monotonic_vad_without_reordering_batch():
    tool = _load_tool()
    case = SimpleNamespace(
        audio_path=Path("/audio.wav"),
        segments=(
            {"index": 1, "start_ms": 300, "end_ms": 400},
            {"index": 0, "start_ms": 100, "end_ms": 200},
        ),
    )

    class Client:
        def run_aed_plan(self, audio_path, batch_plan, vad_segments_ms=None):
            assert [item["index"] for item in batch_plan[0]] == [1, 0]
            assert vad_segments_ms == [[100, 200], [300, 400]]
            return {
                "aed_results": [
                    {"index": item["index"], "result": {"text": "ok"}}
                    for item in batch_plan[0]
                ],
                "batch_metrics": [{
                    "actual_seconds": 1.0,
                    "peak_allocated_mib": 1,
                    "peak_reserved_mib": 2,
                }],
            }

    tool._measure_case(Client(), case)


def test_measure_case_accepts_results_filtered_by_worker():
    tool = _load_tool()
    case = SimpleNamespace(
        audio_path=Path("/audio.wav"),
        segments=(
            {"index": 0, "start_ms": 100, "end_ms": 200},
            {"index": 1, "start_ms": 300, "end_ms": 400},
        ),
    )

    class Client:
        def run_aed_plan(self, *args, **kwargs):
            return {
                "aed_results": [{"index": 1, "result": {"text": "ok"}}],
                "batch_metrics": [{"actual_seconds": 1.0}],
            }

    measurement = tool._measure_case(Client(), case)

    assert measurement["actual_seconds"] == 1.0


def test_tool_collects_artifact_and_resume_skips_completed_cases(
    tmp_path: Path,
    monkeypatch,
):
    tool = _load_tool()
    manifest = _write_manifest(tmp_path)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    real_python = tmp_path / "python-real"
    real_python.write_text("", encoding="utf-8")
    worker_python = tmp_path / "python"
    worker_python.symlink_to(real_python)
    output = tmp_path / "output"
    runtime = {
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "gpu_name": "NVIDIA Test GPU",
        "compute_capability": "8.6",
        "driver_version": "580.1",
        "cuda_version": "12.test",
        "torch_version": "2.test",
        "firered_source_commit": "f" * 40,
        "aed_checkpoint_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        tool,
        "build_expected_runtime_identity",
        lambda *args, **kwargs: dict(runtime),
    )
    args = tool._parser().parse_args(
        [
            "--manifest", str(manifest),
            "--model-dir", str(model_dir),
            "--worker-python", str(worker_python),
            "--output", str(output),
        ]
    )
    FakeClient.constructed = 0
    FakeClient.calls = 0
    FakeClient.worker_pythons = []

    assert tool.run(args, client_factory=FakeClient, gpu_query=lambda _: _gpu()) == 0
    assert FakeClient.worker_pythons == [worker_python.absolute()]
    assert (output / "plan.json").is_file()
    assert (output / "records.jsonl").is_file()
    assert (output / "calibration-report.json").is_file()
    assert (output / "weighted-aed-calibration.json").is_file()
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["warmups_per_case"] == 1
    assert FakeClient.calls == len(plan["cases"]) * (args.repeats + 1)
    first_calls = FakeClient.calls
    constructed = FakeClient.constructed

    try:
        tool.run(args, client_factory=FakeClient, gpu_query=lambda _: _gpu())
    except ValueError as exc:
        assert "--resume" in str(exc)
    else:
        raise AssertionError("existing plan was overwritten without --resume")
    assert FakeClient.constructed == constructed

    resume = tool._parser().parse_args(
        [
            "--manifest", str(manifest),
            "--model-dir", str(model_dir),
            "--worker-python", str(worker_python),
            "--output", str(output),
            "--resume",
        ]
    )
    assert tool.run(
        resume,
        client_factory=FakeClient,
        gpu_query=lambda _: _gpu(),
    ) == 0
    assert FakeClient.calls == first_calls


def test_tool_resume_rejects_manifest_change_before_worker(tmp_path: Path, monkeypatch):
    tool = _load_tool()
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "plan.json").write_text(
        json.dumps({"manifest_checksum": "0" * 64}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        manifest=manifest,
        audio_root=None,
        model_dir=tmp_path,
        worker_python=tmp_path / "python",
        output=output,
        repeats=3,
        use_half=True,
        resume=True,
    )
    constructed = FakeClient.constructed

    try:
        tool.run(args, client_factory=FakeClient, gpu_query=lambda _: _gpu())
    except ValueError as exc:
        assert "manifest" in str(exc)
    else:
        raise AssertionError("resume accepted a changed manifest")
    assert FakeClient.constructed == constructed

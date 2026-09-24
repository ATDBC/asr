from pathlib import Path
import os

import pytest


def test_benchmark_requires_explicit_gpu0_confirmation(tmp_path: Path):
    from tools.benchmark_nemotron_direct_crop import run_benchmark

    with pytest.raises(ValueError, match="confirm"):
        run_benchmark(
            config_path=tmp_path / "missing.toml",
            output_root=tmp_path / "result",
            confirm_gpu0=False,
        )
    assert not (tmp_path / "result").exists()


@pytest.mark.skipif(
    os.environ.get("RUN_NEMOTRON_GPU_TEST") != "1",
    reason="explicit GPU0 acceptance only",
)
def test_nemotron_gpu0_direct_crop_acceptance(tmp_path: Path):
    import json

    from tools.benchmark_nemotron_direct_crop import run_benchmark

    project_root = Path(__file__).parents[2]
    config_path = Path(
        os.environ.get(
            "NEMOTRON_BENCHMARK_CONFIG",
            project_root / "outputs" / "runtime" / "nemotron.toml",
        )
    )
    output_root = Path(
        os.environ.get(
            "NEMOTRON_BENCHMARK_OUTPUT_ROOT",
            project_root
            / "outputs"
            / "measurements"
            / "20260924T-nemotron3-direct-crop-acceptance",
        )
    )
    result = run_benchmark(
        config_path=config_path,
        output_root=output_root,
        confirm_gpu0=True,
    )

    assert result["admission"]["index"] == 0
    assert result["ending_gpu"]["index"] == 0
    overlap = result["cases"]["real_overlap"]
    turns = overlap["transcript"]["turns"]
    assert len({turn["speaker"] for turn in turns}) >= 2
    assert any(turn["overlap"] for turn in turns)
    assert overlap["diagnostics"]["runtime"]["nemotron"]["device"] == "cuda:0"
    deep = json.loads(
        (
            output_root
            / "real_overlap"
            / "diagnostics"
            / "scheduler.json"
        ).read_text(encoding="utf-8")
    )
    assert len(deep["crop_plans"]) >= 2
    assert len({item["path"] for item in deep["crop_plans"]}) >= 2
    for case in result["cases"].values():
        assert case["stage_seconds"]
        assert case["project_owned_peak"]

def test_benchmark_end_snapshot_follows_own_host_close(
    tmp_path: Path, monkeypatch
):
    import json

    import tools.benchmark_nemotron_direct_crop as benchmark
    from funasr_transcriber.config import load_config
    from funasr_transcriber.model_registry import ModelVerification

    config = load_config(
        cli_overrides={
            "diarization.backend": "nemotron3",
            "paths.work_dir": str(tmp_path / "work"),
            "paths.model_dir": str(tmp_path / "models"),
        }
    )
    source = tmp_path / "input.wav"
    source.write_bytes(b"test")
    monkeypatch.setattr(benchmark, "INPUTS", {"real_overlap": source})
    monkeypatch.setattr(benchmark, "load_config", lambda path: config)
    monkeypatch.setattr(
        benchmark, "verify_models",
        lambda path, speaker_backend: ModelVerification(True, ()),
    )
    monkeypatch.setattr(
        benchmark, "_observed_reference_text", lambda: []
    )

    class Host:
        closed = False
        inventories = {"speaker": {"device": "cuda:0"}}

        def start(self):
            return self

        def close(self):
            self.closed = True

    host = Host()
    monkeypatch.setattr(
        benchmark, "build_hot_model_host", lambda config, paths: host
    )
    snapshots = []

    def gpu_snapshot():
        snapshots.append(host.closed)
        return {
            "index": 0, "total_mib": 24576,
            "used_mib": 1000, "free_mib": 23576,
            "utilization_percent": 0,
        }

    monkeypatch.setattr(benchmark, "_gpu0_snapshot", gpu_snapshot)

    class Runner:
        def __init__(self, config, host):
            pass

        def run(self, request):
            output = Path(request.output_dir)
            output.mkdir()
            (output / "diagnostics.json").write_text(
                json.dumps({
                    "stage_seconds": {"end_to_end_after_load": 1.0},
                    "resource_profile": {
                        "process_peak_mib": {"speaker": 123}
                    },
                }),
                encoding="utf-8",
            )
            (output / "transcript.json").write_text(
                json.dumps({"turns": [], "full_text": ""}),
                encoding="utf-8",
            )

    monkeypatch.setattr(benchmark, "ServiceJobRunner", Runner)
    benchmark.run_benchmark(
        tmp_path / "config.toml",
        tmp_path / "result",
        confirm_gpu0=True,
    )

    assert snapshots == [False, True]

"""Explicit GPU0 benchmark for optional Nemotron direct-crop transcription.

This command runs no training and reads overlap reference text only after
inference, for an observational comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from time import perf_counter
import traceback
from typing import Any
import uuid

from funasr_transcriber.cli import require_physical_gpu_zero
from funasr_transcriber.config import load_config
from funasr_transcriber.model_registry import verify_models
from funasr_transcriber.service import ServiceJobRunner, build_hot_model_host
from funasr_transcriber.service_protocol import (
    ServicePaths,
    ServiceRequest,
    config_fingerprint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "real_overlap": (
        PROJECT_ROOT
        / "artifacts"
        / "release_validation_corpus"
        / "overlap"
        / "audio"
        / "real-overlap-0001.wav"
    ),
    "long_recording": (
        PROJECT_ROOT
        / "artifacts"
        / "concat_eval"
        / "CTS-298-431-1368-concat.wav"
    ),
}
REFERENCE_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "release_validation_corpus"
    / "overlap"
    / "manifest.json"
)


def _gpu0_snapshot() -> dict[str, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"GPU0 device query failed: {completed.stderr.strip()}"
        )
    fields = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 5:
        raise RuntimeError("GPU0 device query returned invalid fields")
    values = dict(
        zip(
            ("index", "total_mib", "used_mib", "free_mib", "utilization_percent"),
            (int(value) for value in fields),
        )
    )
    if values["index"] != 0:
        raise RuntimeError("GPU0 device query returned another device")
    return values


def _write_new(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _observed_reference_text() -> list[str]:
    rows = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("id") == "real-overlap-0001":
            return [
                str(turn["text"])
                for turn in row.get("reference_turns", ())
            ]
    raise RuntimeError("real-overlap-0001 reference is missing")


def run_benchmark(
    config_path: Path,
    output_root: Path,
    *,
    confirm_gpu0: bool,
) -> dict[str, Any]:
    if not confirm_gpu0:
        raise ValueError("confirm_gpu0 is required for GPU0 benchmark")
    config = load_config(Path(config_path))
    if config.diarization.backend != "nemotron3":
        raise ValueError("benchmark requires diarization.backend=nemotron3")
    if config.overlap.enabled or config.asr.aed_scheduler != "weighted_duration":
        raise ValueError(
            "Nemotron benchmark requires overlap disabled and weighted_duration"
        )
    require_physical_gpu_zero(config.general.device)
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"benchmark output already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    host = None
    report: dict[str, Any] = {
        "status": "running",
        "config_fingerprint": config_fingerprint(config),
        "cases": {},
        "inputs": {name: str(path) for name, path in INPUTS.items()},
        "reference_usage": "observation only, after inference; no training or calibration",
    }
    try:
        admission = _gpu0_snapshot()
        report["admission"] = admission
        _write_new(root / "admission.json", admission)
        if (
            admission["total_mib"] < config.service.minimum_total_mib
            or admission["free_mib"] < config.service.minimum_free_mib
        ):
            report["status"] = "pending"
            report["reason"] = "insufficient GPU0 free or total memory"
            _write_new(root / "result.json", report)
            return report

        verification = verify_models(
            config.paths.model_dir, speaker_backend="nemotron3"
        )
        if not verification.ok:
            raise RuntimeError(
                "selected model verification failed: "
                + ", ".join(verification.missing)
            )
        for name, path in INPUTS.items():
            if not path.is_file():
                raise FileNotFoundError(f"benchmark input missing: {name}: {path}")

        benchmark_config = replace(
            config,
            paths=replace(config.paths, work_dir=root / "work"),
        )
        paths = ServicePaths.from_work_dir(benchmark_config.paths.work_dir)
        host = build_hot_model_host(benchmark_config, paths)
        load_started = perf_counter()
        host.start()
        report["model_load_seconds"] = perf_counter() - load_started
        report["runtime_identity"] = dict(host.inventories)
        runner = ServiceJobRunner(benchmark_config, host)
        for name, audio in INPUTS.items():
            output = root / name
            started = perf_counter()
            runner.run(
                ServiceRequest(
                    identifier=f"benchmark-{name}-{uuid.uuid4().hex[:8]}",
                    action="submit",
                    input_audio=audio,
                    output_dir=output,
                    config_fingerprint=config_fingerprint(benchmark_config),
                    diagnostics_level="deep",
                )
            )
            diagnostics = json.loads(
                (output / "diagnostics.json").read_text(encoding="utf-8")
            )
            transcript = json.loads(
                (output / "transcript.json").read_text(encoding="utf-8")
            )
            profile = diagnostics.get("resource_profile") or {}
            report["cases"][name] = {
                "elapsed_seconds": perf_counter() - started,
                "transcript": transcript,
                "diagnostics": diagnostics,
                "stage_seconds": diagnostics.get("stage_seconds"),
                "project_owned_peak": profile.get("process_peak_mib"),
                "net_service_peak_mib": profile.get("net_service_peak_mib"),
            }
        report["cases"]["real_overlap"]["reference_texts"] = (
            _observed_reference_text()
        )
        closing = host
        host = None
        closing.close()
        report["ending_gpu"] = _gpu0_snapshot()
        report["status"] = "complete"
        _write_new(root / "result.json", report)
        return report
    except Exception as exc:
        if host is not None:
            closing = host
            host = None
            try:
                closing.close()
            except Exception as close_exc:
                report["close_error"] = str(close_exc)
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["oom"] = "out of memory" in str(exc).lower()
        report["traceback"] = traceback.format_exc()
        try:
            report["ending_gpu"] = _gpu0_snapshot()
        except Exception as snapshot_exc:
            report["ending_gpu_error"] = str(snapshot_exc)
        _write_new(root / "failure.json", report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--confirm-gpu0", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        args.config, args.output_root, confirm_gpu0=args.confirm_gpu0
    )
    print(json.dumps(
        {
            "status": result["status"],
            "output_root": str(args.output_root),
        },
        ensure_ascii=False,
        sort_keys=True,
    ))
    return 0 if result["status"] == "complete" else 4


if __name__ == "__main__":
    raise SystemExit(main())

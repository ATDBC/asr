#!/usr/bin/env python3
"""Calibrate the two weighted AED scheduler coefficients on physical GPU0."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Callable, Mapping, Sequence

from funasr_transcriber.firered_client import FireRedWorkerClient
from funasr_transcriber.gpu import GpuSample, query_gpu
from funasr_transcriber.weighted_aed_calibration import (
    build_calibration_cases,
    build_expected_runtime_identity,
    build_ready_artifact,
    canonical_sha256,
    evaluate_calibration,
    load_calibration_sources,
    sha256_file,
    validate_calibration_coverage,
)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit device-specific weighted AED scheduler coefficients.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument(
        "--use-half",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _write_create_only(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _gpu_dict(sample: GpuSample) -> dict[str, object]:
    return {
        "index": int(sample.index),
        "name": str(sample.name),
        "total_mib": int(sample.total_mib),
        "free_mib": int(sample.free_mib),
        "driver_version": str(sample.driver_version),
    }


def _plan_body(
    *,
    manifest_checksum: str,
    repeats: int,
    use_half: bool,
    runtime_identity: Mapping[str, object],
    coverage: Mapping[str, object],
    cases: Sequence[object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "weighted_aed_calibration_plan",
        "device": "cuda:0",
        "manifest_checksum": manifest_checksum,
        "repeats": int(repeats),
        "use_half": bool(use_half),
        "requested_runtime_identity": dict(runtime_identity),
        "coverage": dict(coverage),
        "cases": [case.as_dict() for case in cases],
    }


def _load_records(
    path: Path,
    *,
    case_ids: set[str],
    plan_checksum: str,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    runtime_checksum: str | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"record {line_number} must be an object")
        case_id = str(record.get("case_id") or "")
        if case_id not in case_ids:
            raise ValueError(f"record {line_number} has unknown case_id")
        if case_id in seen:
            raise ValueError(f"duplicate record for case_id {case_id}")
        seen.add(case_id)
        if record.get("plan_checksum") != plan_checksum:
            raise ValueError(f"record {line_number} plan checksum mismatch")
        identity = record.get("runtime_identity")
        identity_checksum = str(record.get("runtime_identity_checksum") or "")
        if not isinstance(identity, dict) or canonical_sha256(identity) != identity_checksum:
            raise ValueError(f"record {line_number} runtime identity mismatch")
        if runtime_checksum is None:
            runtime_checksum = identity_checksum
        elif runtime_checksum != identity_checksum:
            raise ValueError("calibration records use different runtime identities")
        if record.get("status") != "complete":
            raise ValueError(f"preserved failed calibration case: {case_id}")
        records.append(record)
    return records


def _validate_worker_runtime(
    expected: Mapping[str, object],
    status: Mapping[str, object],
) -> dict[str, object]:
    runtime = status.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("FireRed worker status has no runtime identity")
    for key in ("device", "dtype", "compute_capability"):
        if runtime.get(key) != expected.get(key):
            raise ValueError(
                f"FireRed worker {key} mismatch: "
                f"expected {expected.get(key)!r}, got {runtime.get(key)!r}"
            )
    return {**expected, **runtime}


def _aed_output_fingerprint(results: Sequence[object]) -> str:
    semantic = []
    for item in results:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("result"), Mapping
        ):
            raise ValueError("FireRed worker returned invalid AED results")
        result = item["result"]
        semantic.append({
            "index": int(item["index"]),
            "text": str(result.get("text") or ""),
            "confidence": result.get("confidence"),
            "timestamp": result.get("timestamp"),
        })
    return canonical_sha256(semantic)


def _measure_case(client: object, case: object) -> dict[str, object]:
    segments = [dict(segment) for segment in case.segments]
    response = client.run_aed_plan(
        case.audio_path,
        [segments],
        vad_segments_ms=[
            [int(segment["start_ms"]), int(segment["end_ms"])]
            for segment in segments
        ],
    )
    results = response.get("aed_results")
    metrics = response.get("batch_metrics")
    if not isinstance(results, list) or not isinstance(metrics, list) or len(metrics) != 1:
        raise ValueError("FireRed worker returned invalid AED calibration result")
    expected_indexes = sorted(int(segment["index"]) for segment in segments)
    try:
        returned_indexes = sorted(int(item["index"]) for item in results)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("FireRed worker returned invalid AED result indexes") from exc
    if returned_indexes != expected_indexes:
        raise ValueError("FireRed worker returned incomplete AED calibration results")
    metric = metrics[0]
    if not isinstance(metric, Mapping):
        raise ValueError("FireRed worker returned invalid AED batch metrics")
    actual = float(metric.get("actual_seconds") or 0.0)
    if not math.isfinite(actual) or actual <= 0.0:
        raise ValueError("FireRed worker returned invalid AED timing")
    return {
        "actual_seconds": actual,
        "output_fingerprint": _aed_output_fingerprint(results),
        "peak_allocated_mib": metric.get("peak_allocated_mib"),
        "peak_reserved_mib": metric.get("peak_reserved_mib"),
    }


def run(
    args: argparse.Namespace,
    *,
    client_factory: Callable[..., object] = FireRedWorkerClient,
    gpu_query: Callable[[str], GpuSample] = query_gpu,
) -> int:
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    worker_python = Path(args.worker_python).expanduser().absolute()
    plan_path = output / "plan.json"
    records_path = output / "records.jsonl"
    report_path = output / "calibration-report.json"
    artifact_path = output / "weighted-aed-calibration.json"
    manifest_checksum = sha256_file(manifest)

    existing_plan: dict[str, object] | None = None
    if plan_path.exists():
        if not args.resume:
            raise ValueError("calibration plan already exists; use --resume")
        existing_plan = _read_json(plan_path)
        if existing_plan.get("manifest_checksum") != manifest_checksum:
            raise ValueError("calibration manifest checksum mismatch")
    elif args.resume:
        raise ValueError("cannot resume without plan.json")

    sources = load_calibration_sources(manifest, args.audio_root)
    cases = build_calibration_cases(sources)
    coverage = validate_calibration_coverage(cases)
    gpu_start = gpu_query("cuda:0")
    expected_runtime = build_expected_runtime_identity(
        model_dir,
        gpu_name=gpu_start.name,
        driver_version=gpu_start.driver_version,
        use_half=bool(args.use_half),
    )
    body = _plan_body(
        manifest_checksum=manifest_checksum,
        repeats=int(args.repeats),
        use_half=bool(args.use_half),
        runtime_identity=expected_runtime,
        coverage=coverage,
        cases=cases,
    )
    plan_checksum = canonical_sha256(body)
    plan = {**body, "plan_checksum": plan_checksum}
    output.mkdir(parents=True, exist_ok=True)
    if existing_plan is None:
        _write_create_only(plan_path, plan)
    elif existing_plan != plan:
        raise ValueError("calibration plan does not match resume request")

    case_by_id = {case.case_id: case for case in cases}
    records = _load_records(
        records_path,
        case_ids=set(case_by_id),
        plan_checksum=plan_checksum,
    )
    completed = {str(record["case_id"]) for record in records}
    if len(completed) == len(cases) and report_path.is_file() and artifact_path.is_file():
        if _read_json(report_path).get("status") != "ready":
            raise ValueError("completed calibration report is not ready")
        if _read_json(artifact_path).get("status") != "ready":
            raise ValueError("completed calibration artifact is not ready")
        return 0
    pending = [case for case in cases if case.case_id not in completed]
    actual_runtime = (
        dict(records[0]["runtime_identity"]) if records else None
    )

    client = None
    current_case = pending[0] if pending else None
    try:
        if pending:
            client = client_factory(
                worker_python=worker_python,
                model_root=model_dir,
                device="cuda:0",
                timeout_seconds=1800.0,
                stderr_log_path=output / "firered-calibration.log",
                use_half=bool(args.use_half),
                enable_vad=False,
                enable_punc=False,
            )
            client.start()
            worker_runtime = _validate_worker_runtime(expected_runtime, client.status())
            if actual_runtime is not None and worker_runtime != actual_runtime:
                raise ValueError("FireRed worker runtime changed during resume")
            actual_runtime = worker_runtime
            _measure_case(client, pending[0])
            runtime_checksum = canonical_sha256(actual_runtime)
            for current_case in pending:
                measurements = [
                    _measure_case(client, current_case)
                    for _ in range(int(args.repeats))
                ]
                fingerprints = [
                    str(item["output_fingerprint"]) for item in measurements
                ]
                if len(set(fingerprints)) != 1:
                    raise ValueError("AED output fingerprints do not match")
                repeat_seconds = [
                    float(item["actual_seconds"]) for item in measurements
                ]
                record = {
                    "status": "complete",
                    "case_id": current_case.case_id,
                    "source_recording_group": current_case.source_recording_group,
                    "segment_indexes": [
                        int(segment["index"]) for segment in current_case.segments
                    ],
                    "item_count": current_case.item_count,
                    "max_duration_seconds": current_case.max_duration_seconds,
                    "total_duration_seconds": current_case.total_duration_seconds,
                    "repeat_seconds": repeat_seconds,
                    "median_actual_seconds": statistics.median(repeat_seconds),
                    "output_fingerprints": fingerprints,
                    "peak_allocated_mib": [
                        item["peak_allocated_mib"] for item in measurements
                    ],
                    "peak_reserved_mib": [
                        item["peak_reserved_mib"] for item in measurements
                    ],
                    "plan_checksum": plan_checksum,
                    "runtime_identity": actual_runtime,
                    "runtime_identity_checksum": runtime_checksum,
                }
                _append_jsonl(records_path, record)
                records.append(record)
    except Exception as exc:
        if current_case is not None:
            failure_runtime = actual_runtime or expected_runtime
            _append_jsonl(
                records_path,
                {
                    "status": "failed",
                    "case_id": current_case.case_id,
                    "source_recording_group": current_case.source_recording_group,
                    "plan_checksum": plan_checksum,
                    "runtime_identity": failure_runtime,
                    "runtime_identity_checksum": canonical_sha256(failure_runtime),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        raise
    finally:
        if client is not None:
            client.close()

    gpu_end = gpu_query("cuda:0")
    if len(records) != len(cases) or actual_runtime is None:
        raise ValueError("calibration collection is incomplete")
    weights, fit_report = evaluate_calibration(records)
    report = {
        **fit_report,
        "manifest_checksum": manifest_checksum,
        "plan_checksum": plan_checksum,
        "records_checksum": sha256_file(records_path),
        "runtime_identity": actual_runtime,
        "gpu0": {"start": _gpu_dict(gpu_start), "end": _gpu_dict(gpu_end)},
        "outcomes": {
            "batches": len(cases) * int(args.repeats),
            "failure_records": 0,
            "oom_records": 0,
            "safety_violations": 0,
            "invalid_responses": 0,
        },
    }
    _atomic_write_json(report_path, report)
    if report["status"] != "ready":
        return 2
    artifact = build_ready_artifact(
        weights=weights,
        report=report,
        runtime_identity=actual_runtime,
        manifest_checksum=manifest_checksum,
        plan_checksum=plan_checksum,
        records_checksum=str(report["records_checksum"]),
        gpu_start=_gpu_dict(gpu_start),
        gpu_end=_gpu_dict(gpu_end),
    )
    _atomic_write_json(artifact_path, artifact)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(_parser().parse_args(argv))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

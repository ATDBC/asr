#!/usr/bin/env python
"""Calibrate official FireRed AED/Punc batching on physical GPU 0."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "models" / "fireredasr2s" / "source"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from funasr_transcriber.firered_benchmark import (  # noqa: E402
    choose_batch_candidate,
    stable_response_fingerprint,
    summarize_stage_calls,
)
from funasr_transcriber.firered_contract import normalize_official_result  # noqa: E402
from funasr_transcriber.firered_worker import (  # noqa: E402
    _scheduled_process,
    build_firered_system,
)


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _query_gpu0() -> dict[str, int]:
    import subprocess

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
    if len(fields) != 3 or fields[0] != "0":
        raise RuntimeError("physical GPU 0 is unavailable")
    return {
        "index": int(fields[0]),
        "total_mib": int(fields[1]),
        "free_mib": int(fields[2]),
    }


def _own_gpu_memory_mib() -> int:
    """Read this process's CUDA allocation from nvidia-smi only."""

    import subprocess

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        return 0
    pid = os.getpid()
    peak = 0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
            try:
                peak = max(peak, int(fields[1]))
            except ValueError:
                pass
    return peak


def _synchronize(torch: Any) -> None:
    torch.cuda.synchronize(0)


def _instrument(
    system: Any,
    torch: Any,
    calls: dict[str, list[float]],
) -> list[tuple[Any, str, Callable[..., Any]]]:
    restores: list[tuple[Any, str, Callable[..., Any]]] = []

    def wrap(owner: Any, attribute: str, stage: str) -> None:
        original = getattr(owner, attribute)

        def measured(*args: Any, **kwargs: Any) -> Any:
            _synchronize(torch)
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                _synchronize(torch)
                calls[stage].append(time.perf_counter() - started)

        setattr(owner, attribute, measured)
        restores.append((owner, attribute, original))

    wrap(system.vad, "detect", "vad")
    wrap(system.asr, "transcribe", "aed")
    wrap(system.punc, "process_with_timestamp", "punc")
    return restores


def _run_once(
    system: Any,
    audio: Path,
    request_id: str,
    torch: Any,
    scheduled_segments: list[tuple[float, float]] | None = None,
    batch_size: int = 1,
) -> tuple[dict[str, object], dict[str, list[float]], float, int]:
    calls = {"vad": [], "aed": [], "punc": []}
    if scheduled_segments is None:
        restores = _instrument(system, torch, calls)
    else:
        restores = []
        for owner, attribute, stage in (
            (system.asr, "transcribe", "aed"),
            (system.punc, "process_with_timestamp", "punc"),
        ):
            original = getattr(owner, attribute)

            def measured(*args: Any, _original=original, _stage=stage, **kwargs: Any) -> Any:
                _synchronize(torch)
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    _synchronize(torch)
                    calls[_stage].append(time.perf_counter() - started)

            setattr(owner, attribute, measured)
            restores.append((owner, attribute, original))
        calls["vad"].append(0.0)
    try:
        _synchronize(torch)
        started = time.perf_counter()
        if scheduled_segments is None:
            raw = system.process(str(audio), uttid=request_id)
        else:
            raw = _scheduled_process(
                system,
                str(audio),
                request_id,
                scheduled_segments,
                batch_size,
            )
        _synchronize(torch)
        elapsed = time.perf_counter() - started
    finally:
        for owner, attribute, original in reversed(restores):
            setattr(owner, attribute, original)
    response = normalize_official_result(request_id, raw)
    return response, calls, elapsed, _own_gpu_memory_mib()


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--warmup-audio", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=_positive_int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--maximum-process-mib", type=_positive_int, default=12_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--use-half",
        action="store_true",
        help="enable FireRed AED half-precision path (BF16 when supported)",
    )
    parser.add_argument(
        "--baseline-fingerprint",
        help="known stable fingerprint; allows benchmarking scheduled batches without batch 1",
    )
    parser.add_argument(
        "--vad-segments-json",
        type=Path,
        help="optional JSON list of [start_ms, end_ms] intervals to schedule",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.device != "cuda:0":
        raise ValueError("only physical GPU device cuda:0 is supported")
    if len(set(args.batches)) != len(args.batches):
        raise ValueError("--batches must not contain duplicates")
    if args.baseline_fingerprint is None and 1 not in args.batches:
        raise ValueError("--batches must include baseline batch 1")
    for path, label in ((args.audio, "audio"), (args.warmup_audio, "warmup-audio")):
        if path is not None and not path.expanduser().is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if not args.model_dir.expanduser().is_dir():
        raise ValueError(f"model-dir does not exist: {args.model_dir}")
    if args.vad_segments_json is not None and not args.vad_segments_json.expanduser().is_file():
        raise ValueError(f"vad-segments-json does not exist: {args.vad_segments_json}")


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    gpu = _query_gpu0()
    warmup_audio = (args.warmup_audio or args.audio).expanduser().resolve()
    audio = args.audio.expanduser().resolve()
    scheduled_segments = None
    if args.vad_segments_json is not None:
        raw_segments = json.loads(
            args.vad_segments_json.expanduser().read_text(encoding="utf-8")
        )
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("vad-segments-json must contain a non-empty list")
        scheduled_segments = [
            (float(item[0]) / 1000.0, float(item[1]) / 1000.0)
            for item in raw_segments
        ]
    candidates: list[dict[str, object]] = []
    baseline_fingerprint: str | None = args.baseline_fingerprint
    batches = (
        list(args.batches)
        if baseline_fingerprint is not None
        else [1, *(batch for batch in args.batches if batch != 1)]
    )
    baseline_valid = baseline_fingerprint is not None
    for batch_size in batches:
        before = _query_gpu0()
        candidate: dict[str, object] = {
            "batch_size": batch_size,
            "repeats": args.repeats,
            "maximum_process_mib": args.maximum_process_mib,
            "gpu_free_before_mib": before["free_mib"],
            "fingerprints": [],
            "end_to_end_seconds": [],
            "peak_process_mib": 0,
            "stage_calls": {"vad": [], "aed": [], "punc": []},
            "successful_repetitions": 0,
        }
        system = None
        try:
            load_started = time.perf_counter()
            system = build_firered_system(
                args.model_dir,
                args.device,
                batch_size,
                use_half=args.use_half,
            )
            candidate["load_seconds"] = time.perf_counter() - load_started
            # Warm-up is deliberately excluded from all reported timing and peak stats.
            system.process(str(warmup_audio), uttid=f"warmup-b{batch_size}")
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            for repeat in range(args.repeats):
                response, calls, elapsed, process_mib = _run_once(
                    system,
                    audio,
                    f"benchmark-b{batch_size}-r{repeat}",
                    torch,
                    scheduled_segments,
                    batch_size,
                )
                fingerprint = stable_response_fingerprint(response)
                if baseline_fingerprint is None:
                    baseline_fingerprint = fingerprint
                candidate["fingerprints"].append(fingerprint)
                candidate["end_to_end_seconds"].append(elapsed)
                candidate["peak_process_mib"] = max(
                    int(candidate["peak_process_mib"]), process_mib
                )
                stage_calls = candidate["stage_calls"]
                for stage, values in calls.items():
                    stage_calls[stage].extend(values)
                candidate["successful_repetitions"] = repeat + 1
            candidate["stage_summary"] = summarize_stage_calls(candidate["stage_calls"])
            candidate["torch_peak_allocated_mib"] = int(
                torch.cuda.max_memory_allocated(0) / (1024 * 1024)
            )
        except torch.cuda.OutOfMemoryError as exc:
            candidate["oom"] = True
            candidate["error"] = str(exc)
        except Exception as exc:
            candidate["failed"] = True
            candidate["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if system is not None:
                del system
            gc.collect()
            torch.cuda.empty_cache()
            candidate["gpu_free_after_mib"] = _query_gpu0()["free_mib"]
        if batch_size == 1 and args.baseline_fingerprint is None:
            fingerprints = candidate["fingerprints"]
            if (
                len(fingerprints) == args.repeats
                and len(set(fingerprints)) == 1
                and candidate.get("failed") is not True
                and candidate.get("oom") is not True
            ):
                baseline_fingerprint = fingerprints[0]
                baseline_valid = True
        if baseline_fingerprint is not None:
            candidate["baseline_fingerprint"] = baseline_fingerprint
        candidate["baseline_valid"] = baseline_valid
        candidates.append(candidate)
    recommendation = choose_batch_candidate(candidates)
    return {
        "device": args.device,
        "gpu": gpu,
        "audio": str(audio),
        "warmup_audio": str(warmup_audio),
        "repeats": args.repeats,
        "maximum_process_mib": args.maximum_process_mib,
        "use_half": args.use_half,
        "candidates": candidates,
        "recommended_segment_batch_size": recommendation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        payload = run(args)
        _atomic_write(args.output, payload)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.exit(1, f"benchmark failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Create and run 30/60/120-minute deterministic CLI stress cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from funasr_transcriber.benchmarking import stress_durations_minutes


def build_audio(source: Path, output: Path, minutes: int) -> None:
    completed = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-stream_loop", "-1", "-i", str(source),
            "-t", str(minutes * 60),
            "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le", str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())


def run_monitored(command: list[str]) -> dict:
    started = time.perf_counter()
    process = subprocess.Popen(command)
    monitored = psutil.Process(process.pid)
    peak_rss = 0
    peak_gpu_memory_mib = 0
    gpu_samples = 0
    gpu_monitor_available = False
    while process.poll() is None:
        try:
            peak_rss = max(peak_rss, monitored.memory_info().rss)
        except psutil.Error:
            pass
        try:
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            gpu_monitor_available = gpu_monitor_available or gpu.returncode == 0
            process_ids = {process.pid}
            try:
                process_ids.update(child.pid for child in monitored.children(recursive=True))
            except psutil.Error:
                pass
            for line in gpu.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 2 and int(fields[0]) in process_ids:
                    peak_gpu_memory_mib = max(peak_gpu_memory_mib, int(fields[1]))
                    gpu_samples += 1
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        time.sleep(0.2)
    return {
        "returncode": process.returncode,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "gpu_samples": gpu_samples,
        "gpu_monitor_available": gpu_monitor_available,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    executable = os.getenv("FUNASR_TRANSCRIBE_BIN", "funasr-transcribe")
    for minutes in stress_durations_minutes():
        audio = args.output_dir / f"stress-{minutes}m.wav"
        build_audio(args.source, audio, minutes)
        hashes = []
        runs = []
        for run_index in (1, 2):
            output = args.output_dir / f"stress-{minutes}m-run{run_index}"
            command = [
                executable, str(audio),
                "--config", str(args.config),
                "--output-dir", str(output),
                "--disable-overlap",
            ]
            run = run_monitored(command)
            if run["returncode"] != 0:
                raise RuntimeError(
                    f"{minutes} minute run failed: {run['returncode']}"
                )
            hashes.append(sha256(output / "transcript.json"))
            transcript = json.loads(
                (output / "transcript.json").read_text(encoding="utf-8")
            )
            diagnostics = json.loads(
                (output / "diagnostics.json").read_text(encoding="utf-8")
            )
            sentences = (diagnostics.get("asr") or {}).get("sentences") or []
            turns = transcript.get("turns") or []
            run["lid_sentences"] = len(sentences)
            run["lid_covered"] = sum(
                bool(item.get("language"))
                and item.get("language_confidence") is not None
                for item in sentences
            )
            run["timestamp_checked"] = len(turns)
            run["timestamp_valid"] = sum(
                0 <= int(item.get("start_ms", -1))
                < int(item.get("end_ms", -1))
                for item in turns
            )
            run["gpu_stages"] = diagnostics.get("gpu_stages") or {}
            leftovers = list(output.parent.glob(f".{output.name}.staging-*"))
            run["temporary_cleanup_ok"] = not leftovers
            runs.append(run)
        results.append(
            {
                "minutes": minutes,
                "runs": runs,
                "result_hashes": hashes,
                "deterministic": hashes[0] == hashes[1],
            }
        )
    all_runs = [
        (item["minutes"], index, run)
        for item in results
        for index, run in enumerate(item["runs"], start=1)
    ]
    lid_sentences = sum(run["lid_sentences"] for _, _, run in all_runs)
    lid_covered = sum(run["lid_covered"] for _, _, run in all_runs)
    timestamp_checked = sum(
        run["timestamp_checked"] for _, _, run in all_runs
    )
    timestamp_valid = sum(run["timestamp_valid"] for _, _, run in all_runs)
    gpu_stages = {
        f"{minutes}m-run{index}:{stage}": record
        for minutes, index, run in all_runs
        for stage, record in run["gpu_stages"].items()
    }
    release_evidence = {
        "lid_coverage": {
            "sentences": lid_sentences,
            "covered": lid_covered,
            "ratio": lid_covered / lid_sentences if lid_sentences else 0.0,
        },
        "timestamp_integrity": {
            "checked": timestamp_checked,
            "valid": timestamp_valid,
            "passed": (
                timestamp_checked > 0 and timestamp_valid == timestamp_checked
            ),
        },
        "gpu_stages": gpu_stages,
        "deterministic_hashes": {
            str(item["minutes"]): item["result_hashes"] for item in results
        },
    }
    evidence_passed = (
        release_evidence["lid_coverage"]["ratio"] == 1.0
        and release_evidence["timestamp_integrity"]["passed"]
        and bool(gpu_stages)
        and all(
            record.get("released") is True
            for record in gpu_stages.values()
        )
    )
    payload = {
        "results": results,
        "release_evidence": release_evidence,
        "passed": all(
            item["deterministic"]
            and all(
                run["temporary_cleanup_ok"]
                and run["gpu_monitor_available"]
                and run["gpu_samples"] > 0
                for run in item["runs"]
            )
            for item in results
        ) and evidence_passed,
    }
    (args.output_dir / "stress-summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

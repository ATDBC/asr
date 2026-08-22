#!/usr/bin/env python3
"""Run a same-service AED transient-budget matrix on physical GPU0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import uuid

from funasr_transcriber.cli import _connected_service
from funasr_transcriber.config import load_config
from funasr_transcriber.service_protocol import ServiceRequest, config_fingerprint


def build_matrix_manifest(
    fractions: list[float],
    repeats: int,
) -> list[dict[str, object]]:
    if not fractions:
        raise ValueError("at least one budget fraction is required")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    normalized = []
    for fraction in fractions:
        value = float(fraction)
        if not 0.1 <= value <= 1.0:
            raise ValueError("budget fractions must stay within 0.1..1.0")
        normalized.append(value)
    return [
        {
            "fraction": fraction,
            "repeat": repeat,
            "id": uuid.uuid4().hex,
        }
        for fraction in normalized
        for repeat in range(1, repeats + 1)
    ]


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update((path / "transcript.json").read_bytes())
    return digest.hexdigest()


def run_matrix(
    audio: Path,
    config_path: Path,
    output_root: Path,
    fractions: list[float],
    repeats: int,
    diagnostics_level: str = "deep",
) -> dict[str, object]:
    config = load_config(config_path)
    if config.general.device != "cuda:0":
        raise ValueError("budget matrix is restricted to physical cuda:0")
    args = argparse.Namespace(config=config_path)
    client = _connected_service(args, config)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_matrix_manifest(fractions, repeats)
    records: list[dict[str, object]] = []
    fingerprints: dict[float, str] = {}
    for item in manifest:
        fraction = float(item["fraction"])
        repeat = int(item["repeat"])
        output_dir = output_root / f"fraction-{fraction:.2f}" / f"run-{repeat}"
        request = ServiceRequest(
            identifier=str(item["id"]),
            action="submit",
            input_audio=Path(audio).expanduser().resolve(),
            output_dir=output_dir.resolve(),
            config_fingerprint=config_fingerprint(config),
            overwrite=True,
            profile_resources=diagnostics_level in {"resources", "deep"},
            diagnostics_level=diagnostics_level,
            aed_budget_fraction=fraction,
        )
        started = perf_counter()
        record: dict[str, object] = {
            "fraction": fraction,
            "repeat": repeat,
            "output_dir": str(output_dir.resolve()),
        }
        try:
            accepted = client.submit(request)
            terminal = client.wait(
                str(accepted["job_id"]),
                uuid.uuid4().hex,
            )
            diagnostics = json.loads(
                (output_dir / "diagnostics.json").read_text(encoding="utf-8")
            )
            fingerprint = _fingerprint(output_dir)
            previous = fingerprints.setdefault(fraction, fingerprint)
            record.update(
                {
                    "status": "completed",
                    "wall_seconds": perf_counter() - started,
                    "fingerprint": fingerprint,
                    "fingerprint_stable": previous == fingerprint,
                    "result": terminal.get("result"),
                    "stage_seconds": diagnostics.get("stage_seconds", {}),
                    "resource_profile": diagnostics.get(
                        "resource_profile", {}
                    ),
                    "precision": diagnostics.get("config", {})
                    .get("asr", {})
                    .get("use_half"),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "wall_seconds": perf_counter() - started,
                    "error": str(exc),
                }
            )
        records.append(record)
        (output_root / "manifest.json").write_text(
            json.dumps(
                {
                    "audio": str(Path(audio).expanduser().resolve()),
                    "config": str(Path(config_path).expanduser().resolve()),
                    "diagnostics": diagnostics_level,
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {"records": records}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fractions", nargs="+", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--diagnostics",
        choices=("resources", "deep"),
        default="deep",
    )
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    result = run_matrix(
        args.audio,
        args.config,
        args.output_root,
        args.fractions,
        args.repeats,
        args.diagnostics,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

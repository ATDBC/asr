"""Runtime diagnostics that are separate from transcript business data."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping
from time import perf_counter
from typing import Iterator


@dataclass(slots=True)
class Diagnostics:
    stage_seconds: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    speaker_count: int | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    asr: dict[str, Any] = field(default_factory=dict)
    gpu_stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    resource_profile: dict[str, Any] = field(
        default_factory=lambda: {"enabled": False}
    )

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[name] = perf_counter() - started

    def warn(self, message: str) -> None:
        self.warnings.append(message)


SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY")


def sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        upper = str(key).upper()
        if any(marker in upper for marker in SECRET_MARKERS):
            sanitized[str(key)] = "<redacted>"
        elif isinstance(item, Mapping):
            sanitized[str(key)] = sanitize_mapping(item)
        elif isinstance(item, Path):
            sanitized[str(key)] = str(item)
        elif isinstance(item, (list, tuple)):
            sanitized[str(key)] = [
                str(element) if isinstance(element, Path) else element
                for element in item
            ]
        else:
            sanitized[str(key)] = item
    return sanitized


def collect_runtime_info() -> dict[str, Any]:
    packages = {}
    for name in ("funasr", "modelscope", "torch", "torchaudio", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    runtime: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        runtime["cuda_available"] = torch.cuda.is_available()
        runtime["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            runtime["gpu"] = torch.cuda.get_device_name(0)
            runtime["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    except ImportError:
        runtime["cuda_available"] = False
    return runtime


def export_diagnostics(
    diagnostics: Diagnostics,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    directory = Path(output_dir)
    target = directory / "diagnostics.json"
    if target.exists() and not overwrite:
        raise FileExistsError(f"diagnostics output already exists: {target}")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".diagnostics.json.tmp"
    payload = {
        "schema_version": "1.0",
        "speaker_count": diagnostics.speaker_count,
        "stage_seconds": {
            key: round(value, 6)
            for key, value in sorted(diagnostics.stage_seconds.items())
        },
        "warnings": list(diagnostics.warnings),
        "runtime": diagnostics.runtime,
        "config": sanitize_mapping(diagnostics.config),
        "asr": diagnostics.asr,
        "gpu_stages": diagnostics.gpu_stages,
        "resource_profile": sanitize_mapping(diagnostics.resource_profile),
    }
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target

"""Read-only NVIDIA GPU admission and release measurements."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Any, Callable

from .errors import EnvironmentError as TranscriberEnvironmentError


GPU_RELEASE_TOLERANCE_MIB = 512


@dataclass(frozen=True, slots=True)
class GpuSample:
    index: int
    name: str
    total_mib: int
    free_mib: int
    driver_version: str


def _device_index(device: str) -> int:
    parts = str(device).split(":", 1)
    if len(parts) != 2 or parts[0] != "cuda":
        raise TranscriberEnvironmentError(
            f"an explicit CUDA device is required: {device}"
        )
    try:
        index = int(parts[1])
    except ValueError as exc:
        raise TranscriberEnvironmentError(
            f"invalid CUDA device: {device}"
        ) from exc
    if index < 0:
        raise TranscriberEnvironmentError(f"invalid CUDA device: {device}")
    return index


def query_gpu(
    device: str,
    runner: Callable[..., Any] = subprocess.run,
) -> GpuSample:
    """Query one physical GPU without changing or signaling any process."""

    index = _device_index(device)
    completed = runner(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise TranscriberEnvironmentError(
            "nvidia-smi GPU monitoring failed: "
            f"{str(completed.stderr).strip()}"
        )
    for line in str(completed.stdout).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            row_index = int(fields[0])
            total_mib = int(fields[2])
            free_mib = int(fields[3])
        except ValueError:
            continue
        if row_index == index:
            return GpuSample(
                index=row_index,
                name=fields[1],
                total_mib=total_mib,
                free_mib=free_mib,
                driver_version=fields[4],
            )
    raise TranscriberEnvironmentError(
        f"nvidia-smi did not report CUDA device {index}"
    )


def require_gpu_capacity(
    device: str,
    runner: Callable[..., Any] = subprocess.run,
    minimum_total_mib: int = 23 * 1024,
    minimum_free_mib: int = 12 * 1024,
) -> GpuSample:
    sample = query_gpu(device, runner=runner)
    if sample.total_mib < minimum_total_mib:
        raise TranscriberEnvironmentError(
            f"GPU {sample.index} total memory is {sample.total_mib} MiB; "
            f"at least {minimum_total_mib} MiB is required"
        )
    if sample.free_mib < minimum_free_mib:
        raise TranscriberEnvironmentError(
            f"GPU {sample.index} free memory is {sample.free_mib} MiB; "
            f"at least {minimum_free_mib} MiB is required"
        )
    return sample


def require_concurrent_gpu_capacity(
    device: str,
    minimum_total_mib: int,
    required_free_mib: int,
    runner: Callable[..., Any] = subprocess.run,
) -> GpuSample:
    """Require the pre-calibrated free-memory budget for concurrent owners."""

    return require_gpu_capacity(
        device,
        runner=runner,
        minimum_total_mib=minimum_total_mib,
        minimum_free_mib=required_free_mib,
    )


def query_gpu_process_memory(
    device: str,
    allowed_pids: set[int],
    runner: Callable[..., Any] = subprocess.run,
) -> dict[int, int]:
    """Read memory only for the explicitly registered project PIDs."""

    _device_index(device)
    completed = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise TranscriberEnvironmentError(
            "nvidia-smi process monitoring failed: "
            f"{str(completed.stderr).strip()}"
        )
    from .resource_profile import parse_process_memory_mib

    return parse_process_memory_mib(str(completed.stdout), allowed_pids)


def build_gpu_stage_record(
    before: GpuSample,
    after: GpuSample,
    peak_gpu_bytes: int | None,
    load_seconds: float | None,
    inference_seconds: float | None,
) -> dict[str, object]:
    if before.index != after.index:
        raise ValueError("GPU samples must refer to the same device")
    return {
        "device_index": before.index,
        "before_free_mib": before.free_mib,
        "after_free_mib": after.free_mib,
        "peak_gpu_bytes": peak_gpu_bytes,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "released": (
            after.free_mib
            >= before.free_mib - GPU_RELEASE_TOLERANCE_MIB
        ),
    }

"""Opt-in resource inventory and read-only sampling for owned processes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
import subprocess
import threading
from typing import Any


def parse_process_memory_mib(
    output: str,
    allowed_pids: set[int],
) -> dict[int, int]:
    """Return only known project PID memory rows from nvidia-smi CSV output."""

    result: dict[int, int] = {}
    for line in str(output).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
            memory_mib = int(fields[1])
        except ValueError:
            continue
        if pid in allowed_pids:
            result[pid] = memory_mib
    return result


def _tensor_groups(tensors: list[Any]) -> dict[str, dict[str, int]]:
    groups: dict[str, dict[str, int]] = {}
    for tensor in tensors:
        key = f"{tensor.device.type}:{str(tensor.dtype).removeprefix('torch.')}"
        group = groups.setdefault(
            key,
            {"elements": 0, "bytes": 0, "tensors": 0},
        )
        group["elements"] += int(tensor.numel())
        group["bytes"] += int(tensor.numel() * tensor.element_size())
        group["tensors"] += 1
    return groups


def module_inventory(name: str, module: Any) -> dict[str, object]:
    """Serialize parameter and persistent-buffer facts for one torch module."""

    parameters = list(module.parameters())
    buffers = list(module.buffers())
    parameter_bytes = sum(
        int(tensor.numel() * tensor.element_size())
        for tensor in parameters
    )
    buffer_bytes = sum(
        int(tensor.numel() * tensor.element_size())
        for tensor in buffers
    )
    return {
        "name": str(name),
        "parameter_count": sum(int(tensor.numel()) for tensor in parameters),
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "parameters": _tensor_groups(parameters),
        "buffers": _tensor_groups(buffers),
    }


def _device_index(device: str) -> int:
    prefix, separator, value = str(device).partition(":")
    if prefix != "cuda" or not separator:
        raise ValueError(f"an explicit CUDA device is required: {device}")
    index = int(value)
    if index < 0:
        raise ValueError(f"an explicit CUDA device is required: {device}")
    return index


def _process_rss_kib(pid: int) -> int | None:
    try:
        with open(f"/proc/{int(pid)}/status", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


class ResourceSampler:
    """Sample only service-owned PIDs without signalling any process."""

    def __init__(
        self,
        device: str,
        interval_ms: int,
        pid_getters: Mapping[str, Callable[[], int | None]],
        query: Callable[[], tuple[dict[str, int], dict[int, int]]] | None = None,
        rss_query: Callable[[int], int | None] = _process_rss_kib,
    ):
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.device = str(device)
        self.interval_ms = int(interval_ms)
        self._pid_getters = dict(pid_getters)
        self._query = query or self._default_query
        self._rss_query = rss_query
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample_count = 0
        self._minimum_free_mib: int | None = None
        self._total_mib: int | None = None
        self._process_peak_mib: dict[str, int] = {}
        self._cpu_rss_peak_kib: dict[str, int] = {}
        self._pid_observed = {name: False for name in self._pid_getters}

    def _registered_pids(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for name, getter in self._pid_getters.items():
            try:
                pid = getter()
            except Exception:
                continue
            if isinstance(pid, int) and pid > 0:
                result[name] = pid
        return result

    def _default_query(self) -> tuple[dict[str, int], dict[int, int]]:
        index = _device_index(self.device)
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if gpu.returncode != 0:
            raise RuntimeError("nvidia-smi GPU monitoring failed")
        sample: dict[str, int] | None = None
        for line in gpu.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                if int(fields[0]) == index:
                    sample = {
                        "total_mib": int(fields[1]),
                        "free_mib": int(fields[2]),
                    }
                    break
            except ValueError:
                continue
        if sample is None:
            raise RuntimeError(f"nvidia-smi did not report CUDA device {index}")
        processes = subprocess.run(
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
        if processes.returncode != 0:
            raise RuntimeError("nvidia-smi process monitoring failed")
        allowed = set(self._registered_pids().values())
        return sample, parse_process_memory_mib(processes.stdout, allowed)

    def sample_once(self) -> None:
        registered = self._registered_pids()
        gpu, process_memory = self._query()
        with self._lock:
            self._sample_count += 1
            total = gpu.get("total_mib")
            free = gpu.get("free_mib")
            if isinstance(total, int):
                self._total_mib = total
            if isinstance(free, int):
                self._minimum_free_mib = (
                    free
                    if self._minimum_free_mib is None
                    else min(self._minimum_free_mib, free)
                )
            for name, pid in registered.items():
                memory_mib = process_memory.get(pid)
                if memory_mib is not None:
                    self._pid_observed[name] = True
                    self._process_peak_mib[name] = max(
                        self._process_peak_mib.get(name, 0),
                        int(memory_mib),
                    )
                rss_kib = self._rss_query(pid)
                if rss_kib is not None:
                    self._cpu_rss_peak_kib[name] = max(
                        self._cpu_rss_peak_kib.get(name, 0),
                        int(rss_kib),
                    )

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_ms / 1000):
            try:
                self.sample_once()
            except Exception:
                # Profiling is observational; keep the job running and report
                # the samples that were collected successfully.
                continue

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="resource-profile-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, object]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=(self.interval_ms / 1000) + 1)
            self._thread = None
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "sampling_interval_ms": self.interval_ms,
                "sample_count": self._sample_count,
                "gpu": {
                    "total_mib": self._total_mib,
                    "minimum_free_mib": self._minimum_free_mib,
                },
                "process_peak_mib": dict(self._process_peak_mib),
                "cpu_rss_peak_kib": dict(self._cpu_rss_peak_kib),
                "pid_observed": dict(self._pid_observed),
            }

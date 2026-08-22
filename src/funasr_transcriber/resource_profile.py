"""Opt-in resource inventory and read-only sampling for owned processes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
import subprocess
import threading
from time import perf_counter
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
        trace_enabled: bool = False,
    ):
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.device = str(device)
        self.interval_ms = int(interval_ms)
        self._pid_getters = dict(pid_getters)
        self._query = query or self._default_query
        self._rss_query = rss_query
        self._trace_enabled = bool(trace_enabled)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample_count = 0
        self._minimum_free_mib: int | None = None
        self._total_mib: int | None = None
        self._other_project_baseline_mib: int | None = None
        self._peak_total_used_mib: int | None = None
        self._process_peak_mib: dict[str, int] = {}
        self._cpu_rss_peak_kib: dict[str, int] = {}
        self._pid_observed = {name: False for name in self._pid_getters}
        self._samples: list[dict[str, object]] = []

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
        sampled_at = perf_counter()
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
                if isinstance(total, int):
                    used = max(0, total - free)
                    self._peak_total_used_mib = max(
                        self._peak_total_used_mib or 0,
                        used,
                    )
                    known_project_mib = sum(
                        int(value)
                        for pid, value in process_memory.items()
                        if pid in set(registered.values())
                    )
                    if self._other_project_baseline_mib is None:
                        self._other_project_baseline_mib = max(
                            0,
                            used - known_project_mib,
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
            if self._trace_enabled:
                total = gpu.get("total_mib")
                free = gpu.get("free_mib")
                if isinstance(total, int) and isinstance(free, int):
                    used = max(0, total - free)
                    named_process_memory = {
                        name: int(process_memory[pid])
                        for name, pid in registered.items()
                        if pid in process_memory
                    }
                    process_pid_memory = {
                        str(pid): int(process_memory[pid])
                        for pid in set(registered.values())
                        if pid in process_memory
                    }
                    known_project_mib = sum(process_pid_memory.values())
                    self._samples.append(
                        {
                            "timestamp_monotonic": sampled_at,
                            "total_mib": total,
                            "free_mib": free,
                            "used_mib": used,
                            "net_service_mib": known_project_mib,
                            "process_mib": named_process_memory,
                            "process_pid_mib": process_pid_memory,
                        }
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
        try:
            self.sample_once()
        except Exception:
            # Profiling remains observational even when nvidia-smi is
            # temporarily unavailable at service startup.
            pass
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
                "other_project_baseline_mib": self._other_project_baseline_mib,
                "peak_total_used_mib": self._peak_total_used_mib,
                "net_service_peak_mib": (
                    None
                    if self._peak_total_used_mib is None
                    or self._other_project_baseline_mib is None
                    else max(
                        0,
                        self._peak_total_used_mib
                        - self._other_project_baseline_mib,
                    )
                ),
                "process_peak_mib": dict(self._process_peak_mib),
                "cpu_rss_peak_kib": dict(self._cpu_rss_peak_kib),
                "pid_observed": dict(self._pid_observed),
            }

    def trace_stage(
        self,
        started_at_monotonic: float,
        ended_at_monotonic: float,
        *,
        effective_budget_mib: int | None = None,
        safety_margin_mib: int | None = None,
    ) -> dict[str, object]:
        """Return a compact time series and summary for one measured stage."""

        with self._lock:
            samples = [
                dict(sample)
                for sample in self._samples
                if started_at_monotonic
                <= float(sample["timestamp_monotonic"])
                <= ended_at_monotonic
            ]
        public_samples: list[dict[str, object]] = []
        for sample in samples:
            public_samples.append(
                {
                    "offset_seconds": round(
                        float(sample["timestamp_monotonic"])
                        - started_at_monotonic,
                        6,
                    ),
                    "total_mib": int(sample["total_mib"]),
                    "free_mib": int(sample["free_mib"]),
                    "used_mib": int(sample["used_mib"]),
                    "net_service_mib": int(sample["net_service_mib"]),
                    "process_mib": dict(sample["process_mib"]),
                }
            )
        total_values = [int(sample["total_mib"]) for sample in samples]
        used_values = [int(sample["used_mib"]) for sample in samples]
        free_values = [int(sample["free_mib"]) for sample in samples]
        net_values = [int(sample["net_service_mib"]) for sample in samples]
        process_names = sorted(
            {
                name
                for sample in samples
                for name in sample["process_mib"]
            }
        )
        process_values = {
            name: [
                int(sample["process_mib"][name])
                for sample in samples
                if name in sample["process_mib"]
            ]
            for name in process_names
        }
        process_peak = {
            name: max(values) for name, values in process_values.items() if values
        }
        process_average = {
            name: round(sum(values) / len(values), 3)
            for name, values in process_values.items()
            if values
        }
        near_peak_process_fraction = {
            name: round(
                sum(value >= 0.95 * process_peak[name] for value in values)
                / len(values),
                3,
            )
            for name, values in process_values.items()
            if values and process_peak.get(name, 0) > 0
        }
        peak_total = max(total_values, default=0)
        peak_used = max(used_values, default=0)
        average_used = (
            round(sum(used_values) / len(used_values), 3)
            if used_values
            else 0.0
        )
        stage_start_used = used_values[0] if used_values else 0
        transient_increment = max(0, peak_used - stage_start_used)
        budget_utilization = (
            round(100 * transient_increment / effective_budget_mib, 3)
            if effective_budget_mib and effective_budget_mib > 0
            else None
        )
        remaining_over_safety = (
            max(0, min(free_values) - safety_margin_mib)
            if free_values and safety_margin_mib is not None
            else None
        )
        process_pid_values = {
            pid: [
                int(sample["process_pid_mib"][pid])
                for sample in samples
                if pid in sample["process_pid_mib"]
            ]
            for pid in sorted(
                {
                    pid
                    for sample in samples
                    for pid in sample["process_pid_mib"]
                }
            )
        }
        return {
            "stage_start_monotonic": started_at_monotonic,
            "stage_end_monotonic": ended_at_monotonic,
            "sampling_interval_ms": self.interval_ms,
            "sample_count": len(public_samples),
            "summary": {
                "total_mib": peak_total,
                "stage_start_used_mib": stage_start_used,
                "aed_transient_increment_mib": transient_increment,
                "effective_budget_mib": effective_budget_mib,
                "budget_utilization_pct": budget_utilization,
                "remaining_over_safety_mib": remaining_over_safety,
                "peak_used_mib": peak_used,
                "average_used_mib": average_used,
                "minimum_free_mib": min(free_values, default=0),
                "peak_net_service_mib": max(net_values, default=0),
                "average_net_service_mib": (
                    round(sum(net_values) / len(net_values), 3)
                    if net_values
                    else 0.0
                ),
                "peak_utilization_pct": round(
                    100 * peak_used / peak_total, 3
                )
                if peak_total
                else 0.0,
                "average_utilization_pct": round(
                    100 * average_used / peak_total, 3
                )
                if peak_total
                else 0.0,
                "near_peak_threshold_pct": 95,
                "near_peak_used_fraction": round(
                    sum(value >= 0.95 * peak_used for value in used_values)
                    / len(used_values),
                    3,
                )
                if peak_used and used_values
                else 0.0,
                "near_peak_net_service_fraction": round(
                    sum(value >= 0.95 * max(net_values) for value in net_values)
                    / len(net_values),
                    3,
                )
                if net_values and max(net_values) > 0
                else 0.0,
                "process_peak_mib": process_peak,
                "process_average_mib": process_average,
                "near_peak_process_fraction": near_peak_process_fraction,
                "unique_process_peak_mib": {
                    pid: max(values)
                    for pid, values in process_pid_values.items()
                    if values
                },
            },
            "samples": public_samples,
        }

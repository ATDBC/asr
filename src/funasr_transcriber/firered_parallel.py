"""Pure planning helpers for fixed-plan multi-AED inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from .firered_scheduler import BatchDecision, MemoryFirstBatchPacker
from .schemas import AsrResult


@dataclass(frozen=True, slots=True)
class GpuMemorySnapshot:
    total_mib: int
    free_mib: int
    process_mib: dict[int, int]


def derive_transient_budget_mib(
    snapshot: GpuMemorySnapshot,
    *,
    safety_margin_mib: int,
) -> int:
    if safety_margin_mib < 0:
        raise ValueError("safety_margin_mib must not be negative")
    return max(0, int(snapshot.free_mib) - int(safety_margin_mib))


def required_concurrent_increment_mib(
    assignments: Sequence[Sequence[BatchDecision]],
) -> int:
    return sum(
        max(
            (batch.estimate.predicted_incremental_mib for batch in lane),
            default=0,
        )
        for lane in assignments
    )


def _default_gpu_memory_provider(pids: Sequence[int]) -> GpuMemorySnapshot:
    import subprocess

    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    wanted = {int(pid) for pid in pids if int(pid) > 0}
    process_mib: dict[int, int] = {}
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode == 0:
        for line in query.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                continue
            try:
                pid, memory = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            if pid in wanted:
                process_mib[pid] = memory
    return GpuMemorySnapshot(
        total_mib=int(total_bytes / (1024 * 1024)),
        free_mib=int(free_bytes / (1024 * 1024)),
        process_mib=process_mib,
    )


@dataclass(frozen=True, slots=True)
class PartitionSummary:
    """A deterministic worker assignment and its predicted runtime."""

    worker_index: int
    batches: tuple[BatchDecision, ...]
    predicted_seconds: float


def build_global_plan(
    segments: Sequence[tuple[float, float]],
    *,
    shared_budget_mib: int,
    worker_resident_mib: tuple[int, ...],
) -> tuple[BatchDecision, ...]:
    """Build the one immutable memory-first plan shared by AED workers."""

    return tuple(
        MemoryFirstBatchPacker(
            segments,
            shared_budget_mib=shared_budget_mib,
            worker_resident_mib=worker_resident_mib,
        ).plan()
    )


def _loads(assignments: Sequence[Sequence[BatchDecision]]) -> list[float]:
    return [
        sum(float(batch.estimate.predicted_seconds) for batch in batches)
        for batches in assignments
    ]


def partition_batches(
    batches: Sequence[BatchDecision],
    *,
    worker_count: int = 2,
) -> tuple[tuple[BatchDecision, ...], ...]:
    """Assign complete batches using deterministic longest-processing-time.

    The scheduler's batch boundaries are never changed here. A batch is an
    indivisible unit because splitting it would invalidate its padding and
    memory estimate. Ties are resolved by worker index and original plan order.
    """

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    assignments: list[list[BatchDecision]] = [
        [] for _ in range(worker_count)
    ]
    # Keep the original plan position as a stable tie-breaker. Python's sort is
    # stable, so equal predicted runtimes retain the global plan order.
    ordered = sorted(
        enumerate(batches),
        key=lambda pair: (-float(pair[1].estimate.predicted_seconds), pair[0]),
    )
    for _, batch in ordered:
        loads = _loads(assignments)
        target = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[target].append(batch)

    # Preserve execution order within each worker. This is useful for stable
    # diagnostics and does not affect the load-balancing decision above.
    plan_position = {id(batch): position for position, batch in enumerate(batches)}
    for worker_batches in assignments:
        worker_batches.sort(key=lambda batch: plan_position[id(batch)])

    return tuple(tuple(worker_batches) for worker_batches in assignments)


def serialize_batch_plan(
    batches: Sequence[BatchDecision],
) -> list[list[dict[str, int]]]:
    """Serialize planned batches for the JSONL worker protocol."""

    return [
        [
            {
                "index": int(item.index),
                "start_ms": int(round(item.start_s * 1000)),
                "end_ms": int(round(item.end_s * 1000)),
            }
            for item in batch.items
        ]
        for batch in batches
    ]


def summarize_partition(
    assignments: Sequence[Sequence[BatchDecision]],
) -> tuple[PartitionSummary, ...]:
    """Return compact immutable diagnostics for a worker partition."""

    return tuple(
        PartitionSummary(
            worker_index=index,
            batches=tuple(batches),
            predicted_seconds=sum(
                float(batch.estimate.predicted_seconds) for batch in batches
            ),
        )
        for index, batches in enumerate(assignments)
    )


def _prediction_accuracy(
    assignments: Sequence[Sequence[BatchDecision]],
    responses: Sequence[dict[str, object]],
    resident_process_mib: dict[int, int],
    workers: Sequence[Any],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for worker_index, (lane, response, worker) in enumerate(
        zip(assignments, responses, workers)
    ):
        raw_metrics = response.get("batch_metrics", [])
        if not isinstance(raw_metrics, list):
            continue
        pid = getattr(worker, "pid", None)
        fallback_resident = (
            resident_process_mib.get(int(pid), 0)
            if isinstance(pid, int)
            else 0
        )
        response_resident = response.get("resident_allocated_mib")
        resident = (
            int(response_resident)
            if isinstance(response_resident, (int, float))
            else fallback_resident
        )
        for batch_index, (decision, raw) in enumerate(zip(lane, raw_metrics)):
            if not isinstance(raw, dict):
                continue
            actual_seconds = float(raw.get("actual_seconds") or 0.0)
            actual_peak = raw.get("peak_allocated_mib")
            actual_increment = (
                None if actual_peak is None else max(0, int(actual_peak) - resident)
            )
            predicted_seconds = float(decision.estimate.predicted_seconds)
            predicted_increment = decision.estimate.predicted_incremental_mib
            records.append({
                "worker_index": worker_index,
                "batch_index": batch_index,
                "indices": [item.index for item in decision.items],
                "predicted_seconds": predicted_seconds,
                "actual_seconds": actual_seconds,
                "seconds_error": actual_seconds - predicted_seconds,
                "predicted_incremental_mib": predicted_increment,
                "actual_incremental_mib": actual_increment,
                "incremental_mib_error": (
                    None
                    if actual_increment is None
                    else actual_increment - predicted_increment
                ),
            })
    return {
        "batches": len(records),
        "records": records,
        "predicted_seconds": sum(float(item["predicted_seconds"]) for item in records),
        "actual_seconds": sum(float(item["actual_seconds"]) for item in records),
    }


class ParallelFireRedWorkerClient:
    """Two AED clients with worker 0 as the single VAD/Punc owner."""

    def __init__(
        self,
        worker0: Any,
        worker1: Any,
        *,
        gpu_safety_margin_mib: int = 1_536,
        gpu_memory_provider: Any | None = None,
    ):
        self.worker0 = worker0
        self.worker1 = worker1
        self.gpu_safety_margin_mib = int(gpu_safety_margin_mib)
        self.gpu_memory_provider = (
            gpu_memory_provider or _default_gpu_memory_provider
        )
        self.worker_resident_mib: tuple[int, ...] = ()
        self.last_response: dict[str, object] = {}

    @property
    def pid(self) -> int | None:
        return getattr(self.worker0, "pid", None)

    @property
    def pids(self) -> tuple[int, ...]:
        return tuple(
            pid for pid in (
                getattr(self.worker0, "pid", None),
                getattr(self.worker1, "pid", None),
            ) if pid is not None
        )

    def start(self) -> "ParallelFireRedWorkerClient":
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(self.worker0.start)
                second = executor.submit(self.worker1.start)
                first.result()
                second.result()
        except Exception:
            self.close()
            raise
        return self

    def status(self) -> dict[str, object]:
        return {
            "workers": [self.worker0.status(), self.worker1.status()],
            "worker_count": 2,
        }

    def warmup(self, audio_path: Path | None = None) -> dict[str, object]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(client.warmup, audio_path)
                for client in (self.worker0, self.worker1)
            ]
        results = [future.result() for future in futures]
        self.worker_resident_mib = tuple(
            int(item.get("resident_allocated_mib") or 0) for item in results
        )
        return {
            "excluded": all(bool(item.get("excluded")) for item in results),
            "seconds": max(float(item.get("seconds") or 0.0) for item in results),
            "workers": results,
        }

    def transcribe(
        self,
        audio_path: Path,
        vad_segments_ms: list[list[int]] | None = None,
    ) -> AsrResult:
        if vad_segments_ms is None:
            return self.worker0.transcribe(audio_path)
        started = perf_counter()
        vad_started = perf_counter()
        refined = self.worker0.refine_vad(audio_path, vad_segments_ms)
        vad_seconds = perf_counter() - vad_started
        refined_seconds = [
            (float(start) / 1000.0, float(end) / 1000.0)
            for start, end in refined
        ]
        before_plan = self.gpu_memory_provider(self.pids)
        effective_budget = derive_transient_budget_mib(
            before_plan,
            safety_margin_mib=self.gpu_safety_margin_mib,
        )
        plan_started = perf_counter()
        plan = build_global_plan(
            refined_seconds,
            shared_budget_mib=effective_budget,
            worker_resident_mib=(
                self.worker_resident_mib
                or tuple(before_plan.process_mib.get(pid, 0) for pid in self.pids)
            ),
        )
        plan_seconds = perf_counter() - plan_started
        assignments = partition_batches(plan, worker_count=2)
        from .firered_parallel import serialize_batch_plan as _serialize

        plans = [_serialize(batches) for batches in assignments]
        aed_started = perf_counter()
        def execute(client: Any, worker_plan: list[list[dict[str, int]]]):
            started = perf_counter()
            response = client.run_aed_plan(
                audio_path,
                worker_plan,
                refined,
            )
            return response, perf_counter() - started

        before_dispatch = self.gpu_memory_provider(self.pids)
        dispatch_budget = derive_transient_budget_mib(
            before_dispatch,
            safety_margin_mib=self.gpu_safety_margin_mib,
        )
        dispatch_mode = "concurrent"
        if required_concurrent_increment_mib(assignments) > dispatch_budget:
            dispatch_mode = "serial"
        largest_single_increment = max(
            (batch.estimate.predicted_incremental_mib for batch in plan),
            default=0,
        )
        if largest_single_increment > dispatch_budget:
            raise RuntimeError(
                "AED transient memory budget cannot fit even one planned batch: "
                f"required={largest_single_increment} MiB, budget={dispatch_budget} MiB"
            )
        if dispatch_mode == "concurrent":
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(execute, client, worker_plan)
                    for client, worker_plan in zip(
                        (self.worker0, self.worker1), plans
                    )
                ]
                timed_responses = [future.result() for future in futures]
        else:
            timed_responses = [
                execute(client, worker_plan)
                for client, worker_plan in zip(
                    (self.worker0, self.worker1), plans
                )
            ]
        responses = [item[0] for item in timed_responses]
        worker_aed_seconds = [float(item[1]) for item in timed_responses]
        aed_seconds = perf_counter() - aed_started
        merged: list[dict[str, object]] = []
        duration_ms = 0
        for response in responses:
            raw = response.get("aed_results")
            if isinstance(raw, list):
                merged.extend(item for item in raw if isinstance(item, dict))
            duration_ms = max(duration_ms, int(response.get("duration_ms") or 0))
        punc_started = perf_counter()
        punctuated = self.worker0.punctuate(
            audio_path,
            merged,
            duration_ms,
            refined,
        )
        punc_seconds = perf_counter() - punc_started
        self.last_response = {
            "metrics": {
                "parallel": {
                "warmup_excluded": True,
                "dispatch_mode": dispatch_mode,
                "memory": {
                    "planned_free_mib": before_plan.free_mib,
                    "dispatch_free_mib": before_dispatch.free_mib,
                    "effective_transient_budget_mib": effective_budget,
                    "dispatch_transient_budget_mib": dispatch_budget,
                    "required_concurrent_increment_mib": (
                        required_concurrent_increment_mib(assignments)
                    ),
                    "resident_process_mib": dict(before_plan.process_mib),
                },
                "prediction_accuracy": _prediction_accuracy(
                    assignments,
                    responses,
                    before_plan.process_mib,
                    (self.worker0, self.worker1),
                ),
                "vad_seconds": vad_seconds,
                "plan_seconds": plan_seconds,
                "aed_seconds": aed_seconds,
                "aed_worker_seconds": worker_aed_seconds,
                "punc_seconds": punc_seconds,
                "total_seconds": perf_counter() - started,
                "worker_count": 2,
                "planned_batches": len(plan),
                "worker_batch_counts": [len(items) for items in assignments],
                },
                "scheduler": {
                "planner": "memory_first",
                "shared_budget_mib": effective_budget,
                "lane_budget_mib": max(1, effective_budget // 2),
                "worker_resident_mib": list(
                    self.worker_resident_mib
                    or tuple(before_plan.process_mib.get(pid, 0) for pid in self.pids)
                ),
                "planned_batches": len(plan),
                "planned_items": sum(len(batch.items) for batch in plan),
                "batch_sizes": [len(batch.items) for batch in plan],
                "worker_predicted_seconds": [
                    summary.predicted_seconds
                    for summary in summarize_partition(assignments)
                ],
                "worker_plans": [
                    serialize_batch_plan(items) for items in assignments
                ],
                },
            },
        }
        return punctuated

    def verify(self, audio_path: Path) -> AsrResult:
        return self.worker0.verify(audio_path)

    def close(self) -> None:
        try:
            self.worker0.close()
        finally:
            self.worker1.close()

    def __enter__(self) -> "ParallelFireRedWorkerClient":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

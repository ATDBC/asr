"""Pure planning helpers for fixed-plan multi-AED inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from .firered_scheduler import AdaptiveBatchScheduler, BatchDecision
from .schemas import AsrResult


@dataclass(frozen=True, slots=True)
class PartitionSummary:
    """A deterministic worker assignment and its predicted runtime."""

    worker_index: int
    batches: tuple[BatchDecision, ...]
    predicted_seconds: float


def build_global_plan(
    segments: Sequence[tuple[float, float]],
    *,
    batch_size: int,
    frame_budget: int = 8000,
    memory_limit_mib: int | None = 10_000,
    memory_safety_mib: int = 512,
) -> tuple[BatchDecision, ...]:
    """Build the one immutable plan shared by every AED worker."""

    scheduler = AdaptiveBatchScheduler(
        segments,
        max_batch_size=batch_size,
        frame_budget=frame_budget,
        memory_limit_mib=memory_limit_mib,
        safety_mib=memory_safety_mib,
    )
    return tuple(scheduler.plan())


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


class ParallelFireRedWorkerClient:
    """Two AED clients with worker 0 as the single VAD/Punc owner."""

    def __init__(
        self,
        worker0: Any,
        worker1: Any,
        *,
        batch_size: int,
        frame_budget: int = 8000,
        memory_limit_mib: int | None = 10_000,
        memory_safety_mib: int = 512,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.worker0 = worker0
        self.worker1 = worker1
        self.batch_size = int(batch_size)
        self.frame_budget = int(frame_budget)
        self.memory_limit_mib = memory_limit_mib
        self.memory_safety_mib = int(memory_safety_mib)
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
        plan_started = perf_counter()
        plan = build_global_plan(
            refined_seconds,
            batch_size=self.batch_size,
            frame_budget=self.frame_budget,
            memory_limit_mib=self.memory_limit_mib,
            memory_safety_mib=self.memory_safety_mib,
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

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    execute,
                    client,
                    worker_plan,
                )
                for client, worker_plan in zip(
                    (self.worker0, self.worker1), plans
                )
            ]
            timed_responses = [future.result() for future in futures]
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
                "planned_batches": len(plan),
                "planned_items": sum(len(batch.items) for batch in plan),
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

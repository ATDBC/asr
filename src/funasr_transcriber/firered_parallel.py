"""Pure planning helpers for fixed-plan multi-AED inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from time import perf_counter

from .firered_scheduler import BatchCostModel, BatchDecision, MemoryFirstBatchPacker
from .firered_contract import normalize_official_result
from .firered_client import parse_worker_response
from .firered_worker import (
    _aed_plan_summary,
    plan_aed_batches,
)
from .weighted_aed_calibration import (
    DEFAULT_WEIGHTED_AED_WEIGHTS,
    WeightedAedWeights,
)
from .punc_scheduler import (
    DEFAULT_PUNC_MAX_ITEMS,
    DEFAULT_PUNC_TOKEN_BUDGET,
    assemble_punc_items,
)
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
    cost_model: BatchCostModel | None = None,
) -> tuple[BatchDecision, ...]:
    """Build the one immutable memory-first plan shared by AED workers."""

    return tuple(
        MemoryFirstBatchPacker(
            segments,
            shared_budget_mib=shared_budget_mib,
            worker_resident_mib=worker_resident_mib,
            cost_model=cost_model or BatchCostModel(),
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

    # Improve the longest-processing-time result with deterministic one-batch
    # moves.  This preserves complete batch boundaries while fixing cases in
    # which an early LPT choice leaves a better local split available.
    while worker_count > 1:
        current_loads = _loads(assignments)
        current_makespan = max(current_loads, default=0.0)
        source = max(range(worker_count), key=lambda i: (current_loads[i], -i))
        target = min(range(worker_count), key=lambda i: (current_loads[i], i))
        if source == target or not assignments[source]:
            break
        best_move: tuple[float, int] | None = None
        for batch_index, batch in enumerate(assignments[source]):
            duration = float(batch.estimate.predicted_seconds)
            candidate_loads = list(current_loads)
            candidate_loads[source] -= duration
            candidate_loads[target] += duration
            candidate_makespan = max(candidate_loads)
            if candidate_makespan < current_makespan:
                candidate = (candidate_makespan, batch_index)
                if best_move is None or candidate < best_move:
                    best_move = candidate
        if best_move is None:
            break
        _, batch_index = best_move
        assignments[target].append(assignments[source].pop(batch_index))

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
        response_reserved = response.get("resident_reserved_mib")
        reserved = (
            int(response_reserved)
            if isinstance(response_reserved, (int, float))
            else None
        )
        for batch_index, (decision, raw) in enumerate(zip(lane, raw_metrics)):
            if not isinstance(raw, dict):
                continue
            actual_seconds = float(raw.get("actual_seconds") or 0.0)
            actual_peak = raw.get("peak_allocated_mib")
            actual_increment = (
                None if actual_peak is None else max(0, int(actual_peak) - resident)
            )
            actual_reserved_peak = raw.get("peak_reserved_mib")
            actual_reserved_increment = (
                None
                if actual_reserved_peak is None or reserved is None
                else max(0, int(actual_reserved_peak) - reserved)
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
                "actual_allocated_incremental_mib": actual_increment,
                "actual_reserved_incremental_mib": actual_reserved_increment,
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
        punc_owner: Any | None = None,
        punc_token_budget: int = DEFAULT_PUNC_TOKEN_BUDGET,
        punc_max_items: int = DEFAULT_PUNC_MAX_ITEMS,
        gpu_safety_margin_mib: int = 1_536,
        gpu_memory_provider: Any | None = None,
    ):
        self.worker0 = worker0
        self.worker1 = worker1
        self.punc_owner = punc_owner or worker0
        self.punc_token_budget = int(punc_token_budget)
        self.punc_max_items = int(punc_max_items)
        self.gpu_safety_margin_mib = int(gpu_safety_margin_mib)
        self.gpu_memory_provider = (
            gpu_memory_provider or _default_gpu_memory_provider
        )
        self.worker_resident_mib: tuple[int, ...] = ()
        self.cost_model = BatchCostModel()
        self.aed_budget_fraction = 0.9
        self.last_response: dict[str, object] = {}

    @property
    def pid(self) -> int | None:
        return getattr(self.worker0, "pid", None)

    @property
    def pids(self) -> tuple[int, ...]:
        pids: list[int] = []
        for client in (self.worker0, self.worker1, self.punc_owner):
            pid = getattr(client, "pid", None)
            if pid is not None and pid not in pids:
                pids.append(pid)
        return tuple(pids)

    def start(self) -> "ParallelFireRedWorkerClient":
        try:
            clients = []
            for client in (self.worker0, self.worker1, self.punc_owner):
                if client not in clients:
                    clients.append(client)
            with ThreadPoolExecutor(max_workers=len(clients)) as executor:
                futures = [executor.submit(client.start) for client in clients]
                for future in futures:
                    future.result()
        except Exception:
            self.close()
            raise
        return self

    def status(self) -> dict[str, object]:
        return {
            "workers": [self.worker0.status(), self.worker1.status()],
            "punc": self.punc_owner.status()
            if self.punc_owner is not self.worker0
            else None,
            "worker_count": 2,
        }

    def warmup(self, audio_path: Path | None = None) -> dict[str, object]:
        clients = []
        for client in (self.worker0, self.worker1, self.punc_owner):
            if client not in clients:
                clients.append(client)
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            futures = [
                executor.submit(client.warmup, audio_path)
                for client in clients
            ]
        results = [future.result() for future in futures]
        self.worker_resident_mib = tuple(
            int(results[index].get("resident_allocated_mib") or 0)
            for index in range(2)
        )
        return {
            "excluded": all(bool(item.get("excluded")) for item in results),
            "seconds": max(float(item.get("seconds") or 0.0) for item in results),
            "workers": results[:2],
            "punc": results[2] if len(results) > 2 else None,
            "cost_model": self.cost_model.summary(),
        }

    def cleanup_cuda_cache(self) -> dict[str, object]:
        """Concurrently trim unused allocator cache in both AED workers."""

        started = perf_counter()
        workers = (self.worker0, self.worker1)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(worker.cleanup_cuda_cache)
                for worker in workers
            ]
            results = [future.result() for future in futures]
        records: list[dict[str, object]] = []
        for index, (worker, result) in enumerate(zip(workers, results)):
            record = dict(result)
            record["worker_index"] = index
            record["pid"] = getattr(worker, "pid", None)
            records.append(record)
        return {
            "seconds": perf_counter() - started,
            "workers": records,
        }

    def _transcribe_streaming(
        self,
        audio_path: Path,
        refined: list[list[int]],
        plan: tuple[BatchDecision, ...],
        assignments: tuple[tuple[BatchDecision, ...], ...],
        plans: list[list[list[dict[str, int]]]],
        dispatch_mode: str,
        before_plan: GpuMemorySnapshot,
        before_dispatch: GpuMemorySnapshot,
        effective_budget: int,
        dispatch_budget: int,
        started: float,
        vad_seconds: float,
        plan_seconds: float,
    ) -> AsrResult:
        """Overlap AED batch events with one independent Punc owner."""

        sentinel = object()
        queue: Queue[object] = Queue(maxsize=8)
        punc_items: list[dict[str, object]] = []
        punc_batch_metrics: list[dict[str, object]] = []
        punc_batches = 0
        punc_model_seconds = 0.0
        max_queue_depth = 0

        def flush_pending(pending: list[dict[str, object]]) -> None:
            nonlocal punc_batches, punc_model_seconds
            if not pending:
                return
            punc_started = perf_counter()
            chunk = self.punc_owner.punctuate_chunk(audio_path, pending)
            punc_model_seconds += perf_counter() - punc_started
            punc_batches += 1
            punc_items.extend(chunk)
            response = getattr(self.punc_owner, "last_response", {})
            metrics = response.get("punc_metrics") if isinstance(response, dict) else None
            if isinstance(metrics, dict):
                punc_batch_metrics.append(dict(metrics))
            pending.clear()

        def consume_punc() -> dict[str, object]:
            nonlocal max_queue_depth
            pending: list[dict[str, object]] = []
            completed_lanes = 0
            punc_started = perf_counter()
            while completed_lanes < len(assignments):
                try:
                    item = queue.get(timeout=0.05)
                except Empty:
                    flush_pending(pending)
                    continue
                if item is sentinel:
                    completed_lanes += 1
                    flush_pending(pending)
                    continue
                if isinstance(item, list):
                    pending.extend(
                        record for record in item if isinstance(record, dict)
                    )
                    max_queue_depth = max(max_queue_depth, queue.qsize())
                    token_count = 0
                    for record in pending:
                        result = record.get("result")
                        if isinstance(result, dict):
                            timestamp = result.get("timestamp", [])
                            if isinstance(timestamp, (list, tuple)):
                                token_count += len(timestamp)
                    if (
                        token_count >= self.punc_token_budget
                        or len(pending) >= self.punc_max_items
                    ):
                        flush_pending(pending)
            flush_pending(pending)
            return {
                "wall_seconds": perf_counter() - punc_started,
                "batches": punc_batches,
                "model_seconds": punc_model_seconds,
                "max_queue_depth": max_queue_depth,
            }

        def execute(
            client: Any,
            worker_plan: list[list[dict[str, int]]],
        ) -> tuple[dict[str, object], float]:
            def on_event(event: dict[str, object]) -> None:
                raw = event.get("aed_results")
                if isinstance(raw, list) and raw:
                    queue.put(raw)

            worker_started = perf_counter()
            try:
                response = client.run_aed_plan_stream(
                    audio_path,
                    worker_plan,
                    refined,
                    on_event,
                )
                return response, perf_counter() - worker_started
            finally:
                queue.put(sentinel)

        aed_started = perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            punc_future = executor.submit(consume_punc)
            futures = [
                executor.submit(execute, client, worker_plan)
                for client, worker_plan in zip(
                    (self.worker0, self.worker1), plans
                )
            ]
            timed_responses = [future.result() for future in futures]
            aed_ended = perf_counter()
            punc_metrics = punc_future.result()
        aed_wall_seconds = aed_ended - aed_started
        responses = [item[0] for item in timed_responses]
        worker_aed_seconds = [float(item[1]) for item in timed_responses]
        durations = [int(response.get("duration_ms") or 0) for response in responses]
        duration_ms = max(durations, default=0)
        raw_started = perf_counter()
        raw = assemble_punc_items(
            "parallel",
            punc_items,
            duration_ms=duration_ms,
            vad_segments_ms=refined,
        )
        normalized = normalize_official_result("parallel", raw)
        result = parse_worker_response(normalized)
        normalization_seconds = perf_counter() - raw_started
        aed_batch_metrics = [
            metric
            for response in responses
            for metric in response.get("batch_metrics", ())
            if isinstance(metric, dict)
        ]

        def peak(records: list[dict[str, object]], key: str) -> int | None:
            values = [
                int(record[key])
                for record in records
                if isinstance(record.get(key), (int, float))
            ]
            return max(values, default=None)

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
                    "aed_wall_seconds": aed_wall_seconds,
                    "aed_started_at_monotonic": aed_started,
                    "aed_ended_at_monotonic": aed_ended,
                    "aed_worker_seconds": worker_aed_seconds,
                    "punc_queue_wall_seconds": float(punc_metrics["wall_seconds"]),
                    "punc_model_seconds": float(punc_metrics["model_seconds"]),
                    "punc_normalization_seconds": normalization_seconds,
                    "punc_batches": int(punc_metrics["batches"]),
                    "punc_max_queue_depth": int(punc_metrics["max_queue_depth"]),
                    "punc_token_budget": self.punc_token_budget,
                    "punc_max_items": self.punc_max_items,
                    "memory_peaks": {
                        "aed": {
                            "allocated_mib": peak(
                                aed_batch_metrics, "peak_allocated_mib"
                            ),
                            "reserved_mib": peak(
                                aed_batch_metrics, "peak_reserved_mib"
                            ),
                        },
                        "punc": {
                            "allocated_mib": peak(
                                punc_batch_metrics, "peak_allocated_mib"
                            ),
                            "reserved_mib": peak(
                                punc_batch_metrics, "peak_reserved_mib"
                            ),
                        },
                    },
                    "total_seconds": perf_counter() - started,
                    "worker_count": 2,
                    "planned_batches": len(plan),
                    "worker_batch_counts": [len(items) for items in assignments],
                    "cost_model_observations": 0,
                },
                "scheduler": {
                    "planner": "memory_first",
                    "model": self.cost_model.summary(),
                    "shared_budget_mib": effective_budget,
                    "lane_budget_mib": max(1, effective_budget // 2),
                    "worker_resident_mib": list(self.worker_resident_mib),
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
        return result

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
        budget_fraction = min(max(float(self.aed_budget_fraction), 0.1), 1.0)
        effective_budget = int(round(derive_transient_budget_mib(
            before_plan,
            safety_margin_mib=self.gpu_safety_margin_mib,
        ) * budget_fraction))
        plan_started = perf_counter()
        plan = build_global_plan(
            refined_seconds,
            shared_budget_mib=effective_budget,
            worker_resident_mib=(
                self.worker_resident_mib
                or tuple(before_plan.process_mib.get(pid, 0) for pid in self.pids)
            ),
            cost_model=self.cost_model,
        )
        plan_seconds = perf_counter() - plan_started
        assignments = partition_batches(plan, worker_count=2)
        from .firered_parallel import serialize_batch_plan as _serialize

        plans = [_serialize(batches) for batches in assignments]

        before_dispatch = self.gpu_memory_provider(self.pids)
        dispatch_budget = int(round(derive_transient_budget_mib(
            before_dispatch,
            safety_margin_mib=self.gpu_safety_margin_mib,
        ) * budget_fraction))
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
        if (
            self.punc_owner is not self.worker0
            and all(hasattr(client, "run_aed_plan_stream") for client in (self.worker0, self.worker1))
            and hasattr(self.punc_owner, "punctuate_chunk")
        ):
            return self._transcribe_streaming(
                audio_path,
                refined,
                tuple(plan),
                tuple(tuple(items) for items in assignments),
                plans,
                dispatch_mode,
                before_plan,
                before_dispatch,
                effective_budget,
                dispatch_budget,
                started,
                vad_seconds,
                plan_seconds,
            )
        aed_started = perf_counter()
        def execute(client: Any, worker_plan: list[list[dict[str, int]]]):
            started = perf_counter()
            response = client.run_aed_plan(
                audio_path,
                worker_plan,
                refined,
            )
            return response, perf_counter() - started
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
        aed_ended = perf_counter()
        aed_wall_seconds = aed_ended - aed_started
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
        punc_queue_wall_seconds = perf_counter() - punc_started
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
                "aed_wall_seconds": aed_wall_seconds,
                "aed_started_at_monotonic": aed_started,
                "aed_ended_at_monotonic": aed_ended,
                "aed_worker_seconds": worker_aed_seconds,
                "punc_queue_wall_seconds": punc_queue_wall_seconds,
                "total_seconds": perf_counter() - started,
                "worker_count": 2,
                "planned_batches": len(plan),
                "worker_batch_counts": [len(items) for items in assignments],
                "cost_model_observations": 0,
                },
                "scheduler": {
                "model": self.cost_model.summary(),
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
        clients = []
        for client in (self.worker0, self.worker1, self.punc_owner):
            if client not in clients:
                clients.append(client)
        first_error: BaseException | None = None
        for client in clients:
            try:
                client.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "ParallelFireRedWorkerClient":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class WeightedAedPuncWorkerClient(ParallelFireRedWorkerClient):
    """One or two weighted AED lanes streaming into one Punc owner."""

    def __init__(
        self,
        worker0: Any,
        punc_owner: Any,
        *,
        worker1: Any | None = None,
        weights: WeightedAedWeights = DEFAULT_WEIGHTED_AED_WEIGHTS,
        weight_source: str = "builtin",
        weight_artifact_checksum: str | None = None,
        weight_warnings: tuple[str, ...] = (),
        punc_token_budget: int = DEFAULT_PUNC_TOKEN_BUDGET,
        punc_max_items: int = DEFAULT_PUNC_MAX_ITEMS,
    ):
        super().__init__(
            worker0,
            worker1 or worker0,
            punc_owner=punc_owner,
            punc_token_budget=punc_token_budget,
            punc_max_items=punc_max_items,
        )
        self.aed_workers = (
            (worker0,) if worker1 is None else (worker0, worker1)
        )
        self.weights = weights
        self.weight_source = str(weight_source)
        self.weight_artifact_checksum = weight_artifact_checksum
        self.weight_warnings = tuple(str(item) for item in weight_warnings)
        self.cost_model = BatchCostModel(
            base_transient_mib=0,
            memory_per_padded_frame_mib=0,
            base_seconds=0,
            seconds_per_padded_frame=weights.padded_weight / 100,
            seconds_per_total_frame=0,
            seconds_per_max_frame=weights.longest_weight / 100,
            seconds_per_item=0,
        )

    @property
    def pids(self) -> tuple[int, ...]:
        return tuple(
            pid
            for worker in self.aed_workers
            if isinstance((pid := getattr(worker, "pid", None)), int)
            and pid > 0
        )

    def status(self) -> dict[str, object]:
        return {
            "workers": [worker.status() for worker in self.aed_workers],
            "punc": self.punc_owner.status(),
            "worker_count": len(self.aed_workers),
        }

    def warmup(self, audio_path: Path | None = None) -> dict[str, object]:
        clients = (*self.aed_workers, self.punc_owner)
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            futures = [
                executor.submit(client.warmup, audio_path)
                for client in clients
            ]
            results = [future.result() for future in futures]
        worker_count = len(self.aed_workers)
        self.worker_resident_mib = tuple(
            int(result.get("resident_allocated_mib") or 0)
            for result in results[:worker_count]
        )
        return {
            "excluded": all(bool(item.get("excluded")) for item in results),
            "seconds": max(float(item.get("seconds") or 0.0) for item in results),
            "workers": results[:worker_count],
            "punc": results[-1],
            "cost_model": self.cost_model.summary(),
        }

    def cleanup_cuda_cache(self) -> dict[str, object]:
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=len(self.aed_workers)) as executor:
            futures = [
                executor.submit(worker.cleanup_cuda_cache)
                for worker in self.aed_workers
            ]
            results = [future.result() for future in futures]
        records = []
        for index, (worker, result) in enumerate(
            zip(self.aed_workers, results)
        ):
            record = dict(result)
            record["worker_index"] = index
            record["pid"] = getattr(worker, "pid", None)
            records.append(record)
        return {"seconds": perf_counter() - started, "workers": records}

    def transcribe(
        self,
        audio_path: Path,
        vad_segments_ms: list[list[int]] | None = None,
    ) -> AsrResult:
        if vad_segments_ms is None:
            raise ValueError("weighted AED/Punc streaming requires VAD segments")

        started = perf_counter()
        vad_started = perf_counter()
        refined = self.worker0.refine_vad(audio_path, vad_segments_ms)
        vad_seconds = perf_counter() - vad_started
        plan_started = perf_counter()
        batches = plan_aed_batches(
            [
                (float(start) / 1000.0, float(end) / 1000.0)
                for start, end in refined
            ],
            weights=self.weights,
        )
        plan = tuple(
            BatchDecision(
                items=tuple(batch),
                estimate=self.cost_model.estimate(batch),
                decision="weighted_duration",
            )
            for batch in batches
        )
        plan_seconds = perf_counter() - plan_started
        worker_count = len(self.aed_workers)
        assignments = (
            (plan,)
            if worker_count == 1
            else partition_batches(plan, worker_count=worker_count)
        )
        worker_plans = [
            serialize_batch_plan(worker_plan)
            for worker_plan in assignments
        ]
        empty_memory = GpuMemorySnapshot(0, 0, {})
        result = self._transcribe_streaming(
            audio_path,
            refined,
            plan,
            assignments,
            worker_plans,
            "single" if worker_count == 1 else "concurrent",
            empty_memory,
            empty_memory,
            0,
            0,
            started,
            vad_seconds,
            plan_seconds,
        )
        parallel = self.last_response["metrics"]["parallel"]
        parallel.pop("memory", None)
        parallel.pop("prediction_accuracy", None)
        parallel["worker_count"] = worker_count
        scheduler = _aed_plan_summary(
            batches,
            weights=self.weights,
            source=self.weight_source,
            artifact_checksum=self.weight_artifact_checksum,
            warnings=self.weight_warnings,
        )
        if worker_count == 2:
            scheduler.update({
                "planner": "weighted_duration_dual_worker",
                "worker_count": worker_count,
                "worker_batch_counts": [
                    len(worker_plan) for worker_plan in assignments
                ],
                "worker_predicted_seconds": [
                    summary.predicted_seconds
                    for summary in summarize_partition(assignments)
                ],
                "worker_plans": worker_plans,
            })
        self.last_response["metrics"]["scheduler"] = scheduler
        return result


SingleAedPuncWorkerClient = WeightedAedPuncWorkerClient

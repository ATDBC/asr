"""Memory-first deterministic packing for FireRed AED segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


class MemoryCapacityError(RuntimeError):
    """A segment cannot fit in the currently available AED budget."""


@dataclass(frozen=True, slots=True)
class ScheduledItem:
    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class BatchEstimate:
    item_count: int
    total_frames: int
    padded_frames: int
    predicted_seconds: float
    predicted_peak_mib: int
    predicted_incremental_mib: int


@dataclass(slots=True)
class BatchCostModel:
    """Predict cost from padded frames without imposing a batch cardinality."""

    frame_rate: float = 100.0
    # Calibrated against driver-visible BF16 worker peaks. This intentionally
    # predicts above torch allocated tensors to cover allocator/runtime state.
    base_transient_mib: float = 384.0
    memory_per_padded_frame_mib: float = 0.045
    base_seconds: float = 0.05
    seconds_per_padded_frame: float = 0.0003
    seconds_per_item: float = 0.01
    smoothing: float = 0.25
    observations: int = 0

    def _frames(self, item: ScheduledItem) -> int:
        return max(1, int(round(item.duration_s * self.frame_rate)))

    def estimate(
        self,
        items: Sequence[ScheduledItem | tuple[float, float]],
        *,
        resident_mib: int = 0,
    ) -> BatchEstimate:
        normalized = [
            item
            if isinstance(item, ScheduledItem)
            else ScheduledItem(index, float(item[0]), float(item[1]))
            for index, item in enumerate(items)
        ]
        if not normalized:
            raise ValueError("cannot estimate an empty batch")
        frames = [self._frames(item) for item in normalized]
        total_frames = sum(frames)
        padded_frames = max(frames) * len(frames)
        predicted_seconds = (
            self.base_seconds
            + self.seconds_per_padded_frame * padded_frames
            + self.seconds_per_item * len(frames)
        )
        predicted_incremental_mib = int(
            round(self.base_transient_mib
                  + self.memory_per_padded_frame_mib * padded_frames)
        )
        return BatchEstimate(
            item_count=len(normalized),
            total_frames=total_frames,
            padded_frames=padded_frames,
            predicted_seconds=predicted_seconds,
            predicted_peak_mib=int(resident_mib) + predicted_incremental_mib,
            predicted_incremental_mib=predicted_incremental_mib,
        )

    def observe(
        self,
        estimate: BatchEstimate,
        actual_seconds: float,
        actual_peak_mib: int | None,
    ) -> None:
        if estimate.padded_frames <= 0:
            return
        alpha = min(max(self.smoothing, 0.0), 1.0)
        observed_seconds = max(0.0, float(actual_seconds))
        seconds_slope = max(
            0.0,
            (observed_seconds - self.base_seconds
             - self.seconds_per_item * estimate.item_count)
            / estimate.padded_frames,
        )
        self.seconds_per_padded_frame = (
            (1.0 - alpha) * self.seconds_per_padded_frame
            + alpha * seconds_slope
        )
        if actual_peak_mib is not None:
            memory_slope = max(
                0.0,
                (int(actual_peak_mib) - estimate.predicted_peak_mib)
                / estimate.padded_frames,
            )
            self.memory_per_padded_frame_mib = (
                (1.0 - alpha) * self.memory_per_padded_frame_mib
                + alpha * memory_slope
            )
        self.observations += 1


@dataclass(frozen=True, slots=True)
class BatchDecision:
    items: tuple[ScheduledItem, ...]
    estimate: BatchEstimate
    decision: str
    free_memory_mib: int | None = None


@dataclass(slots=True)
class MemoryFirstBatchPacker:
    """Greedily fill batches until the shared-memory lane ceiling is reached."""

    segments: Sequence[tuple[float, float]]
    shared_budget_mib: int
    worker_resident_mib: tuple[int, ...] = ()
    cost_model: BatchCostModel = field(default_factory=BatchCostModel)
    _plan: list[BatchDecision] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.shared_budget_mib <= 0:
            raise ValueError("shared_budget_mib must be positive")
        if not self.worker_resident_mib:
            self.worker_resident_mib = (0,)
        if any(int(value) < 0 for value in self.worker_resident_mib):
            raise ValueError("worker resident memory must not be negative")

    @property
    def lane_budget_mib(self) -> int:
        return max(1, self.shared_budget_mib // max(1, len(self.worker_resident_mib)))

    def plan(self) -> list[BatchDecision]:
        if self._plan is not None:
            return list(self._plan)
        remaining = [
            ScheduledItem(index, float(start), float(end))
            for index, (start, end) in enumerate(self.segments)
        ]
        remaining.sort(key=lambda item: (item.duration_s, item.index))
        resident = max(self.worker_resident_mib)
        planned: list[BatchDecision] = []
        while remaining:
            candidate: list[ScheduledItem] = []
            for item in remaining:
                trial = candidate + [item]
                estimate = self.cost_model.estimate(trial, resident_mib=resident)
                if estimate.predicted_incremental_mib <= self.lane_budget_mib:
                    candidate = trial
                    continue
                if not candidate:
                    raise MemoryCapacityError(
                        "singleton segment cannot fit AED memory budget: "
                        f"index={item.index}, required={estimate.predicted_incremental_mib} MiB, "
                        f"lane_budget={self.lane_budget_mib} MiB"
                    )
                break
            selected = tuple(candidate)
            estimate = self.cost_model.estimate(selected, resident_mib=resident)
            planned.append(
                BatchDecision(
                    selected,
                    estimate,
                    "memory_fit",
                    self.shared_budget_mib,
                )
            )
            selected_indices = {item.index for item in selected}
            remaining = [item for item in remaining if item.index not in selected_indices]
        self._plan = planned
        return list(planned)

    def summary(self) -> dict[str, object]:
        plan = self.plan()
        return {
            "planner": "memory_first",
            "shared_budget_mib": self.shared_budget_mib,
            "lane_budget_mib": self.lane_budget_mib,
            "worker_resident_mib": list(self.worker_resident_mib),
            "planned_batches": len(plan),
            "planned_items": sum(len(batch.items) for batch in plan),
            "model": {
                "base_transient_mib": self.cost_model.base_transient_mib,
                "memory_per_padded_frame_mib": self.cost_model.memory_per_padded_frame_mib,
                "base_seconds": self.cost_model.base_seconds,
                "seconds_per_padded_frame": self.cost_model.seconds_per_padded_frame,
                "observations": self.cost_model.observations,
            },
            "plan": [
                {
                    "item_count": len(batch.items),
                    "indices": [item.index for item in batch.items],
                    "total_frames": batch.estimate.total_frames,
                    "padded_frames": batch.estimate.padded_frames,
                    "predicted_seconds": round(batch.estimate.predicted_seconds, 6),
                    "predicted_peak_mib": batch.estimate.predicted_peak_mib,
                    "predicted_incremental_mib": batch.estimate.predicted_incremental_mib,
                }
                for batch in plan
            ],
        }

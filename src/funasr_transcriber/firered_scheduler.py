"""Adaptive, memory-bounded scheduling for FireRed AED micro-batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


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
    """Small observed-cost model using feature-frame padding as the proxy."""

    frame_rate: float = 100.0
    base_memory_mib: float = 5_000.0
    memory_per_padded_frame_mib: float = 0.03
    base_seconds: float = 0.05
    seconds_per_padded_frame: float = 0.0003
    seconds_per_item: float = 0.01
    smoothing: float = 0.25
    observations: int = 0

    def _frames(self, item: ScheduledItem) -> int:
        return max(1, int(round(item.duration_s * self.frame_rate)))

    def estimate(self, items: Sequence[ScheduledItem | tuple[float, float]]) -> BatchEstimate:
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
        predicted_peak_mib = int(
            round(self.base_memory_mib
                  + self.memory_per_padded_frame_mib * padded_frames)
        )
        predicted_incremental_mib = max(
            0,
            predicted_peak_mib - int(round(self.base_memory_mib)),
        )
        return BatchEstimate(
            item_count=len(normalized),
            total_frames=total_frames,
            padded_frames=padded_frames,
            predicted_seconds=predicted_seconds,
            predicted_peak_mib=predicted_peak_mib,
            predicted_incremental_mib=predicted_incremental_mib,
        )

    def observe(
        self,
        estimate: BatchEstimate,
        actual_seconds: float,
        actual_peak_mib: int | None,
    ) -> None:
        """Update slopes conservatively from one completed AED batch."""

        alpha = min(max(self.smoothing, 0.0), 1.0)
        if estimate.padded_frames <= 0:
            return
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
                (int(actual_peak_mib) - self.base_memory_mib)
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
    free_memory_mib: int | None


@dataclass(slots=True)
class AdaptiveBatchScheduler:
    """Build a fixed duration-ordered plan before executing any batch."""

    segments: Sequence[tuple[float, float]]
    max_batch_size: int
    frame_budget: int
    memory_limit_mib: int | None = 10_000
    safety_mib: int = 512
    cost_model: BatchCostModel = field(default_factory=BatchCostModel)
    free_memory_provider: Callable[[], int | None] | None = None
    transient_memory_limit_mib: int | None = None
    _remaining: list[ScheduledItem] = field(init=False, repr=False)
    _plan: list[BatchDecision] | None = field(default=None, init=False, repr=False)
    _plan_cursor: int = field(default=0, init=False, repr=False)
    _observations: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )
    _prediction_overruns: list[dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )
    planned_free_memory_mib: int | None = field(default=None, init=False)
    decisions: list[BatchDecision] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.frame_budget <= 0:
            raise ValueError("frame_budget must be positive")
        if self.memory_limit_mib is not None and self.memory_limit_mib <= 0:
            raise ValueError("memory_limit_mib must be positive when provided")
        if self.safety_mib < 0:
            raise ValueError("safety_mib must not be negative")
        if (
            self.transient_memory_limit_mib is not None
            and self.transient_memory_limit_mib <= 0
        ):
            raise ValueError("transient_memory_limit_mib must be positive when provided")
        self._remaining = sorted(
            [
                ScheduledItem(index, float(start), float(end))
                for index, (start, end) in enumerate(self.segments)
            ],
            key=lambda item: (item.duration_s, item.index),
        )

    def _candidate(
        self,
        remaining: Sequence[ScheduledItem],
        free_memory: int | None,
    ) -> tuple[list[ScheduledItem], str]:
        candidate = list(remaining[: self.max_batch_size])
        if not candidate:
            return [], "empty"
        decision = "batch_limit"
        while len(candidate) > 1:
            estimate = self.cost_model.estimate(candidate)
            frame_violation = estimate.total_frames > self.frame_budget
            memory_violation = (
                self.memory_limit_mib is not None
                and estimate.predicted_peak_mib + self.safety_mib
                > self.memory_limit_mib
            )
            transient_limit_violation = (
                self.transient_memory_limit_mib is not None
                and estimate.predicted_incremental_mib > self.transient_memory_limit_mib
            )
            transient_mib = max(
                0,
                estimate.predicted_peak_mib
                - int(round(self.cost_model.base_memory_mib)),
            )
            free_violation = (
                free_memory is not None
                and transient_mib + self.safety_mib > free_memory
            )
            if not (
                frame_violation
                or memory_violation
                or free_violation
                or transient_limit_violation
            ):
                break
            if free_violation:
                decision = "free_memory"
            elif memory_violation:
                decision = "memory_limit"
            elif transient_limit_violation:
                decision = "shared_memory_limit"
            else:
                decision = "frame_budget"
            candidate.pop()
        estimate = self.cost_model.estimate(candidate)
        if len(candidate) == 1:
            transient_mib = max(
                0,
                estimate.predicted_peak_mib
                - int(round(self.cost_model.base_memory_mib)),
            )
            if (
                estimate.total_frames > self.frame_budget
                or (
                    self.memory_limit_mib is not None
                    and estimate.predicted_peak_mib + self.safety_mib
                    > self.memory_limit_mib
                )
                or (
                    self.transient_memory_limit_mib is not None
                    and estimate.predicted_incremental_mib
                    > self.transient_memory_limit_mib
                )
            ):
                decision = "over_budget" if decision != "shared_memory_limit" else decision
            elif (
                free_memory is not None
                and transient_mib + self.safety_mib > free_memory
            ):
                decision = "over_free_memory"
        return candidate, decision

    def plan(self) -> list[BatchDecision]:
        """Partition every segment once using one initial resource snapshot.

        The returned decisions are the execution plan. Later cost-model
        observations are intentionally excluded from planning so that batch
        membership and ordering cannot change while work is in flight.
        """

        if self._plan is not None:
            return list(self._plan)
        self.planned_free_memory_mib = (
            self.free_memory_provider()
            if self.free_memory_provider is not None
            else None
        )
        remaining = list(self._remaining)
        planned: list[BatchDecision] = []
        while remaining:
            candidate, decision = self._candidate(
                remaining,
                self.planned_free_memory_mib,
            )
            estimate = self.cost_model.estimate(candidate)
            selected = tuple(candidate)
            planned.append(
                BatchDecision(
                    selected,
                    estimate,
                    decision,
                    self.planned_free_memory_mib,
                )
            )
            del remaining[: len(candidate)]
        self._plan = planned
        return list(planned)

    def next_batch(self) -> BatchDecision | None:
        if self._plan is None:
            self.plan()
        assert self._plan is not None
        if self._plan_cursor >= len(self._plan):
            return None
        result = self._plan[self._plan_cursor]
        self._plan_cursor += 1
        del self._remaining[: len(result.items)]
        self.decisions.append(result)
        return result

    def observe(
        self,
        decision: BatchDecision,
        actual_seconds: float,
        actual_peak_mib: int | None,
    ) -> None:
        observation = {
            "item_count": len(decision.items),
            "indices": [item.index for item in decision.items],
            "actual_seconds": round(float(actual_seconds), 6),
            "actual_peak_mib": actual_peak_mib,
        }
        self._observations.append(observation)
        if (
            actual_peak_mib is not None
            and int(actual_peak_mib) > decision.estimate.predicted_peak_mib
        ):
            self._prediction_overruns.append(
                {
                    **observation,
                    "predicted_peak_mib": decision.estimate.predicted_peak_mib,
                    "predicted_incremental_mib": decision.estimate.predicted_incremental_mib,
                }
            )
        self.cost_model.observe(
            decision.estimate,
            actual_seconds,
            actual_peak_mib,
        )

    def summary(self) -> dict[str, object]:
        return {
            "batches": len(self.decisions),
            "items": sum(len(item.items) for item in self.decisions),
            "frame_budget": self.frame_budget,
            "memory_limit_mib": self.memory_limit_mib,
            "safety_mib": self.safety_mib,
            "transient_memory_limit_mib": self.transient_memory_limit_mib,
            "planned_free_memory_mib": self.planned_free_memory_mib,
            "planned_batches": len(self._plan or []),
            "planned_items": sum(len(item.items) for item in (self._plan or [])),
            "model": {
                "base_memory_mib": self.cost_model.base_memory_mib,
                "memory_per_padded_frame_mib": self.cost_model.memory_per_padded_frame_mib,
                "base_seconds": self.cost_model.base_seconds,
                "seconds_per_padded_frame": self.cost_model.seconds_per_padded_frame,
                "observations": self.cost_model.observations,
            },
            "decisions": [
                {
                    "item_count": len(decision.items),
                    "indices": [item.index for item in decision.items],
                    "total_frames": decision.estimate.total_frames,
                    "padded_frames": decision.estimate.padded_frames,
                    "predicted_seconds": round(
                        decision.estimate.predicted_seconds, 6
                    ),
                    "predicted_peak_mib": decision.estimate.predicted_peak_mib,
                    "predicted_incremental_mib": decision.estimate.predicted_incremental_mib,
                    "decision": decision.decision,
                    "free_memory_mib": decision.free_memory_mib,
                }
                for decision in self.decisions
            ],
            "plan": [
                {
                    "item_count": len(decision.items),
                    "indices": [item.index for item in decision.items],
                    "total_frames": decision.estimate.total_frames,
                    "padded_frames": decision.estimate.padded_frames,
                    "predicted_seconds": round(
                        decision.estimate.predicted_seconds, 6
                    ),
                    "predicted_peak_mib": decision.estimate.predicted_peak_mib,
                    "decision": decision.decision,
                    "free_memory_mib": decision.free_memory_mib,
                }
                for decision in (self._plan or [])
            ],
            "observations": list(self._observations),
            "prediction_overruns": list(self._prediction_overruns),
        }

from __future__ import annotations

from funasr_transcriber.firered_scheduler import (
    AdaptiveBatchScheduler,
    BatchCostModel,
)


def test_scheduler_uses_frame_and_memory_limits_and_preserves_duration_order():
    scheduler = AdaptiveBatchScheduler(
        [(0.0, 1.0), (1.0, 5.0), (5.0, 6.0)],
        max_batch_size=3,
        frame_budget=500,
        memory_limit_mib=6_000,
        safety_mib=100,
        cost_model=BatchCostModel(
            base_memory_mib=5_000,
            memory_per_padded_frame_mib=0.5,
        ),
        free_memory_provider=lambda: 20_000,
    )

    first = scheduler.next_batch()
    second = scheduler.next_batch()

    assert [item.index for item in first.items] == [0, 2]
    assert [item.index for item in second.items] == [1]
    assert first.estimate.predicted_peak_mib == 5_100
    assert first.decision == "frame_budget"


def test_scheduler_shrinks_when_current_free_memory_is_low():
    scheduler = AdaptiveBatchScheduler(
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
        max_batch_size=3,
        frame_budget=1_000,
        memory_limit_mib=10_000,
        safety_mib=200,
        cost_model=BatchCostModel(
            base_memory_mib=5_000,
            memory_per_padded_frame_mib=1.0,
        ),
        free_memory_provider=lambda: 250,
    )

    batch = scheduler.next_batch()

    assert len(batch.items) == 1
    assert batch.decision == "over_free_memory"


def test_cost_model_updates_from_observed_batch():
    model = BatchCostModel(
        base_memory_mib=5_000,
        memory_per_padded_frame_mib=0.1,
        base_seconds=0.1,
        seconds_per_padded_frame=0.001,
    )
    estimate = model.estimate([(0, 2), (2, 4)])

    model.observe(
        estimate,
        actual_seconds=1.1,
        actual_peak_mib=5_800,
    )

    assert model.memory_per_padded_frame_mib > 0.1
    assert model.seconds_per_padded_frame > 0.001


def test_scheduler_processes_single_over_budget_item_and_marks_it():
    scheduler = AdaptiveBatchScheduler(
        [(0.0, 20.0)],
        max_batch_size=8,
        frame_budget=100,
        memory_limit_mib=5_100,
        safety_mib=100,
    )

    batch = scheduler.next_batch()

    assert batch is not None
    assert batch.decision == "over_budget"


def test_scheduler_precomputes_full_plan_before_execution():
    free_values = iter((8_000, 100))
    scheduler = AdaptiveBatchScheduler(
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
        max_batch_size=2,
        frame_budget=1_000,
        free_memory_provider=lambda: next(free_values),
    )

    plan = scheduler.plan()
    first = scheduler.next_batch()
    scheduler.observe(first, actual_seconds=99.0, actual_peak_mib=9_999)
    second = scheduler.next_batch()

    assert [[item.index for item in batch.items] for batch in plan] == [
        [0, 1],
        [2],
    ]
    assert [item.index for item in first.items] == [0, 1]
    assert [item.index for item in second.items] == [2]
    assert scheduler.planned_free_memory_mib == 8_000


def test_scheduler_reads_free_memory_once_for_the_complete_plan():
    calls = 0

    def free_memory():
        nonlocal calls
        calls += 1
        return 8_000

    scheduler = AdaptiveBatchScheduler(
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
        max_batch_size=2,
        frame_budget=1_000,
        free_memory_provider=free_memory,
    )

    scheduler.plan()
    while scheduler.next_batch() is not None:
        pass

    assert calls == 1


def test_observe_does_not_replan_remaining_batches():
    scheduler = AdaptiveBatchScheduler(
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
        max_batch_size=2,
        frame_budget=1_000,
        cost_model=BatchCostModel(
            base_memory_mib=5_000,
            memory_per_padded_frame_mib=0.1,
        ),
    )

    plan = scheduler.plan()
    first = scheduler.next_batch()
    scheduler.observe(first, actual_seconds=99.0, actual_peak_mib=50_000)
    remaining = scheduler.next_batch()

    assert [item.index for item in remaining.items] == [
        item.index for item in plan[1].items
    ]

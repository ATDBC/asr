from __future__ import annotations

import pytest

from funasr_transcriber.firered_scheduler import (
    BatchCostModel,
    MemoryCapacityError,
    MemoryFirstBatchPacker,
)


def test_memory_first_packer_uses_budget_not_fixed_batch_size():
    packer = MemoryFirstBatchPacker(
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
        shared_budget_mib=1000,
        worker_resident_mib=(3000, 3200),
        cost_model=BatchCostModel(base_transient_mib=20, memory_per_padded_frame_mib=0.1),
    )

    plan = packer.plan()

    assert [len(batch.items) for batch in plan] == [4]
    assert [item.index for item in plan[0].items] == [0, 1, 2, 3]
    assert plan[0].estimate.predicted_incremental_mib <= 100


def test_memory_first_packer_cardinality_changes_with_segment_duration():
    segments = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    short_budget = MemoryFirstBatchPacker(
        segments,
        shared_budget_mib=200,
        worker_resident_mib=(3000, 3200),
        cost_model=BatchCostModel(base_transient_mib=20, memory_per_padded_frame_mib=0.1),
    ).plan()
    tight_budget = MemoryFirstBatchPacker(
        segments,
        shared_budget_mib=100,
        worker_resident_mib=(3000, 3200),
        cost_model=BatchCostModel(base_transient_mib=20, memory_per_padded_frame_mib=0.1),
    ).plan()

    assert len(short_budget) < len(tight_budget)
    assert all(batch.items for batch in short_budget + tight_budget)


def test_memory_first_packer_preserves_all_original_indices():
    plan = MemoryFirstBatchPacker(
        [(0.0, 2.0), (2.0, 2.5), (2.5, 8.0)],
        shared_budget_mib=1000,
        worker_resident_mib=(3000, 3200),
    ).plan()

    assert sorted(item.index for batch in plan for item in batch.items) == [0, 1, 2]


def test_memory_first_packer_rejects_singleton_that_cannot_fit():
    packer = MemoryFirstBatchPacker(
        [(0.0, 100.0)],
        shared_budget_mib=10,
        worker_resident_mib=(3000, 3200),
    )

    with pytest.raises(MemoryCapacityError, match="singleton"):
        packer.plan()


def test_cost_model_exposes_total_and_max_frame_features():
    model = BatchCostModel(
        base_transient_mib=10,
        memory_per_padded_frame_mib=0.1,
        seconds_per_padded_frame=0.001,
        seconds_per_total_frame=0.002,
        seconds_per_max_frame=0.01,
    )
    estimate = model.estimate([(0.0, 1.0), (1.0, 5.0)])

    assert estimate.total_frames == 500
    assert estimate.padded_frames == 800
    assert estimate.max_frames == 400
    assert estimate.predicted_seconds > 1.0


def test_cost_model_observe_updates_from_reserved_increment():
    model = BatchCostModel(
        base_transient_mib=10,
        memory_per_padded_frame_mib=0.1,
        seconds_per_padded_frame=0.001,
        seconds_per_total_frame=0.001,
        seconds_per_max_frame=0.001,
    )
    estimate = model.estimate([(0.0, 1.0), (1.0, 2.0)])
    before = model.estimate([(0.0, 1.0), (1.0, 2.0)])
    model.observe(estimate, actual_seconds=0.5, actual_incremental_reserved_mib=5)
    after = model.estimate([(0.0, 1.0), (1.0, 2.0)])

    assert model.observations == 1
    assert after.predicted_seconds != before.predicted_seconds
    assert after.predicted_incremental_mib != before.predicted_incremental_mib

from __future__ import annotations

import pytest

from funasr_transcriber.firered_parallel import (
    GpuMemorySnapshot,
    ParallelFireRedWorkerClient,
    build_global_plan,
    derive_transient_budget_mib,
    partition_batches,
    required_concurrent_increment_mib,
    serialize_batch_plan,
)
from funasr_transcriber.schemas import AsrResult, RecognizedSentence, TimedToken


def test_shared_transient_budget_subtracts_configured_margin():
    snapshot = GpuMemorySnapshot(
        total_mib=24_576,
        free_mib=13_444,
        process_mib={101: 3_126, 102: 2_704},
    )

    assert derive_transient_budget_mib(snapshot, safety_margin_mib=1_536) == 11_908


def test_required_concurrent_increment_uses_each_lanes_largest_batch():
    plan = build_global_plan(
        [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0), (25.0, 30.0)],
        batch_size=1,
        frame_budget=10_000,
    )
    assignments = partition_batches(plan, worker_count=2)

    expected = sum(
        max(batch.estimate.predicted_incremental_mib for batch in lane)
        for lane in assignments
    )
    assert required_concurrent_increment_mib(assignments) == expected


def test_global_plan_shrinks_batch_size_until_two_lane_peak_fits():
    plan = build_global_plan(
        [(float(index), float(index + 20)) for index in range(8)],
        batch_size=8,
        frame_budget=10_000,
        transient_memory_budget_mib=130,
        worker_count=2,
    )

    assert all(len(batch.items) < 8 for batch in plan)
    assert required_concurrent_increment_mib(
        partition_batches(plan, worker_count=2)
    ) <= 130


def test_partition_assigns_complete_batches_to_the_least_loaded_worker():
    plan = build_global_plan(
        [(0.0, 10.0), (10.0, 9.0 + 10.0), (20.0, 25.0), (25.0, 27.0)],
        batch_size=1,
        frame_budget=10_000,
    )

    workers = partition_batches(plan, worker_count=2)

    assert all(batch.items for batches in workers for batch in batches)
    assert sorted(
        item.index
        for batches in workers
        for batch in batches
        for item in batch.items
    ) == [0, 1, 2, 3]
    loads = [
        sum(batch.estimate.predicted_seconds for batch in batches)
        for batches in workers
    ]
    # With four indivisible batches, this is the best deterministic LPT split
    # for the deliberately uneven durations.
    assert max(loads) - min(loads) <= 0.81


def test_partition_does_not_split_a_batch_and_is_deterministic():
    plan = build_global_plan(
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
        batch_size=2,
        frame_budget=10_000,
    )

    first = partition_batches(plan, worker_count=2)
    second = partition_batches(plan, worker_count=2)

    def signature(result):
        return [
            [[item.index for item in batch.items] for batch in batches]
            for batches in result
        ]

    assert signature(first) == signature(second)
    assert signature(first) == [[[0, 1]], [[2, 3]]]


def test_serialized_plan_contains_original_indices_and_milliseconds():
    plan = build_global_plan(
        [(1.25, 2.5), (0.0, 1.0)],
        batch_size=2,
        frame_budget=10_000,
    )

    assert serialize_batch_plan(plan) == [
        [
            {"index": 1, "start_ms": 0, "end_ms": 1000},
            {"index": 0, "start_ms": 1250, "end_ms": 2500},
        ]
    ]


class _FakeWorker:
    def __init__(self, results=None, pid=None):
        self.results = results or []
        self._pid = pid
        self.started = 0
        self.closed = 0
        self.calls = []
        self.last_response = {}

    @property
    def pid(self):
        return self._pid

    def start(self):
        self.started += 1
        return self

    def close(self):
        self.closed += 1

    def status(self):
        return {"ok": True}

    def refine_vad(self, path, segments):
        self.calls.append(("vad", segments))
        return segments

    def run_aed_plan(self, path, plan, vad):
        self.calls.append(("aed", plan))
        records = []
        for batch in plan:
            for item in batch:
                records.append({
                    "index": item["index"],
                    "result": {
                        "uttid": f"job_s{item['start_ms']}_e{item['end_ms']}",
                        "text": str(item["index"]),
                        "confidence": 0.9,
                        "timestamp": [(str(item["index"]), 0.0, 0.5)],
                    },
                })
        return {
            "aed_results": records,
            "duration_ms": 4000,
            "batch_metrics": [
                {
                    "batch_index": index,
                    "actual_seconds": 0.5,
                    "peak_allocated_mib": 3000,
                    "peak_reserved_mib": 3200,
                }
                for index, _ in enumerate(plan)
            ],
        }

    def punctuate(self, path, records, duration_ms, vad):
        self.calls.append(("punc", [item["index"] for item in records]))
        return AsrResult(
            text="0123。",
            tokens=(TimedToken("0123。", 0, 2000, 0.9),),
            sentences=(RecognizedSentence("0123。", 0, 2000, 0.9),),
            vad_segments_ms=tuple(tuple(item) for item in vad),
            duration_ms=duration_ms,
        )

    def verify(self, path):
        raise AssertionError


def test_parallel_owner_uses_global_vad_two_fixed_plans_and_one_punc(tmp_path):
    first = _FakeWorker()
    second = _FakeWorker()
    owner = ParallelFireRedWorkerClient(
        first, second, batch_size=2, frame_budget=10_000
    )
    owner.start()
    result = owner.transcribe(
        tmp_path / "audio.wav",
        [[0, 1000], [1000, 2000], [2000, 3000], [3000, 4000]],
    )
    owner.close()

    assert result.text == "0123。"
    assert first.started == second.started == 1
    assert first.closed == second.closed == 1
    assert first.calls[0][0] == "vad"
    assert owner.last_response["metrics"]["parallel"]["warmup_excluded"] is True
    assert owner.last_response["metrics"]["parallel"]["aed_seconds"] >= 0
    assert owner.last_response["metrics"]["parallel"]["punc_seconds"] >= 0
    assert [call[0] for call in first.calls].count("punc") == 1
    assert [call[0] for call in second.calls].count("punc") == 0
    assert sorted(
        item["index"]
        for worker in (first, second)
        for kind, payload in worker.calls
        if kind == "aed"
        for batch in payload
        for item in batch
    ) == [0, 1, 2, 3]


def test_parallel_owner_uses_post_warmup_budget_and_serial_fallback(tmp_path):
    first = _FakeWorker(pid=101)
    second = _FakeWorker(pid=102)
    snapshots = iter(
        (
            GpuMemorySnapshot(24_576, 1_800, {101: 3_126, 102: 2_704}),
            GpuMemorySnapshot(24_576, 1_542, {101: 3_126, 102: 2_704}),
        )
    )
    owner = ParallelFireRedWorkerClient(
        first,
        second,
        batch_size=2,
        frame_budget=10_000,
        gpu_safety_margin_mib=1_536,
        gpu_memory_provider=lambda pids: next(snapshots),
    )

    owner.transcribe(
        tmp_path / "audio.wav",
        [[0, 1000], [1000, 2000], [2000, 3000], [3000, 4000]],
    )

    parallel = owner.last_response["metrics"]["parallel"]
    assert parallel["memory"]["planned_free_mib"] == 1_800
    assert parallel["memory"]["effective_transient_budget_mib"] == 264
    assert parallel["dispatch_mode"] == "serial"
    assert parallel["prediction_accuracy"]["batches"] == 2


def test_parallel_owner_fails_before_aed_when_one_batch_cannot_fit(tmp_path):
    first = _FakeWorker(pid=101)
    second = _FakeWorker(pid=102)
    snapshots = iter(
        (
            GpuMemorySnapshot(24_576, 1_800, {101: 3_126, 102: 2_704}),
            GpuMemorySnapshot(24_576, 1_540, {101: 3_126, 102: 2_704}),
        )
    )
    owner = ParallelFireRedWorkerClient(
        first,
        second,
        batch_size=2,
        frame_budget=10_000,
        gpu_safety_margin_mib=1_536,
        gpu_memory_provider=lambda pids: next(snapshots),
    )

    with pytest.raises(RuntimeError, match="cannot fit even one planned batch"):
        owner.transcribe(
            tmp_path / "audio.wav",
            [[0, 1000], [1000, 2000]],
        )

    assert not any(call[0] == "aed" for call in first.calls + second.calls)

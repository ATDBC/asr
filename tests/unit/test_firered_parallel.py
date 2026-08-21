from __future__ import annotations

from funasr_transcriber.firered_parallel import (
    ParallelFireRedWorkerClient,
    build_global_plan,
    partition_batches,
    serialize_batch_plan,
)
from funasr_transcriber.schemas import AsrResult, RecognizedSentence, TimedToken


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
    def __init__(self, results=None):
        self.results = results or []
        self.started = 0
        self.closed = 0
        self.calls = []
        self.last_response = {}

    @property
    def pid(self):
        return None

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
        return {"aed_results": records, "duration_ms": 4000}

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

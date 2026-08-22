from pathlib import Path

from funasr_transcriber.diagnostic_traces import (
    compact_scheduler_summary,
    write_deep_traces,
)


def test_compact_scheduler_summary_removes_large_plan_fields():
    summary = {
        "planned_batches": 2,
        "worker_plans": [[{"index": 1}]],
        "parallel": {
            "prediction_accuracy": {
                "batches": 1,
                "records": [{"index": 1}],
            }
        },
    }

    compact = compact_scheduler_summary(summary)

    assert compact == {
        "planned_batches": 2,
        "parallel": {
            "prediction_accuracy": {"batches": 1}
        },
    }


def test_write_deep_traces_writes_scheduler_and_batch_events(tmp_path: Path):
    scheduler = {
        "planned_batches": 1,
        "worker_plans": [[{"index": 0}]],
        "parallel": {
            "prediction_accuracy": {
                "batches": 1,
                "records": [{"batch_index": 0, "actual_seconds": 1.2}],
            }
        },
    }

    paths = write_deep_traces(tmp_path, scheduler)

    assert {path.name for path in paths} == {"scheduler.json", "batch-events.jsonl"}
    assert '"planned_batches": 1' in (tmp_path / "scheduler.json").read_text()
    assert '"batch_index": 0' in (tmp_path / "batch-events.jsonl").read_text()

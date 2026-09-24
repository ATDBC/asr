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

def test_nemotron_crop_plan_is_deep_only_with_lane_events(tmp_path: Path):
    import json

    scheduler = {
        "crop_plans": [
            {
                "crop_index": 0,
                "speaker_id": 1,
                "path": "/job/crop.wav",
                "batches": [[{"index": 0}]],
            }
        ],
        "worker_crop_indexes": [[0]],
    }

    assert "crop_plans" not in compact_scheduler_summary(scheduler)
    write_deep_traces(tmp_path, scheduler)
    events = [
        json.loads(line)
        for line in (tmp_path / "batch-events.jsonl").read_text().splitlines()
    ]
    assert events == [
        {
            "type": "speaker_crop",
            "crop_index": 0,
            "speaker_id": 1,
            "path": "/job/crop.wav",
            "worker_index": 0,
            "batches": [[{"index": 0}]],
        }
    ]

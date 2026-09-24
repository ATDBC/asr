"""Optional deep diagnostic trace publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


_LARGE_KEYS = {"worker_plans", "records", "crop_plans"}


def compact_scheduler_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove full plans and per-batch records from the normal summary."""

    def compact(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): compact(value)
                for key, value in item.items()
                if str(key) not in _LARGE_KEYS
            }
        if isinstance(item, list):
            return [compact(element) for element in item]
        if isinstance(item, tuple):
            return [compact(element) for element in item]
        return item

    return compact(value)


def _atomic_write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        with temporary.open("r+", encoding="utf-8", newline="\n") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_deep_traces(
    directory: Path,
    scheduler: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write complete scheduler JSON and prediction records as JSONL."""

    target = Path(directory)
    scheduler_path = _atomic_write(
        target / "scheduler.json",
        json.dumps(dict(scheduler), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    records: list[Mapping[str, Any]] = []
    parallel = scheduler.get("parallel")
    if isinstance(parallel, Mapping):
        prediction = parallel.get("prediction_accuracy")
        if isinstance(prediction, Mapping):
            raw_records = prediction.get("records")
            if isinstance(raw_records, list):
                records = [item for item in raw_records if isinstance(item, Mapping)]
    crop_plans = scheduler.get("crop_plans")
    worker_crop_indexes = scheduler.get("worker_crop_indexes")
    if isinstance(crop_plans, list) and isinstance(worker_crop_indexes, list):
        by_index = {
            item.get("crop_index"): item
            for item in crop_plans
            if isinstance(item, Mapping)
        }
        for worker_index, indexes in enumerate(worker_crop_indexes):
            if not isinstance(indexes, list):
                continue
            for crop_index in indexes:
                crop = by_index.get(crop_index)
                if crop is not None:
                    records.append({
                        "type": "speaker_crop",
                        "worker_index": worker_index,
                        **crop,
                    })
    events = "".join(
        json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    events_path = _atomic_write(target / "batch-events.jsonl", events)
    return scheduler_path, events_path

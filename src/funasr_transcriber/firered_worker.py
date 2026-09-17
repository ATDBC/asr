"""FireRed AED/Punc JSON Lines worker with a hosted VAD helper.

The worker is the AED/Punc functional owner. FireRedVAD is hosted in the same
`.venv-asr` process only to refine long speaker-separation intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, IO, Protocol
from types import SimpleNamespace

from .firered_contract import normalize_official_result
from .firered_scheduler import ScheduledItem
from .weighted_aed_calibration import (
    DEFAULT_WEIGHTED_AED_WEIGHTS,
    WeightedAedWeights,
)
from .punc_scheduler import (
    DEFAULT_PUNC_MAX_ITEMS,
    DEFAULT_PUNC_TOKEN_BUDGET,
    assemble_punc_items,
    compact_aed_record,
    pack_punc_records,
)


LOGGER = logging.getLogger(__name__)

SCRIPTED_VAD_THRESHOLD_SECONDS = 10.0
PUNC_INTERNAL_BATCH_SIZE = DEFAULT_PUNC_MAX_ITEMS
PUNC_TOKEN_BUDGET = DEFAULT_PUNC_TOKEN_BUDGET
AED_BATCH_DURATION_LIMIT_MS = 200_000
AED_BATCH_SIZE = 32
# Train-only fit from 726 repeat-median solo server batches (GPU 0, BF16).
AED_LONGEST_WEIGHT = DEFAULT_WEIGHTED_AED_WEIGHTS.longest_weight
AED_PADDED_WEIGHT = DEFAULT_WEIGHTED_AED_WEIGHTS.padded_weight


@dataclass(frozen=True, slots=True)
class AudioInfo:
    sample_rate: int
    channels: int
    subtype: str


def _item_duration_ms(item: ScheduledItem) -> int:
    return int(round(item.duration_s * 1000))


def _batch_duration_ms(batch: list[ScheduledItem]) -> int:
    return sum(_item_duration_ms(item) for item in batch)


def _batch_cost(
    batch: list[ScheduledItem],
    weights: WeightedAedWeights = DEFAULT_WEIGHTED_AED_WEIGHTS,
) -> float:
    longest = max(_item_duration_ms(item) for item in batch) / 1000
    return longest * (
        weights.longest_weight + weights.padded_weight * len(batch)
    )


def plan_aed_batches(
    segments: list[tuple[float, float]],
    *,
    weights: WeightedAedWeights = DEFAULT_WEIGHTED_AED_WEIGHTS,
) -> list[list[ScheduledItem]]:
    """Port the Mac duration-capacity planner with server-fitted weights."""

    items = sorted(
        (
            ScheduledItem(index, float(start), float(end))
            for index, (start, end) in enumerate(segments)
        ),
        key=lambda item: (-_item_duration_ms(item), item.index),
    )
    batches: list[list[ScheduledItem]] = []
    for item in items:
        duration_ms = _item_duration_ms(item)
        if duration_ms > AED_BATCH_DURATION_LIMIT_MS:
            raise ValueError("AED segment exceeds batch duration limit")
        candidates = [
            (
                AED_BATCH_DURATION_LIMIT_MS
                - _batch_duration_ms(batch)
                - duration_ms,
                index,
            )
            for index, batch in enumerate(batches)
            if len(batch) < AED_BATCH_SIZE
            and _batch_duration_ms(batch) + duration_ms
            <= AED_BATCH_DURATION_LIMIT_MS
        ]
        if candidates:
            batches[min(candidates)[1]].append(item)
        else:
            batches.append([item])

    # ponytail: O(n^3) best-single-move search; replace only if planning is visible.
    while True:
        best_delta = 0.0
        best_move = None
        for source_index, source in enumerate(batches):
            if len(source) <= 1:
                continue
            for item_index, item in enumerate(source):
                for target_index, target in enumerate(batches):
                    if source_index == target_index or len(target) >= AED_BATCH_SIZE:
                        continue
                    if (
                        _batch_duration_ms(target) + _item_duration_ms(item)
                        > AED_BATCH_DURATION_LIMIT_MS
                    ):
                        continue
                    new_source = source[:item_index] + source[item_index + 1 :]
                    new_target = [*target, item]
                    delta = (
                        _batch_cost(new_source, weights)
                        + _batch_cost(new_target, weights)
                        - _batch_cost(source, weights)
                        - _batch_cost(target, weights)
                    )
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_move = (
                            source_index,
                            target_index,
                            new_source,
                            new_target,
                        )
        if best_move is None:
            break
        source_index, target_index, new_source, new_target = best_move
        batches[source_index] = new_source
        batches[target_index] = new_target

    return [
        batch
        for _, batch in sorted(
            enumerate(batches),
            key=lambda pair: (
                max(_item_duration_ms(item) for item in pair[1]),
                _batch_duration_ms(pair[1]),
                pair[0],
            ),
        )
    ]


def _aed_plan_summary(
    plan: list[list[ScheduledItem]],
    *,
    weights: WeightedAedWeights = DEFAULT_WEIGHTED_AED_WEIGHTS,
    source: str = "builtin",
    artifact_checksum: str | None = None,
    warnings: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "planner": "weighted_duration_single_worker",
        "duration_limit_seconds": AED_BATCH_DURATION_LIMIT_MS / 1000,
        "max_items": AED_BATCH_SIZE,
        "longest_weight": weights.longest_weight,
        "padded_weight": weights.padded_weight,
        "weight_source": source,
        "weight_artifact_checksum": artifact_checksum,
        "weight_warnings": list(warnings),
        "planned_batches": len(plan),
        "planned_items": sum(len(batch) for batch in plan),
        "plan": [
            {
                "item_count": len(batch),
                "indices": [item.index for item in batch],
                "total_duration_seconds": _batch_duration_ms(batch) / 1000,
                "max_duration_seconds": (
                    max(_item_duration_ms(item) for item in batch) / 1000
                ),
                "predicted_seconds": round(_batch_cost(batch, weights), 6),
            }
            for batch in plan
        ],
    }


def _synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _cuda_peak_memory_mib() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except ImportError:
        pass
    return None


def _cuda_memory_allocated_mib() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except ImportError:
        pass
    return None


def _cuda_peak_reserved_mib() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_reserved() / (1024 * 1024))
    except ImportError:
        pass
    return None


def _cuda_reserved_memory_mib() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_reserved() / (1024 * 1024))
    except ImportError:
        pass
    return None


def _cuda_free_memory_mib() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            return int(free_bytes / (1024 * 1024))
    except ImportError:
        pass
    return None


def _cleanup_cuda_cache() -> dict[str, object]:
    """Release this worker's unused PyTorch CUDA allocator cache."""

    import torch

    allocated_before = _cuda_memory_allocated_mib()
    reserved_before = _cuda_reserved_memory_mib()
    collected = gc.collect()
    _synchronize_cuda()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _synchronize_cuda()
    return {
        "gc_collected": int(collected),
        "allocated_before_mib": allocated_before,
        "reserved_before_mib": reserved_before,
        "allocated_after_mib": _cuda_memory_allocated_mib(),
        "reserved_after_mib": _cuda_reserved_memory_mib(),
    }


def _scheduled_process(
    system: FireRedSystem,
    wav_path: str,
    uttid: str,
    vad_segments: list[tuple[float, float]],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Run the pinned modules with duration-aware AED batches.

    This mirrors the small orchestration layer in the pinned FireRed system,
    but keeps the project-owned execution scheduler outside the ignored model
    checkout. All result lists are restored to the caller's VAD order before
    punctuation and response formatting.
    """

    import soundfile as sf

    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    duration_s = wav_np.shape[0] / sample_rate
    config = system.config
    records: list[tuple[int, dict[str, object], object]] = []
    plan = plan_aed_batches(vad_segments)
    for batch in plan:
        batch_uttid: list[str] = []
        batch_wav: list[tuple[int, object]] = []
        for item in batch:
            start = int(item.start_s * sample_rate)
            end = int(item.end_s * sample_rate)
            batch_uttid.append(
                f"{uttid}_s{int(item.start_s * 1000)}_e{int(item.end_s * 1000)}"
            )
            batch_wav.append((sample_rate, wav_np[start:end]))
        _reset_cuda_peak_memory()
        asr_started = clock()
        asr_results = system.asr.transcribe(batch_uttid, batch_wav)
        _synchronize_cuda()
        lid_results = (
            system.lid.process(batch_uttid, batch_wav)
            if getattr(config, "enable_lid", False)
            else [None] * len(asr_results)
        )
        item_by_uttid = {
            uttid: item for uttid, item in zip(batch_uttid, batch)
        }
        lid_by_uttid = {
            str(result.get("uttid")): result
            for result in lid_results
            if isinstance(result, dict) and result.get("uttid") is not None
        }
        for asr_result in asr_results:
            result_uttid = str(asr_result.get("uttid") or "")
            item = item_by_uttid.get(result_uttid)
            if item is None:
                raise ValueError(
                    "FireRed AED returned an unknown segment uttid: "
                    f"{result_uttid}"
                )
            lid_result = lid_by_uttid.get(result_uttid)
            text = str(asr_result.get("text", "")).strip()
            if not text or re.search(r"(<blank>)|(<sil>)", text):
                continue
            records.append((item.index, asr_result, lid_result))

    records.sort(key=lambda item: item[0])
    asr_results = [item[1] for item in records]
    lid_results = [item[2] for item in records]
    punc_results: list[dict[str, object]]
    if getattr(config, "enable_punc", False):
        punc_results = []
        punc_batch: list[tuple[dict[str, object], str]] = []
        for index, asr_result in enumerate(asr_results):
            timestamp = asr_result.get("timestamp", [])
            punc_batch.append((asr_result, timestamp))
            if len(punc_batch) < PUNC_INTERNAL_BATCH_SIZE and index != len(asr_results) - 1:
                continue
            punc_results.extend(
                system.punc.process_with_timestamp(
                    [item[1] for item in punc_batch],
                    [str(item[0]["uttid"]) for item in punc_batch],
                )
            )
            punc_batch = []
    else:
        punc_results = [item for item in asr_results]

    sentences: list[dict[str, object]] = []
    words: list[dict[str, object]] = []
    for asr_result, punc_result, lid_result in zip(
        asr_results, punc_results, lid_results
    ):
        start_ms, end_ms = str(asr_result["uttid"]).split("_")[-2:]
        start_ms = int(start_ms[1:])
        end_ms = int(end_ms[1:])
        if getattr(config.asr_config, "return_timestamp", False):
            for index, punc_sentence in enumerate(
                punc_result["punc_sentences"]
            ):
                start = start_ms + int(float(punc_sentence["start_s"]) * 1000)
                end = start_ms + int(float(punc_sentence["end_s"]) * 1000)
                if index == 0:
                    start = start_ms
                if index == len(punc_result["punc_sentences"]) - 1:
                    end = end_ms
                sentence = {
                    "start_ms": start,
                    "end_ms": end,
                    "text": punc_sentence["punc_text"],
                    "asr_confidence": asr_result["confidence"],
                    "lang": None,
                    "lang_confidence": 0,
                }
                if lid_result:
                    sentence["lang"] = lid_result["lang"]
                    sentence["lang_confidence"] = lid_result["confidence"]
                sentences.append(sentence)
        else:
            text = (
                punc_result["punc_text"]
                if getattr(config, "enable_punc", False)
                else asr_result["text"]
            )
            sentences.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "asr_confidence": asr_result["confidence"],
                    "lang": None,
                    "lang_confidence": 0,
                }
            )
        for word, start, end in asr_result.get("timestamp", []):
            words.append(
                {
                    "start_ms": int(float(start) * 1000 + start_ms),
                    "end_ms": int(float(end) * 1000 + start_ms),
                    "text": word,
                }
            )

    text = "".join(str(sentence["text"]) for sentence in sentences)
    text = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", text)
    return {
        "uttid": uttid,
        "text": text,
        "sentences": sentences,
        "vad_segments_ms": [
            (int(start * 1000), int(end * 1000))
            for start, end in vad_segments
        ],
        "dur_s": duration_s,
        "words": words,
        "wav_path": wav_path,
        "_scheduler": _aed_plan_summary(plan),
    }


def _explicit_aed_process(
    system: FireRedSystem,
    wav_path: str,
    uttid: str,
    batch_plan: list[list[dict[str, object]]],
    vad_segments: list[tuple[float, float]],
    clock: Callable[[], float] = time.perf_counter,
    on_batch: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Execute a caller-owned immutable batch plan without VAD or Punc."""

    import soundfile as sf

    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    duration_s = wav_np.shape[0] / sample_rate
    records: list[dict[str, object]] = []
    batch_metrics: list[dict[str, object]] = []
    resident_allocated_mib = _cuda_memory_allocated_mib()
    resident_reserved_mib = _cuda_reserved_memory_mib()
    for batch_index, raw_batch in enumerate(batch_plan):
        batch_uttid: list[str] = []
        batch_wav: list[tuple[int, object]] = []
        batch_items: list[dict[str, object]] = []
        for raw_item in raw_batch:
            if not isinstance(raw_item, dict):
                raise ValueError("batch_plan items must be objects")
            index = int(raw_item["index"])
            start_ms = int(raw_item["start_ms"])
            end_ms = int(raw_item["end_ms"])
            if start_ms < 0 or end_ms <= start_ms:
                raise ValueError("batch_plan item bounds are invalid")
            start = int(start_ms * sample_rate / 1000)
            end = int(end_ms * sample_rate / 1000)
            item_uttid = f"{uttid}_s{start_ms}_e{end_ms}"
            batch_uttid.append(item_uttid)
            batch_wav.append((sample_rate, wav_np[start:end]))
            batch_items.append({"index": index, "uttid": item_uttid})
        if not batch_uttid:
            continue
        _reset_cuda_peak_memory()
        started = clock()
        results = system.asr.transcribe(batch_uttid, batch_wav)
        _synchronize_cuda()
        actual_seconds = clock() - started
        batch_metrics.append({
            "batch_index": batch_index,
            "item_count": len(batch_items),
            "indices": [int(item["index"]) for item in batch_items],
            "actual_seconds": actual_seconds,
            "peak_allocated_mib": _cuda_peak_memory_mib(),
            "peak_reserved_mib": _cuda_peak_reserved_mib(),
            "resident_allocated_mib": resident_allocated_mib,
            "resident_reserved_mib": resident_reserved_mib,
        })
        returned = {str(result.get("uttid")): result for result in results}
        batch_records: list[dict[str, object]] = []
        for item in batch_items:
            result = returned.get(str(item["uttid"]))
            if result is None:
                continue
            text = str(result.get("text", "")).strip()
            if not text or re.search(r"(<blank>)|(<sil>)", text):
                continue
            record = {"index": item["index"], "result": result}
            records.append(record)
            batch_records.append(record)
        if on_batch is not None:
            on_batch({
                "batch_index": batch_index,
                "aed_results": batch_records,
                "batch_metrics": batch_metrics[-1],
            })
        LOGGER.debug("explicit AED batch completed in %.3fs", actual_seconds)

    records.sort(key=lambda item: int(item["index"]))
    return {
        "uttid": uttid,
        "duration_ms": int(round(duration_s * 1000)),
        "vad_segments_ms": [
            [int(round(start * 1000)), int(round(end * 1000))]
            for start, end in vad_segments
        ],
        "aed_results": records,
        "batch_metrics": batch_metrics,
        "resident_allocated_mib": resident_allocated_mib,
        "resident_reserved_mib": resident_reserved_mib,
    }


def _punctuate_aed_items(
    system: FireRedSystem,
    aed_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Punctuate compact AED records and retain their original indexes."""

    ordered = sorted(aed_results, key=lambda item: int(item["index"]))
    records = [compact_aed_record(item) for item in ordered]
    output: list[dict[str, object]] = []
    token_budget = int(
        getattr(system, "punc_token_budget", PUNC_TOKEN_BUDGET)
    )
    max_items = int(
        getattr(system, "punc_max_items", PUNC_INTERNAL_BATCH_SIZE)
    )
    for group in pack_punc_records(
        records,
        token_budget=token_budget,
        max_items=max_items,
    ):
        punctuated = system.punc.process_with_timestamp(
            [record.timestamp for record in group],
            [record.uttid for record in group],
        )
        for record, punc_result in zip(group, punctuated):
            output.append({
                "index": record.index,
                "uttid": record.uttid,
                "confidence": record.confidence,
                "timestamp": [list(item) for item in record.timestamp],
                "punc_sentences": punc_result["punc_sentences"],
            })
    return output


def _punctuate_aed_results(
    system: FireRedSystem,
    request_id: str,
    aed_results: list[dict[str, object]],
    duration_ms: int,
    vad_segments_ms: list[list[int]],
) -> dict[str, object]:
    """Apply one shared Punc model and produce the normal raw result shape."""

    return assemble_punc_items(
        request_id,
        _punctuate_aed_items(system, aed_results),
        duration_ms=duration_ms,
        vad_segments_ms=vad_segments_ms,
    )


def _warmup_process(system: FireRedSystem, wav_path: str) -> dict[str, object]:
    """Exercise the resident AED/Punc CUDA paths before production timing."""

    import numpy as np
    import soundfile as sf

    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    shapes = [
        ("short_single", 1, 1),
        ("short_batch", 1, 4),
        ("medium", 10, 1),
        ("long", 30, 1),
    ]
    if getattr(system, "asr", None) is not None:
        punc = getattr(system, "punc", None)
        for shape, seconds, count in shapes:
            samples = np.resize(wav_np, sample_rate * seconds)
            uttids = [f"__firered_warmup_{shape}_{index}__" for index in range(count)]
            asr_results = system.asr.transcribe(
                uttids, [(sample_rate, samples) for _ in uttids]
            )
            if punc is not None and asr_results:
                punc.process_with_timestamp(
                    [item.get("timestamp", []) for item in asr_results],
                    uttids,
                )
            _synchronize_cuda()
    elif getattr(system, "punc", None) is not None:
        system.punc.process_with_timestamp(
            [[("预热", 0.0, 0.2)]],
            ["__firered_punc_warmup__"],
        )
    _synchronize_cuda()
    return {
        "shapes": [item[0] for item in shapes]
        if getattr(system, "asr", None) is not None
        else ["punc"]
    }


class FireRedSystem(Protocol):
    def process(self, path: str, uttid: str) -> dict[str, object]:
        """Run the complete VAD, AED, and punctuation path."""


class GuardedVad:
    """Reject unsafe VAD segments before the official system reaches AED."""

    def __init__(self, vad: object):
        self._vad = vad

    def detect(self, path: str):
        result, probabilities = self._vad.detect(path)
        for start_s, end_s in result.get("timestamps", []):
            _check_segment_bounds(float(start_s), float(end_s))
        return result, probabilities

    def __getattr__(self, name: str) -> object:
        return getattr(self._vad, name)


def _check_segment_bounds(start_s: float, end_s: float) -> None:
    if end_s - start_s > 60.0:
        raise ValueError(
            "FireRedVAD segment exceeds the 60 second AED limit: "
            f"{start_s}..{end_s}"
        )


class ScriptedVad:
    """Replay speaker segments, refining long turns from one global VAD pass."""

    def __init__(
        self,
        segments_s: list[tuple[float, float]],
        fallback_vad: object | None = None,
    ):
        self._segments = list(segments_s)
        self._fallback_vad = fallback_vad

    def detect(self, path: str):
        long_segment = any(
            end - start > SCRIPTED_VAD_THRESHOLD_SECONDS
            for start, end in self._segments
        )
        if not long_segment:
            return {"timestamps": list(self._segments)}, None
        neural_segments: list[tuple[float, float]] = []
        if self._fallback_vad is not None:
            neural_result, _ = self._fallback_vad.detect(path)
            neural_segments = [
                (float(start), float(end))
                for start, end in neural_result.get("timestamps", [])
            ]
        output: list[tuple[float, float]] = []
        for start_s, end_s in self._segments:
            duration = end_s - start_s
            if duration <= SCRIPTED_VAD_THRESHOLD_SECONDS:
                output.append((start_s, end_s))
                continue
            subdivided: list[tuple[float, float]] = []
            for neural_start, neural_end in neural_segments:
                intersection_start = max(start_s, neural_start)
                intersection_end = min(end_s, neural_end)
                if intersection_end > intersection_start:
                    subdivided.append((intersection_start, intersection_end))
            if subdivided:
                output.extend(subdivided)
                continue
            cursor = start_s
            while end_s - cursor > SCRIPTED_VAD_THRESHOLD_SECONDS:
                output.append(
                    (cursor, cursor + SCRIPTED_VAD_THRESHOLD_SECONDS)
                )
                cursor += SCRIPTED_VAD_THRESHOLD_SECONDS
            output.append((cursor, end_s))
        for start_s, end_s in output:
            _check_segment_bounds(start_s, end_s)
        return {"timestamps": output}, None


def _parse_scripted_segments(raw: object) -> list[tuple[float, float]]:
    """Validate request-provided VAD segments and convert ms to seconds."""

    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("vad_segments_ms must be a non-empty list")
    segments: list[tuple[float, float]] = []
    previous_start = -1.0
    previous_end = -1.0
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("vad_segments_ms entries must contain two timestamps")
        try:
            start_s = float(item[0]) / 1000.0
            end_s = float(item[1]) / 1000.0
        except (TypeError, ValueError) as exc:
            raise ValueError("vad_segments_ms entries must be numeric") from exc
        if not start_s >= 0.0 or not end_s > start_s:
            raise ValueError("vad_segments_ms bounds are invalid")
        if start_s < previous_start or end_s < previous_end:
            raise ValueError("vad_segments_ms must be monotonic")
        segments.append((start_s, end_s))
        previous_start = start_s
        previous_end = end_s
    return segments


def read_audio_info(path: Path) -> AudioInfo:
    import soundfile as sf

    info = sf.info(str(path))
    return AudioInfo(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        subtype=str(info.subtype),
    )


def _validate_audio(
    raw_path: object,
    audio_info: Callable[[Path], AudioInfo],
) -> Path:
    path = Path(str(raw_path or ""))
    if not path.is_absolute():
        raise ValueError("audio_path must be absolute")
    if path.suffix.lower() != ".wav":
        raise ValueError("audio_path must be a WAV file")
    if not path.is_file():
        raise ValueError(f"audio_path does not exist: {path}")
    info = audio_info(path)
    if (
        info.sample_rate != 16000
        or info.channels != 1
        or info.subtype != "PCM_16"
    ):
        raise ValueError("audio must be normalized to 16 kHz mono PCM_16")
    return path


def _write_response(output: IO[str], response: dict[str, object]) -> None:
    output.write(
        json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )
    output.flush()


def _error_response(
    identifier: str | None,
    error: Exception,
) -> dict[str, object]:
    return {
        "id": identifier,
        "ok": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _no_peak_sample() -> int | None:
    return None


def _peak_gpu_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def firered_inventory(system: FireRedSystem) -> dict[str, object]:
    """Return physical inventory facts, including the hosted VAD helper.

    The ``vad`` entry describes a speaker-separation helper even though the
    model is physically loaded by this worker process.
    """

    from .resource_profile import module_inventory

    owners = (
        ("vad", "vad_model"),
        ("asr", "model"),
        ("punc", "model"),
    )
    modules: dict[str, object] = {}
    for owner_name, model_name in owners:
        owner = getattr(system, owner_name, None)
        model = getattr(owner, model_name, None)
        if model is None:
            modules[owner_name] = {"available": False}
        else:
            modules[owner_name] = module_inventory(owner_name, model)
    memory: dict[str, int | None] = {
        "allocated_bytes": None,
        "reserved_bytes": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            memory = {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
            }
    except ImportError:
        pass
    return {"modules": modules, "memory": memory}


def serve(
    input_stream: IO[str],
    protocol_output: IO[str],
    system_factory: Callable[[], FireRedSystem],
    audio_info: Callable[[Path], AudioInfo] = read_audio_info,
    clock: Callable[[], float] = time.perf_counter,
    peak_sampler: Callable[[], int | None] = _no_peak_sample,
    inventory_provider: Callable[[FireRedSystem], dict[str, object]] = (
        firered_inventory
    ),
) -> None:
    """Load FireRed once, then serve requests until shutdown or EOF."""

    load_started = clock()
    system = system_factory()
    load_seconds = clock() - load_started
    inference_seconds = 0.0
    seen_ids: set[str] = set()
    for line in input_stream:
        if not line.strip():
            continue
        identifier: str | None = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            identifier = str(request.get("id") or "")
            if not identifier:
                raise ValueError("request id must not be empty")
            if identifier in seen_ids:
                raise ValueError(f"duplicate request id: {identifier}")
            seen_ids.add(identifier)
            action = str(request.get("action") or "")
            if action == "shutdown":
                _write_response(
                    protocol_output,
                    {"id": identifier, "ok": True, "shutdown": True},
                )
                return
            if action == "cleanup_cuda_cache":
                cleanup_started = clock()
                cleanup = _cleanup_cuda_cache()
                cleanup["seconds"] = clock() - cleanup_started
                _write_response(
                    protocol_output,
                    {
                        "id": identifier,
                        "ok": True,
                        "cache_cleanup": cleanup,
                    },
                )
                continue
            if action == "warmup":
                path = _validate_audio(request.get("audio_path"), audio_info)
                warmup_started = clock()
                warmup_info = _warmup_process(system, str(path))
                resident_allocated_mib = _cuda_memory_allocated_mib()
                resident_reserved_mib = _cuda_reserved_memory_mib()
                _write_response(
                    protocol_output,
                    {
                        "id": identifier,
                        "ok": True,
                        "warmup_seconds": clock() - warmup_started,
                        "warmup_excluded": True,
                        "warmup_shapes": warmup_info.get("shapes", []),
                        "resident_allocated_mib": resident_allocated_mib,
                        "resident_reserved_mib": resident_reserved_mib,
                    },
                )
                continue
            if action == "status":
                _write_response(
                    protocol_output,
                    {
                        "id": identifier,
                        "ok": True,
                        "status": inventory_provider(system),
                        "metrics": {
                            "load_seconds": load_seconds,
                            "peak_gpu_bytes": peak_sampler(),
                        },
                    },
                )
                continue
            if action not in {
                "transcribe", "verify", "refine_vad", "aed_plan",
                "punctuate", "punctuate_chunk",
            }:
                raise ValueError(f"unsupported worker action: {action}")
            path = _validate_audio(request.get("audio_path"), audio_info)
            raw_segments = request.get("vad_segments_ms")
            scripted_segments = (
                None
                if raw_segments is None
                else _parse_scripted_segments(raw_segments)
            )
            if action == "refine_vad":
                if scripted_segments is None:
                    raise ValueError("refine_vad requires vad_segments_ms")
                original_vad = getattr(system, "vad", None)
                if original_vad is None:
                    raise ValueError("worker was started without VAD")
                refined, _ = ScriptedVad(scripted_segments, original_vad).detect(str(path))
                _write_response(
                    protocol_output,
                    {
                        "id": identifier,
                        "ok": True,
                        "vad_segments_ms": [
                            [int(round(start * 1000)), int(round(end * 1000))]
                            for start, end in refined["timestamps"]
                        ],
                    },
                )
                continue
            if action == "aed_plan":
                if scripted_segments is None:
                    raise ValueError("aed_plan requires vad_segments_ms")
                raw_plan = request.get("batch_plan")
                if not isinstance(raw_plan, list):
                    raise ValueError("aed_plan requires batch_plan")
                on_batch = None
                if bool(request.get("stream")):
                    def on_batch(event: dict[str, object]) -> None:
                        _write_response(
                            protocol_output,
                            {
                                "id": identifier,
                                "ok": True,
                                "event": "aed_batch",
                                **event,
                            },
                        )

                response = _explicit_aed_process(
                    system,
                    str(path),
                    identifier,
                    raw_plan,
                    scripted_segments,
                    on_batch=on_batch,
                )
                _write_response(
                    protocol_output,
                    {"id": identifier, "ok": True, **response},
                )
                continue
            if action == "punctuate":
                raw_results = request.get("aed_results")
                if not isinstance(raw_results, list):
                    raise ValueError("punctuate requires aed_results")
                raw_vad_ms = request.get("vad_segments_ms")
                if not isinstance(raw_vad_ms, list):
                    raise ValueError("punctuate requires vad_segments_ms")
                duration_ms = int(request.get("duration_ms") or 0)
                if duration_ms <= 0:
                    raise ValueError("punctuate requires positive duration_ms")
                response = _punctuate_aed_results(
                    system,
                    identifier,
                    raw_results,
                    duration_ms,
                    raw_vad_ms,
                )
                _write_response(
                    protocol_output,
                    normalize_official_result(identifier, response),
                )
                continue
            if action == "punctuate_chunk":
                raw_results = request.get("aed_results")
                if not isinstance(raw_results, list):
                    raise ValueError("punctuate_chunk requires aed_results")
                resident_allocated_mib = _cuda_memory_allocated_mib()
                resident_reserved_mib = _cuda_reserved_memory_mib()
                _reset_cuda_peak_memory()
                punc_started = clock()
                response = _punctuate_aed_items(system, raw_results)
                _synchronize_cuda()
                punc_metrics = {
                    "actual_seconds": clock() - punc_started,
                    "peak_allocated_mib": _cuda_peak_memory_mib(),
                    "peak_reserved_mib": _cuda_peak_reserved_mib(),
                    "resident_allocated_mib": resident_allocated_mib,
                    "resident_reserved_mib": resident_reserved_mib,
                }
                _write_response(
                    protocol_output,
                    {
                        "id": identifier,
                        "ok": True,
                        "punc_results": response,
                        "punc_metrics": punc_metrics,
                    },
                )
                continue
            inference_started = clock()
            if scripted_segments is None:
                raw = system.process(str(path), uttid=identifier)
            else:
                original_vad = getattr(system, "vad", None)
                scripted_vad = ScriptedVad(scripted_segments, original_vad)
                system.vad = scripted_vad
                try:
                    if (
                        hasattr(system, "asr")
                        and hasattr(system, "config")
                    ):
                        vad_result, _ = scripted_vad.detect(str(path))
                        raw = _scheduled_process(
                            system,
                            str(path),
                            identifier,
                            list(vad_result["timestamps"]),
                        )
                    else:
                        raw = system.process(str(path), uttid=identifier)
                finally:
                    system.vad = original_vad
            inference_seconds += clock() - inference_started
            response = normalize_official_result(identifier, raw)
            response["metrics"] = {
                "load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "peak_gpu_bytes": peak_sampler(),
            }
            scheduler_metrics = raw.get("_scheduler")
            if isinstance(scheduler_metrics, dict):
                response["metrics"]["scheduler"] = scheduler_metrics
            if action == "verify":
                response["verified"] = True
            _write_response(protocol_output, response)
        except Exception as exc:
            LOGGER.exception("FireRed worker request failed")
            _write_response(
                protocol_output,
                _error_response(identifier or None, exc),
            )


def build_firered_system(
    model_root: Path,
    device: str,
    use_half: bool = False,
    enable_vad: bool = True,
    enable_punc: bool = True,
    punc_token_budget: int = PUNC_TOKEN_BUDGET,
    punc_max_items: int = PUNC_INTERNAL_BATCH_SIZE,
) -> FireRedSystem:
    """Construct the AED/Punc worker and its hosted speaker-VAD helper."""

    import torch
    from .firered_compat import install_bf16_forced_align_compat
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    from fireredasr2s.fireredvad import FireRedVadConfig

    if not device.startswith("cuda:"):
        raise ValueError("FireRed worker requires an explicit CUDA device")
    try:
        device_index = int(device.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid CUDA device: {device}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the FireRed environment")
    torch.cuda.set_device(device_index)
    install_bf16_forced_align_compat()

    root = Path(model_root).expanduser().resolve() / "fireredasr2s"
    config = FireRedAsr2SystemConfig(
        vad_model_dir=str(root / "FireRedVAD" / "VAD"),
        asr_type="aed",
        asr_model_dir=str(root / "FireRedASR2-AED"),
        punc_model_dir=str(root / "FireRedPunc"),
        vad_config=FireRedVadConfig(
            use_gpu=False,
            smooth_window_size=5,
            speech_threshold=0.4,
            min_speech_frame=20,
            max_speech_frame=2000,
            min_silence_frame=20,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        ),
        asr_config=FireRedAsr2Config(
            use_gpu=True,
            use_half=use_half,
            beam_size=3,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=True,
        ),
        punc_config=FireRedPuncConfig(use_gpu=True),
        asr_batch_size=1,
        punc_batch_size=PUNC_INTERNAL_BATCH_SIZE,
        enable_vad=enable_vad,
        enable_lid=False,
        enable_punc=enable_punc,
    )
    system = FireRedAsr2System(config)
    system.punc_token_budget = int(punc_token_budget)
    system.punc_max_items = int(punc_max_items)
    if system.vad is not None:
        system.vad = GuardedVad(system.vad)
    return system


def build_punc_system(
    model_root: Path,
    device: str,
    punc_token_budget: int = PUNC_TOKEN_BUDGET,
    punc_max_items: int = PUNC_INTERNAL_BATCH_SIZE,
) -> FireRedSystem:
    """Construct the single-model Punc owner used by the streaming path."""

    import torch
    from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig

    if not device.startswith("cuda:"):
        raise ValueError("Punc worker requires an explicit CUDA device")
    try:
        device_index = int(device.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid CUDA device: {device}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Punc environment")
    torch.cuda.set_device(device_index)
    root = Path(model_root).expanduser().resolve() / "fireredasr2s"
    system = SimpleNamespace(
        asr=None,
        vad=None,
        lid=None,
        punc=FireRedPunc.from_pretrained(
            str(root / "FireRedPunc"),
            FireRedPuncConfig(use_gpu=True),
        ),
        config=SimpleNamespace(
            enable_vad=False,
            enable_lid=False,
            enable_punc=True,
        ),
    )
    system.punc_token_budget = int(punc_token_budget)
    system.punc_max_items = int(punc_max_items)
    return system


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-half", action="store_true")
    parser.add_argument("--disable-vad", action="store_true")
    parser.add_argument("--disable-punc", action="store_true")
    parser.add_argument("--punc-only", action="store_true")
    parser.add_argument("--punc-token-budget", type=int, default=PUNC_TOKEN_BUDGET)
    parser.add_argument("--punc-max-items", type=int, default=PUNC_INTERNAL_BATCH_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    protocol_output = sys.stdout
    sys.stdout = sys.stderr
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    serve(
        sys.stdin,
        protocol_output,
        lambda: (
            build_punc_system(
                args.model_root,
                args.device,
                args.punc_token_budget,
                args.punc_max_items,
            )
            if args.punc_only
            else build_firered_system(
                args.model_root,
                args.device,
                args.use_half,
                not args.disable_vad,
                not args.disable_punc,
                args.punc_token_budget,
                args.punc_max_items,
            )
        ),
        peak_sampler=_peak_gpu_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

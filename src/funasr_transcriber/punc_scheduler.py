"""Compact Punc inputs and deterministic token-budgeted microbatching."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

DEFAULT_PUNC_TOKEN_BUDGET = 256
DEFAULT_PUNC_MAX_ITEMS = 16


@dataclass(frozen=True, slots=True)
class PuncRecord:
    """The minimum AED result data required by FireRedPunc."""

    index: int
    uttid: str
    timestamp: tuple[tuple[str, float, float], ...]
    confidence: float


def assemble_punc_items(
    request_id: str,
    items: Sequence[Mapping[str, object]],
    *,
    duration_ms: int,
    vad_segments_ms: Sequence[Sequence[int]],
) -> dict[str, object]:
    """Build the raw stable-response shape from indexed Punc results."""

    sentences: list[dict[str, object]] = []
    words: list[dict[str, object]] = []
    for item in sorted(items, key=lambda value: int(value["index"])):
        start_token, end_token = str(item["uttid"]).split("_")[-2:]
        start_ms = int(start_token[1:])
        end_ms = int(end_token[1:])
        raw_sentences = item.get("punc_sentences")
        if not isinstance(raw_sentences, (list, tuple)):
            raise ValueError("Punc result sentences must be a sequence")
        for sentence_index, sentence in enumerate(raw_sentences):
            if not isinstance(sentence, Mapping):
                raise ValueError("Punc sentence must be an object")
            sentence_start = start_ms + int(float(sentence["start_s"]) * 1000)
            sentence_end = (
                end_ms
                if sentence_index == len(raw_sentences) - 1
                else start_ms + int(float(sentence["end_s"]) * 1000)
            )
            if sentence_index == 0:
                sentence_start = start_ms
            sentences.append({
                "start_ms": sentence_start,
                "end_ms": sentence_end,
                "text": str(sentence["punc_text"]),
                "asr_confidence": float(item["confidence"]),
            })
        raw_timestamp = item.get("timestamp", ())
        if not isinstance(raw_timestamp, (list, tuple)):
            raise ValueError("Punc timestamp must be a sequence")
        for word, start, end in raw_timestamp:
            words.append({
                "start_ms": int(float(start) * 1000 + start_ms),
                "end_ms": int(float(end) * 1000 + start_ms),
                "text": str(word),
            })
    text = "".join(str(sentence["text"]) for sentence in sentences)
    text = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", text)
    return {
        "uttid": request_id,
        "text": text,
        "sentences": sentences,
        "vad_segments_ms": [list(segment) for segment in vad_segments_ms],
        "dur_s": int(duration_ms) / 1000.0,
        "words": words,
    }


def compact_aed_record(record: Mapping[str, object]) -> PuncRecord:
    """Drop AED fields that Punc and final assembly do not need."""

    raw_result = record.get("result")
    if not isinstance(raw_result, Mapping):
        raise ValueError("AED record result must be an object")
    raw_timestamp = raw_result.get("timestamp", ())
    if not isinstance(raw_timestamp, (list, tuple)):
        raise ValueError("AED timestamp must be a sequence")
    timestamp: list[tuple[str, float, float]] = []
    for item in raw_timestamp:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError("AED timestamp entries must contain three values")
        timestamp.append((str(item[0]), float(item[1]), float(item[2])))
    return PuncRecord(
        index=int(record["index"]),
        uttid=str(raw_result["uttid"]),
        timestamp=tuple(timestamp),
        confidence=float(raw_result.get("confidence", 0.0)),
    )


def pack_punc_records(
    records: Sequence[PuncRecord] | Iterable[PuncRecord],
    *,
    token_budget: int = 128,
    max_items: int = 16,
) -> list[tuple[PuncRecord, ...]]:
    """Pack records by timestamp-token cost without reordering them."""

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    batches: list[tuple[PuncRecord, ...]] = []
    current: list[PuncRecord] = []
    current_cost = 0
    for record in records:
        cost = max(1, len(record.timestamp))
        would_exceed = (
            current
            and (current_cost + cost > token_budget or len(current) >= max_items)
        )
        if would_exceed:
            batches.append(tuple(current))
            current = []
            current_cost = 0
        current.append(record)
        current_cost += cost
    if current:
        batches.append(tuple(current))
    return batches

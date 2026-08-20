"""Label-invariant Chinese transcription and speaker metrics."""

from __future__ import annotations

import itertools
import re
import unicodedata
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


PUNCTUATION_RE = re.compile(r"[\s，。！？；：、,.!?;:…\"'“”‘’（）()\[\]【】]+")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    return PUNCTUATION_RE.sub("", normalized)


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def cpcer(
    reference_by_speaker: dict[Any, str],
    hypothesis_by_speaker: dict[Any, str],
) -> float:
    references = [
        normalize_text(text)
        for _, text in sorted(reference_by_speaker.items(), key=lambda item: str(item[0]))
    ]
    hypotheses = [
        normalize_text(text)
        for _, text in sorted(hypothesis_by_speaker.items(), key=lambda item: str(item[0]))
    ]
    size = max(len(references), len(hypotheses))
    if size == 0:
        return 0.0
    costs = np.zeros((size, size), dtype="float64")
    for row in range(size):
        for column in range(size):
            if row < len(references) and column < len(hypotheses):
                costs[row, column] = edit_distance(
                    references[row],
                    hypotheses[column],
                )
            elif row < len(references):
                costs[row, column] = len(references[row])
            elif column < len(hypotheses):
                costs[row, column] = len(hypotheses[column])
    rows, columns = linear_sum_assignment(costs)
    errors = float(costs[rows, columns].sum())
    denominator = sum(len(text) for text in references)
    if denominator == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / denominator


def speaker_count_metrics(
    reference_counts: list[int],
    hypothesis_counts: list[int],
) -> dict[str, float]:
    if not reference_counts or len(reference_counts) != len(hypothesis_counts):
        raise ValueError("speaker count lists must be non-empty and equal length")
    differences = [
        hypothesis - reference
        for reference, hypothesis in zip(reference_counts, hypothesis_counts)
    ]
    return {
        "signed_mean": sum(differences) / len(differences),
        "mae": sum(abs(value) for value in differences) / len(differences),
        "exact_accuracy": (
            sum(value == 0 for value in differences) / len(differences)
        ),
    }


def _overlap(first: dict[str, Any], second: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(first["end"]), float(second["end"]))
        - max(float(first["start"]), float(second["start"])),
    )


def _label(item: dict[str, Any]) -> str:
    for key in ("speaker", "spk", "speaker_id"):
        if item.get(key) is not None:
            return str(item[key])
    raise ValueError(f"interval has no speaker label: {item}")


def _normalized_intervals(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for item in items:
        start, end = float(item["start"]), float(item["end"])
        if end > start:
            output.append({"start": start, "end": end, "label": _label(item)})
    return sorted(output, key=lambda item: (item["start"], item["end"]))


def _mapping(overlaps: dict[str, dict[str, float]]) -> dict[str, str]:
    clusters = sorted(overlaps)
    speakers = sorted(
        {speaker for values in overlaps.values() for speaker in values}
    )
    if not clusters or not speakers:
        return {}
    if len(clusters) > len(speakers):
        return {
            cluster: max(
                speakers,
                key=lambda speaker: overlaps[cluster].get(speaker, 0.0),
            )
            for cluster in clusters
        }
    costs = np.zeros((len(clusters), len(speakers)), dtype="float64")
    for row, cluster in enumerate(clusters):
        for column, speaker in enumerate(speakers):
            costs[row, column] = -overlaps[cluster].get(speaker, 0.0)
    rows, columns = linear_sum_assignment(costs)
    return {
        clusters[row]: speakers[column]
        for row, column in zip(rows, columns)
    }


def _label_at(intervals: list[dict[str, Any]], position: float) -> str | None:
    for item in intervals:
        if item["start"] <= position < item["end"]:
            return item["label"]
    return None


def evaluate_speaker_tracks(
    reference: list[dict[str, Any]],
    hypothesis: list[dict[str, Any]],
) -> dict[str, float]:
    refs = _normalized_intervals(reference)
    hyps = _normalized_intervals(hypothesis)
    overlaps: dict[str, dict[str, float]] = {}
    for hyp in hyps:
        for ref in refs:
            seconds = _overlap(hyp, ref)
            if seconds > 0:
                values = overlaps.setdefault(hyp["label"], {})
                values[ref["label"]] = values.get(ref["label"], 0.0) + seconds
    total = sum(sum(values.values()) for values in overlaps.values())
    dominant = sum(max(values.values()) for values in overlaps.values())
    purity = dominant / total if total else 0.0
    mapping = _mapping(overlaps)

    points = sorted(
        {
            point
            for item in refs + hyps
            for point in (item["start"], item["end"])
        }
    )
    ref_speech = miss = false_alarm = confusion = 0.0
    for start, end in itertools.pairwise(points):
        midpoint = (start + end) / 2
        duration = end - start
        ref_label = _label_at(refs, midpoint)
        hyp_label = _label_at(hyps, midpoint)
        if ref_label is not None:
            ref_speech += duration
        if ref_label is None and hyp_label is not None:
            false_alarm += duration
        elif ref_label is not None and hyp_label is None:
            miss += duration
        elif (
            ref_label is not None
            and hyp_label is not None
            and mapping.get(hyp_label) != ref_label
        ):
            confusion += duration
    der = (
        (miss + false_alarm + confusion) / ref_speech
        if ref_speech
        else 0.0
    )
    return {
        "weighted_purity": round(purity, 6),
        "der_like": round(der, 6),
    }


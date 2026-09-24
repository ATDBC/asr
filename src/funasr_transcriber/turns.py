"""Build stable human-readable speaker turns."""

from __future__ import annotations

from .schemas import AlignmentConfidence, AttributedToken, Turn


_CONFIDENCE_RANK = {
    AlignmentConfidence.LOW: 0,
    AlignmentConfidence.MEDIUM: 1,
    AlignmentConfidence.HIGH: 2,
}


def _lower_confidence(
    first: AlignmentConfidence,
    second: AlignmentConfidence,
) -> AlignmentConfidence:
    return min((first, second), key=_CONFIDENCE_RANK.__getitem__)


def build_turns(
    tokens: tuple[AttributedToken, ...],
    merge_gap_ms: int,
    *,
    preserve_input_ties: bool = False,
) -> tuple[Turn, ...]:
    if merge_gap_ms < 0:
        raise ValueError("merge_gap_ms must be non-negative")

    accumulated: list[dict] = []
    for token in sorted(
        tokens,
        key=lambda item: (
            (item.start_ms, item.speaker_id)
            if preserve_input_ties
            else (item.start_ms, item.speaker_id, item.end_ms)
        ),
    ):
        previous = accumulated[-1] if accumulated else None
        gap_ms = token.start_ms - previous["end_ms"] if previous else None
        can_merge = (
            previous is not None
            and previous["speaker_id"] == token.speaker_id
            and gap_ms is not None
            and 0 <= gap_ms <= merge_gap_ms
            and not previous["overlap"]
            and not token.overlap
        )
        if can_merge:
            previous["end_ms"] = max(previous["end_ms"], token.end_ms)
            previous["text"] += token.text
            previous["confidence"] = _lower_confidence(
                previous["confidence"],
                token.confidence,
            )
            if previous["source"] != token.source:
                previous["source"] = "mixed"
            continue
        accumulated.append(
            {
                "speaker_id": token.speaker_id,
                "start_ms": token.start_ms,
                "end_ms": token.end_ms,
                "text": token.text,
                "source": token.source,
                "confidence": token.confidence,
                "overlap": token.overlap,
            }
        )

    return tuple(
        Turn(index=index, **item)
        for index, item in enumerate(accumulated, start=1)
    )

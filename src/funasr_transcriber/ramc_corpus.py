"""Parse and deterministically select material from MagicData-RAMC."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
import wave


TRANSCRIPT_RANGE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?),(?P<end>\d+(?:\.\d+)?)\]$"
)


@dataclass(frozen=True, slots=True)
class TranscriptRow:
    start_ms: int
    end_ms: int
    speaker_id: str
    gender: str
    language: str
    text: str


@dataclass(frozen=True, slots=True)
class RecordingInfo:
    wav_name: str
    speaker_ids: tuple[str, str]
    topic: str
    valid_minutes: float
    total_minutes: float
    environment: str


@dataclass(frozen=True, slots=True)
class SpeakerInfo:
    speaker_id: str
    gender: str
    region: str


@dataclass(frozen=True, slots=True)
class CoreCandidate:
    wav_name: str
    wav_path: Path
    transcript_path: Path
    rows: tuple[TranscriptRow, ...]
    speaker_ids: tuple[str, str]
    speaker_speech_ms: tuple[tuple[str, int], ...]
    character_count: int
    topic: str
    gender_pair: tuple[str, str]
    regions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CleanClip:
    wav_name: str
    wav_path: Path
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class SyntheticMix:
    case_id: str
    first: CleanClip
    second: CleanClip
    snr_db: int
    overlap_ratio: float
    second_offset_ms: int
    first_offset_ms: int = 0


@dataclass(frozen=True, slots=True)
class WindowClip:
    case_id: str
    kind: str
    wav_name: str
    wav_path: Path
    start_ms: int
    end_ms: int
    reference_rows: tuple[TranscriptRow, ...]
    gap_ms: int | None = None
    source_marker: TranscriptRow | None = None


@dataclass(frozen=True, slots=True)
class OverlapPlan:
    synthetic: tuple[SyntheticMix, ...]
    real: tuple[WindowClip, ...]
    negative: tuple[WindowClip, ...]


def _row_error(path: Path, line_number: int, message: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {message}")


def parse_transcript(path: Path) -> tuple[TranscriptRow, ...]:
    source = Path(path)
    output: list[TranscriptRow] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = raw_line.split("\t", 3)
        if len(fields) != 4:
            raise _row_error(source, line_number, "expected four tab-separated fields")
        match = TRANSCRIPT_RANGE.fullmatch(fields[0].strip())
        if match is None:
            raise _row_error(source, line_number, "invalid timestamp range")
        speaker_id = fields[1].strip()
        gender_language = fields[2].split(",", 1)
        text = fields[3].strip()
        if not speaker_id:
            raise _row_error(source, line_number, "speaker ID is empty")
        if len(gender_language) != 2 or not all(
            item.strip() for item in gender_language
        ):
            raise _row_error(source, line_number, "invalid gender/language field")
        if not text:
            raise _row_error(source, line_number, "transcription is empty")
        start_ms = int(round(float(match.group("start")) * 1000))
        end_ms = int(round(float(match.group("end")) * 1000))
        if start_ms < 0 or end_ms <= start_ms:
            raise _row_error(source, line_number, "timestamp range is not positive")
        output.append(
            TranscriptRow(
                start_ms,
                end_ms,
                speaker_id,
                gender_language[0].strip(),
                gender_language[1].strip(),
                text,
            )
        )
    return tuple(output)


def _metadata_rows(path: Path) -> tuple[csv.DictReader, object]:
    stream = Path(path).open("r", encoding="utf-8", newline="")
    return csv.DictReader(stream, delimiter="\t"), stream


def parse_utterance_info(path: Path) -> dict[str, RecordingInfo]:
    source = Path(path)
    reader, stream = _metadata_rows(source)
    output: dict[str, RecordingInfo] = {}
    required = {
        "UTTRANS_ID",
        "SPEAKER_ID-1",
        "SPEAKER_ID-2",
        "TOPIC",
        "VALID(min)",
        "TOTAL(min)",
        "Environment",
    }
    try:
        if not required.issubset(set(reader.fieldnames or ())):
            raise _row_error(source, 1, "missing required metadata columns")
        for row in reader:
            line_number = reader.line_num
            try:
                wav_name = str(row["UTTRANS_ID"] or "").strip()
                first = str(row["SPEAKER_ID-1"] or "").strip()
                second = str(row["SPEAKER_ID-2"] or "").strip()
                topic = str(row["TOPIC"] or "").strip()
                environment = str(row["Environment"] or "").strip()
                valid = float(str(row["VALID(min)"] or ""))
                total = float(str(row["TOTAL(min)"] or ""))
            except (KeyError, TypeError, ValueError) as exc:
                raise _row_error(source, line_number, "invalid metadata values") from exc
            if not all((wav_name, first, second, topic, environment)):
                raise _row_error(source, line_number, "metadata value is empty")
            if wav_name in output:
                raise _row_error(source, line_number, f"duplicate recording: {wav_name}")
            if valid <= 0 or total <= 0 or valid > total:
                raise _row_error(source, line_number, "invalid recording minutes")
            output[wav_name] = RecordingInfo(
                wav_name,
                (first, second),
                topic,
                valid,
                total,
                environment,
            )
    finally:
        stream.close()
    return output


def parse_speaker_info(path: Path) -> dict[str, SpeakerInfo]:
    source = Path(path)
    reader, stream = _metadata_rows(source)
    output: dict[str, SpeakerInfo] = {}
    required = {"SPEAKER_ID", "GENDER", "REGION,CITY"}
    try:
        if not required.issubset(set(reader.fieldnames or ())):
            raise _row_error(source, 1, "missing required speaker columns")
        for row in reader:
            line_number = reader.line_num
            speaker_id = str(row.get("SPEAKER_ID") or "").strip()
            gender = str(row.get("GENDER") or "").strip()
            region = str(row.get("REGION,CITY") or "").strip()
            if not all((speaker_id, gender, region)):
                raise _row_error(source, line_number, "speaker value is empty")
            if speaker_id in output:
                raise _row_error(
                    source,
                    line_number,
                    f"duplicate speaker: {speaker_id}",
                )
            output[speaker_id] = SpeakerInfo(speaker_id, gender, region)
    finally:
        stream.close()
    return output


CORE_START_MS = 60_000
CORE_END_MS = 480_000
MIN_SPEAKER_MS = 30_000
MIN_CORE_CHARACTERS = 500
CHINESE_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _intersection_ms(
    start_ms: int,
    end_ms: int,
    window_start_ms: int,
    window_end_ms: int,
) -> int:
    return max(0, min(end_ms, window_end_ms) - max(start_ms, window_start_ms))


def analyze_core_candidate(
    wav_path: Path,
    transcript_path: Path,
    rows: tuple[TranscriptRow, ...],
    recording: RecordingInfo,
    speakers: dict[str, SpeakerInfo],
    duration_ms: int,
) -> CoreCandidate | None:
    if duration_ms < CORE_END_MS:
        return None
    expected_speakers = recording.speaker_ids
    if len(set(expected_speakers)) != 2:
        return None
    if any(speaker_id not in speakers for speaker_id in expected_speakers):
        return None
    speech_ms = {speaker_id: 0 for speaker_id in expected_speakers}
    character_count = 0
    window_rows: list[TranscriptRow] = []
    effective_speakers: set[str] = set()
    for row in rows:
        overlap_ms = _intersection_ms(
            row.start_ms,
            row.end_ms,
            CORE_START_MS,
            CORE_END_MS,
        )
        if overlap_ms <= 0:
            continue
        if row.speaker_id != "G00000000" and strip_control_markers(row.text):
            effective_speakers.add(row.speaker_id)
        if row.speaker_id not in speech_ms:
            continue
        speech_ms[row.speaker_id] += overlap_ms
        character_count += len(CHINESE_CHARACTER.findall(row.text))
        window_rows.append(row)
    if effective_speakers != set(expected_speakers):
        return None
    if any(value < MIN_SPEAKER_MS for value in speech_ms.values()):
        return None
    if character_count < MIN_CORE_CHARACTERS:
        return None
    gender_pair = tuple(sorted(speakers[item].gender for item in expected_speakers))
    regions = tuple(sorted({speakers[item].region for item in expected_speakers}))
    return CoreCandidate(
        wav_name=recording.wav_name,
        wav_path=Path(wav_path),
        transcript_path=Path(transcript_path),
        rows=tuple(rows),
        speaker_ids=expected_speakers,
        speaker_speech_ms=tuple(
            (speaker_id, speech_ms[speaker_id])
            for speaker_id in expected_speakers
        ),
        character_count=character_count,
        topic=recording.topic,
        gender_pair=gender_pair,
        regions=regions,
    )


def _wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as stream:
            return int(round(stream.getnframes() * 1000 / stream.getframerate()))
    except (wave.Error, OSError, ZeroDivisionError) as exc:
        raise ValueError(f"unable to read WAV duration: {path}: {exc}") from exc


def discover_core_candidates(
    dataset_root: Path,
    rejections: list[str] | None = None,
) -> tuple[CoreCandidate, ...]:
    root = Path(dataset_root)
    recordings = parse_utterance_info(root / "UTTERANCEINFO.txt")
    speakers = parse_speaker_info(root / "SPKINFO.txt")
    output: list[CoreCandidate] = []
    for wav_name in sorted(recordings):
        wav_path = root / "WAV" / wav_name
        transcript_path = root / "TXT" / f"{Path(wav_name).stem}.txt"
        try:
            if not wav_path.is_file() or not transcript_path.is_file():
                raise ValueError("missing paired WAV or transcript")
            candidate = analyze_core_candidate(
                wav_path,
                transcript_path,
                parse_transcript(transcript_path),
                recordings[wav_name],
                speakers,
                _wav_duration_ms(wav_path),
            )
        except (OSError, ValueError) as exc:
            if rejections is not None:
                detail = str(exc)
                for private_path in (root, wav_path, transcript_path):
                    detail = detail.replace(str(private_path), private_path.name)
                rejections.append(f"{wav_name}: {type(exc).__name__}: {detail}")
            continue
        if candidate is not None:
            output.append(candidate)
        elif rejections is not None:
            rejections.append(f"{wav_name}: ineligible core recording")
    return tuple(output)


def select_core_candidates(
    candidates: tuple[CoreCandidate, ...],
    count: int = 20,
) -> tuple[CoreCandidate, ...]:
    if count <= 0:
        raise ValueError("core candidate count must be positive")
    if len(candidates) < count:
        raise ValueError(
            f"requested {count} core recordings, only {len(candidates)} eligible"
        )
    remaining = {item.wav_name: item for item in candidates}
    if len(remaining) != len(candidates):
        raise ValueError("core candidate WAV names must be unique")
    selected: list[CoreCandidate] = []
    covered_topics: set[str] = set()
    covered_regions: set[str] = set()
    gender_counts: dict[tuple[str, str], int] = {}
    while len(selected) < count:
        ordered = sorted(
            remaining.values(),
            key=lambda item: (
                -(item.topic not in covered_topics),
                gender_counts.get(item.gender_pair, 0),
                -len(set(item.regions) - covered_regions),
                item.wav_name,
            ),
        )
        chosen = ordered[0]
        selected.append(chosen)
        remaining.pop(chosen.wav_name)
        covered_topics.add(chosen.topic)
        covered_regions.update(chosen.regions)
        gender_counts[chosen.gender_pair] = (
            gender_counts.get(chosen.gender_pair, 0) + 1
        )
    return tuple(selected)


CONTROL_MARKERS = ("[+]", "[*]", "[LAUGHTER]", "[SONANT]", "[MUSIC]")
SNR_SCHEDULE = (-6, -3, 0, 3, 6)
OVERLAP_RATIO_SCHEDULE = (0.25, 0.5, 0.75, 1.0)


def _contains_control_marker(text: str) -> bool:
    return any(marker in text for marker in CONTROL_MARKERS)


def strip_control_markers(text: str) -> str:
    output = text
    for marker in CONTROL_MARKERS:
        output = output.replace(marker, "")
    return output.strip()


def _rows_overlap(first: TranscriptRow, second: TranscriptRow) -> bool:
    return (
        first.speaker_id != second.speaker_id
        and _intersection_ms(
            first.start_ms,
            first.end_ms,
            second.start_ms,
            second.end_ms,
        ) > 0
    )


def _row_gap_ms(first: TranscriptRow, second: TranscriptRow) -> int:
    if first.end_ms <= second.start_ms:
        return second.start_ms - first.end_ms
    if second.end_ms <= first.start_ms:
        return first.start_ms - second.end_ms
    return 0


def clean_clips(candidates: tuple[CoreCandidate, ...]) -> tuple[CleanClip, ...]:
    output: list[CleanClip] = []
    for candidate in sorted(candidates, key=lambda item: item.wav_name):
        rows = tuple(sorted(candidate.rows, key=lambda item: (item.start_ms, item.end_ms)))
        active: list[tuple[int, TranscriptRow]] = []
        overlaps: set[int] = set()
        for index, row in enumerate(rows):
            active = [item for item in active if item[1].end_ms > row.start_ms]
            for other_index, other in active:
                if other.speaker_id != row.speaker_id:
                    overlaps.update((index, other_index))
            active.append((index, row))
        for index, row in enumerate(rows):
            if row.speaker_id == "G00000000":
                continue
            duration_ms = row.end_ms - row.start_ms
            if not 3_000 <= duration_ms <= 10_000:
                continue
            if _contains_control_marker(row.text):
                continue
            if len(CHINESE_CHARACTER.findall(row.text)) < 8:
                continue
            if index in overlaps:
                continue
            output.append(
                CleanClip(
                    candidate.wav_name,
                    candidate.wav_path,
                    row.speaker_id,
                    row.start_ms,
                    row.end_ms,
                    row.text,
                )
            )
    return tuple(output)


def _shift_reference_rows(
    rows: tuple[TranscriptRow, ...],
    start_ms: int,
    end_ms: int,
) -> tuple[TranscriptRow, ...]:
    output: list[TranscriptRow] = []
    for row in rows:
        if row.speaker_id == "G00000000":
            continue
        clipped_start = max(start_ms, row.start_ms)
        clipped_end = min(end_ms, row.end_ms)
        text = strip_control_markers(row.text)
        if clipped_end <= clipped_start or not text:
            continue
        output.append(
            TranscriptRow(
                clipped_start - start_ms,
                clipped_end - start_ms,
                row.speaker_id,
                row.gender,
                row.language,
                text,
            )
        )
    return tuple(output)


def _padded_window(start_ms: int, end_ms: int) -> tuple[int, int] | None:
    start = max(CORE_START_MS, start_ms - 2_000)
    end = min(CORE_END_MS, end_ms + 2_000)
    if end - start < 8_000:
        missing = 8_000 - (end - start)
        start = max(CORE_START_MS, start - (missing + 1) // 2)
        end = min(CORE_END_MS, end + missing // 2)
    if end - start > 20_000:
        center = (start_ms + end_ms) // 2
        start = max(CORE_START_MS, center - 10_000)
        end = min(CORE_END_MS, start + 20_000)
        start = max(CORE_START_MS, end - 20_000)
    if not 8_000 <= end - start <= 20_000:
        return None
    return start, end


def _select_synthetic(
    clips: tuple[CleanClip, ...],
    count: int,
) -> tuple[SyntheticMix, ...]:
    if count == 0:
        return ()
    if len(clips) < 2:
        raise ValueError("insufficient clean clips for synthetic overlap")
    groups: dict[str, list[CleanClip]] = {}
    for clip in sorted(clips, key=lambda item: (item.wav_name, item.start_ms)):
        groups.setdefault(clip.wav_name, []).append(clip)
    wav_names = sorted(groups)
    if len(wav_names) < 2:
        raise ValueError("synthetic overlap requires clips from two recordings")
    pair_candidates: list[tuple[CleanClip, CleanClip]] = []
    maximum_combinations = max(
        len(groups[left]) * len(groups[right])
        for left in wav_names
        for right in wav_names
        if left != right
    )
    for combination_index in range(maximum_combinations):
        for shift in range(1, len(wav_names)):
            for left_index, left_name in enumerate(wav_names):
                right_name = wav_names[(left_index + shift) % len(wav_names)]
                combinations = [
                    (first, second)
                    for first in groups[left_name]
                    for second in groups[right_name]
                    if first.speaker_id != second.speaker_id
                ]
                if combination_index < len(combinations):
                    pair_candidates.append(combinations[combination_index])
                    if len(pair_candidates) >= count:
                        break
            if len(pair_candidates) >= count:
                break
        if len(pair_candidates) >= count:
            break
    if len(pair_candidates) < count:
        raise ValueError("insufficient distinct clips for synthetic overlap")
    output: list[SyntheticMix] = []
    for index, (first, second) in enumerate(pair_candidates[:count]):
        ratio = OVERLAP_RATIO_SCHEDULE[(index // len(SNR_SCHEDULE)) % len(OVERLAP_RATIO_SCHEDULE)]
        shared_ms = min(
            first.end_ms - first.start_ms,
            second.end_ms - second.start_ms,
        )
        if index % 2:
            first_offset_ms = max(
                0,
                (second.end_ms - second.start_ms) - int(round(shared_ms * ratio)),
            )
            second_offset_ms = 0
        else:
            first_offset_ms = 0
            second_offset_ms = max(
                0,
                (first.end_ms - first.start_ms) - int(round(shared_ms * ratio)),
            )
        output.append(
            SyntheticMix(
                f"synthetic-{index + 1:04d}",
                first,
                second,
                SNR_SCHEDULE[index % len(SNR_SCHEDULE)],
                ratio,
                second_offset_ms,
                first_offset_ms,
            )
        )
    return tuple(output)


def _diverse_candidate_order(
    candidates: tuple[CoreCandidate, ...],
) -> tuple[CoreCandidate, ...]:
    remaining = {item.wav_name: item for item in candidates}
    output: list[CoreCandidate] = []
    covered_topics: set[str] = set()
    gender_counts: dict[tuple[str, str], int] = {}
    while remaining:
        chosen = sorted(
            remaining.values(),
            key=lambda item: (
                -(item.topic not in covered_topics),
                gender_counts.get(item.gender_pair, 0),
                item.wav_name,
            ),
        )[0]
        output.append(chosen)
        remaining.pop(chosen.wav_name)
        covered_topics.add(chosen.topic)
        gender_counts[chosen.gender_pair] = gender_counts.get(chosen.gender_pair, 0) + 1
    return tuple(output)


def _select_real(
    candidates: tuple[CoreCandidate, ...],
    count: int,
) -> tuple[WindowClip, ...]:
    if count == 0:
        return ()
    output: list[WindowClip] = []
    for candidate in _diverse_candidate_order(candidates):
        rows = tuple(sorted(candidate.rows, key=lambda item: (item.start_ms, item.end_ms)))
        for row in rows:
            if "[+]" not in row.text:
                continue
            if row.speaker_id == "G00000000":
                continue
            partners = sorted(
                (
                    other
                    for other in rows
                    if other.speaker_id != row.speaker_id
                    and other.speaker_id != "G00000000"
                    and strip_control_markers(other.text)
                    and _row_gap_ms(row, other) <= 2_000
                ),
                key=lambda other: (
                    _row_gap_ms(row, other),
                    other.start_ms,
                    other.speaker_id,
                ),
            )
            if not partners:
                continue
            partner = partners[0]
            bounds = _padded_window(
                min(row.start_ms, partner.start_ms),
                max(row.end_ms, partner.end_ms),
            )
            if bounds is None:
                continue
            start_ms, end_ms = bounds
            references = _shift_reference_rows(rows, start_ms, end_ms)
            if len({item.speaker_id for item in references}) < 2:
                continue
            output.append(
                WindowClip(
                    f"real-overlap-{len(output) + 1:04d}",
                    "real_overlap",
                    candidate.wav_name,
                    candidate.wav_path,
                    start_ms,
                    end_ms,
                    references,
                    source_marker=row,
                )
            )
            break
        if len(output) >= count:
            break
    if len(output) != count:
        raise ValueError(f"requested {count} real overlaps, found {len(output)}")
    return tuple(output)


def _select_negative(
    candidates: tuple[CoreCandidate, ...],
    count: int,
    excluded_wavs: frozenset[str] = frozenset(),
) -> tuple[WindowClip, ...]:
    if count == 0:
        return ()
    output: list[WindowClip] = []
    eligible = tuple(
        item for item in candidates if item.wav_name not in excluded_wavs
    )
    for candidate in _diverse_candidate_order(eligible):
        rows = tuple(sorted(candidate.rows, key=lambda item: (item.start_ms, item.end_ms)))
        for first, second in zip(rows, rows[1:]):
            gap_ms = second.start_ms - first.end_ms
            if (
                first.speaker_id == second.speaker_id
                or "G00000000" in {first.speaker_id, second.speaker_id}
                or not 0 <= gap_ms <= 1_000
            ):
                continue
            if _contains_control_marker(first.text) or _contains_control_marker(second.text):
                continue
            bounds = _padded_window(first.start_ms, second.end_ms)
            if bounds is None:
                continue
            start_ms, end_ms = bounds
            marker_buffer_start = max(CORE_START_MS, start_ms - 2_000)
            marker_buffer_end = min(CORE_END_MS, end_ms + 2_000)
            if any(
                "[+]" in row.text
                and _intersection_ms(
                    row.start_ms,
                    row.end_ms,
                    marker_buffer_start,
                    marker_buffer_end,
                ) > 0
                for row in rows
            ):
                continue
            window_rows = tuple(
                row
                for row in rows
                if _intersection_ms(
                    row.start_ms,
                    row.end_ms,
                    start_ms,
                    end_ms,
                ) > 0
            )
            if any("[+]" in row.text for row in window_rows):
                continue
            references = _shift_reference_rows(window_rows, start_ms, end_ms)
            if len({row.speaker_id for row in references}) < 2:
                continue
            if any(
                _rows_overlap(left, right)
                for position, left in enumerate(references)
                for right in references[position + 1:]
            ):
                continue
            output.append(
                WindowClip(
                    f"speaker-change-negative-{len(output) + 1:04d}",
                    "speaker_change_negative",
                    candidate.wav_name,
                    candidate.wav_path,
                    start_ms,
                    end_ms,
                    references,
                    gap_ms,
                )
            )
            break
        if len(output) >= count:
            break
    if len(output) != count:
        raise ValueError(
            f"requested {count} speaker-change negatives, found {len(output)}"
        )
    return tuple(output)


def select_overlap_material(
    candidates: tuple[CoreCandidate, ...],
    synthetic_count: int = 30,
    real_count: int = 10,
    negative_count: int = 10,
) -> OverlapPlan:
    if min(synthetic_count, real_count, negative_count) < 0:
        raise ValueError("overlap material counts must be non-negative")
    ordered = tuple(sorted(candidates, key=lambda item: item.wav_name))
    real = _select_real(ordered, real_count)
    return OverlapPlan(
        synthetic=_select_synthetic(clean_clips(ordered), synthetic_count),
        real=real,
        negative=_select_negative(
            ordered,
            negative_count,
            frozenset(item.wav_name for item in real),
        ),
    )

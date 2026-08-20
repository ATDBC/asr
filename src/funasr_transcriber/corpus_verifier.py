"""Independent structural verification for release validation corpora."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping
import wave

from .ramc_corpus import parse_transcript, strip_control_markers


CHECKSUM_FILE = "checksums.sha256"
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
WINDOWS_ABSOLUTE_ANYWHERE = re.compile(r"[A-Za-z]:[\\/]")
SECRET_VALUE = re.compile(r"(?:hf_|token\s*[=:]|api[_-]?key\s*[=:])", re.I)
RELEASE_COUNTS: Mapping[str, int] = {
    "core": 20,
    "synthetic": 30,
    "real_overlap": 10,
    "speaker_change_negative": 10,
    "seeds": 3,
}


@dataclass(frozen=True, slots=True)
class CorpusVerification:
    ok: bool
    errors: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in Path(root).rglob("*")
                if path.is_file() and path.name != CHECKSUM_FILE
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def write_checksums(root: Path) -> Path:
    directory = Path(root)
    target = directory / CHECKSUM_FILE
    lines = [
        f"{_sha256(path)}  {path.relative_to(directory).as_posix()}"
        for path in _artifact_files(directory)
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return target


def _load_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path.name}: {exc}")
        return None


def _verify_checksums(root: Path, errors: list[str]) -> None:
    path = root / CHECKSUM_FILE
    if not path.is_file():
        errors.append(f"missing {CHECKSUM_FILE}")
        return
    expected: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"invalid checksum line {line_number}")
            continue
        expected[parts[1]] = parts[0]
    actual_names = {
        item.relative_to(root).as_posix() for item in _artifact_files(root)
    }
    if set(expected) != actual_names:
        for missing in sorted(actual_names - set(expected)):
            errors.append(f"checksum entry missing: {missing}")
        for extra in sorted(set(expected) - actual_names):
            errors.append(f"checksum entry has no file: {extra}")
    for relative, digest in expected.items():
        target = root / Path(relative)
        if target.is_file() and _sha256(target) != digest:
            errors.append(f"checksum mismatch: {relative}")


def _wav_properties(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as stream:
        rate = stream.getframerate()
        duration_ms = int(round(stream.getnframes() * 1000 / rate))
        return duration_ms, rate, stream.getnchannels(), stream.getsampwidth()


def _verify_audio(root: Path, errors: list[str]) -> dict[str, int]:
    durations: dict[str, int] = {}
    for path in sorted(root.rglob("*.wav")):
        relative = path.relative_to(root).as_posix()
        try:
            duration, rate, channels, width = _wav_properties(path)
        except (OSError, wave.Error, ZeroDivisionError) as exc:
            errors.append(f"invalid WAV {relative}: {exc}")
            continue
        durations[relative] = duration
        if duration <= 0 or rate != 16_000 or channels != 1 or width != 2:
            errors.append(
                f"invalid WAV properties {relative}: "
                f"duration={duration}, rate={rate}, channels={channels}, width={width}"
            )
    return durations


def _string_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)
    elif isinstance(value, str):
        yield value


def _verify_path_hygiene(payloads: tuple[object, ...], errors: list[str]) -> None:
    for value in (item for payload in payloads for item in _string_values(payload)):
        if WINDOWS_ABSOLUTE.match(value) or value.startswith("/"):
            errors.append(f"absolute path leaked into manifest: {value}")
        if SECRET_VALUE.search(value):
            errors.append("secret-like value leaked into manifest")


def _verify_text_hygiene(root: Path, errors: list[str]) -> None:
    for path in _artifact_files(root):
        if path.suffix.lower() not in {".json", ".tsv", ".sh", ".txt"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root).as_posix()
        if WINDOWS_ABSOLUTE_ANYWHERE.search(content):
            errors.append(f"absolute path leaked into {relative}")
        if SECRET_VALUE.search(content):
            errors.append(f"secret-like value leaked into {relative}")


def _turn_value(turn: dict, name: str):
    if name == "speaker":
        return turn.get("speaker") or turn.get("speaker_id")
    return turn.get(name)


def _verify_reference_turns(
    item: dict,
    duration_ms: int,
    errors: list[str],
) -> None:
    case_id = str(item.get("id") or "unknown")
    turns = item.get("reference_turns") or []
    valid: list[tuple[int, int, str]] = []
    for turn in turns:
        try:
            start = int(_turn_value(turn, "start_ms"))
            end = int(_turn_value(turn, "end_ms"))
            speaker = str(_turn_value(turn, "speaker") or "")
            text = str(_turn_value(turn, "text") or "")
        except (TypeError, ValueError):
            errors.append(f"invalid reference turn: {case_id}")
            continue
        if start < 0 or end <= start or end > duration_ms or not speaker or not text:
            errors.append(f"reference turn outside audio or empty: {case_id}")
            continue
        valid.append((start, end, speaker))
    if item.get("kind") == "speaker_change_negative":
        for position, first in enumerate(valid):
            for second in valid[position + 1:]:
                if (
                    first[2] != second[2]
                    and min(first[1], second[1]) > max(first[0], second[0])
                ):
                    errors.append(f"negative reference overlaps: {case_id}")
                    return


def _verify_declared_sha(
    root: Path,
    relative: str,
    declared: object,
    errors: list[str],
    label: str,
) -> None:
    target = root / Path(relative)
    if not target.is_file():
        errors.append(f"declared file missing: {label}: {relative}")
        return
    if not isinstance(declared, str) or len(declared) != 64:
        errors.append(f"missing or invalid declared sha256: {label}")
    elif _sha256(target) != declared:
        errors.append(f"declared sha256 mismatch: {label}")


def _verify_core_layout(
    root: Path,
    durations: Mapping[str, int],
    errors: list[str],
) -> set[str]:
    tsv_path = root / "ramc_core" / "test.tsv"
    tsv_names: set[str] = set()
    if tsv_path.is_file():
        for line_number, line in enumerate(
            tsv_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 2 or not fields[0].lower().endswith(".wav"):
                errors.append(f"invalid core TSV row: {line_number}")
                continue
            name = Path(fields[0].replace("\\", "/")).name
            if name in tsv_names:
                errors.append(f"duplicate core TSV WAV: {name}")
            tsv_names.add(name)
    wav_names = {path.name for path in (root / "ramc_core" / "WAV").glob("*.wav")}
    txt_names = {
        f"{path.stem}.wav" for path in (root / "ramc_core" / "TXT").glob("*.txt")
    }
    if not (tsv_names == wav_names == txt_names):
        errors.append(
            "core TSV/WAV/TXT mismatch: "
            f"tsv={sorted(tsv_names)}, wav={sorted(wav_names)}, txt={sorted(txt_names)}"
        )
    for transcript in sorted((root / "ramc_core" / "TXT").glob("*.txt")):
        wav_relative = f"ramc_core/WAV/{transcript.stem}.wav"
        duration = durations.get(wav_relative)
        if duration is None:
            continue
        try:
            rows = parse_transcript(transcript)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        starts = [row.start_ms for row in rows]
        if starts != sorted(starts):
            errors.append(f"core transcript not monotonic: {transcript.name}")
        if any(row.end_ms > duration for row in rows):
            errors.append(f"core transcript outside audio: {transcript.name}")
    return tsv_names


def _verify_overlap_semantics(
    item: dict,
    duration_ms: int,
    errors: list[str],
) -> None:
    case_id = str(item.get("id") or "unknown")
    kind = item.get("kind")
    turns = item.get("reference_turns") or []
    speakers = {
        str(_turn_value(turn, "speaker") or "")
        for turn in turns
        if str(_turn_value(turn, "speaker") or "") != "G00000000"
    }
    speakers.discard("")
    if kind == "synthetic":
        if len(turns) != 2 or len(speakers) != 2:
            errors.append(f"synthetic speakers must be exactly two: {case_id}")
        sources = item.get("sources")
        if not isinstance(sources, list) or len(sources) != 2:
            errors.append(f"synthetic source provenance missing: {case_id}")
        first_offset = item.get("first_offset_ms")
        second_offset = item.get("second_offset_ms")
        if not isinstance(first_offset, int) or not isinstance(second_offset, int):
            errors.append(f"synthetic offsets missing: {case_id}")
        if not isinstance(item.get("linear_gain"), (int, float)):
            errors.append(f"synthetic gain missing: {case_id}")
        elif isinstance(item.get("snr_db"), (int, float)) and not math.isclose(
            float(item["linear_gain"]),
            math.pow(10.0, float(item["snr_db"]) / 20.0),
            rel_tol=1e-6,
        ):
            errors.append(f"synthetic gain inconsistent with SNR: {case_id}")
        if isinstance(sources, list) and len(sources) == 2 and len(turns) == 2:
            source_speakers = {str(source.get("speaker") or "") for source in sources}
            if source_speakers != speakers:
                errors.append(f"synthetic source/reference speakers mismatch: {case_id}")
            for source, turn in zip(sources, turns):
                if (
                    source.get("output_start_ms") != _turn_value(turn, "start_ms")
                    or source.get("speaker") != _turn_value(turn, "speaker")
                    or source.get("text") != _turn_value(turn, "text")
                ):
                    errors.append(f"synthetic source/reference mismatch: {case_id}")
                    break
                try:
                    source_duration = int(source["end_ms"]) - int(source["start_ms"])
                    turn_duration = int(_turn_value(turn, "end_ms")) - int(
                        _turn_value(turn, "start_ms")
                    )
                except (KeyError, TypeError, ValueError):
                    source_duration = -1
                    turn_duration = -2
                if source_duration != turn_duration:
                    errors.append(f"synthetic source/reference duration mismatch: {case_id}")
                    break
            if (
                first_offset != _turn_value(turns[0], "start_ms")
                or second_offset != _turn_value(turns[1], "start_ms")
            ):
                errors.append(f"synthetic offsets/reference mismatch: {case_id}")
            try:
                if max(int(_turn_value(turn, "end_ms")) for turn in turns) != duration_ms:
                    errors.append(f"synthetic output duration mismatch: {case_id}")
            except (TypeError, ValueError):
                errors.append(f"synthetic output duration invalid: {case_id}")
            try:
                first_start = int(_turn_value(turns[0], "start_ms"))
                first_end = int(_turn_value(turns[0], "end_ms"))
                second_start = int(_turn_value(turns[1], "start_ms"))
                second_end = int(_turn_value(turns[1], "end_ms"))
                overlap_ms = max(
                    0,
                    min(first_end, second_end) - max(first_start, second_start),
                )
                shared_ms = min(first_end - first_start, second_end - second_start)
                actual_ratio = overlap_ms / shared_ms
                if not math.isclose(
                    actual_ratio,
                    float(item.get("overlap_ratio")),
                    abs_tol=0.001,
                ):
                    errors.append(f"synthetic overlap ratio mismatch: {case_id}")
            except (TypeError, ValueError, ZeroDivisionError):
                errors.append(f"synthetic overlap metadata invalid: {case_id}")
    elif kind == "real_overlap":
        evidence = item.get("marker_evidence")
        valid_evidence = (
            isinstance(evidence, dict)
            and evidence.get("marker") == "[+]"
            and bool(evidence.get("source_wav"))
            and bool(evidence.get("source_speaker"))
            and isinstance(evidence.get("source_start_ms"), int)
            and isinstance(evidence.get("source_end_ms"), int)
        )
        if len(speakers) < 2 or not valid_evidence:
            errors.append(f"real overlap lacks marker evidence or two speakers: {case_id}")
        if not isinstance(item.get("source_window"), dict):
            errors.append(f"real overlap source window missing: {case_id}")
        elif valid_evidence:
            window = item["source_window"]
            try:
                evidence_in_window = (
                    evidence.get("source_wav") == window.get("wav_name")
                    and int(window.get("start_ms"))
                    <= int(evidence["source_start_ms"])
                    < int(evidence["source_end_ms"])
                    <= int(window.get("end_ms"))
                )
            except (TypeError, ValueError):
                evidence_in_window = False
            if not evidence_in_window:
                errors.append(f"real marker outside source window: {case_id}")
            else:
                marker_start = int(evidence["source_start_ms"]) - int(window["start_ms"])
                marker_end = int(evidence["source_end_ms"]) - int(window["start_ms"])
                nearby_other = False
                for turn in turns:
                    if _turn_value(turn, "speaker") == evidence.get("source_speaker"):
                        continue
                    try:
                        turn_start = int(_turn_value(turn, "start_ms"))
                        turn_end = int(_turn_value(turn, "end_ms"))
                    except (TypeError, ValueError):
                        continue
                    gap = max(0, turn_start - marker_end, marker_start - turn_end)
                    if gap <= 2_000:
                        nearby_other = True
                        break
                if not nearby_other:
                    errors.append(f"real marker has no nearby other speaker: {case_id}")
    elif kind == "speaker_change_negative":
        if not isinstance(item.get("source_window"), dict):
            errors.append(f"negative source window missing: {case_id}")
        if len(speakers) < 2:
            errors.append(f"negative requires two speakers: {case_id}")
        if not isinstance(item.get("gap_ms"), int) or not 0 <= item["gap_ms"] <= 1_000:
            errors.append(f"negative gap outside contract: {case_id}")
        else:
            try:
                ordered = sorted(
                    turns,
                    key=lambda turn: (
                        int(_turn_value(turn, "start_ms")),
                        int(_turn_value(turn, "end_ms")),
                    ),
                )
                matching_gap = any(
                    _turn_value(first, "speaker")
                    != _turn_value(second, "speaker")
                    and int(_turn_value(second, "start_ms"))
                    - int(_turn_value(first, "end_ms"))
                    == item["gap_ms"]
                    for first, second in zip(ordered, ordered[1:])
                )
            except (TypeError, ValueError):
                matching_gap = False
            if not matching_gap:
                errors.append(f"negative gap does not match reference: {case_id}")
    else:
        errors.append(f"unknown overlap kind: {case_id}: {kind}")


def verify_corpus(
    root: Path,
    expected_counts: Mapping[str, int] | None = RELEASE_COUNTS,
) -> CorpusVerification:
    directory = Path(root)
    errors: list[str] = []
    if not directory.is_dir():
        return CorpusVerification(False, (f"corpus directory missing: {directory}",))
    manifest_path = directory / "manifest.json"
    overlap_path = directory / "overlap" / "manifest.json"
    manifest = _load_json(manifest_path, errors) if manifest_path.is_file() else None
    overlap = _load_json(overlap_path, errors) if overlap_path.is_file() else None
    if manifest is not None and not isinstance(manifest, dict):
        errors.append("manifest JSON top level must be an object")
    if overlap is not None and not isinstance(overlap, list):
        errors.append("overlap manifest JSON top level must be an array")
    for required in (
        manifest_path,
        overlap_path,
        directory / "ramc_core" / "test.tsv",
        directory / "SOURCE_NOTICE.txt",
        directory / "run_validation_linux.sh",
    ):
        if not required.is_file():
            errors.append(f"required file missing: {required.relative_to(directory)}")
    _verify_checksums(directory, errors)
    _verify_text_hygiene(directory, errors)
    durations = _verify_audio(directory, errors)
    core_tsv_names = _verify_core_layout(directory, durations, errors)
    strict_release = (
        expected_counts is not None
        and dict(expected_counts) == dict(RELEASE_COUNTS)
    )
    if strict_release:
        for relative, duration in durations.items():
            if relative.startswith("ramc_core/WAV/") and duration != 420_000:
                errors.append(
                    f"core duration must be 420000 ms: {relative}: {duration}"
                )
        required_seed_durations = {
            "seeds/format_fixture.wav": (30_000, 30_000),
            "seeds/stress_source.wav": (600_000, 600_000),
            "seeds/doctor_smoke.wav": (5_000, 10_000),
        }
        for relative, (minimum, maximum) in required_seed_durations.items():
            duration = durations.get(relative)
            if duration is not None and not minimum <= duration <= maximum:
                errors.append(
                    f"seed duration outside contract: {relative}: {duration}"
                )
    if isinstance(manifest, dict):
        _verify_path_hygiene((manifest,), errors)
        counts = manifest.get("counts") or {}
        if expected_counts is not None:
            for name, expected in expected_counts.items():
                if int(counts.get(name, -1)) != expected:
                    errors.append(
                        f"release count {name} must be {expected}, "
                        f"got {counts.get(name)}"
                    )
        core_count = len(tuple((directory / "ramc_core" / "WAV").glob("*.wav")))
        if int(counts.get("core", -1)) != core_count:
            errors.append("manifest core count mismatch")
        seeds = manifest.get("seeds") or {}
        if int(counts.get("seeds", -1)) != len(seeds):
            errors.append("manifest seed count mismatch")
        expected_seed_paths = {
            "format_fixture": "seeds/format_fixture.wav",
            "stress_source": "seeds/stress_source.wav",
            "doctor_smoke": "seeds/doctor_smoke.wav",
        }
        if set(seeds) != set(expected_seed_paths):
            errors.append("manifest seed names mismatch")
        if manifest.get("builder_version") != "1.0":
            errors.append("manifest builder version missing")
        if manifest.get("checksum_file") != CHECKSUM_FILE:
            errors.append("manifest checksum filename mismatch")
        if manifest.get("total_audio_duration_ms") != sum(durations.values()):
            errors.append("manifest total audio duration mismatch")
        core_items = manifest.get("core_items") or []
        if len(core_items) != len(core_tsv_names):
            errors.append("manifest core item count mismatch")
        declared_core_names: set[str] = set()
        tsv_metadata: dict[str, list[str]] = {}
        tsv_path = directory / "ramc_core" / "test.tsv"
        if tsv_path.is_file():
            for line in tsv_path.read_text(encoding="utf-8").splitlines():
                fields = line.split("\t")
                if len(fields) >= 5:
                    tsv_metadata[Path(fields[0].replace("\\", "/")).name] = fields[1:5]
        for item in core_items:
            wav_relative = str(item.get("wav_path") or "")
            txt_relative = str(item.get("transcript_path") or "")
            declared_core_names.add(Path(wav_relative).name)
            core_name = Path(wav_relative).name
            expected_tsv = [
                str(item.get("source_wav") or ""),
                *(str(value) for value in (item.get("speaker_ids") or [])),
                str(item.get("topic") or ""),
            ]
            if tsv_metadata.get(core_name) != expected_tsv:
                errors.append(f"core TSV metadata mismatch: {core_name}")
            _verify_declared_sha(
                directory, wav_relative, item.get("wav_sha256"), errors, wav_relative
            )
            _verify_declared_sha(
                directory,
                txt_relative,
                item.get("transcript_sha256"),
                errors,
                txt_relative,
            )
            if (
                item.get("source_start_ms") != 60_000
                or item.get("source_end_ms") != 480_000
            ):
                errors.append(f"core source interval invalid: {wav_relative}")
            transcript = directory / Path(txt_relative)
            if transcript.is_file():
                try:
                    rows = parse_transcript(transcript)
                    actual_speakers = {
                        row.speaker_id
                        for row in rows
                        if row.speaker_id != "G00000000"
                        and strip_control_markers(row.text)
                    }
                except ValueError:
                    actual_speakers = set()
                declared_speakers = set(item.get("speaker_ids") or [])
                if len(actual_speakers) != 2 or actual_speakers != declared_speakers:
                    errors.append(f"core speaker set mismatch: {txt_relative}")
        if declared_core_names != core_tsv_names:
            errors.append("manifest core items do not match TSV")
        for name, item in seeds.items():
            relative = str(item.get("audio_path") or "")
            if expected_seed_paths.get(name) != relative:
                errors.append(f"seed path mismatch: {name}: {relative}")
            if relative not in durations:
                errors.append(f"seed audio missing: {name}: {relative}")
                continue
            _verify_declared_sha(
                directory, relative, item.get("sha256"), errors, f"seed {name}"
            )
            if item.get("duration_ms") != durations[relative]:
                errors.append(f"seed duration metadata mismatch: {name}")
            if not isinstance(item.get("source"), dict):
                errors.append(f"seed source provenance missing: {name}")
    if isinstance(overlap, list):
        _verify_path_hygiene((overlap,), errors)
        kind_counts: dict[str, int] = {}
        seen_ids: set[str] = set()
        seen_audio: set[str] = set()
        for item in overlap:
            case_id = str(item.get("id") or "")
            if not case_id or case_id in seen_ids:
                errors.append(f"duplicate or empty overlap id: {case_id}")
            seen_ids.add(case_id)
            kind = str(item.get("kind") or "")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            relative = str(item.get("audio_path") or "")
            if not relative or relative in seen_audio:
                errors.append(f"duplicate or empty overlap audio path: {relative}")
            seen_audio.add(relative)
            if relative not in durations:
                errors.append(f"overlap audio missing: {item.get('id')}: {relative}")
                continue
            _verify_reference_turns(item, durations[relative], errors)
            _verify_overlap_semantics(item, durations[relative], errors)
            _verify_declared_sha(
                directory, relative, item.get("sha256"), errors, str(item.get("id"))
            )
            if kind in {"real_overlap", "speaker_change_negative"}:
                duration = durations.get(relative)
                if duration is not None and not 8_000 <= duration <= 20_000:
                    errors.append(f"overlap window duration outside contract: {case_id}")
        overlap_audio = {
            path.relative_to(directory).as_posix()
            for path in (directory / "overlap" / "audio").glob("*.wav")
        }
        if seen_audio != overlap_audio:
            errors.append("overlap manifest/audio mismatch")
        real_sources = {
            str(item.get("source_window", {}).get("wav_name") or "")
            for item in overlap
            if item.get("kind") == "real_overlap"
        }
        negative_sources = {
            str(item.get("source_window", {}).get("wav_name") or "")
            for item in overlap
            if item.get("kind") == "speaker_change_negative"
        }
        if (real_sources - {""}) & (negative_sources - {""}):
            errors.append("real and negative source recordings must be disjoint")
        if isinstance(manifest, dict):
            counts = manifest.get("counts") or {}
            for key, kind in (
                ("synthetic", "synthetic"),
                ("real_overlap", "real_overlap"),
                ("speaker_change_negative", "speaker_change_negative"),
            ):
                if int(counts.get(key, -1)) != kind_counts.get(kind, 0):
                    errors.append(f"manifest {key} count mismatch")
    return CorpusVerification(not errors, tuple(errors))

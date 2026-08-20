"""Audio and annotation primitives for portable validation corpora."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import uuid
import wave

from .corpus_verifier import verify_corpus, write_checksums
from .exporters import publish_output_directory
from .release_validation import (
    EVIDENCE_FIELDS,
    FIRERED_MODULES,
    STRESS_DURATIONS,
)
from .ramc_corpus import (
    CHINESE_CHARACTER,
    CORE_END_MS,
    CORE_START_MS,
    CoreCandidate,
    OverlapPlan,
    SyntheticMix,
    TranscriptRow,
    clean_clips,
    discover_core_candidates,
    select_core_candidates,
    select_overlap_material,
)


@dataclass(frozen=True, slots=True)
class AudioProperties:
    duration_ms: int
    sample_rate: int
    channels: int
    sample_width: int


@dataclass(frozen=True, slots=True)
class CorpusCounts:
    core: int = 20
    synthetic: int = 30
    real_overlap: int = 10
    negative: int = 10


@dataclass(frozen=True, slots=True)
class SeedMaterial:
    wav_path: Path
    start_ms: int
    end_ms: int
    reference_rows: tuple[TranscriptRow, ...]


@dataclass(frozen=True, slots=True)
class CorpusPlan:
    core: tuple[CoreCandidate, ...]
    overlap: OverlapPlan
    format_fixture: SeedMaterial
    stress_source: SeedMaterial
    doctor_smoke: SeedMaterial
    rejections: tuple[str, ...] = ()


def probe_wav(path: Path) -> AudioProperties:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as stream:
            frame_rate = stream.getframerate()
            if frame_rate <= 0:
                raise ValueError("frame rate must be positive")
            if stream.getcomptype() != "NONE":
                raise ValueError("WAV must contain uncompressed PCM")
            return AudioProperties(
                duration_ms=int(round(stream.getnframes() * 1000 / frame_rate)),
                sample_rate=frame_rate,
                channels=stream.getnchannels(),
                sample_width=stream.getsampwidth(),
            )
    except (OSError, wave.Error, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"unable to probe WAV {source}: {exc}") from exc


def clip_audio(
    source: Path,
    target: Path,
    start_ms: int,
    end_ms: int,
    ffmpeg: str = "ffmpeg",
) -> None:
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("audio clip range is invalid")
    input_path = Path(source)
    output_path = Path(target)
    if not input_path.is_file():
        raise FileNotFoundError(f"source audio does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{(end_ms - start_ms) / 1000:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(60, int((end_ms - start_ms) / 1000) * 2),
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"unable to create audio clip {output_path}: "
            f"{completed.stderr.strip()}"
        )
    properties = probe_wav(output_path)
    if (
        properties.sample_rate != 16_000
        or properties.channels != 1
        or properties.sample_width != 2
    ):
        raise RuntimeError(f"normalized clip has invalid properties: {properties}")


def shift_rows(
    rows: tuple[TranscriptRow, ...],
    source_start_ms: int,
    source_end_ms: int,
) -> tuple[TranscriptRow, ...]:
    if source_start_ms < 0 or source_end_ms <= source_start_ms:
        raise ValueError("transcript window is invalid")
    output: list[TranscriptRow] = []
    for row in sorted(rows, key=lambda item: (item.start_ms, item.end_ms)):
        start_ms = max(source_start_ms, row.start_ms)
        end_ms = min(source_end_ms, row.end_ms)
        if end_ms <= start_ms:
            continue
        output.append(
            TranscriptRow(
                start_ms=start_ms - source_start_ms,
                end_ms=end_ms - source_start_ms,
                speaker_id=row.speaker_id,
                gender=row.gender,
                language=row.language,
                text=row.text,
            )
        )
    return tuple(output)


def write_transcript(path: Path, rows: tuple[TranscriptRow, ...]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        (
            f"[{row.start_ms / 1000:.3f},{row.end_ms / 1000:.3f}]\t"
            f"{row.speaker_id}\t{row.gender},{row.language}\t{row.text}\n"
        )
        for row in rows
    )
    target.write_text(content, encoding="utf-8", newline="\n")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _intersects(row: TranscriptRow, start_ms: int, end_ms: int) -> bool:
    return min(row.end_ms, end_ms) > max(row.start_ms, start_ms)


def _seed_rows(
    candidate: CoreCandidate,
    start_ms: int,
    end_ms: int,
) -> tuple[TranscriptRow, ...]:
    return shift_rows(candidate.rows, start_ms, end_ms)


def _select_format_seed(
    candidates: tuple[CoreCandidate, ...],
    excluded: set[str],
) -> tuple[SeedMaterial, str]:
    for candidate in sorted(candidates, key=lambda item: item.wav_name):
        if candidate.wav_name in excluded:
            continue
        rows = sorted(candidate.rows, key=lambda item: (item.start_ms, item.end_ms))
        for row in rows:
            start_ms = max(60_000, row.start_ms)
            end_ms = start_ms + 30_000
            window = [item for item in rows if _intersects(item, start_ms, end_ms)]
            transitions = sum(
                left.speaker_id != right.speaker_id
                for left, right in zip(window, window[1:])
            )
            if transitions >= 2 and window:
                return (
                    SeedMaterial(
                        candidate.wav_path,
                        start_ms,
                        end_ms,
                        _seed_rows(candidate, start_ms, end_ms),
                    ),
                    candidate.wav_name,
                )
    raise ValueError("unable to find format fixture seed")


def _select_stress_seed(
    candidates: tuple[CoreCandidate, ...],
    excluded: set[str],
) -> tuple[SeedMaterial, str]:
    for candidate in sorted(candidates, key=lambda item: item.wav_name):
        if candidate.wav_name in excluded:
            continue
        if probe_wav(candidate.wav_path).duration_ms < 660_000:
            continue
        return (
            SeedMaterial(
                candidate.wav_path,
                60_000,
                660_000,
                _seed_rows(candidate, 60_000, 660_000),
            ),
            candidate.wav_name,
        )
    raise ValueError("unable to find stress source seed")


def _select_doctor_seed(
    candidates: tuple[CoreCandidate, ...],
    excluded: set[str],
) -> tuple[SeedMaterial, str]:
    for clip in clean_clips(candidates):
        if clip.wav_name in excluded:
            continue
        if clip.end_ms - clip.start_ms < 5_000:
            continue
        if len(CHINESE_CHARACTER.findall(clip.text)) < 15:
            continue
        row = TranscriptRow(
            clip.start_ms,
            clip.end_ms,
            clip.speaker_id,
            "未知",
            "普通话",
            clip.text,
        )
        return (
            SeedMaterial(
                clip.wav_path,
                clip.start_ms,
                clip.end_ms,
                shift_rows((row,), clip.start_ms, clip.end_ms),
            ),
            clip.wav_name,
        )
    raise ValueError("unable to find doctor smoke seed")


def create_corpus_plan(
    source_root: Path,
    counts: CorpusCounts,
    seed: int = 20260820,
) -> CorpusPlan:
    if seed != 20260820:
        raise ValueError("version 1 requires seed 20260820")
    rejections: list[str] = []
    candidates = discover_core_candidates(source_root, rejections)
    core = select_core_candidates(candidates, counts.core)
    overlap = select_overlap_material(
        candidates,
        counts.synthetic,
        counts.real_overlap,
        counts.negative,
    )
    excluded = {item.wav_name for item in core}
    for item in overlap.synthetic:
        excluded.update((item.first.wav_name, item.second.wav_name))
    excluded.update(item.wav_name for item in overlap.real)
    excluded.update(item.wav_name for item in overlap.negative)
    format_fixture, name = _select_format_seed(candidates, excluded)
    excluded.add(name)
    stress_source, name = _select_stress_seed(candidates, excluded)
    excluded.add(name)
    doctor_smoke, _ = _select_doctor_seed(candidates, excluded)
    return CorpusPlan(
        core, overlap, format_fixture, stress_source, doctor_smoke, tuple(rejections)
    )


def mix_synthetic(
    item: SyntheticMix,
    target: Path,
    work_dir: Path,
    ffmpeg: str = "ffmpeg",
) -> None:
    work = Path(work_dir) / item.case_id
    work.mkdir(parents=True, exist_ok=True)
    first_path = work / "first.wav"
    second_path = work / "second.wav"
    clip_audio(
        item.first.wav_path,
        first_path,
        item.first.start_ms,
        item.first.end_ms,
        ffmpeg,
    )
    clip_audio(
        item.second.wav_path,
        second_path,
        item.second.start_ms,
        item.second.end_ms,
        ffmpeg,
    )
    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    gain = math.pow(10.0, item.snr_db / 20.0)
    first_delay = item.first_offset_ms
    second_delay = item.second_offset_ms
    filter_graph = (
        f"[0:a]asetpts=PTS-STARTPTS,adelay={first_delay}|{first_delay}[a0];"
        f"[1:a]volume={gain:.8f},adelay={second_delay}|{second_delay}[a1];"
        "[a0][a1]amix=inputs=2:duration=longest:normalize=0[out]"
    )
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(first_path),
            "-i",
            str(second_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"unable to mix {item.case_id}: {completed.stderr.strip()}")


def _row_payload(row: TranscriptRow) -> dict:
    return asdict(row)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_validation_corpus_contents(
    source_root: Path,
    output_dir: Path,
    counts: CorpusCounts = CorpusCounts(),
    seed: int = 20260820,
    ffmpeg: str = "ffmpeg",
) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"validation corpus output exists: {output}")
    output.mkdir(parents=True)
    plan = create_corpus_plan(Path(source_root), counts, seed)

    core_rows = []
    core_items = []
    for item in plan.core:
        wav_target = output / "ramc_core" / "WAV" / item.wav_name
        txt_target = output / "ramc_core" / "TXT" / f"{Path(item.wav_name).stem}.txt"
        clip_audio(
            item.wav_path,
            wav_target,
            CORE_START_MS,
            CORE_END_MS,
            ffmpeg,
        )
        shifted = shift_rows(item.rows, CORE_START_MS, CORE_END_MS)
        write_transcript(txt_target, shifted)
        wav_relative = wav_target.relative_to(output).as_posix()
        txt_relative = txt_target.relative_to(output).as_posix()
        core_items.append(
            {
                "wav_path": wav_relative,
                "transcript_path": txt_relative,
                "source_wav": item.wav_name,
                "source_start_ms": CORE_START_MS,
                "source_end_ms": CORE_END_MS,
                "speaker_ids": list(item.speaker_ids),
                "topic": item.topic,
                "gender_pair": list(item.gender_pair),
                "regions": list(item.regions),
                "wav_sha256": _sha256(wav_target),
                "transcript_sha256": _sha256(txt_target),
            }
        )
        core_rows.append(
            "\t".join(
                (
                    f"WAV/{item.wav_name}",
                    item.wav_name,
                    item.speaker_ids[0],
                    item.speaker_ids[1],
                    item.topic,
                )
            )
        )
    core_tsv = output / "ramc_core" / "test.tsv"
    core_tsv.parent.mkdir(parents=True, exist_ok=True)
    core_tsv.write_text("\n".join(core_rows) + "\n", encoding="utf-8")

    overlap_items = []
    mix_work = output / ".mix-work"
    for item in plan.overlap.synthetic:
        relative = Path("overlap") / "audio" / f"{item.case_id}.wav"
        mix_synthetic(item, output / relative, mix_work, ffmpeg)
        first_end = item.first_offset_ms + item.first.end_ms - item.first.start_ms
        second_end = item.second_offset_ms + item.second.end_ms - item.second.start_ms
        overlap_items.append(
            {
                "id": item.case_id,
                "kind": "synthetic",
                "audio_path": relative.as_posix(),
                "snr_db": item.snr_db,
                "overlap_ratio": item.overlap_ratio,
                "first_offset_ms": item.first_offset_ms,
                "second_offset_ms": item.second_offset_ms,
                "linear_gain": round(math.pow(10.0, item.snr_db / 20.0), 8),
                "sha256": _sha256(output / relative),
                "sources": [
                    {
                        "role": "first",
                        "wav_name": item.first.wav_name,
                        "start_ms": item.first.start_ms,
                        "end_ms": item.first.end_ms,
                        "speaker": item.first.speaker_id,
                        "text": item.first.text,
                        "output_start_ms": item.first_offset_ms,
                    },
                    {
                        "role": "second",
                        "wav_name": item.second.wav_name,
                        "start_ms": item.second.start_ms,
                        "end_ms": item.second.end_ms,
                        "speaker": item.second.speaker_id,
                        "text": item.second.text,
                        "output_start_ms": item.second_offset_ms,
                    },
                ],
                "reference_turns": [
                    {
                        "speaker": item.first.speaker_id,
                        "start_ms": item.first_offset_ms,
                        "end_ms": first_end,
                        "text": item.first.text,
                    },
                    {
                        "speaker": item.second.speaker_id,
                        "start_ms": item.second_offset_ms,
                        "end_ms": second_end,
                        "text": item.second.text,
                    },
                ],
            }
        )
    for item in (*plan.overlap.real, *plan.overlap.negative):
        relative = Path("overlap") / "audio" / f"{item.case_id}.wav"
        clip_audio(item.wav_path, output / relative, item.start_ms, item.end_ms, ffmpeg)
        overlap_items.append(
            {
                "id": item.case_id,
                "kind": item.kind,
                "audio_path": relative.as_posix(),
                "gap_ms": item.gap_ms,
                "sha256": _sha256(output / relative),
                "source_window": {
                    "wav_name": item.wav_name,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                },
                "marker_evidence": (
                    {
                        "source_wav": item.wav_name,
                        "source_start_ms": item.source_marker.start_ms,
                        "source_end_ms": item.source_marker.end_ms,
                        "source_speaker": item.source_marker.speaker_id,
                        "marker": "[+]",
                    }
                    if item.source_marker is not None
                    else None
                ),
                "timing_reference": (
                    "unavailable" if item.kind == "real_overlap" else "serialized"
                ),
                "reference_turns": [_row_payload(row) for row in item.reference_rows],
            }
        )
    shutil.rmtree(mix_work, ignore_errors=True)
    _write_json(output / "overlap" / "manifest.json", overlap_items)

    seed_payload = {}
    for name, material in (
        ("format_fixture", plan.format_fixture),
        ("stress_source", plan.stress_source),
        ("doctor_smoke", plan.doctor_smoke),
    ):
        relative = Path("seeds") / f"{name}.wav"
        clip_audio(
            material.wav_path,
            output / relative,
            material.start_ms,
            material.end_ms,
            ffmpeg,
        )
        seed_payload[name] = {
            "audio_path": relative.as_posix(),
            "sha256": _sha256(output / relative),
            "duration_ms": probe_wav(output / relative).duration_ms,
            "source": {
                "wav_name": material.wav_path.name,
                "start_ms": material.start_ms,
                "end_ms": material.end_ms,
            },
            "reference_turns": [_row_payload(row) for row in material.reference_rows],
        }

    notice_source = Path(source_root) / "README.txt"
    notice = (
        notice_source.read_text(encoding="utf-8")
        if notice_source.is_file()
        else "MagicData-RAMC MDT2021S003 validation excerpts\n"
    )
    (output / "SOURCE_NOTICE.txt").write_text(notice, encoding="utf-8")
    runner_source = Path(__file__).resolve().parents[2] / "scripts" / "run_validation_linux.sh"
    runner_target = output / "run_validation_linux.sh"
    if not runner_source.is_file():
        raise FileNotFoundError(f"Linux validation runner is missing: {runner_source}")
    shutil.copy2(runner_source, runner_target)
    runner_target.chmod(0o755)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "1.0",
            "builder_version": "1.0",
            "source_dataset": "MagicData-RAMC/MDT2021S003",
            "seed": seed,
            "audio": {
                "sample_rate": 16_000,
                "channels": 1,
                "sample_width": 2,
            },
            "counts": {
                "core": len(plan.core),
                "synthetic": len(plan.overlap.synthetic),
                "real_overlap": len(plan.overlap.real),
                "speaker_change_negative": len(plan.overlap.negative),
                "seeds": 3,
            },
            "total_audio_duration_ms": sum(
                probe_wav(path).duration_ms for path in output.rglob("*.wav")
            ),
            "checksum_file": "checksums.sha256",
            "release_evidence": {
                "schema_version": "1.0",
                "required": list(EVIDENCE_FIELDS),
                "firered_modules": list(FIRERED_MODULES),
                "stress_durations_minutes": [
                    int(value) for value in STRESS_DURATIONS
                ],
            },
            "core_test_tsv": "ramc_core/test.tsv",
            "core_items": core_items,
            "core_distribution": {
                "topics": sorted({item.topic for item in plan.core}),
                "gender_pairs": sorted({"/".join(item.gender_pair) for item in plan.core}),
                "regions": sorted({region for item in plan.core for region in item.regions}),
            },
            "overlap_manifest": "overlap/manifest.json",
            "seeds": seed_payload,
            "rejections": {
                "count": len(plan.rejections),
                "items": list(plan.rejections),
            },
        },
    )
    return output


def build_validation_corpus(
    source_root: Path,
    output_dir: Path,
    counts: CorpusCounts = CorpusCounts(),
    seed: int = 20260820,
    ffmpeg: str = "ffmpeg",
    overwrite: bool = False,
) -> Path:
    target = Path(output_dir)
    if target.exists() and not overwrite:
        raise FileExistsError(f"validation corpus output exists: {target}")
    staging = target.with_name(
        f".{target.name}.building-{uuid.uuid4().hex}"
    )
    try:
        _build_validation_corpus_contents(
            source_root,
            staging,
            counts=counts,
            seed=seed,
            ffmpeg=ffmpeg,
        )
        write_checksums(staging)
        verification = verify_corpus(
            staging,
            expected_counts={
                "core": counts.core,
                "synthetic": counts.synthetic,
                "real_overlap": counts.real_overlap,
                "speaker_change_negative": counts.negative,
                "seeds": 3,
            },
        )
        if not verification.ok:
            raise ValueError(
                "validation corpus verification failed: "
                + "; ".join(verification.errors)
            )
        publish_output_directory(staging, target, overwrite=overwrite)
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)

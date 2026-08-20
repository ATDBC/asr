import json
from pathlib import Path

import pytest
import funasr_transcriber.exporters as exporters

from funasr_transcriber.audio import AudioInfo
from funasr_transcriber.exporters import (
    export_transcript,
    publish_output_directory,
)
from funasr_transcriber.schemas import AlignmentConfidence, Transcript, Turn


def sample_transcript():
    return Transcript(
        language="zh",
        speaker_count=1,
        turns=(
            Turn(
                index=1,
                speaker_id=1,
                start_ms=3_723_000,
                end_ms=3_724_000,
                text="超过一小时。",
                source="primary_asr",
                confidence=AlignmentConfidence.HIGH,
                overlap=False,
            ),
        ),
        warnings=(),
    )


def test_export_uses_hh_mm_ss_and_utf8(tmp_path: Path):
    export_transcript(sample_transcript(), tmp_path, overwrite=False)

    assert (tmp_path / "transcript.txt").read_text(encoding="utf-8") == (
        "[01:02:03] 说话人1：超过一小时。\n"
    )
    payload = json.loads(
        (tmp_path / "transcript.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "success"
    assert payload["turns"][0]["start_ms"] == 3_723_000
    assert payload["full_text"] == "超过一小时。"


def test_export_refuses_existing_result(tmp_path: Path):
    (tmp_path / "transcript.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_transcript(sample_transcript(), tmp_path, overwrite=False)

    assert (tmp_path / "transcript.txt").read_text(encoding="utf-8") == "existing"
    assert not (tmp_path / "transcript.json").exists()


def test_repeated_exports_are_byte_identical(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    export_transcript(sample_transcript(), first, overwrite=False)
    export_transcript(sample_transcript(), second, overwrite=False)

    assert (first / "transcript.txt").read_bytes() == (
        second / "transcript.txt"
    ).read_bytes()
    assert (first / "transcript.json").read_bytes() == (
        second / "transcript.json"
    ).read_bytes()


def test_export_includes_input_audio_metadata(tmp_path: Path):
    info = AudioInfo(Path("/data/call.mp3"), 12_345, 44_100, 2, "mp3")

    export_transcript(
        sample_transcript(),
        tmp_path,
        overwrite=False,
        input_info=info,
    )

    payload = json.loads((tmp_path / "transcript.json").read_text("utf-8"))
    assert payload["input"] == {
        "path": str(Path("/data/call.mp3")),
        "format": "mp3",
        "duration_ms": 12_345,
        "sample_rate": 44_100,
        "channels": 2,
    }


def test_publish_rejects_any_existing_output_directory(tmp_path: Path):
    staging = tmp_path / "staging"
    target = tmp_path / "result"
    staging.mkdir()
    target.mkdir()
    (staging / "transcript.txt").write_text("new", encoding="utf-8")

    with pytest.raises(FileExistsError):
        publish_output_directory(staging, target, overwrite=False)

    assert staging.exists()
    assert target.exists()


def test_publish_does_not_report_failure_after_new_output_is_visible(
    monkeypatch,
    tmp_path: Path,
):
    staging = tmp_path / "staging"
    target = tmp_path / "result"
    staging.mkdir()
    target.mkdir()
    (staging / "transcript.txt").write_text("new", encoding="utf-8")
    (target / "transcript.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        exporters.shutil,
        "rmtree",
        lambda path: (_ for _ in ()).throw(OSError("cleanup failed")),
    )

    publish_output_directory(staging, target, overwrite=True)

    assert (target / "transcript.txt").read_text("utf-8") == "new"

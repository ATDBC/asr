#!/usr/bin/env bash
set -euo pipefail

CORPUS_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=""
CONFIG=""
SKIP_STRESS=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --skip-stress) SKIP_STRESS=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$PROJECT_ROOT" ] || [ -z "$CONFIG" ]; then
    echo "Usage: $0 --project-root PATH --config PATH [--skip-stress]" >&2
    exit 2
fi

CORE_PYTHON="$PROJECT_ROOT/.venv/bin/python"
ASR_PYTHON="$PROJECT_ROOT/.venv-asr/bin/python"
TRANSCRIBE_BIN="$PROJECT_ROOT/.venv/bin/funasr-transcribe"
RESULTS_ROOT="$CORPUS_ROOT/validation_results"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RESULTS="$RESULTS_ROOT/$RUN_ID"
SUMMARY="$RESULTS/summary.json"
SUMMARY_TEXT="$RESULTS/summary.txt"
LATEST_SUMMARY="$RESULTS_ROOT/summary.json"
LATEST_SUMMARY_TEXT="$RESULTS_ROOT/summary.txt"
CURRENT_STAGE="startup"
SERVICE_STARTED=0
mkdir -p "$RESULTS"

finish() {
    local status="$1"
    if [ "$SERVICE_STARTED" -eq 1 ]; then
        "$TRANSCRIBE_BIN" service stop --config "$CONFIG" \
            > "$RESULTS/service-stop.json" 2>&1 || true
    fi
    "$CORE_PYTHON" - "$SUMMARY" "$SUMMARY_TEXT" "$LATEST_SUMMARY" \
        "$LATEST_SUMMARY_TEXT" "$status" "$CURRENT_STAGE" "$SKIP_STRESS" <<'PY'
import json, pathlib, sys
from funasr_transcriber.release_validation import (
    FIRERED_MODULES,
    build_release_summary,
)
path = pathlib.Path(sys.argv[1])
text_path = pathlib.Path(sys.argv[2])
latest_path = pathlib.Path(sys.argv[3])
latest_text_path = pathlib.Path(sys.argv[4])
status = int(sys.argv[5])
stress_skipped = bool(int(sys.argv[7]))
stage_results = {}
for candidate in sorted(path.parent.rglob("*-summary.json")):
    if candidate == path:
        continue
    try:
        stage_results[candidate.relative_to(path.parent).as_posix()] = json.loads(
            candidate.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        stage_results[candidate.relative_to(path.parent).as_posix()] = {
            "read_error": str(exc)
        }
doctor = next(
    (
        value
        for key, value in stage_results.items()
        if key.endswith("doctor-summary.json") and isinstance(value, dict)
    ),
    {},
)
doctor_evidence = doctor.get("release_evidence") or {}
doctor_modules = doctor_evidence.get("firered_modules") or {}
stress = next(
    (
        value
        for key, value in stage_results.items()
        if key.endswith("stress-summary.json") and isinstance(value, dict)
    ),
    {},
)
deterministic_hashes = {
    str(item.get("minutes")): item.get("result_hashes")
    for item in stress.get("results", [])
    if isinstance(item, dict)
}
release_evidence = {
    "firered_modules": {
        name: doctor_modules.get(name) for name in FIRERED_MODULES
    },
    "lid_coverage": doctor_evidence.get("lid_coverage"),
    "timestamp_integrity": doctor_evidence.get("timestamp_integrity"),
    "gpu_stages": doctor_evidence.get("gpu_stages"),
    "deterministic_hashes": deterministic_hashes,
}
payload = build_release_summary(
    exit_code=status,
    stress_skipped=stress_skipped,
    last_stage=sys.argv[6],
    stage_results=stage_results,
    release_evidence=release_evidence,
)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = [
    f"passed: {payload['passed']}",
    f"complete: {payload['complete']}",
    f"exit_code: {status}",
    f"last_stage: {payload['last_stage']}",
    f"stage_result_files: {len(stage_results)}",
    f"missing_evidence: {','.join(payload['missing_evidence']) or 'none'}",
]
text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
latest_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
latest_text_path.write_text(text_path.read_text(encoding="utf-8"), encoding="utf-8")
PY
}
trap 'finish $?' EXIT

cd "$CORPUS_ROOT"
if [ ! -x "$ASR_PYTHON" ]; then
    echo "FireRed ASR Python is missing: $ASR_PYTHON" >&2
    exit 4
fi
"$CORE_PYTHON" -c "import pytest"
"$CORE_PYTHON" - "$CONFIG" "$PROJECT_ROOT" <<'PY'
import pathlib, sys
from funasr_transcriber.config import load_config

config = load_config(pathlib.Path(sys.argv[1]))
expected_python = pathlib.Path(sys.argv[2]) / ".venv-asr" / "bin" / "python"
if config.asr.worker_python.resolve() != expected_python.resolve():
    raise SystemExit(
        f"config ASR worker must be {expected_python}: {config.asr.worker_python}"
    )
root = config.paths.model_dir.resolve() / "fireredasr2s"
required = {
    "FireRedVAD": root / "FireRedVAD" / "VAD" / "model.pth.tar",
    "FireRedLID": root / "FireRedLID" / "model.pth.tar",
    "FireRedASR2-AED": root / "FireRedASR2-AED" / "model.pth.tar",
    "FireRedPunc": root / "FireRedPunc" / "model.pth.tar",
}
missing = [name for name, path in required.items() if not path.is_file()]
if missing:
    raise SystemExit("missing FireRed models: " + ", ".join(missing))
PY
CURRENT_STAGE="checksums"
sha256sum -c checksums.sha256

CURRENT_STAGE="models"
"$TRANSCRIBE_BIN" models verify --config "$CONFIG"

CURRENT_STAGE="doctor"
"$TRANSCRIBE_BIN" doctor --config "$CONFIG" | tee "$RESULTS/doctor-checks.txt"
SERVICE_STARTED=1
"$TRANSCRIBE_BIN" "$CORPUS_ROOT/seeds/doctor_smoke.wav" \
    --config "$CONFIG" \
    --output-dir "$RESULTS/doctor-smoke" \
    --disable-overlap \
    --profile-resources \
    --overwrite
"$CORE_PYTHON" - "$RESULTS/doctor-smoke/transcript.json" \
    "$RESULTS/doctor-smoke/diagnostics.json" \
    "$RESULTS/doctor-checks.txt" "$RESULTS/doctor-summary.json" <<'PY'
import hashlib, json, pathlib, sys
transcript = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
diagnostics = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
profile = diagnostics.get("resource_profile") or {}
checks = pathlib.Path(sys.argv[3]).read_text(encoding="utf-8")
turns = transcript.get("turns") or []
sentences = (diagnostics.get("asr") or {}).get("sentences") or []
covered = sum(
    bool(item.get("language")) and item.get("language_confidence") is not None
    for item in sentences
)
ranges = [
    (int(item.get("start_ms", -1)), int(item.get("end_ms", -1)))
    for item in [*turns, *sentences]
]
valid_ranges = sum(0 <= start < end for start, end in ranges)
module_check = all(
    marker in checks
    for marker in (
        "firered_source: verified",
        "firered_models: verified",
        "firered_worker: verified",
    )
)
release_evidence = {
    "firered_modules": {
        "vad": module_check,
        "lid": module_check,
        "aed": module_check,
        "punc": module_check,
    },
    "lid_coverage": {
        "sentences": len(sentences),
        "covered": covered,
        "ratio": covered / len(sentences) if sentences else 0.0,
    },
    "timestamp_integrity": {
        "checked": len(ranges),
        "valid": valid_ranges,
        "passed": bool(ranges) and valid_ranges == len(ranges),
    },
    "gpu_stages": diagnostics.get("gpu_stages") or {},
}
payload = {
    "passed": bool(turns) and all(release_evidence["firered_modules"].values())
    and release_evidence["lid_coverage"]["ratio"] == 1.0
    and release_evidence["timestamp_integrity"]["passed"]
    and bool(release_evidence["gpu_stages"])
    and all(
        item.get("released") is True
        for item in release_evidence["gpu_stages"].values()
    ),
    "speaker_count": transcript.get("speaker_count"),
    "warnings": transcript.get("warnings", []),
    "transcript_sha256": hashlib.sha256(
        pathlib.Path(sys.argv[1]).read_bytes()
    ).hexdigest(),
    "release_evidence": release_evidence,
}
if profile.get("enabled") is not True:
    raise SystemExit("service profile was not written")
unknown_owners = set((profile.get("process_peak_mib") or {})) - {
    "service", "firered", "overlap"
}
if unknown_owners:
    raise SystemExit(f"profile exposed unexpected owners: {sorted(unknown_owners)}")
pathlib.Path(sys.argv[4]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
if not payload["passed"]:
    raise SystemExit("doctor smoke release evidence is incomplete")
PY

CURRENT_STAGE="service_reuse"
"$TRANSCRIBE_BIN" "$CORPUS_ROOT/seeds/doctor_smoke.wav" \
    --config "$CONFIG" \
    --output-dir "$RESULTS/doctor-smoke-reuse" \
    --disable-overlap \
    --overwrite
"$TRANSCRIBE_BIN" service status --config "$CONFIG" \
    > "$RESULTS/service-status.json"

export FUNASR_TRANSCRIBER_RUN_BENCHMARKS=1
export FUNASR_TRANSCRIBER_RAMC_ROOT="$CORPUS_ROOT/ramc_core"
export FUNASR_TRANSCRIBER_RAMC_TSV="$CORPUS_ROOT/ramc_core/test.tsv"
export FUNASR_TRANSCRIBER_OVERLAP_MANIFEST="$CORPUS_ROOT/overlap/manifest.json"
export FUNASR_TRANSCRIBER_FORMAT_FIXTURE="$CORPUS_ROOT/seeds/format_fixture.wav"
export FUNASR_TRANSCRIBER_STRESS_SOURCE="$CORPUS_ROOT/seeds/stress_source.wav"
export FUNASR_TRANSCRIBER_CONFIG="$CONFIG"
export FUNASR_TRANSCRIBE_BIN="$TRANSCRIBE_BIN"

CURRENT_STAGE="ramc"
"$CORE_PYTHON" -m pytest "$PROJECT_ROOT/tests/benchmark/test_ramc_gate.py" -q \
    --basetemp "$RESULTS/pytest-ramc"
CURRENT_STAGE="overlap"
"$CORE_PYTHON" -m pytest "$PROJECT_ROOT/tests/benchmark/test_overlap_gate.py" -q \
    --basetemp "$RESULTS/pytest-overlap"
CURRENT_STAGE="input_matrix"
"$CORE_PYTHON" -m pytest "$PROJECT_ROOT/tests/benchmark/test_input_matrix.py" -q \
    --basetemp "$RESULTS/pytest-input-matrix"
if [ "$SKIP_STRESS" -eq 0 ]; then
    CURRENT_STAGE="long_audio"
    "$CORE_PYTHON" -m pytest "$PROJECT_ROOT/tests/benchmark/test_long_audio.py" -q \
        --basetemp "$RESULTS/pytest-long-audio"
fi
CURRENT_STAGE="complete"

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
TRANSCRIBE_BIN="$PROJECT_ROOT/.venv/bin/funasr-transcribe"
RESULTS_ROOT="$CORPUS_ROOT/validation_results"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RESULTS="$RESULTS_ROOT/$RUN_ID"
SUMMARY="$RESULTS/summary.json"
SUMMARY_TEXT="$RESULTS/summary.txt"
LATEST_SUMMARY="$RESULTS_ROOT/summary.json"
LATEST_SUMMARY_TEXT="$RESULTS_ROOT/summary.txt"
CURRENT_STAGE="startup"
mkdir -p "$RESULTS"

finish() {
    local status="$1"
    "$CORE_PYTHON" - "$SUMMARY" "$SUMMARY_TEXT" "$LATEST_SUMMARY" \
        "$LATEST_SUMMARY_TEXT" "$status" "$CURRENT_STAGE" "$SKIP_STRESS" <<'PY'
import json, pathlib, sys
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
payload = {
    "passed": status == 0 and not stress_skipped,
    "complete": status == 0 and not stress_skipped,
    "stress_skipped": stress_skipped,
    "exit_code": status,
    "last_stage": sys.argv[6],
    "stage_results": stage_results,
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = [
    f"passed: {payload['passed']}",
    f"exit_code: {status}",
    f"last_stage: {payload['last_stage']}",
    f"stage_result_files: {len(stage_results)}",
]
text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
latest_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
latest_text_path.write_text(text_path.read_text(encoding="utf-8"), encoding="utf-8")
PY
}
trap 'finish $?' EXIT

cd "$CORPUS_ROOT"
"$CORE_PYTHON" -c "import pytest"
CURRENT_STAGE="checksums"
sha256sum -c checksums.sha256

CURRENT_STAGE="models"
"$TRANSCRIBE_BIN" models verify --config "$CONFIG"

CURRENT_STAGE="doctor"
"$TRANSCRIBE_BIN" doctor --config "$CONFIG"
"$TRANSCRIBE_BIN" "$CORPUS_ROOT/seeds/doctor_smoke.wav" \
    --config "$CONFIG" \
    --output-dir "$RESULTS/doctor-smoke" \
    --disable-overlap \
    --overwrite
"$CORE_PYTHON" - "$RESULTS/doctor-smoke/transcript.json" \
    "$RESULTS/doctor-summary.json" <<'PY'
import json, pathlib, sys
transcript = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "passed": bool(transcript.get("turns")),
    "speaker_count": transcript.get("speaker_count"),
    "warnings": transcript.get("warnings", []),
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if not transcript.get("turns"):
    raise SystemExit("doctor smoke transcript has no turns")
PY

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

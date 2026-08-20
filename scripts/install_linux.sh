#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CORE_VENV="$PROJECT_ROOT/.venv"
ASR_VENV="$PROJECT_ROOT/.venv-asr"
OVERLAP_VENV="$PROJECT_ROOT/.venv-overlap"
FIRERED_COMMIT="4e7d9aaf4482a47cec1724807026b9b151926eb5"
FIRERED_REPOSITORY="$(printenv FIRERED_REPOSITORY || true)"
if [ -z "$FIRERED_REPOSITORY" ]; then
    FIRERED_REPOSITORY="https://github.com/FireRedTeam/FireRedASR2S.git"
fi
MODEL_ROOT="$(printenv FUNASR_TRANSCRIBER_MODEL_ROOT || true)"
if [ -z "$MODEL_ROOT" ]; then
    MODEL_ROOT="$PROJECT_ROOT/models"
fi
FIRERED_ROOT="$MODEL_ROOT/fireredasr2s"
FIRERED_SOURCE="$FIRERED_ROOT/source"
TORCH_INDEX_URL="$(printenv TORCH_INDEX_URL || true)"
if [ -z "$TORCH_INDEX_URL" ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
fi

PYTHON_BIN="$(printenv PYTHON_BIN || true)"
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3.11 || true)"
fi
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
    echo "Python 3.11 is missing; set PYTHON_BIN to its executable" >&2
    exit 4
fi
"$PYTHON_BIN" -c \
    "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

for command in ffmpeg ffprobe nvidia-smi git; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Missing required command: $command" >&2
        exit 4
    fi
done

"$PYTHON_BIN" -m venv "$CORE_VENV"
"$PYTHON_BIN" -m venv "$ASR_VENV"
"$PYTHON_BIN" -m venv "$OVERLAP_VENV"

CORE_PYTHON="$CORE_VENV/bin/python"
ASR_PYTHON="$ASR_VENV/bin/python"
OVERLAP_PYTHON="$OVERLAP_VENV/bin/python"

mkdir -p "$FIRERED_ROOT"
if [ -e "$FIRERED_SOURCE" ]; then
    if [ ! -d "$FIRERED_SOURCE/.git" ]; then
        echo "Existing FireRed source is not a Git checkout: $FIRERED_SOURCE" >&2
        exit 4
    fi
    SOURCE_HEAD="$(git -C "$FIRERED_SOURCE" rev-parse HEAD)"
    if [ "$SOURCE_HEAD" != "$FIRERED_COMMIT" ]; then
        echo "Existing FireRed source has wrong commit: $SOURCE_HEAD" >&2
        exit 4
    fi
else
    SOURCE_TMP="$(mktemp -d "$FIRERED_ROOT/.source.download-XXXXXX")"
    cleanup_source_tmp() {
        if [ -n "$SOURCE_TMP" ] && [ -d "$SOURCE_TMP" ]; then
            case "$SOURCE_TMP" in
                "$FIRERED_ROOT"/.source.download-*)
                    rm -rf -- "$SOURCE_TMP"
                    ;;
            esac
        fi
    }
    trap cleanup_source_tmp EXIT
    git clone "$FIRERED_REPOSITORY" "$SOURCE_TMP"
    git -C "$SOURCE_TMP" checkout --detach "$FIRERED_COMMIT"
    SOURCE_HEAD="$(git -C "$SOURCE_TMP" rev-parse HEAD)"
    if [ "$SOURCE_HEAD" != "$FIRERED_COMMIT" ]; then
        echo "Downloaded FireRed source has wrong commit: $SOURCE_HEAD" >&2
        exit 4
    fi
    mv "$SOURCE_TMP" "$FIRERED_SOURCE"
    SOURCE_TMP=""
    trap - EXIT
fi

"$CORE_PYTHON" -m pip install --upgrade \
    "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
"$CORE_PYTHON" -m pip install \
    --index-url "$TORCH_INDEX_URL" \
    "torch==2.11.0" "torchaudio==2.11.0"
"$CORE_PYTHON" -m pip install -e "$PROJECT_ROOT[dev]"

"$ASR_PYTHON" -m pip install --upgrade \
    "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
"$ASR_PYTHON" -m pip install -e "$FIRERED_SOURCE"
"$ASR_PYTHON" -m pip install \
    -r "$PROJECT_ROOT/requirements-asr.txt"
"$ASR_PYTHON" -m pip install --no-deps -e "$PROJECT_ROOT"

"$OVERLAP_PYTHON" -m pip install --upgrade \
    "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
"$OVERLAP_PYTHON" -m pip install \
    --index-url "$TORCH_INDEX_URL" \
    "torch==2.8.0" "torchaudio==2.8.0"
"$OVERLAP_PYTHON" -m pip install \
    --extra-index-url "$TORCH_INDEX_URL" \
    -r "$PROJECT_ROOT/requirements-overlap.txt"

"$CORE_PYTHON" -c \
    "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'"
"$ASR_PYTHON" -c \
    "import torch; assert torch.cuda.is_available(), 'ASR CUDA is unavailable'"
"$OVERLAP_PYTHON" -c \
    "import torch, torchaudio, torchvision, pyannote.audio, clearvoice; assert torch.cuda.is_available(), 'overlap CUDA is unavailable'"

echo "Installation complete."
echo "Activate with: source $CORE_VENV/bin/activate"
echo "Next: funasr-transcribe models download"
echo "Then: funasr-transcribe models verify"
echo "Finally: funasr-transcribe doctor"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NEMOTRON_VENV="$PROJECT_ROOT/.venv-nemotron"
TRANSFORMERS_REVISION="f324707307757d9c0b8dac1c4462eceff911fa2f"

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

TORCH_INDEX_URL="$(printenv TORCH_INDEX_URL || true)"
if [ -z "$TORCH_INDEX_URL" ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
fi

"$PYTHON_BIN" -m venv "$NEMOTRON_VENV"
NEMOTRON_PYTHON="$NEMOTRON_VENV/bin/python"
"$NEMOTRON_PYTHON" -m pip install --upgrade \
    "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
"$NEMOTRON_PYTHON" -m pip install \
    --index-url "$TORCH_INDEX_URL" \
    "torch==2.8.0" "torchaudio==2.8.0"
"$NEMOTRON_PYTHON" -m pip install \
    -r "$PROJECT_ROOT/requirements-nemotron.txt"
"$NEMOTRON_PYTHON" -m pip install \
    -c "$PROJECT_ROOT/requirements-nemotron.txt" \
    "transformers @ git+https://github.com/huggingface/transformers.git@$TRANSFORMERS_REVISION"
"$NEMOTRON_PYTHON" -m pip install --no-deps -e "$PROJECT_ROOT"
"$NEMOTRON_PYTHON" -c \
    "import torch, torchaudio, transformers; assert torch.cuda.is_available(), 'Nemotron CUDA is unavailable'"

echo "Optional Nemotron environment installed at $NEMOTRON_VENV"
echo "Select [diarization] backend = 'nemotron3', then run models download and models verify."

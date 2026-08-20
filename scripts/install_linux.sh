#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CORE_VENV="$PROJECT_ROOT/.venv"
OVERLAP_VENV="$PROJECT_ROOT/.venv-overlap"
TORCH_INDEX_URL="$(printenv TORCH_INDEX_URL || true)"
if [ -z "$TORCH_INDEX_URL" ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
fi

for command in python3.11 ffmpeg ffprobe nvidia-smi git; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Missing required command: $command" >&2
        exit 4
    fi
done

python3.11 -m venv "$CORE_VENV"
python3.11 -m venv "$OVERLAP_VENV"

CORE_PYTHON="$CORE_VENV/bin/python"
OVERLAP_PYTHON="$OVERLAP_VENV/bin/python"

"$CORE_PYTHON" -m pip install --upgrade \
    "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
"$CORE_PYTHON" -m pip install \
    --index-url "$TORCH_INDEX_URL" \
    "torch==2.11.0" "torchaudio==2.11.0"
"$CORE_PYTHON" -m pip install -e "$PROJECT_ROOT[dev]"

"$OVERLAP_PYTHON" -m pip install --upgrade \
    "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
"$OVERLAP_PYTHON" -m pip install \
    --index-url "$TORCH_INDEX_URL" \
    "torch==2.11.0" "torchaudio==2.11.0"
"$OVERLAP_PYTHON" -m pip install \
    --extra-index-url "$TORCH_INDEX_URL" \
    -r "$PROJECT_ROOT/requirements-overlap.txt"
"$OVERLAP_PYTHON" -m pip install --no-deps -e "$PROJECT_ROOT"

"$CORE_PYTHON" -c \
    "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'"
"$OVERLAP_PYTHON" -c \
    "import torch; assert torch.cuda.is_available(), 'overlap CUDA is unavailable'"

echo "Installation complete."
echo "Activate with: source $CORE_VENV/bin/activate"
echo "Next: funasr-transcribe models download"
echo "Then: funasr-transcribe models verify"
echo "Finally: funasr-transcribe doctor"

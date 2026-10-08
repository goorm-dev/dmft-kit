#!/usr/bin/env bash
# Install dependencies, picking the torch build for this machine
#   scripts/setup.sh              NVIDIA GPU (nvidia-smi) -> CUDA torch (gpu), otherwise CPU torch (cpu; includes MPS on Apple Silicon)
#   scripts/setup.sh --torch cpu  force a build
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=".venv-$(uname -s | tr '[:upper:]' '[:lower:]')"
cd "$(dirname "$0")/.."

torch=auto
while [ $# -gt 0 ]; do
    case "$1" in
        --torch) torch="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ "$torch" = auto ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        torch=gpu; echo "Detected: $(nvidia-smi -L | head -1) -> torch build 'gpu'"
    else
        torch=cpu; echo "Detected: no NVIDIA GPU -> torch build 'cpu'"
    fi
fi
case "$torch" in gpu|cpu) ;; *) echo "--torch must be auto, gpu or cpu" >&2; exit 2 ;; esac

uv sync --extra "$torch"
uv run --no-sync python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| mps:', torch.backends.mps.is_available())"

# Shared setup, sourced by every script:  . "$(dirname "$0")/_env.sh"
set -euo pipefail
export PYTHONUTF8=1
# `uv run` would re-sync to the default (no extra) and drop the gpu/cpu torch chosen by setup.sh
export UV_NO_SYNC=1
# Per-OS venv, so WSL and Windows can share one checkout (Windows uses .venv)
export UV_PROJECT_ENVIRONMENT=".venv-$(uname -s | tr '[:upper:]' '[:lower:]')"
cd "$(dirname "$0")/.."

# .env (KEY=VALUE) -> environment; values already set in the shell win
if [ -f .env ]; then
    while IFS='=' read -r key value || [ -n "$key" ]; do
        key=$(echo "$key" | tr -d '[:space:]')
        case "$key" in ''|\#*) continue ;; esac
        value=$(echo "$value" | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e "s/^[\"']//" -e "s/[\"']$//")
        if [ -z "${!key:-}" ]; then export "$key=$value"; fi
    done < .env
fi

if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
    echo "No $UV_PROJECT_ENVIRONMENT: run scripts/setup.sh first" >&2
    exit 2
fi

# GPU by default (CUDA, then Apple MPS); CPU when forced (arg 1) or when neither is usable
resolve_device() {
    if [ "${1:-0}" = 1 ]; then echo cpu; return; fi
    local dev
    dev=$(uv run python -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')" 2>/dev/null) || dev=cpu
    if [ "$dev" = cpu ]; then echo "CUDA/MPS not available -> falling back to CPU" >&2; fi
    echo "$dev"
}

#!/usr/bin/env bash
# Run script.py against the local vLLM build.
#
# Wraps ./vllm-env/bin/python so the venv, CUDA paths, and the checks that
# catch the two failure modes worth catching early -- extensions not built,
# and the GPU already occupied -- happen before we spend a minute loading
# weights only to OOM.
#
# Usage:
#   ./run.sh                                    # built-in prompts
#   ./run.sh --prompt "Explain paged attention."
#   ./run.sh --model Qwen/Qwen3-0.6B
#   ./run.sh --offline                          # never touch the network
#   VENV=/other/venv ./run.sh
#
# Any option not listed above is passed straight through to script.py.
set -euo pipefail

cd "$(dirname "$0")"

VENV="${VENV:-vllm-env}"
PYTHON="${VENV}/bin/python"
SCRIPT="${SCRIPT:-script.py}"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: no virtualenv python at ${PYTHON}" >&2
    echo "       run ./build.sh first, or set VENV=<path>" >&2
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "error: ${SCRIPT} not found" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# --offline is ours, not script.py's, so pull it out of the argument list.
script_args=()
for arg in "$@"; do
    case "$arg" in
        --offline)
            export HF_HUB_OFFLINE=1
            export TRANSFORMERS_OFFLINE=1
            ;;
        *)
            script_args+=("$arg")
            ;;
    esac
done

# The compiled extensions live in ./vllm and are what ./build_dev.sh installs.
# Importing them is a far cheaper way to find a broken build than waiting for
# the engine to come up.
if ! "$PYTHON" -c "import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch" 2>/dev/null; then
    echo "error: vLLM's compiled extensions failed to import" >&2
    echo "       run ./build_dev.sh (or ./build.sh) first" >&2
    exit 1
fi

# script.py asks for a fraction of *total* VRAM, so memory already held by
# another process comes out of its budget and turns into an OOM at load time.
if command -v nvidia-smi >/dev/null; then
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [[ "$used_mib" -gt 1024 ]]; then
        echo "warning: ${used_mib} MiB of VRAM is already in use by another process." >&2
        echo "         script.py budgets a share of total VRAM, so this may OOM." >&2
        echo "         free it, or lower --gpu-memory-utilization." >&2
    fi
fi

exec "$PYTHON" "$SCRIPT" "${script_args[@]}"

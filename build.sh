#!/usr/bin/env bash
# Build vLLM from source without starving the machine.
#
# Usage:
#   ./build.sh              # full build (C++/CUDA kernels)
#   ./build.sh --python     # Python-only changes, reuse precompiled kernels (fast)
#   MAX_JOBS=2 ./build.sh   # override parallelism
set -euo pipefail

cd "$(dirname "$0")"

VENV="${VENV:-vllm-env}"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "error: no virtualenv at ${VENV}/ (set VENV=<path>)" >&2
    exit 1
fi
source "${VENV}/bin/activate"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# Only build for this machine's GPU (RTX 5070 Ti = Blackwell sm_120).
# Cuts build time and peak memory by an order of magnitude vs. all archs.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

# Memory, not cores, is the limit: each nvcc job peaks around 2-3 GB.
# Budget one job per 3 GB of *available* RAM, capped at half the cores.
avail_gb=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
cores=$(nproc)
jobs=$(( avail_gb / 3 ))
(( jobs > cores / 2 )) && jobs=$(( cores / 2 ))
(( jobs < 1 )) && jobs=1
export MAX_JOBS="${MAX_JOBS:-$jobs}"

# nvcc's own internal threads multiply against MAX_JOBS, so keep this small.
export NVCC_THREADS="${NVCC_THREADS:-2}"

if command -v ccache >/dev/null; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
else
    echo "note: ccache not found; 'sudo apt install ccache' makes rebuilds far cheaper"
fi

# qutlass.cmake reuses an existing .deps/qutlass-src without checking its
# revision, so a checkout left behind by an older vLLM commit silently wins over
# the pinned tag and fails to compile. Drop it when it doesn't match the pin.
pin=$(sed -n 's/.*_QUTLASS_UPSTREAM_TAG "\([0-9a-f]*\)".*/\1/p' \
    cmake/external_projects/qutlass.cmake)
if [[ -n "${pin}" && -d .deps/qutlass-src ]]; then
    have=$(git -C .deps/qutlass-src rev-parse HEAD 2>/dev/null || true)
    if [[ "${have}" != "${pin}" ]]; then
        echo ">> stale qutlass checkout (${have:0:9} != ${pin:0:9}); re-fetching"
        rm -rf .deps/qutlass-src .deps/qutlass-build .deps/qutlass-subbuild
    fi
fi

# Build deps (torch, cmake, ninja, ...) must be present in the venv itself,
# since we build without isolation to guarantee build/runtime torch match.
if ! "${VENV}/bin/python" -c "import torch" 2>/dev/null; then
    echo ">> installing build requirements (this downloads torch, ~3 GB)"
    "${VENV}/bin/python" -m pip install -r requirements/build/cuda.txt
fi

if [[ "${1:-}" == "--python" ]]; then
    export VLLM_USE_PRECOMPILED=1
    echo ">> python-only build (precompiled kernels)"
else
    echo ">> full build: MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS} arch=${TORCH_CUDA_ARCH_LIST}"
fi

# Deprioritise CPU and disk so the desktop stays responsive during the build.
exec nice -n 15 ionice -c 3 \
    "${VENV}/bin/python" -m pip install -v -e . --no-build-isolation

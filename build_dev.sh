#!/usr/bin/env bash
#
# Incremental C++/CUDA builds against a persistent CMake tree.
#
# ./build.sh drives pip, which builds in a throwaway temp dir and therefore
# recompiles all ~400 objects every run. This script keeps the build tree in
# ./build, so ninja rebuilds only what actually changed, then installs the
# resulting .so files into ./vllm where the editable install picks them up.
#
# Requires ./build.sh to have been run once first, for the editable install
# and for the build dependencies (torch, cmake, ninja) inside the venv.
#
# Usage:
#   ./build_dev.sh                       # build the default targets
#   ./build_dev.sh _C_stable_libtorch    # build one specific target
#   ./build_dev.sh --all                 # build every target setup.py builds
#   ./build_dev.sh --reconfigure         # force a fresh cmake configure
#
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$PWD"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
VENV_DIR="${VENV:-${ROOT}/vllm-env}"
PYTHON="${VENV_DIR}/bin/python"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"

# Build only for this machine's GPU (RTX 5070 Ti is Blackwell, sm_120).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"

# Targets built by setup.py for a CUDA build.
ALL_TARGETS=(
    _C_stable_libtorch
    _moe_C_stable_libtorch
    _qutlass_C
    _flashkda_C
    _deep_gemm_C
    fmha_sm100
    tml_fa4
    triton_kernels
    cumem_allocator
    spinloop
    fs_io_C
)

# The two targets that hold the bulk of the kernels, and so are the ones you
# normally iterate on.
DEFAULT_TARGETS=(
    _C_stable_libtorch
    _moe_C_stable_libtorch
)


# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

reconfigure="no"
requested_targets=()

for arg in "$@"; do
    case "$arg" in
        --reconfigure)
            reconfigure="yes"
            ;;
        --all)
            requested_targets=("${ALL_TARGETS[@]}")
            ;;
        -*)
            echo "error: unknown option '$arg'" >&2
            exit 1
            ;;
        *)
            requested_targets+=("$arg")
            ;;
    esac
done

if [[ ${#requested_targets[@]} -eq 0 ]]; then
    targets=("${DEFAULT_TARGETS[@]}")
else
    targets=("${requested_targets[@]}")
fi


# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------

if [[ ! -x "$PYTHON" ]]; then
    echo "error: no virtualenv python at ${PYTHON}" >&2
    echo "       run ./build.sh first, or set VENV=<path>" >&2
    exit 1
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "error: nvcc not found at ${CUDA_HOME}/bin/nvcc" >&2
    echo "       set CUDA_HOME to your CUDA install" >&2
    exit 1
fi


# ---------------------------------------------------------------------------
# Decide how much parallelism this machine can absorb
#
# Memory is the limit here, not cores. Two things make it unforgiving:
#
#   - There is no swap on this machine, so overcommitting does not degrade
#     into slowness, it goes straight to OOM-kill or a livelocked desktop.
#   - vLLM's heavy template units (CUTLASS, Marlin) peak far above the 2-3 GB
#     a typical .cu costs, and -DNVCC_THREADS multiplies against the job count.
#
# So: reserve a fixed slice for the desktop, charge each job for the nvcc
# threads it will actually spawn, and never use more than half the cores.
# ---------------------------------------------------------------------------

# nvcc's own internal threads multiply against the job count, so keep it small.
nvcc_threads="${NVCC_THREADS:-2}"

# Held back for the desktop, page cache, and the tail of a heavy compile.
reserve_gb="${RESERVE_GB:-6}"

if [[ -n "${MAX_JOBS:-}" ]]; then
    jobs="$MAX_JOBS"
else
    available_gb=$(awk '/MemAvailable/ { print int($2 / 1024 / 1024) }' /proc/meminfo)
    cores=$(nproc)

    budget_gb=$(( available_gb - reserve_gb ))
    if [[ $budget_gb -lt 1 ]]; then
        budget_gb=1
    fi

    # A job costs ~3 GB of baseline nvcc footprint plus roughly a GB per
    # internal thread it is allowed to fan out to.
    per_job_gb=$(( 3 + nvcc_threads ))

    jobs=$(( budget_gb / per_job_gb ))

    half_the_cores=$(( cores / 2 ))
    if [[ $jobs -gt $half_the_cores ]]; then
        jobs=$half_the_cores
    fi

    if [[ $jobs -lt 1 ]]; then
        jobs=1
    fi
fi

# Linking these extensions is the single biggest memory spike in the build --
# well above any one compile -- so it gets its own, much narrower pool rather
# than riding along on the compile count.
link_jobs="${LINK_JOBS:-2}"
if [[ $link_jobs -gt $jobs ]]; then
    link_jobs=$jobs
fi

if [[ $(awk '/SwapTotal/ { print $2 }' /proc/meminfo) -eq 0 ]]; then
    echo ">> note: no swap configured; keeping parallelism conservative"
fi


# ---------------------------------------------------------------------------
# Configure
#
# These flags mirror what setup.py passes (see its `configure` method), so this
# build tree stays interchangeable with the one pip produces.
# ---------------------------------------------------------------------------

if [[ "$reconfigure" == "yes" ]]; then
    echo ">> forcing reconfigure"
    rm -f "${BUILD_DIR}/CMakeCache.txt"
fi

# Reusing the cache blindly is how a build silently keeps whatever job pools
# and GPU architecture it was first configured with, no matter what this run
# asks for. Only skip the configure step when the cached values still match.
# TORCH_CUDA_ARCH_LIST is "12.0"; cmake stores it as "120".
want_archs="${TORCH_CUDA_ARCH_LIST//./}"
# A cmake list is semicolon-separated; a comma here silently collapses into one
# bogus pool, and ninja then fails on the undefined pool name.
want_pools="compile=${jobs};link=${link_jobs}"

config_is_current="no"
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    # The values themselves contain '=', so strip only through the first one
    # rather than splitting on every '='.
    cached_pools=$(sed -n 's/^CMAKE_JOB_POOLS:[^=]*=//p' "${BUILD_DIR}/CMakeCache.txt")
    cached_archs=$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[^=]*=//p' "${BUILD_DIR}/CMakeCache.txt")

    if [[ "$cached_pools" == "$want_pools" && "$cached_archs" == *"$want_archs"* ]]; then
        config_is_current="yes"
    else
        echo ">> cached configuration is stale, reconfiguring"
        [[ "$cached_archs" == *"$want_archs"* ]] || \
            echo "   arch:  cached '${cached_archs}' != requested '${want_archs}'"
        [[ "$cached_pools" == "$want_pools" ]] || \
            echo "   pools: cached '${cached_pools}' != requested '${want_pools}'"
    fi
fi

if [[ "$config_is_current" == "yes" ]]; then
    echo ">> reusing existing cmake configuration in ${BUILD_DIR}"
else
    echo ">> configuring ${BUILD_DIR}"

    # ccache turns a rebuild-from-scratch into mostly cache hits. setup.py
    # detects it the same way for pip builds.
    compiler_launcher_args=()
    if command -v ccache >/dev/null; then
        compiler_launcher_args=(
            -DCMAKE_C_COMPILER_LAUNCHER=ccache
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
            -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
        )
    else
        echo "   note: ccache not installed; first build of each target will be slow"
        echo "         install it with: sudo apt-get install -y ccache"
    fi

    # Letting cmake see the venv's sys.path is how it finds torch's cmake
    # packages without us hardcoding paths.
    python_path=$("$PYTHON" -c 'import sys; print(":".join(sys.path))')

    cmake -G Ninja -B "${BUILD_DIR}" -S "${ROOT}" \
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
        -DVLLM_TARGET_DEVICE=cuda \
        -DVLLM_PYTHON_EXECUTABLE="${PYTHON}" \
        -DVLLM_PYTHON_PATH="${python_path}" \
        -DFETCHCONTENT_BASE_DIR="${ROOT}/.deps" \
        -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
        -DNVCC_THREADS="${nvcc_threads}" \
        -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
        -DCMAKE_JOB_POOL_LINK:STRING=link \
        -DCMAKE_JOB_POOLS:STRING="${want_pools}" \
        "${compiler_launcher_args[@]}"
fi


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

echo ">> building: ${targets[*]}"
echo "   jobs=${jobs} link_jobs=${link_jobs} nvcc_threads=${nvcc_threads} arch=${TORCH_CUDA_ARCH_LIST}"

target_args=()
for target in "${targets[@]}"; do
    target_args+=("--target=${target}")
done

# nice/ionice keep the desktop responsive while this runs.
nice -n 15 ionice -c 3 \
    cmake --build "${BUILD_DIR}" -j"${jobs}" "${target_args[@]}"


# ---------------------------------------------------------------------------
# Install
#
# Each extension is its own cmake component, installed with the repo root as
# the prefix. That puts each .so next to the Python sources the editable
# install already points at, which is exactly what setup.py does.
# ---------------------------------------------------------------------------

for target in "${targets[@]}"; do
    cmake --install "${BUILD_DIR}" --prefix "${ROOT}" --component "${target}"
done

echo ">> done"

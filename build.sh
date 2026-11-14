#!/usr/bin/env bash
set -euo pipefail

apt-get update || true
apt-get install -y curl ca-certificates

# Ubuntu 24.04 -> ubuntu2404 (not ubuntu24.04)
source /etc/os-release
distribution="${ID}${VERSION_ID//./}"
arch="x86_64"

REPO="https://developer.download.nvidia.com/compute/cuda/repos/${distribution}/${arch}"

# Remove the invalid source added previously
rm -f /etc/apt/sources.list.d/nvidia-devtools.list

# Install NVIDIA repository keyring
curl -fsSL \
    "${REPO}/cuda-keyring_1.1-1_all.deb" \
    -o /tmp/cuda-keyring.deb

dpkg -i /tmp/cuda-keyring.deb

apt-get update

# Install Nsight tools for CUDA 12.8
CUDA_VERSION="${CUDA_VERSION:-12-8}"

apt-get install -y \
    "cuda-nsight-systems-${CUDA_VERSION}" \
    "cuda-nsight-compute-${CUDA_VERSION}"

# Ensure profiler executables are on PATH
export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.8/bin:$PATH"

echo "Nsight Systems:"
nsys --version

echo "Nsight Compute:"
ncu --version

# Install vLLM
VLLM_USE_PRECOMPILED=1 uv pip install -e .
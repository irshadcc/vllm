# Add the NVIDIA devtools repository list for your architecture
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -O https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

# Add the devtools source specifically
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/ /" | tee /etc/pki/rpm-gpg /etc/apt/sources.list.d/nvidia-devtools.list 2>/dev/null || true
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/ /" | tee /etc/apt/sources.list.d/nvidia-devtools.list

#Update package lists and install prerequisite utilities
apt-get update && apt-get install -y wget gnupg

# (Optional) If you need the full CUDA toolkit which includes nsys and ncu:
# Make sure to check your CUDA version first with `nvcc --version`
apt-get install -y cuda-toolkit-$(nvcc --version | grep "Release" | awk '{print $NF}' | cut -d'.' -f1,2 | tr '.' '-')

# Alternatively, you can install just the profiling tools package if available in your repo:
apt-get install -y nsight-systems-cli


VLLM_USE_PRECOMPILED=1 uv pip install -e .
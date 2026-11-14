#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nsys profile \
    --force-overwrite=true \
    --stats=true \
    --trace=cuda,nvtx,osrt \
    --trace-fork-before-exec=true \
    --sample=process-tree \
    --cpuctxsw=process-tree \
    --cuda-memory-usage=true \
    --cuda-graph-trace=node \
    -o vllm_a6000_profile \
    vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --enforce-eager
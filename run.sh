# vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
#     --tensor-parallel-size 2 \
#     --max-model-len 32768 \
#     --port 8000 \
#     --host 0.0.0.0


nsys profile \
    --stats=true \
    --cuda-memory-usage=true \
    --trace=cuda,nvtx,osrt \
    -o vllm_a6000_profile \
    vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --enforce-eager
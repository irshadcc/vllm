"""Run a local model with vLLM on this machine's RTX 5070 Ti (16 GB, sm_120).

Defaults to the AWQ-quantized Qwen2.5-VL-7B already present in the local
HuggingFace cache, so no download is needed. Run with:

    ./vllm-env/bin/python script.py
    ./vllm-env/bin/python script.py --prompt "Explain paged attention."
    ./vllm-env/bin/python script.py --model Qwen/Qwen3-0.6B
"""

import argparse

from vllm import LLM, SamplingParams

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

DEFAULT_PROMPTS = [
    "Explain what paged attention is in two sentences.",
    "Write a Python function that reverses a linked list.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to send; repeat for a batch. Defaults to a built-in pair.",
    )
    # 16 GB total: leave headroom for the CUDA context and activations.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    # A short context keeps the KV cache small enough to fit alongside weights.
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    conversations = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(conversations, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("=" * 70)
        print(f"PROMPT: {prompt}")
        print("-" * 70)
        print(output.outputs[0].text.strip())
    print("=" * 70)


if __name__ == "__main__":
    main()

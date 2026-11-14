import multiprocessing as mp
from vllm import LLM, SamplingParams

def main():
    # Recommended: Set spawn method explicitly for safety across platforms and PyTorch CUDA workers
    mp.set_start_method("spawn", force=True)

    print("Initializing vLLM engine for Qwen2.5-Coder-14B-Instruct...")
    
    # 1. Initialize the engine directly
    llm = LLM(
        model="Qwen/Qwen2.5-Coder-14B-Instruct",
        max_model_len=32768,
        tensor_parallel_size=1,  # Change this if you are using multiple GPUs
        enforce_eager=False,     # Set to True if you want to disable CUDA graphs for debugging
    )

    # 2. Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=512
    )

    llm.

    for i in range(100):
    # 3. Provide prompts to run generation
        prompts = [
            "Write a Python function using binary search to find an element in a sorted list.",
        ]

        print("Running generation...")
        outputs = llm.generate(prompts, sampling_params)

        # 4. Print results
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt: {prompt}\nGenerated Text:\n{generated_text}")

if __name__ == "__main__":
    main()
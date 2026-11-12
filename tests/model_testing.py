from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
import torch
import time

model_name = "Qwen/Qwen3-1.7B"

current_file_path = Path(__file__).resolve().parent

model_path = Path.joinpath(current_file_path, "../models/Qwen3-1.7B")
output_path = Path.joinpath(current_file_path, "../results/results.ndjson")

MAX_NEW_TOKENS = 64

PROMPTS = [
    "Hi",
    "Give me a short introduction to large language model",
    "Explain the difference between prefill and decode in LLM inference, "
    "and why decode is usually memory-bound rather than compute-bound.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Hi",
    "Give me a short introduction to large language model",
    "Explain the difference between prefill and decode in LLM inference, "
    "and why decode is usually memory-bound rather than compute-bound.",
    "Write a Python function that returns the n-th Fibonacci number.",
]

print("MODEL_PATH", model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="mps", dtype=torch.float32).eval()

print("DTYPE", model.dtype)

generation_config_decode = GenerationConfig.from_dict({
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
})

generation_config_prefill = GenerationConfig.from_dict({
    "max_new_tokens": 1,
    "do_sample": False,
})

# "w" overwrites on each run, so the file always reflects exactly one run's settings.
with open(output_path, "w") as f:
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_ids = model_inputs.input_ids[0].tolist()

        torch.mps.synchronize()
        start = time.perf_counter()
        generated_ids = model.generate(
            **model_inputs,
            generation_config=generation_config_prefill,
        )
        torch.mps.synchronize()
        end = time.perf_counter()

        t1 = end - start

        torch.mps.synchronize()
        start = time.perf_counter()
        generated_ids = model.generate(
            **model_inputs,
            generation_config=generation_config_decode,
        )
        torch.mps.synchronize()
        end = time.perf_counter()

        decode_time = end - start - t1


        output_ids = generated_ids[0][len(input_ids):].tolist()

        record = {
            "model": model_name,
            "dtype": str(model.dtype),
            "device": str(model.device),
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt": prompt,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "time": end - start,
            "prefill_time": t1,
            "decode_time": decode_time,
            "decode_time_per_token": decode_time / (len(output_ids) - 1)
        }

        f.write(json.dumps(record) + "\n")

        print(prompt)
        print(output_ids)
        print(tokenizer.decode(output_ids))
        print()

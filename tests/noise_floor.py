"""Measure the numerical noise floor between two *correct* HF fp32 runs.

The goal is to choose a data-backed tolerance for comparing our own forward pass
against HF. Do not run this while a timing benchmark is running.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

current_file_path = Path(__file__).resolve().parent
model_path = current_file_path / "../models/Qwen3-1.7B"

PROMPTS = [
    "Hi",
    "Give me a short introduction to large language model",
    "Explain the difference between prefill and decode in LLM inference, "
    "and why decode is usually memory-bound rather than compute-bound.",
    "Write a Python function that returns the n-th Fibonacci number.",
]

tokenizer = AutoTokenizer.from_pretrained(model_path)


def load(device: str):
    return AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map=device
    ).eval()


def encode(prompt: str) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tokenizer([text], return_tensors="pt").input_ids


@torch.no_grad()
def logits_for(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids.to(model.device)).logits  # [1, seq, vocab]
    return out[0].to("cpu", torch.float32)  # [seq, vocab]


def report(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    diff = (a - b).abs()
    per_pos = diff.amax(dim=-1)  # max abs diff at each position
    top2 = a[-1].topk(2).values
    print(
        f"  {name:<18} max|diff|={diff.max().item():.3e}  "
        f"last-pos max={per_pos[-1].item():.3e}  "
        f"worst pos={per_pos.argmax().item()}  "
        f"argmax agree (all pos)={(a.argmax(-1) == b.argmax(-1)).all().item()}  "
        f"top1-top2 gap (last)={(top2[0] - top2[1]).item():.3f}"
    )


cpu_model = load("cpu")
cpu_logits = {p: logits_for(cpu_model, encode(p)) for p in PROMPTS}
cpu_logits_again = {p: logits_for(cpu_model, encode(p)) for p in PROMPTS}
del cpu_model

mps_model = load("mps")
mps_logits = {p: logits_for(mps_model, encode(p)) for p in PROMPTS}
mps_logits_again = {p: logits_for(mps_model, encode(p)) for p in PROMPTS}

for p in PROMPTS:
    print(f"\n{p[:60]!r}  (seq_len={cpu_logits[p].shape[0]})")
    report("cpu vs cpu (rerun)", cpu_logits[p], cpu_logits_again[p])
    report("mps vs mps (rerun)", mps_logits[p], mps_logits_again[p])
    report("cpu vs mps", cpu_logits[p], mps_logits[p])

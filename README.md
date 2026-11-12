# inference-engine

A vLLM-style LLM inference engine, built from scratch in PyTorch to learn how inference works. Every component is hand-written, checked for correctness against a reference, and benchmarked as it is added.

> **Status: early.** Phase 0 (reference and baseline) is done. The engine itself is not built yet, so there is nothing to serve. See the [roadmap](ROADMAP.md).

## What this is

* A learning project: the aim is understanding, not matching vLLM's performance.
* Target model: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), a dense GQA transformer.
* Planned: forward pass, KV cache, sampling, batching, continuous batching, paged attention, prefix caching, and an OpenAI-compatible API. GPU-only work (kernels, CUDA graphs, quantization, tensor parallelism) comes later on rented hardware.
* Hugging Face `transformers` is used only as a **reference oracle** for tests and baselines, not for the forward pass.

Full plan, requirements and exit criteria: [`ROADMAP.md`](ROADMAP.md).

## Current status

| Phase | State |
|---|---|
| 0. HF reference and baseline | Done |
| 1. Naive forward pass (no cache) | Next |
| 2+ | See [roadmap](ROADMAP.md) |

**Baseline** (HF `generate`, Qwen3-1.7B, fp32, MPS on Apple M5 Pro, batch size 1, greedy): about 28-30 tok/s decode and about 0.08-0.13 s prefill for short prompts. These are rough single-run numbers. Details and caveats: [`notes/phase0-hf-baseline.md`](notes/phase0-hf-baseline.md).

## Getting started

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.14. The reference scripts currently target Apple Silicon (MPS).

```sh
# install dependencies
uv sync

# download the model weights (about 3.8 GB, git-ignored)
hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B

# generate reference tokens and timings with Hugging Face (writes results/results.ndjson)
uv run python tests/model_testing.py

# measure fp32 logit noise between CPU and MPS
uv run python tests/noise_floor.py
```

The `inference-engine generate` CLI exists as a stub and raises `NotImplementedError` until the engine is built.

## Repository layout

```
inference_engine/
├── README.md
├── ROADMAP.md          # goals, requirements, phases, benchmark spec
├── CLAUDE.md           # working rules for AI coding assistants
├── pyproject.toml
├── src/inference_engine/
│   ├── cli.py          # CLI entry point (stub)
│   ├── device/         # device selection and synchronization helpers
│   ├── model/          # weight loading, layers, forward pass (in progress)
│   ├── cache/          # KV cache, block allocator, prefix cache (planned)
│   ├── scheduler/      # request queue, batching policy (planned)
│   ├── sampling/       # (planned)
│   ├── kernels/        # custom attention and other kernels (GPU phases)
│   └── server/         # async API layer (planned)
├── tests/              # reference scripts and correctness tests
├── benchmarks/         # harness, workloads, scripts
├── results/            # logged benchmark and reference outputs
└── notes/              # learning notes, design decisions per phase
```

## Principles

* **Correctness first.** A faster engine that gives different outputs is a bug until proven otherwise.
* **One change at a time**, each benchmarked against a recorded baseline.
* **Readable over clever.** The code exists to be learned from.

## Contributing and AI assistants

This is a personal learning project, so the code is written by hand by the author. Rules for AI coding assistants (mentor role, no engine code generation) live in [`CLAUDE.md`](CLAUDE.md).

# Roadmap

The plan for building the engine: goals, requirements, how correctness and performance are checked, and the phases in order. For the project overview and quickstart see [`README.md`](README.md).

## Goals

1. Understand inference end to end: model forward pass, KV cache, batching, scheduling, memory management, kernels, quantization, speculative decoding, parallelism, serving.
2. Build a working engine that serves a real open-weights model and exposes an OpenAI-style API.
3. Measure the effect of each optimization with a reproducible benchmark and write down why the numbers came out the way they did.

## Non-goals

* Training or fine-tuning.
* Beating vLLM in absolute performance. The aim is to understand the gap, not close it.
* Supporting many model architectures. One model family is enough.
* Production hardening (auth, multi-tenant isolation, autoscaling).

## Environment and constraints

| Item | Current | Notes |
|---|---|---|
| Dev machine | Apple M5 Pro, arm64, 48 GB unified memory, 307 GB/s | PyTorch MPS and CPU only. No CUDA, no Triton. |
| Python | 3.14 (`.python-version`) | Working so far with the current PyTorch wheel. |
| Package manager | `uv` | `pyproject.toml` + `uv.lock` |
| GPU work | Not yet available | Rent an NVIDIA GPU (e.g. A10/L4/A100/H100) for the kernel and CUDA-graph phases. |

**Consequence:** phases that are hardware-independent (model, KV cache, sampling, scheduler, paged memory management, prefix caching, serving) are built and verified locally. Phases that need CUDA (Triton/CUDA attention kernels, CUDA graphs, FP8, tensor parallelism) are deferred to a rented GPU box, and the code must keep the device backend swappable.

## Target model

**Qwen3-1.7B** (`Qwen/Qwen3-1.7B`, bf16 weights ~3.8 GB, stored in `models/Qwen3-1.7B`, git-ignored). A standard dense transformer: 28 layers, hidden 2048, 16 query heads / 8 KV heads (GQA), head_dim 128, RMSNorm, RoPE, SwiGLU, tied embeddings.

**Why not Qwen3.5-2B:** it is a hybrid (18 of 24 layers are Gated DeltaNet linear attention with recurrent state, 6 are full attention) and multimodal, with an attention output gate, partial/multimodal RoPE and MTP. That would complicate the KV cache, paged memory and prefix caching phases. It is a good later stretch phase for hybrid-model support.

* Load weights from `safetensors` directly. Do not use `transformers` for the forward pass.
* Hugging Face `transformers` is allowed only as a **reference oracle** for correctness tests and baseline benchmarks.
* Later: a second, larger model (e.g. 7-8B) on a rented GPU for realistic benchmarks and tensor parallelism.

## Functional requirements

| ID | Requirement |
|---|---|
| F1 | Load weights and tokenizer for the target model from a local directory. |
| F2 | Hand-written forward pass (embedding, RMSNorm, RoPE, GQA attention, SwiGLU MLP, LM head). |
| F3 | Greedy generation matching the reference token for token. |
| F4 | KV cache with separate prefill and decode paths. |
| F5 | Sampling: temperature, top-k, top-p, seeded determinism, stop tokens, max length. |
| F6 | Batched generation over multiple concurrent requests. |
| F7 | Continuous (iteration-level) batching with a request queue and a scheduler. |
| F8 | Paged KV cache: block allocator, per-sequence block tables, preemption. |
| F9 | Chunked prefill and a configurable scheduling policy. |
| F10 | Prefix caching (shared blocks across requests). |
| F11 | Streaming token output. |
| F12 | OpenAI-compatible HTTP API (`/v1/completions`, `/v1/chat/completions`, streaming). |
| F13 | Optimizations added incrementally and individually toggleable by flag: fused/flash/paged attention kernels, CUDA graphs, quantization (weight-only, KV cache), speculative decoding, tensor parallelism. |

## Non-functional requirements

| ID | Requirement |
|---|---|
| N1 | **Correctness first.** Every phase must pass a correctness test against the reference before its benchmarks count. |
| N2 | **Reproducibility.** Benchmarks run from a single command, with fixed seeds, recorded config, git commit and hardware. |
| N3 | **Toggleable optimizations.** Each optimization is behind a flag so before/after comparisons run on the same code. |
| N4 | **Device abstraction.** Model and cache code must not hardcode a device (CPU, MPS, CUDA). |
| N5 | **Observability.** Per-request timing (queue, prefill, decode), tokens/s, and memory usage are logged. |
| N6 | **Readable over clever.** The code exists to be learned from. |

## Correctness strategy

* **Logit match:** for fixed prompts, compare per-position logits with the HF reference within a stated tolerance (per dtype). The fp32 CPU-vs-MPS noise floor measured in Phase 0 is 1e-4 to 3e-4 max abs diff (see [`notes/phase0-hf-baseline.md`](notes/phase0-hf-baseline.md)); the Phase 1 tolerance is chosen above that floor.
* **Token match:** greedy output must be identical to the reference for N test prompts, in fp32. In bf16, near-ties make token match unreliable, so compare logits instead.
* **Invariants:** allocator never double-frees or leaks blocks; a sequence's output is independent of which other sequences share its batch; prefix-cached output equals uncached output.
* **Quality checks** for lossy optimizations (quantization, speculative decoding with sampling): perplexity or task-eval delta against the unoptimized engine.

## Benchmark specification

**Metrics**

* Throughput (output tokens/s, requests/s)
* TTFT (time to first token)
* TPOT / inter-token latency
* End-to-end latency at p50 / p95 / p99
* Peak memory, KV cache capacity, max concurrent sequences
* Utilization (achieved bandwidth and FLOPs versus hardware peak)

**Workloads**

* *Offline:* fixed batch of prompts, measure max throughput.
* *Online:* Poisson arrivals at several request rates, measure latency distributions.
* Prompt/output length distributions are fixed and documented (short chat, long prompt, mixed). A shared-prefix workload is used for prefix caching.

**Method**

* Warmup runs, then repeated measurements with variance reported.
* Correct device synchronization before timing.
* One change at a time against a recorded baseline.
* Compare against HF `generate` (and optionally vLLM on the GPU box) to calibrate.
* Each result is logged with date, commit, hardware, config, numbers and a short interpretation.

## Phases and exit criteria

| Phase | Scope | Exit criteria | Env | Status |
|---|---|---|---|---|
| 0 | Setup: env, weights, tokenizer, HF reference + baseline benchmark | HF baseline numbers recorded | Mac | Done |
| 1 | Naive forward pass, no cache | Logits match reference | Mac | Next |
| 2 | KV cache (prefill/decode) | Tokens match; speedup measured | Mac | |
| 3 | Sampling | Seeded determinism test passes | Mac | |
| 4 | Static batching | Batch outputs equal single-request outputs | Mac | |
| 5 | Continuous batching + scheduler | Throughput beats static batching on mixed lengths | Mac | |
| 6 | Paged KV cache | Allocator invariants hold; memory waste quantified | Mac | |
| 7 | Chunked prefill + scheduling policy | TTFT/TPOT trade-off curves | Mac | |
| 8 | Prefix caching | Identical outputs; speedup on shared-prefix workload | Mac | |
| 9 | Serving layer (async, streaming, OpenAI API) | Load test passes | Mac | |
| 10 | Attention kernels (paged, flash-style, flash-decoding) | Match reference; bandwidth analysis | GPU | |
| 11 | CUDA graphs, `torch.compile`, op fusion | Launch overhead reduction measured | GPU | |
| 12 | Quantization (weight-only, KV, FP8) | Quality delta and speedup reported | GPU (partly Mac) | |
| 13 | Speculative decoding | Acceptance rate and speedup reported | Mac / GPU | |
| 14 | Tensor parallelism (optional: PP, disaggregated prefill/decode) | Multi-GPU scaling measured | multi-GPU | |
| 15 | Stretch: structured output, LoRA serving, MoE | TBD | TBD | |

Phases 0 to 9 can all be done on the Mac.

## Open decisions

1. ~~Target model~~ Decided: Qwen3-1.7B.
2. ~~Python version~~ 3.14 works with the current PyTorch wheel so far. Revisit if a later dependency lacks wheels.
3. **GPU access:** which provider and when, since phases 10+ depend on it.
4. **Dtype for early phases:** fp32 is the reference dtype for exact matching (bf16 diverges from fp32 on near-ties). Open: which dtype the engine runs by default after Phase 1.
5. **Server framework:** FastAPI/uvicorn versus a lighter option.
6. **Reference oracle scope:** HF only, or also vLLM on the GPU box for benchmark calibration.
7. **Phase 1 logit tolerance:** value and rtol/atol split, to be set from the Phase 0 noise floor.

# Phase 0: HF reference and baseline

**Date:** 2026-11-11
**Hardware:** Apple M5 Pro, 48 GB unified memory, 307 GB/s memory bandwidth (MPS)
**Model:** Qwen3-1.7B, fp32, HF `transformers` `generate`, greedy, `max_new_tokens=64`, batch size 1
**Code:** `tests/model_testing.py` (reference tokens + timing), `tests/noise_floor.py` (HF-vs-HF logit noise)

## Reference outputs
- `results/results.ndjson` holds prompt ids, generated ids, dtype and device for 4 fixed prompts (lengths 9, 17, 20, 34 tokens).
- Greedy output is deterministic: repeated runs gave identical tokens.
- bf16 and fp32 diverge on the same prompt after a few dozen tokens (rounding flips near-ties), so the fp32 run is the reference. Token-match over 64 steps in bf16 would not be a fair test.

## Noise floor (fp32, logits)
| comparison | max abs diff |
|---|---|
| CPU vs CPU rerun | 0 |
| MPS vs MPS rerun | 0 |
| CPU vs MPS | 1.1e-4 to 2.9e-4 (worst position is often position 0) |

Top-1 vs top-2 logit gap at the last position was about 11-12, far above the noise. Proposed tolerance for Phase 1: about 1e-3 max abs diff (3-10x the observed floor). Still to confirm on a longer prompt and to combine with an rtol.

## Timing (HF generate, MPS, fp32)
- **Decode:** about 28-30 tok/s steady state (about 0.034-0.036 s/token).
- **Prefill (includes `generate` setup):** about 0.08-0.13 s for 9-34 tokens. Does not grow clearly with prompt length.
- **Back-of-envelope ceiling:** 6.8 GB of fp32 weights / 307 GB/s is about 22 ms/token, so about 45 tok/s at best. Measured steady state is about 62-67% of that.

## Caveats and open questions
- One measurement per prompt, no warmup, no shuffled order. This is a rough baseline, not a rigorous benchmark.
- The very first prompt of a run was faster (36-38 tok/s) than later ones. When "Hi" was repeated later in the same run it ran at 28.2 tok/s, so this is an order effect and not a prompt effect. Cause not identified (thermal or clock state is a guess).
- Prefill at 4-6x the weight-read floor suggests `generate` overhead dominates. Not investigated.
- Gap between 28-30 and 45 tok/s is not explained yet.
- Timing and reference tokens are written to the same file, so a timing run overwrites the reference. Split them before relying on the reference.

## Next
Phase 1: list layer-0 tensor names and shapes from `model.safetensors.index.json`, derive shapes from `config.json`, then build and test each component against HF.

# Co-design: GQA decode kernel + mini schedule IR

Design doc for the compiler-level toy project: a ~10-construct schedule IR
whose lowering emits CUDA for the GQA decode family, co-designed with a
first-principles "ideal kernel" prediction that the IR must be able to
express — and that exhaustive enumeration over the IR can confirm or
refute. Kernel-side analysis first (it dictates the IR), then the IR,
then gates, cost model, and integration.

## 0. Problem (frozen from EXPERIMENT.md)

Decode attention, q_len=1: B=4, H_q=32, H_kv=8 (G=4 query heads per KV
head), D=128, L=8192 KV length, BF16 in/out, FP32 accumulate, contiguous
KV cache `[B, H_kv, L, D]`. Output `O[B, H_q, D]`.

## 1. First-principles analysis (what "ideal" means here)

- **Traffic:** K+V = 2 x B x H_kv x L x D x 2B = **268.4 MB** must be read
  once. Q and O are ~32 KB each — noise.
- **Compute:** 4 x B x H_q x L x D ≈ **0.54 GFLOP**.
- **Intensity:** ~2 FLOP/byte vs a B200 BF16 ridge of ~280 → as
  memory-bound as kernels get. GQA's G=4 sharing is already counted (each
  KV element serves 4 query heads; MHA would read 4x more).
- **Speed of light:** 268.4 MB / read ceiling (~6.2 TB/s measured proxy)
  ≈ **43-45 us**. The harness's read_peak_gbps is the honest denominator;
  pct_measured_peak vs the copy proxy will understate this kernel.
- **Parallelism problem:** natural work units = B x H_kv = **32** vs 148
  SMs on B200. An unsplit kernel leaves >75% of the chip idle — this is
  THE decode pathology (Flash-Decoding's 36-43x win is exactly this fix).
  Split each (b, h_kv)'s KV range into S chunks → B x H_kv x S CTAs, with
  partial (m, l, O_acc) results merged by log-sum-exp.
- **Consequences:** compute engine choice barely matters (scores for 4
  q-heads per KV row is a [4x128]x[128xT] micro-GEMM — FMA units keep up
  with DRAM easily); what matters is (1) SM fill via S, (2) a load
  pipeline that never stalls (multi-stage cp.async/TMA), (3) not wasting
  bandwidth (perfectly coalesced, read-once, no partial-result thrash).

### Predicted ideal schedule (falsifiable — enumeration will judge it)

S=8-16 (256-512 CTAs, ~2-3 waves), tile_rows=64, stages=3, TMA loads,
8 warps uniform (no warp specialization needed at this pipeline depth),
FMA score path, conditional softmax rescale, two-kernel combine.
Predicted time ~45-50 us (≥90% of read ceiling). If enumeration finds
something meaningfully better, the analysis above was wrong somewhere —
which is the interesting outcome.

## 2. The mini IR

One schedule = one small typed record. Everything an agent or enumerator
chooses is here; everything mechanical (indexing, barriers, tail handling,
partial-buffer layout, combine launch) is derived by the lowering.

```
GqaDecodeSchedule:
  # --- work decomposition ---
  split_kv:        S in {1, 2, 4, 8, 16, 32}    # chunks per (b, h_kv)
  tile_rows:       {32, 64, 128}                # KV rows per pipeline tile
  # --- load pipeline ---
  stages:          {2, 3, 4}                    # smem buffering depth
  load_engine:     {cp_async, tma}
  vector_width:    {4, 8}                       # bf16 elems/inst (cp_async only)
  num_warps:       {4, 8}
  warp_roles:      {uniform, producer_consumer} # v0 may ship uniform-only
  # --- compute ---
  score_engine:    {fma, mma_m16}               # QK^T path; PV mirrors it
  softmax_rescale: {always, conditional}
  # --- combine ---
  combine:         {second_kernel}              # v1: + fused_atomic_flag
```

Full space ≈ 6 x 3 x 3 x 2 x 2 x 2 x 2 x 2 x 2 ≈ 3.5k points; after
static gates and v0 restrictions (uniform roles, second_kernel) it's
**300-600 legal specs — exhaustively enumerable**. Batch-measured on
Modal (many candidates per container), that is a $20-40 experiment.

Design rules borrowed from CAKE, scaled down:
- **Declarative commitments, derived mechanics.** The spec names WHAT
  (tile size, engine, stages); the lowering derives HOW (cp.async groups,
  mbarrier phases, smem offsets, LSE-merge math). Agents never write
  barrier code, so an entire class of correctness failures is
  unrepresentable.
- **No layout algebra.** D=128 BF16 rows are 256 B — trivially coalesced;
  the IR doesn't let you express an uncoalesced layout at all.
- **The IR grows bottom-up.** v0 spans exactly this family. When a
  schedule idea can't be expressed (stream-K assignment, TMEM accumators,
  fused combine via arrival counters), that's an IR extension with its
  own commit — CAKE's corpus-growth loop, one person scale.

## 3. Static gates (reject before GPU time)

1. smem: stages x tile_rows x D x 2B x 2 (K+V) ≤ 200 KB (leave slack
   below the 228 KB/SM limit for the Q tile + barriers).
2. chunk length L/S must be ≥ tile_rows; tail iterations derived when
   tile_rows doesn't divide it (L=8192 divides everything in-space, but
   the gate keeps the IR honest for future shapes).
3. mma_m16 requires tile_rows % 16 == 0.
4. tma requires vector_width unset (TMA moves whole tiles; the knob is
   meaningless — gate rejects contradictory specs rather than silently
   ignoring them).
5. Register estimate: accumulators G x D x 4B / (32 x num_warps) per
   thread + pipeline overhead ≤ 255 regs/thread (formula documented in
   the lowering; refined against ptxas --resource-usage output).
6. Workspace: S x B x H_q x (D + 2) x 4B partials ≤ preallocated buffer.

## 4. Cost model v0 (rank before measuring)

time ≈ max( KV_bytes / BW_eff , combine_cost )
BW_eff = read_peak x fill(CTAs) x pipeline_eff(stages)
  - fill(CTAs): saturating curve in CTAs/148, ~linear below 1 wave —
    fitted to the directional-proxy data the harness already logs.
  - pipeline_eff: 1 - 1/(2^(stages-1)) style penalty at stage depth 1-2.
combine_cost ≈ second-kernel launch (~3-5 us) + S x 66 KB partial traffic.

Predicts: S=1 catastrophic (~4.6x slow), knee at S≈8, flat 8→32 with a
slight combine-cost rise at 32. v1: fit coefficients to enumeration
results in results.jsonl (spec hash lives in the `note` field, so the
training set assembles itself).

## 5. Lowering & integration

- `tools/gqa_lower.py` (driver-side Python, like modal/ — the C++/Make
  verifier stays pure): spec (JSON) → `kernels/generated/gqa_decode_<hash>.cu`,
  spec embedded as a header comment and as the run's NOTE. Both passes
  (main + combine) live in one .cu; the launch lambda runs main then
  combine so the harness times the pair — bytes_moved accounts for both
  (partials add S x 66 KB ≈ ≤2 MB, <1% of 268 MB).
- Verified by the existing harness unchanged: CPU FP64 reference,
  poison / validate / time / re-validate. Tolerance pinned during task
  authoring per EXPERIMENT.md (proposal to validate empirically: abs err
  ≤ 3e-2 on O(1)-scale outputs; BF16 output quantization alone is ~0.8%
  relative).
- Enumeration driver batches N specs per Modal container (compile all,
  run sequentially, one JSON line each) to amortize container spin-up.

## 6. What this buys, in order

1. **A certainty result:** the best schedule in the family, found by
   exhaustive search, vs the ~44 us speed of light — no agent luck
   involved. Either the predicted-ideal spec wins (analysis validated)
   or something surprising does (better outcome).
2. **A baseline-calibrated artifact:** the winning kernel vs torch SDPA /
   FlexAttention / FlashInfer at this exact shape (EXPERIMENT.md
   milestone-2 baseline lane).
3. **A fourth experiment arm:** agent-authors-IR vs agent-authors-CUDA
   vs exhaustive search, same verifier and budget — CAKE's core ablation,
   reproduced end-to-end with the IR the paper didn't publish.
4. **A cost-model training set** (spec → measured us) for free.

## Non-goals (v0)

Tensor-core-maximal paths (tcgen05/TMEM — the kernel is bandwidth-bound;
revisit for prefill), paged KV, variable L across the batch, stream-K
assignment, fused combine, portability off sm_90/sm_100. Each is an IR
extension when its family member arrives, not v0 scope.

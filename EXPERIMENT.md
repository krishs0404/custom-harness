# Experiment: does the medium matter?

A toy-scale replication of CAKE's controlled ablation (arXiv 2608.12629,
§clean-start experiment) with off-the-shelf kernel representations instead
of a proprietary IR.

## Hypothesis

CAKE showed that with the model and budget held fixed, an agent authoring
their typed schedule IR reached 1.144x a tuned baseline while an agent
writing raw CUDA/PTX reached 0.928x. If the effect is real and transfers,
agents writing a hardware-explicit DSL (CuTe-DSL) should plateau higher
and faster than agents writing raw CUDA, with a tile DSL (Triton) in
between — and the gap should widen with kernel difficulty. If the arms
tie, the published effect may not transfer off NVIDIA's in-house IR.
Either outcome is a finding.

## Arms (per kernel, identical everything except the medium)

| Arm | Medium | Notes |
|---|---|---|
| A | Raw CUDA C++ | existing harness path (`kernels/*.cu`) |
| B | Triton | tile abstraction, no explicit warp/TMA control |
| C | CuTe-DSL (Python) | hardware-explicit; the medium FA4 itself is written in |

Held fixed across arms: agent model, prompt scaffold (adapted only for
medium syntax), measurement budget, stop rule, GPU (Modal B200), shapes,
verifier, timing policy. The agent may not see other arms' transcripts.

## Kernels (ascending difficulty)

1. **transpose** — square n=8195 FP32 (arm A is already complete: 9 runs,
   plateau 4634 GB/s = ~77% of copy proxy; see results.jsonl + git
   history; reuse as-is, do not re-run).
2. **row softmax** — 8192 rows x 8192 cols, BF16 in/out, FP32 accumulate.
   Online (single-pass) algorithms legal; tolerance set from a CPU
   FP64 reference (proposal: max abs error <= 2e-3, to be pinned down
   during task authoring, before any arm runs).
3. **GQA decode attention** — the winnable-slice toy: batch 4, 32 query
   heads / 8 KV heads, head_dim 128, KV length 8192, q_len 1, BF16
   in/out FP32 accumulate, contiguous KV cache. One fixed shape,
   deliberately. CPU FP64 reference; tolerance pinned during authoring.

Shapes/tolerances/bytes-moved formulas are the problem statement: frozen
in a task-definition commit BEFORE any optimization run, per README
contract.

## Protocol

- **Budget:** 10 measured attempts per arm per kernel (a `make run` that
  prints status ok or a validation failure both count as one attempt;
  compile errors do not count but are logged). Budget is enforced by the
  driver (code, not prompt) — see infrastructure below.
- **Stop rule:** two consecutive attempts with <2% improvement on
  median_us, or budget exhausted, whichever first.
- **Seeding:** every arm starts from the same naive reference
  implementation, ported to its medium and verified before the campaign
  starts (port is not counted against budget).
- **Metrics per arm x kernel:**
  1. best `median_us` at plateau, and as % of the relevant ceiling
     (copy/read/write proxies; for attention, roofline from
     bytes_moved + FLOPs);
  2. attempts-to-plateau;
  3. correctness incidents (validation failures, sanitizer findings);
  4. token cost of the campaign (from agent logs).
- **Timing policy:** identical for all arms — the harness's
  warm_same_buffer batched CUDA-event protocol. Python-medium candidates
  are timed by the same verifier (see lane below), never by
  triton.testing or torch timers.
- **Runs happen on Modal B200** via the driver; one arm at a time
  (no interleaving, to keep GPU-allocation variance comparable).

## Infrastructure additions (build before arm B/C runs)

1. **DSL candidate lane:** a way for the C++ verifier to time and
   validate a candidate that is a Python-defined kernel. Design intent
   (simplest honest version): the candidate process precomputes nothing;
   a thin Python shim loads inputs written by the harness (or
   regenerates from the fixed seed), launches the kernel, and the
   poison/validate/re-validate + JSON protocol is reproduced exactly.
   The shim is verifier code (agents may not edit it). Alternative if
   shim fidelity is in doubt: candidates AOT-compile to a .so the C++
   harness dlopens. Decide during implementation; document which.
2. **Code-enforced budget:** driver counts measured attempts per
   arm x kernel and refuses beyond 10 (roadmap item, now justified —
   arms must have identical budgets for the comparison to mean
   anything).
3. **Baseline lane (attention only, context not competition):** one-off
   measurement of FlashInfer / FA at the GQA shape for calibration of
   how far the toy winners are from production kernels. Not an arm.

## Threats to validity (accepted at toy scale, but log them)

- **Measured-peak wobble across Modal containers:** compare on
  median_us; record gbps and ceilings per run (already in JSON).
- **Anti-cheat asymmetry:** the Python lane must enforce the same
  poison/re-validate/device-sync rules or arm B/C wins are suspect.
- **One model, one budget point:** this is a pilot; no claim about
  other models or budget scaling.
- **Agent prompt-tuning bias:** write all three prompt scaffolds before
  running any arm; freeze them in a commit.
- **n=1 campaigns per cell:** if a result looks surprising, re-run that
  cell once before believing it (CAKE ran 3 per arm; we accept noisier
  conclusions at toy scale).

## Cost estimate

~90 Modal B200 runs x ~1 min ≈ $10-15 GPU, plus agent tokens
(transpose campaign was ~120k tokens/round; budget ~$100-200 total).

## Deliverable

A short writeup: the 3x3 table (best % of ceiling, attempts-to-plateau,
incidents), attempt trajectories from results.jsonl notes, and an honest
paragraph on whether the medium effect appeared. Publish alongside the
repo if the result is interesting either way.

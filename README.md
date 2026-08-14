# Quick CUDA harness

One self-contained `.cu` per kernel problem under `kernels/`, one shared
verifier header (`harness.cuh`), and a Makefile. C++17 + CUDA + Make, nothing
else.

## The loop

```bash
make lock-clocks                 # once per box/reboot: biggest noise reducer
make run KERNEL=vector_add       # verify, time, log; N=16777219 ARCH=sm_100a etc.
make profile KERNEL=vector_add   # ncu speed-of-light: WHY is it slow?
make sanitize KERNEL=vector_add  # memcheck + racecheck before trusting a win
```

Edit `kernels/<name>.cu`, `make run`, read the numbers, `make profile` when
you need to know why, edit again. Every successful `make run` appends one
JSON line (median, GB/s, % of peak, git SHA, timestamp) to `results.jsonl` —
the optimization trajectory is a file, not a memory. Label attempts with
`NOTE="64x64 padded tile"`; the note lands in the JSON line, making the log
a self-documenting attempt table.

`make run` output ends with the speed-of-light line, for example:

```
median 116.090 us  4628.0 GB/s  77.0% of measured peak (copy 6011.0, read 6300.2, write 5850.7 GB/s)
```

No GPU on this machine? `modal/run_kernel.py` runs the same targets on a
Modal GPU (default B200) and appends the JSON line to the **local**
`results.jsonl`, so history survives the ephemeral container:

```bash
python3 -m modal run modal/run_kernel.py --kernel transpose --n 8195 --note "attempt label"
```

The driver is optional infrastructure, not the verifier — timing and
verification live entirely in `harness.cuh` + the Makefile.

## When are you done?

For memory-bound kernels (most of them): **`pct_measured_peak` ≳ 90–95%
means you are at the wall — stop.** The measured peak is a saturating
device-to-device copy run in the same process; the gap between it and
`pct_theoretical_peak` (datasheet clock × bus width) is vendor/physics tax,
not your problem. A perfectly coalesced streaming kernel can land slightly
above 100% of the copy proxy; that also means done.

The JSON also reports **directional ceilings** (`read_peak_gbps` from a
load-only sweep, `write_peak_gbps` from a memset). Judge a kernel against
the mix it actually performs: one that streams reads but scatters writes is
bounded by something between the two, and the copy number alone will
under-explain the gap. When one direction's achieved bandwidth is near its
ceiling and the other is far below, the far one is the lever — that is a
profiler-grade diagnosis with no profiler, which matters on hosts (Modal)
that block ncu's counters.

If the percentage is low, `make profile` says why:

- **Memory throughput high in ncu too** → genuinely bandwidth-bound; only
  moving fewer bytes (fusion, better data types) helps.
- **Both compute and memory throughput below ~60%** → latency-bound: fix
  occupancy, launch configuration, or access patterns first. Roofline
  reasoning is not valid until the kernel has enough parallelism in flight.
- **Compute throughput high** → compute-bound; bandwidth SOL is the wrong
  ceiling for this kernel (FLOP-side speed-of-light is deliberately deferred
  from this harness for now).

Profiler numbers are for diagnosis only. Only `make run` numbers count:
ncu locks clocks, flushes caches, and serializes launches, so timings taken
under it are not comparable.

## Adding a kernel

Copy `kernels/vector_add.cu`, replace the kernel, the CPU reference, and the
`bytes_moved` formula, then `make run KERNEL=<name>`. That is the entire
registration story.

## Contract for agents (Sol Ultra, Fable — read this)

- You may edit files under `kernels/` only. Do not touch `harness.cuh`, the
  `Makefile`, `results.jsonl`, or `modal/`.
- Label every measured attempt: `NOTE="what changed"` (or `--note` via the
  Modal driver). Unlabeled log lines are indistinguishable later.
- Success is `make run` printing `"status":"ok"` with a lower `median_us`
  than the previous entry in `results.jsonl` for the same kernel and N.
- The output buffer is NaN-poisoned before correctness checking **and
  re-poisoned and re-validated after timing**; timing does a device-wide
  sync before the stop event. Stale outputs, partial writes, and
  side-stream tricks fail or are counted.
- The CPU reference and the `bytes_moved` formula are the problem statement,
  not the optimization surface. Keep them honest.
- `make profile` output is for reasoning; only `make run` numbers count.

## Measurement policy and caveats

- Reported policy is `warm_same_buffer`: batched, same-stream CUDA-event
  timing over a hot cache, median of 15 samples × 100 launches. Cold-cache
  and production-traffic measurements are separate questions, deliberately
  deferred.
- `WARNING: GPU clocks not locked` means run-to-run noise is on you;
  `make lock-clocks` (uses `nvidia-smi --lock-gpu-clocks=tdp,tdp`, plus
  memory clocks where supported) fixes it. Clock status is recorded in each
  JSON line.
- Exit codes: `0` ok, `1` wrong answer or error, `77` no usable GPU (skip),
  `2` sanitizer findings.
- `make profile` requires direct hardware access. Sandboxed GPU providers
  (verified on Modal: root, `RmProfilingAdminOnly: 0`, yet ncu still fails
  with "failed to prepare kernel") do not pass the profiler's device ioctls
  through their runtime, and hardware counters are a cross-tenant
  side-channel besides. Profile on bare metal or a full VM; on shared hosts,
  the directional ceilings above are the fallback diagnosis.

## Roadmap (known, deliberately deferred)

- Multi-shape correctness sweep (`make check` over several N) — anti-
  overfitting once agents run less supervised.
- Optional per-kernel library baseline (cuBLAS/CUB) timed under identical
  conditions — the true "can we do better" ceiling.
- FLOP-side speed-of-light for compute-bound kernels.
- GPU UUID + live SM clock in the JSON line, to attribute run-to-run
  denominator wobble.
- Code-enforced agent run budgets — an orchestration concern that stays
  outside this repo by design.

The design borrows the small, load-bearing ideas from recent work:
verifier-first execution and deterministic control from
[KernelFalcon](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/),
varied correctness conditions from [robust-kbench](https://sakana.ai/ai-cuda-engineer/),
explicit cache policy and warmup from
[high-fidelity GPU benchmarking](https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/),
and correctness-gated measurement from
[SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench/blog/introducing-sol-execbench).
The anti-cheat details (candidate-first evaluation, poison-revalidate,
device-wide sync before the stop event) answer documented exploits from the
Sakana AI CUDA Engineer incident, Kevin-32B, and CUDA-L1.

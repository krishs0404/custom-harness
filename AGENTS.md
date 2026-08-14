# Agent guide

This repository is a CUDA kernel optimization harness designed to be
operated by coding agents. This file is the map; [README.md](README.md)
has the full story.

## The contract

- Edit files under `kernels/` ONLY. Never touch `harness.cuh`, `Makefile`,
  `modal/`, `results.jsonl`, or this file.
- Inside a kernel file, the CPU reference, the `bytes_moved` formula, the
  tolerance, and the seed are the problem statement — off-limits to
  cleverness. The kernel, its launch configuration, and everything else in
  the file are fair game.
- Success = `make run` printing `"status":"ok"` with a lower `median_us`
  than the previous `results.jsonl` entry for the same kernel and N. Only
  harness numbers count; profiler output is diagnosis, never scoring.
- Label every measured attempt with `NOTE="what changed"`.
- Anti-cheat is structural, so don't bother: outputs are NaN-poisoned and
  re-validated after timing, and timing does a device-wide sync before the
  stop event (side-stream work is counted).

## Commands

```bash
make run KERNEL=<name> [N=...] [NOTE="..."]   # verify + time + append to log
make profile KERNEL=<name>                    # ncu diagnosis (bare metal/VM only)
make profile-remote HOST=user@gpubox          # same, via SSH; CSV saved locally
make sanitize KERNEL=<name>                   # memcheck + racecheck
python3 -m modal run modal/run_kernel.py \
    --kernel <name> --n <N> --note "..."      # same loop on a Modal GPU;
                                              # persists the log locally
```

## Reading the output

- Memory-bound kernels: `pct_measured_peak` >= ~90% means done — stop.
- `read_peak_gbps` / `write_peak_gbps` are directional ceilings measured in
  the same process; judge a kernel against the traffic mix it actually
  performs, not just the copy number.
- If achieved bandwidth is far below every ceiling and ncu is unavailable,
  suspect latency/occupancy before hunting exotic memory effects.

## Adding a kernel

Copy `kernels/vector_add.cu`; replace the kernel, the CPU reference, and
the `bytes_moved` formula. That is the entire registration story.

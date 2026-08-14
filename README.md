# Quick CUDA harness

A tiny scratchpad for editing and timing one CUDA kernel. It is intentionally
separate from the repository's existing Python/ThunderKittens evaluator.

```bash
cd cuda_harness
make run                         # auto-detect the local GPU architecture
make run ARCH=sm_100a N=16777219 # example: B200 and a larger odd workload
make sanitize                    # slow correctness/memory check
```

Edit the kernel at the top of `vector_add.cu`, then run `make run` again. The
harness provides:

- fixed-seed CPU reference checking;
- NaN-poisoned output before and after timing;
- real warmup launches;
- batched, same-stream CUDA-event timing;
- median plus raw samples in a final JSON line;
- clear CUDA errors and a no-GPU skip exit (`77`).

The reported policy is explicitly `warm_same_buffer`; cold-cache or production
traffic measurements are separate questions and deliberately deferred.

The design borrows the small, load-bearing ideas from recent work: verifier-first
execution and deterministic control from [KernelFalcon](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/),
varied correctness conditions from [robust-kbench](https://sakana.ai/ai-cuda-engineer/),
explicit cache policy and warmup from [high-fidelity GPU benchmarking](https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/),
and correctness-gated measurement from [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench/blog/introducing-sol-execbench).

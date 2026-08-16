"""Baseline lane for the GQA decode task (EXPERIMENT.md milestone 2).

Times the incumbent attention implementations at the frozen shape
(B=4, H_q=32, H_kv=8, D=128, L=8192, BF16) on a Modal GPU and appends
one harness-style JSON line per baseline to the LOCAL results.jsonl.

This is driver infrastructure, not the verifier, and not an experiment
arm: baselines are context that defines what "fast" means at this shape.
Per the README contract, agents may not edit this file.

Timing mirrors the harness policy as closely as PyTorch allows: warmups,
batched launches per CUDA-event sample, device-wide sync before the stop
event, median of samples. Each implementation is checked against an FP32
reference before it is timed; a wrong baseline is reported, not timed.
FlashInfer is timed on its run phase only (plan happens once, outside).
"""

import json
import os
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
GPU = os.environ.get("HARNESS_GPU", "B200")

app = modal.App("gqa-baselines")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("flashinfer-python", "numpy")
)

# The frozen problem (must match EXPERIMENT.md / IR_DESIGN.md).
B, H_Q, H_KV, D, L = 4, 32, 8, 128, 8192
SEED = 20260813
TOLERANCE = 3.0e-2
# K + V mandatory traffic; Q and O are noise but counted for consistency.
BYTES_MOVED = 2 * B * H_KV * L * D * 2 + 2 * B * H_Q * D * 2


@app.function(image=image, gpu=GPU, timeout=1800)
def bench():
    import torch
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)

    # Layout [B, heads, seq, D] for torch APIs.
    q = torch.randn(B, H_Q, 1, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H_KV, L, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H_KV, L, D, device=device, dtype=torch.bfloat16)
    group = H_Q // H_KV

    # FP32 reference for correctness gating (not timing).
    with torch.no_grad():
        kf = k.float().repeat_interleave(group, dim=1)
        vf = v.float().repeat_interleave(group, dim=1)
        scores = (q.float() @ kf.transpose(-1, -2)) / (D ** 0.5)
        reference = (torch.softmax(scores, dim=-1) @ vf)

    def check(output):
        error = (output.float() - reference).abs().max().item()
        return error, error <= TOLERANCE

    def time_us(fn, warmups=10, samples=15, iters=20):
        for _ in range(warmups):
            fn()
        torch.cuda.synchronize()
        times = []
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(samples):
            start.record()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop) * 1000.0 / iters)
        times.sort()
        return times[len(times) // 2], times

    results = []

    def add(label, fn, note=""):
        try:
            out = fn()
            error, ok = check(out)
            if not ok:
                results.append({"kernel": f"gqa_decode_baseline_{label}",
                                "status": "wrong", "max_error": error})
                return
            median, samples_us = time_us(fn)
            results.append({
                "status": "ok",
                "kernel": f"gqa_decode_baseline_{label}",
                "n": L, "seed": SEED, "cache": "warm_same_buffer",
                "median_us": round(median, 3),
                "samples_us": [round(t, 3) for t in samples_us],
                "bytes_moved": BYTES_MOVED,
                "gbps": round(BYTES_MOVED / (median * 1e-6) / 1e9, 1),
                "max_error": round(error, 6),
                "clocks": "unlocked", "device": name, "note": note,
            })
        except Exception as exc:  # report, never abort the lane
            results.append({"kernel": f"gqa_decode_baseline_{label}",
                            "status": "error", "error": repr(exc)[:300]})

    # 1. Eager composition (the weak baseline of the literature).
    def eager():
        ke = k.repeat_interleave(group, dim=1)
        ve = v.repeat_interleave(group, dim=1)
        s = (q @ ke.transpose(-1, -2)) / (D ** 0.5)
        return torch.softmax(s, dim=-1) @ ve
    add("eager", eager, note=f"torch {torch.__version__}")

    # 2. torch SDPA, default dispatch and per-backend.
    add("sdpa_default",
        lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True),
        note=f"torch {torch.__version__}")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for backend, label in [(SDPBackend.CUDNN_ATTENTION, "sdpa_cudnn"),
                           (SDPBackend.FLASH_ATTENTION, "sdpa_flash"),
                           (SDPBackend.EFFICIENT_ATTENTION, "sdpa_efficient")]:
        def run(b=backend):
            with sdpa_kernel([b]):
                return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        add(label, run, note=f"torch {torch.__version__}")

    # 3. FlexAttention (compiled; compile cost excluded via warmups).
    try:
        from torch.nn.attention.flex_attention import flex_attention
        flex = torch.compile(flex_attention, dynamic=False)
        add("flex", lambda: flex(q, k, v, enable_gqa=True),
            note=f"torch {torch.__version__} compiled")
    except Exception as exc:
        results.append({"kernel": "gqa_decode_baseline_flex",
                        "status": "error", "error": repr(exc)[:300]})

    # 4. FlashInfer batch decode (paged wrapper over contiguous KV;
    #    plan() once outside the timed region, run() timed).
    try:
        import flashinfer
        page = 16
        pages_per_seq = L // page
        # [B, H_kv, L, D] -> [total_pages, page, H_kv, D] (NHD layout)
        paged_k = (k.permute(0, 2, 1, 3).reshape(B * pages_per_seq, page, H_KV, D)
                   .contiguous())
        paged_v = (v.permute(0, 2, 1, 3).reshape(B * pages_per_seq, page, H_KV, D)
                   .contiguous())
        indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * pages_per_seq
        indices = torch.arange(0, B * pages_per_seq, device=device, dtype=torch.int32)
        last_len = torch.full((B,), page, device=device, dtype=torch.int32)
        workspace = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)
        fi_q = q.reshape(B, H_Q, D)
        # Two configs: default, and the tensor-core path FlashInfer
        # recommends for GQA (grouped heads raise arithmetic intensity).
        for tc in (False, True):
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace, "NHD", use_tensor_cores=tc)
            wrapper.plan(indptr, indices, last_len, H_Q, H_KV, D, page,
                         q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            def flashinfer_run(w=wrapper):
                return w.run(fi_q, (paged_k, paged_v)).reshape(B, H_Q, 1, D)
            add(f"flashinfer{'_tc' if tc else ''}", flashinfer_run,
                note=f"flashinfer {flashinfer.__version__} page={page} "
                     f"tensor_cores={tc} run-only")
    except Exception as exc:
        results.append({"kernel": "gqa_decode_baseline_flashinfer",
                        "status": "error", "error": repr(exc)[:300]})

    return results


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    rows = bench.remote()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    appended = 0
    with open(REPO / "results.jsonl", "a") as log:
        for row in rows:
            if row.get("status") == "ok":
                row["git_sha"] = "baseline-lane"
                row["dirty"] = False
                row["timestamp"] = stamp
                log.write(json.dumps(row) + "\n")
                appended += 1
                print(f"{row['kernel']:38s} {row['median_us']:>10.3f} us  "
                      f"{row['gbps']:>7.1f} GB/s  (max_error {row['max_error']})")
            else:
                print(f"{row['kernel']:38s} {row['status'].upper()}: "
                      f"{row.get('error', row.get('max_error', ''))}")
    print(f"[baseline-lane] appended {appended} line(s) to results.jsonl")

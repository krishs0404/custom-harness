"""Driver: run the harness on a Modal GPU and persist results locally.

This is optional infrastructure, not the verifier — the verifier is
harness.cuh + the Makefile, and they never depend on this file. Per the
README contract, agents may not edit this either.

Usage (from the repo root or anywhere):

    python3 -m modal run modal/run_kernel.py --kernel transpose --n 8195 \
        --note "64x64 padded tile" [--sanitize]

Every successful run's JSON line is appended to the LOCAL results.jsonl,
so the attempt history survives the ephemeral container. Set HARNESS_GPU
to override the GPU type (default B200; needs a CUDA image new enough for
the target arch).
"""

import os
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
GPU = os.environ.get("HARNESS_GPU", "B200")

app = modal.App("cuda-harness")

image = (
    # B200 (Blackwell, sm_100) needs CUDA 12.8+.
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("make")
    .add_local_dir(
        str(REPO),
        remote_path="/repo",
        ignore=["build/**", ".git/**", "results.jsonl", "modal/**"],
    )
)


@app.function(image=image, gpu=GPU, timeout=900)
def run_remote(kernel: str, n: str, note: str, sanitize: bool):
    import subprocess

    env = dict(os.environ)
    if note:
        env["HARNESS_NOTE"] = note
    cmd = f"cd /repo && make run KERNEL={kernel}"
    if n:
        cmd += f" N={n}"
    if sanitize:
        cmd += f" && make sanitize KERNEL={kernel}"
    proc = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, env=env
    )
    return {"exit": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


@app.local_entrypoint()
def main(kernel: str = "vector_add", n: str = "", note: str = "", sanitize: bool = False):
    result = run_remote.remote(kernel, n, note, sanitize)
    print(result["stderr"])
    print(result["stdout"])

    json_lines = [
        line
        for line in result["stdout"].splitlines()
        if line.startswith('{"status":"ok"')
    ]
    if result["exit"] == 0 and json_lines:
        with open(REPO / "results.jsonl", "a") as log:
            for line in json_lines:
                log.write(line + "\n")
        print(f"[driver] appended {len(json_lines)} line(s) to {REPO / 'results.jsonl'}")
    else:
        print(f"[driver] exit {result['exit']}; nothing appended to results.jsonl")
    if result["exit"] != 0:
        raise SystemExit(result["exit"])

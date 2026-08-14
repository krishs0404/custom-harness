SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

NVCC ?= nvcc
NCU ?= ncu
ARCH ?= native
KERNEL ?= vector_add
N ?=
NOTE ?=
GPU_CLOCK ?=
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo

GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
GIT_DIRTY := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo 1 || echo 0)

BIN := build/$(KERNEL)

.PHONY: all build run profile sanitize lock-clocks unlock-clocks clean

all: build

build: $(BIN)

$(BIN): kernels/$(KERNEL).cu harness.cuh
	@command -v $(NVCC) >/dev/null || { \
	  echo "error: $(NVCC) not found; this repo builds on the GPU box, not this machine" >&2; exit 1; }
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -I. \
	  -DHARNESS_KERNEL_NAME='"$(KERNEL)"' \
	  -DHARNESS_GIT_SHA='"$(GIT_SHA)"' \
	  -DHARNESS_GIT_DIRTY=$(GIT_DIRTY) \
	  kernels/$(KERNEL).cu -o $(BIN)

# Correctness gate + timing. Human output on stderr; the one JSON line on
# stdout is appended to results.jsonl. Clock status is advisory metadata:
# the stamp file records our own lock action, and the live SM clock is
# cross-checked in case someone rebooted or unlocked since.
run: build
	@clocks=unknown; \
	if command -v nvidia-smi >/dev/null; then \
	  clocks=unlocked; \
	  if [ -f build/.clocks_locked ]; then \
	    want=$$(cat build/.clocks_locked); \
	    have=$$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1); \
	    if [ "$$have" -eq "$$have" ] 2>/dev/null; then \
	      diff=$$(( want > have ? want - have : have - want )); \
	      if [ $$diff -le 100 ]; then clocks=locked; \
	      else echo "WARNING: clock stamp ($$want MHz) != live SM clock ($$have MHz); treating as unlocked" >&2; fi; \
	    fi; \
	  fi; \
	  if [ "$$clocks" = unlocked ]; then \
	    echo "WARNING: GPU clocks not locked; run 'make lock-clocks' for stable numbers" >&2; \
	  fi; \
	fi; \
	HARNESS_CLOCKS=$$clocks HARNESS_NOTE="$${HARNESS_NOTE:-$(NOTE)}" ./$(BIN) $(N) | tee -a results.jsonl

# Diagnosis only, never timing: HARNESS_MODE=check skips the timing loop and
# JSON, and --launch-skip 1 skips the correctness launch, so ncu profiles one
# plain launch. Speed-of-light + memory sections answer "what is the limiter".
profile: build
	@command -v $(NCU) >/dev/null || { \
	  echo "error: $(NCU) not found. Install Nsight Compute; if counters are blocked" >&2; \
	  echo "(ERR_NVGPUCTRPERM), set NVreg_RestrictProfilingToAdminUsers=0 or run via sudo" >&2; exit 1; }
	HARNESS_MODE=check $(NCU) --section SpeedOfLight --section MemoryWorkloadAnalysis \
	  --launch-skip 1 --launch-count 1 --csv ./$(BIN) $(N)

# Small odd N: fast, and still exercises the tail path.
sanitize: build
	HARNESS_MODE=check compute-sanitizer --tool memcheck --error-exitcode=2 ./$(BIN) 257
	HARNESS_MODE=check compute-sanitizer --tool racecheck --error-exitcode=2 ./$(BIN) 257

# tdp is nvidia-smi's keyword for the sustainable clock (nvbench's default
# recommendation); pass GPU_CLOCK=<MHz> to pin a specific rate instead.
# Memory clocks matter most for bandwidth numbers but locking them is not
# supported everywhere, hence the soft failure.
lock-clocks:
	@command -v nvidia-smi >/dev/null || { echo "error: nvidia-smi not found" >&2; exit 1; }
	@if [ -n "$(GPU_CLOCK)" ]; then \
	  sudo nvidia-smi --lock-gpu-clocks=$(GPU_CLOCK),$(GPU_CLOCK); \
	else \
	  sudo nvidia-smi --lock-gpu-clocks=tdp,tdp; \
	fi
	@mem=$$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits 2>/dev/null | head -1); \
	if [ "$$mem" -eq "$$mem" ] 2>/dev/null; then \
	  sudo nvidia-smi --lock-memory-clocks=$$mem,$$mem \
	    || echo "note: memory clock locking unsupported on this GPU" >&2; \
	fi
	@mkdir -p build; \
	nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1 > build/.clocks_locked; \
	echo "locked; stamped $$(cat build/.clocks_locked) MHz" >&2

unlock-clocks:
	@sudo nvidia-smi --reset-gpu-clocks; \
	sudo nvidia-smi --reset-memory-clocks 2>/dev/null || true; \
	rm -f build/.clocks_locked

# ncu needs direct hardware access (sandboxed providers block it — README).
# Runs the profile on a remote GPU box over SSH and saves the CSV locally.
# Usage: make profile-remote HOST=user@gpubox [SSH_KEY=~/.ssh/key] [KERNEL=] [N=]
HOST ?=
SSH_KEY ?=
SSH_FLAGS := $(if $(SSH_KEY),-i $(SSH_KEY) -o IdentitiesOnly=yes)

profile-remote:
	@test -n "$(HOST)" || { echo "usage: make profile-remote HOST=user@gpubox [SSH_KEY=...]" >&2; exit 1; }
	@mkdir -p build
	rsync -az -e "ssh $(SSH_FLAGS)" --exclude build --exclude .git \
	  --exclude results.jsonl ./ $(HOST):.cuda-harness-remote/
	ssh $(SSH_FLAGS) $(HOST) 'cd .cuda-harness-remote && make build KERNEL=$(KERNEL) >/dev/null \
	  && sudo HARNESS_MODE=check $$(command -v ncu) --section SpeedOfLight \
	  --section MemoryWorkloadAnalysis --launch-skip 1 --launch-count 1 \
	  --csv ./build/$(KERNEL) $(N)' | tee build/$(KERNEL)-profile.csv >/dev/null
	@echo "saved build/$(KERNEL)-profile.csv" >&2

# results.jsonl survives clean: it is measurement history, not a build product.
clean:
	rm -rf build

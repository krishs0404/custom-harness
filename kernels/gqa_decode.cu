#include "harness.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// GQA decode attention, the frozen task from EXPERIMENT.md / IR_DESIGN.md:
// q_len = 1, B=4 sequences, 32 query heads sharing 8 KV heads (G=4),
// head_dim 128, BF16 in/out with FP32 accumulation, contiguous KV cache
// [B, H_kv, L, D]. N (the harness size argument) is the KV length L;
// the frozen benchmark value is L=8192, small L exercises tails/sanitize.
//
// Mandatory traffic is K+V read once: 2*B*H_kv*L*D*2 bytes = 134.2 MB at
// L=8192 -> ~21.5 us speed of light on B200. Measured baselines (see
// results.jsonl): cuDNN SDPA ~29.4 us, FlashInfer tensor-core ~34.5 us.
//
// This is the NAIVE rung: one thread block per (b, q_head), so each KV
// head is re-read G=4 times (4x mandatory traffic), rows are processed
// one at a time with a shared-memory dot-product reduction, and there is
// no split-KV, no pipelining, no vectorized loads. Every one of those
// sins is a rung on the optimization ladder.
constexpr int kBatch = 4;
constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kGroup = kQueryHeads / kKvHeads;
constexpr int kDim = 128;

__global__ void gqa_decode(
    const __nv_bfloat16* __restrict__ q,   // [B, H_q, D]
    const __nv_bfloat16* __restrict__ k,   // [B, H_kv, L, D]
    const __nv_bfloat16* __restrict__ v,   // [B, H_kv, L, D]
    __nv_bfloat16* __restrict__ output,    // [B, H_q, D]
    size_t length) {
    const int batch = blockIdx.x / kQueryHeads;
    const int q_head = blockIdx.x % kQueryHeads;
    const int kv_head = q_head / kGroup;
    const int d = threadIdx.x;  // one thread per dimension, blockDim.x == kDim

    const __nv_bfloat16* k_head =
        k + ((size_t{batch} * kKvHeads + kv_head) * length) * kDim;
    const __nv_bfloat16* v_head =
        v + ((size_t{batch} * kKvHeads + kv_head) * length) * kDim;
    const float q_d =
        __bfloat162float(q[(size_t{batch} * kQueryHeads + q_head) * kDim + d]);
    const float scale = rsqrtf(static_cast<float>(kDim));

    __shared__ float reduction[kDim];

    // Online softmax; m/l are replicated across threads (identical values),
    // the accumulator is distributed one dimension per thread.
    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;

    for (size_t row = 0; row < length; ++row) {
        reduction[d] = q_d * __bfloat162float(k_head[row * kDim + d]);
        __syncthreads();
        for (int stride = kDim / 2; stride > 0; stride /= 2) {
            if (d < stride) {
                reduction[d] += reduction[d + stride];
            }
            __syncthreads();
        }
        const float score = reduction[0] * scale;
        __syncthreads();  // everyone has read reduction[0]; safe to reuse

        const float new_m = fmaxf(m, score);
        const float correction = __expf(m - new_m);
        const float p = __expf(score - new_m);
        l = l * correction + p;
        acc = acc * correction +
              p * __bfloat162float(v_head[row * kDim + d]);
        m = new_m;
    }

    output[(size_t{batch} * kQueryHeads + q_head) * kDim + d] =
        __float2bfloat16(acc / l);
}

int main(int argc, char** argv) {
    const size_t length = argc > 1 ? std::stoull(argv[1]) : 8192;
    constexpr uint32_t seed = 20260813;

    try {
        cuda_harness::require_gpu_or_exit();
        if (length == 0) {
            throw std::invalid_argument("L must be greater than zero");
        }

        // Fixed-seed inputs and an FP64 CPU reference over the
        // BF16-rounded values. These define the problem; keep them honest.
        const size_t q_count = size_t{kBatch} * kQueryHeads * kDim;
        const size_t kv_count = size_t{kBatch} * kKvHeads * length * kDim;
        std::mt19937 generator(seed);
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        std::vector<__nv_bfloat16> q(q_count);
        std::vector<__nv_bfloat16> k(kv_count);
        std::vector<__nv_bfloat16> v(kv_count);
        for (auto* tensor : {&q, &k, &v}) {
            for (auto& value : *tensor) {
                value = __float2bfloat16(distribution(generator));
            }
        }

        std::vector<__nv_bfloat16> expected(q_count);
        std::vector<double> scores(length);
        for (int batch = 0; batch < kBatch; ++batch) {
            for (int q_head = 0; q_head < kQueryHeads; ++q_head) {
                const int kv_head = q_head / kGroup;
                const __nv_bfloat16* q_ptr =
                    &q[(size_t{batch} * kQueryHeads + q_head) * kDim];
                const __nv_bfloat16* k_head =
                    &k[(size_t{batch} * kKvHeads + kv_head) * length * kDim];
                const __nv_bfloat16* v_head =
                    &v[(size_t{batch} * kKvHeads + kv_head) * length * kDim];

                double max_score = -1.0e30;
                for (size_t row = 0; row < length; ++row) {
                    double dot = 0.0;
                    for (int d = 0; d < kDim; ++d) {
                        dot += static_cast<double>(__bfloat162float(q_ptr[d])) *
                               static_cast<double>(
                                   __bfloat162float(k_head[row * kDim + d]));
                    }
                    scores[row] = dot / std::sqrt(static_cast<double>(kDim));
                    max_score = std::max(max_score, scores[row]);
                }
                double sum = 0.0;
                for (size_t row = 0; row < length; ++row) {
                    scores[row] = std::exp(scores[row] - max_score);
                    sum += scores[row];
                }
                for (int d = 0; d < kDim; ++d) {
                    double out = 0.0;
                    for (size_t row = 0; row < length; ++row) {
                        out += scores[row] *
                               static_cast<double>(
                                   __bfloat162float(v_head[row * kDim + d]));
                    }
                    expected[(size_t{batch} * kQueryHeads + q_head) * kDim + d] =
                        __float2bfloat16(static_cast<float>(out / sum));
                }
            }
        }

        cuda_harness::DeviceBuffer<__nv_bfloat16> device_q(q_count);
        cuda_harness::DeviceBuffer<__nv_bfloat16> device_k(kv_count);
        cuda_harness::DeviceBuffer<__nv_bfloat16> device_v(kv_count);
        cuda_harness::DeviceBuffer<__nv_bfloat16> device_output(q_count);
        CUDA_CHECK(cudaMemcpy(device_q.data(), q.data(),
                              q_count * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_k.data(), k.data(),
                              kv_count * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_v.data(), v.data(),
                              kv_count * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));

        auto launch = [&] {
            gqa_decode<<<kBatch * kQueryHeads, kDim>>>(
                device_q.data(), device_k.data(), device_v.data(),
                device_output.data(), length);
        };

        cuda_harness::RunConfig config;
        config.n = length;
        config.seed = seed;
        // Mandatory traffic: K+V read once, plus Q and O. A schedule that
        // re-reads KV (like this naive one) shows up as low gbps/pct —
        // that is the point, not an accounting error.
        config.bytes_moved =
            2.0 * kv_count * sizeof(__nv_bfloat16) +
            2.0 * q_count * sizeof(__nv_bfloat16);
        // Frozen per EXPERIMENT.md: BF16 output quantization alone is
        // ~4e-4 absolute at these magnitudes; measured baseline kernels
        // (cuDNN, FlashInfer, FA) land at max error ~3e-4 vs an FP32
        // reference. 3e-3 gives ~7x margin for FP32-accumulation
        // reordering while still failing any actually-wrong result.
        config.tolerance = 3.0e-3;

        return cuda_harness::run_and_report(
            config, launch, device_output.data(), expected);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

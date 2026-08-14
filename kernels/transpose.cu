#include "harness.cuh"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Naive n x n transpose: reads are coalesced (consecutive threads read
// consecutive input), writes are strided by n. This is the classic starting
// rung — the optimization ladder is shared-memory tiles, then padding the
// tile to kill bank conflicts. Edit this kernel and its launch configuration
// while experimenting.
__global__ void transpose(const float* input, float* output, size_t n) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

int main(int argc, char** argv) {
    // N is the matrix side. An odd default exercises the tile tail paths;
    // on big-L2 GPUs (B200: 126 MB) prefer N >= 8195 so the working set
    // does not fit in cache.
    const size_t n = argc > 1 ? std::stoull(argv[1]) : 4099;
    constexpr uint32_t seed = 20260813;
    const dim3 threads(32, 8);

    try {
        cuda_harness::require_gpu_or_exit();
        if (n == 0) {
            throw std::invalid_argument("N must be greater than zero");
        }

        // Fixed-seed input and a CPU reference. These define the problem;
        // keep them honest.
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<float> input(n * n);
        std::vector<float> expected(n * n);
        for (size_t i = 0; i < n * n; ++i) {
            input[i] = distribution(generator);
        }
        for (size_t row = 0; row < n; ++row) {
            for (size_t col = 0; col < n; ++col) {
                expected[col * n + row] = input[row * n + col];
            }
        }

        cuda_harness::DeviceBuffer<float> device_input(n * n);
        cuda_harness::DeviceBuffer<float> device_output(n * n);
        CUDA_CHECK(cudaMemcpy(
            device_input.data(), input.data(), n * n * sizeof(float),
            cudaMemcpyHostToDevice));

        const auto blocks_for = [](size_t extent, unsigned block) {
            return static_cast<unsigned>((extent + block - 1) / block);
        };
        const dim3 blocks(blocks_for(n, threads.x), blocks_for(n, threads.y));
        auto launch = [&] {
            transpose<<<blocks, threads>>>(
                device_input.data(), device_output.data(), n);
        };

        cuda_harness::RunConfig config;
        config.n = n;
        config.seed = seed;
        // One float read and one float write per element.
        config.bytes_moved = 2.0 * static_cast<double>(n * n) * sizeof(float);
        // A transpose moves values without arithmetic, so it must be exact.
        config.tolerance = 0.0f;

        return cuda_harness::run_and_report(
            config, launch, device_output.data(), expected);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

#include "harness.cuh"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Tiled n x n transpose: a TILE x TILE shared-memory tile staged so both
// global reads and global writes are coalesced. The tile is padded to
// [TILE][TILE + 1] so the transposed shared-memory access pattern is
// bank-conflict free. Each TILE x BLOCK_ROWS thread block is coarsened:
// every thread loads/stores TILE / BLOCK_ROWS rows of the tile. Bounds
// checks on both phases handle the n=8195 tail tiles.
constexpr int TILE = 64;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose(
    const float* __restrict__ input, float* __restrict__ output, size_t n) {
    __shared__ float tile[TILE][TILE + 1];

    const size_t x_in = blockIdx.x * size_t{TILE} + threadIdx.x;
    const size_t y_in = blockIdx.y * size_t{TILE} + threadIdx.y;

    if (x_in < n) {
#pragma unroll
        for (int j = 0; j < TILE; j += BLOCK_ROWS) {
            if (y_in + j < n) {
                tile[threadIdx.y + j][threadIdx.x] = input[(y_in + j) * n + x_in];
            }
        }
    }

    __syncthreads();

    const size_t x_out = blockIdx.y * size_t{TILE} + threadIdx.x;
    const size_t y_out = blockIdx.x * size_t{TILE} + threadIdx.y;

    if (x_out < n) {
#pragma unroll
        for (int j = 0; j < TILE; j += BLOCK_ROWS) {
            if (y_out + j < n) {
                output[(y_out + j) * n + x_out] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    // N is the matrix side. An odd default exercises the tile tail paths;
    // on big-L2 GPUs (B200: 126 MB) prefer N >= 8195 so the working set
    // does not fit in cache.
    const size_t n = argc > 1 ? std::stoull(argv[1]) : 4099;
    constexpr uint32_t seed = 20260813;
    const dim3 threads(TILE, BLOCK_ROWS);

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
        // Each block covers a TILE x TILE tile regardless of blockDim.y.
        const dim3 blocks(blocks_for(n, TILE), blocks_for(n, TILE));
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

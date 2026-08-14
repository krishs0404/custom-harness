#include "harness.cuh"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Edit this kernel and its launch configuration while experimenting.
__global__ void vector_add(
    const float* x,
    const float* y,
    float* output,
    size_t n) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = x[index] + y[index];
    }
}

int main(int argc, char** argv) {
    // An odd default size exercises the kernel's tail path.
    const size_t n = argc > 1 ? std::stoull(argv[1]) : (1u << 20) + 3;
    constexpr uint32_t seed = 20260813;
    constexpr int threads = 256;

    try {
        cuda_harness::require_gpu_or_exit();
        if (n == 0) {
            throw std::invalid_argument("N must be greater than zero");
        }

        // Fixed-seed inputs and a CPU reference. These define the problem;
        // keep them honest.
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<float> x(n);
        std::vector<float> y(n);
        std::vector<float> expected(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = distribution(generator);
            y[i] = distribution(generator);
            expected[i] = x[i] + y[i];
        }

        cuda_harness::DeviceBuffer<float> device_x(n);
        cuda_harness::DeviceBuffer<float> device_y(n);
        cuda_harness::DeviceBuffer<float> device_output(n);
        CUDA_CHECK(cudaMemcpy(
            device_x.data(), x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            device_y.data(), y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        const int blocks = static_cast<int>((n + threads - 1) / threads);
        auto launch = [&] {
            vector_add<<<blocks, threads>>>(
                device_x.data(), device_y.data(), device_output.data(), n);
        };

        cuda_harness::RunConfig config;
        config.n = n;
        config.seed = seed;
        // Two float reads and one float write per element.
        config.bytes_moved = 3.0 * static_cast<double>(n) * sizeof(float);
        config.tolerance = 1.0e-6f;

        return cuda_harness::run_and_report(
            config, launch, device_output.data(), expected);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

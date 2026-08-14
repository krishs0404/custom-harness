#include "harness.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
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

bool validate(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance = 1.0e-6f) {
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(actual[i]) || error > tolerance) {
            std::cerr << "wrong value at " << i << ": got " << actual[i]
                      << ", expected " << expected[i] << '\n';
            return false;
        }
    }
    std::cout << "correct (max error " << max_error << ")\n";
    return true;
}

int main(int argc, char** argv) {
    // An odd default size exercises the kernel's tail path.
    const size_t n = argc > 1 ? std::stoull(argv[1]) : (1u << 20) + 3;
    constexpr uint32_t seed = 20260813;
    constexpr int threads = 256;
    constexpr int samples = 15;
    constexpr int launches_per_sample = 100;

    try {
        if (n == 0) {
            throw std::invalid_argument("N must be greater than zero");
        }

        int device_count = 0;
        const cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (device_status == cudaErrorNoDevice ||
            device_status == cudaErrorInsufficientDriver ||
            device_count == 0) {
            std::cerr << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(device_status);

        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<float> x(n);
        std::vector<float> y(n);
        std::vector<float> expected(n);
        std::vector<float> output(n);
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

        // NaN poisoning catches missing or partial writes before timing.
        CUDA_CHECK(cudaMemset(device_output.data(), 0xff, n * sizeof(float)));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            output.data(), device_output.data(), n * sizeof(float), cudaMemcpyDeviceToHost));
        if (!validate(output, expected)) {
            return 1;
        }

        // Same-buffer, warm-cache device timing. Each sample batches launches
        // so sub-10 us kernels are not dominated by event resolution.
        const auto times = cuda_harness::benchmark(
            launch, 10, samples, launches_per_sample);
        const float median_us = cuda_harness::median(times);

        // Re-poison and validate after timing to catch stale-output tricks.
        CUDA_CHECK(cudaMemset(device_output.data(), 0xff, n * sizeof(float)));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            output.data(), device_output.data(), n * sizeof(float), cudaMemcpyDeviceToHost));
        if (!validate(output, expected)) {
            return 1;
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "median " << median_us << " us  (n=" << n
                  << ", " << samples << " samples x "
                  << launches_per_sample << " launches, warm cache)\n";
        std::cout << "{\"status\":\"ok\",\"n\":" << n
                  << ",\"seed\":" << seed
                  << ",\"cache\":\"warm_same_buffer\""
                  << ",\"median_us\":" << median_us
                  << ",\"samples_us\":[";
        for (size_t i = 0; i < times.size(); ++i) {
            std::cout << (i == 0 ? "" : ",") << times[i];
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

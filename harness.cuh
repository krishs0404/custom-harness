#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_harness {

inline void check(cudaError_t error, const char* expression) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(expression) + ": " + cudaGetErrorName(error) +
            " (" + cudaGetErrorString(error) + ")");
    }
}

#define CUDA_CHECK(expression) \
    ::cuda_harness::check((expression), #expression)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

class Event {
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    operator cudaEvent_t() const { return event_; }

private:
    cudaEvent_t event_{};
};

template <typename Launch>
std::vector<float> benchmark(
    Launch launch,
    int warmups = 10,
    int samples = 15,
    int launches_per_sample = 100) {
    for (int i = 0; i < warmups; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Event start;
    Event stop;
    std::vector<float> times_us;
    times_us.reserve(samples);

    for (int sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < launches_per_sample; ++i) {
            launch();
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        times_us.push_back(elapsed_ms * 1000.0f / launches_per_sample);
    }

    return times_us;
}

inline float median(std::vector<float> values) {
    if (values.empty()) {
        throw std::invalid_argument("median requires at least one sample");
    }
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0f
        : values[middle];
}

}  // namespace cuda_harness

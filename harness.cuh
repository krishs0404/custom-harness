#pragma once

// Verifier and measurement code shared by every kernel under kernels/.
// Kernel files compose the pieces below; they should never need to edit
// this header. Layout:
//
//   Layer 1: primitives   — error checking, RAII buffers/events, timing
//   Layer 2: verifier     — GPU gate, poisoning, validation, bandwidth peaks
//   Layer 3: run_and_report — the full poison/validate/time/re-validate flow
//
// Layer 3 lives here rather than in each kernel's main() so a kernel file
// cannot reorder or drop verification steps; Layers 1-2 stay public as the
// escape hatch for kernels with unusual shapes.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Injected by the Makefile so a kernel cannot mislabel its own log entries.
#ifndef HARNESS_KERNEL_NAME
#define HARNESS_KERNEL_NAME "unknown"
#endif
#ifndef HARNESS_GIT_SHA
#define HARNESS_GIT_SHA "unknown"
#endif
#ifndef HARNESS_GIT_DIRTY
#define HARNESS_GIT_DIRTY 0
#endif

namespace cuda_harness {

// ---------------------------------------------------------------------------
// Layer 1: primitives
// ---------------------------------------------------------------------------

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
        // Device-wide sync before the stop event: work a kernel enqueues on
        // side streams gets counted too (a documented eval exploit). Costs
        // one host wakeup per sample, amortized over launches_per_sample;
        // the bias direction is over-reporting time, never under.
        CUDA_CHECK(cudaDeviceSynchronize());
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

// ---------------------------------------------------------------------------
// Layer 2: verifier pieces
// ---------------------------------------------------------------------------

// Call first in main(), before any allocation. Exit 77 = "no GPU here",
// so CI and the GPU-less laptop can tell a skip from a failure.
inline void require_gpu_or_exit() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        device_count == 0) {
        std::cerr << "SKIP: no usable CUDA device\n";
        std::exit(77);
    }
    CUDA_CHECK(status);
}

// 0xff bytes make every float a NaN, so a kernel that skips writes (whole
// buffer, tail, or "only on the first call") is caught by validation.
template <typename T>
void poison(T* device_ptr, size_t count) {
    CUDA_CHECK(cudaMemset(device_ptr, 0xff, count * sizeof(T)));
}

template <typename T>
bool validate(
    const std::vector<T>& actual,
    const std::vector<T>& expected,
    T tolerance) {
    T max_error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const T error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(actual[i]) || error > tolerance) {
            std::cerr << "wrong value at " << i << ": got " << actual[i]
                      << ", expected " << expected[i] << '\n';
            return false;
        }
    }
    std::cerr << "correct (max error " << max_error << ")\n";
    return true;
}

// Datasheet-style ceiling: memory clock x 2 (DDR) x bus width. Nullopt when
// the driver does not report the attributes (they are deprecated on some
// stacks); the JSON field becomes null rather than blocking the run.
inline std::optional<double> theoretical_peak_gbps(int device = 0) {
    int clock_khz = 0;
    int bus_bits = 0;
    if (cudaDeviceGetAttribute(
            &clock_khz, cudaDevAttrMemoryClockRate, device) != cudaSuccess ||
        cudaDeviceGetAttribute(
            &bus_bits, cudaDevAttrGlobalMemoryBusWidth, device) != cudaSuccess) {
        cudaGetLastError();  // clear the sticky error
        return std::nullopt;
    }
    if (clock_khz <= 0 || bus_bits <= 0) {
        return std::nullopt;
    }
    return clock_khz * 1.0e3 * 2.0 * (bus_bits / 8.0) / 1.0e9;
}

// Achievable ceiling: a saturating device-to-device copy sweep. Max of five
// reps because we want the ceiling, not the typical rep. A perfectly
// coalesced streaming kernel can slightly beat this (README explains).
inline double measured_peak_gbps() {
    const size_t bytes = size_t{256} << 20;  // 256 MiB per buffer
    DeviceBuffer<unsigned char> source(bytes);
    DeviceBuffer<unsigned char> destination(bytes);

    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMemcpy(
            destination.data(), source.data(), bytes, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    Event start;
    Event stop;
    double best_gbps = 0.0;
    for (int rep = 0; rep < 5; ++rep) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(
            destination.data(), source.data(), bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        // A copy reads and writes every byte, hence 2x.
        const double gbps = 2.0 * bytes / (elapsed_ms * 1.0e-3) / 1.0e9;
        best_gbps = std::max(best_gbps, gbps);
    }
    return best_gbps;
}

inline std::string device_name(int device = 0) {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.name;
}

inline std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm parts{};
    gmtime_r(&now, &parts);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &parts);
    return buffer;
}

struct Result {
    size_t n = 0;
    uint32_t seed = 0;
    float median_us = 0.0f;
    std::vector<float> samples_us;
    double bytes_moved = 0.0;
    double gbps = 0.0;
    std::optional<double> theoretical_peak_gbps;
    double measured_peak_gbps = 0.0;
    std::string clocks;
    std::string device;
};

// Single owner of the JSON schema. One line on stdout and nothing else, so
// `make run` can append stdout verbatim to results.jsonl.
inline void print_result_json(const Result& result) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);
    out << "{\"status\":\"ok\""
        << ",\"kernel\":\"" << HARNESS_KERNEL_NAME << "\""
        << ",\"n\":" << result.n
        << ",\"seed\":" << result.seed
        << ",\"cache\":\"warm_same_buffer\""
        << ",\"median_us\":" << result.median_us
        << ",\"samples_us\":[";
    for (size_t i = 0; i < result.samples_us.size(); ++i) {
        out << (i == 0 ? "" : ",") << result.samples_us[i];
    }
    out << "]";
    out.precision(0);
    out << ",\"bytes_moved\":" << result.bytes_moved;
    out.precision(1);
    out << ",\"gbps\":" << result.gbps;
    if (result.theoretical_peak_gbps) {
        out << ",\"theoretical_peak_gbps\":" << *result.theoretical_peak_gbps
            << ",\"pct_theoretical_peak\":"
            << 100.0 * result.gbps / *result.theoretical_peak_gbps;
    } else {
        out << ",\"theoretical_peak_gbps\":null,\"pct_theoretical_peak\":null";
    }
    out << ",\"measured_peak_gbps\":" << result.measured_peak_gbps
        << ",\"pct_measured_peak\":"
        << 100.0 * result.gbps / result.measured_peak_gbps
        << ",\"clocks\":\"" << result.clocks << "\""
        << ",\"git_sha\":\"" << HARNESS_GIT_SHA << "\""
        << ",\"dirty\":" << (HARNESS_GIT_DIRTY ? "true" : "false")
        << ",\"device\":\"" << result.device << "\""
        << ",\"timestamp\":\"" << utc_timestamp() << "\"}";
    std::cout << out.str() << std::endl;
}

// ---------------------------------------------------------------------------
// Layer 3: the composed flow
// ---------------------------------------------------------------------------

struct RunConfig {
    size_t n = 0;
    uint32_t seed = 0;
    // The kernel's declared traffic (e.g. 3 * n * sizeof(float) for c=a+b).
    // Part of the problem statement, not the optimization surface.
    double bytes_moved = 0.0;
    float tolerance = 1.0e-6f;
    int warmups = 10;
    int samples = 15;
    // Batched so sub-10 us kernels are not dominated by event resolution.
    int launches_per_sample = 100;
};

// The whole verifier sequence:
//   poison -> launch -> validate          (correctness gate)
//   benchmark                             (warm, batched, device-synced)
//   re-poison -> launch -> re-validate    (catches stale-output tricks)
//   speed-of-light math -> stderr summary + one JSON line on stdout
//
// HARNESS_MODE=check stops after the correctness gate plus a few plain
// launches: that is what `make profile` and `make sanitize` run, so profiler
// numbers are structurally never mixed with timing numbers.
//
// Returns the process exit code: 0 ok, 1 wrong.
template <typename T, typename Launch>
int run_and_report(
    const RunConfig& config,
    Launch launch,
    T* device_output,
    const std::vector<T>& expected) {
    const char* mode = std::getenv("HARNESS_MODE");
    const bool check_only = mode != nullptr && std::string(mode) == "check";
    const char* clocks = std::getenv("HARNESS_CLOCKS");

    const size_t count = expected.size();
    std::vector<T> actual(count);
    const auto launch_and_validate = [&] {
        poison(device_output, count);
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            actual.data(), device_output, count * sizeof(T),
            cudaMemcpyDeviceToHost));
        return validate(actual, expected, static_cast<T>(config.tolerance));
    };

    if (!launch_and_validate()) {
        return 1;
    }

    if (check_only) {
        // A few plain launches for ncu / compute-sanitizer to attach to
        // (`--launch-skip 1` in the Makefile skips the correctness launch).
        for (int i = 0; i < 3; ++i) {
            launch();
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cerr << "check ok\n";
        return 0;
    }

    // Ceilings before warmup, so the copy sweep's cache pollution cannot
    // touch the timed region.
    Result result;
    result.measured_peak_gbps = measured_peak_gbps();
    result.theoretical_peak_gbps = theoretical_peak_gbps();

    result.samples_us = benchmark(
        launch, config.warmups, config.samples, config.launches_per_sample);
    result.median_us = median(result.samples_us);

    if (!launch_and_validate()) {
        return 1;
    }

    result.n = config.n;
    result.seed = config.seed;
    result.bytes_moved = config.bytes_moved;
    result.gbps = config.bytes_moved / (result.median_us * 1.0e-6) / 1.0e9;
    result.clocks = clocks != nullptr ? clocks : "unknown";
    result.device = device_name();

    std::cerr.setf(std::ios::fixed);
    std::cerr.precision(3);
    std::cerr << "median " << result.median_us << " us  ";
    std::cerr.precision(1);
    std::cerr << result.gbps << " GB/s  "
              << 100.0 * result.gbps / result.measured_peak_gbps
              << "% of measured peak (" << result.measured_peak_gbps
              << " GB/s)\n";
    print_result_json(result);
    return 0;
}

}  // namespace cuda_harness

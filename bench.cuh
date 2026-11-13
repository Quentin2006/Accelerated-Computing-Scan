// bench.cuh - drop-in timing and reporting for the scan kernels.
//
// GPU usage (in a .cu file compiled with nvcc):
//
//   #include "../bench.cuh"
//   // ./scan.out 16777216  or  ./scan.out 2^24
//   size_t N = bench::arg_n(argc, argv, 1 << 24);
//   ...
//   // multi-kernel launches are fine too
//   bench::Stats s = bench::time_gpu([&] {
//     scan<<<gd, threads>>>(dst_d, src_d, N, ADD);
//   });
//   // the last two arguments (the kernel's own estimate) are optional
//   bench::report("kogge-stone", "add", N, s,
//                 bench::scan_min_bytes(N), bench::scan_min_flops(N),
//                 est_bytes, est_flops);
//
// CPU usage (compiled as C++ with g++/clang++ or nvcc - not as C):
//
//   bench::Stats s = bench::time_cpu([&] { scan(dst, src, N, ADD); });
//   bench::report("seq", "add", N, s,
//                 bench::scan_min_bytes(N), bench::scan_min_flops(N));
//
// Output:
//   - One human-readable line per report() on stdout.
//   - If the environment variable BENCH_CSV is set, one CSV row is appended to
//     that file (header written when the file is new/empty), e.g.
//       for k in $(seq 16 28); do BENCH_CSV=results.csv ./scan.out 2^$k; done
//   - bytes/flops are the work credited to every scan (the minimum, so GB/s
//     compares kernels fairly). est_bytes/est_flops are the kernel's own
//     estimate of what it actually moves and computes, worked out by hand
//     from its loads, stores and binop calls (not measured); they default to
//     bytes/flops when not given.
//
// Notes:
//   - GPU times come from cudaEvents recorded around each repetition, so they
//     cover only the GPU work issued inside the lambda (no host overhead, no
//     memcpy unless you put it in the lambda).
//   - Launches inside the lambda must go to the same stream passed to
//     time_gpu (default stream 0 if you don't pass one).
//   - Every launch is followed by cudaGetLastError(), and the stream is
//     synchronized before results are read, so a failed launch aborts instead
//     of producing a suspiciously fast time.
//   - With flush_l2 on, the L2 is overwritten (outside the timed region)
//     before each repetition so repeated runs don't read a warm cache.
//   - Throughput figures use the median time.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace bench {

// ---------------------------------------------------------------------------
// Configuration and statistics
// ---------------------------------------------------------------------------

struct Config {
  int warmup = 3;       // untimed runs first (module load, clocks ramping up)
  int reps = 20;        // timed runs
  bool flush_l2 = true; // GPU only: overwrite L2 before each timed run
};

struct Stats {
  double min_ms = 0, median_ms = 0, mean_ms = 0, stddev_ms = 0;
  int reps = 0;
  bool on_gpu = false;
};

inline Stats summarize(std::vector<double> ms) {
  Stats s;
  s.reps = (int)ms.size();
  if (ms.empty())
    return s;
  std::sort(ms.begin(), ms.end());
  size_t m = ms.size() / 2;
  s.min_ms = ms.front();
  s.median_ms = (ms.size() % 2) ? ms[m] : 0.5 * (ms[m - 1] + ms[m]);
  double sum = 0;
  for (double x : ms)
    sum += x;
  s.mean_ms = sum / ms.size();
  double var = 0;
  for (double x : ms)
    var += (x - s.mean_ms) * (x - s.mean_ms);
  s.stddev_ms = ms.size() > 1 ? std::sqrt(var / (ms.size() - 1)) : 0.0;
  return s;
}

// ---------------------------------------------------------------------------
// Argument parsing: accepts a plain integer ("16777216") or a power of two
// ("2^24")
// ---------------------------------------------------------------------------

inline size_t arg_n(int argc, char **argv, size_t fallback, int index = 1) {
  if (argc <= index)
    return fallback;
  const char *s = argv[index];
  char *end = nullptr;
  if (std::strncmp(s, "2^", 2) == 0) {
    unsigned long k = std::strtoul(s + 2, &end, 10);
    if (end != s + 2 && *end == '\0' && k < 63)
      return (size_t)1 << k;
  } else {
    unsigned long long v = std::strtoull(s, &end, 10);
    if (end != s && *end == '\0' && v > 0)
      return (size_t)v;
  }
  std::fprintf(stderr, "bad element count '%s' (use e.g. 16777216 or 2^24)\n",
               s);
  std::exit(EXIT_FAILURE);
}

// ---------------------------------------------------------------------------
// Analytic work for a scan of n floats.
//
// These are the *minimum* a work-efficient inclusive scan must do: read n
// floats, write n floats, perform n-1 binary ops. Using the same numbers for
// every kernel makes GB/s and GFLOP/s directly comparable ("useful work per
// second"). For the measured roofline, use ncu's dram__bytes / FP32 op
// counters instead - the gap between the two is the kernel's overhead.
// ---------------------------------------------------------------------------

inline double scan_min_bytes(size_t n) {
  return 2.0 * (double)n * sizeof(float);
}
inline double scan_min_flops(size_t n) { return n ? (double)(n - 1) : 0.0; }

// ---------------------------------------------------------------------------
// Device description (peaks are 0 when unknown, e.g. for CPU runs)
// ---------------------------------------------------------------------------

struct DeviceInfo {
  std::string name = "cpu";
  int cc_major = 0, cc_minor = 0, sm_count = 0;
  size_t l2_bytes = 0;
  double peak_gbs = 0; // theoretical DRAM bandwidth, GB/s (1e9 bytes)
  double peak_gflops =
      0; // theoretical FP32 throughput (FMA = 2 FLOPs), GFLOP/s
};

inline const DeviceInfo &cpu_info() {
  static DeviceInfo info;
  return info;
}

// ---------------------------------------------------------------------------
// CPU timing
// ---------------------------------------------------------------------------

template <class F> Stats time_cpu(F &&fn, Config cfg = Config()) {
  using clk = std::chrono::steady_clock;
  for (int i = 0; i < cfg.warmup; ++i)
    fn();
  std::vector<double> ms;
  ms.reserve(cfg.reps > 0 ? cfg.reps : 0);
  for (int i = 0; i < cfg.reps; ++i) {
    auto t0 = clk::now();
    fn();
    auto t1 = clk::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  Stats s = summarize(ms);
  s.on_gpu = false;
  return s;
}

// ---------------------------------------------------------------------------
// GPU timing (only when compiled by nvcc)
// ---------------------------------------------------------------------------

#ifdef __CUDACC__

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n  in: %s\n", __FILE__,    \
                   __LINE__, cudaGetErrorString(err_), #call);                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
#endif

// FP32 lanes per SM by architecture; 0 = unknown (peak FLOP/s left blank).
inline int fp32_cores_per_sm(int major, int minor) {
  switch (major) {
  case 5:
    return 128; // Maxwell
  case 6:
    return minor == 0 ? 64 : 128; // Pascal (P100 = 64)
  case 7:
    return 64; // Volta, Turing
  case 8:
    return minor == 0 ? 64 : 128; // Ampere (A100 = 64, GA10x = 128), Ada
  case 9:
    return 128; // Hopper
  case 10:
  case 11:
  case 12:
    return 128; // Blackwell
  default:
    return 0;
  }
}

inline const DeviceInfo &gpu_info() {
  static DeviceInfo info;
  static bool ready = false;
  if (ready)
    return info;

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  info.name = prop.name;

  // Attributes rather than cudaDeviceProp fields: the clock fields were
  // deprecated in CUDA 12 and removed in CUDA 13.
  int sms = 0, l2 = 0, core_khz = 0, mem_khz = 0, bus_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&info.cc_major,
                                    cudaDevAttrComputeCapabilityMajor, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&info.cc_minor,
                                    cudaDevAttrComputeCapabilityMinor, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&core_khz, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(
      cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));

  info.sm_count = sms;
  info.l2_bytes = (size_t)l2;
  // DDR: two transfers per reported memory clock.
  info.peak_gbs = 2.0 * mem_khz * 1e3 * (bus_bits / 8.0) / 1e9;
  // One FMA (2 FLOPs) per FP32 lane per cycle at the max (boost) clock.
  info.peak_gflops = 2.0 * sms *
                     fp32_cores_per_sm(info.cc_major, info.cc_minor) *
                     core_khz * 1e3 / 1e9;
  ready = true;
  return info;
}

template <class F>
Stats time_gpu(F &&launch, Config cfg = Config(), cudaStream_t stream = 0) {
  // Scratch buffer larger than L2; memset over it evicts the kernel's data.
  void *flush = nullptr;
  size_t flush_bytes = 0;
  if (cfg.flush_l2) {
    flush_bytes = std::max<size_t>(2 * gpu_info().l2_bytes, (size_t)8 << 20);
    CUDA_CHECK(cudaMalloc(&flush, flush_bytes));
  }

  for (int i = 0; i < cfg.warmup; ++i) {
    launch();
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int reps = cfg.reps > 0 ? cfg.reps : 0;
  std::vector<cudaEvent_t> start(reps), stop(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventCreate(&start[i]));
    CUDA_CHECK(cudaEventCreate(&stop[i]));
  }

  for (int i = 0; i < reps; ++i) {
    if (flush)
      CUDA_CHECK(cudaMemsetAsync(flush, i & 0xff, flush_bytes, stream));
    CUDA_CHECK(cudaEventRecord(start[i], stream));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop[i], stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream)); // surfaces async kernel faults

  std::vector<double> ms(reps);
  for (int i = 0; i < reps; ++i) {
    float t = 0;
    CUDA_CHECK(cudaEventElapsedTime(&t, start[i], stop[i]));
    ms[i] = t;
    CUDA_CHECK(cudaEventDestroy(start[i]));
    CUDA_CHECK(cudaEventDestroy(stop[i]));
  }
  if (flush)
    CUDA_CHECK(cudaFree(flush));

  Stats s = summarize(ms);
  s.on_gpu = true;
  return s;
}

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

inline const DeviceInfo &info_for(const Stats &s) {
#ifdef __CUDACC__
  if (s.on_gpu)
    return gpu_info();
#endif
  (void)s;
  return cpu_info();
}

inline void print_device(FILE *out = stdout) {
#ifdef __CUDACC__
  const DeviceInfo &d = gpu_info();
  std::fprintf(out,
               "[bench] %s  sm_%d%d  %d SMs  L2 %.1f MB  "
               "peak %.1f GB/s  %.1f GFLOP/s (FP32)  ridge %.1f FLOP/B\n",
               d.name.c_str(), d.cc_major, d.cc_minor, d.sm_count,
               d.l2_bytes / 1048576.0, d.peak_gbs, d.peak_gflops,
               d.peak_gbs > 0 ? d.peak_gflops / d.peak_gbs : 0.0);
#else
  std::fprintf(out, "[bench] cpu (no peak figures)\n");
#endif
}

// op: the scan operator's name ("add", "mul", ...), recorded so timings for
//   different operators can be told apart.
// bytes / flops: the work you want credited for one call of the timed lambda.
// est_bytes / est_flops: the kernel's own estimate of its traffic and op count
// (negative = same as bytes / flops).
inline void report(const char *kernel, const char *op, size_t n,
                   const Stats &s, double bytes, double flops,
                   double est_bytes = -1, double est_flops = -1) {
  if (est_bytes < 0)
    est_bytes = bytes;
  if (est_flops < 0)
    est_flops = flops;
  const DeviceInfo &d = info_for(s);
  double sec = s.median_ms / 1e3;
  double gbs = sec > 0 ? bytes / sec / 1e9 : 0;
  double gflops = sec > 0 ? flops / sec / 1e9 : 0;
  double ai = bytes > 0 ? flops / bytes : 0;
  double pct_bw = d.peak_gbs > 0 ? 100.0 * gbs / d.peak_gbs : 0;
  double pct_fl = d.peak_gflops > 0 ? 100.0 * gflops / d.peak_gflops : 0;
  double est_gbs = sec > 0 ? est_bytes / sec / 1e9 : 0;
  double est_gflops = sec > 0 ? est_flops / sec / 1e9 : 0;
  double est_ai = est_bytes > 0 ? est_flops / est_bytes : 0;

  FILE *out = stdout;
  std::fprintf(
      out,
      "[bench] %-28s %-3s N=%-11zu median %9.4f ms (min %.4f, sd %.4f, %d reps)"
      "  %8.2f GB/s",
      kernel, op, n, s.median_ms, s.min_ms, s.stddev_ms, s.reps, gbs);
  if (d.peak_gbs > 0)
    std::fprintf(out, " (%5.1f%% peak)", pct_bw);
  std::fprintf(out, "  %8.2f GFLOP/s  AI %.3f FLOP/B", gflops, ai);
  std::fprintf(out, "  | est %8.2f GB/s  %8.2f GFLOP/s  AI %.3f FLOP/B\n",
               est_gbs, est_gflops, est_ai);

  const char *path = std::getenv("BENCH_CSV");
  if (!path || !*path)
    return;
  FILE *f = std::fopen(path, "a");
  if (!f) {
    std::fprintf(stderr, "[bench] cannot open BENCH_CSV=%s\n", path);
    return;
  }
  std::fseek(f, 0, SEEK_END);
  if (std::ftell(f) == 0)
    std::fprintf(f, "kernel,op,device,n,median_ms,min_ms,mean_ms,stddev_ms,reps,"
                    "bytes,flops,gbs,gflops,ai,pct_peak_bw,pct_peak_flops,"
                    "peak_gbs,peak_gflops,est_bytes,est_flops,est_ai,est_gbs,"
                    "est_gflops\n");
  std::fprintf(f,
               "%s,%s,\"%s\",%zu,%.6f,%.6f,%.6f,%.6f,%d,%.0f,%.0f,%.4f,%.4f,%.6f,"
               "%.3f,%.3f,%.2f,%.2f,%.0f,%.0f,%.6f,%.4f,%.4f\n",
               kernel, op, d.name.c_str(), n, s.median_ms, s.min_ms, s.mean_ms,
               s.stddev_ms, s.reps, bytes, flops, gbs, gflops, ai, pct_bw,
               pct_fl, d.peak_gbs, d.peak_gflops, est_bytes, est_flops, est_ai,
               est_gbs, est_gflops);
  std::fclose(f);
}

} // namespace bench

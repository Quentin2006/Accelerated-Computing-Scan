#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../bench.cuh" // BENCH: timing/reporting header (bench::, CUDA_CHECK)
#include "../common.cuh" // BINOP, binop, identity, CLI flags, output

__global__ void scan(float *dst, const float *src, int N, enum BINOP op) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
    float res = src[0];
    for (int j = 1; j <= i; ++j)
      res = binop(src[j], res, op);
    dst[i] = res;
  }
}

int main(int argc, char **argv) { // BENCH: was main(void); argv carries N
  float *src, *dst;
  // BENCH: N from the command line (./scan.out 2^16) instead of hardcoded 10.
  // Default is deliberately small: naive does N(N-1)/2 ops, so 2^24 would
  // take far too long across 23 runs. Try up to about 2^20.
  // --op and --input pick the operator and the repeated input values.
  Args args = parse_args(argc, argv, 1 << 16);
  size_t numElements = args.n;
  // BENCH: size_t, an int byte count overflows past 2^29 elements
  size_t numBytes = sizeof(float) * numElements;

  src = (float *)malloc(numBytes);
  dst = (float *)malloc(numBytes);

  fill_input(src, numElements, args.input);

  bench::print_device(); // BENCH: one-line GPU summary (peaks, ridge point)

  // calling cuda ker
  {
    float *src_d, *dst_d;
    int threads = 256;
    // BENCH: integer ceil-divide; the old float ceil() is inexact above 2^24
    int gd = (int)((numElements + threads - 1) / threads);

    // BENCH: CUDA_CHECK (from bench.cuh) aborts with file/line on any error
    CUDA_CHECK(cudaMalloc(&src_d, numBytes));
    CUDA_CHECK(cudaMalloc(&dst_d, numBytes));

    CUDA_CHECK(cudaMemcpy(src_d, src, numBytes, cudaMemcpyHostToDevice));

    // BENCH: the launch now runs inside time_gpu (3 warmup + 20 timed runs,
    // L2 cleared before each). Only the kernel is timed, not the memcpys.
    bench::Stats s = bench::time_gpu([&] {
      scan<<<gd, threads>>>(dst_d, src_d, numElements, args.op);
    });
    // BENCH: credited work is the scan minimum (8N bytes, N-1 ops).
    // Estimate: thread i does i ops, so N(N-1)/2 ops in total. Thread i also
    // reads src[0..i], but neighbouring threads read the same values and those
    // repeats hit L1/L2, so DRAM traffic stays near 8N bytes (read src once,
    // write dst once).
    double n = (double)numElements;
    bench::report("naive", numElements, s,
                  bench::scan_min_bytes(numElements),
                  bench::scan_min_flops(numElements),
                  8.0 * n, n * (n - 1.0) / 2.0);

    CUDA_CHECK(cudaMemcpy(dst, dst_d, numBytes, cudaMemcpyDeviceToHost));
    cudaFree(src_d); // BENCH: added frees
    cudaFree(dst_d);
  }

  write_output(dst, numElements); // all N results, one line on stdout

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}

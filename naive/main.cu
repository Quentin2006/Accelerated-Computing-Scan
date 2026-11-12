#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../bench.cuh" // BENCH: timing/reporting header (bench::, CUDA_CHECK)

// NOTE: we can use any bianary operator
enum BINOP { ADD, SUB, MUL, DIV };

__device__ float binop(float a, float b, enum BINOP op) {
  switch (op) {
  case ADD:
    return a + b;
  case SUB:
    return a - b;
  case MUL:
    return a * b;
  case DIV:
    return a / b;
  }
  return a + b;
}

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
  // BENCH: N from the command line (./scan 2^16) instead of hardcoded 10.
  // Default is deliberately small: naive does N(N-1)/2 ops, so 2^24 would
  // take far too long across 23 runs. Try up to about 2^20.
  size_t numElements = bench::arg_n(argc, argv, 1 << 16);
  // BENCH: size_t, an int byte count overflows past 2^29 elements
  size_t numBytes = sizeof(float) * numElements;

  src = (float *)malloc(numBytes);
  dst = (float *)malloc(numBytes);

  for (size_t i = 0; i < numElements; ++i) // BENCH: size_t to match N
    src[i] = 1.0f;

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
      scan<<<gd, threads>>>(dst_d, src_d, numElements, ADD);
    });
    // BENCH: minimum scan work (8N bytes, N-1 ops). Actual ops are N(N-1)/2,
    // and most of the repeated src reads hit cache rather than DRAM.
    bench::report("naive", numElements, s,
                  bench::scan_min_bytes(numElements),
                  bench::scan_min_flops(numElements));

    CUDA_CHECK(cudaMemcpy(dst, dst_d, numBytes, cudaMemcpyDeviceToHost));
    cudaFree(src_d); // BENCH: added frees
    cudaFree(dst_d);
  }

  // BENCH: spot-check the first few values instead of printing all N
  for (size_t i = 0; i < 4 && i < numElements; ++i)
    printf("dst[%zu] == %f\n", i, dst[i]);

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}

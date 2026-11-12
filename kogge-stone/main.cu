#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../bench.cuh" // BENCH: timing/reporting header (bench::, CUDA_CHECK)

#define SEG_SIZE 256

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

// copied from the textbook, currently doesnt support many blocks
__global__ void scan(float *dst, const float *src, unsigned int N,
                     enum BINOP op) {
  __shared__ float buffer_s[SEG_SIZE];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    buffer_s[threadIdx.x] = src[i];
  } else {
    buffer_s[threadIdx.x] = 0.0f;
  }
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    float temp;
    if (threadIdx.x >= stride)
      temp = binop(buffer_s[threadIdx.x], buffer_s[threadIdx.x - stride], op);
    __syncthreads();
    if (threadIdx.x >= stride)
      buffer_s[threadIdx.x] = temp;
  }
  if (i < N) {
    dst[i] = buffer_s[threadIdx.x];
  }
}

int main(int argc, char **argv) { // BENCH: was main(void); argv carries N
  float *src, *dst;
  // BENCH: N from the command line (./scan 2^24) instead of hardcoded 10
  size_t numElements = bench::arg_n(argc, argv, 1 << 24);
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
    int threads = SEG_SIZE;
    // BENCH: integer ceil-divide; the old float ceil() is inexact above 2^24
    int gd = (int)((numElements + threads - 1) / threads);

    // BENCH: CUDA_CHECK (from bench.cuh) aborts with file/line on any error
    CUDA_CHECK(cudaMalloc(&src_d, numBytes));
    CUDA_CHECK(cudaMalloc(&dst_d, numBytes));

    CUDA_CHECK(cudaMemcpy(src_d, src, numBytes, cudaMemcpyHostToDevice));

    // BENCH: the launch now runs inside time_gpu (3 warmup + 20 timed runs,
    // L2 cleared before each). Only the kernel is timed, not the memcpys.
    // NOTE: each block scans only its own SEG_SIZE segment, so for
    // N > SEG_SIZE this times a segmented scan (dst restarts every 256).
    bench::Stats s = bench::time_gpu([&] {
      scan<<<gd, threads>>>(dst_d, src_d, numElements, ADD);
    });
    // BENCH: minimum scan work (8N bytes, N-1 ops). Actual ops are ~7N:
    // log2(256) = 8 strides, minus the threads below each stride that idle.
    bench::report("kogge-stone", numElements, s,
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

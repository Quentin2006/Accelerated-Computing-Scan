#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../bench.cuh" // BENCH: timing/reporting header (bench::, CUDA_CHECK)
#include "../common.cuh" // BINOP, binop, identity, CLI flags, output

#define SEG_SIZE 256

// copied from the textbook, currently doesnt support many blocks
__global__ void scan(float *dst, const float *src, unsigned int N,
                     enum BINOP op) {
  __shared__ float buffer_s[SEG_SIZE];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    buffer_s[threadIdx.x] = src[i];
  } else {
    buffer_s[threadIdx.x] = identity(op);
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
  // BENCH: N from the command line (./scan.out 2^24) instead of hardcoded 10.
  // --op and --input pick the operator and the repeated input values.
  Args args = parse_args(argc, argv, 1 << 24);
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
      scan<<<gd, threads>>>(dst_d, src_d, numElements, args.op);
    });
    // BENCH: credited work is the scan minimum (8N bytes, N-1 ops).
    // Estimate per 256-element block: stride s has 256-s active threads, so
    // sum over s=1,2,...,128 of (256-s) = 256*8 - 255 = 1793 ops (~7N total).
    // Traffic is 8N bytes: src read once, dst written once.
    const double blockOps = SEG_SIZE * std::log2((double)SEG_SIZE) - (SEG_SIZE - 1);
    bench::report("kogge-stone", op_name(args.op), numElements, s,
                  bench::scan_min_bytes(numElements),
                  bench::scan_min_flops(numElements),
                  8.0 * numElements, blockOps * gd);

    CUDA_CHECK(cudaMemcpy(dst, dst_d, numBytes, cudaMemcpyDeviceToHost));
    cudaFree(src_d); // BENCH: added frees
    cudaFree(dst_d);
  }

  write_output(dst, numElements); // all N results, one line on stdout

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}

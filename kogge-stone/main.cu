#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../bench.cuh" // BENCH: timing/reporting header (bench::, CUDA_CHECK)
#include "../common.cuh" // BINOP, binop, identity, CLI flags, output

#define SEG_SIZE 256

// copied from the textbook (Fig 11.3): Kogge-Stone scan of one value per
// thread through shared memory. Returns this thread's inclusive result.
// buffer_s must hold blockDim.x floats.
__device__ float koggeStone(float val, float *buffer_s, enum BINOP op) {
  buffer_s[threadIdx.x] = val;
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    float temp;
    if (threadIdx.x >= stride)
      temp = binop(buffer_s[threadIdx.x], buffer_s[threadIdx.x - stride], op);
    __syncthreads();
    if (threadIdx.x >= stride)
      buffer_s[threadIdx.x] = temp;
  }
  return buffer_s[threadIdx.x];
}

// step 1: scan each block, and keep each block's total for step 2
__global__ void scanBlock(float *dst, float *blockTotals, const float *src,
                          unsigned int N, enum BINOP op) {
  __shared__ float buffer_s[SEG_SIZE];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = koggeStone((i < N) ? src[i] : identity(op), buffer_s, op);
  if (i < N) {
    dst[i] = val;
  }
  // last thread holds the whole block's total
  if (threadIdx.x == blockDim.x - 1) {
    blockTotals[blockIdx.x] = val;
  }
}

// step 2: scan the block totals. one block walks them a chunk at a time,
// carrying each chunk's total into the next, so any number of blocks works
__global__ void scanBlockTotals(float *blockTotals, unsigned int numBlocks,
                                enum BINOP op) {
  __shared__ float buffer_s[SEG_SIZE];
  __shared__ float carry_s;
  if (threadIdx.x == 0) {
    carry_s = identity(op);
  }
  __syncthreads();

  for (unsigned int base = 0; base < numBlocks; base += blockDim.x) {
    unsigned int t = base + threadIdx.x;
    float val = (t < numBlocks) ? blockTotals[t] : identity(op);
    val = koggeStone(val, buffer_s, op);
    if (t < numBlocks) {
      blockTotals[t] = binop(val, carry_s, op);
    }
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) {
      carry_s = binop(carry_s, val, op);
    }
    __syncthreads();
  }
}

// step 3: combine each element with the total of all the blocks before it
__global__ void addBlockOffsets(float *dst, const float *blockTotals,
                                unsigned int N, enum BINOP op) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0 && i < N) {
    dst[i] = binop(dst[i], blockTotals[blockIdx.x - 1], op);
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
    float *src_d, *dst_d, *blockTotals_d;
    int threads = SEG_SIZE;
    // BENCH: integer ceil-divide; the old float ceil() is inexact above 2^24
    int gd = (int)((numElements + threads - 1) / threads);

    // BENCH: CUDA_CHECK (from bench.cuh) aborts with file/line on any error
    CUDA_CHECK(cudaMalloc(&src_d, numBytes));
    CUDA_CHECK(cudaMalloc(&dst_d, numBytes));
    CUDA_CHECK(cudaMalloc(&blockTotals_d, sizeof(float) * gd));

    CUDA_CHECK(cudaMemcpy(src_d, src, numBytes, cudaMemcpyHostToDevice));

    // BENCH: all three kernels run inside one timed lambda (3 warmup + 20
    // timed runs, L2 cleared before each). Only the kernels are timed, not the
    // memcpys. Each run rewrites dst_d and blockTotals_d from scratch.
    bench::Stats s = bench::time_gpu([&] {
      scanBlock<<<gd, threads>>>(dst_d, blockTotals_d, src_d, numElements,
                                 args.op);
      scanBlockTotals<<<1, threads>>>(blockTotals_d, gd, args.op);
      addBlockOffsets<<<gd, threads>>>(dst_d, blockTotals_d, numElements,
                                       args.op);
    });
    // BENCH: credited work is the scan minimum (8N bytes, N-1 ops).
    // Estimate, with S = 256 elements per block and G = gd blocks:
    //   bytes: step 1 reads src and writes dst (8N) plus G totals (4G),
    //          step 2 reads and writes the totals (8G),
    //          step 3 reads and writes dst again (8N) plus one total per
    //          block (4G)                           -> 16N + 16G
    //   ops:   one Kogge-Stone pass over 256 values: stride s has 256-s
    //          active threads, so sum over s=1..128 of (256-s) = 1793 ops.
    //          Step 1 does G passes, step 2 does ceil(G/S) passes plus 257
    //          carry ops per chunk, step 3 does N - S ops   -> ~8N
    const double G = gd, n = (double)numElements;
    const double passOps = SEG_SIZE * std::log2((double)SEG_SIZE) - (SEG_SIZE - 1);
    const double chunks = (gd + SEG_SIZE - 1) / SEG_SIZE;
    double estBytes = 16.0 * n + 16.0 * G;
    double estOps = passOps * G + chunks * (passOps + SEG_SIZE + 1) +
                    (n > SEG_SIZE ? n - SEG_SIZE : 0.0);
    bench::report("kogge-stone", op_name(args.op), numElements, s,
                  bench::scan_min_bytes(numElements),
                  bench::scan_min_flops(numElements), estBytes, estOps);

    CUDA_CHECK(cudaMemcpy(dst, dst_d, numBytes, cudaMemcpyDeviceToHost));
    cudaFree(src_d); // BENCH: added frees
    cudaFree(dst_d);
    cudaFree(blockTotals_d);
  }

  write_output(dst, numElements); // all N results, one line on stdout

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}

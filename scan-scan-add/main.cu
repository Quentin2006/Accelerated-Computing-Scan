#include <cuda.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define SEG_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (SEG_SIZE / WARP_SIZE)

__device__ inline unsigned int laneIdx(void) { return threadIdx.x % WARP_SIZE; }
__device__ inline unsigned int warpIdx(void) { return threadIdx.x / WARP_SIZE; }

// FIGURE 11.8
// warp level scan. shuffles read other lanes registers so no shared mem or
// __syncthreads needed in here
__device__ inline float warpScan(float val) {
  for (unsigned int stride = 1; stride < WARP_SIZE; stride *= 2) {
    // shfl_up(mask, var, delta): mask is which lanes take part (all 32 here),
    // var is the value we pass in, delta is how many lanes to look left.
    // lanes below delta just get their own value back, so we skip them below
    float leftVal = __shfl_up_sync(0xffffffff, val, stride);
    if (laneIdx() >= stride) {
      val += leftVal;
    }
  }
  return val;
}

// block wide scan: warp scan, then scan the warp sums in warp 0
__device__ inline float blockScan(float val) {
  // warp level scan
  val = warpScan(val);
  // a shuffle cant leave its warp, so the warp sums go through shared mem
  __shared__ float warpsums_s[NUM_WARPS];
  const unsigned int numWarps = blockDim.x / WARP_SIZE;
  if (laneIdx() == WARP_SIZE - 1) {
    warpsums_s[warpIdx()] = val;
  }
  __syncthreads();
  // scan the warp sums
  if (warpIdx() == 0) {
    float warpSum = (threadIdx.x < numWarps) ? warpsums_s[threadIdx.x] : 0.0f;
    warpSum = warpScan(warpSum);
    if (threadIdx.x < numWarps) {
      warpsums_s[threadIdx.x] = warpSum;
    }
  }
  __syncthreads();
  // add the previous warps sums
  if (warpIdx() > 0) {
    val += warpsums_s[warpIdx() - 1];
  }
  return val;
}

// scan scan add step 1: scan each block and keep the block sums
// NOTE: blockDim has to be a multiple of WARP_SIZE and <= SEG_SIZE or
// warpsums_s overflows
__global__ void scanBlock(float *dst, float *blockTotals, const float *src,
                          unsigned int N) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = (i < N) ? src[i] : 0.0f;
  val = blockScan(val);
  if (i < N) {
    dst[i] = val;
  }
  // last thread holds the whole block sum
  if (threadIdx.x == blockDim.x - 1) {
    blockTotals[blockIdx.x] = val;
  }
}

// scan scan add step 2: scan the block sums. one block walks them a chunk at a
// time so the number of blocks doesnt matter
__global__ void scanBlockTotals(float *blockTotals, unsigned int numBlocks) {
  __shared__ float carry_s;
  if (threadIdx.x == 0) {
    carry_s = 0.0f;
  }
  __syncthreads();

  for (unsigned int base = 0; base < numBlocks; base += blockDim.x) {
    unsigned int tid = base + threadIdx.x;
    float val = (tid < numBlocks) ? blockTotals[tid] : 0.0f;
    val = blockScan(val);
    if (tid < numBlocks) {
      blockTotals[tid] = val + carry_s;
    }
    __syncthreads();
    // carry the chunk sum into the next chunk
    if (threadIdx.x == blockDim.x - 1) {
      carry_s += val;
    }
    __syncthreads();
  }
}

// scan scan add step 3: add the block sums of all the blocks before this one
__global__ void addBlockOffsets(float *dst, const float *blockTotals,
                                unsigned int N) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0 && i < N) {
    dst[i] += blockTotals[blockIdx.x - 1];
  }
}

int main(void) {
  float *src, *dst;
  int numElements = 10;

  // numBytes is an int so a bad count would wrap around
  if (numElements <= 0 || numElements > INT_MAX / (int)sizeof(float)) {
    fprintf(stderr, "bad element count %d: must be between 1 and %d\n",
            numElements, INT_MAX / (int)sizeof(float));
    return EXIT_FAILURE;
  }

  int numBytes = sizeof(float) * numElements;

  src = (float *)malloc(numBytes);
  dst = (float *)malloc(numBytes);

  for (int i = 0; i < numElements; ++i)
    src[i] = 1.0f;

  // calling cuda kernels
  {
    float *src_d, *dst_d, *blockTotals_d;
    int threads = SEG_SIZE;

    // bad block sizes break the shuffle mask or overflow warpsums_s
    if (threads % WARP_SIZE != 0 || threads > SEG_SIZE) {
      fprintf(stderr,
              "bad block size %d: must be a multiple of %d and at most %d\n",
              threads, WARP_SIZE, SEG_SIZE);
      return EXIT_FAILURE;
    }

    int gd = (numElements + threads - 1) / threads;
    if (gd < 1)
      gd = 1;
    int totalsBytes = sizeof(float) * gd;

    cudaMalloc(&src_d, numBytes);
    cudaMalloc(&dst_d, numBytes);
    cudaMalloc(&blockTotals_d, totalsBytes);

    cudaMemcpy(src_d, src, numBytes, cudaMemcpyHostToDevice);
    scanBlock<<<gd, threads>>>(dst_d, blockTotals_d, src_d, numElements);
    // step 2 runs on one block and walks the totals in chunks so gd can be
    // any size
    scanBlockTotals<<<1, threads>>>(blockTotals_d, gd);
    addBlockOffsets<<<gd, threads>>>(dst_d, blockTotals_d, numElements);
    cudaMemcpy(dst, dst_d, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(src_d);
    cudaFree(dst_d);
    cudaFree(blockTotals_d);
  }

  for (int i = 0; i < numElements; ++i)
    printf("dst[%i] == %f\n", i, dst[i]);

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}

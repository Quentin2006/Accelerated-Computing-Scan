
#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../bench.cuh"
#include "../common.cuh" // BINOP, binop, identity, CLI flags, output
//#define SEG_SIZE 256
#define BLOCK_DIM 256
#define COARSE_FACTOR 8
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_DIM/WARP_SIZE)

//pg 238
__device__ unsigned int warpIdx() {return threadIdx.x/WARP_SIZE;}


//pg 239
__device__ unsigned int laneIdx() {return threadIdx.x%WARP_SIZE;}

//pg 264; fig 11.8
__device__ inline float warpScan(float val, enum BINOP op)
{
	for(unsigned int stride = 1; stride < WARP_SIZE; stride *= 2)
	{
		float leftVal = __shfl_up_sync(0xffffffff, val, stride);
		if(laneIdx() >= stride)
		{
			val = binop(val, leftVal, op);
		}
	}
	return val;
}

//pg 264, fig 11.9
__device__ inline float blockScan(float val, enum BINOP op)
{
	//warp-level scan
	val = warpScan(val, op);
	//collect warp ops
	__shared__ float warpOps_s[NUM_WARPS];
	if (laneIdx() == WARP_SIZE - 1)
	{
		warpOps_s[warpIdx()] = val;
	}
	__syncthreads();

	if(warpIdx() == 0)
	{
		float warpOp = (threadIdx.x<NUM_WARPS)?warpOps_s[threadIdx.x]:identity(op);
		warpOp = warpScan(warpOp, op);
		if (threadIdx.x < NUM_WARPS)
		{
			warpOps_s[threadIdx.x] = warpOp;

		}
	}
	__syncthreads();

	if(warpIdx() > 0)
	{
		val = binop(val, warpOps_s[warpIdx() - 1], op);
	}
	return val;

		
}

// step 1: scan one COARSE_FACTOR*BLOCK_DIM (2048) element tile per block.
// From the textbook, pg 271, fig 11.13, with two additions: bounds checks on
// the global load and store, so N doesn't have to be a multiple of 2048, and
// the tile's total is saved for step 2.
__global__ void scanBlock(float *dst, float *blockTotals, const float *src,
                          unsigned int N, enum BINOP op)
{
	unsigned int blockSegment = blockIdx.x*COARSE_FACTOR*BLOCK_DIM;
	//load data to shared memory. Slots past N get the operator's identity,
	//which never changes a result. The check guards the access instead of
	//returning early, because every thread has to reach the __syncthreads and
	//the full-warp shuffles below.
	__shared__ float buffer_s[COARSE_FACTOR*BLOCK_DIM];
	for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
	{
		unsigned int idx = blockSegment + c*BLOCK_DIM + threadIdx.x;
		buffer_s[c*BLOCK_DIM + threadIdx.x] = (idx < N) ? src[idx] : identity(op);
	}
	__syncthreads();
	//scan thread subsegment
	unsigned int threadSegment = threadIdx.x*COARSE_FACTOR;
	float buffer_r[COARSE_FACTOR];
	buffer_r[0] = buffer_s[threadSegment];
	#pragma unroll
	for (unsigned int c= 1; c < COARSE_FACTOR; ++c)
	{
		buffer_r[c] = binop(buffer_s[threadSegment + c], buffer_r[c-1], op);
		
	}
	// block-wide scan of thread ops
	float threadOp = buffer_r[COARSE_FACTOR-1];
	threadOp = blockScan(threadOp, op);
	//the last thread's inclusive result is the whole tile's total
	if (threadIdx.x == BLOCK_DIM - 1)
	{
		blockTotals[blockIdx.x] = threadOp;
	}
	//collect thread partial ops
	__shared__ float threadOps[BLOCK_DIM];
	threadOps[threadIdx.x] = threadOp;
	__syncthreads();
	//add/sub/etc. previous thread's partial ops
	float prevPartialOp = (threadIdx.x>0)?threadOps[threadIdx.x-1]:identity(op);
	#pragma unroll
	for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
	{
		buffer_s[threadSegment + c] = binop(buffer_r[c], prevPartialOp, op);
	}
	__syncthreads();
	//write output, skipping the padded slots past N
	for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
	{
		unsigned int idx = blockSegment + c*BLOCK_DIM + threadIdx.x;
		if (idx < N)
		{
			dst[idx] = buffer_s[c*BLOCK_DIM + threadIdx.x];
		}
	}
}

// step 2: scan the tile totals in place. One block walks them BLOCK_DIM at a
// time with blockScan, carrying each chunk's total into the next, so any
// number of tiles works.
__global__ void scanBlockTotals(float *blockTotals, unsigned int numBlocks,
                                enum BINOP op)
{
	__shared__ float carry_s;
	if (threadIdx.x == 0)
	{
		carry_s = identity(op);
	}
	__syncthreads();

	for (unsigned int base = 0; base < numBlocks; base += BLOCK_DIM)
	{
		unsigned int t = base + threadIdx.x;
		float val = (t < numBlocks) ? blockTotals[t] : identity(op);
		val = blockScan(val, op);
		if (t < numBlocks)
		{
			blockTotals[t] = binop(val, carry_s, op);
		}
		__syncthreads();
		//the last thread's result is this chunk's total
		if (threadIdx.x == BLOCK_DIM - 1)
		{
			carry_s = binop(carry_s, val, op);
		}
		__syncthreads();
	}
}

// step 3: combine every element of tile b > 0 with the total of tiles 0..b-1.
// One block per tile and COARSE_FACTOR elements per thread, using the same
// coalesced indexing as the load in step 1.
__global__ void addBlockOffsets(float *dst, const float *blockTotals,
                                unsigned int N, enum BINOP op)
{
	//the first tile has nothing before it. Returning early is fine here: this
	//kernel has no __syncthreads, and the whole block returns together.
	if (blockIdx.x == 0)
	{
		return;
	}
	float offset = blockTotals[blockIdx.x - 1];
	unsigned int blockSegment = blockIdx.x*COARSE_FACTOR*BLOCK_DIM;
	for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
	{
		unsigned int idx = blockSegment + c*BLOCK_DIM + threadIdx.x;
		if (idx < N)
		{
			dst[idx] = binop(dst[idx], offset, op);
		}
	}
}

int main(int argc, char** argv) {
  float *src, *dst;
  //Num elements now taken as arg; --op and --input pick the operator and the
  //repeated input values
  Args args = parse_args(argc, argv, 1 << 24);
  size_t numElements = args.n;
  size_t numBytes = sizeof(float) * numElements;

  src = (float *)malloc(numBytes);
  dst = (float *)malloc(numBytes);

  fill_input(src, numElements, args.input);

	bench::print_device();//added from bench.cuh
  // calling cuda ker
  {
    float *src_d, *dst_d, *blockTotals_d;
    int threads = BLOCK_DIM;//was SEG_SIZE
	const size_t perBlock = COARSE_FACTOR * BLOCK_DIM;
    int gd = (int)((numElements + perBlock - 1)/perBlock);

    CUDA_CHECK(cudaMalloc(&src_d, numBytes));
    CUDA_CHECK(cudaMalloc(&dst_d, numBytes));
    CUDA_CHECK(cudaMalloc(&blockTotals_d, sizeof(float) * gd)); // one total per tile

    CUDA_CHECK(cudaMemcpy(src_d, src, numBytes, cudaMemcpyHostToDevice));
	// all three steps run inside one timed lambda, so the time covers the
	// whole scan. Each run rewrites dst_d and blockTotals_d from scratch, so
	// repeating it is safe.
	bench::Stats s = bench::time_gpu([&]{
		scanBlock<<<gd, threads>>>(dst_d, blockTotals_d, src_d, numElements,
		                           args.op);
		scanBlockTotals<<<1, threads>>>(blockTotals_d, gd, args.op);
		addBlockOffsets<<<gd, threads>>>(dst_d, blockTotals_d, numElements,
		                                 args.op);
	});
    // BENCH: credited work is the scan minimum (8N bytes, N-1 ops).
    // Estimate, with S = 2048 elements per tile and G = gd tiles:
    //   bytes: step 1 reads src and writes dst (8N) plus G totals (4G),
    //          step 2 reads and writes the totals (8G),
    //          step 3 reads and writes dst again (8N) plus one total per
    //          tile (4G)                              -> 16N + 16G
    //   ops:   step 1, per tile: 7 ops per thread for the register scan
    //          (7*256 = 1792), a 256-value blockScan (8 warp scans of 129
    //          ops, a 129-op scan of warp totals, 7*32 = 224 adds = 1385),
    //          then 8 ops per thread to apply the earlier threads' total
    //          (2048) -> 5225 ops (~2.55 per element). Step 2 does
    //          ceil(G/256) blockScans plus 257 carry ops per chunk, and step 3
    //          does N - S ops                         -> ~3.55N
    const double G = gd, n = (double)numElements;
    const double blockScanOps = 8 * 129 + 129 + 224;
    const double tileOps = 7 * BLOCK_DIM + blockScanOps +
                           COARSE_FACTOR * BLOCK_DIM;
    const double chunks = (gd + BLOCK_DIM - 1) / BLOCK_DIM;
    double estBytes = 16.0 * n + 16.0 * G;
    double estOps = tileOps * G + chunks * (blockScanOps + BLOCK_DIM + 1) +
                    (n > perBlock ? n - perBlock : 0.0);
    bench::report("register-tiling", op_name(args.op), numElements, s,
	bench::scan_min_bytes(numElements),
	bench::scan_min_flops(numElements),
	estBytes, estOps);

    CUDA_CHECK(cudaMemcpy(dst, dst_d, numBytes, cudaMemcpyDeviceToHost));
	cudaFree(src_d);
	cudaFree(dst_d);
	cudaFree(blockTotals_d);
  }

  write_output(dst, numElements); // all N results, one line on stdout

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}

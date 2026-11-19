#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda/atomic>
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

__device__ inline float interBlockScan(float val, unsigned int bid, float* partialOps, unsigned int* flags, enum BINOP op)
{
    __shared__ float prevBlockPartialOp_s;
    if (threadIdx.x == BLOCK_DIM - 1)
    {
        if (bid >0)
        {
            //wait for previous block to pass its partial op. An acquire load
            //gives the same guarantee as the textbook's fetch_add(0) without
            //an atomic write on every check.
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                prevBLockFlag_ref(flags[bid-1]);
                while(prevBLockFlag_ref.load(cuda::memory_order_acquire)==0){}
                //read previous block's partial op
                prevBlockPartialOp_s = partialOps[bid-1];

        }else{
            prevBlockPartialOp_s = identity(op);
        }
        //Write this block's partial op
        partialOps[bid] = binop(prevBlockPartialOp_s, val, op);
        //Set this block's flag
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            myBlockFlag_ref(flags[bid]);
        myBlockFlag_ref.fetch_add(1, cuda::memory_order_release);
    }
    __syncthreads();
    return prevBlockPartialOp_s;
}

// single-pass scan, from the textbook's register-tiled kernel plus the chained
// hand-off between blocks above. blockCounter and flags must be zero before
// every launch (main resets them inside the timed lambda).
__global__ void scan_kernel(const float* input, float* output, unsigned int* blockCounter,
float* partialOps, unsigned int* flags, unsigned int N, enum BINOP op)
{
    //assign block index dynamically
    __shared__ unsigned int bid_s;
    if (threadIdx.x == 0)
    {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            blockCounter_ref(*blockCounter);
            bid_s = blockCounter_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    unsigned int bid = bid_s;
    //BLOCK_DIM rather than blockDim.x, so this always matches the shared
    //buffer sizes below
    unsigned int blockSegment = COARSE_FACTOR*bid*BLOCK_DIM;
    //load data to shared memory. Slots past N get the operator's identity,
    //which never changes a result. The check guards the access instead of
    //returning early, because every thread has to reach the __syncthreads and
    //the full-warp shuffles below.
    __shared__ float buffer_s[COARSE_FACTOR*BLOCK_DIM];
    for (unsigned int c= 0 ; c < COARSE_FACTOR; ++c)
    {
        unsigned int idx = blockSegment + c*BLOCK_DIM + threadIdx.x;
        buffer_s[c*BLOCK_DIM + threadIdx.x] = (idx < N) ? input[idx] : identity(op);
    }
    __syncthreads();
    //Scan thread subsegment
    unsigned int threadSegment = threadIdx.x*COARSE_FACTOR;
    float buffer_r[COARSE_FACTOR];
    buffer_r[0] = buffer_s[threadSegment];
    #pragma unroll
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c)
    {
        buffer_r[c] = binop(buffer_s[threadSegment+c], buffer_r[c-1], op);
    }
    //Block-wide scan of thread ops
    float threadOp = buffer_r[COARSE_FACTOR - 1];
    threadOp = blockScan(threadOp, op);
    //Collect thread partial ops
    __shared__ float threadOps[BLOCK_DIM];
    threadOps[threadIdx.x] = threadOp;
    __syncthreads();
    //Apply previous thread's partial op
    float prevPartialOp = (threadIdx.x>0)?threadOps[threadIdx.x-1]:identity(op);
    #pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
    {
        buffer_r[c] = binop(buffer_r[c], prevPartialOp, op);
    }
    //Scan block partial ops
    float prevBlockPartialOp =
        interBlockScan(buffer_r[COARSE_FACTOR-1], bid, partialOps, flags, op);
    //Apply previous block's partial op
    #pragma unroll
    for (unsigned int c= 0; c < COARSE_FACTOR; ++c)
    {
        buffer_s[threadSegment + c] = binop(buffer_r[c], prevBlockPartialOp, op);
    }
    __syncthreads();
    //write output, skipping the padded slots past N
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
    {
        unsigned int idx = blockSegment + c*BLOCK_DIM + threadIdx.x;
        if (idx < N)
        {
            output[idx] = buffer_s[c*BLOCK_DIM + threadIdx.x];
        }
    }
}

int main(int argc, char** argv) {
  float *src, *dst;
  //Num elements taken as arg; --op and --input pick the operator and the
  //repeated input values
  Args args = parse_args(argc, argv, 1 << 24);
  size_t numElements = args.n;
  size_t numBytes = sizeof(float) * numElements;

  src = (float *)malloc(numBytes);
  dst = (float *)malloc(numBytes);

  fill_input(src, numElements, args.input);

  bench::print_device();
  // calling cuda ker
  {
    float *src_d, *dst_d, *partialOps_d;
    unsigned int *sync_d;
    int threads = BLOCK_DIM;
    const size_t perBlock = COARSE_FACTOR * BLOCK_DIM;
    int gd = (int)((numElements + perBlock - 1)/perBlock);

    CUDA_CHECK(cudaMalloc(&src_d, numBytes));
    CUDA_CHECK(cudaMalloc(&dst_d, numBytes));
    CUDA_CHECK(cudaMalloc(&partialOps_d, sizeof(float) * gd)); // one per tile
    // the block counter and the gd flags share one allocation, so a single
    // memset resets both: sync_d[0] is the counter, sync_d[1..gd] the flags
    const size_t syncBytes = sizeof(unsigned int) * (gd + 1);
    CUDA_CHECK(cudaMalloc(&sync_d, syncBytes));
    unsigned int *blockCounter_d = sync_d;
    unsigned int *flags_d = sync_d + 1;

    CUDA_CHECK(cudaMemcpy(src_d, src, numBytes, cudaMemcpyHostToDevice));
    // the counter and flags have to start at zero on every run, or the second
    // run hands out block IDs past the end of the data and skips every wait.
    // The reset is inside the timed lambda because it's part of what this
    // scan costs (CUB does the same internally). partialOps_d needs no reset:
    // each value is written before anything reads it.
    bench::Stats s = bench::time_gpu([&]{
      CUDA_CHECK(cudaMemsetAsync(sync_d, 0, syncBytes));
      scan_kernel<<<gd, threads>>>(src_d, dst_d, blockCounter_d, partialOps_d,
                                   flags_d, numElements, args.op);
    });
    // BENCH: credited work is the scan minimum (8N bytes, N-1 ops).
    // Estimate, with S = 2048 elements per tile and G = gd tiles:
    //   bytes: src read once and dst written once (8N). Per tile, the
    //          hand-off adds the counter fetch_add (8), polling the previous
    //          flag (4), reading the previous partial op (4), writing this
    //          one (4), the flag fetch_add (8), and the memset of its flag
    //          (4) -> 32 bytes                     -> 8N + 32G
    //   ops:   per tile, the same 5225 ops as register-tiling's step 1
    //          (7*256 register scan + 1385 blockScan + 2048 to apply the
    //          earlier threads' total), plus 1 op to build this tile's
    //          partial op and 2048 to apply the earlier tiles' total
    //          -> 7274 ops                          -> ~3.55N
    const double G = gd, n = (double)numElements;
    const double blockScanOps = 8 * 129 + 129 + 224;
    const double tileOps = 7 * BLOCK_DIM + blockScanOps +
                           COARSE_FACTOR * BLOCK_DIM + 1 +
                           COARSE_FACTOR * BLOCK_DIM;
    double estBytes = 8.0 * n + 32.0 * G;
    double estOps = tileOps * G;
    bench::report("inter-block-scan", op_name(args.op), numElements, s,
                  bench::scan_min_bytes(numElements),
                  bench::scan_min_flops(numElements),
                  estBytes, estOps);

    CUDA_CHECK(cudaMemcpy(dst, dst_d, numBytes, cudaMemcpyDeviceToHost));
    cudaFree(src_d);
    cudaFree(dst_d);
    cudaFree(partialOps_d);
    cudaFree(sync_d);
  }

  write_output(dst, numElements); // all N results, one line on stdout

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}
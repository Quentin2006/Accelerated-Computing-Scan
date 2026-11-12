
#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../bench.cuh"
//#define SEG_SIZE 256
#define BLOCK_DIM 256
#define COARSE_FACTOR 8
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_DIM/WARP_SIZE)

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
		float warpOp = (threadIdx.x<NUM_WARPS)?warpOps_s[threadIdx.x]:0.0f;
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

// copied from the textbook, pg 271, fig 11.13
__global__ void scan(float *dst, const float *src, unsigned int N,
                     enum BINOP op) 
{
	unsigned int blockSegment = blockIdx.x*COARSE_FACTOR*BLOCK_DIM;
	//load data to shared memory
	__shared__ float buffer_s[COARSE_FACTOR*BLOCK_DIM];
	for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
	{
		buffer_s[c*BLOCK_DIM + threadIdx.x] = 
			src[blockSegment + c*BLOCK_DIM + threadIdx.x];
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
	//collect thread partial ops
	__shared__ float threadOps[BLOCK_DIM];
	threadOps[threadIdx.x] = threadOp;
	__syncthreads();
	//add/sub/etc. previous thread's partial ops
	float prevPartialOp = (threadIdx.x>0)?threadOps[threadIdx.x-1]:0.0f;
	#pragma unroll
	for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
	{
		buffer_s[threadSegment + c] = binop(buffer_r[c], prevPartialOp, op);
	}
	__syncthreads();
	//write output
	for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
	{
		dst[blockSegment + c*BLOCK_DIM + threadIdx.x] = 
			buffer_s[c*BLOCK_DIM + threadIdx.x];
	}
}

int main(int argc, char** argv) {
  float *src, *dst;
  size_t numElements = bench::arg_n(argc, argv, 1 << 24); //Num elements now taken as arg
  size_t numBytes = sizeof(float) * numElements;

  src = (float *)malloc(numBytes);
  dst = (float *)malloc(numBytes);

  for (size_t i = 0; i < numElements; ++i)
    src[i] = 1.0f;

	bench::print_device();//added from bench.cuh
  // calling cuda ker
  {
    float *src_d, *dst_d;
    int threads = BLOCK_DIM;//was SEG_SIZE
	const size_t perBlock = COARSE_FACTOR * BLOCK_DIM;
    int gd = (int)((numElements + perBlock - 1)/perBlock);

    CUDA_CHECK(cudaMalloc(&src_d, numBytes));
    CUDA_CHECK(cudaMalloc(&dst_d, numBytes));

    CUDA_CHECK(cudaMemcpy(src_d, src, numBytes, cudaMemcpyHostToDevice));
	bench::Stats s = bench::time_gpu([&]{
		scan<<<gd, threads>>>(dst_d, src_d, numElements, ADD);
	});
    bench::report("register-tiling", numElements, s,
	bench::scan_min_bytes(numElements),
	bench::scan_min_flops(numElements));

    CUDA_CHECK(cudaMemcpy(dst, dst_d, numBytes, cudaMemcpyDeviceToHost));
	cudaFree(src_d);
	cudaFree(dst_d);
  }

  for (size_t i = 0; i < 4 && i < numElements; ++i)
    printf("dst[%zu] == %f\n", i, dst[i]);

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}

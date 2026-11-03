#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

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

int main(void) {
  float *src, *dst;
  int numElements = 10;
  int numBytes = sizeof(float) * numElements;

  src = (float *)malloc(numBytes);
  dst = (float *)malloc(numBytes);

  for (int i = 0; i < numElements; ++i)
    src[i] = 1.0f;

  // calling cuda ker
  {
    float *src_d, *dst_d;
    int threads = SEG_SIZE;
    int gd = ceil(numElements / (float)threads);

    cudaMalloc(&src_d, numBytes);
    cudaMalloc(&dst_d, numBytes);

    cudaMemcpy(src_d, src, numBytes, cudaMemcpyHostToDevice);
    scan<<<gd, threads>>>(dst_d, src_d, numElements, ADD);
    cudaMemcpy(dst, dst_d, numBytes, cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < numElements; ++i)
    printf("dst[%i] == %f\n", i, dst[i]);

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}

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

// double buffered kogge stone scan, currently doesnt support many blocks
__global__ void scan(float *dst, const float *src, unsigned int N,
                     enum BINOP op) {
  __shared__ float buffer1_s[SEG_SIZE];
  __shared__ float buffer2_s[SEG_SIZE];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  float *inBuffer_s = buffer1_s;
  float *outBuffer_s = buffer2_s;

  inBuffer_s[threadIdx.x] = (i < N) ? src[i] : 0.0f;

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (threadIdx.x >= stride) {
      outBuffer_s[threadIdx.x] =
          binop(inBuffer_s[threadIdx.x], inBuffer_s[threadIdx.x - stride], op);
    } else {
      outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
    }
    float *temp = inBuffer_s;
    inBuffer_s = outBuffer_s;
    outBuffer_s = temp;
  }
  if (i < N) {
    dst[i] = inBuffer_s[threadIdx.x];
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

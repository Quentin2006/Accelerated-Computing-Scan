#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

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
    int threads = 256;
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

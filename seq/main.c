#include <stdio.h>
#include <stdlib.h>

// NOTE: we can use any bianary operator
enum BINOP { ADD, SUB, MUL, DIV };

float binop(float a, float b, enum BINOP op) {
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

void scan(float *dst, const float *src, int N, enum BINOP op) {
  if (N <= 0 || !dst || !src)
    return;

  dst[0] = src[0];

  for (int i = 1; i < N; ++i) {
    dst[i] = binop(src[i], dst[i - 1], op);
  }
}

int main(void) {
  float *src, *dst;
  int numElements = 10;
  int numBytes = sizeof(float) * numElements;

  src = malloc(numBytes);
  dst = malloc(numBytes);

  for (int i = 0; i < numElements; ++i)
    src[i] = 1.0f;

  scan(dst, src, numElements, ADD);

  for (int i = 0; i < numElements; ++i)
    printf("dst[%i] == %f\n", i, dst[i]);

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}

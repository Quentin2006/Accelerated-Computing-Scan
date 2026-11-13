#include <cub/cub.cuh>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../bench.cuh"  // BENCH: timing/reporting header (bench::, CUDA_CHECK)
#include "../common.cuh" // BINOP, binop, identity, CLI flags, output

// PMPP 11.11: the library baseline. cub::DeviceScan::InclusiveScan is a
// single-pass scan with decoupled lookback, tuned per architecture. The other
// kernels are compared against its output and its speed.

// CUB takes the operator as a functor; this one applies --op.
struct ScanOp {
  enum BINOP op;
  __device__ float operator()(float a, float b) const { return binop(a, b, op); }
};

int main(int argc, char **argv) {
  Args args = parse_args(argc, argv, 1 << 24);
  size_t numElements = args.n;
  size_t numBytes = sizeof(float) * numElements;

  float *src = (float *)malloc(numBytes);
  float *dst = (float *)malloc(numBytes);
  fill_input(src, numElements, args.input);

  bench::print_device();

  {
    float *src_d, *dst_d;
    CUDA_CHECK(cudaMalloc(&src_d, numBytes));
    CUDA_CHECK(cudaMalloc(&dst_d, numBytes));
    CUDA_CHECK(cudaMemcpy(src_d, src, numBytes, cudaMemcpyHostToDevice));

    // First call with no storage only asks how much temporary storage CUB
    // needs. It's allocated here, outside the timed region.
    ScanOp op{args.op};
    void *temp_d = nullptr;
    size_t tempBytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(temp_d, tempBytes, src_d, dst_d,
                                              op, numElements));
    CUDA_CHECK(cudaMalloc(&temp_d, tempBytes));

    bench::Stats s = bench::time_gpu([&] {
      CUDA_CHECK(cub::DeviceScan::InclusiveScan(temp_d, tempBytes, src_d,
                                                dst_d, op, numElements));
    });
    // BENCH: credited work is the scan minimum (8N bytes, N-1 ops).
    // Estimate: one pass, so src is read once and dst written once (8N
    // bytes); the per-tile lookback status words are small enough to ignore.
    // Each element is combined about once (~N ops).
    double n = (double)numElements;
    bench::report("cub", op_name(args.op), numElements, s,
                  bench::scan_min_bytes(numElements),
                  bench::scan_min_flops(numElements), 8.0 * n, n);

    CUDA_CHECK(cudaMemcpy(dst, dst_d, numBytes, cudaMemcpyDeviceToHost));
    cudaFree(temp_d);
    cudaFree(src_d);
    cudaFree(dst_d);
  }

  write_output(dst, numElements); // all N results, one line on stdout

  free(src);
  free(dst);
  return EXIT_SUCCESS;
}

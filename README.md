# Accelerated-Computing-Scan

Inclusive scan (prefix sum) kernels in CUDA, plus a sequential CPU baseline in `seq/`.
Every CUDA kernel is timed with the shared header [`bench.cuh`](bench.cuh).

| Directory | Kernel | Notes |
|---|---|---|
| `naive/` | One thread per output, loops over all earlier inputs | O(N²): keep N ≤ 2^20 |
| `kogge-stone/` | Kogge-Stone in shared memory | Scans each 256-element block separately |
| `kogge-stone-double-buffering/` | Kogge-Stone with two shared buffers | Scans each 256-element block separately |
| `scan-scan-add/` | Warp-shuffle block scan → scan block totals → add offsets | Full scan of any N |
| `register-tiling/` | 8 elements per thread in registers + block scan | Scans each 2048-element block separately; N must be a multiple of 2048 (2^11 and up) |

## Benchmark one kernel

```bash
cd kogge-stone
make                  # nvcc -O3 -arch=sm_86 (RTX A2000)
make run              # N = 2^24 (2^16 for naive)
make run N=2^20       # any size
./scan 16777216       # same as ./scan 2^24
```

Each run prints the GPU's specs and one timing line. Example output (exact numbers vary):

```
[bench] NVIDIA RTX A2000 12GB  sm_86  26 SMs  L2 3.0 MB  peak 288.0 GB/s  7987.2 GFLOP/s (FP32)  ridge 27.7 FLOP/B
[bench] register-tiling   N=16777216  median 0.5239 ms (min 0.5233, sd 0.0511, 20 reps)  256.20 GB/s (88.9% peak)  32.03 GFLOP/s  AI 0.125 FLOP/B
```

To sweep one kernel over many sizes into a CSV file:

```bash
cd kogge-stone && make
for k in $(seq 11 28); do BENCH_CSV=results.csv ./scan 2^$k; done
```

`BENCH_CSV=<file>` appends one row per run and writes the header row if the file is new.
Delete the file first to start fresh.

## Sweep all kernels

From the repo root:

```bash
rm -f results.csv
for d in naive kogge-stone kogge-stone-double-buffering scan-scan-add register-tiling; do
  (cd $d && make -s)
done
for d in kogge-stone kogge-stone-double-buffering scan-scan-add register-tiling; do
  for k in $(seq 11 28); do (cd $d && BENCH_CSV=../results.csv ./scan 2^$k); done
done
for k in $(seq 11 20); do (cd naive && BENCH_CSV=../results.csv ./scan 2^$k); done
```

All kernels then share one `results.csv`, with a `kernel` column telling the rows apart.

## CSV columns

| Column | Meaning |
|---|---|
| `kernel`, `device`, `n` | What ran, where, and on how many elements |
| `median_ms`, `min_ms`, `mean_ms`, `stddev_ms`, `reps` | Kernel time over 20 timed runs (after 3 warm-up runs, with L2 cleared before each) |
| `bytes`, `flops` | Work credited to the kernel: by default the minimum for any scan, 8N bytes and N−1 ops |
| `gbs`, `gflops` | `bytes` and `flops` divided by the median time |
| `ai` | Arithmetic intensity, `flops / bytes` |
| `pct_peak_bw`, `pct_peak_flops` | Percentage of the GPU's theoretical peak |
| `peak_gbs`, `peak_gflops` | Theoretical peaks, from the driver |

## Notes

- **What's timed:** only the GPU work inside `bench::time_gpu(...)`, not `cudaMalloc` or `cudaMemcpy`.
  For `scan-scan-add`, that's all three kernels together.
- **Default work counts:** `bytes` and `flops` are the minimum work for any scan, so GB/s compares kernels fairly.
  Each kernel's actual operation count is in a `// BENCH:` comment next to its `bench::report` call.
- **Memory check:** run `compute-sanitizer ./scan 2^16` after changing a kernel to catch out-of-bounds accesses.
- **Profiling:** `ncu` needs GPU performance-counter permission, which the lab machine currently blocks
  (`ERR_NVGPUCTRPERM`). `nsys profile --stats=true ./scan 2^24` works without it.

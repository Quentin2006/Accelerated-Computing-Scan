# Accelerated-Computing-Scan

Inclusive scan (prefix sum) kernels in CUDA, plus a sequential CPU baseline in `seq/`.
Every CUDA kernel is timed with the shared header [`bench.cuh`](bench.cuh) and shares its
operators, command-line flags and output code through [`common.cuh`](common.cuh).

| Directory | Kernel | Notes |
|---|---|---|
| `naive/` | One thread per output, loops over all earlier inputs | O(N²): keep N ≤ 2^20 |
| `kogge-stone/` | Kogge-Stone in shared memory | Scans each 256-element block separately |
| `kogge-stone-double-buffering/` | Kogge-Stone with two shared buffers | Scans each 256-element block separately |
| `scan-scan-add/` | Warp-shuffle block scan → scan block totals → add offsets | Full scan of any N |
| `register-tiling/` | 8 elements per thread in registers + block scan | Scans each 2048-element block separately; N must be a multiple of 2048 (2^11 and up), otherwise it reads and writes out of bounds |
| `cub/` | `cub::DeviceScan::InclusiveScan` (PMPP 11.11) | Library baseline: the reference output and speed the others are compared against |

## Benchmark one kernel

```bash
cd kogge-stone
make                  # nvcc -O3 -arch=sm_86 (Ampere: RTX 3060 Ti, RTX A2000)
make run              # N = 2^24 (2^16 for naive)
make run N=2^20       # any size
./scan.out 16777216   # same as ./scan.out 2^24
```

### Command line

```
./scan.out [N] [-o|--op add|mul|max|min] [-i|--input v1,v2,...]
```

| Argument | Default | Meaning |
|---|---|---|
| `N` | 2^24 (2^16 for naive) | Element count, as a number (`16777216`) or a power of two (`2^24`) |
| `-o`, `--op` | `add` | The scan operator: `add`, `mul`, `max` or `min` |
| `-i`, `--input` | `1` | Comma-separated values, repeated until they fill N elements |

Order doesn't matter; `--op=mul`, `-o mul` and `-omul` all work; `-h` prints help.
A bad argument prints the usage to stderr and exits with status 1.

```bash
$ ./scan.out 5 --input 1,2 | tail -n 1
1,3,4,6,7
$ ./scan.out 6 --op max --input 3,1,4,1,5,9 | tail -n 1
3,3,4,4,5,9
```

### Output

Everything goes to stdout, in this order:

1. the GPU's specs (`[bench]` line),
2. the timing (`[bench]` line),
3. all N results on one line, comma-separated, printed with `%.9g` (enough digits to
   round-trip a float). This is always the last line, so `tail -n 1` gets the values.

At 2^28 elements the last line is about 3 GB, so redirect stdout (`> /dev/null`) when you only
want timings; `BENCH_CSV` still records them. Example (exact numbers vary; values cut short):

```
[bench] NVIDIA GeForce RTX 3060 Ti  sm_86  38 SMs  L2 3.0 MB  peak 448.1 GB/s  17510.4 GFLOP/s (FP32)  ridge 39.1 FLOP/B
[bench] register-tiling          N=16777216    median    0.3164 ms (min 0.3154, sd 0.0819, 20 reps)    424.18 GB/s ( 94.7% peak)     53.02 GFLOP/s  AI 0.125 FLOP/B  | est   424.18 GB/s    135.28 GFLOP/s  AI 0.319 FLOP/B
1,2,3,4,5,6,7,8,9,10,...
```

The figures before the `|` use the minimum work for any scan. The `est` figures use the kernel's
own estimate of the bytes it moves and the operations it does (see [CSV columns](#csv-columns)).

To sweep one kernel over many sizes into a CSV file:

```bash
cd kogge-stone && make
for k in $(seq 11 28); do BENCH_CSV=results.csv ./scan.out 2^$k > /dev/null; done
```

`BENCH_CSV=<file>` appends one row per run and writes the header row if the file is new.
Delete the file first to start fresh.

## Build, check and benchmark everything

[`roofline.py`](roofline.py) does the whole loop from the repo root. It is a
[uv script](https://docs.astral.sh/uv/guides/scripts/), so uv installs its dependencies on first run.

```bash
uv run roofline.py                        # default: N in range(1, 1100000, 157143)
uv run roofline.py --range 1 5000 7       # N in range(1, 5000, 7)
uv run roofline.py --range 2049 2050      # just N = 2049
```

`--range START STOP [STEP]` takes the same arguments as Python's `range()` and is the only flag.
For every run the script:

1. runs `make clean` and `make` in every directory (in parallel) and deletes each `<dir>/results.csv`;
2. for each operator and each N, runs `cub/scan.out`, then every other kernel, with
   `BENCH_CSV=results.csv`, so each run appends its timing to its own directory's CSV;
3. compares each kernel's output (the last stdout line) with CUB's, to a relative tolerance of
   1e-6, and prints `PASS` or `FAIL` per kernel with the first wrong index of up to 5 failures.

Naive stops at N = 2^16. Runs happen one at a time so they don't share the GPU and skew each
other's timings. The exit status is 1 if any kernel fails.

The inputs keep every result exact in float32: `add` uses `1,2,3`, `max`/`min` use `3,1,4,1,5,9`,
and `mul` uses `2,-1,0.5`, a pattern whose product over any power-of-two block is −2, so a kernel
that restarts the scan at a block boundary gets a visibly wrong answer.

To lint and type-check the script:

```bash
uvx ruff check .
uvx ty check --python "$(uv python find --script roofline.py)" roofline.py
```

## CSV columns

| Column | Meaning |
|---|---|
| `kernel`, `op`, `device`, `n` | What ran, with which operator, where, and on how many elements |
| `median_ms`, `min_ms`, `mean_ms`, `stddev_ms`, `reps` | Kernel time over 20 timed runs (after 3 warm-up runs, with L2 cleared before each) |
| `bytes`, `flops` | Work credited to the kernel: by default the minimum for any scan, 8N bytes and N−1 ops |
| `gbs`, `gflops` | `bytes` and `flops` divided by the median time |
| `ai` | Arithmetic intensity, `flops / bytes` |
| `pct_peak_bw`, `pct_peak_flops` | Percentage of the GPU's theoretical peak |
| `peak_gbs`, `peak_gflops` | Theoretical peaks, from the driver |
| `est_bytes`, `est_flops` | The kernel's own estimate of the bytes it moves to and from DRAM and the operations it does, worked out by hand from the code (not measured). The derivation is in the `// BENCH:` comment next to each `bench::report` call |
| `est_ai`, `est_gbs`, `est_gflops` | `est_flops / est_bytes`, and the estimates divided by the median time |

## Notes

- **What's timed:** only the GPU work inside `bench::time_gpu(...)`, not `cudaMalloc` or `cudaMemcpy`.
  For `scan-scan-add`, that's all three kernels together.
- **Work counts:** `bytes` and `flops` are the minimum work for any scan, so GB/s compares kernels fairly.
  `est_bytes` and `est_flops` are each kernel's own estimate, worked out by counting the global loads
  and stores (4 bytes each) and the `binop` calls in its code, so they show the extra work a kernel
  does on top of the minimum. They are not measured; repeated reads that should hit the cache are
  left out.
- **Padding:** slots past N (and starting values such as a block's carry) hold the operator's
  identity (0 for add, 1 for mul, −inf for max, +inf for min), so they never change a result.
- **Memory check:** run `compute-sanitizer ./scan.out 2049 --op mul --input 1,-1` after changing a kernel to catch
  out-of-bounds accesses.
- **Profiling:** `ncu` needs GPU performance-counter permission, which the lab machine currently blocks
  (`ERR_NVGPUCTRPERM`). `nsys profile --stats=true ./scan.out 2^24` works without it.

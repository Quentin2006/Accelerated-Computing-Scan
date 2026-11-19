# Accelerated-Computing-Scan

Inclusive scan (prefix sum) kernels in CUDA, plus a sequential CPU baseline in `seq/`.
Every CUDA kernel is timed with the shared header [`bench.cuh`](bench.cuh) and shares its
operators, command-line flags and output code through [`common.cuh`](common.cuh).

| Directory | Kernel | Notes |
|---|---|---|
| `naive/` | One thread per output, loops over all earlier inputs | O(N²): keep N ≤ 2^20 |
| `kogge-stone/` | Kogge-Stone block scan in shared memory → scan block totals → add offsets | Full scan of any N; 256 elements per block |
| `kogge-stone-double-buffering/` | Same, with two shared buffers (one `__syncthreads` per stride) | Full scan of any N; 256 elements per block |
| `scan-scan-add/` | Warp-shuffle block scan → scan block totals → add offsets | Full scan of any N; 256 elements per block |
| `register-tiling/` | 8 elements per thread in registers + block scan | Full scan of any N (bounds-checked) |
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

[`bench.py`](bench.py) does the timing loop from the repo root: build, run, check against
CUB. It is a [uv script](https://docs.astral.sh/uv/guides/scripts/), so uv installs its
dependencies (numpy) on first run. It does not draw graphs; it ends by printing
`uv run graph.py` as the next step.

```bash
uv run bench.py                        # default: 40 sizes from 1 to 2^24 (about 10 minutes)
uv run bench.py --points 8 --max 65536 # fewer, smaller sizes
uv run bench.py --range 1 5000 7       # N in range(1, 5000, 7)
uv run bench.py --range 2049 2050      # just N = 2049
```

Only `add` is timed. By default it sweeps `--points 40` sizes, log-spaced from 1 to
`--max 16777216` (2^24), and rounds each one to an odd number so runs land in the middle of a
block, where a kernel that only scans whole blocks goes visibly wrong. Rounding can make
neighbouring sizes collide at the small end; those move up by 2 instead, so the list stays
distinct and increasing, and the largest odd size at or below `--max` is always included.
Each `add` run appends its timing to its directory's `results.csv` (via `BENCH_CSV`) and is
also checked against CUB. The other operators are never timed: `mul`, `max` and `min` are
checked against CUB with no CSV row, at sizes that hit every edge a block scan has — 1, 255,
256, 257, 2047, 2048, 2049, 65535, 1048575 — with the inputs that keep float32 exact (`mul`
uses `2,-1,0.5`, whose product over any power-of-two block is −2, so a kernel that restarts
the scan at a block boundary gets a visibly wrong answer; `max`/`min` use `3,1,4,1,5,9`).

For every run the script:

1. runs `make clean` and `make` in every directory, one at a time, and deletes each `<dir>/results.csv`;
2. for each timed N, runs `cub/scan.out`, then every other kernel, with `BENCH_CSV=results.csv`,
   so each run appends its timing to its own directory's CSV; the correctness-only runs use
   the same commands without `BENCH_CSV`;
3. compares each kernel's output (the last stdout line) with CUB's, to a relative tolerance of
   1e-6, and prints `PASS` or `FAIL` per kernel with the first wrong index of up to 5 failures.

The exit status is 1 if any kernel failed.

Naive is left out of the list (it's O(N²), so it would take far too long at these sizes); run it
by hand with a small N. Runs happen one at a time so they don't share the GPU and skew each
other's timings.

## Graphs

[`graph.py`](graph.py) reads the CSVs `bench.py` wrote and draws the graphs. It needs no GPU
and no `nvcc`, so it runs wherever the CSVs are committed.

```bash
uv run graph.py   # the 15 PNGs; exit 1 if there is nothing to graph
```

Each figure's title names the GPU. There are 15 timing PNGs: five in `graphs/` that put every
kernel on one panel, and a `runtime.png` and `bandwidth.png` in each `<kernel>/` directory next
to the implementation.

The colour and shape of a kernel are the same in every figure: CUB blue circle, Kogge-Stone
orange square, Kogge-Stone-double-buffering green triangle, scan-scan-add red diamond,
register-tiling purple cross.

| File | What it shows, and how to read it |
|---|---|
| `graphs/runtime.png` | Median time against N (N > 1), one line per kernel, with asymmetric error bars: down to the fastest run, up by one standard deviation. Parallel lines mean the kernels scale alike; a jump means a kernel's extra passes start to cost. |
| `graphs/bandwidth.png` | Useful GB/s — the minimum 8N bytes any scan must move, over the median time — as a percentage of the GPU's peak DRAM bandwidth, on a fixed 0–100 % axis. Counting only the minimum work keeps kernels comparable no matter how much extra traffic they create; the line that tops out highest uses the GPU's memory system best. |
| `graphs/per-element.png` | Median time per element, in ns, against N. The floor is the time to move one float in and one float out; anything above it is per-element overhead (extra passes, block scans, barriers). |
| `graphs/vs-cub.png` | Each kernel's median time divided by CUB's at the same N. Below the dashed 1-line means faster than CUB; the shape of the curve shows where a kernel wins or loses. |
| `graphs/roofline.png` | The roofline: x is the minimum ops per estimated byte (`flops / est_bytes`), y is useful GFLOP/s, the dashed roof is the machine's `min(peak GFLOP/s, peak GB/s × AI)`, colour is N (one shared log scale, ticked at the smallest, middle and largest N on the colourbar) and marker shape is the kernel. The x axis is fixed at 0.04–0.2 FLOP/B, where every kernel's intensity sits, and y is zoomed to the data with the roof over that window in view. The legend states both peaks and the ridge point, which is far to the right of the plotted window. |
| `<kernel>/runtime.png` | That kernel's median time against N (N > 1) alone. All five share one y-range, so the kernels compare by flipping between the figures. |
| `<kernel>/bandwidth.png` | That kernel's useful bandwidth as % of peak, on the same fixed 0–100 % axis as `graphs/bandwidth.png`. |

Every N in the CSV except N = 1 is plotted (a single element times launch overhead, not
throughput). Log axes get plain-number tick labels, and minor ticks stay unlabelled.

The graphs are written from `<kernel>/results.csv`, so `uv run graph.py` redraws them from
whatever the last benchmark run left behind. `graphs/` is created if it doesn't exist.

To lint and type-check the scripts:

```bash
uvx ruff check .
uvx ty check --python "$(uv python find --script bench.py)" bench.py
uvx ty check --python "$(uv python find --script graph.py)" graph.py
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
  For the kernels that scan block totals and add offsets, that's all three kernels together.
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

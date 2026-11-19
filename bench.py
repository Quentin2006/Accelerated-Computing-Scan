# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Rebuild every scan, time each one on add, and check every operator against
CUB, appending the timings to <kernel>/results.csv (via BENCH_CSV). Run from
the repo root; `uv run graph.py` draws the graphs from what this records.

uv run bench.py                        # 40 odd sizes from 1 to 2^24 (about 10 minutes)
uv run bench.py --points 8 --max 65536 # fewer, smaller sizes
uv run bench.py --range 1 5000 7       # N in range(1, 5000, 7)
uv run bench.py --range 2049 2050      # just N = 2049

Only add is timed. The other operators are still checked against CUB, with no
timing recorded, at sizes that hit every edge a block scan has:
1, 255, 256, 257, 2047, 2048, 2049, 65535, 1048575.
"""

import argparse
import os
import subprocess

import numpy as np

REFERENCE = "cub"  # library scan the others are checked against
KERNELS = [
    # "naive", # slow
    "kogge-stone",
    "kogge-stone-double-buffering",
    "scan-scan-add",
    "register-tiling",
]
ALL_KERNELS = [REFERENCE, *KERNELS]
CSV_NAME = "results.csv"
OP = "add"  # the only operator that is timed
CHECK_OPS = ("mul", "max", "min")  # checked against CUB, not timed
# Sizes whose edges a block scan can get wrong: empty, one short of / exactly
# at / one past a 256 block, a 2048 register-tiling block, a scanBlockTotals
# chunk (256 blocks), and one short of a 1M sweep point.
CHECK_SIZES = (1, 255, 256, 257, 2047, 2048, 2049, 65535, 1048575)
# --input per operator.
INPUTS = {"add": "1,2,3", "mul": "2,-1,0.5", "max": "3,1,4,1,5,9", "min": "3,1,4,1,5,9"}
TOLERANCE = 1e-6
MAX_FAILS_SHOWN = 5
DEFAULT_POINTS = 40  # how many sizes a plain run sweeps
DEFAULT_MAX = 2**24  # the largest size a plain run sweeps


def build_sizes(points: int, max_n: int) -> list[int]:
    """`points` sizes, log-spaced from 1 to max_n, nudged to odd numbers.

    Odd sizes land in the middle of a block, so a kernel that only scans whole
    blocks gets a visibly wrong answer. Rounding to odd can make the small sizes
    collide; those step up by 2 instead, so the list stays distinct and
    increasing. The largest odd size at or below max_n is always included."""
    top = max_n if max_n % 2 else max_n - 1  # the largest odd N allowed
    if points <= 1 or top < 3:
        return [1]
    sizes: list[int] = []
    for target in np.geomspace(1, top, points):
        n = max(round(target), 1)
        n = n if n % 2 else n - 1  # odd: land on a partial block
        if sizes and n <= sizes[-1]:
            n = sizes[-1] + 2  # keep it distinct and increasing
        if n > top:
            break
        sizes.append(n)
    return sizes


def build() -> None:
    """make clean, make, and start each CSV fresh"""
    for kernel in ALL_KERNELS:
        if os.path.exists(f"{kernel}/{CSV_NAME}"):
            os.remove(f"{kernel}/{CSV_NAME}")
        subprocess.run(["make", "-s", "-C", kernel, "clean", "all"], check=True)


def run_scan(kernel: str, op: str, n: int, *, record: bool) -> np.ndarray | None:
    """Run ./scan.out and return its results (the last stdout line), or None if
    it crashed.

    record=True appends the timing to <kernel>/results.csv (via BENCH_CSV);
    record=False leaves the CSV alone."""
    env = {**os.environ}
    if record:
        env["BENCH_CSV"] = CSV_NAME
    result = subprocess.run(
        ["./scan.out", str(n), "--op", op, "--input", INPUTS[op]],
        cwd=kernel,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    last_line = result.stdout.splitlines()[-1]  # the N comma-separated values
    return np.array(last_line.split(","), dtype=float)


def compare(got: np.ndarray | None, want: np.ndarray) -> str | None:
    """None if got matches want, else what went wrong."""
    if got is None:
        return "crashed"
    if len(got) != len(want):
        return f"printed {len(got)} values, expected {len(want)}"
    wrong = np.flatnonzero(~np.isclose(got, want, rtol=TOLERANCE))
    if len(wrong) > 0:
        return f"first wrong index {wrong[0]}"
    return None


def run_and_check(sizes: list[int]) -> int:
    """Build, run and compare every kernel on add (timed) and the other
    operators (checked only). 1 if any kernel failed, else 0."""
    build()

    failures: dict[str, list[str]] = {kernel: [] for kernel in KERNELS}
    cases: dict[str, int] = {kernel: 0 for kernel in KERNELS}

    def check(want: np.ndarray, op: str, n: int, record: bool) -> None:
        """Compare every kernel with CUB's output; `record` says whether this
        case is timed and appends a CSV row."""
        for kernel in KERNELS:
            cases[kernel] += 1
            problem = compare(run_scan(kernel, op, n, record=record), want)
            if problem is not None:
                failures[kernel].append(f"FAIL {kernel} {op} N={n}: {problem}")

    # Runs are one at a time. add sweeps every size with timing on and is still
    # checked against CUB; the other operators are checked only, at the edge
    # sizes, and leave the CSVs alone.
    runs = [(OP, n, True) for n in sizes]
    runs += [(op, n, False) for op in CHECK_OPS for n in CHECK_SIZES]

    for op, n, record in runs:
        want = run_scan(REFERENCE, op, n, record=record)
        if want is None:
            print(f"{REFERENCE} crashed on {op} N={n}; can't check the others")
            return 1
        check(want, op, n, record)

    for kernel in KERNELS:
        for failure in failures[kernel][:MAX_FAILS_SHOWN]:
            print(failure)
        passed = cases[kernel] - len(failures[kernel])
        status = "FAIL" if failures[kernel] else "PASS"
        print(f"{status} {kernel}: {passed}/{cases[kernel]}")

    return 1 if any(failures.values()) else 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--range",
        nargs="+",
        type=int,
        metavar="N",
        help="START STOP [STEP], as in range(); overrides --points and --max",
    )
    p.add_argument(
        "--points",
        type=int,
        default=DEFAULT_POINTS,
        metavar="N",
        help=f"how many log-spaced sizes to run (default: {DEFAULT_POINTS})",
    )
    p.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX,
        metavar="N",
        help="largest size for the log-spaced sweep, nudged down to odd "
        f"(default: {DEFAULT_MAX})",
    )
    args = p.parse_args()

    if args.range is not None:
        if not 2 <= len(args.range) <= 3:
            p.error("--range takes START STOP [STEP]")
        sizes = list(range(*args.range))
    else:
        if args.points < 1:
            p.error("--points must be at least 1")
        if args.max < 1:
            p.error("--max must be at least 1")
        sizes = build_sizes(args.points, args.max)

    status = run_and_check(sizes)
    print("next: uv run graph.py  # graph these CSVs")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

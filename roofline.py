# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Rebuild every scan, run each one and check it against CUB.

Each run appends its timing to <kernel>/results.csv (via BENCH_CSV), so the
CSVs hold everything needed for the roofline. Run from the repo root.

uv run roofline.py                        # default range of N
uv run roofline.py --range 1 5000 7       # N in range(1, 5000, 7)
uv run roofline.py --range 2049 2050      # just N = 2049
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
CSV_NAME = "results.csv"
# Odd step so N lands on partial blocks
DEFAULT_RANGE = (1, 1100000, 157143)
# --input per operator.
INPUTS = {"add": "1,2,3", "mul": "2,-1,0.5", "max": "3,1,4,1,5,9", "min": "3,1,4,1,5,9"}
TOLERANCE = 1e-6
MAX_FAILS_SHOWN = 5


def build() -> None:
    """make clean, make, and start each CSV fresh"""
    for kernel in [REFERENCE, *KERNELS]:
        if os.path.exists(f"{kernel}/{CSV_NAME}"):
            os.remove(f"{kernel}/{CSV_NAME}")
        subprocess.run(["make", "-s", "-C", kernel, "clean", "all"], check=True)


def run_scan(kernel: str, op: str, n: int) -> np.ndarray | None:
    """Run ./scan.out (which appends a row to its CSV) and return its results,
    or None if it crashed."""
    cmd = f"BENCH_CSV={CSV_NAME} ./scan.out {n} --op {op} --input {INPUTS[op]}"
    result = subprocess.run(
        cmd, shell=True, cwd=kernel, capture_output=True, text=True, check=False
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


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--range",
        nargs="+",
        type=int,
        default=list(DEFAULT_RANGE),
        metavar="N",
        help="START STOP [STEP], as in range() "
        f"(default: {' '.join(map(str, DEFAULT_RANGE))})",
    )
    args = p.parse_args()
    if not 2 <= len(args.range) <= 3:
        p.error("--range takes START STOP [STEP]")
    sizes = range(*args.range)

    build()

    # Runs are one at a time
    failures: dict[str, list[str]] = {kernel: [] for kernel in KERNELS}
    cases: dict[str, int] = {kernel: 0 for kernel in KERNELS}
    for op in INPUTS:
        for n in sizes:
            want = run_scan(REFERENCE, op, n)

            if want is None:
                print(f"{REFERENCE} crashed on {op} N={n}; can't check the others")
                return 1

            for kernel in KERNELS:
                cases[kernel] += 1
                problem = compare(run_scan(kernel, op, n), want)
                if problem is not None:
                    failures[kernel].append(f"FAIL {kernel} {op} N={n}: {problem}")

    for kernel in KERNELS:
        for failure in failures[kernel][:MAX_FAILS_SHOWN]:
            print(failure)
        passed = cases[kernel] - len(failures[kernel])
        status = "FAIL" if failures[kernel] else "PASS"
        print(f"{status} {kernel}: {passed}/{cases[kernel]}")

    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    main()

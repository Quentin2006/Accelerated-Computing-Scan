# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Graph every <kernel>/results.csv that exists: five figures in graphs/ that
compare all kernels on add, and a runtime and bandwidth next to each
implementation.

uv run graph.py   # write the 15 PNGs; exit 1 if there is nothing to graph

Reading the CSVs needs no GPU and no nvcc, so this runs wherever they are
committed. Run it after `uv run bench.py` has recorded the timings.
"""

import csv
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display needed: write the figures to files

import matplotlib.pyplot as plt  # after matplotlib.use: only the Agg backend
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

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
GRAPH_DIR = "graphs"
OP = "add"  # the only operator that is timed and graphed
CMAP = "viridis_r"  # colour by N: light for small N, dark for large
NEUTRAL = "0.35"  # roofline legend markers: colour there means N, so shape only
# One colour per kernel, the same in every figure (matplotlib's tab10 cycle)
COLORS = {kernel: f"C{i}" for i, kernel in enumerate(ALL_KERNELS)}
# One marker shape per kernel, the same in every figure: the roofline uses
# colour for N, so shape is what distinguishes the kernels there.
MARKERS = {
    "cub": "o",
    "kogge-stone": "s",
    "kogge-stone-double-buffering": "^",
    "scan-scan-add": "D",
    "register-tiling": "X",
}


@dataclass(frozen=True)
class Row:
    """The columns of one results.csv row that the graphs use."""

    op: str
    device: str
    n: float
    median_ms: float
    min_ms: float
    stddev_ms: float
    flops: float
    gflops: float
    pct_peak_bw: float
    peak_gbs: float
    peak_gflops: float
    est_bytes: float


def load_results(kernel: str) -> list[Row] | None:
    """Every row of <kernel>/results.csv, or None if it is missing or empty."""
    path = Path(kernel) / CSV_NAME
    if not path.exists():
        return None
    rows: list[Row] = []
    with path.open(newline="") as f:
        # device names hold commas, so let the csv module handle the quoting
        for record in csv.DictReader(f):
            values: dict[str, Any] = {
                field.name: record[field.name]
                if field.type is str
                else float(record[field.name])
                for field in fields(Row)
            }
            rows.append(Row(**values))
    return rows or None


def no_data(ax: Axes, what: str) -> None:
    """Say so in the middle of an empty panel."""
    ax.text(0.5, 0.5, f"no {what} data", transform=ax.transAxes, ha="center", va="center")


def machine_peaks(points: list[Row]) -> tuple[float, float] | None:
    """The GPU's peak GB/s and peak GFLOP/s from the CSV, or None without them."""
    peak_gbs = max(r.peak_gbs for r in points)
    peak_gflops = max(r.peak_gflops for r in points)
    return (peak_gbs, peak_gflops) if peak_gbs > 0 and peak_gflops > 0 else None


def roof(peak_gbs: float, peak_gflops: float, ai: np.ndarray) -> np.ndarray:
    """The machine's ceiling at each arithmetic intensity: min(peak GFLOP/s, peak GB/s x AI)."""
    return np.minimum(peak_gflops, peak_gbs * ai)


def draw_ceiling(ax: Axes, points: list[Row], xlim: tuple[float, float]) -> list[Line2D]:
    """The machine's ceiling across xlim. Returns the extra legend handles: the
    ceiling and, as text only, its ridge point (which this x window never
    reaches, so there is no marker to draw)."""
    peaks = machine_peaks(points)
    if peaks is None:
        return []  # no driver peaks in the CSV: points only
    peak_gbs, peak_gflops = peaks

    ridge_ai = peak_gflops / peak_gbs  # FLOP/B where the ceiling changes slope
    ai = np.geomspace(xlim[0], xlim[1], 400)
    ax.plot(
        ai,
        roof(peak_gbs, peak_gflops, ai),
        linestyle="--",
        linewidth=1.5,
        color=NEUTRAL,
    )
    return [
        Line2D(
            [],
            [],
            linestyle="--",
            linewidth=1.5,
            color=NEUTRAL,
            label=f"ceiling: {peak_gbs:.0f} GB/s, {peak_gflops:.0f} GFLOP/s",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="x",
            markersize=8,
            color="black",
            label=f"ridge: AI {ridge_ai:.1f} FLOP/B",
        ),
    ]


def draw_roofline(
    ax: Axes, series: dict[str, list[Row]], extent: tuple[float, float] | None
) -> list[Line2D]:
    """Every N > 1 on add: x is the minimum ops per estimated byte moved,
    colour means N, marker shape means kernel.

    Returns the extra legend handles (ceiling, ridge)."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/B, estimated traffic)")
    ax.set_ylabel("useful GFLOP/s")

    points: list[Row] = []
    if extent is not None:
        norm = LogNorm(vmin=extent[0], vmax=extent[1])
        for kernel, rows in series.items():
            at = [
                r
                for r in rows
                if r.n > 1 and r.est_bytes > 0 and r.flops > 0 and r.gflops > 0
            ]
            if not at:
                continue
            ax.scatter(
                [r.flops / r.est_bytes for r in at],
                [r.gflops for r in at],
                c=[r.n for r in at],
                norm=norm,
                cmap=CMAP,
                marker=MARKERS[kernel],
                s=30,
                linewidths=0.5,
                edgecolors="0.2",
                zorder=3,
            )
            points.extend(at)
    if not points:
        no_data(ax, OP)
        return []

    # A fixed, sensible window: every kernel's intensity sits at one of two
    # values (0.0625 for the ~16N movers, 0.125 for CUB's one pass), so two
    # ticks name every cluster, and the minor labels matplotlib would add
    # on a log axis this narrow stay off.
    xlim = (0.04, 0.2)
    ax.set_xlim(*xlim)
    ax.set_xticks([0.0625, 0.125], ["0.0625", "0.125"])
    ax.xaxis.set_minor_formatter(NullFormatter())

    # The y axis covers the points and the ceiling over the visible x range:
    # across it the roof climbs from peak GB/s x 0.04 to peak GB/s x 0.2, so
    # it sits just above the data there even though the compute peak is far
    # away. Plain numbers on the majors, nothing on the minors.
    ys = [r.gflops for r in points]
    peaks = machine_peaks(points)
    if peaks is not None:
        ys += [float(v) for v in roof(*peaks, np.array(xlim)) if v > 0]
    ax.set_ylim(min(ys) / 2, max(ys) * 2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    return draw_ceiling(ax, points, xlim)


def draw_vs_n(
    ax: Axes,
    series: dict[str, list[Row]],
    *,
    y: Callable[[Row], float],
    ylabel: str,
    yerr: Callable[[Row], tuple[float, float]] | None = None,
    log_y: bool = True,
    ylim: tuple[float, float] | None = None,
    reference: float | None = None,
) -> None:
    """One quantity against N > 1, one line per kernel, marker shape and colour
    per kernel.

    `y` picks the plotted value. `yerr` adds asymmetric error bars around it
    (down to the value it returns first, up by the second); without it there
    are no bars. `reference` draws a dashed horizontal line (CUB's ratio of 1
    on vs-cub). A linear y axis with a fixed `ylim` is how the bandwidth
    figures hold their 0-100 % scale."""
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("N (elements)")
    ax.set_ylabel(ylabel)
    if reference is not None:
        ax.axhline(reference, color=NEUTRAL, linestyle="--", linewidth=1.5)

    drawn = False
    for kernel, rows in series.items():
        points = [r for r in rows if r.n > 1 and y(r) > 0]
        if not points:
            continue
        points.sort(key=lambda r: r.n)
        ys = [y(r) for r in points]
        bars: np.ndarray | None = None
        if yerr is not None:
            errs = [yerr(r) for r in points]  # one per point, computed once
            bars = np.array([[down for down, _ in errs], [up for _, up in errs]])
        ax.errorbar(
            [r.n for r in points],
            ys,
            yerr=bars,
            fmt=f"{MARKERS[kernel]}-",
            markersize=4,
            linewidth=1.2,
            capsize=2,
            color=COLORS[kernel],
        )
        drawn = True

    if not drawn:
        no_data(ax, OP)


def runtime_limits(series: dict[str, list[Row]]) -> tuple[float, float] | None:
    """One log y-range for every per-kernel runtime figure: all kernels."""
    values: list[float] = []
    for rows in series.values():
        for r in rows:
            if r.n > 1 and r.median_ms > 0 and r.min_ms > 0:
                values += [r.min_ms, r.median_ms, r.median_ms + r.stddev_ms]
    if not values:
        return None
    lo, hi = min(values), max(values)
    return (lo / 2, hi * 2)  # headroom, so error bars stay off the edges


def n_extent(series: dict[str, list[Row]]) -> tuple[float, float] | None:
    """The smallest and largest N > 1 in series, or None if there is none."""
    ns = [r.n for rows in series.values() for r in rows if r.n > 1]
    if not ns:
        return None
    lo, hi = min(ns), max(ns)
    return (lo, hi) if lo < hi else (lo / 2, hi * 2)


def add_colorbar(fig: Figure, extent: tuple[float, float]) -> None:
    """N colour scale beside the panel, ticked at its smallest, middle and largest N."""
    bar = fig.colorbar(
        ScalarMappable(norm=LogNorm(vmin=extent[0], vmax=extent[1]), cmap=CMAP),
        ax=list(fig.axes),
        label="N (elements)",
    )
    ticks = [extent[0], float(np.sqrt(extent[0] * extent[1])), extent[1]]
    bar.minorticks_off()  # its log minor labels would pile onto ours
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{tick:,.0f}" for tick in ticks])
    bar.ax.tick_params(labelsize=8)


def add_runtime_note(fig: Figure) -> None:
    """Say what the error bars are, under the panel of a runtime figure."""
    fig.supxlabel("error bars: fastest run to +1σ", fontsize=8, color=NEUTRAL)


def add_legend(
    fig: Figure, series: dict[str, list[Row]], extra: list[Line2D], *, neutral: bool = False
) -> None:
    """One legend beside the panel: inside an axes it would cover the data.

    `neutral` greys the kernel markers, for the roofline where colour means N."""
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=MARKERS[kernel],
            markersize=7,
            color=NEUTRAL if neutral else COLORS[kernel],
            label=kernel,
        )
        for kernel in series
    ]
    if not handles and not extra:
        return
    fig.legend(
        handles=handles + extra,
        # centre height, not top: a legend at the top would run into the
        # figure title
        loc="outside right center",
        fontsize=9,
        title=None
        if not handles
        else ("kernel" if len(series) == 1 else "kernels"),
    )


def device_names(series: dict[str, list[Row]]) -> str:
    """The GPU(s) the CSV rows were measured on."""
    names = sorted({r.device for rows in series.values() for r in rows})
    return ", ".join(names) if names else "unknown device"


def save(fig: Figure, path: Path) -> None:
    """Write one figure and close it."""
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def single_panel(figsize: tuple[float, float] = (8.5, 5.5)) -> tuple[Figure, Axes]:
    """One empty, gridded panel with the constrained layout every figure uses."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.grid(True, which="both", alpha=0.3)
    return fig, ax


def runtime_error(r: Row) -> tuple[float, float]:
    """The asymmetric bar around a median: down to the fastest run, up by 1σ."""
    return (max(r.median_ms - r.min_ms, 0.0), r.stddev_ms)


def runtime_figure(
    series: dict[str, list[Row]], title: str, ylim: tuple[float, float] | None = None
) -> Figure:
    """A median-time panel; `title` prefixes the figure title, e.g. "add" or
    "register-tiling, add". `ylim` shares one time scale across kernels."""
    fig, ax = single_panel()
    fig.suptitle(f"{title}: median time on {device_names(series)}")
    draw_vs_n(
        ax,
        series,
        y=lambda r: r.median_ms,
        ylabel="median time (ms)",
        yerr=runtime_error,
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    add_runtime_note(fig)
    return fig


def bandwidth_figure(series: dict[str, list[Row]], title: str) -> Figure:
    """A useful-bandwidth panel on a fixed 0-100 % axis; `title` as above."""
    fig, ax = single_panel()
    fig.suptitle(f"{title}: useful bandwidth on {device_names(series)}")
    draw_vs_n(
        ax,
        series,
        y=lambda r: r.pct_peak_bw,
        ylabel="useful bandwidth (% of peak DRAM)",
        log_y=False,
        ylim=(0, 100),
    )
    return fig


def write_shared_figures(
    series: dict[str, list[Row]], extent: tuple[float, float] | None
) -> None:
    """graphs/: the five figures that put every kernel on one panel."""
    out = Path(GRAPH_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig = runtime_figure(series, "add")
    add_legend(fig, series, [])
    save(fig, out / "runtime.png")

    fig = bandwidth_figure(series, "add")
    add_legend(fig, series, [])
    save(fig, out / "bandwidth.png")

    fig, ax = single_panel()
    fig.suptitle(f"add: time per element on {device_names(series)}")
    draw_vs_n(
        ax,
        series,
        y=lambda r: r.median_ms * 1e6 / r.n,
        ylabel="time per element (ns)",
    )
    add_legend(fig, series, [])
    save(fig, out / "per-element.png")

    # CUB itself is only the dashed ratio-1 line, not a plotted series, so it
    # leaves the legend's markers to the kernels that are plotted.
    cub = {
        r.n: r.median_ms
        for r in series.get(REFERENCE, [])
        if r.n > 1 and r.median_ms > 0
    }
    others = {
        kernel: [r for r in rows if r.n > 1 and r.median_ms > 0 and r.n in cub]
        for kernel, rows in series.items()
        if kernel != REFERENCE
    }
    fig, ax = single_panel()
    fig.suptitle(f"add: time relative to CUB on {device_names(series)}")
    draw_vs_n(
        ax,
        others,
        y=lambda r: r.median_ms / cub[r.n],
        ylabel="median time ÷ CUB's median time",
        reference=1.0,
    )
    cub_line = Line2D(
        [], [], linestyle="--", linewidth=1.5, color=NEUTRAL, label="CUB (ratio 1)"
    )
    add_legend(fig, others, [cub_line])
    save(fig, out / "vs-cub.png")

    fig, ax = single_panel()
    fig.suptitle(f"add: roofline on {device_names(series)}")
    extras = draw_roofline(ax, series, extent)
    add_legend(fig, series, extras, neutral=True)
    if extent is not None:
        add_colorbar(fig, extent)
    save(fig, out / "roofline.png")


def write_kernel_figures(
    kernel: str,
    rows: list[Row],
    ylim: tuple[float, float] | None,
) -> None:
    """<kernel>/: runtime and bandwidth for that kernel alone, next to the
    implementation. The runtime figures share one y-range across kernels."""
    series = {kernel: rows}
    fig = runtime_figure(series, f"{kernel}, add", ylim)
    save(fig, Path(kernel) / "runtime.png")
    fig = bandwidth_figure(series, f"{kernel}, add")
    save(fig, Path(kernel) / "bandwidth.png")


def main() -> int:
    """Graph every CSV that exists. 0 if anything was drawn, 1 if none was."""
    series: dict[str, list[Row]] = {}
    for kernel in ALL_KERNELS:
        rows = load_results(kernel)
        if rows is None:
            print(f"skipping {kernel}: no rows in {kernel}/{CSV_NAME}")
            continue
        adds = [row for row in rows if row.op == OP]
        if not adds:
            print(f"skipping {kernel}: no {OP} rows in {kernel}/{CSV_NAME}")
            continue
        series[kernel] = adds

    if not series:
        print(f"nothing to graph: no {OP} rows in any {CSV_NAME} file")
        return 1

    # One N colour scale for the roofline, and one time scale for the
    # per-kernel runtime figures, so figures of different kernels compare.
    extent = n_extent(series)
    ylim = runtime_limits(series)
    write_shared_figures(series, extent)
    for kernel, rows in series.items():
        write_kernel_figures(kernel, rows, ylim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

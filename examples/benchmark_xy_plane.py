#!/usr/bin/env python3
"""Fixed-`xy`-plane resampling benchmark.

This script resamples the reference 3D datasets onto one fixed Cartesian
midplane across a resolution ramp, compares octree output with SciPy nearest,
and writes plots and timing reports under `artifacts/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import numpy as np
from batread.dataset import Dataset
from scipy.interpolate import NearestNDInterpolator

from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS

if __package__ in {None, ""}:
    from benchmark_helpers import _configure_progress_logging
    from benchmark_helpers import _configure_builder_logging
    from benchmark_helpers import _build_octree
    from benchmark_helpers import _ProgressReporter
    from benchmark_helpers import _resolution_ramp
    from benchmark_helpers import _time_call
    from benchmark_helpers import DatasetCase
    from benchmark_helpers import resolve_data_file
else:
    from .benchmark_helpers import _configure_progress_logging
    from .benchmark_helpers import _configure_builder_logging
    from .benchmark_helpers import _build_octree
    from .benchmark_helpers import _ProgressReporter
    from .benchmark_helpers import _resolution_ramp
    from .benchmark_helpers import _time_call
    from .benchmark_helpers import DatasetCase
    from .benchmark_helpers import resolve_data_file


def _xy_plane_image(
    *,
    n_plane: int,
    z_plane: float,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one regular `xy` query plane."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if not (float(zmin) <= float(z_plane) <= float(zmax)):
        raise ValueError(f"z_plane={z_plane} lies outside dataset z-bounds [{zmin}, {zmax}].")

    x = np.linspace(xmin, xmax, int(n_plane), dtype=float)
    y = np.linspace(ymin, ymax, int(n_plane), dtype=float)
    xg, yg = np.meshgrid(x, y, indexing="xy")
    zg = np.full_like(xg, float(z_plane), dtype=float)
    query = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))
    return xg, yg, query


def _octree_xy_plane_image(
    interp: OctreeInterpolator,
    *,
    query: np.ndarray,
    n_plane: int,
) -> np.ndarray:
    """Evaluate one octree interpolator on one prepared `xy` plane query."""
    return np.asarray(
        interp(query, query_coord="xyz", log_outside_domain=False),
        dtype=float,
    ).reshape(int(n_plane), int(n_plane))


def _nearest_xy_plane_image(
    nearest_interp: NearestNDInterpolator,
    *,
    query: np.ndarray,
    n_plane: int,
) -> np.ndarray:
    """Evaluate one SciPy nearest interpolator on one prepared `xy` plane query."""
    return np.asarray(nearest_interp(query), dtype=float).reshape(int(n_plane), int(n_plane))


def _array_stats(a: np.ndarray) -> tuple[int, int]:
    """Return `(nan_count, zero_count_exact)`."""
    finite = np.isfinite(a)
    nan_count = int(np.size(a) - np.count_nonzero(finite))
    zero_count = int(np.count_nonzero(finite & (a == 0.0)))
    return nan_count, zero_count


def _finite_min_max(a: np.ndarray) -> tuple[float, float]:
    """Return finite min/max, or `(nan, nan)` if none exist."""
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.nan, np.nan
    vals = np.asarray(a[finite], dtype=float)
    return float(np.min(vals)), float(np.max(vals))


def _pairwise_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    """Return overlap and error metrics between two plane images."""
    a_finite = np.isfinite(a)
    b_finite = np.isfinite(b)
    overlap = a_finite & b_finite
    out: dict[str, float | int] = {
        "a_finite": int(np.count_nonzero(a_finite)),
        "b_finite": int(np.count_nonzero(b_finite)),
        "finite_overlap": int(np.count_nonzero(overlap)),
        "positive_overlap": 0,
        "abs_mae": float("nan"),
        "abs_rmse": float("nan"),
        "log10_mae": float("nan"),
        "log10_rmse": float("nan"),
    }
    if not np.any(overlap):
        return out

    diff = np.asarray(a[overlap] - b[overlap], dtype=float)
    out["abs_mae"] = float(np.mean(np.abs(diff)))
    out["abs_rmse"] = float(np.sqrt(np.mean(diff * diff)))

    positive = overlap & (a > 0.0) & (b > 0.0)
    out["positive_overlap"] = int(np.count_nonzero(positive))
    if not np.any(positive):
        return out

    log_diff = np.log10(a[positive]) - np.log10(b[positive])
    out["log10_mae"] = float(np.mean(np.abs(log_diff)))
    out["log10_rmse"] = float(np.sqrt(np.mean(log_diff * log_diff)))
    return out


def _fastest_label(*, octree_s: float, nearest_s: float) -> str:
    """Return the faster steady-state method label for reports."""
    if float(octree_s) <= float(nearest_s):
        return "octree"
    return "nearest"


def _has_positive_finite(*arrays: np.ndarray) -> bool:
    """Return whether any provided array contains a positive finite value."""
    for arr in arrays:
        vals = np.asarray(arr, dtype=float)
        if np.any(np.isfinite(vals) & (vals > 0.0)):
            return True
    return False


def _image_extent(xg: np.ndarray, yg: np.ndarray) -> tuple[float, float, float, float]:
    """Return image extent from one regular `xy` plane grid."""
    return (
        float(np.min(xg)),
        float(np.max(xg)),
        float(np.min(yg)),
        float(np.max(yg)),
    )


def _save_xy_plane_figure(
    out_path: Path,
    *,
    dataset_label: str,
    variable: str,
    n_plane: int,
    z_plane: float,
    extent: tuple[float, float, float, float],
    img: np.ndarray,
    time_s: float,
) -> None:
    """Save one resampled `xy` plane figure."""
    finite = np.isfinite(img)
    pos = finite & (img > 0.0)
    stats = _array_stats(img)
    vmin, vmax = _finite_min_max(img)

    if np.any(pos):
        pos_vals = np.asarray(img[pos], dtype=float)
        lo = float(np.min(pos_vals))
        hi = float(np.max(pos_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = 1.0
            hi = 10.0
        norm = LogNorm(vmin=lo, vmax=hi)
        cbar_label = variable
        img_disp = np.where(pos, img, np.nan)
    else:
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = 0.0
            vmax = 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar_label = variable
        img_disp = img

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="#dddddd")

    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    im = ax.imshow(
        img_disp,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )
    ax.set_title(f"{dataset_label} | {n_plane}x{n_plane} | z={z_plane:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.text(
        0.02,
        0.98,
        f"time={time_s:.4f}s\nnan={stats[0]}\nzero={stats[1]}\nmin={vmin:.6e}\nmax={vmax:.6e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        fontsize=9,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_timing_table(
    rows: list[dict[str, float | int]],
    out_path: Path,
    *,
    octree_tree_s: float,
    octree_interp_s: float,
    nearest_build_s: float,
    cold_resolution: int,
    octree_cold_s: float,
    nearest_cold_s: float,
) -> None:
    """Write markdown timing table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "## Build",
        "",
        "| method | seconds |",
        "|---|---:|",
        f"| octree tree | {float(octree_tree_s):.6f} |",
        f"| octree interpolator | {float(octree_interp_s):.6f} |",
        f"| octree total | {float(octree_tree_s + octree_interp_s):.6f} |",
        f"| scipy nearest | {float(nearest_build_s):.6f} |",
        "",
        "## Cold Start",
        "",
        f"- first plane query after build: `{int(cold_resolution)}x{int(cold_resolution)}`",
        "",
        "| method | seconds |",
        "|---|---:|",
        f"| octree | {float(octree_cold_s):.6f} |",
        f"| scipy nearest | {float(nearest_cold_s):.6f} |",
        "",
        "## Steady-State Runtime",
        "",
        "| resolution | pixels | octree_s | nearest_s | fastest | octree_nan | nearest_nan |",
        "|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{int(row['resolution'])}x{int(row['resolution'])} | "
            f"{int(row['pixels'])} | "
            f"{float(row['octree_plane_s']):.6f} | "
            f"{float(row['nearest_plane_s']):.6f} | "
            f"{_fastest_label(octree_s=float(row['octree_plane_s']), nearest_s=float(row['nearest_plane_s']))} | "
            f"{int(row['octree_nan'])} | "
            f"{int(row['nearest_nan'])} |"
        )
    lines.extend(
        [
            "",
            "## Octree Vs Nearest",
            "",
            "| resolution | finite_overlap | positive_overlap | abs_mae | abs_rmse | log10_mae | log10_rmse |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{int(row['resolution'])}x{int(row['resolution'])} | "
            f"{int(row['finite_overlap'])} | "
            f"{int(row['positive_overlap'])} | "
            f"{float(row['abs_mae']):.6e} | "
            f"{float(row['abs_rmse']):.6e} | "
            f"{float(row['log10_mae']):.6e} | "
            f"{float(row['log10_rmse']):.6e} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tail_power_fit(resolution: np.ndarray, runtime: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Fit one tail power law `runtime ~ N**alpha` on the last few samples."""
    tail_count = min(4, int(resolution.size))
    if tail_count < 2:
        return None

    tail_resolution = resolution[-tail_count:]
    tail_runtime = runtime[-tail_count:]
    log_tail_resolution = np.log(tail_resolution)
    log_tail_runtime = np.log(tail_runtime)
    centered_resolution = log_tail_resolution - np.mean(log_tail_resolution)
    alpha = float(
        np.dot(centered_resolution, log_tail_runtime - np.mean(log_tail_runtime))
        / np.dot(centered_resolution, centered_resolution)
    )
    log_scale = float(np.mean(log_tail_runtime - alpha * log_tail_resolution))
    return np.exp(log_scale) * np.power(resolution, alpha), alpha


def _save_runtime_plot(
    rows: list[dict[str, float | int]],
    out_path: Path,
    *,
    title: str,
    cold_resolution: int,
    octree_cold_s: float,
    nearest_cold_s: float,
) -> None:
    """Save one figure with cold-start and steady-state plane runtimes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolution = np.asarray([int(row["resolution"]) for row in rows], dtype=float)
    octree_t = np.asarray([float(row["octree_plane_s"]) for row in rows], dtype=float)
    nearest_t = np.asarray([float(row["nearest_plane_s"]) for row in rows], dtype=float)

    fig, (ax_cold, ax_steady) = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.5),
        gridspec_kw={"width_ratios": [0.9, 2.4]},
        constrained_layout=True,
    )
    cold_x = np.arange(2, dtype=float)
    ax_cold.bar(cold_x, [float(octree_cold_s), float(nearest_cold_s)], color=["C0", "C2"])
    ax_cold.set_xticks(cold_x)
    ax_cold.set_xticklabels(["octree", "scipy nearest"], rotation=15, ha="right")
    ax_cold.set_yscale("log")
    ax_cold.set_ylabel("Runtime [s]")
    ax_cold.set_title(f"Cold start ({int(cold_resolution)}x{int(cold_resolution)})")
    ax_cold.grid(True, axis="y", which="both", alpha=0.25)

    ax_steady.plot(resolution, octree_t, "o-", color="C0", label="octree")
    ax_steady.plot(resolution, nearest_t, "o-", color="C2", label="scipy nearest")
    for label, runtime, color in (
        ("octree", octree_t, "C0"),
        ("scipy nearest", nearest_t, "C2"),
    ):
        tail_fit = _tail_power_fit(resolution, runtime)
        if tail_fit is None:
            continue
        fit, alpha = tail_fit
        ax_steady.plot(
            resolution,
            fit,
            "--",
            color=color,
            linewidth=1.6,
            alpha=0.8,
            label=f"{label} tail fit ~ N^{alpha:.2f}",
        )
    ax_steady.set_xscale("log")
    ax_steady.set_yscale("log")
    _set_resolution_ticks(ax_steady, resolution)
    ax_steady.set_xlabel("Image resolution")
    ax_steady.set_ylabel("Runtime [s]")
    ax_steady.set_title("Steady state")
    ax_steady.grid(True, which="both", alpha=0.25)
    ax_steady.legend()
    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_comparison_plot(
    rows: list[dict[str, float | int]],
    out_path: Path,
    *,
    title: str,
) -> None:
    """Save overlap and error curves between octree and SciPy nearest."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolution = np.asarray([int(row["resolution"]) for row in rows], dtype=float)
    abs_mae = np.asarray([float(row["abs_mae"]) for row in rows], dtype=float)
    abs_rmse = np.asarray([float(row["abs_rmse"]) for row in rows], dtype=float)
    log10_mae = np.asarray([float(row["log10_mae"]) for row in rows], dtype=float)
    log10_rmse = np.asarray([float(row["log10_rmse"]) for row in rows], dtype=float)
    finite_overlap = np.asarray([int(row["finite_overlap"]) for row in rows], dtype=float)
    positive_overlap = np.asarray([int(row["positive_overlap"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)

    axes[0].plot(resolution, abs_mae, "o-", label="MAE")
    axes[0].plot(resolution, abs_rmse, "o--", label="RMSE")
    axes[0].set_xscale("log")
    if _has_positive_finite(abs_mae, abs_rmse):
        axes[0].set_yscale("log")
    _set_resolution_ticks(axes[0], resolution)
    axes[0].set_xlabel("Image resolution")
    axes[0].set_ylabel("Absolute error")
    axes[0].set_title("Octree vs nearest")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].plot(resolution, log10_mae, "o-", label="MAE")
    axes[1].plot(resolution, log10_rmse, "o--", label="RMSE")
    axes[1].set_xscale("log")
    if _has_positive_finite(log10_mae, log10_rmse):
        axes[1].set_yscale("log")
    _set_resolution_ticks(axes[1], resolution)
    axes[1].set_xlabel("Image resolution")
    axes[1].set_ylabel("log10 error")
    axes[1].set_title("Positive overlap only")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    axes[2].plot(resolution, finite_overlap, "o-", label="finite overlap")
    axes[2].plot(resolution, positive_overlap, "o--", label="positive overlap")
    axes[2].set_xscale("log")
    if _has_positive_finite(finite_overlap, positive_overlap):
        axes[2].set_yscale("log")
    _set_resolution_ticks(axes[2], resolution)
    axes[2].set_xlabel("Image resolution")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Overlap counts")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend()

    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _set_resolution_ticks(ax: plt.Axes, resolution: np.ndarray) -> None:
    """Label a log-scaled axis with explicit square-image resolutions."""
    ax.set_xticks(resolution)
    ax.set_xticklabels([f"{int(n)}x{int(n)}" for n in resolution], rotation=35, ha="right")


def _artifact_path(out_root: Path, *, case_label: str, name: str) -> Path:
    """Return one flat artifact path under the benchmark output root."""
    return out_root / f"benchmark_xy_plane_{case_label}_{name}"


def _run_case(
    *,
    case: DatasetCase,
    repo_root: Path,
    out_root: Path,
    progress: _ProgressReporter,
    resolutions: list[int],
    max_seconds_per_image: float,
    z_plane: float,
    variable: str,
) -> None:
    """Run one dataset case through the full `xy`-plane benchmark."""
    progress.note(f"[{case.label}] file={case.file_name}")
    progress.note(f"[{case.label}] artifact_prefix=benchmark_xy_plane_{case.label}_*")
    progress.start(f"[{case.label}] resolve data file")
    data_path, resolve_s = _time_call(resolve_data_file, repo_root, case.file_name)
    progress.complete(f"[{case.label}] resolve data file", resolve_s, detail=f"-> {data_path}")
    progress.start(f"[{case.label}] read dataset")
    ds, read_s = _time_call(Dataset.from_file, str(data_path))
    progress.complete(f"[{case.label}] read dataset", read_s)
    progress.start(f"[{case.label}] prepare octree")
    tree, tree_s = _time_call(_build_octree, ds)
    progress.complete(
        f"[{case.label}] prepare octree",
        tree_s,
        detail=f"coord={tree.tree_coord}",
    )
    progress.start(f"[{case.label}] build interpolator")
    interp, interp_s = _time_call(OctreeInterpolator, tree, np.asarray(ds[variable], dtype=float))
    progress.complete(f"[{case.label}] build interpolator", interp_s)
    xyz = np.column_stack(tuple(np.asarray(ds[name], dtype=float) for name in XYZ_VARS))
    values = np.asarray(ds[variable], dtype=float)
    progress.start(f"[{case.label}] build scipy NearestND")
    nearest_interp, nearest_build_s = _time_call(NearestNDInterpolator, xyz, values)
    progress.complete(f"[{case.label}] build scipy NearestND", nearest_build_s)

    dmin = np.min(xyz, axis=0)
    dmax = np.max(xyz, axis=0)
    bounds = (
        float(dmin[0]),
        float(dmax[0]),
        float(dmin[1]),
        float(dmax[1]),
        float(dmin[2]),
        float(dmax[2]),
    )

    warm_n = int(resolutions[0])
    progress.start(f"[{case.label}] cold start {warm_n}x{warm_n}")
    _, _, warm_query = _xy_plane_image(n_plane=warm_n, z_plane=z_plane, bounds=bounds)
    _, octree_warm_s = _time_call(_octree_xy_plane_image, interp, query=warm_query, n_plane=warm_n)
    _, nearest_warm_s = _time_call(
        _nearest_xy_plane_image,
        nearest_interp,
        query=warm_query,
        n_plane=warm_n,
    )
    progress.complete(f"[{case.label}] cold start {warm_n}x{warm_n}", float(octree_warm_s + nearest_warm_s))

    rows: list[dict[str, float | int]] = []
    for n in resolutions:
        progress.start(f"[{case.label}] run {n}x{n}")
        t_step = time.perf_counter()
        xg, yg, query = _xy_plane_image(n_plane=int(n), z_plane=z_plane, bounds=bounds)
        t0 = time.perf_counter()
        octree_img = _octree_xy_plane_image(interp, query=query, n_plane=int(n))
        octree_plane_s = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        nearest_img = _nearest_xy_plane_image(nearest_interp, query=query, n_plane=int(n))
        nearest_plane_s = float(time.perf_counter() - t0)
        octree_nan, _octree_zero = _array_stats(octree_img)
        nearest_nan, _nearest_zero = _array_stats(nearest_img)
        metrics = _pairwise_metrics(octree_img, nearest_img)
        pixels = int(n * n)

        row = {
            "resolution": int(n),
            "pixels": pixels,
            "octree_plane_s": float(octree_plane_s),
            "nearest_plane_s": float(nearest_plane_s),
            "octree_nan": int(octree_nan),
            "nearest_nan": int(nearest_nan),
            "finite_overlap": int(metrics["finite_overlap"]),
            "positive_overlap": int(metrics["positive_overlap"]),
            "abs_mae": float(metrics["abs_mae"]),
            "abs_rmse": float(metrics["abs_rmse"]),
            "log10_mae": float(metrics["log10_mae"]),
            "log10_rmse": float(metrics["log10_rmse"]),
        }
        rows.append(row)

        _save_xy_plane_figure(
            _artifact_path(out_root, case_label=case.label, name=f"xy_plane_{n}x{n}.png"),
            dataset_label=f"{case.label}:{case.file_name}",
            variable=variable,
            n_plane=int(n),
            z_plane=z_plane,
            extent=_image_extent(xg, yg),
            img=octree_img,
            time_s=octree_plane_s,
        )
        _write_timing_table(
            rows,
            _artifact_path(out_root, case_label=case.label, name="timing_report.md"),
            octree_tree_s=float(tree_s),
            octree_interp_s=float(interp_s),
            nearest_build_s=float(nearest_build_s),
            cold_resolution=warm_n,
            octree_cold_s=float(octree_warm_s),
            nearest_cold_s=float(nearest_warm_s),
        )
        _save_runtime_plot(
            rows,
            _artifact_path(out_root, case_label=case.label, name="runtime_vs_pixels.png"),
            title=f"{case.label}: xy plane runtime",
            cold_resolution=warm_n,
            octree_cold_s=float(octree_warm_s),
            nearest_cold_s=float(nearest_warm_s),
        )
        _save_comparison_plot(
            rows,
            _artifact_path(out_root, case_label=case.label, name="octree_vs_nearest.png"),
            title=f"{case.label}: octree vs scipy nearest",
        )
        progress.complete(
            f"[{case.label}] run {n}x{n}",
            float(time.perf_counter() - t_step),
        )
        if max(octree_plane_s, nearest_plane_s) > max_seconds_per_image:
            progress.note(f"[{case.label}] stop at {n}x{n}: reached {max_seconds_per_image:.2f}s limit")
            break
    progress.note(f"[{case.label}] done -> benchmark_xy_plane_{case.label}_*")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample the reference 3D files onto one fixed xy plane.")
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=2,
        help="Smallest square plane resolution (default: 2).",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=1024,
        help="Largest square plane resolution (default: 1024).",
    )
    parser.add_argument(
        "--max-seconds-per-image",
        type=float,
        default=0.5,
        help="Stop increasing resolution for one dataset once one plane image exceeds this many seconds.",
    )
    parser.add_argument(
        "--z-plane",
        type=float,
        default=0.0,
        help="Cartesian z coordinate of the sampled xy plane (default: 0.0).",
    )
    parser.add_argument(
        "--variable",
        default="Rho [g/cm^3]",
        help="Dataset variable to resample.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Output directory for PNGs and tables.",
    )
    args = parser.parse_args()

    resolutions = _resolution_ramp(int(args.min_resolution), int(args.max_resolution))
    max_seconds_per_image = float(args.max_seconds_per_image)
    if max_seconds_per_image <= 0.0:
        raise ValueError("max_seconds_per_image must be positive.")

    repo_root = Path(__file__).resolve().parent.parent
    out_root = (repo_root / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    progress_log_path = out_root / "benchmark_xy_plane.log"
    progress_log_path.write_text("", encoding="utf-8")
    _configure_progress_logging(log_path=progress_log_path)
    _configure_builder_logging(log_path=progress_log_path)
    progress = _ProgressReporter(log_path=progress_log_path)

    cases = [
        DatasetCase("example", "3d__var_1_n00000000.plt"),
        DatasetCase("ih", "3d__var_4_n00005000.plt"),
        DatasetCase("sc", "3d__var_4_n00044000.plt"),
    ]

    progress.note(f"output_dir={out_root}")
    for case in cases:
        _run_case(
            case=case,
            repo_root=repo_root,
            out_root=out_root,
            progress=progress,
            resolutions=resolutions,
            max_seconds_per_image=max_seconds_per_image,
            z_plane=float(args.z_plane),
            variable=str(args.variable),
        )


if __name__ == "__main__":
    main()

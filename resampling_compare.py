#!/usr/bin/env python3
"""Compare grid-sum resampling vs ray integration across resolutions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tarfile
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import numpy as np
import pooch
from starwinds_readplt.dataset import Dataset

from batcamp import OctreeInterpolator
from batcamp import OctreeRayInterpolator


_G2211_URL = "https://zenodo.org/records/7110555/files/run-Sun-G2211.tar.gz"
_G2211_SHA256 = "c31a32aab08cc20d5b643bba734fd7220e6b369e691f55f88a3a08cc5b2a2136"


@dataclass(frozen=True)
class DatasetCase:
    label: str
    file_name: str


def _unique_match(paths: list[Path], *, name: str) -> Path:
    """Return one matched path by name, otherwise raise."""
    if not paths:
        raise FileNotFoundError(name)
    if len(paths) > 1:
        raise FileNotFoundError(f"Expected unique match for {name}, found {len(paths)}: {paths}")
    return paths[0]


def _find_in_example_data(root: Path, name: str) -> Path:
    """Find one file by basename under example_data."""
    return _unique_match(sorted(root.rglob(name)), name=name)


def _fetch_from_g2211_archive(name: str) -> Path:
    """Fetch one named file from the Zenodo G2211 archive."""
    archive_path = Path(
        pooch.retrieve(
            url=_G2211_URL,
            known_hash=_G2211_SHA256,
            progressbar=False,
        )
    )
    with tarfile.open(archive_path, "r:gz") as tar:
        member_names = sorted(m.name for m in tar.getmembers() if m.isfile() and Path(m.name).name == name)
    member = _unique_match([Path(m) for m in member_names], name=name).as_posix()
    extracted = pooch.retrieve(
        url=_G2211_URL,
        known_hash=_G2211_SHA256,
        progressbar=False,
        processor=pooch.Untar(members=[member]),
    )
    if isinstance(extracted, (list, tuple)):
        extracted = extracted[0]
    return Path(extracted)


def resolve_data_file(repo_root: Path, name: str) -> Path:
    """Resolve data file from example_data first, then pooch fallback."""
    try:
        return _find_in_example_data(repo_root / "example_data", name)
    except FileNotFoundError:
        return _fetch_from_g2211_archive(name)


def parse_resolutions(raw: str) -> list[int]:
    """Parse comma-separated resolution integers."""
    out: list[int] = []
    for token in raw.split(","):
        part = token.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError("All resolutions must be positive integers.")
        out.append(n)
    if not out:
        raise ValueError("No resolutions parsed.")
    return out


def _grid_sum_image(
    interp: OctreeInterpolator,
    *,
    n_plane: int,
    nx_sum: int,
    bounds: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Resample to XYZ grid and integrate along x; return image as (z, y)."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x = np.linspace(xmin, xmax, int(nx_sum), dtype=float)
    y = np.linspace(ymin, ymax, int(n_plane), dtype=float)
    z = np.linspace(zmin, zmax, int(n_plane), dtype=float)

    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
    query = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))
    vals = np.asarray(interp(query, query_coord="xyz", log_outside_domain=False), dtype=float).reshape(
        x.size,
        y.size,
        z.size,
    )

    finite = np.isfinite(vals)
    summed = np.trapezoid(np.where(finite, vals, 0.0), x=x, axis=0)
    any_finite = np.any(finite, axis=0)
    out = np.full_like(summed, np.nan, dtype=float)
    out[any_finite] = summed[any_finite]
    # Summation result is (y, z); transpose to canonical image layout (z, y).
    return out.T


def _ray_image(
    ray: OctreeRayInterpolator,
    *,
    origins: np.ndarray,
    direction: np.ndarray,
    t_end: float,
    n_plane: int,
    chunk_size: int,
) -> np.ndarray:
    """Integrate directly with OctreeRayInterpolator along +x rays as (z, y)."""
    values = np.asarray(
        ray.integrate_field_along_rays(
            origins,
            direction,
            0.0,
            t_end,
            chunk_size=int(chunk_size),
        ),
        dtype=float,
    ).reshape((int(n_plane), int(n_plane)))
    return values


def _ray_setup(
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build origins/direction/extent for one (z, y) ray image plane."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    y = np.linspace(ymin, ymax, int(n_plane), dtype=float)
    z = np.linspace(zmin, zmax, int(n_plane), dtype=float)
    yg, zg = np.meshgrid(y, z, indexing="xy")
    x_span = float(xmax - xmin)
    x0 = float(xmin - 1.0e-6 * max(1.0, x_span))
    origins = np.column_stack((np.full(yg.size, x0, dtype=float), yg.ravel(), zg.ravel()))
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    t_end = float((xmax - x0) * 0.999999)
    return origins, direction, t_end


def _ray_segment_counts(
    ray: OctreeRayInterpolator,
    *,
    origins: np.ndarray,
    direction: np.ndarray,
    t_end: float,
    n_plane: int,
    chunk_size: int,
) -> np.ndarray:
    """Return per-pixel segment counts from adaptive-midpoint ray offsets."""
    _mids, _weights, offsets = ray.adaptive_midpoint_rule(
        origins,
        direction,
        0.0,
        t_end,
        chunk_size=int(chunk_size),
    )
    counts = np.diff(np.asarray(offsets, dtype=np.int64))
    return counts.reshape((int(n_plane), int(n_plane)))


def _array_stats(a: np.ndarray) -> tuple[int, int]:
    """Return `(nan_count, zero_count_exact)`."""
    finite = np.isfinite(a)
    nan_count = int(np.size(a) - np.count_nonzero(finite))
    zero_count = int(np.count_nonzero(finite & (a == 0.0)))
    return nan_count, zero_count


def _equality_deviation(
    img0: np.ndarray,
    img1: np.ndarray,
) -> tuple[int, int, float, float, float, float]:
    """Return deviation metrics relative to the plot0==plot1 line."""
    finite = np.isfinite(img0) & np.isfinite(img1)
    finite_overlap = int(np.count_nonzero(finite))
    if finite_overlap == 0:
        return 0, 0, np.nan, np.nan, np.nan, np.nan

    x = img0[finite].reshape(-1).astype(float)
    y = img1[finite].reshape(-1).astype(float)
    diff = y - x
    abs_l1 = float(np.sum(np.abs(diff)))
    abs_rmse = float(np.sqrt(np.mean(diff * diff)))

    positive = (x > 0.0) & (y > 0.0)
    pos_overlap = int(np.count_nonzero(positive))
    if pos_overlap == 0:
        return finite_overlap, 0, abs_l1, abs_rmse, np.nan, np.nan

    log_diff = np.log10(y[positive]) - np.log10(x[positive])
    log_l1 = float(np.sum(np.abs(log_diff)))
    log_rmse = float(np.sqrt(np.mean(log_diff * log_diff)))
    return finite_overlap, pos_overlap, abs_l1, abs_rmse, log_l1, log_rmse


def _save_four_panel_figure(
    out_path: Path,
    *,
    dataset_label: str,
    n_plane: int,
    img0: np.ndarray,
    img1: np.ndarray,
    ray_segment_counts: np.ndarray,
    grid_segment_count: int,
    time0: float,
    time1: float,
    nx_sum: int,
    eq_abs_l1: float,
    eq_abs_rmse: float,
    eq_log_l1: float,
    eq_log_rmse: float,
    eq_pos_overlap: int,
) -> None:
    """Save 2x2 panels: images, comparison scatter, and segment-count histogram."""
    stats0 = _array_stats(img0)
    stats1 = _array_stats(img1)

    pos0 = np.isfinite(img0) & (img0 > 0.0)
    pos1 = np.isfinite(img1) & (img1 > 0.0)
    both_pos = pos0 & pos1
    pos_vals = np.concatenate((img0[pos0], img1[pos1])) if (np.any(pos0) or np.any(pos1)) else np.array(
        [],
        dtype=float,
    )

    if pos_vals.size > 0:
        vmin = float(np.min(pos_vals))
        vmax = float(np.max(pos_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0.0 or vmax <= vmin:
            vmin, vmax = 1.0, 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cbar_label = "value (log scale)"
    else:
        vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar_label = "value"

    img0_disp = np.where(pos0, img0, np.nan)
    img1_disp = np.where(pos1, img1, np.nan)

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="#dddddd")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    im0 = axes[0, 0].imshow(img0_disp, origin="lower", cmap=cmap, norm=norm, aspect="equal")
    axes[0, 0].set_title("Plot 0: 3D grid-sum")
    axes[0, 0].text(
        0.02,
        0.98,
        f"time={time0:.4f}s\nnan={stats0[0]}\nzero={stats0[1]}",
        transform=axes[0, 0].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        fontsize=9,
    )

    im1 = axes[0, 1].imshow(img1_disp, origin="lower", cmap=cmap, norm=norm, aspect="equal")
    axes[0, 1].set_title("Plot 1: Ray integration")
    axes[0, 1].text(
        0.02,
        0.98,
        f"time={time1:.4f}s\nnan={stats1[0]}\nzero={stats1[1]}",
        transform=axes[0, 1].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        fontsize=9,
    )
    cbar = fig.colorbar(im0, ax=[axes[0, 0], axes[0, 1]], fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label)

    axes[1, 0].set_title("Comparison: plot0 vs plot1")
    if np.any(both_pos):
        x = img0[both_pos].reshape(-1)
        y = img1[both_pos].reshape(-1)
        axes[1, 0].scatter(x, y, s=12, alpha=0.8)
        lo_data = float(min(np.min(x), np.min(y)))
        hi_data = float(max(np.max(x), np.max(y)))
        if not np.isfinite(lo_data) or not np.isfinite(hi_data) or lo_data <= 0.0 or hi_data <= lo_data:
            lo_data, hi_data = 1.0, 10.0
        pad = 1.12
        lo = lo_data / pad
        hi = hi_data * pad
        axes[1, 0].plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
        axes[1, 0].set_xlim(lo, hi)
        axes[1, 0].set_ylim(lo, hi)
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
    else:
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_xlim(1.0, 10.0)
        axes[1, 0].set_ylim(1.0, 10.0)
        axes[1, 0].text(0.5, 0.5, "no positive overlap", transform=axes[1, 0].transAxes, ha="center", va="center")
    axes[1, 0].set_xlabel("plot0 values")
    axes[1, 0].set_ylabel("plot1 values")
    axes[1, 0].grid(True, which="both", alpha=0.25)
    log_l1_text = "n/a" if not np.isfinite(eq_log_l1) else f"{eq_log_l1:.3e}"
    log_rmse_text = "n/a" if not np.isfinite(eq_log_rmse) else f"{eq_log_rmse:.3e}"
    axes[1, 0].text(
        0.02,
        0.98,
        "deviation from y=x\n"
        f"L1={eq_abs_l1:.3e}\n"
        f"RMSE={eq_abs_rmse:.3e}\n"
        f"log10 L1={log_l1_text}\n"
        f"log10 RMSE={log_rmse_text}\n"
        f"positive overlap={eq_pos_overlap}",
        transform=axes[1, 0].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        fontsize=9,
    )

    axes[1, 1].set_title("Ray Segment Count Histogram")
    seg = np.asarray(ray_segment_counts, dtype=np.int64).reshape(-1)
    if seg.size > 0:
        max_seg = int(np.max(seg))
        bins = np.arange(-0.5, max_seg + 1.5, 1.0)
        axes[1, 1].hist(seg, bins=bins, color="tab:blue", alpha=0.8, edgecolor="black", linewidth=0.5)
        avg_seg = float(np.mean(seg))
        axes[1, 1].axvline(
            avg_seg,
            color="tab:green",
            linestyle=":",
            linewidth=1.8,
            label=f"ray mean = {avg_seg:.2f}",
        )
        axes[1, 1].axvline(
            float(grid_segment_count),
            color="tab:red",
            linestyle="--",
            linewidth=1.6,
            label=f"grid const = {grid_segment_count}",
        )
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "no rays", transform=axes[1, 1].transAxes, ha="center", va="center")
    axes[1, 1].set_xlabel("segments per ray")
    axes[1, 1].set_ylabel("pixel count")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"{dataset_label} | plane={n_plane}x{n_plane} | nx_sum={nx_sum}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_timing_table(rows: list[dict[str, float | int]], out_path: Path) -> None:
    """Write markdown timing table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| resolution | pixels | grid_s | ray_s | ray/grid | grid_nan | grid_zero | ray_nan | ray_zero | finite_overlap | positive_overlap | eq_abs_l1 | eq_abs_rmse | eq_log10_l1 | eq_log10_rmse |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ratio = float(r["ray_s"]) / max(float(r["grid_s"]), 1.0e-15)
        eq_log10_l1 = float(r["eq_log10_l1"])
        eq_log10_rmse = float(r["eq_log10_rmse"])
        eq_log10_l1_text = f"{eq_log10_l1:.6e}" if np.isfinite(eq_log10_l1) else "nan"
        eq_log10_rmse_text = f"{eq_log10_rmse:.6e}" if np.isfinite(eq_log10_rmse) else "nan"
        lines.append(
            "| "
            f"{int(r['resolution'])}x{int(r['resolution'])} | "
            f"{int(r['pixels'])} | "
            f"{float(r['grid_s']):.6f} | "
            f"{float(r['ray_s']):.6f} | "
            f"{ratio:.3f} | "
            f"{int(r['grid_nan'])} | "
            f"{int(r['grid_zero'])} | "
            f"{int(r['ray_nan'])} | "
            f"{int(r['ray_zero'])} | "
            f"{int(r['finite_overlap'])} | "
            f"{int(r['positive_overlap'])} | "
            f"{float(r['eq_abs_l1']):.6e} | "
            f"{float(r['eq_abs_rmse']):.6e} | "
            f"{eq_log10_l1_text} | "
            f"{eq_log10_rmse_text} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_runtime_plot(rows: list[dict[str, float | int]], out_path: Path, *, title: str) -> None:
    """Save runtime vs pixel-count line plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray([int(r["pixels"]) for r in rows], dtype=float)
    grid_t = np.asarray([float(r["grid_s"]) for r in rows], dtype=float)
    ray_t = np.asarray([float(r["ray_s"]) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(pixels, grid_t, "o-", label="plot0: 3D grid-sum")
    ax.plot(pixels, ray_t, "o-", label="plot1: ray integration")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Pixel count (N x N)")
    ax.set_ylabel("Runtime [s]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 3D grid-sum vs ray integration resampling.")
    parser.add_argument(
        "--resolutions",
        default="2,4,8,16,32",
        help="Comma-separated plane resolutions (default: 2,4,8,16,32).",
    )
    parser.add_argument(
        "--nx-sum",
        type=int,
        default=256,
        help="Number of x-samples for 3D-grid summation baseline (default: 256).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Chunk size for ray integrator (default: 4096).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/resampling_compare",
        help="Output directory for PNGs and tables.",
    )
    args = parser.parse_args()

    resolutions = parse_resolutions(args.resolutions)
    repo_root = Path(__file__).resolve().parent
    out_root = (repo_root / args.output_dir).resolve()

    cases = [
        DatasetCase("example", "3d__var_1_n00000000.plt"),
        DatasetCase("sc", "3d__var_4_n00044000.plt"),
        DatasetCase("ih", "3d__var_4_n00005000.plt"),
    ]

    print(
        "dataset,resolution,pixels,grid_s,ray_s,grid_nan,grid_zero,ray_nan,ray_zero,"
        "finite_overlap,positive_overlap,eq_abs_l1,eq_abs_rmse,eq_log10_l1,eq_log10_rmse"
    )
    for case in cases:
        case_dir = out_root / case.label
        case_dir.mkdir(parents=True, exist_ok=True)

        data_path = resolve_data_file(repo_root, case.file_name)
        ds = Dataset.from_file(str(data_path))
        interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"])
        ray = OctreeRayInterpolator(interp)

        dmin, dmax = interp.tree.domain_bounds(coord="xyz")
        bounds = (
            float(dmin[0]),
            float(dmax[0]),
            float(dmin[1]),
            float(dmax[1]),
            float(dmin[2]),
            float(dmax[2]),
        )

        warm_n = int(resolutions[0])
        _ = _grid_sum_image(interp, n_plane=warm_n, nx_sum=int(args.nx_sum), bounds=bounds)
        warm_origins, warm_direction, warm_t_end = _ray_setup(n_plane=warm_n, bounds=bounds)
        _ = _ray_image(
            ray,
            origins=warm_origins,
            direction=warm_direction,
            t_end=warm_t_end,
            n_plane=warm_n,
            chunk_size=int(args.chunk_size),
        )

        rows: list[dict[str, float | int]] = []
        for n in resolutions:
            origins, direction, t_end = _ray_setup(n_plane=int(n), bounds=bounds)

            t0 = time.perf_counter()
            img0 = _grid_sum_image(interp, n_plane=int(n), nx_sum=int(args.nx_sum), bounds=bounds)
            grid_s = float(time.perf_counter() - t0)

            t0 = time.perf_counter()
            img1 = _ray_image(
                ray,
                origins=origins,
                direction=direction,
                t_end=t_end,
                n_plane=int(n),
                chunk_size=int(args.chunk_size),
            )
            ray_s = float(time.perf_counter() - t0)
            ray_seg_counts = _ray_segment_counts(
                ray,
                origins=origins,
                direction=direction,
                t_end=t_end,
                n_plane=int(n),
                chunk_size=int(args.chunk_size),
            )
            grid_seg_const = max(int(args.nx_sum) - 1, 1)

            grid_nan, grid_zero = _array_stats(img0)
            ray_nan, ray_zero = _array_stats(img1)
            finite_overlap, pos_overlap, eq_abs_l1, eq_abs_rmse, eq_log_l1, eq_log_rmse = _equality_deviation(
                img0,
                img1,
            )
            pixels = int(n * n)

            row = {
                "resolution": int(n),
                "pixels": pixels,
                "grid_s": grid_s,
                "ray_s": ray_s,
                "grid_nan": int(grid_nan),
                "grid_zero": int(grid_zero),
                "ray_nan": int(ray_nan),
                "ray_zero": int(ray_zero),
                "finite_overlap": finite_overlap,
                "positive_overlap": int(pos_overlap),
                "eq_abs_l1": float(eq_abs_l1),
                "eq_abs_rmse": float(eq_abs_rmse),
                "eq_log10_l1": float(eq_log_l1),
                "eq_log10_rmse": float(eq_log_rmse),
            }
            rows.append(row)

            figure_path = case_dir / f"resampling_compare_{n}x{n}.png"
            _save_four_panel_figure(
                figure_path,
                dataset_label=f"{case.label}:{case.file_name}",
                n_plane=int(n),
                img0=img0,
                img1=img1,
                ray_segment_counts=ray_seg_counts,
                grid_segment_count=grid_seg_const,
                time0=grid_s,
                time1=ray_s,
                nx_sum=int(args.nx_sum),
                eq_abs_l1=float(eq_abs_l1),
                eq_abs_rmse=float(eq_abs_rmse),
                eq_log_l1=float(eq_log_l1),
                eq_log_rmse=float(eq_log_rmse),
                eq_pos_overlap=int(pos_overlap),
            )

            _write_timing_table(rows, case_dir / "timing_report.md")
            _save_runtime_plot(
                rows,
                case_dir / "runtime_vs_pixels.png",
                title=f"{case.label}: runtime vs pixel count",
            )

            print(
                f"{case.label},{n}x{n},{pixels},{grid_s:.6f},{ray_s:.6f},"
                f"{grid_nan},{grid_zero},{ray_nan},{ray_zero},{finite_overlap},"
                f"{pos_overlap},{eq_abs_l1:.6e},{eq_abs_rmse:.6e},{eq_log_l1:.6e},{eq_log_rmse:.6e}"
            )


if __name__ == "__main__":
    main()

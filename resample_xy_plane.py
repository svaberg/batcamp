#!/usr/bin/env python3
"""Resample the reference 3D files onto one fixed `xy` plane across resolutions."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import numpy as np
from batread.dataset import Dataset

from batcamp import OctreeInterpolator
from resampling_compare import _load_or_build_octree
from resampling_compare import _progress
from resampling_compare import _resolution_ramp
from resampling_compare import _time_call
from resampling_compare import DatasetCase
from resampling_compare import resolve_data_file


def _xy_plane_image(
    interp: OctreeInterpolator,
    *,
    n_plane: int,
    z_plane: float,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample one scalar field onto one regular `xy` plane."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if not (float(zmin) <= float(z_plane) <= float(zmax)):
        raise ValueError(f"z_plane={z_plane} lies outside dataset z-bounds [{zmin}, {zmax}].")

    x = np.linspace(xmin, xmax, int(n_plane), dtype=float)
    y = np.linspace(ymin, ymax, int(n_plane), dtype=float)
    xg, yg = np.meshgrid(x, y, indexing="xy")
    zg = np.full_like(xg, float(z_plane), dtype=float)
    query = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))
    img = np.asarray(
        interp(query, query_coord="xyz", log_outside_domain=False),
        dtype=float,
    ).reshape(int(n_plane), int(n_plane))
    return xg, yg, img


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


def _save_xy_plane_figure(
    out_path: Path,
    *,
    dataset_label: str,
    variable: str,
    n_plane: int,
    z_plane: float,
    xg: np.ndarray,
    yg: np.ndarray,
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
        extent=(float(np.min(xg)), float(np.max(xg)), float(np.min(yg)), float(np.max(yg))),
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


def _write_timing_table(rows: list[dict[str, float | int]], out_path: Path) -> None:
    """Write markdown timing table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| resolution | pixels | plane_s | nan | zero | finite_min | finite_max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{int(row['resolution'])}x{int(row['resolution'])} | "
            f"{int(row['pixels'])} | "
            f"{float(row['plane_s']):.6f} | "
            f"{int(row['nan'])} | "
            f"{int(row['zero'])} | "
            f"{float(row['finite_min']):.6e} | "
            f"{float(row['finite_max']):.6e} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_runtime_plot(rows: list[dict[str, float | int]], out_path: Path, *, title: str) -> None:
    """Save runtime vs pixel-count line plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray([int(row["pixels"]) for row in rows], dtype=float)
    plane_t = np.asarray([float(row["plane_s"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(pixels, plane_t, "o-", label="xy plane resample")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pixel count (N x N)")
    ax.set_ylabel("Runtime [s]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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
        default="artifacts/resample_xy_plane",
        help="Output directory for PNGs and tables.",
    )
    args = parser.parse_args()

    resolutions = _resolution_ramp(int(args.min_resolution), int(args.max_resolution))
    max_seconds_per_image = float(args.max_seconds_per_image)
    if max_seconds_per_image <= 0.0:
        raise ValueError("max_seconds_per_image must be positive.")

    repo_root = Path(__file__).resolve().parent
    out_root = (repo_root / args.output_dir).resolve()
    cache_root = (repo_root / "build" / "resampling_compare_octree_cache").resolve()
    progress_log_path = out_root / "progress.log"
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_log_path.write_text("", encoding="utf-8")

    cases = [
        DatasetCase("example", "3d__var_1_n00000000.plt"),
        DatasetCase("ih", "3d__var_4_n00005000.plt"),
        DatasetCase("sc", "3d__var_4_n00044000.plt"),
    ]

    _progress(f"output_dir={out_root}", log_path=progress_log_path)
    print("dataset,resolution,pixels,plane_s,nan,zero,finite_min,finite_max", flush=True)
    for case in cases:
        case_dir = out_root / case.label
        case_dir.mkdir(parents=True, exist_ok=True)

        _progress(f"[{case.label}] start file={case.file_name}", log_path=progress_log_path)
        data_path, resolve_s = _time_call(resolve_data_file, repo_root, case.file_name)
        _progress(f"[{case.label}] resolved path={data_path} ({resolve_s:.2f}s)", log_path=progress_log_path)
        ds, read_s = _time_call(Dataset.from_file, str(data_path))
        _progress(f"[{case.label}] read dataset ({read_s:.2f}s)", log_path=progress_log_path)
        (tree, tree_source), tree_s = _time_call(_load_or_build_octree, ds, data_path, cache_root)
        _progress(f"[{case.label}] octree {tree_source} ({tree_s:.2f}s)", log_path=progress_log_path)
        interp, interp_s = _time_call(OctreeInterpolator, ds, [args.variable], tree=tree)
        _progress(f"[{case.label}] interpolator ready ({interp_s:.2f}s)", log_path=progress_log_path)

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
        _, warm_s = _time_call(_xy_plane_image, interp, n_plane=warm_n, z_plane=float(args.z_plane), bounds=bounds)
        _progress(f"[{case.label}] warmup plane={warm_s:.2f}s", log_path=progress_log_path)

        rows: list[dict[str, float | int]] = []
        for n in resolutions:
            _progress(f"[{case.label}] run {n}x{n}", log_path=progress_log_path)
            t0 = time.perf_counter()
            xg, yg, img = _xy_plane_image(
                interp,
                n_plane=int(n),
                z_plane=float(args.z_plane),
                bounds=bounds,
            )
            plane_s = float(time.perf_counter() - t0)
            nan_count, zero_count = _array_stats(img)
            finite_min, finite_max = _finite_min_max(img)
            pixels = int(n * n)

            row = {
                "resolution": int(n),
                "pixels": pixels,
                "plane_s": plane_s,
                "nan": int(nan_count),
                "zero": int(zero_count),
                "finite_min": float(finite_min),
                "finite_max": float(finite_max),
            }
            rows.append(row)

            _save_xy_plane_figure(
                case_dir / f"xy_plane_{n}x{n}.png",
                dataset_label=f"{case.label}:{case.file_name}",
                variable=args.variable,
                n_plane=int(n),
                z_plane=float(args.z_plane),
                xg=xg,
                yg=yg,
                img=img,
                time_s=plane_s,
            )
            _write_timing_table(rows, case_dir / "timing_report.md")
            _save_runtime_plot(rows, case_dir / "runtime_vs_pixels.png", title=f"{case.label}: xy plane runtime")

            print(
                f"{case.label},{n}x{n},{pixels},{plane_s:.6f},{nan_count},{zero_count},"
                f"{finite_min:.6e},{finite_max:.6e}",
                flush=True,
            )
            if plane_s > max_seconds_per_image:
                _progress(
                    f"[{case.label}] stop at {n}x{n}: plane={plane_s:.2f}s > {max_seconds_per_image:.2f}s",
                    log_path=progress_log_path,
                )
                break
        _progress(f"[{case.label}] done -> {case_dir}", log_path=progress_log_path)


if __name__ == "__main__":
    main()

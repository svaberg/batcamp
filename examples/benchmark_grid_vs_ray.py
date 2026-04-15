#!/usr/bin/env python3
"""Compare 3D grid-sum resampling with octree ray accumulation."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.transforms import blended_transform_factory
import numpy as np
from batread.dataset import Dataset

from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer
from batcamp.constants import XYZ_VARS
from benchmark_helpers import _build_octree
from benchmark_helpers import _configure_builder_logging
from benchmark_helpers import _configure_progress_logging
from benchmark_helpers import _ProgressReporter
from benchmark_helpers import _resolution_ramp
from benchmark_helpers import _time_call
from benchmark_helpers import DatasetCase
from benchmark_helpers import resolve_data_file


_GRID_CACHE_VERSION = 1


def _grid_sum_image(
    interp: OctreeInterpolator,
    *,
    n_plane: int,
    nx_sum: int,
    bounds: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Resample to one uniform XYZ grid and integrate along x; return image as (z, y)."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x = np.linspace(xmin, xmax, int(nx_sum), dtype=float)
    y = _plane_axis_points(ymin, ymax, int(n_plane))
    z = _plane_axis_points(zmin, zmax, int(n_plane))

    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
    query = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))
    vals = np.asarray(
        interp(query, query_coord="xyz", log_outside_domain=False),
        dtype=float,
    ).reshape(x.size, y.size, z.size)

    finite = np.isfinite(vals)
    summed = np.trapezoid(np.where(finite, vals, 0.0), x=x, axis=0)
    any_finite = np.any(finite, axis=0)
    out = np.full_like(summed, np.nan, dtype=float)
    out[any_finite] = summed[any_finite]
    return out.T


def _grid_cache_file(
    out_root: Path,
    *,
    case_label: str,
    data_path: Path,
    variable: str,
    n_plane: int,
    nx_sum: int,
    bounds: tuple[float, float, float, float, float, float],
) -> Path:
    data_stat = data_path.stat()
    key = "|".join(
        (
            str(data_path.resolve()),
            str(int(data_stat.st_size)),
            str(int(data_stat.st_mtime_ns)),
            str(variable),
            str(int(n_plane)),
            str(int(nx_sum)),
            ",".join(f"{float(value):.17g}" for value in bounds),
        )
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return (
        out_root
        / f"benchmark_grid_vs_ray_{case_label}_grid_cache_{int(n_plane)}x{int(n_plane)}_{digest}.npz"
    )


def _grid_cache_matches(
    cache: np.lib.npyio.NpzFile,
    *,
    data_path: Path,
    variable: str,
    n_plane: int,
    nx_sum: int,
    bounds: tuple[float, float, float, float, float, float],
) -> bool:
    data_stat = data_path.stat()
    return (
        int(cache["version"]) == _GRID_CACHE_VERSION
        and str(cache["data_path"]) == str(data_path.resolve())
        and int(cache["data_size"]) == int(data_stat.st_size)
        and int(cache["data_mtime_ns"]) == int(data_stat.st_mtime_ns)
        and str(cache["variable"]) == str(variable)
        and int(cache["n_plane"]) == int(n_plane)
        and int(cache["nx_sum"]) == int(nx_sum)
        and np.allclose(
            np.asarray(cache["bounds"], dtype=float),
            np.asarray(bounds, dtype=float),
            atol=0.0,
            rtol=0.0,
        )
    )


def _read_grid_cache(
    cache_path: Path,
    *,
    data_path: Path,
    variable: str,
    n_plane: int,
    nx_sum: int,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, float] | None:
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as cache:
        if not _grid_cache_matches(
            cache,
            data_path=data_path,
            variable=variable,
            n_plane=int(n_plane),
            nx_sum=int(nx_sum),
            bounds=bounds,
        ):
            return None
        return np.asarray(cache["image"], dtype=float), float(cache["grid_s"])


def _write_grid_cache(
    cache_path: Path,
    image: np.ndarray,
    *,
    grid_s: float,
    data_path: Path,
    variable: str,
    n_plane: int,
    nx_sum: int,
    bounds: tuple[float, float, float, float, float, float],
) -> None:
    data_stat = data_path.stat()
    np.savez_compressed(
        cache_path,
        version=np.array(_GRID_CACHE_VERSION, dtype=np.int64),
        image=np.asarray(image, dtype=float),
        grid_s=np.array(float(grid_s), dtype=float),
        data_path=np.array(str(data_path.resolve())),
        data_size=np.array(int(data_stat.st_size), dtype=np.int64),
        data_mtime_ns=np.array(int(data_stat.st_mtime_ns), dtype=np.int64),
        variable=np.array(str(variable)),
        n_plane=np.array(int(n_plane), dtype=np.int64),
        nx_sum=np.array(int(nx_sum), dtype=np.int64),
        bounds=np.asarray(bounds, dtype=float),
    )


def _grid_sum_image_cached(
    interp: OctreeInterpolator,
    *,
    cache_path: Path,
    data_path: Path,
    variable: str,
    n_plane: int,
    nx_sum: int,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, float, bool]:
    cached = _read_grid_cache(
        cache_path,
        data_path=data_path,
        variable=variable,
        n_plane=int(n_plane),
        nx_sum=int(nx_sum),
        bounds=bounds,
    )
    if cached is not None:
        image, grid_s = cached
        return image, float(grid_s), True

    image, grid_s = _time_call(
        _grid_sum_image,
        interp,
        n_plane=int(n_plane),
        nx_sum=int(nx_sum),
        bounds=bounds,
    )
    _write_grid_cache(
        cache_path,
        image,
        grid_s=float(grid_s),
        data_path=data_path,
        variable=variable,
        n_plane=int(n_plane),
        nx_sum=int(nx_sum),
        bounds=bounds,
    )
    return image, float(grid_s), False


def _plane_axis_points(lo: float, hi: float, n: int) -> np.ndarray:
    """Return one regular image-axis coordinate array at pixel centers."""
    if int(n) <= 0:
        raise ValueError("n must be positive.")
    step = (float(hi) - float(lo)) / float(n)
    return float(lo) + (np.arange(int(n), dtype=float) + 0.5) * step


def _ray_setup(
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build one `(z, y)` batch of parallel +x rays."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    y = _plane_axis_points(ymin, ymax, int(n_plane))
    z = _plane_axis_points(zmin, zmax, int(n_plane))
    yg, zg = np.meshgrid(y, z, indexing="xy")
    xg = np.full_like(yg, float(xmin), dtype=float)
    origins = np.stack((xg, yg, zg), axis=-1)
    directions = np.zeros_like(origins)
    directions[..., 0] = 1.0
    return origins, directions, float(xmax - xmin)


def _ray_image_and_segment_counts(
    interp: OctreeInterpolator,
    tracer: OctreeRayTracer,
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
    ray_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one plane of parallel rays and return one accumulated image and segment counts."""
    origins, directions, t_end = _ray_setup(n_plane=int(n_plane), bounds=bounds)
    if str(ray_method) == "midpoint":
        image, counts = tracer.accumulate_midpoint_image(
            interp,
            origins,
            directions,
            t_min=0.0,
            t_max=float(t_end),
        )
    elif str(ray_method) == "exact":
        image, counts = tracer.accumulate_exact_image(
            interp,
            origins,
            directions,
            t_min=0.0,
            t_max=float(t_end),
        )
    else:
        raise ValueError(f"Unsupported ray_method '{ray_method}'.")
    image = np.asarray(image, dtype=float)
    counts = np.asarray(counts, dtype=np.int64)
    return image, counts


def _ray_methods(ray_method: str) -> list[str]:
    """Return the concrete ray methods to run for one benchmark invocation."""
    if str(ray_method) == "both":
        return ["midpoint", "exact"]
    if str(ray_method) in {"midpoint", "exact"}:
        return [str(ray_method)]
    raise ValueError(f"Unsupported ray_method '{ray_method}'.")


def _pixel_plane_coordinates(
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-pixel y, z, and r=sqrt(y^2+z^2) on the image plane as (z, y)."""
    _xmin, _xmax, ymin, ymax, zmin, zmax = bounds
    y = _plane_axis_points(ymin, ymax, int(n_plane))
    z = _plane_axis_points(zmin, zmax, int(n_plane))
    yg, zg = np.meshgrid(y, z, indexing="xy")
    rg = np.sqrt(yg * yg + zg * zg)
    return yg, zg, rg


def _array_stats(a: np.ndarray) -> tuple[int, int]:
    """Return (nan_count, zero_count_exact)."""
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


def _discrepancy_rows(
    img0: np.ndarray,
    img1: np.ndarray,
    *,
    pixel_y: np.ndarray,
    pixel_z: np.ndarray,
    pixel_r: np.ndarray,
    log10_threshold: float = 0.1,
    max_finite_rows: int = 256,
) -> list[dict[str, float | int | str]]:
    """Return one sorted discrepancy list for one comparison image pair."""
    pos0 = np.isfinite(img0) & (img0 > 0.0)
    pos1 = np.isfinite(img1) & (img1 > 0.0)

    categories = [
        ("grid_pos_ray_nan", pos0 & np.isnan(img1), np.inf),
        ("grid_pos_ray_zero", pos0 & np.isfinite(img1) & (img1 == 0.0), np.inf),
        ("grid_zero_ray_pos", np.isfinite(img0) & (img0 == 0.0) & pos1, np.inf),
        ("grid_nan_ray_pos", np.isnan(img0) & pos1, np.inf),
    ]

    rows: list[dict[str, float | int | str]] = []
    for kind, mask, log_mag in categories:
        iz, iy = np.nonzero(mask)
        for k in range(iz.size):
            i = int(iz[k])
            j = int(iy[k])
            grid_val = float(img0[i, j])
            ray_val = float(img1[i, j])
            abs_diff = np.nan if (not np.isfinite(grid_val) or not np.isfinite(ray_val)) else abs(ray_val - grid_val)
            rows.append(
                {
                    "kind": kind,
                    "iz": i,
                    "iy": j,
                    "y": float(pixel_y[i, j]),
                    "z": float(pixel_z[i, j]),
                    "r": float(pixel_r[i, j]),
                    "grid_value": grid_val,
                    "ray_value": ray_val,
                    "abs_diff": abs_diff,
                    "abs_log10_diff": float(log_mag),
                }
            )

    both_pos = pos0 & pos1
    if np.any(both_pos):
        abs_log10 = np.abs(np.log10(img1[both_pos]) - np.log10(img0[both_pos]))
        if np.any(abs_log10 >= log10_threshold):
            coords = np.column_stack(np.nonzero(both_pos))
            selected = np.nonzero(abs_log10 >= log10_threshold)[0]
            selected = selected[np.argsort(abs_log10[selected])[::-1]]
            selected = selected[: int(max_finite_rows)]
            for idx in selected:
                i = int(coords[idx, 0])
                j = int(coords[idx, 1])
                rows.append(
                    {
                        "kind": "finite_log10_mismatch",
                        "iz": i,
                        "iy": j,
                        "y": float(pixel_y[i, j]),
                        "z": float(pixel_z[i, j]),
                        "r": float(pixel_r[i, j]),
                        "grid_value": float(img0[i, j]),
                        "ray_value": float(img1[i, j]),
                        "abs_diff": float(abs(img1[i, j] - img0[i, j])),
                        "abs_log10_diff": float(abs_log10[idx]),
                    }
                )

    rows.sort(
        key=lambda row: (
            str(row["kind"]) != "grid_pos_ray_nan",
            str(row["kind"]) != "grid_pos_ray_zero",
            str(row["kind"]) != "grid_zero_ray_pos",
            str(row["kind"]) != "grid_nan_ray_pos",
            -float(row["abs_log10_diff"]) if np.isfinite(float(row["abs_log10_diff"])) else float("inf"),
            -float(row["r"]),
        )
    )
    return rows


def _write_discrepancy_csv(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    """Write one CSV list of flagged discrepancies for one resolution."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "iz",
        "iy",
        "y",
        "z",
        "r",
        "grid_value",
        "ray_value",
        "abs_diff",
        "abs_log10_diff",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_four_panel_figure(
    out_path: Path,
    *,
    dataset_label: str,
    ray_label: str,
    n_plane: int,
    img0: np.ndarray,
    img1: np.ndarray,
    pixel_r: np.ndarray,
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
    plot0_only = pos0 & np.isfinite(img1) & (img1 == 0.0)
    plot1_only = np.isfinite(img0) & (img0 == 0.0) & pos1
    plot0_nan = pos0 & np.isnan(img1)
    plot1_nan = np.isnan(img0) & pos1
    pos_vals = np.concatenate((img0[pos0], img1[pos1])) if (np.any(pos0) or np.any(pos1)) else np.array([], dtype=float)

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

    fig = plt.figure(figsize=(10.4, 10.8))

    panel_size = 0.34
    col_gap = 0.055
    row_gap = 0.060
    cbar_height = 0.018
    cbar_gap = 0.042
    grid_width = 2.0 * panel_size + col_gap
    grid_height = 2.0 * panel_size + row_gap
    x0 = 0.5 - 0.5 * grid_width
    y0 = 0.11

    cax = fig.add_axes([x0, y0 + grid_height + cbar_gap, grid_width, cbar_height])
    axes = np.array(
        [
            [
                fig.add_axes([x0, y0 + panel_size + row_gap, panel_size, panel_size]),
                fig.add_axes([x0 + panel_size + col_gap, y0 + panel_size + row_gap, panel_size, panel_size]),
            ],
            [
                fig.add_axes([x0, y0, panel_size, panel_size]),
                fig.add_axes([x0 + panel_size + col_gap, y0, panel_size, panel_size]),
            ],
        ],
        dtype=object,
    )

    im0 = axes[0, 0].imshow(img0_disp, origin="lower", cmap=cmap, norm=norm, aspect="equal")
    axes[0, 0].set_title("Grid")
    axes[0, 0].set_xlabel("y index")
    axes[0, 0].set_ylabel("z index", labelpad=1)
    axes[0, 0].text(
        0.02,
        0.98,
        f"time={time0:.4f}s\nnan={stats0[0]}\nzero={stats0[1]}",
        transform=axes[0, 0].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        fontsize=7,
    )

    axes[0, 1].imshow(img1_disp, origin="lower", cmap=cmap, norm=norm, aspect="equal")
    axes[0, 1].set_title(ray_label)
    axes[0, 1].set_xlabel("y index")
    axes[0, 1].set_ylabel("")
    axes[0, 1].text(
        0.02,
        0.98,
        f"time={time1:.4f}s\nnan={stats1[0]}\nzero={stats1[1]}",
        transform=axes[0, 1].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        fontsize=7,
    )

    cbar = fig.colorbar(im0, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=8, pad=1)

    axes[1, 0].set_title(f"Comparison: grid vs {ray_label.lower()}")
    if pos_vals.size > 0:
        lo_data = float(np.min(pos_vals))
        hi_data = float(np.max(pos_vals))
        if not np.isfinite(lo_data) or not np.isfinite(hi_data) or lo_data <= 0.0 or hi_data <= lo_data:
            lo_data, hi_data = 1.0, 10.0
        pad = 1.12
        lo = lo_data / pad
        hi = hi_data * pad
        x_boundary_transform = blended_transform_factory(axes[1, 0].transAxes, axes[1, 0].transData)
        y_boundary_transform = blended_transform_factory(axes[1, 0].transData, axes[1, 0].transAxes)
        boundary_frac_zero = 0.015
        boundary_frac_nan = 0.045
        r_mask = both_pos | plot0_only | plot1_only | plot0_nan | plot1_nan
        r_vals = pixel_r[r_mask].reshape(-1) if np.any(r_mask) else np.array([0.0], dtype=float)
        r_lo = float(np.min(r_vals))
        r_hi = float(np.max(r_vals))
        if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo:
            r_lo = 0.0
            r_hi = max(r_lo + 1.0, r_hi)
        r_norm = Normalize(vmin=r_lo, vmax=r_hi)
        if np.any(both_pos):
            axes[1, 0].scatter(
                img0[both_pos].reshape(-1),
                img1[both_pos].reshape(-1),
                c=pixel_r[both_pos].reshape(-1),
                cmap="cividis",
                norm=r_norm,
                s=12,
                alpha=0.85,
                linewidths=0.0,
            )
        if np.any(plot0_only):
            axes[1, 0].scatter(
                img0[plot0_only].reshape(-1),
                np.full(int(np.count_nonzero(plot0_only)), boundary_frac_zero, dtype=float),
                c=pixel_r[plot0_only].reshape(-1),
                cmap="cividis",
                norm=r_norm,
                marker="v",
                s=22,
                alpha=0.95,
                linewidths=0.0,
                transform=y_boundary_transform,
            )
        if np.any(plot1_only):
            axes[1, 0].scatter(
                np.full(int(np.count_nonzero(plot1_only)), boundary_frac_zero, dtype=float),
                img1[plot1_only].reshape(-1),
                c=pixel_r[plot1_only].reshape(-1),
                cmap="cividis",
                norm=r_norm,
                marker="<",
                s=22,
                alpha=0.95,
                linewidths=0.0,
                transform=x_boundary_transform,
            )
        if np.any(plot0_nan):
            axes[1, 0].scatter(
                img0[plot0_nan].reshape(-1),
                np.full(int(np.count_nonzero(plot0_nan)), boundary_frac_nan, dtype=float),
                c=pixel_r[plot0_nan].reshape(-1),
                cmap="cividis",
                norm=r_norm,
                marker="^",
                s=24,
                alpha=0.95,
                linewidths=0.0,
                transform=y_boundary_transform,
            )
        if np.any(plot1_nan):
            axes[1, 0].scatter(
                np.full(int(np.count_nonzero(plot1_nan)), boundary_frac_nan, dtype=float),
                img1[plot1_nan].reshape(-1),
                c=pixel_r[plot1_nan].reshape(-1),
                cmap="cividis",
                norm=r_norm,
                marker=">",
                s=24,
                alpha=0.95,
                linewidths=0.0,
                transform=x_boundary_transform,
            )
        axes[1, 0].set_xlim(lo, hi)
        axes[1, 0].set_ylim(lo, hi)
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
        axes[1, 0].plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
    else:
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_xlim(1.0, 10.0)
        axes[1, 0].set_ylim(1.0, 10.0)
        axes[1, 0].text(0.5, 0.5, "no positive overlap", transform=axes[1, 0].transAxes, ha="center", va="center")
    axes[1, 0].set_xlabel("grid values")
    axes[1, 0].set_ylabel(f"{ray_label.lower()} values", labelpad=1)
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
        f"positive overlap={eq_pos_overlap}\n"
        f"grid>0, {ray_label.lower()}=0: {int(np.count_nonzero(plot0_only))}\n"
        f"{ray_label.lower()}>0, grid=0: {int(np.count_nonzero(plot1_only))}\n"
        f"grid>0, {ray_label.lower()}=nan: {int(np.count_nonzero(plot0_nan))}\n"
        f"{ray_label.lower()}>0, grid=nan: {int(np.count_nonzero(plot1_nan))}",
        transform=axes[1, 0].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        fontsize=7,
    )

    axes[1, 1].set_title(f"{ray_label} Segment Count Histogram")
    seg = np.asarray(ray_segment_counts, dtype=np.int64).reshape(-1)
    if seg.size > 0:
        max_seg = int(np.max(seg))
        bins = np.arange(-0.5, max_seg + 1.5, 1.0)
        axes[1, 1].hist(seg, bins=bins, color="tab:blue", alpha=0.8, edgecolor="black", linewidth=0.5)
        min_seg = int(np.min(seg))
        median_seg = float(np.median(seg))
        max_seg_value = int(np.max(seg))
        median_seg_label = f"{int(median_seg)}" if float(median_seg).is_integer() else f"{median_seg:.1f}"
        axes[1, 1].axvline(
            float(min_seg),
            color="tab:orange",
            linestyle="--",
            linewidth=1.6,
            label=f"ray min = {min_seg}",
        )
        axes[1, 1].axvline(
            median_seg,
            color="tab:green",
            linestyle=":",
            linewidth=1.8,
            label=f"ray median = {median_seg_label}",
        )
        axes[1, 1].axvline(
            float(max_seg_value),
            color="tab:purple",
            linestyle="-.",
            linewidth=1.6,
            label=f"ray max = {max_seg_value}",
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
    axes[1, 1].set_ylabel("")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"{dataset_label} | plane={n_plane}x{n_plane} | nx_sum={nx_sum}", fontsize=11, y=0.94)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_timing_table(
    rows: list[dict[str, float | int]],
    out_path: Path,
    *,
    ray_label: str,
    octree_tree_s: float,
    octree_interp_s: float,
    ray_tracer_s: float,
    cold_resolution: int,
    cold_grid_s: float,
    cold_ray_s: float,
) -> None:
    """Write one old-style timing table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "## Build",
        "",
        "| method | seconds |",
        "|---|---:|",
        f"| octree tree | {float(octree_tree_s):.6f} |",
        f"| octree interpolator | {float(octree_interp_s):.6f} |",
        f"| {ray_label.lower()} tracer | {float(ray_tracer_s):.6f} |",
        f"| total | {float(octree_tree_s + octree_interp_s + ray_tracer_s):.6f} |",
        "",
        "## Cold Start",
        "",
        f"- first image pair after build: `{int(cold_resolution)}x{int(cold_resolution)}`",
        "",
        f"| resolution | grid_s | {ray_label.lower()}_s | {ray_label.lower()}/grid |",
        "|---:|---:|---:|---:|",
        (
            f"| {int(cold_resolution)}x{int(cold_resolution)} | "
            f"{float(cold_grid_s):.6f} | "
            f"{float(cold_ray_s):.6f} | "
            f"{(float(cold_ray_s) / max(float(cold_grid_s), 1.0e-15)):.3f} |"
        ),
        "",
        "## Steady-State Runtime",
        "",
        f"| resolution | pixels | grid_s | {ray_label.lower()}_s | {ray_label.lower()}/grid | grid_nan | grid_zero | {ray_label.lower()}_nan | {ray_label.lower()}_zero | finite_overlap | positive_overlap | eq_abs_l1 | eq_abs_rmse | eq_log10_l1 | eq_log10_rmse |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = float(row["ray_s"]) / max(float(row["grid_s"]), 1.0e-15)
        eq_log10_l1 = float(row["eq_log10_l1"])
        eq_log10_rmse = float(row["eq_log10_rmse"])
        eq_log10_l1_text = f"{eq_log10_l1:.6e}" if np.isfinite(eq_log10_l1) else "nan"
        eq_log10_rmse_text = f"{eq_log10_rmse:.6e}" if np.isfinite(eq_log10_rmse) else "nan"
        lines.append(
            "| "
            f"{int(row['resolution'])}x{int(row['resolution'])} | "
            f"{int(row['pixels'])} | "
            f"{float(row['grid_s']):.6f} | "
            f"{float(row['ray_s']):.6f} | "
            f"{ratio:.3f} | "
            f"{int(row['grid_nan'])} | "
            f"{int(row['grid_zero'])} | "
            f"{int(row['ray_nan'])} | "
            f"{int(row['ray_zero'])} | "
            f"{int(row['finite_overlap'])} | "
            f"{int(row['positive_overlap'])} | "
            f"{float(row['eq_abs_l1']):.6e} | "
            f"{float(row['eq_abs_rmse']):.6e} | "
            f"{eq_log10_l1_text} | "
            f"{eq_log10_rmse_text} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_method_comparison_table(
    rows_by_method: dict[str, list[dict[str, float | int]]],
    out_path: Path,
) -> None:
    """Write one direct midpoint-versus-exact timing comparison table."""
    midpoint_rows = rows_by_method.get("midpoint", [])
    exact_rows = rows_by_method.get("exact", [])
    if not midpoint_rows or not exact_rows:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "## Ray Method Comparison",
        "",
        "| resolution | pixels | midpoint_s | exact_s | exact/midpoint |",
        "|---:|---:|---:|---:|---:|",
    ]
    for midpoint_row, exact_row in zip(midpoint_rows, exact_rows, strict=True):
        resolution = int(midpoint_row["resolution"])
        if resolution != int(exact_row["resolution"]):
            raise ValueError("midpoint/exact comparison rows must share the same resolutions.")
        midpoint_s = float(midpoint_row["ray_s"])
        exact_s = float(exact_row["ray_s"])
        lines.append(
            f"| {resolution}x{resolution} | "
            f"{int(midpoint_row['pixels'])} | "
            f"{midpoint_s:.6f} | "
            f"{exact_s:.6f} | "
            f"{(exact_s / max(midpoint_s, 1.0e-15)):.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_runtime_plot(
    rows: list[dict[str, float | int]],
    out_path: Path,
    *,
    title: str,
    ray_label: str,
    cold_resolution: int,
    cold_grid_s: float,
    cold_ray_s: float,
) -> None:
    """Save one figure with cold-start and steady-state runtimes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray([int(row["pixels"]) for row in rows], dtype=float)
    grid_t = np.asarray([float(row["grid_s"]) for row in rows], dtype=float)
    ray_t = np.asarray([float(row["ray_s"]) for row in rows], dtype=float)

    fig, (ax_cold, ax_steady) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.5),
        gridspec_kw={"width_ratios": [1.0, 2.3]},
        constrained_layout=True,
    )
    ax_cold.bar([0.0, 1.0], [float(cold_grid_s), float(cold_ray_s)], color=["C0", "C1"])
    ax_cold.set_xticks([0.0, 1.0])
    ax_cold.set_xticklabels(["plot0: 3D grid-sum", f"plot1: {ray_label.lower()}"], rotation=15, ha="right")
    ax_cold.set_yscale("log")
    ax_cold.set_ylabel("Runtime [s]")
    ax_cold.set_title(f"Cold start ({int(cold_resolution)}x{int(cold_resolution)})")
    ax_cold.grid(True, axis="y", which="both", alpha=0.25)

    ax_steady.plot(pixels, grid_t, "o-", label="plot0: 3D grid-sum")
    ax_steady.plot(pixels, ray_t, "o-", label=f"plot1: {ray_label.lower()}")
    ax_steady.set_xscale("log")
    ax_steady.set_yscale("log")
    ax_steady.set_xlabel("Pixel count (N x N)")
    ax_steady.set_ylabel("Runtime [s]")
    ax_steady.set_title("Steady state")
    ax_steady.grid(True, which="both", alpha=0.25)
    ax_steady.legend()
    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _artifact_path(out_root: Path, *, case_label: str, ray_method: str, name: str) -> Path:
    """Return one flat artifact path under the benchmark output root."""
    return out_root / f"benchmark_grid_vs_ray_{case_label}_{ray_method}_{name}"


def _run_case(
    *,
    case: DatasetCase,
    repo_root: Path,
    out_root: Path,
    progress: _ProgressReporter,
    resolutions: list[int],
    max_seconds_per_image: float,
    nx_sum: int,
    ray_method: str,
    variable: str,
) -> None:
    """Run one dataset case through the grid-vs-ray benchmark."""
    requested_ray_methods = _ray_methods(ray_method)
    progress.note(f"[{case.label}] file={case.file_name}")
    progress.note(f"[{case.label}] artifact_prefix=benchmark_grid_vs_ray_{case.label}_<method>_*")
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
    if str(tree.tree_coord) == "rpa":
        ray_methods = ["midpoint"]
        if requested_ray_methods != ray_methods:
            progress.note(f"[{case.label}] note: spherical ray benchmark uses midpoint accumulation")
    else:
        ray_methods = requested_ray_methods
    progress.start(f"[{case.label}] build interpolator")
    interp, interp_s = _time_call(OctreeInterpolator, tree, np.asarray(ds[variable], dtype=float))
    progress.complete(f"[{case.label}] build interpolator", interp_s)
    progress.start(f"[{case.label}] build ray tracer")
    tracer, tracer_s = _time_call(OctreeRayTracer, tree)
    progress.complete(f"[{case.label}] build ray tracer", tracer_s)

    xyz = np.column_stack(tuple(np.asarray(ds[name], dtype=float) for name in XYZ_VARS))
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
    progress.start(f"[{case.label}] cold start check")
    t0 = time.perf_counter()
    cold_grid_cache_path = _grid_cache_file(
        out_root,
        case_label=case.label,
        data_path=data_path,
        variable=variable,
        n_plane=warm_n,
        nx_sum=int(nx_sum),
        bounds=bounds,
    )
    _, cold_grid_s, cold_grid_cached = _grid_sum_image_cached(
        interp,
        cache_path=cold_grid_cache_path,
        data_path=data_path,
        variable=variable,
        n_plane=warm_n,
        nx_sum=int(nx_sum),
        bounds=bounds,
    )
    if cold_grid_cached:
        progress.note(f"[{case.label}] grid cache hit {warm_n}x{warm_n}")
    cold_ray_s_by_method: dict[str, float] = {}
    for method in ray_methods:
        try:
            (_cold_ray_img, _cold_counts), cold_ray_s = _time_call(
                _ray_image_and_segment_counts,
                interp,
                tracer,
                n_plane=warm_n,
                bounds=bounds,
                ray_method=method,
            )
        except ValueError as exc:
            cold_ray_s = float("nan")
            progress.note(f"[{case.label}] cold-start {method} ray trace failed at {warm_n}x{warm_n}: {exc}")
        cold_ray_s_by_method[method] = float(cold_ray_s)
    progress.complete(
        f"[{case.label}] cold start check",
        float(time.perf_counter() - t0),
        detail=" ".join(
            [f"grid={cold_grid_s:.2f}s" + (" cached" if cold_grid_cached else "")]
            + [f"{method}={cold_ray_s_by_method[method]:.2f}s" for method in ray_methods]
        ),
    )
    rows_by_method: dict[str, list[dict[str, float | int]]] = {method: [] for method in ray_methods}
    for n in resolutions:
        progress.start(f"[{case.label}] run {n}x{n}")
        t_step = time.perf_counter()

        grid_cache_path = _grid_cache_file(
            out_root,
            case_label=case.label,
            data_path=data_path,
            variable=variable,
            n_plane=int(n),
            nx_sum=int(nx_sum),
            bounds=bounds,
        )
        img0, grid_s, grid_cached = _grid_sum_image_cached(
            interp,
            cache_path=grid_cache_path,
            data_path=data_path,
            variable=variable,
            n_plane=int(n),
            nx_sum=int(nx_sum),
            bounds=bounds,
        )
        if grid_cached:
            progress.note(f"[{case.label}] grid cache hit {int(n)}x{int(n)}")
        pixel_y, pixel_z, pixel_r = _pixel_plane_coordinates(n_plane=int(n), bounds=bounds)
        grid_nan, grid_zero = _array_stats(img0)
        pixels = int(n * n)
        stop_after_resolution = False
        method_details: list[str] = [f"grid={grid_s:.2f}s" + (" cached" if grid_cached else "")]
        for method in ray_methods:
            ray_label = f"Ray {method}"
            try:
                (img1, ray_seg_counts), ray_s = _time_call(
                    _ray_image_and_segment_counts,
                    interp,
                    tracer,
                    n_plane=int(n),
                    bounds=bounds,
                    ray_method=method,
                )
            except ValueError as exc:
                progress.note(f"[{case.label}] {method} ray trace failed at {n}x{n}: {exc}")
                stop_after_resolution = True
                continue

            ray_nan, ray_zero = _array_stats(img1)
            finite_overlap, pos_overlap, eq_abs_l1, eq_abs_rmse, eq_log_l1, eq_log_rmse = _equality_deviation(img0, img1)
            row = {
                "resolution": int(n),
                "pixels": pixels,
                "grid_s": float(grid_s),
                "ray_s": float(ray_s),
                "grid_nan": int(grid_nan),
                "grid_zero": int(grid_zero),
                "ray_nan": int(ray_nan),
                "ray_zero": int(ray_zero),
                "finite_overlap": int(finite_overlap),
                "positive_overlap": int(pos_overlap),
                "eq_abs_l1": float(eq_abs_l1),
                "eq_abs_rmse": float(eq_abs_rmse),
                "eq_log10_l1": float(eq_log_l1),
                "eq_log10_rmse": float(eq_log_rmse),
            }
            rows_by_method[method].append(row)

            _save_four_panel_figure(
                _artifact_path(out_root, case_label=case.label, ray_method=method, name=f"resample_grid_vs_ray_{n}x{n}.png"),
                dataset_label=f"{case.label}:{case.file_name}",
                ray_label=ray_label,
                n_plane=int(n),
                img0=img0,
                img1=img1,
                pixel_r=pixel_r,
                ray_segment_counts=ray_seg_counts,
                grid_segment_count=max(int(nx_sum) - 1, 1),
                time0=float(grid_s),
                time1=float(ray_s),
                nx_sum=int(nx_sum),
                eq_abs_l1=float(eq_abs_l1),
                eq_abs_rmse=float(eq_abs_rmse),
                eq_log_l1=float(eq_log_l1),
                eq_log_rmse=float(eq_log_rmse),
                eq_pos_overlap=int(pos_overlap),
            )
            _write_discrepancy_csv(
                _discrepancy_rows(
                    img0,
                    img1,
                    pixel_y=pixel_y,
                    pixel_z=pixel_z,
                    pixel_r=pixel_r,
                ),
                _artifact_path(out_root, case_label=case.label, ray_method=method, name=f"discrepancies_{n}x{n}.csv"),
            )
            _write_timing_table(
                rows_by_method[method],
                _artifact_path(out_root, case_label=case.label, ray_method=method, name="timing_report.md"),
                ray_label=ray_label,
                octree_tree_s=float(tree_s),
                octree_interp_s=float(interp_s),
                ray_tracer_s=float(tracer_s),
                cold_resolution=warm_n,
                cold_grid_s=float(cold_grid_s),
                cold_ray_s=float(cold_ray_s_by_method[method]),
            )
            _save_runtime_plot(
                rows_by_method[method],
                _artifact_path(out_root, case_label=case.label, ray_method=method, name="runtime_vs_pixels.png"),
                title=f"{case.label}: grid vs {ray_label.lower()} runtime",
                ray_label=ray_label,
                cold_resolution=warm_n,
                cold_grid_s=float(cold_grid_s),
                cold_ray_s=float(cold_ray_s_by_method[method]),
            )
            method_details.append(f"{method}={ray_s:.2f}s")
            progress.note(
                f"{case.label},{method},{n}x{n},{pixels},{grid_s:.6f},{ray_s:.6f},"
                f"{grid_nan},{grid_zero},{ray_nan},{ray_zero},{finite_overlap},"
                f"{pos_overlap},{eq_abs_l1:.6e},{eq_abs_rmse:.6e},{eq_log_l1:.6e},{eq_log_rmse:.6e}"
            )
            if max(float(grid_s), float(ray_s)) > max_seconds_per_image:
                stop_after_resolution = True

        if len(ray_methods) > 1:
            _write_method_comparison_table(
                rows_by_method,
                _artifact_path(out_root, case_label=case.label, ray_method="compare", name="ray_method_compare.md"),
            )

        progress.complete(
            f"[{case.label}] run {n}x{n}",
            float(time.perf_counter() - t_step),
            detail=" ".join(method_details),
        )
        if stop_after_resolution:
            progress.note(f"[{case.label}] stop at {n}x{n}: reached {max_seconds_per_image:.2f}s limit")
            break
    progress.note(f"[{case.label}] done -> benchmark_grid_vs_ray_{case.label}_<method>_*")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 3D grid-sum resampling with octree ray accumulation.")
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
        default=20.0,
        help="Stop increasing resolution for one dataset once grid or ray time exceeds this many seconds.",
    )
    parser.add_argument(
        "--nx-sum",
        type=int,
        default=256,
        help="Number of x-samples for the 3D-grid summation baseline (default: 256).",
    )
    parser.add_argument(
        "--variable",
        default="Rho [g/cm^3]",
        help="Dataset variable to resample.",
    )
    parser.add_argument(
        "--ray-method",
        choices=("midpoint", "exact", "both"),
        default="exact",
        help="Ray accumulation method to benchmark (default: exact). Use 'both' for one-process midpoint/exact comparison.",
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
    progress_log_path = out_root / "benchmark_grid_vs_ray.log"
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
    progress.note(
        "dataset,method,resolution,pixels,grid_s,ray_s,grid_nan,grid_zero,ray_nan,ray_zero,"
        "finite_overlap,positive_overlap,eq_abs_l1,eq_abs_rmse,eq_log10_l1,eq_log10_rmse"
    )
    for case in cases:
        _run_case(
            case=case,
            repo_root=repo_root,
            out_root=out_root,
            progress=progress,
            resolutions=resolutions,
            max_seconds_per_image=max_seconds_per_image,
            nx_sum=int(args.nx_sum),
            ray_method=str(args.ray_method),
            variable=str(args.variable),
        )


if __name__ == "__main__":
    main()

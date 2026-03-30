#!/usr/bin/env python3
"""Compare grid-sum resampling vs ray integration across resolutions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import sys
import tarfile
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.transforms import blended_transform_factory
import numpy as np
from numba import njit
import pooch
from batread.dataset import Dataset

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS


_G2211_URL = "https://zenodo.org/records/7110555/files/run-Sun-G2211.tar.gz"
_G2211_SHA256 = "c31a32aab08cc20d5b643bba734fd7220e6b369e691f55f88a3a08cc5b2a2136"
_OCTREE_CACHE_VERSION = "v4"


@dataclass(frozen=True)
class DatasetCase:
    label: str
    file_name: str


@dataclass(frozen=True)
class XRayIntegrator:
    """Packed cell data for exact `+x` line integrals on one scalar field."""

    cell_y0: np.ndarray
    cell_dy: np.ndarray
    cell_z0: np.ndarray
    cell_dz: np.ndarray
    coeff00: np.ndarray
    coeff10: np.ndarray
    coeff01: np.ndarray
    coeff11: np.ndarray


class _ProgressReporter:
    """Simple progress logger for script stages."""

    def __init__(self, *, log_path: Path | None = None) -> None:
        self._log_path = log_path
        self._logger = logging.getLogger("resample.progress")

    def note(self, message: str) -> None:
        """Write one ordinary progress line."""
        self._logger.info(message)

    def start(self, message: str) -> None:
        """Start one timed stage."""
        self._logger.info("%s...", message)

    def complete(self, message: str, seconds: float, *, detail: str | None = None) -> None:
        """Finish one timed stage."""
        line = f"{message} complete ({seconds:.2f}s)"
        if detail:
            line = f"{line} {detail}"
        self._logger.info(line)


def _configure_progress_logging(*, log_path: Path) -> None:
    """Route script progress logs to stdout and the per-run progress log."""
    progress_logger = logging.getLogger("resample.progress")
    for handler in list(progress_logger.handlers):
        progress_logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    progress_logger.addHandler(stream_handler)
    progress_logger.addHandler(file_handler)
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False


def _configure_builder_logging(*, log_path: Path) -> None:
    """Route batcamp build/materialize logs to stdout and the per-run progress log."""
    formatter = logging.Formatter("  [%(filename)s:%(funcName)s:%(lineno)d] %(message)s")
    for logger_name in ("batcamp.builder", "batcamp.octree"):
        logger_obj = logging.getLogger(logger_name)
        for handler in list(logger_obj.handlers):
            logger_obj.removeHandler(handler)
            handler.close()
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(stream_handler)
        logger_obj.addHandler(file_handler)
        logger_obj.setLevel(logging.INFO)
        logger_obj.propagate = False


def _unique_match(paths: list[Path], *, name: str) -> Path:
    """Return one matched path by name, otherwise raise."""
    if not paths:
        raise FileNotFoundError(name)
    if len(paths) > 1:
        raise FileNotFoundError(f"Expected unique match for {name}, found {len(paths)}: {paths}")
    return paths[0]


def _find_in_sample_data(root: Path, name: str) -> Path:
    """Find one file by basename under sample_data."""
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
    """Resolve data file from sample_data first, then pooch fallback."""
    try:
        return _find_in_sample_data(repo_root / "sample_data", name)
    except FileNotFoundError:
        return _fetch_from_g2211_archive(name)


def _octree_cache_path(cache_root: Path, data_path: Path) -> Path:
    """Return one persistent octree cache path keyed by file contents."""
    stat = data_path.stat()
    cache_name = (
        f"{data_path.name}.{int(stat.st_size)}.{int(stat.st_mtime_ns)}."
        f"{_OCTREE_CACHE_VERSION}.octree.npz"
    )
    return cache_root / cache_name


def _load_or_build_octree(
    ds: Dataset,
    data_path: Path,
    cache_root: Path,
    *,
    use_cache: bool = True,
) -> tuple[Octree, str]:
    """Load one cached octree or rebuild it fresh."""
    cache_path = _octree_cache_path(cache_root, data_path)
    if use_cache and cache_path.exists():
        try:
            return (
                Octree.load(
                    cache_path,
                    points=np.column_stack(tuple(np.asarray(ds[name], dtype=np.float64) for name in XYZ_VARS)),
                    corners=np.asarray(ds.corners, dtype=np.int64),
                ),
                "cache",
            )
        except ValueError as exc:
            if "Missing required octree fields" not in str(exc):
                raise
            cache_path.unlink()

    tree = Octree.from_ds(ds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tree.save(cache_path)
    return tree, "build"


def _xyz_octree_cache_path(cache_root: Path, data_path: Path) -> Path:
    """Return one persistent XYZ-octree cache path keyed by file contents."""
    stat = data_path.stat()
    cache_name = (
        f"{data_path.name}.{int(stat.st_size)}.{int(stat.st_mtime_ns)}."
        "xyz.v1.octree.npz"
    )
    return cache_root / cache_name


def _load_or_build_xyz_octree(ds: Dataset, data_path: Path, cache_root: Path) -> tuple[Octree, str]:
    """Load one cached XYZ octree or build and persist it once."""
    cache_path = _xyz_octree_cache_path(cache_root, data_path)
    if cache_path.exists():
        try:
            return (
                Octree.load(
                    cache_path,
                    points=np.column_stack(tuple(np.asarray(ds[name], dtype=np.float64) for name in XYZ_VARS)),
                    corners=np.asarray(ds.corners, dtype=np.int64),
                ),
                "cache",
            )
        except ValueError as exc:
            if "Missing required octree fields" not in str(exc):
                raise
            cache_path.unlink()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tree = Octree.from_ds(ds, tree_coord="xyz")
    tree.save(cache_path)
    return tree, "build"


def _octree_prepare_detail(*, tree_source: str, tree_coord: str, no_cache: bool) -> str:
    """Return one readable octree-prep detail string."""
    if no_cache:
        return f"coord={tree_coord}"
    if tree_source == "cache":
        return f"cached octree coord={tree_coord}"
    return f"built octree and refreshed cache coord={tree_coord}"


def _resolution_ramp(min_resolution: int, max_resolution: int) -> list[int]:
    """Return the doubled resolution ramp `min, 2*min, ...` up to `max`."""
    if int(min_resolution) <= 0:
        raise ValueError("min_resolution must be positive.")
    if int(max_resolution) < int(min_resolution):
        raise ValueError("max_resolution must be >= min_resolution.")

    out: list[int] = []
    n = int(min_resolution)
    while n <= int(max_resolution):
        out.append(n)
        n *= 2
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


@njit(cache=True)
def _sample_index_range(
    cell_start: float,
    cell_width: float,
    axis_min: float,
    axis_max: float,
    n_axis: int,
) -> tuple[int, int]:
    """Return half-open sample-index coverage for one cell interval."""
    if cell_width <= 0.0 or n_axis <= 0:
        return 0, 0
    cell_stop = cell_start + cell_width
    tol = 1.0e-12 * max(1.0, abs(axis_max - axis_min), abs(cell_width))
    if n_axis == 1:
        coord = axis_min
        if coord < (cell_start - tol) or coord > (cell_stop + tol):
            return 0, 0
        return 0, 1

    axis_step = (axis_max - axis_min) / float(n_axis - 1)
    if axis_step <= 0.0:
        return 0, 0

    start_idx = int(math.ceil((cell_start - axis_min - tol) / axis_step))
    if cell_stop >= (axis_max - tol):
        stop_idx = int(n_axis)
    else:
        stop_idx = int(math.ceil((cell_stop - axis_min - tol) / axis_step))

    if start_idx < 0:
        start_idx = 0
    if stop_idx > n_axis:
        stop_idx = int(n_axis)
    if stop_idx <= start_idx:
        return 0, 0
    return start_idx, stop_idx


@njit(cache=True)
def _accumulate_xray_image(
    cell_y0: np.ndarray,
    cell_dy: np.ndarray,
    cell_z0: np.ndarray,
    cell_dz: np.ndarray,
    coeff00: np.ndarray,
    coeff10: np.ndarray,
    coeff01: np.ndarray,
    coeff11: np.ndarray,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    n_plane: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate exact full-cell `+x` line integrals on one uniform `(z, y)` grid."""
    image = np.zeros((n_plane, n_plane), dtype=np.float64)
    segment_counts = np.zeros((n_plane, n_plane), dtype=np.int64)
    y_step = 0.0 if n_plane <= 1 else (ymax - ymin) / float(n_plane - 1)
    z_step = 0.0 if n_plane <= 1 else (zmax - zmin) / float(n_plane - 1)

    n_cells = int(cell_y0.shape[0])
    for cell_id in range(n_cells):
        iy0, iy1 = _sample_index_range(float(cell_y0[cell_id]), float(cell_dy[cell_id]), ymin, ymax, int(n_plane))
        if iy1 <= iy0:
            continue
        iz0, iz1 = _sample_index_range(float(cell_z0[cell_id]), float(cell_dz[cell_id]), zmin, zmax, int(n_plane))
        if iz1 <= iz0:
            continue

        y0 = float(cell_y0[cell_id])
        dy = float(cell_dy[cell_id])
        z0 = float(cell_z0[cell_id])
        dz = float(cell_dz[cell_id])
        c00 = float(coeff00[cell_id])
        c10 = float(coeff10[cell_id])
        c01 = float(coeff01[cell_id])
        c11 = float(coeff11[cell_id])

        for iz in range(iz0, iz1):
            z = zmin if n_plane <= 1 else (zmin + float(iz) * z_step)
            wz1 = (z - z0) / dz
            if wz1 < 0.0:
                wz1 = 0.0
            elif wz1 > 1.0:
                wz1 = 1.0
            wz0 = 1.0 - wz1

            for iy in range(iy0, iy1):
                y = ymin if n_plane <= 1 else (ymin + float(iy) * y_step)
                wy1 = (y - y0) / dy
                if wy1 < 0.0:
                    wy1 = 0.0
                elif wy1 > 1.0:
                    wy1 = 1.0
                wy0 = 1.0 - wy1

                face0 = wy0 * c00 + wy1 * c10
                face1 = wy0 * c01 + wy1 * c11
                image[iz, iy] += wz0 * face0 + wz1 * face1
                segment_counts[iz, iy] += 1

    for iz in range(int(n_plane)):
        for iy in range(int(n_plane)):
            if segment_counts[iz, iy] == 0:
                image[iz, iy] = np.nan
    return image, segment_counts


def _build_xray_integrator(interp: OctreeInterpolator) -> XRayIntegrator:
    """Pack one scalar field into per-cell bilinear coefficients on the `yz` plane."""
    if str(interp.tree.tree_coord) != "xyz":
        raise ValueError("Grid-vs-ray comparison expects tree_coord='xyz'.")
    if int(interp.n_value_components) != 1:
        raise ValueError("Grid-vs-ray comparison expects exactly one scalar field.")

    leaf_ids = np.flatnonzero(np.asarray(interp.tree.cell_levels, dtype=np.int64) >= 0).astype(np.int64)
    cell_bounds = np.asarray(interp.tree._cell_bounds[leaf_ids], dtype=np.float64)
    corners = np.asarray(interp.tree.corners[leaf_ids], dtype=np.int64)
    point_values = np.asarray(interp._point_values_2d[:, 0], dtype=np.float64)
    corner_values = point_values[corners]
    dx = np.asarray(cell_bounds[:, 0, 1], dtype=np.float64)
    return XRayIntegrator(
        cell_y0=np.asarray(cell_bounds[:, 1, 0], dtype=np.float64),
        cell_dy=np.asarray(cell_bounds[:, 1, 1], dtype=np.float64),
        cell_z0=np.asarray(cell_bounds[:, 2, 0], dtype=np.float64),
        cell_dz=np.asarray(cell_bounds[:, 2, 1], dtype=np.float64),
        coeff00=dx * 0.5 * (corner_values[:, 0] + corner_values[:, 1]),
        coeff10=dx * 0.5 * (corner_values[:, 2] + corner_values[:, 3]),
        coeff01=dx * 0.5 * (corner_values[:, 4] + corner_values[:, 5]),
        coeff11=dx * 0.5 * (corner_values[:, 6] + corner_values[:, 7]),
    )


def _ray_setup(
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, float]:
    """Build origin points and extent for one `(z, y)` ray image plane."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    y = np.linspace(ymin, ymax, int(n_plane), dtype=float)
    z = np.linspace(zmin, zmax, int(n_plane), dtype=float)
    yg, zg = np.meshgrid(y, z, indexing="xy")
    xg = np.full_like(yg, float(xmin), dtype=float)
    origins = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))
    return origins, float(xmax - xmin)


def _lookup_cell_ids_along_x(
    tree: Octree,
    origins: np.ndarray,
    t_values: np.ndarray,
) -> np.ndarray:
    """Resolve containing cell ids at `origins + t * ex`."""
    query = np.array(origins, copy=True, dtype=np.float64, order="C")
    query[:, 0] += np.asarray(t_values, dtype=np.float64)
    return np.asarray(tree.lookup_points(query, coord="xyz"), dtype=np.int64)


def _interp_values_along_x(
    interp: OctreeInterpolator,
    origins: np.ndarray,
    t_values: np.ndarray,
) -> np.ndarray:
    """Evaluate interpolated scalar values at `origins + t * ex`."""
    query = np.array(origins, copy=True, dtype=np.float64, order="C")
    query[:, 0] += np.asarray(t_values, dtype=np.float64)
    return np.asarray(interp(query, query_coord="xyz", log_outside_domain=False), dtype=np.float64).reshape(-1)


def _adaptive_ray_image_and_segment_counts(
    interp: OctreeInterpolator,
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
    max_depth: int = 12,
    min_dt_fraction: float = 1.0 / 4096.0,
    fallback_substeps: int = 4,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate along `+x` rays by adaptive subdivision in `xyz` query space."""
    origins, t_end = _ray_setup(n_plane=int(n_plane), bounds=bounds)
    n_rays = int(origins.shape[0])
    image = np.zeros(n_rays, dtype=np.float64)
    segment_counts = np.zeros(n_rays, dtype=np.int64)
    if n_rays == 0 or t_end <= 0.0:
        return image.reshape((int(n_plane), int(n_plane))), segment_counts.reshape((int(n_plane), int(n_plane)))

    min_dt = max(float(t_end) * float(min_dt_fraction), 1.0e-12)
    ray_ids = np.arange(n_rays, dtype=np.int64)
    t0 = np.zeros(n_rays, dtype=np.float64)
    t1 = np.full(n_rays, float(t_end), dtype=np.float64)
    depth = np.zeros(n_rays, dtype=np.int16)

    while ray_ids.size > 0:
        next_ray_ids: list[np.ndarray] = []
        next_t0: list[np.ndarray] = []
        next_t1: list[np.ndarray] = []
        next_depth: list[np.ndarray] = []

        for start in range(0, int(ray_ids.size), int(chunk_size)):
            stop = min(int(ray_ids.size), start + int(chunk_size))
            rid = np.asarray(ray_ids[start:stop], dtype=np.int64)
            a = np.asarray(t0[start:stop], dtype=np.float64)
            b = np.asarray(t1[start:stop], dtype=np.float64)
            d = np.asarray(depth[start:stop], dtype=np.int16)
            dt = b - a
            origins_sub = np.asarray(origins[rid], dtype=np.float64)

            tq0 = a + 0.25 * dt
            tqm = a + 0.50 * dt
            tq1 = a + 0.75 * dt
            cid0 = _lookup_cell_ids_along_x(interp.tree, origins_sub, tq0)
            cidm = _lookup_cell_ids_along_x(interp.tree, origins_sub, tqm)
            cid1 = _lookup_cell_ids_along_x(interp.tree, origins_sub, tq1)

            same = (cid0 == cidm) & (cidm == cid1)
            accept = same & (cidm >= 0)
            outside = same & (cidm < 0)
            unresolved = ~(accept | outside)

            if np.any(accept):
                rid_acc = rid[accept]
                a_acc = a[accept]
                b_acc = b[accept]
                origins_acc = origins_sub[accept]
                half = 0.5 * (b_acc - a_acc)
                mid = 0.5 * (b_acc + a_acc)
                gauss = half / np.sqrt(3.0)
                vl = _interp_values_along_x(interp, origins_acc, mid - gauss)
                vr = _interp_values_along_x(interp, origins_acc, mid + gauss)
                image[rid_acc] += half * np.nan_to_num(vl, nan=0.0) + half * np.nan_to_num(vr, nan=0.0)
                segment_counts[rid_acc] += 1

            if np.any(unresolved):
                split = unresolved & (d < int(max_depth)) & (dt > float(min_dt))
                if np.any(split):
                    rid_split = rid[split]
                    a_split = a[split]
                    b_split = b[split]
                    d_split = (d[split] + 1).astype(np.int16)
                    mid = 0.5 * (a_split + b_split)
                    next_ray_ids.append(np.concatenate((rid_split, rid_split)))
                    next_t0.append(np.concatenate((a_split, mid)))
                    next_t1.append(np.concatenate((mid, b_split)))
                    next_depth.append(np.concatenate((d_split, d_split)))

                fallback = unresolved & ~split
                if np.any(fallback):
                    rid_fb = rid[fallback]
                    a_fb = a[fallback]
                    b_fb = b[fallback]
                    origins_fb = origins_sub[fallback]
                    dt_fb = (b_fb - a_fb) / float(fallback_substeps)
                    contrib = np.zeros(rid_fb.shape[0], dtype=np.float64)
                    for substep in range(int(fallback_substeps)):
                        tm = a_fb + (float(substep) + 0.5) * dt_fb
                        vals = _interp_values_along_x(interp, origins_fb, tm)
                        contrib += dt_fb * np.nan_to_num(vals, nan=0.0)
                    image[rid_fb] += contrib
                    segment_counts[rid_fb] += int(fallback_substeps)

        if next_ray_ids:
            ray_ids = np.concatenate(next_ray_ids)
            t0 = np.concatenate(next_t0)
            t1 = np.concatenate(next_t1)
            depth = np.concatenate(next_depth)
        else:
            ray_ids = np.empty((0,), dtype=np.int64)
            t0 = np.empty((0,), dtype=np.float64)
            t1 = np.empty((0,), dtype=np.float64)
            depth = np.empty((0,), dtype=np.int16)

    image_2d = image.reshape((int(n_plane), int(n_plane)))
    segment_counts_2d = segment_counts.reshape((int(n_plane), int(n_plane)))
    image_2d[segment_counts_2d == 0] = np.nan
    return image_2d, segment_counts_2d


def _ray_image(
    ray: XRayIntegrator,
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Integrate exactly along `+x` rays on one uniform `(z, y)` sample grid."""
    _xmin, _xmax, ymin, ymax, zmin, zmax = bounds
    values, _segment_counts = _accumulate_xray_image(
        ray.cell_y0,
        ray.cell_dy,
        ray.cell_z0,
        ray.cell_dz,
        ray.coeff00,
        ray.coeff10,
        ray.coeff01,
        ray.coeff11,
        float(ymin),
        float(ymax),
        float(zmin),
        float(zmax),
        int(n_plane),
    )
    return values


def _pixel_plane_coordinates(
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-pixel `y`, `z`, and `r = sqrt(y^2 + z^2)` on the image plane as `(z, y)`."""
    _xmin, _xmax, ymin, ymax, zmin, zmax = bounds
    y = np.linspace(ymin, ymax, int(n_plane), dtype=float)
    z = np.linspace(zmin, zmax, int(n_plane), dtype=float)
    yg, zg = np.meshgrid(y, z, indexing="xy")
    rg = np.sqrt(yg * yg + zg * zg)
    return yg, zg, rg


def _ray_segment_counts(
    ray: XRayIntegrator,
    *,
    n_plane: int,
    bounds: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Return per-pixel counts of crossed cells along `+x`."""
    _xmin, _xmax, ymin, ymax, zmin, zmax = bounds
    _image, counts = _accumulate_xray_image(
        ray.cell_y0,
        ray.cell_dy,
        ray.cell_z0,
        ray.cell_dz,
        ray.coeff00,
        ray.coeff10,
        ray.coeff01,
        ray.coeff11,
        float(ymin),
        float(ymax),
        float(zmin),
        float(zmax),
        int(n_plane),
    )
    return counts


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

    axes[0, 1].imshow(img1_disp, origin="lower", cmap=cmap, norm=norm, aspect="equal")
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
        scatter_artist = None
        if np.any(both_pos):
            x = img0[both_pos].reshape(-1)
            y = img1[both_pos].reshape(-1)
            scatter_artist = axes[1, 0].scatter(
                x,
                y,
                c=pixel_r[both_pos].reshape(-1),
                cmap="cividis",
                norm=r_norm,
                s=12,
                alpha=0.85,
                linewidths=0.0,
            )
        if np.any(plot0_only):
            scatter_artist = axes[1, 0].scatter(
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
            scatter_artist = axes[1, 0].scatter(
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
            scatter_artist = axes[1, 0].scatter(
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
            scatter_artist = axes[1, 0].scatter(
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
        if scatter_artist is not None:
            cbar_r = fig.colorbar(scatter_artist, ax=axes[1, 0], fraction=0.046, pad=0.02)
            cbar_r.set_label("r = sqrt(y^2 + z^2)")
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
        f"positive overlap={eq_pos_overlap}\n"
        f"plot0>0, plot1=0: {int(np.count_nonzero(plot0_only))}\n"
        f"plot1>0, plot0=0: {int(np.count_nonzero(plot1_only))}\n"
        f"plot0>0, plot1=nan: {int(np.count_nonzero(plot0_nan))}\n"
        f"plot1>0, plot0=nan: {int(np.count_nonzero(plot1_nan))}",
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


def _write_timing_table(
    rows: list[dict[str, float | int]],
    out_path: Path,
    *,
    cold_resolution: int,
    cold_grid_s: float,
    cold_ray_s: float,
) -> None:
    """Write markdown timing table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "## Cold Start",
        "",
        f"- first image pair after build: `{int(cold_resolution)}x{int(cold_resolution)}`",
        "",
        "| resolution | grid_s | ray_s | ray/grid |",
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


def _save_runtime_plot(
    rows: list[dict[str, float | int]],
    out_path: Path,
    *,
    title: str,
    cold_resolution: int,
    cold_grid_s: float,
    cold_ray_s: float,
) -> None:
    """Save one figure with cold-start and steady-state runtimes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray([int(r["pixels"]) for r in rows], dtype=float)
    grid_t = np.asarray([float(r["grid_s"]) for r in rows], dtype=float)
    ray_t = np.asarray([float(r["ray_s"]) for r in rows], dtype=float)

    fig, (ax_cold, ax_steady) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.5),
        gridspec_kw={"width_ratios": [1.0, 2.3]},
        constrained_layout=True,
    )
    ax_cold.bar([0.0, 1.0], [float(cold_grid_s), float(cold_ray_s)], color=["C0", "C1"])
    ax_cold.set_xticks([0.0, 1.0])
    ax_cold.set_xticklabels(["plot0: 3D grid-sum", "plot1: ray integration"], rotation=15, ha="right")
    ax_cold.set_yscale("log")
    ax_cold.set_ylabel("Runtime [s]")
    ax_cold.set_title(f"Cold start ({int(cold_resolution)}x{int(cold_resolution)})")
    ax_cold.grid(True, axis="y", which="both", alpha=0.25)

    ax_steady.plot(pixels, grid_t, "o-", label="plot0: 3D grid-sum")
    ax_steady.plot(pixels, ray_t, "o-", label="plot1: ray integration")
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


def _time_call(fn, /, *args, **kwargs):
    """Run one callable and return `(result, seconds)`."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, float(time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 3D grid-sum vs ray integration resampling.")
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
        help="Stop increasing resolution for one dataset once grid or ray time exceeds this many seconds.",
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
        default="artifacts/resample_grid_vs_ray",
        help="Output directory for PNGs and tables.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore any existing octree cache, rebuild fresh, and update the shared cache.",
    )
    args = parser.parse_args()

    resolutions = _resolution_ramp(int(args.min_resolution), int(args.max_resolution))
    max_seconds_per_image = float(args.max_seconds_per_image)
    if max_seconds_per_image <= 0.0:
        raise ValueError("max_seconds_per_image must be positive.")
    repo_root = Path(__file__).resolve().parent
    out_root = (repo_root / args.output_dir).resolve()
    cache_root = (repo_root / "artifacts" / "resampling_compare_octree_cache").resolve()
    progress_log_path = out_root / "progress.log"
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
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
        "dataset,resolution,pixels,grid_s,ray_s,grid_nan,grid_zero,ray_nan,ray_zero,"
        "finite_overlap,positive_overlap,eq_abs_l1,eq_abs_rmse,eq_log10_l1,eq_log10_rmse"
    )
    for case in cases:
        case_dir = out_root / case.label
        case_dir.mkdir(parents=True, exist_ok=True)

        progress.note(f"[{case.label}] file={case.file_name}")
        progress.start(f"[{case.label}] resolve data file")
        data_path, resolve_s = _time_call(resolve_data_file, repo_root, case.file_name)
        progress.complete(f"[{case.label}] resolve data file", resolve_s, detail=f"-> {data_path}")
        progress.start(f"[{case.label}] read dataset")
        ds, read_s = _time_call(Dataset.from_file, str(data_path))
        progress.complete(f"[{case.label}] read dataset", read_s)
        progress.start(f"[{case.label}] prepare octree")
        (tree, tree_source), tree_s = _time_call(
            _load_or_build_octree,
            ds,
            data_path,
            cache_root,
            use_cache=not bool(args.no_cache),
        )
        progress.complete(
            f"[{case.label}] prepare octree",
            tree_s,
            detail=_octree_prepare_detail(
                tree_source=tree_source,
                tree_coord=str(tree.tree_coord),
                no_cache=bool(args.no_cache),
            ),
        )
        progress.start(f"[{case.label}] build interpolator")
        interp, interp_s = _time_call(OctreeInterpolator, tree, np.asarray(ds["Rho [g/cm^3]"], dtype=float))
        progress.complete(f"[{case.label}] build interpolator", interp_s)
        progress.start(f"[{case.label}] build ray interpolator")
        ray = interp
        ray_s0 = 0.0
        ray_detail = "mode=adaptive_xyz_queries"
        progress.complete(f"[{case.label}] build ray interpolator", ray_s0, detail=ray_detail)

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
        _, warm_grid_s = _time_call(_grid_sum_image, interp, n_plane=warm_n, nx_sum=int(args.nx_sum), bounds=bounds)
        _, warm_ray_s = _time_call(_adaptive_ray_image_and_segment_counts, interp, n_plane=warm_n, bounds=bounds)
        progress.complete(
            f"[{case.label}] cold start check",
            float(time.perf_counter() - t0),
            detail=f"first image {warm_n}x{warm_n} grid={warm_grid_s:.2f}s ray={warm_ray_s:.2f}s",
        )

        rows: list[dict[str, float | int]] = []
        for n in resolutions:
            progress.start(f"[{case.label}] run {n}x{n}")
            t_step = time.perf_counter()

            t0 = time.perf_counter()
            img0 = _grid_sum_image(interp, n_plane=int(n), nx_sum=int(args.nx_sum), bounds=bounds)
            grid_s = float(time.perf_counter() - t0)

            t0 = time.perf_counter()
            img1, ray_seg_counts = _adaptive_ray_image_and_segment_counts(
                interp,
                n_plane=int(n),
                bounds=bounds,
            )
            ray_s = float(time.perf_counter() - t0)
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

            figure_path = case_dir / f"resample_grid_vs_ray_{n}x{n}.png"
            pixel_y, pixel_z, pixel_r = _pixel_plane_coordinates(n_plane=int(n), bounds=bounds)
            _save_four_panel_figure(
                figure_path,
                dataset_label=f"{case.label}:{case.file_name}",
                n_plane=int(n),
                img0=img0,
                img1=img1,
                pixel_r=pixel_r,
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
            discrepancy_rows = _discrepancy_rows(
                img0,
                img1,
                pixel_y=pixel_y,
                pixel_z=pixel_z,
                pixel_r=pixel_r,
            )
            _write_discrepancy_csv(
                discrepancy_rows,
                case_dir / f"discrepancies_{n}x{n}.csv",
            )

            _write_timing_table(
                rows,
                case_dir / "timing_report.md",
                cold_resolution=warm_n,
                cold_grid_s=float(warm_grid_s),
                cold_ray_s=float(warm_ray_s),
            )
            _save_runtime_plot(
                rows,
                case_dir / "runtime_vs_pixels.png",
                title=f"{case.label}: grid vs ray runtime",
                cold_resolution=warm_n,
                cold_grid_s=float(warm_grid_s),
                cold_ray_s=float(warm_ray_s),
            )
            progress.complete(
                f"[{case.label}] run {n}x{n}",
                float(time.perf_counter() - t_step),
                detail=f"grid={grid_s:.2f}s ray={ray_s:.2f}s",
            )

            progress.note(
                f"{case.label},{n}x{n},{pixels},{grid_s:.6f},{ray_s:.6f},"
                f"{grid_nan},{grid_zero},{ray_nan},{ray_zero},{finite_overlap},"
                f"{pos_overlap},{eq_abs_l1:.6e},{eq_abs_rmse:.6e},{eq_log_l1:.6e},{eq_log_rmse:.6e}"
            )
            if max(grid_s, ray_s) > max_seconds_per_image:
                progress.note(
                    (
                        f"[{case.label}] stop at {n}x{n}: "
                        f"max(grid={grid_s:.2f}s, ray={ray_s:.2f}s) > {max_seconds_per_image:.2f}s"
                    )
                )
                break
        progress.note(f"[{case.label}] done -> {case_dir}")


if __name__ == "__main__":
    main()

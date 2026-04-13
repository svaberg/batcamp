#!/usr/bin/env python3
"""Cartesian octree ray tracing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ._xyz_refined_event_walk import trace_xyz_refined_event_path
from .octree import Octree

__all__ = ["OctreeRayTracer", "RaySegments", "render_midpoint_image"]


def _normalize_ray_arrays(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return flat finite ray arrays plus the leading broadcast shape."""
    o = np.array(origins, dtype=np.float64, order="C")
    d = np.array(directions, dtype=np.float64, order="C")
    if o.ndim == 0 or o.shape[-1] != 3:
        raise ValueError("origins must have shape (..., 3).")
    if d.shape != o.shape:
        raise ValueError("directions must have the same shape as origins.")
    if not np.all(np.isfinite(o)):
        raise ValueError("origins must contain only finite values.")
    if not np.all(np.isfinite(d)):
        raise ValueError("directions must contain only finite values.")
    d_flat = d.reshape(-1, 3)
    if np.any(np.linalg.norm(d_flat, axis=1) <= 0):
        raise ValueError("directions must be nonzero.")
    shape = (1,) if o.ndim == 1 else o.shape[:-1]
    return o.reshape(-1, 3), d_flat, shape


def _trace_xyz_ray_batch(
    tree: Octree,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    ray_shape: tuple[int, ...],
) -> "RaySegments":
    """Trace one flat Cartesian ray batch with the refined event walker."""
    o_flat = np.asarray(origins, dtype=np.float64)
    d_flat = np.asarray(directions, dtype=np.float64)
    n_rays = int(o_flat.shape[0])
    ray_offsets = np.empty(n_rays + 1, dtype=np.int64)
    time_offsets = np.empty(n_rays + 1, dtype=np.int64)
    ray_offsets[0] = 0
    time_offsets[0] = 0
    cell_rows: list[np.ndarray] = []
    time_rows: list[np.ndarray] = []
    n_cell = 0
    n_time = 0

    for ray_id in range(n_rays):
        cell_ids, times = trace_xyz_refined_event_path(
            tree,
            -1,
            o_flat[ray_id],
            d_flat[ray_id],
            t_min=t_min,
            t_max=t_max,
        )
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        times = np.asarray(times, dtype=np.float64)
        cell_rows.append(cell_ids)
        time_rows.append(times)
        n_cell += int(cell_ids.size)
        n_time += int(times.size)
        ray_offsets[ray_id + 1] = n_cell
        time_offsets[ray_id + 1] = n_time

    cell_ids_out = np.concatenate(cell_rows).astype(np.int64, copy=False) if n_cell else np.empty(0, dtype=np.int64)
    times_out = np.concatenate(time_rows).astype(np.float64, copy=False) if n_time else np.empty(0, dtype=np.float64)
    return RaySegments(
        ray_offsets=ray_offsets,
        time_offsets=time_offsets,
        cell_ids=cell_ids_out,
        times=times_out,
        ray_shape=ray_shape,
    )


@dataclass(frozen=True)
class RaySegments:
    """Packed per-ray event trace with one boundary-time list per ray."""

    ray_offsets: np.ndarray
    time_offsets: np.ndarray
    cell_ids: np.ndarray
    times: np.ndarray
    ray_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        ray_offsets = np.asarray(self.ray_offsets, dtype=np.int64)
        time_offsets = np.asarray(self.time_offsets, dtype=np.int64)
        cell_ids = np.asarray(self.cell_ids, dtype=np.int64)
        times = np.asarray(self.times, dtype=np.float64)
        if ray_offsets.ndim != 1 or time_offsets.ndim != 1:
            raise ValueError("ray_offsets and time_offsets must be 1D arrays.")
        if cell_ids.ndim != 1 or times.ndim != 1:
            raise ValueError("cell_ids and times must be 1D arrays.")
        if ray_offsets.size == 0 or time_offsets.size == 0 or ray_offsets[0] != 0 or time_offsets[0] != 0:
            raise ValueError("ray_offsets and time_offsets must start at 0.")
        if ray_offsets.shape != time_offsets.shape:
            raise ValueError("ray_offsets and time_offsets must have the same shape.")
        if np.any(np.diff(ray_offsets) < 0) or np.any(np.diff(time_offsets) < 0):
            raise ValueError("ray_offsets and time_offsets must be nondecreasing.")
        if int(ray_offsets[-1]) != int(cell_ids.size):
            raise ValueError("ray_offsets[-1] must equal the packed cell count.")
        if int(time_offsets[-1]) != int(times.size):
            raise ValueError("time_offsets[-1] must equal the packed time count.")

        cell_counts = np.diff(ray_offsets)
        time_counts = np.diff(time_offsets)
        expected_time_counts = np.where(cell_counts == 0, 0, cell_counts + 1)
        if not np.array_equal(time_counts, expected_time_counts):
            raise ValueError("Each nonempty ray must pack one more time than cell id.")

        for ray_id, cell_count in enumerate(cell_counts):
            if cell_count == 0:
                continue
            time_lo = int(time_offsets[ray_id])
            time_hi = int(time_offsets[ray_id + 1])
            ray_times = times[time_lo:time_hi]
            if np.any(ray_times[1:] < ray_times[:-1]):
                raise ValueError("Each ray must have nondecreasing boundary times.")

        object.__setattr__(self, "ray_offsets", ray_offsets)
        object.__setattr__(self, "time_offsets", time_offsets)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "ray_shape", tuple(int(v) for v in self.ray_shape))

    @property
    def n_rays(self) -> int:
        """Return the number of packed rays."""
        return int(self.ray_offsets.size - 1)


class OctreeRayTracer:
    """Trace Cartesian rays through one octree."""

    def __init__(self, tree: Octree) -> None:
        """Bind one tracer to one built Cartesian octree."""
        if not isinstance(tree, Octree):
            raise TypeError("OctreeRayTracer requires a built Octree as its first argument.")
        if str(tree.tree_coord) != "xyz":
            raise NotImplementedError("OctreeRayTracer currently supports only tree_coord='xyz'.")
        self.tree = tree

    def trace(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float = 0,
        t_max: float = np.inf,
    ) -> RaySegments:
        """Trace one Cartesian ray batch and return packed event segments."""
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")
        o_flat, d_flat, ray_shape = _normalize_ray_arrays(origins, directions)
        return _trace_xyz_ray_batch(
            self.tree,
            o_flat,
            d_flat,
            t_min=t_lo,
            t_max=t_hi,
            ray_shape=ray_shape,
        )

    def __str__(self) -> str:
        """Return a compact tracer summary."""
        return f"OctreeRayTracer(tree_coord={self.tree.tree_coord})"


def render_midpoint_image(
    interpolator,
    origins: np.ndarray,
    directions: np.ndarray,
    segments: RaySegments,
) -> np.ndarray:
    """Render one midpoint-sampled line integral over packed traced segments."""
    from .interpolator import OctreeInterpolator

    if not isinstance(interpolator, OctreeInterpolator):
        raise TypeError("render_midpoint_image requires one OctreeInterpolator.")

    o_flat, d_flat, ray_shape = _normalize_ray_arrays(origins, directions)
    if tuple(ray_shape) != tuple(segments.ray_shape):
        raise ValueError("segments.ray_shape must match the ray origin/direction shape.")

    n_rays = int(o_flat.shape[0])
    n_components = int(interpolator.n_components)
    accum = np.zeros((n_rays, n_components), dtype=np.float64)
    for ray_id in range(n_rays):
        cell_lo = int(segments.ray_offsets[ray_id])
        cell_hi = int(segments.ray_offsets[ray_id + 1])
        if cell_lo == cell_hi:
            continue
        time_lo = int(segments.time_offsets[ray_id])
        time_hi = int(segments.time_offsets[ray_id + 1])
        ray_cell_ids = segments.cell_ids[cell_lo:cell_hi]
        ray_times = segments.times[time_lo:time_hi]
        segment_lengths = np.diff(ray_times)
        active = (segment_lengths > 0) & (ray_cell_ids >= 0)
        if not np.any(active):
            continue
        mid_t = ((ray_times[:-1] + ray_times[1:]) / 2)[active]
        mid_xyz = o_flat[ray_id] + mid_t[:, None] * d_flat[ray_id]
        samples = np.asarray(interpolator(mid_xyz, query_coord="xyz", log_outside_domain=False), dtype=np.float64)
        samples_2d = samples.reshape(samples.shape[0], -1)
        accum[ray_id] = np.sum(samples_2d * segment_lengths[active, None], axis=0)

    out = accum.reshape(tuple(ray_shape) + interpolator.value_shape)
    if interpolator.value_shape:
        return out
    return out.reshape(tuple(ray_shape))

#!/usr/bin/env python3
"""Cartesian octree ray tracing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import cartesian_crossing_trace
from .octree import Octree

TRACE_CHUNK_SIZE = 256
DEFAULT_CROSSING_BUFFER_SIZE = 256


def normalize_rays(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return flat finite ray arrays plus the broadcast shape."""
    o = np.array(origins, dtype=np.float64, order="C")
    d = np.array(directions, dtype=np.float64, order="C")
    if o.ndim == 0 or o.shape[-1] != 3:
        raise ValueError("origins must have shape (..., 3).")
    if d.shape == (3,):
        d = np.broadcast_to(d, o.shape).copy()
    elif d.shape != o.shape:
        raise ValueError("directions must have shape (..., 3) or (3,).")
    if not np.all(np.isfinite(o)):
        raise ValueError("origins must contain only finite values.")
    if not np.all(np.isfinite(d)):
        raise ValueError("directions must contain only finite values.")
    d_flat = d.reshape(-1, 3)
    if np.any(np.linalg.norm(d_flat, axis=1) <= 0):
        raise ValueError("directions must be nonzero.")
    shape = (1,) if o.ndim == 1 else o.shape[:-1]
    return o.reshape(-1, 3), d_flat, shape


def trace_segments(
    tree: Octree,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    ray_shape: tuple[int, ...],
) -> "RaySegments":
    """Trace rays and return packed crossing segments."""
    o_flat = np.asarray(origins, dtype=np.float64)
    d_flat = np.asarray(directions, dtype=np.float64)
    n_rays = int(o_flat.shape[0])
    ray_offsets = np.empty(n_rays + 1, dtype=np.int64)
    time_offsets = np.empty(n_rays + 1, dtype=np.int64)
    ray_offsets[0] = 0
    time_offsets[0] = 0
    cell_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    n_cell = 0
    n_time = 0
    for chunk_lo in range(0, n_rays, TRACE_CHUNK_SIZE):
        chunk_hi = min(chunk_lo + TRACE_CHUNK_SIZE, n_rays)
        chunk_origins = o_flat[chunk_lo:chunk_hi]
        chunk_directions = d_flat[chunk_lo:chunk_hi]
        chunk_n_rays = int(chunk_hi - chunk_lo)
        cell_counts, time_counts, cell_buffer, time_buffer = fill_chunk(
            tree,
            chunk_origins,
            chunk_directions,
            t_min=t_min,
            t_max=t_max,
        )
        chunk_cell_offsets = np.empty(chunk_n_rays + 1, dtype=np.int64)
        chunk_time_offsets = np.empty(chunk_n_rays + 1, dtype=np.int64)
        chunk_cell_offsets[0] = 0
        chunk_time_offsets[0] = 0
        np.cumsum(cell_counts, out=chunk_cell_offsets[1:])
        np.cumsum(time_counts, out=chunk_time_offsets[1:])
        chunk_cell_total = int(chunk_cell_offsets[-1])
        chunk_time_total = int(chunk_time_offsets[-1])
        chunk_cells = np.empty(chunk_cell_total, dtype=np.int64)
        chunk_times = np.empty(chunk_time_total, dtype=np.float64)
        cartesian_crossing_trace.pack_buffer(
            cell_counts,
            time_counts,
            cell_buffer,
            time_buffer,
            chunk_cell_offsets,
            chunk_time_offsets,
            chunk_cells,
            chunk_times,
        )
        cell_chunks.append(chunk_cells)
        time_chunks.append(chunk_times)
        n_cell += chunk_cell_total
        n_time += chunk_time_total
        ray_offsets[chunk_lo + 1 : chunk_hi + 1] = n_cell - chunk_cell_total + chunk_cell_offsets[1:]
        time_offsets[chunk_lo + 1 : chunk_hi + 1] = n_time - chunk_time_total + chunk_time_offsets[1:]
    cell_ids_out = np.concatenate(cell_chunks).astype(np.int64, copy=False) if n_cell else np.empty(0, dtype=np.int64)
    times_out = np.concatenate(time_chunks).astype(np.float64, copy=False) if n_time else np.empty(0, dtype=np.float64)
    return RaySegments(
        ray_offsets=ray_offsets,
        time_offsets=time_offsets,
        cell_ids=cell_ids_out,
        times=times_out,
        ray_shape=ray_shape,
    )


def fill_chunk(
    tree: Octree,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    t_min: float,
    t_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace one chunk into reusable crossing buffers."""
    chunk_n_rays = int(origins.shape[0])
    crossing_capacity = DEFAULT_CROSSING_BUFFER_SIZE
    while True:
        cell_counts = np.empty(chunk_n_rays, dtype=np.int64)
        time_counts = np.empty(chunk_n_rays, dtype=np.int64)
        cell_buffer = np.empty((chunk_n_rays, crossing_capacity), dtype=np.int64)
        time_buffer = np.empty((chunk_n_rays, crossing_capacity + 1), dtype=np.float64)
        cartesian_crossing_trace.trace_buffer(
            tree._root_cell_ids,
            tree.cell_child,
            tree.cell_bounds,
            tree._domain_bounds,
            tree.cell_neighbor,
            origins,
            directions,
            float(t_min),
            float(t_max),
            cell_counts,
            time_counts,
            cell_buffer,
            time_buffer,
        )
        if np.any(cell_counts == -1) or np.any(time_counts == -1):
            crossing_capacity *= 2
            continue
        if np.any(cell_counts < 0) or np.any(time_counts < 0):
            raise ValueError("Cartesian ray trace encountered an invalid crossing.")
        return cell_counts, time_counts, cell_buffer, time_buffer


def reshape_image(image_flat: np.ndarray, ray_shape: tuple[int, ...], value_shape: tuple[int, ...]) -> np.ndarray:
    """Reshape one flat image onto the requested ray/value shape."""
    out = image_flat.reshape(tuple(ray_shape) + tuple(value_shape))
    if value_shape:
        return out
    return out.reshape(tuple(ray_shape))


def accumulate_midpoints(
    tree: Octree,
    interpolator,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    ray_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Trace rays and accumulate midpoint samples without packing segments."""
    from .interpolator import OctreeInterpolator
    from . import interpolator as interpolator_module

    if not isinstance(interpolator, OctreeInterpolator):
        raise TypeError("accumulate_midpoint_image requires one OctreeInterpolator.")
    if interpolator.tree is not tree:
        raise ValueError("interpolator.tree must match the tracer octree.")

    o_flat = np.asarray(origins, dtype=np.float64)
    d_flat = np.asarray(directions, dtype=np.float64)
    n_rays = int(o_flat.shape[0])
    n_components = int(interpolator.n_components)
    accum = np.zeros((n_rays, n_components), dtype=np.float64)
    cell_counts_out = np.zeros(n_rays, dtype=np.int64)
    for chunk_lo in range(0, n_rays, TRACE_CHUNK_SIZE):
        chunk_hi = min(chunk_lo + TRACE_CHUNK_SIZE, n_rays)
        chunk_origins = o_flat[chunk_lo:chunk_hi]
        chunk_directions = d_flat[chunk_lo:chunk_hi]
        cell_counts, _time_counts, cell_buffer, time_buffer = fill_chunk(
            tree,
            chunk_origins,
            chunk_directions,
            t_min=t_min,
            t_max=t_max,
        )
        cell_counts_out[chunk_lo:chunk_hi] = cell_counts
        accum[chunk_lo:chunk_hi] = interpolator_module.accumulate_midpoint_cells_xyz(
            chunk_origins,
            chunk_directions,
            cell_counts,
            cell_buffer,
            time_buffer,
            tree.cell_bounds,
            tree.corners,
            interpolator._point_values_2d,
        )
    return reshape_image(accum, ray_shape, interpolator.value_shape), cell_counts_out.reshape(tuple(ray_shape))


def accumulate_exact(
    tree: Octree,
    interpolator,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    ray_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Trace rays and accumulate exact trilinear cell integrals without packing segments."""
    from .interpolator import OctreeInterpolator
    from . import interpolator as interpolator_module

    if not isinstance(interpolator, OctreeInterpolator):
        raise TypeError("accumulate_exact_image requires one OctreeInterpolator.")
    if interpolator.tree is not tree:
        raise ValueError("interpolator.tree must match the tracer octree.")

    o_flat = np.asarray(origins, dtype=np.float64)
    d_flat = np.asarray(directions, dtype=np.float64)
    n_rays = int(o_flat.shape[0])
    n_components = int(interpolator.n_components)
    accum = np.zeros((n_rays, n_components), dtype=np.float64)
    cell_counts_out = np.zeros(n_rays, dtype=np.int64)
    for chunk_lo in range(0, n_rays, TRACE_CHUNK_SIZE):
        chunk_hi = min(chunk_lo + TRACE_CHUNK_SIZE, n_rays)
        chunk_origins = o_flat[chunk_lo:chunk_hi]
        chunk_directions = d_flat[chunk_lo:chunk_hi]
        cell_counts, _time_counts, cell_buffer, time_buffer = fill_chunk(
            tree,
            chunk_origins,
            chunk_directions,
            t_min=t_min,
            t_max=t_max,
        )
        cell_counts_out[chunk_lo:chunk_hi] = cell_counts
        accum[chunk_lo:chunk_hi] = interpolator_module.accumulate_exact_cells_xyz(
            chunk_origins,
            chunk_directions,
            cell_counts,
            cell_buffer,
            time_buffer,
            tree.cell_bounds,
            tree.corners,
            interpolator._point_values_2d,
        )
    return reshape_image(accum, ray_shape, interpolator.value_shape), cell_counts_out.reshape(tuple(ray_shape))


@dataclass(frozen=True)
class RaySegments:
    """Packed per-ray crossing segments with one time list per ray."""

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
                raise ValueError("Each ray must have nondecreasing crossing times.")

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
        """Trace Cartesian rays and return packed crossing segments."""
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")
        o_flat, d_flat, ray_shape = normalize_rays(origins, directions)
        return trace_segments(
            self.tree,
            o_flat,
            d_flat,
            t_min=t_lo,
            t_max=t_hi,
            ray_shape=ray_shape,
        )

    def accumulate_midpoint_image(
        self,
        interpolator,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float = 0,
        t_max: float = np.inf,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace rays and accumulate one midpoint-sampled image plus per-ray segment counts."""
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")
        o_flat, d_flat, ray_shape = normalize_rays(origins, directions)
        return accumulate_midpoints(
            self.tree,
            interpolator,
            o_flat,
            d_flat,
            t_min=t_lo,
            t_max=t_hi,
            ray_shape=ray_shape,
        )

    def accumulate_exact_image(
        self,
        interpolator,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float = 0,
        t_max: float = np.inf,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace rays and accumulate exact trilinear cell integrals plus per-ray segment counts."""
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")
        o_flat, d_flat, ray_shape = normalize_rays(origins, directions)
        return accumulate_exact(
            self.tree,
            interpolator,
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
    """Render one midpoint-sampled line integral from packed crossing segments."""
    from .interpolator import OctreeInterpolator

    if not isinstance(interpolator, OctreeInterpolator):
        raise TypeError("render_midpoint_image requires one OctreeInterpolator.")

    o_flat, d_flat, ray_shape = normalize_rays(origins, directions)
    if tuple(ray_shape) != tuple(segments.ray_shape):
        raise ValueError("segments.ray_shape must match the ray origin/direction shape.")

    n_rays = int(o_flat.shape[0])
    n_components = int(interpolator.n_components)
    accum = np.zeros((n_rays, n_components), dtype=np.float64)
    if segments.cell_ids.size == 0:
        out = accum.reshape(tuple(ray_shape) + interpolator.value_shape)
        if interpolator.value_shape:
            return out
        return out.reshape(tuple(ray_shape))

    cell_counts = segments.ray_offsets[1:] - segments.ray_offsets[:-1]
    ray_ids = np.repeat(np.arange(n_rays, dtype=np.int64), cell_counts)
    segment_ord = np.arange(segments.cell_ids.size, dtype=np.int64) - segments.ray_offsets[ray_ids]
    time_lo = segments.time_offsets[ray_ids] + segment_ord
    t0 = segments.times[time_lo]
    t1 = segments.times[time_lo + 1]
    segment_lengths = t1 - t0
    active = (segment_lengths > 0) & (segments.cell_ids >= 0)
    if not np.any(active):
        out = accum.reshape(tuple(ray_shape) + interpolator.value_shape)
        if interpolator.value_shape:
            return out
        return out.reshape(tuple(ray_shape))

    active_ray_ids = ray_ids[active]
    active_cell_ids = segments.cell_ids[active]
    segment_weights = segment_lengths[active]
    mid_t = (t0[active] + t1[active]) / 2
    mid_xyz = o_flat[active_ray_ids] + mid_t[:, None] * d_flat[active_ray_ids]

    samples = np.asarray(interpolator.interp_cells_xyz(mid_xyz, active_cell_ids), dtype=np.float64)
    samples_2d = samples.reshape(mid_xyz.shape[0], -1)
    for component_id in range(n_components):
        accum[:, component_id] = np.bincount(
            active_ray_ids,
            weights=samples_2d[:, component_id] * segment_weights,
            minlength=n_rays,
        )

    out = accum.reshape(tuple(ray_shape) + interpolator.value_shape)
    if interpolator.value_shape:
        return out
    return out.reshape(tuple(ray_shape))

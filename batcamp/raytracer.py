#!/usr/bin/env python3
"""Public octree ray tracing and direct ray accumulation.

This module is the coordinate-agnostic layer above the low-level crossing
kernels. It normalizes ray origin/direction arrays, traces them in chunks
through the Cartesian or spherical traversal backends, packs the resulting
crossing paths into `TracedRays`, and provides the direct midpoint/trilinear
image accumulators built on those paths.

The exact event logic for leaf entry, exit, and neighbor continuation lives in
the coordinate-specific raytracer backend modules. This file owns the public
tracer API, packed segment representation, chunked scratch-buffer management,
and status reporting for failed rays.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import numpy as np
from numba import njit
from numba import prange

from . import interpolator as interpolator_module
from . import raytracer_cartesian
from . import raytracer_spherical
from .octree import Octree
from .shared import TraceScratch

logger = logging.getLogger(__name__)

# Trace chunks are sized to keep the scratch buffers moderate while still
# amortizing the fixed tracing/packing overhead over many rays.
RAY_CHUNK_SIZE = 256

# Ray tracing reports one nonnegative success code plus negative sentinel values
# inherited from the coordinate-specific scratch trace backends.
RAY_STATUS_OK = 0


def _log_ray_status(label: str, coord: str, n_rays: int, ray_status: np.ndarray) -> None:
    """Log one summarized raw status-code line when any ray failed.

    Args:
        label (const): Short label for the calling trace/render path.
        coord (const): Octree coordinate system name.
        n_rays (const): Number of traced rays.
        ray_status (const): Per-ray status codes with `0` for success.
    """
    failed_status = np.asarray(ray_status[ray_status < 0], dtype=np.int64)
    if failed_status.size == 0:
        return
    codes, counts = np.unique(failed_status, return_counts=True)
    status_bits = ", ".join(
        f"-0x{-int(code):04x}:{int(count)}"
        for code, count in zip(codes.tolist(), counts.tolist(), strict=True)
    )
    logger.warning(
        "%s ray status: coord=%s n_rays=%d status_bits={%s} dropped_frac=%.6f",
        label,
        coord,
        n_rays,
        status_bits,
        float(failed_status.size) / float(n_rays),
    )


def prepare_rays(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return flat finite ray arrays plus the broadcast shape.

    Args:
        origins (const): Ray origins with shape `(..., 3)`.
        directions (const): Ray directions with shape `(..., 3)` or `(3,)`.

    Returns:
        `(origins_flat, directions_flat, ray_shape)`.
    """
    o = np.asarray(origins, dtype=np.float64, order="C")
    d = np.asarray(directions, dtype=np.float64, order="C")
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


@njit(cache=True, parallel=True)
def _pack_crossing_buffers(
    cell_counts: np.ndarray,
    time_counts: np.ndarray,
    cell_ids_scratch: np.ndarray,
    times_scratch: np.ndarray,
    cell_offsets: np.ndarray,
    time_offsets: np.ndarray,
    cell_ids_out: np.ndarray,
    times_out: np.ndarray,
) -> None:
    """Pack scratch-traced crossing rows into flat output arrays.

    Args:
        cell_counts (const): Per-ray packed cell counts.
        time_counts (const): Per-ray packed time counts.
        cell_ids_scratch (const): Per-ray scratch cell-id rows.
        times_scratch (const): Per-ray scratch time rows.
        cell_offsets (const): Packed cell offsets for each ray.
        time_offsets (const): Packed time offsets for each ray.
        cell_ids_out (output): Flat packed cell-id array.
        times_out (output): Flat packed time array.
    """
    n_rays = int(cell_counts.shape[0])
    for ray_id in prange(n_rays):
        cell_count = int(cell_counts[ray_id])
        cell_lo = int(cell_offsets[ray_id])
        for cell_pos in range(cell_count):
            cell_ids_out[cell_lo + cell_pos] = int(cell_ids_scratch[ray_id, cell_pos])
        time_count = int(time_counts[ray_id])
        time_lo = int(time_offsets[ray_id])
        for time_pos in range(time_count):
            times_out[time_lo + time_pos] = float(times_scratch[ray_id, time_pos])


def _reshape_image(image_flat: np.ndarray, ray_shape: tuple[int, ...], value_shape: tuple[int, ...]) -> np.ndarray:
    """Reshape one flat image onto the requested ray/value shape.

    Args:
        image_flat (const): Flat image array with one leading ray axis.
        ray_shape (const): Target ray-grid shape.
        value_shape (const): Per-ray value shape.

    Returns:
        Image reshaped onto `ray_shape + value_shape`.
    """
    out = image_flat.reshape(tuple(ray_shape) + tuple(value_shape))
    if value_shape:
        return out
    return out.reshape(tuple(ray_shape))


@dataclass(frozen=True)
class TracedRays:
    """Packed per-ray crossing segments with one time list per ray.

    Each nonempty ray contributes one contiguous slice of `cell_ids` and one
    matching contiguous slice of `times`, where the time slice is exactly one
    entry longer than the cell slice. The traced ray `origins` and
    `directions` are stored with the same shapes passed to `trace`.
    """

    ray_offsets: np.ndarray
    time_offsets: np.ndarray
    cell_ids: np.ndarray
    times: np.ndarray
    origins: np.ndarray
    directions: np.ndarray

    def __post_init__(self) -> None:
        """Validate the packed segment arrays and attached ray geometry."""
        ray_offsets = np.asarray(self.ray_offsets, dtype=np.int64)
        time_offsets = np.asarray(self.time_offsets, dtype=np.int64)
        cell_ids = np.asarray(self.cell_ids, dtype=np.int64)
        times = np.asarray(self.times, dtype=np.float64)
        origins = np.asarray(self.origins, dtype=np.float64, order="C")
        directions = np.asarray(self.directions, dtype=np.float64, order="C")
        if ray_offsets.ndim != 1 or time_offsets.ndim != 1:
            raise ValueError("ray_offsets and time_offsets must be 1D arrays.")
        if cell_ids.ndim != 1 or times.ndim != 1:
            raise ValueError("cell_ids and times must be 1D arrays.")
        if origins.ndim == 0 or origins.shape[-1] != 3:
            raise ValueError("origins must have shape (..., 3).")
        if ray_offsets.size == 0 or time_offsets.size == 0 or ray_offsets[0] != 0 or time_offsets[0] != 0:
            raise ValueError("ray_offsets and time_offsets must start at 0.")
        if ray_offsets.shape != time_offsets.shape:
            raise ValueError("ray_offsets and time_offsets must have the same shape.")
        if np.any(np.diff(ray_offsets) < 0) or np.any(np.diff(time_offsets) < 0):
            raise ValueError("ray_offsets and time_offsets must be nondecreasing.")
        ray_shape = (1,) if origins.ndim == 1 else tuple(int(v) for v in origins.shape[:-1])
        if int(ray_offsets[-1]) != int(cell_ids.size):
            raise ValueError("ray_offsets[-1] must equal the packed cell count.")
        if int(time_offsets[-1]) != int(times.size):
            raise ValueError("time_offsets[-1] must equal the packed time count.")
        if int(np.prod(ray_shape, dtype=np.int64)) != int(ray_offsets.size - 1):
            raise ValueError("ray_shape must match the number of packed rays.")
        if directions.shape != (3,) and directions.shape != origins.shape:
            raise ValueError("directions must have shape origins.shape or (3,).")

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
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "directions", directions)

    @property
    def n_rays(self) -> int:
        """Return the number of packed rays."""
        return int(self.ray_offsets.size - 1)

    @property
    def ray_shape(self) -> tuple[int, ...]:
        """Return the ray-grid shape."""
        return (1,) if self.origins.ndim == 1 else tuple(int(v) for v in self.origins.shape[:-1])

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the ray-grid shape."""
        return self.ray_shape

    @property
    def ndim(self) -> int:
        """Return the number of ray-grid axes."""
        return len(self.ray_shape)

    @property
    def size(self) -> int:
        """Return the total number of rays."""
        return self.n_rays

    def _selected_ray_ids(self, key) -> np.ndarray:
        """Return the selected flat ray ids under NumPy indexing semantics."""
        if not isinstance(key, tuple):
            key = (key,)
        if any(index is None for index in key):
            raise TypeError("TracedRays indexing does not support new axes.")
        return np.asarray(
            np.arange(self.n_rays, dtype=np.int64).reshape(self.ray_shape)[key],
            dtype=np.int64,
        )

    def _ray_index(self, key) -> tuple[int, int | tuple[int, ...]]:
        """Return one flat ray index plus the matching origin-grid index."""
        ray_id_array = self._selected_ray_ids(key)
        if ray_id_array.ndim != 0:
            raise TypeError("TracedRays.xyz and scalar indexing require one ray, not a subset.")
        ray_index = int(ray_id_array)
        if len(self.ray_shape) == 1:
            return ray_index, ray_index
        return ray_index, tuple(int(v) for v in np.unravel_index(ray_index, self.ray_shape))

    def _subset(self, key) -> TracedRays:
        """Return one repacked ray-grid subset."""
        origins_grid = self.origins if self.origins.ndim > 1 else self.origins[None, :]
        subset_origins = np.asarray(origins_grid[key], dtype=np.float64, order="C")
        if self.directions.ndim == 1:
            subset_directions = self.directions
        else:
            subset_directions = np.asarray(self.directions[key], dtype=np.float64, order="C")

        selected_ray_ids = self._selected_ray_ids(key).reshape(-1)
        cell_counts = self.ray_offsets[selected_ray_ids + 1] - self.ray_offsets[selected_ray_ids]
        time_counts = self.time_offsets[selected_ray_ids + 1] - self.time_offsets[selected_ray_ids]
        ray_offsets = np.empty(selected_ray_ids.size + 1, dtype=np.int64)
        time_offsets = np.empty(selected_ray_ids.size + 1, dtype=np.int64)
        ray_offsets[0] = 0
        time_offsets[0] = 0
        np.cumsum(cell_counts, out=ray_offsets[1:])
        np.cumsum(time_counts, out=time_offsets[1:])
        cell_ids = np.empty(int(ray_offsets[-1]), dtype=np.int64)
        times = np.empty(int(time_offsets[-1]), dtype=np.float64)
        for subset_ray_id, ray_id in enumerate(selected_ray_ids):
            cell_lo = int(self.ray_offsets[ray_id])
            cell_hi = int(self.ray_offsets[ray_id + 1])
            time_lo = int(self.time_offsets[ray_id])
            time_hi = int(self.time_offsets[ray_id + 1])
            subset_cell_lo = int(ray_offsets[subset_ray_id])
            subset_cell_hi = int(ray_offsets[subset_ray_id + 1])
            subset_time_lo = int(time_offsets[subset_ray_id])
            subset_time_hi = int(time_offsets[subset_ray_id + 1])
            cell_ids[subset_cell_lo:subset_cell_hi] = self.cell_ids[cell_lo:cell_hi]
            times[subset_time_lo:subset_time_hi] = self.times[time_lo:time_hi]
        return TracedRays(
            ray_offsets=ray_offsets,
            time_offsets=time_offsets,
            cell_ids=cell_ids,
            times=times,
            origins=subset_origins,
            directions=subset_directions,
        )

    def __getitem__(self, key) -> tuple[np.ndarray, np.ndarray] | TracedRays:
        """Return one ray or one repacked subset under NumPy indexing semantics."""
        ray_id_array = self._selected_ray_ids(key)
        if ray_id_array.ndim != 0:
            return self._subset(key)
        ray_index = int(ray_id_array)
        cell_lo = int(self.ray_offsets[ray_index])
        cell_hi = int(self.ray_offsets[ray_index + 1])
        time_lo = int(self.time_offsets[ray_index])
        time_hi = int(self.time_offsets[ray_index + 1])
        return self.cell_ids[cell_lo:cell_hi], self.times[time_lo:time_hi]

    def xyz(self, *key) -> np.ndarray:
        """Return one ray's event positions in Cartesian coordinates."""
        ray_index, origin_key = self._ray_index(key[0] if len(key) == 1 else key)
        time_lo = int(self.time_offsets[ray_index])
        time_hi = int(self.time_offsets[ray_index + 1])
        ray_times = self.times[time_lo:time_hi]
        origin_xyz = self.origins if self.origins.ndim == 1 else self.origins[origin_key]
        direction_xyz = self.directions if self.directions.ndim == 1 else self.directions[origin_key]
        return origin_xyz + ray_times[:, None] * direction_xyz


class OctreeRayTracer:
    """Trace rays through one octree.

    The tracer delegates exact leaf-crossing events to the Cartesian or
    spherical backend selected by `tree.tree_coord`, then either packs those
    events into `TracedRays` or accumulates them directly into image outputs.
    """

    def __init__(self, tree: Octree) -> None:
        """Bind one tracer to one built octree.

        Args:
            tree (const): Built octree whose coordinate system selects the tracing backend.
        """
        if not isinstance(tree, Octree):
            raise TypeError("OctreeRayTracer requires a built Octree as its first argument.")
        tree_coord = str(tree.tree_coord)
        if tree_coord not in {"xyz", "rpa"}:
            raise NotImplementedError("OctreeRayTracer currently supports only tree_coord='xyz' or 'rpa'.")
        self.tree = tree
        self._raytracer_module = raytracer_cartesian if tree_coord == "xyz" else raytracer_spherical
        self._crossing_buffer_size = int(self._raytracer_module.DEFAULT_CROSSING_BUFFER_SIZE)
        logger.info(
            "OctreeRayTracer.__init__: coord=%s crossing_buffer_size=%d",
            tree_coord,
            self._crossing_buffer_size,
        )

    def _trace_chunk_to_scratch(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float,
        t_max: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trace one chunk into scratch buffers, growing capacity on overflow.

        The scratch rows hold one traced crossing path per ray. If any ray
        exhausts the current row width, the whole chunk is retraced with a
        larger shared crossing capacity.

        Args:
            origins (const): Flat origin array with shape `(chunk_n_rays, 3)`.
            directions (const): Flat direction array with shape `(chunk_n_rays, 3)`.
            t_min (const): Lower parameter clip.
            t_max (const): Upper parameter clip.

        Returns:
            `(cell_counts, time_counts, cell_buffer, time_buffer, ray_status)`.
        """
        chunk_n_rays = int(origins.shape[0])
        crossing_capacity = int(self._crossing_buffer_size)
        buffer_overflow = int(self._raytracer_module.TRACE_BUFFER_OVERFLOW)
        logger.debug(
            "_trace_chunk_to_scratch: coord=%s n_rays=%d crossing_buffer_size=%d default=%d",
            self.tree.tree_coord,
            chunk_n_rays,
            crossing_capacity,
            int(self._raytracer_module.DEFAULT_CROSSING_BUFFER_SIZE),
        )
        while True:
            cell_counts = np.empty(chunk_n_rays, dtype=np.int64)
            time_counts = np.empty(chunk_n_rays, dtype=np.int64)
            cell_buffer = np.empty((chunk_n_rays, crossing_capacity), dtype=np.int64)
            time_buffer = np.empty((chunk_n_rays, crossing_capacity + 1), dtype=np.float64)
            self._raytracer_module.trace_rays(
                self.tree.traversal_tree,
                origins,
                directions,
                float(t_min),
                float(t_max),
                cell_counts,
                time_counts,
                cell_buffer,
                time_buffer,
            )
            overflow_rays = (
                (cell_counts == buffer_overflow)
                | (time_counts == buffer_overflow)
            )
            if np.any(overflow_rays):
                next_capacity = 2 * crossing_capacity
                logger.info(
                    "grow crossing buffer: coord=%s n_rays=%d %d -> %d",
                    self.tree.tree_coord,
                    chunk_n_rays,
                    crossing_capacity,
                    next_capacity,
                )
                crossing_capacity *= 2
                continue
            # Collapse the per-array scratch sentinels into one public status
            # array, then zero failed rows so later pack/accumulate code can
            # treat them as ordinary empty rays.
            raw_status = np.where(
                cell_counts < 0,
                cell_counts,
                np.where(time_counts < 0, time_counts, RAY_STATUS_OK),
            )
            ray_status = raw_status.astype(np.int64, copy=False)
            failed_rays = ray_status < 0
            cell_counts[failed_rays] = 0
            time_counts[failed_rays] = 0
            self._crossing_buffer_size = int(crossing_capacity)
            logger.debug(
                "_trace_chunk_to_scratch complete: coord=%s n_rays=%d crossing_buffer_size=%d",
                self.tree.tree_coord,
                chunk_n_rays,
                self._crossing_buffer_size,
            )
            return cell_counts, time_counts, cell_buffer, time_buffer, ray_status

    def _trace_segments(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float,
        t_max: float,
        ray_shape: tuple[int, ...],
        geometry_origins: np.ndarray,
        geometry_directions: np.ndarray,
    ) -> TracedRays:
        """Trace rays and return packed crossing segments.

        Args:
            origins (const): Flat origin array with shape `(n_rays, 3)`.
            directions (const): Flat direction array with shape `(n_rays, 3)`.
            t_min (const): Lower parameter clip.
            t_max (const): Upper parameter clip.
            ray_shape (const): Requested output ray-grid shape.
            geometry_origins (const): Traced origin array with the original input shape.
            geometry_directions (const): Traced direction array with the original input shape.

        Returns:
            Packed per-ray crossing segments for the traced rays.
        """
        o_flat = np.asarray(origins, dtype=np.float64)
        d_flat = np.asarray(directions, dtype=np.float64)
        n_rays = int(o_flat.shape[0])
        logger.debug(
            "_trace_segments: coord=%s n_rays=%d ray_shape=%s chunk_size=%d",
            self.tree.tree_coord,
            n_rays,
            tuple(int(v) for v in ray_shape),
            RAY_CHUNK_SIZE,
        )
        ray_offsets = np.empty(n_rays + 1, dtype=np.int64)
        time_offsets = np.empty(n_rays + 1, dtype=np.int64)
        ray_offsets[0] = 0
        time_offsets[0] = 0
        cell_chunks: list[np.ndarray] = []
        time_chunks: list[np.ndarray] = []
        n_cell = 0
        n_time = 0
        ray_status_all = np.full(n_rays, RAY_STATUS_OK, dtype=np.int64)
        for chunk_lo in range(0, n_rays, RAY_CHUNK_SIZE):
            chunk_hi = min(chunk_lo + RAY_CHUNK_SIZE, n_rays)
            chunk_origins = o_flat[chunk_lo:chunk_hi]
            chunk_directions = d_flat[chunk_lo:chunk_hi]
            chunk_n_rays = int(chunk_hi - chunk_lo)
            logger.debug(
                "_trace_segments chunk: coord=%s ray_lo=%d ray_hi=%d chunk_n_rays=%d",
                self.tree.tree_coord,
                chunk_lo,
                chunk_hi,
                chunk_n_rays,
            )
            cell_counts, time_counts, cell_buffer, time_buffer, ray_status = self._trace_chunk_to_scratch(
                chunk_origins,
                chunk_directions,
                t_min=t_min,
                t_max=t_max,
            )
            ray_status_all[chunk_lo:chunk_hi] = ray_status
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
            _pack_crossing_buffers(
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
            # Convert chunk-local packed offsets into offsets in the final
            # concatenated arrays without doing another prefix sum pass later.
            ray_offsets[chunk_lo + 1:chunk_hi + 1] = n_cell - chunk_cell_total + chunk_cell_offsets[1:]
            time_offsets[chunk_lo + 1:chunk_hi + 1] = n_time - chunk_time_total + chunk_time_offsets[1:]
        if n_cell:
            cell_ids_out = np.concatenate(cell_chunks).astype(np.int64, copy=False)
        else:
            cell_ids_out = np.empty(0, dtype=np.int64)
        if n_time:
            times_out = np.concatenate(time_chunks).astype(np.float64, copy=False)
        else:
            times_out = np.empty(0, dtype=np.float64)
        logger.debug(
            "_trace_segments complete: coord=%s packed_cells=%d packed_times=%d",
            self.tree.tree_coord,
            n_cell,
            n_time,
        )
        _log_ray_status("trace", self.tree.tree_coord, n_rays, ray_status_all)
        return TracedRays(
            ray_offsets=ray_offsets,
            time_offsets=time_offsets,
            cell_ids=cell_ids_out,
            times=times_out,
            origins=geometry_origins,
            directions=geometry_directions,
        )

    def accumulate_chunked(
        self,
        interpolator,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float,
        t_max: float,
        ray_shape: tuple[int, ...],
        accumulator,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace rays and accumulate one backend-provided chunked integral image."""
        if not isinstance(interpolator, interpolator_module.OctreeInterpolator):
            raise TypeError(f"{label} requires one OctreeInterpolator.")
        if interpolator.tree is not self.tree:
            raise ValueError("interpolator.tree must match the tracer octree.")
        o_flat = np.asarray(origins, dtype=np.float64)
        d_flat = np.asarray(directions, dtype=np.float64)
        n_rays = int(o_flat.shape[0])
        n_components = int(interpolator.n_components)
        logger.debug("%s: using %s for %d rays", label, accumulator.__name__, n_rays)
        accum = np.zeros((n_rays, n_components), dtype=np.float64)
        cell_counts_out = np.zeros(n_rays, dtype=np.int64)
        ray_status_all = np.full(n_rays, RAY_STATUS_OK, dtype=np.int64)
        for chunk_lo in range(0, n_rays, RAY_CHUNK_SIZE):
            chunk_hi = min(chunk_lo + RAY_CHUNK_SIZE, n_rays)
            chunk_origins = o_flat[chunk_lo:chunk_hi]
            chunk_directions = d_flat[chunk_lo:chunk_hi]
            cell_counts, _time_counts, cell_buffer, time_buffer, ray_status = self._trace_chunk_to_scratch(
                chunk_origins,
                chunk_directions,
                t_min=t_min,
                t_max=t_max,
            )
            ray_status_all[chunk_lo:chunk_hi] = ray_status
            cell_counts_out[chunk_lo:chunk_hi] = cell_counts
            trace = TraceScratch(cell_counts, cell_buffer, time_buffer)
            accum[chunk_lo:chunk_hi] = accumulator(
                chunk_origins,
                chunk_directions,
                trace,
                interpolator.field,
            )
        _log_ray_status(label, self.tree.tree_coord, n_rays, ray_status_all)
        return _reshape_image(accum, ray_shape, interpolator.value_shape), cell_counts_out.reshape(tuple(ray_shape))

    def render_midpoint_via_segments(
        self,
        interpolator,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float,
        t_max: float,
        ray_shape: tuple[int, ...],
        geometry_origins: np.ndarray,
        geometry_directions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace rays, pack segments, and render midpoint samples from those packed segments."""
        if not isinstance(interpolator, interpolator_module.OctreeInterpolator):
            raise TypeError("midpoint_image requires one OctreeInterpolator.")
        if interpolator.tree is not self.tree:
            raise ValueError("interpolator.tree must match the tracer octree.")
        logger.debug(
            "_render_midpoint_via_segments: coord=%s ray_shape=%s",
            self.tree.tree_coord,
            tuple(int(v) for v in ray_shape),
        )
        segments = self._trace_segments(
            origins,
            directions,
            t_min=t_min,
            t_max=t_max,
            ray_shape=ray_shape,
            geometry_origins=geometry_origins,
            geometry_directions=geometry_directions,
        )
        image = render_midpoint_image(
            interpolator,
            np.asarray(origins, dtype=np.float64).reshape(tuple(ray_shape) + (3,)),
            np.asarray(directions, dtype=np.float64).reshape(tuple(ray_shape) + (3,)),
            segments,
        )
        counts = (segments.ray_offsets[1:] - segments.ray_offsets[:-1]).reshape(tuple(ray_shape))
        return image, counts

    def trace(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float = 0,
        t_max: float = np.inf,
    ) -> TracedRays:
        """Trace rays and return packed crossing segments.

        Args:
            origins (const): Ray origins with shape `(..., 3)`.
            directions (const): Ray directions with shape `(..., 3)` or `(3,)`.
            t_min (const): Lower parameter clip.
            t_max (const): Upper parameter clip.

        Returns:
            Packed per-ray crossing segments with the broadcast ray shape.
        """
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")
        origins_array = np.asarray(origins, dtype=np.float64, order="C")
        directions_array = np.asarray(directions, dtype=np.float64, order="C")
        o_flat, d_flat, ray_shape = prepare_rays(origins_array, directions_array)
        return self._trace_segments(
            o_flat,
            d_flat,
            t_min=t_lo,
            t_max=t_hi,
            ray_shape=ray_shape,
            geometry_origins=origins_array,
            geometry_directions=directions_array,
        )

    def midpoint_image(
        self,
        interpolator,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float = 0,
        t_max: float = np.inf,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace rays and accumulate one midpoint-sampled image plus per-ray segment counts.

        Args:
            interpolator (const): Interpolator bound to the tracer octree.
            origins (const): Ray origins with shape `(..., 3)`.
            directions (const): Ray directions with shape `(..., 3)` or `(3,)`.
            t_min (const): Lower parameter clip.
            t_max (const): Upper parameter clip.

        Returns:
            `(image, counts)` reshaped onto the broadcast ray grid.
        """
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")
        origins_array = np.asarray(origins, dtype=np.float64, order="C")
        directions_array = np.asarray(directions, dtype=np.float64, order="C")
        o_flat, d_flat, ray_shape = prepare_rays(origins_array, directions_array)
        return self._raytracer_module.midpoint_image(
            self,
            interpolator,
            o_flat,
            d_flat,
            t_min=t_lo,
            t_max=t_hi,
            ray_shape=ray_shape,
            geometry_origins=origins_array,
            geometry_directions=directions_array,
        )

    def trilinear_image(
        self,
        interpolator,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        t_min: float = 0,
        t_max: float = np.inf,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace rays and integrate trilinear cell interpolants plus per-ray segment counts.

        Args:
            interpolator (const): Interpolator bound to the tracer octree.
            origins (const): Ray origins with shape `(..., 3)`.
            directions (const): Ray directions with shape `(..., 3)` or `(3,)`.
            t_min (const): Lower parameter clip.
            t_max (const): Upper parameter clip.

        Returns:
            `(image, counts)` reshaped onto the broadcast ray grid.
        """
        t_lo = float(t_min)
        t_hi = float(t_max)
        if not math.isfinite(t_lo):
            raise ValueError("t_min must be finite.")
        if math.isnan(t_hi):
            raise ValueError("t_max must not be NaN.")
        if t_hi < t_lo:
            raise ValueError("t_max must be greater than or equal to t_min.")
        o_flat, d_flat, ray_shape = prepare_rays(origins, directions)
        return self._raytracer_module.trilinear_image(
            self,
            interpolator,
            o_flat,
            d_flat,
            t_min=t_lo,
            t_max=t_hi,
            ray_shape=ray_shape,
        )

    def __str__(self) -> str:
        """Return a compact tracer summary.

        Returns:
            String summary containing the tracer coordinate system.
        """
        return f"OctreeRayTracer(tree_coord={self.tree.tree_coord})"


def render_midpoint_image(
    interpolator,
    origins: np.ndarray,
    directions: np.ndarray,
    segments: TracedRays,
) -> np.ndarray:
    """Render one midpoint-sampled line integral from packed crossing segments.

    This is the packed-segment post-processing path: expand each packed segment
    into one midpoint sample, evaluate the interpolator there, weight by the
    segment length, and bin the result back onto the owning ray id.

    Args:
        interpolator (const): Interpolator used to sample the midpoint values.
        origins (const): Ray origins with shape `(..., 3)`.
        directions (const): Ray directions with shape `(..., 3)` or `(3,)`.
        segments (const): Packed ray segments matching the ray grid.

    Returns:
        Midpoint-sampled image reshaped onto the broadcast ray grid.
    """
    if not isinstance(interpolator, interpolator_module.OctreeInterpolator):
        raise TypeError("render_midpoint_image requires one OctreeInterpolator.")
    o_flat, d_flat, ray_shape = prepare_rays(origins, directions)
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
    # `segment_ord` is the segment index inside each owning ray after expanding
    # the packed cell list into one row per segment.
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
    if str(interpolator.tree.tree_coord) == "xyz":
        samples = np.asarray(interpolator.interp_cells_xyz(mid_xyz, active_cell_ids), dtype=np.float64)
    else:
        samples = np.asarray(
            interpolator(mid_xyz, query_coord="xyz", log_outside_domain=False),
            dtype=np.float64,
        )
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

#!/usr/bin/env python3
"""Octree interpolator and interpolation kernels."""

from __future__ import annotations

import logging
import math
from time import perf_counter
from typing import Literal

from numba import njit
from numba import prange
import numpy as np

from .constants import XYZ_VARS
from .octree import AXIS0
from .octree import AXIS1
from .octree import AXIS2
from .octree import CartesianInterpKernelState
from .octree import Octree
from .octree import SphericalInterpKernelState
from .octree import START
from .octree import WIDTH

logger = logging.getLogger(__name__)

_DEFAULT_SEED_CHUNK_SIZE = 1024
_TWO_PI = 2.0 * math.pi
_TINY = np.finfo(np.float64).tiny


@njit(cache=True)
def _write_trilinear_row(
    out_row: np.ndarray,
    cell_id: int,
    frac_axis0: float,
    frac_axis1: float,
    frac_axis2: float,
    corners: np.ndarray,
    bin_to_corner: np.ndarray,
    point_values: np.ndarray,
) -> None:
    """Write one trilinear interpolation row for one cell from normalized local coordinates."""
    cid = int(cell_id)

    weight000 = (1.0 - frac_axis0) * (1.0 - frac_axis1) * (1.0 - frac_axis2)
    weight100 = frac_axis0 * (1.0 - frac_axis1) * (1.0 - frac_axis2)
    weight010 = (1.0 - frac_axis0) * frac_axis1 * (1.0 - frac_axis2)
    weight110 = frac_axis0 * frac_axis1 * (1.0 - frac_axis2)
    weight001 = (1.0 - frac_axis0) * (1.0 - frac_axis1) * frac_axis2
    weight101 = frac_axis0 * (1.0 - frac_axis1) * frac_axis2
    weight011 = (1.0 - frac_axis0) * frac_axis1 * frac_axis2
    weight111 = frac_axis0 * frac_axis1 * frac_axis2

    local = corners[cid]
    map_row = bin_to_corner[cid]
    c0 = int(local[int(map_row[0])])
    c1 = int(local[int(map_row[1])])
    c2 = int(local[int(map_row[2])])
    c3 = int(local[int(map_row[3])])
    c4 = int(local[int(map_row[4])])
    c5 = int(local[int(map_row[5])])
    c6 = int(local[int(map_row[6])])
    c7 = int(local[int(map_row[7])])

    out_row[:] = (
        weight000 * point_values[c0]
        + weight100 * point_values[c1]
        + weight010 * point_values[c2]
        + weight110 * point_values[c3]
        + weight001 * point_values[c4]
        + weight101 * point_values[c5]
        + weight011 * point_values[c6]
        + weight111 * point_values[c7]
    )


@njit(cache=True)
def _trilinear_from_cell_rpa(
    out_row: np.ndarray,
    cell_id: int,
    r: float,
    polar: float,
    azimuth: float,
    cell_bounds: np.ndarray,
    interp_state: SphericalInterpKernelState,
) -> None:
    """Write one interpolated value row for one spherical query in one leaf cell using flat point values."""
    cid = int(cell_id)

    frac_r = (r - cell_bounds[cid, AXIS0, START]) / max(
        cell_bounds[cid, AXIS0, WIDTH], _TINY
    )
    if frac_r < 0.0:
        frac_r = 0.0
    elif frac_r > 1.0:
        frac_r = 1.0

    frac_p = (polar - cell_bounds[cid, AXIS1, START]) / max(
        cell_bounds[cid, AXIS1, WIDTH], _TINY
    )
    if frac_p < 0.0:
        frac_p = 0.0
    elif frac_p > 1.0:
        frac_p = 1.0

    a_rel = (azimuth - cell_bounds[cid, AXIS2, START]) % _TWO_PI
    if interp_state.cell_a_tiny[cid]:
        frac_a = 0.0
    else:
        if not interp_state.cell_a_full[cid]:
            width = cell_bounds[cid, AXIS2, WIDTH]
            if a_rel < 0.0:
                a_rel = 0.0
            elif a_rel > width:
                a_rel = width
        frac_a = a_rel / max(cell_bounds[cid, AXIS2, WIDTH], _TINY)
        if frac_a < 0.0:
            frac_a = 0.0
        elif frac_a > 1.0:
            frac_a = 1.0

    _write_trilinear_row(
        out_row,
        cell_id,
        frac_r,
        frac_p,
        frac_a,
        interp_state.corners,
        interp_state.bin_to_corner,
        interp_state.point_values,
    )


@njit(cache=True)
def _trilinear_from_cell(
    out_row: np.ndarray,
    cell_id: int,
    x: float,
    y: float,
    z: float,
    cell_bounds: np.ndarray,
    interp_state: CartesianInterpKernelState,
) -> None:
    """Write one Cartesian trilinear interpolation result row for one leaf cell using flat point values."""
    cid = int(cell_id)
    frac_x = (x - cell_bounds[cid, AXIS0, START]) / max(
        cell_bounds[cid, AXIS0, WIDTH], _TINY
    )
    if frac_x < 0.0:
        frac_x = 0.0
    elif frac_x > 1.0:
        frac_x = 1.0
    frac_y = (y - cell_bounds[cid, AXIS1, START]) / max(
        cell_bounds[cid, AXIS1, WIDTH], _TINY
    )
    if frac_y < 0.0:
        frac_y = 0.0
    elif frac_y > 1.0:
        frac_y = 1.0
    frac_z = (z - cell_bounds[cid, AXIS2, START]) / max(
        cell_bounds[cid, AXIS2, WIDTH], _TINY
    )
    if frac_z < 0.0:
        frac_z = 0.0
    elif frac_z > 1.0:
        frac_z = 1.0

    _write_trilinear_row(
        out_row,
        cell_id,
        frac_x,
        frac_y,
        frac_z,
        interp_state.corners,
        interp_state.bin_to_corner,
        interp_state.point_values,
    )


@njit(cache=True, parallel=True)
def _interp_from_cell_ids_rpa(
    queries_rpa: np.ndarray,
    cell_ids: np.ndarray,
    fill_values: np.ndarray,
    cell_bounds: np.ndarray,
    interp_state: SphericalInterpKernelState,
) -> np.ndarray:
    """Evaluate spherical-tree interpolation for spherical queries with known leaf cell ids using flat point values."""
    n_query = queries_rpa.shape[0]
    ncomp = interp_state.point_values.shape[1]
    out = np.empty((n_query, ncomp), dtype=interp_state.point_values.dtype)
    for i in prange(n_query):
        out[i, :] = fill_values
        cid = int(cell_ids[i])
        if cid < 0:
            continue
        _trilinear_from_cell_rpa(
            out[i],
            cid,
            queries_rpa[i, 0],
            queries_rpa[i, 1],
            queries_rpa[i, 2] % _TWO_PI,
            cell_bounds,
            interp_state,
        )
    return out


@njit(cache=True, parallel=True)
def _interp_from_cell_ids_xyz_cartesian(
    queries_xyz: np.ndarray,
    cell_ids: np.ndarray,
    fill_values: np.ndarray,
    cell_bounds: np.ndarray,
    interp_state: CartesianInterpKernelState,
) -> np.ndarray:
    """Evaluate Cartesian-tree interpolation for Cartesian queries with known leaf cell ids using flat point values.

    Assumes the Cartesian backend cell model (axis-aligned per-cell bounds).
    """
    n_query = queries_xyz.shape[0]
    ncomp = interp_state.point_values.shape[1]
    out = np.empty((n_query, ncomp), dtype=interp_state.point_values.dtype)
    for i in prange(n_query):
        out[i, :] = fill_values
        cid = int(cell_ids[i])
        if cid < 0:
            continue
        _trilinear_from_cell(
            out[i],
            cid,
            queries_xyz[i, 0],
            queries_xyz[i, 1],
            queries_xyz[i, 2],
            cell_bounds,
            interp_state,
        )
    return out

class OctreeInterpolator:
    """LinearNDInterpolator-like callable built on octree leaf lookup.

    Query algorithm:
    - Find containing leaf cell with octree lookup.
    - Convert query to backend-local coordinates:
      spherical uses `(r, polar, azimuth)`;
      Cartesian uses normalized `(x, y, z)` from per-cell axis-aligned min/max.
    - Evaluate trilinear interpolation from the 8 corner nodes of that cell.

    Note:
    - For ``tree_coord="xyz"``, Cartesian cell geometry is treated as
      axis-aligned boxes; skewed/non-axis-aligned cells are not modeled exactly.
    """

    def __init__(
        self,
        tree: Octree,
        values: list[str] | np.ndarray | None,
        *,
        fill_value: float | np.ndarray = np.nan,
    ) -> None:
        """Create an interpolator from one built tree and point values."""
        if not isinstance(tree, Octree):
            raise TypeError("OctreeInterpolator requires a built Octree as its first argument.")
        self.tree = tree
        self._ds = tree.ds
        self.fill_value = fill_value

        logger.debug(
            "Initializing OctreeInterpolator: points=%d, cells=%d",
            int(self._ds.points.shape[0]),
            int(self.tree._corners.shape[0]),
        )
        self.value_names: tuple[str, ...] = ()
        self._point_values_2d, self._value_shape_tail = self._coerce_point_values(values)
        self._tree_coord = str(self.tree.tree_coord)
        if self._tree_coord not in {"xyz", "rpa"}:
            raise NotImplementedError(f"Unsupported tree_coord '{self._tree_coord}' for interpolation.")
        self.prepare_kernel_cache()
        self.warmup_kernels()
        logger.info(
            "Interpolator ready: uniform=%s, max_level=%d, leaf_shape=%s",
            self.tree.is_uniform,
            int(self.tree.max_level),
            tuple(self.tree.leaf_shape),
        )

    def _coerce_point_values(self, values: list[str] | np.ndarray | None) -> tuple[np.ndarray, tuple[int, ...]]:
        """Resolve requested fields into one flat `(n_points, n_components)` array plus trailing shape."""
        n_points = int(self._ds.points.shape[0])
        if values is None:
            names: list[str]
            names = [str(name) for name in self._ds.variables]
            if len(names) == 0:
                raise ValueError("Dataset has no variables; cannot interpolate values=None.")
            arrays: list[np.ndarray] = []
            for name in names:
                arr_name = np.array(self._ds[name])
                if arr_name.shape[0] != n_points:
                    logger.error(
                        "Value size mismatch for field %s: values=%d, n_points=%d",
                        name,
                        int(arr_name.shape[0]),
                        n_points,
                    )
                    raise ValueError(f"values length {arr_name.shape[0]} does not match required n_points={n_points}.")
                arrays.append(arr_name)
            self.value_names = tuple(names)
            if len(arrays) == 1:
                arr = arrays[0]
                logger.debug("Using field %s with shape=%s", names[0], tuple(arr.shape))
                return np.array(arr.reshape(n_points, -1), dtype=np.float64, order="C"), tuple(arr.shape[1:])
            merged = np.concatenate([arr.reshape(n_points, -1) for arr in arrays], axis=1)
            logger.debug("Using %d fields with merged shape=%s", len(names), tuple(merged.shape))
            return np.array(merged, dtype=np.float64, order="C"), tuple(merged.shape[1:])

        if isinstance(values, str):
            raise ValueError("values must be None, array-like, or list[str]; single-string values are not supported.")
        if isinstance(values, list):
            if len(values) == 0 or not all(isinstance(v, str) for v in values):
                raise ValueError("values must be None, array-like, or a non-empty list[str] of field names.")
            names = [str(name) for name in values]
            arrays: list[np.ndarray] = []
            for name in names:
                arr_name = np.array(self._ds[name])
                if arr_name.shape[0] != n_points:
                    logger.error(
                        "Value size mismatch for field %s: values=%d, n_points=%d",
                        name,
                        int(arr_name.shape[0]),
                        n_points,
                    )
                    raise ValueError(f"values length {arr_name.shape[0]} does not match required n_points={n_points}.")
                arrays.append(arr_name)
            self.value_names = tuple(names)
            if len(arrays) == 1:
                arr = arrays[0]
                logger.debug("Using field %s with shape=%s", names[0], tuple(arr.shape))
                return np.array(arr.reshape(n_points, -1), dtype=np.float64, order="C"), tuple(arr.shape[1:])
            merged = np.concatenate([arr.reshape(n_points, -1) for arr in arrays], axis=1)
            logger.debug("Using %d fields with merged shape=%s", len(names), tuple(merged.shape))
            return np.array(merged, dtype=np.float64, order="C"), tuple(merged.shape[1:])

        arr = np.asarray(values)
        if arr.shape[0] != n_points:
            raise ValueError(f"values length {arr.shape[0]} does not match required n_points={n_points}.")
        self.value_names = ()
        logger.debug("Using explicit value array with shape=%s", tuple(arr.shape))
        return np.array(arr.reshape(n_points, -1), dtype=np.float64, order="C"), tuple(arr.shape[1:])

    def prepare_kernel_cache(self) -> None:
        """Pack per-point values and bind them to tree-owned interpolation geometry."""
        self._n_value_components = int(self._point_values_2d.shape[1])
        self._interp_state = self.tree._interp_state_from_values(self._point_values_2d)

    def _fill_value_vector(self) -> np.ndarray:
        """Convert `fill_value` to one vector of length `n_components`."""
        ncomp = int(self._n_value_components)
        if np.isscalar(self.fill_value):
            return np.full(ncomp, float(self.fill_value), dtype=np.float64)

        fill = np.array(self.fill_value, dtype=np.float64).reshape(-1)
        if fill.size == 1:
            return np.full(ncomp, float(fill[0]), dtype=np.float64)
        if fill.size != ncomp:
            raise ValueError(
                f"fill_value has {fill.size} entries but interpolated values require {ncomp} components."
            )
        return fill

    def warmup_kernels(self) -> None:
        """Trigger JIT compilation ahead of first real query."""
        if int(self._ds.points.shape[0]) == 0:
            q_xyz = np.zeros((1, 3), dtype=np.float64)
        else:
            q_xyz = np.column_stack(tuple(np.asarray(self._ds[name][:1], dtype=np.float64) for name in XYZ_VARS))
        fill = self._fill_value_vector()
        if self._tree_coord == "rpa":
            q_rpa, cell_ids_xyz = self.tree._lookup_points_local(q_xyz, coord="xyz")
            _q_rpa_direct, cell_ids_rpa = self.tree._lookup_points_local(q_rpa, coord="rpa")
            _interp_from_cell_ids_rpa(q_rpa, cell_ids_xyz, fill, self.tree._cell_bounds, self._interp_state)
            _interp_from_cell_ids_rpa(q_rpa, cell_ids_rpa, fill, self.tree._cell_bounds, self._interp_state)
            return
        _q_local, cell_ids_xyz = self.tree._lookup_points_local(q_xyz, coord="xyz")
        _interp_from_cell_ids_xyz_cartesian(
            q_xyz,
            cell_ids_xyz,
            fill,
            self.tree._cell_bounds,
            self._interp_state,
        )

    @staticmethod
    def prepare_queries(*args) -> tuple[np.ndarray, tuple[int, ...]]:
        """Normalize query inputs to `(N, 3)` plus broadcast output shape.

        Supports:
        - `xi` with shape `(..., 3)`
        - tuple/list of 3 broadcastable arrays
        - three separate coordinate arrays.
        Returns `(q, shape)` where `q` has shape `(N, 3)` and `shape` is the
        broadcasted leading output shape.
        """
        if len(args) == 1:
            xi = args[0]
            if isinstance(xi, tuple):
                if len(xi) != 3:
                    raise ValueError("Tuple input must have exactly 3 arrays.")
                a0, a1, a2 = np.broadcast_arrays(*[np.array(v, dtype=float) for v in xi])
                shape = a0.shape
                q = np.stack((a0, a1, a2), axis=-1).reshape(-1, 3)
                return q, shape

            arr = np.array(xi, dtype=float)
            if arr.ndim == 1:
                if arr.size != 3:
                    raise ValueError("1D xi must have length 3.")
                return arr.reshape(1, 3), ()
            if arr.shape[-1] != 3:
                raise ValueError("xi must have shape (..., 3).")
            return arr.reshape(-1, 3), arr.shape[:-1]

        if len(args) == 3:
            a0, a1, a2 = np.broadcast_arrays(*[np.array(v, dtype=float) for v in args])
            shape = a0.shape
            q = np.stack((a0, a1, a2), axis=-1).reshape(-1, 3)
            return q, shape

        raise ValueError("Call with xi or with x1, x2, x3.")

    def __call__(
        self,
        *args,
        query_coord: Literal["xyz", "rpa"] = "xyz",
        return_cell_ids: bool = False,
        log_outside_domain: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Evaluate interpolation at query points.

        For each query:
        - resolve containing cell via octree lookup,
        - convert to local spherical coordinates,
        - evaluate cached trilinear interpolation.

        Returns values reshaped to the query broadcast shape.
        If `return_cell_ids=True`, also returns the resolved cell ids.
        """
        qs = str(query_coord)
        if qs not in {"xyz", "rpa"}:
            logger.error("Invalid query_coord=%s in call", qs)
            raise ValueError("query_coord must be 'xyz' or 'rpa'.")
        if self._tree_coord == "xyz" and qs == "rpa":
            logger.error("query_coord='rpa' is not supported for Cartesian trees.")
            raise ValueError("query_coord='rpa' is only supported for tree_coord='rpa'.")

        debug_timing = logger.isEnabledFor(logging.DEBUG)
        t0_total = perf_counter() if debug_timing else 0.0

        q, shape = self.prepare_queries(*args)
        t_after_prepare = perf_counter() if debug_timing else 0.0
        q_array = np.array(q, dtype=np.float64, order="C")
        t_after_convert = perf_counter() if debug_timing else 0.0
        n = q_array.shape[0]
        trailing = self._value_shape_tail
        logger.debug("Interpolating %d query points in %s space", int(n), qs)
        fill = self._fill_value_vector()
        t_after_fill = perf_counter() if debug_timing else 0.0

        if self._tree_coord == "rpa":
            q_local, cell_ids = self.tree._lookup_points_local(q_array, coord=qs)
            if debug_timing:
                logger.debug("Interpolation kernel mode: compiled-rpa")
            out2d = _interp_from_cell_ids_rpa(q_local, cell_ids, fill, self.tree._cell_bounds, self._interp_state)
        else:
            _q_local, cell_ids = self.tree._lookup_points_local(q_array, coord="xyz")
            if debug_timing:
                logger.debug("Interpolation kernel mode: compiled-xyz")
            out2d = _interp_from_cell_ids_xyz_cartesian(
                q_array,
                cell_ids,
                fill,
                self.tree._cell_bounds,
                self._interp_state,
            )
        t_after_kernel = perf_counter() if debug_timing else 0.0

        misses = int(np.count_nonzero(cell_ids < 0))
        if log_outside_domain:
            if misses == n and n > 0:
                logger.warning("All query points were outside interpolation domain (%d/%d misses).", misses, n)
            elif misses > 0:
                logger.info("Some query points were outside interpolation domain (%d/%d misses).", misses, n)

        out = out2d.reshape((n,) + trailing).reshape(shape + trailing)
        t_after_post = perf_counter() if debug_timing else 0.0
        if debug_timing:
            prep_s = t_after_prepare - t0_total
            convert_s = t_after_convert - t_after_prepare
            fill_s = t_after_fill - t_after_convert
            kernel_s = t_after_kernel - t_after_fill
            post_s = t_after_post - t_after_kernel
            total_s = t_after_post - t0_total
            logger.debug(
                (
                    "Interpolation timings: "
                    f"n={int(n)} qs={qs} prep={prep_s:.6f}s convert={convert_s:.6f}s "
                    f"fill={fill_s:.6f}s kernel={kernel_s:.6f}s post={post_s:.6f}s "
                    f"total={total_s:.6f}s "
                    f"kernel_share={((kernel_s / total_s) if total_s > 0.0 else float('nan')):.3f}"
                )
            )
        if return_cell_ids:
            return out, cell_ids.reshape(shape)
        return out

    def trilinear_corner_count(self, cell_id: int) -> int:
        """Return number of unique mapped corners used for one cell interpolation map."""
        cid = int(cell_id)
        if cid < 0 or cid >= int(self.tree.cell_count) or int(self.tree.cell_levels[cid]) < 0:
            raise ValueError(f"Invalid cell_id {cid}.")
        return int(np.unique(self.tree._interp_bin_to_corner[cid]).size)

    def cell_has_full_trilinear_corner_map(self, cell_id: int) -> bool:
        """Return whether one cell maps to all 8 logical trilinear corners."""
        return self.trilinear_corner_count(int(cell_id)) == 8

    @property
    def n_value_components(self) -> int:
        """Return flattened component count of the interpolated output."""
        return int(self._n_value_components)

    @property
    def corners(self) -> np.ndarray:
        """Return cell-to-node corner connectivity used by interpolation."""
        return self.tree._corners

    @property
    def point_values(self) -> np.ndarray:
        """Return per-node interpolation values in original component shape."""
        return self._point_values_2d.reshape((int(self._point_values_2d.shape[0]),) + self._value_shape_tail)

    def set_fields(
        self,
        values: list[str] | None,
        *,
        fill_value: float | np.ndarray | None = None,
        warmup: bool = False,
    ) -> None:
        """Experimental: switch interpolated fields without rebuilding geometry.

        This reuses the existing tree/lookup and only repacks value arrays and
        interpolation kernel state.
        """
        if fill_value is not None:
            self.fill_value = fill_value
        self._point_values_2d, self._value_shape_tail = self._coerce_point_values(values)
        self.prepare_kernel_cache()
        # Fail fast when a vector fill value no longer matches component count.
        _ = self._fill_value_vector()
        if warmup:
            self.warmup_kernels()

    def __str__(self) -> str:
        """Return a compact human-readable interpolator description."""
        n_points = int(self._ds.points.shape[0]) if hasattr(self._ds, "points") else -1
        n_cells = int(self.tree._corners.shape[0])
        n_fields = int(len(self.value_names))
        if n_fields == 0:
            field_text = "<none>"
        elif n_fields <= 3:
            field_text = ", ".join(self.value_names)
        else:
            field_text = f"{', '.join(self.value_names[:3])}, ..."
        return (
            "OctreeInterpolator("
            f"tree_coord={self._tree_coord}, "
            f"fields={n_fields}[{field_text}], "
            f"n_points={n_points}, "
            f"n_cells={n_cells}, "
            f"n_components={int(self._n_value_components)}"
            ")"
        )

logger.addHandler(logging.NullHandler())

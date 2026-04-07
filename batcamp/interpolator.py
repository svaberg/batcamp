#!/usr/bin/env python3
"""Octree interpolator and interpolation kernels."""

from __future__ import annotations

import logging
import math
from typing import Literal

from numba import njit
from numba import prange
import numpy as np

from .octree import AXIS0
from .octree import AXIS1
from .octree import AXIS2
from .octree import Octree
from .octree import START
from .octree import WIDTH

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi
_TINY = np.finfo(np.float64).tiny
_XYZ_TRILINEAR_BITS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int8,
)
# BATSRUS spherical cells arrive in a fixed corner order that differs from the
# plain `(axis0, axis1, axis2)` low/high bit order used by Cartesian hexes.
_RPA_TRILINEAR_BITS = np.array(
    [
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
    ],
    dtype=np.int8,
)


@njit(cache=True)
def _accumulate_trilinear(
    out_row: np.ndarray,
    cell_id: int,
    frac_axis0: float,
    frac_axis1: float,
    frac_axis2: float,
    corners: np.ndarray,
    point_values: np.ndarray,
    bits: np.ndarray,
) -> None:
    """Write one trilinear interpolation row from one 8-corner low/high bit table."""
    cell_id = int(cell_id)
    frac_axis0_lo = 1.0 - frac_axis0
    frac_axis1_lo = 1.0 - frac_axis1
    frac_axis2_lo = 1.0 - frac_axis2
    cell_corner_ids = corners[cell_id]
    out_row[:] = 0.0
    for corner_ord in range(8):
        bit0 = bits[corner_ord, AXIS0]
        bit1 = bits[corner_ord, AXIS1]
        bit2 = bits[corner_ord, AXIS2]
        weight = frac_axis0 if bit0 else frac_axis0_lo
        weight *= frac_axis1 if bit1 else frac_axis1_lo
        weight *= frac_axis2 if bit2 else frac_axis2_lo
        corner_point_id = int(cell_corner_ids[corner_ord])
        out_row[:] += weight * point_values[corner_point_id]


@njit(cache=True)
def _interp_cell_rpa(
    out_row: np.ndarray,
    cell_id: int,
    r: float,
    polar: float,
    azimuth: float,
    cell_bounds: np.ndarray,
    corners: np.ndarray,
    point_values: np.ndarray,
) -> None:
    """Write one interpolated value row for one spherical query in one leaf cell using flat point values."""
    cell_id = int(cell_id)

    frac_r = (r - cell_bounds[cell_id, AXIS0, START]) / max(
        cell_bounds[cell_id, AXIS0, WIDTH], _TINY
    )
    if frac_r < 0.0:
        frac_r = 0.0
    elif frac_r > 1.0:
        frac_r = 1.0

    frac_p = (polar - cell_bounds[cell_id, AXIS1, START]) / max(
        cell_bounds[cell_id, AXIS1, WIDTH], _TINY
    )
    if frac_p < 0.0:
        frac_p = 0.0
    elif frac_p > 1.0:
        frac_p = 1.0

    cell_a_width = cell_bounds[cell_id, AXIS2, WIDTH]
    a_rel = (azimuth - cell_bounds[cell_id, AXIS2, START]) % _TWO_PI
    if cell_a_width <= _TINY:
        frac_a = 0.0
    else:
        if cell_a_width < (_TWO_PI - 1.0e-10):
            if a_rel < 0.0:
                a_rel = 0.0
            elif a_rel > cell_a_width:
                a_rel = cell_a_width
        frac_a = a_rel / max(cell_a_width, _TINY)
        if frac_a < 0.0:
            frac_a = 0.0
        elif frac_a > 1.0:
            frac_a = 1.0

    _accumulate_trilinear(
        out_row,
        cell_id,
        frac_r,
        frac_p,
        frac_a,
        corners,
        point_values,
        _RPA_TRILINEAR_BITS,
    )


@njit(cache=True)
def _interp_cell_xyz(
    out_row: np.ndarray,
    cell_id: int,
    x: float,
    y: float,
    z: float,
    cell_bounds: np.ndarray,
    corners: np.ndarray,
    point_values: np.ndarray,
) -> None:
    """Write one Cartesian trilinear interpolation result row for one leaf cell using flat point values."""
    cell_id = int(cell_id)
    frac_x = (x - cell_bounds[cell_id, AXIS0, START]) / max(
        cell_bounds[cell_id, AXIS0, WIDTH], _TINY
    )
    if frac_x < 0.0:
        frac_x = 0.0
    elif frac_x > 1.0:
        frac_x = 1.0
    frac_y = (y - cell_bounds[cell_id, AXIS1, START]) / max(
        cell_bounds[cell_id, AXIS1, WIDTH], _TINY
    )
    if frac_y < 0.0:
        frac_y = 0.0
    elif frac_y > 1.0:
        frac_y = 1.0
    frac_z = (z - cell_bounds[cell_id, AXIS2, START]) / max(
        cell_bounds[cell_id, AXIS2, WIDTH], _TINY
    )
    if frac_z < 0.0:
        frac_z = 0.0
    elif frac_z > 1.0:
        frac_z = 1.0

    _accumulate_trilinear(
        out_row,
        cell_id,
        frac_x,
        frac_y,
        frac_z,
        corners,
        point_values,
        _XYZ_TRILINEAR_BITS,
    )


@njit(cache=True, parallel=True)
def _interp_cells_rpa(
    queries_rpa: np.ndarray,
    cell_ids: np.ndarray,
    fill_values: np.ndarray,
    cell_bounds: np.ndarray,
    corners: np.ndarray,
    point_values: np.ndarray,
) -> np.ndarray:
    """Evaluate spherical-tree interpolation for spherical queries with known leaf cell ids using flat point values."""
    n_query = queries_rpa.shape[0]
    ncomp = point_values.shape[1]
    out = np.empty((n_query, ncomp), dtype=point_values.dtype)
    for i in prange(n_query):
        out[i, :] = fill_values
        cell_id = int(cell_ids[i])
        if cell_id < 0:
            continue
        _interp_cell_rpa(
            out[i],
            cell_id,
            queries_rpa[i, 0],
            queries_rpa[i, 1],
            queries_rpa[i, 2] % _TWO_PI,
            cell_bounds,
            corners,
            point_values,
        )
    return out


@njit(cache=True, parallel=True)
def _interp_cells_xyz(
    queries_xyz: np.ndarray,
    cell_ids: np.ndarray,
    fill_values: np.ndarray,
    cell_bounds: np.ndarray,
    corners: np.ndarray,
    point_values: np.ndarray,
) -> np.ndarray:
    """Evaluate Cartesian-tree interpolation for Cartesian queries with known leaf cell ids using flat point values.

    Assumes the Cartesian backend cell model (axis-aligned per-cell bounds).
    """
    n_query = queries_xyz.shape[0]
    ncomp = point_values.shape[1]
    out = np.empty((n_query, ncomp), dtype=point_values.dtype)
    for i in prange(n_query):
        out[i, :] = fill_values
        cell_id = int(cell_ids[i])
        if cell_id < 0:
            continue
        _interp_cell_xyz(
            out[i],
            cell_id,
            queries_xyz[i, 0],
            queries_xyz[i, 1],
            queries_xyz[i, 2],
            cell_bounds,
            corners,
            point_values,
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
        values: np.ndarray,
        *,
        fill_value: float | np.ndarray = np.nan,
    ) -> None:
        """Create an interpolator from one built tree and point values."""
        if not isinstance(tree, Octree):
            raise TypeError("OctreeInterpolator requires a built Octree as its first argument.")
        self.tree = tree
        self.fill_value = fill_value
        self._point_values_2d, self._value_shape_tail = self._flatten_point_values(values)
        if self.tree.tree_coord not in {"xyz", "rpa"}:
            raise NotImplementedError(f"Unsupported tree_coord '{self.tree.tree_coord}' for interpolation.")

    def _flatten_point_values(self, values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        """Flatten one `(n_points, ...)` value array for interpolation kernels."""
        if values is None or isinstance(values, str) or isinstance(values, list):
            raise ValueError("values must be an array aligned with tree point ids.")
        arr = np.asarray(values)
        n_points_required = int(np.max(self.tree.corners)) + 1
        arr_length = 0 if arr.ndim == 0 else int(arr.shape[0])
        if arr_length < n_points_required:
            raise ValueError(
                f"values length {arr_length} does not cover required point ids 0..{n_points_required - 1}."
            )
        return np.array(arr.reshape(arr_length, -1), dtype=np.float64, order="C"), tuple(arr.shape[1:])

    @staticmethod
    def _normalize_queries(*args) -> tuple[np.ndarray, tuple[int, ...]]:
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
                args = xi
            else:
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
            raise ValueError("query_coord must be 'xyz' or 'rpa'.")
        if self.tree.tree_coord == "xyz" and qs == "rpa":
            raise ValueError("query_coord='rpa' is only supported for tree_coord='rpa'.")

        q, shape = self._normalize_queries(*args)
        q_array = np.array(q, dtype=np.float64, order="C")
        n = q_array.shape[0]
        trailing = self._value_shape_tail
        n_components = int(self._point_values_2d.shape[1])
        if np.isscalar(self.fill_value):
            fill = np.full(n_components, float(self.fill_value), dtype=np.float64)
        else:
            fill = np.array(self.fill_value, dtype=np.float64).reshape(-1)
            if fill.size == 1:
                fill = np.full(n_components, float(fill[0]), dtype=np.float64)
            elif fill.size != n_components:
                raise ValueError(
                    f"fill_value has {fill.size} entries but interpolated values require {n_components} components."
                )

        cell_ids = self.tree.lookup_points(q_array, coord=qs).reshape(-1)
        if self.tree.tree_coord == "rpa":
            if qs == "rpa":
                kernel_queries = q_array
            else:
                from .spherical import xyz_arrays_to_rpa

                kernel_queries = np.column_stack(xyz_arrays_to_rpa(q_array[:, 0], q_array[:, 1], q_array[:, 2]))
            kernel = _interp_cells_rpa
        else:
            kernel_queries = q_array
            kernel = _interp_cells_xyz
        out2d = kernel(
            kernel_queries,
            cell_ids,
            fill,
            self.tree.cell_bounds,
            self.tree.corners,
            self._point_values_2d,
        )

        misses = int(np.count_nonzero(cell_ids < 0))
        if log_outside_domain:
            if misses == n and n > 0:
                logger.warning("All query points were outside interpolation domain (%d/%d misses).", misses, n)
            elif misses > 0:
                logger.info("Some query points were outside interpolation domain (%d/%d misses).", misses, n)

        out = out2d.reshape((n,) + trailing).reshape(shape + trailing)
        if return_cell_ids:
            return out, cell_ids.reshape(shape)
        return out

    def __str__(self) -> str:
        """Return a compact human-readable interpolator summary."""
        return (
            "OctreeInterpolator("
            f"tree_coord={self.tree.tree_coord}, "
            f"n_points={int(self._point_values_2d.shape[0])}, "
            f"n_cells={int(self.tree.corners.shape[0])}, "
            f"n_components={int(self._point_values_2d.shape[1])}"
            ")"
        )

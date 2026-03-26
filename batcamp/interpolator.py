#!/usr/bin/env python3
"""Octree interpolator and interpolation kernels."""

from __future__ import annotations

import logging
import math
from typing import Literal

from numba import njit
from numba import prange
import numpy as np

from .constants import XYZ_VARS
from .octree import AXIS0
from .octree import AXIS1
from .octree import AXIS2
from .octree import Octree
from .octree import START
from .octree import WIDTH

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi
_TINY = np.finfo(np.float64).tiny


@njit(cache=True)
def _accumulate_trilinear(
    out_row: np.ndarray,
    cell_id: int,
    frac_axis0: float,
    frac_axis1: float,
    frac_axis2: float,
    corners: np.ndarray,
    point_values: np.ndarray,
) -> None:
    """Write one trilinear interpolation row for one cell from normalized local coordinates."""
    cell_id = int(cell_id)
    frac_axis0_lo = 1.0 - frac_axis0
    frac_axis1_lo = 1.0 - frac_axis1
    frac_axis2_lo = 1.0 - frac_axis2
    cell_corner_ids = corners[cell_id]
    out_row[:] = 0.0
    for logical_corner in range(8):
        weight = frac_axis0 if (logical_corner & 1) else frac_axis0_lo
        weight *= frac_axis1 if (logical_corner & 2) else frac_axis1_lo
        weight *= frac_axis2 if (logical_corner & 4) else frac_axis2_lo
        corner_point_id = int(cell_corner_ids[logical_corner])
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
        self.value_names: tuple[str, ...] = ()
        self._point_values_2d, self._value_shape_tail = self._flatten_point_values(values)
        self._tree_coord = self.tree.tree_coord
        if self._tree_coord not in {"xyz", "rpa"}:
            raise NotImplementedError(f"Unsupported tree_coord '{self._tree_coord}' for interpolation.")
        self._n_value_components = int(self._point_values_2d.shape[1])
        self.warmup()

    def _flatten_point_values(self, values: list[str] | np.ndarray | None) -> tuple[np.ndarray, tuple[int, ...]]:
        """Resolve requested fields into one flat `(n_points, n_components)` array plus trailing shape."""
        n_points = int(self._ds.points.shape[0])
        if values is None:
            names = [str(name) for name in self._ds.variables]
            if len(names) == 0:
                raise ValueError("Dataset has no variables; cannot interpolate values=None.")
        elif isinstance(values, str):
            raise ValueError("values must be None, array-like, or list[str]; single-string values are not supported.")
        elif isinstance(values, list):
            if len(values) == 0 or not all(isinstance(v, str) for v in values):
                raise ValueError("values must be None, array-like, or a non-empty list[str] of field names.")
            names = [str(name) for name in values]
        else:
            arr = np.asarray(values)
            if arr.shape[0] != n_points:
                raise ValueError(f"values length {arr.shape[0]} does not match required n_points={n_points}.")
            self.value_names = ()
            return np.array(arr.reshape(n_points, -1), dtype=np.float64, order="C"), tuple(arr.shape[1:])

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
            return np.array(arr.reshape(n_points, -1), dtype=np.float64, order="C"), tuple(arr.shape[1:])
        merged = np.concatenate([arr.reshape(n_points, -1) for arr in arrays], axis=1)
        return np.array(merged, dtype=np.float64, order="C"), tuple(merged.shape[1:])

    def _fill_values(self) -> np.ndarray:
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

    def warmup(self) -> None:
        """Trigger JIT compilation ahead of first real query."""
        if int(self._ds.points.shape[0]) == 0:
            q_xyz = np.zeros((1, 3), dtype=np.float64)
        else:
            q_xyz = np.column_stack(tuple(np.asarray(self._ds[name][:1], dtype=np.float64) for name in XYZ_VARS))
        if self._tree_coord == "rpa":
            from .spherical import _xyz_arrays_to_rpa

            q_rpa = np.column_stack(_xyz_arrays_to_rpa(q_xyz[:, 0], q_xyz[:, 1], q_xyz[:, 2]))
            self(q_xyz, query_coord="xyz", log_outside_domain=False)
            self(q_rpa, query_coord="rpa", log_outside_domain=False)
            return
        self(q_xyz, query_coord="xyz", log_outside_domain=False)

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

        q, shape = self._normalize_queries(*args)
        q_array = np.array(q, dtype=np.float64, order="C")
        n = q_array.shape[0]
        trailing = self._value_shape_tail
        fill = self._fill_values()

        cell_ids = self.tree.lookup_points(q_array, coord=qs).reshape(-1)
        if self._tree_coord == "rpa":
            if qs == "rpa":
                kernel_queries = q_array
            else:
                from .spherical import _xyz_arrays_to_rpa

                kernel_queries = np.column_stack(_xyz_arrays_to_rpa(q_array[:, 0], q_array[:, 1], q_array[:, 2]))
            out2d = _interp_cells_rpa(
                kernel_queries,
                cell_ids,
                fill,
                self.tree.cell_bounds,
                self.tree.corners,
                self._point_values_2d,
            )
        else:
            kernel_queries = q_array
            out2d = _interp_cells_xyz(
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

    @property
    def n_value_components(self) -> int:
        """Return flattened component count of the interpolated output."""
        return int(self._n_value_components)

    def __str__(self) -> str:
        """Return a compact human-readable interpolator description."""
        n_points = int(self._ds.points.shape[0])
        n_cells = int(self.tree.corners.shape[0])
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

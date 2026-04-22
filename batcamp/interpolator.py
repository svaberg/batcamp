#!/usr/bin/env python3
"""Public octree interpolator wrapper."""

from __future__ import annotations

import logging
import math
import time
from typing import Literal

import numpy as np
from . import interpolator_cartesian
from . import interpolator_spherical
from .octree import TREE_COORD_AXIS0
from .octree import TREE_COORD_AXIS1
from .octree import TREE_COORD_AXIS2
from .octree import Octree

logger = logging.getLogger(__name__)

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
        logger.info("OctreeInterpolator.__init__: coord=%s", tree.tree_coord)
        t0 = time.perf_counter()
        self.tree = tree
        self._interpolator_module = interpolator_cartesian if tree.tree_coord == "xyz" else interpolator_spherical
        self.fill_value = fill_value
        logger.info("_flatten_point_values...")
        t_flat = time.perf_counter()
        self._point_values_2d, self._value_shape_tail = self._flatten_point_values(values)
        logger.info("_flatten_point_values complete in %.2fs", float(time.perf_counter() - t_flat))
        if self.tree.tree_coord not in {"xyz", "rpa"}:
            raise NotImplementedError(f"Unsupported tree_coord '{self.tree.tree_coord}' for interpolation.")
        logger.info("OctreeInterpolator.__init__ complete in %.2fs", float(time.perf_counter() - t0))

    @property
    def n_components(self) -> int:
        """Return the flat component count carried by each interpolation value."""
        return int(self._point_values_2d.shape[1])

    @property
    def point_values_2d(self) -> np.ndarray:
        """Return flat `(n_points, n_components)` point values for interpolation kernels."""
        return self._point_values_2d

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Return the trailing shape of one interpolated value."""
        return self._value_shape_tail

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
        Returns `(queries_flat, shape)` where `queries_flat` has shape `(N, 3)` and `shape` is the
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
            queries_flat = np.stack((a0, a1, a2), axis=-1).reshape(-1, 3)
            return queries_flat, shape

        raise ValueError("Call with xi or with x1, x2, x3.")

    def _normalize_fill_value(self, n_components: int) -> np.ndarray:
        """Return one flat fill-value vector matching the interpolated component count."""
        if np.isscalar(self.fill_value):
            return np.full(n_components, float(self.fill_value), dtype=np.float64)

        fill = np.array(self.fill_value, dtype=np.float64).reshape(-1)
        if fill.size == 1:
            return np.full(n_components, float(fill[0]), dtype=np.float64)
        if fill.size != n_components:
            raise ValueError(
                f"fill_value has {fill.size} entries but interpolated values require {n_components} components."
            )
        return fill

    def _kernel_inputs(
        self,
        query_points: np.ndarray,
        query_coord: Literal["xyz", "rpa"],
    ) -> tuple[np.ndarray, callable]:
        """Return kernel-local queries and the matching interpolation kernel."""
        return (
            self._interpolator_module.prepare_queries(query_points, str(query_coord)),
            self._interpolator_module.interp_cells,
        )

    def interp_cells_xyz(self, queries_xyz: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
        """Evaluate Cartesian queries in already-known leaf cells.

        This skips octree ownership lookup for callers that already carry the
        exact Cartesian owner cell ids, such as midpoint rendering over traced
        segments. A final directly accumulative renderer should evaluate or
        integrate segment contributions in place instead of routing through this
        point-sampling helper.
        """
        if self.tree.tree_coord != "xyz":
            raise NotImplementedError("interp_cells_xyz requires tree_coord='xyz'.")

        query_points_flat, shape = self._normalize_queries(queries_xyz)
        query_points = np.array(query_points_flat, dtype=np.float64, order="C")
        cell_id_array = np.array(cell_ids, dtype=np.int64, order="C").reshape(-1)
        n = int(query_points.shape[0])
        if int(cell_id_array.size) != n:
            raise ValueError("cell_ids must match the query count.")

        out2d = self._interpolator_module.interp_cells(
            query_points,
            cell_id_array,
            self._normalize_fill_value(int(self._point_values_2d.shape[1])),
            self.tree.cell_bounds,
            self.tree.corners,
            self._point_values_2d,
        )
        return out2d.reshape(shape + self._value_shape_tail)

    def integrate_box(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Return the exact integral of the backend trilinear interpolant over one native box.

        `lower` and `upper` are native-coordinate box corners with shape `(3,)`
        and componentwise `lower <= upper`.

        For `tree_coord="rpa"`, azimuth is one non-wrapping interval inside
        `[0, 2pi]`.
        """
        lower_array = np.asarray(lower, dtype=np.float64)
        upper_array = np.asarray(upper, dtype=np.float64)
        if lower_array.shape != (3,) or upper_array.shape != (3,):
            raise ValueError("lower and upper must both have shape (3,).")
        if not (np.all(np.isfinite(lower_array)) and np.all(np.isfinite(upper_array))):
            raise ValueError("lower and upper must be finite.")
        if np.any(lower_array > upper_array):
            raise ValueError("lower must be componentwise <= upper.")

        if self.tree.tree_coord == "rpa":
            if lower_array[TREE_COORD_AXIS1] < 0.0 or upper_array[TREE_COORD_AXIS1] > math.pi:
                raise ValueError("For tree_coord='rpa', polar bounds must lie in [0, pi].")
            if lower_array[TREE_COORD_AXIS2] < 0.0 or upper_array[TREE_COORD_AXIS2] > (2.0 * math.pi):
                raise ValueError("For tree_coord='rpa', azimuth bounds must lie in [0, 2pi].")

        out1d = self._interpolator_module.integrate_box(self.tree, self._point_values_2d, lower_array, upper_array)
        if len(self._value_shape_tail) == 0:
            return out1d[0]
        return out1d.reshape(self._value_shape_tail)

    def cell_integrals(self, cell_ids: int | np.ndarray | None = None) -> np.ndarray:
        """Return exact whole-cell integrals of the backend trilinear interpolant.

        When `cell_ids` is omitted, the output is aligned with leaf slots and
        unused slots are filled with `NaN`. Explicit `cell_ids` must refer to
        valid leaf cells and preserve the input shape.

        `tree_coord="xyz"` integrates over Cartesian cell volume.
        `tree_coord="rpa"` integrates over physical spherical volume
        `dV = r^2 sin(theta) dr dtheta dphi`.
        """
        if cell_ids is None:
            out2d = np.full((int(self.tree.cell_count), int(self._point_values_2d.shape[1])), np.nan, dtype=np.float64)
            valid_leaf_ids = np.flatnonzero(self.tree.cell_levels >= 0).astype(np.int64)
            if valid_leaf_ids.size:
                out2d[valid_leaf_ids] = self._interpolator_module.cell_integrals(
                    self.tree,
                    self._point_values_2d,
                    valid_leaf_ids,
                )
            return out2d.reshape((int(self.tree.cell_count),) + self._value_shape_tail)

        leaf_ids = self.tree.normalize_leaf_cell_ids(cell_ids)
        shape = leaf_ids.shape
        flat_leaf_ids = np.array(leaf_ids, dtype=np.int64, order="C").reshape(-1)
        out2d = self._interpolator_module.cell_integrals(self.tree, self._point_values_2d, flat_leaf_ids)
        if leaf_ids.ndim == 0:
            return out2d.reshape((1,) + self._value_shape_tail)[0]
        return out2d.reshape(shape + self._value_shape_tail)

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
        resolved_query_coord = str(query_coord)
        if resolved_query_coord not in {"xyz", "rpa"}:
            raise ValueError("query_coord must be 'xyz' or 'rpa'.")
        if self.tree.tree_coord == "xyz" and resolved_query_coord == "rpa":
            raise ValueError("query_coord='rpa' is only supported for tree_coord='rpa'.")

        query_points_flat, shape = self._normalize_queries(*args)
        query_points = np.array(query_points_flat, dtype=np.float64, order="C")
        n = query_points.shape[0]
        trailing = self._value_shape_tail
        n_components = int(self._point_values_2d.shape[1])
        fill = self._normalize_fill_value(n_components)

        cell_ids = self.tree.lookup_points(query_points, coord=resolved_query_coord).reshape(-1)
        kernel_queries, kernel = self._kernel_inputs(query_points, resolved_query_coord)
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

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
from .octree import CartesianInterpKernelState
from .octree import Octree
from .octree import SphericalInterpKernelState

logger = logging.getLogger(__name__)

_DEFAULT_SEED_CHUNK_SIZE = 1024
_TWO_PI = 2.0 * math.pi


@njit(cache=True)
def _trilinear_from_cell_rpa(
    out_row: np.ndarray,
    node_id: int,
    r: float,
    polar: float,
    azimuth: float,
    interp_state: SphericalInterpKernelState,
) -> None:
    """Write one interpolated value row for one spherical query in one leaf node."""
    cid = int(node_id)

    u = (r - interp_state.cell_r0[cid]) / interp_state.cell_rden[cid]
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0

    v = (polar - interp_state.cell_t0[cid]) / interp_state.cell_tden[cid]
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0

    p_rel = (azimuth - interp_state.cell_p_start[cid]) % _TWO_PI
    if interp_state.cell_phi_tiny[cid]:
        w = 0.0
    else:
        if not interp_state.cell_phi_full[cid]:
            width = interp_state.cell_p_width[cid]
            if p_rel < 0.0:
                p_rel = 0.0
            elif p_rel > width:
                p_rel = width
        w = p_rel / interp_state.cell_pden[cid]
        if w < 0.0:
            w = 0.0
        elif w > 1.0:
            w = 1.0

    w0 = (1.0 - u) * (1.0 - v) * (1.0 - w)
    w1 = u * (1.0 - v) * (1.0 - w)
    w2 = (1.0 - u) * v * (1.0 - w)
    w3 = u * v * (1.0 - w)
    w4 = (1.0 - u) * (1.0 - v) * w
    w5 = u * (1.0 - v) * w
    w6 = (1.0 - u) * v * w
    w7 = u * v * w

    local = interp_state.corners[cid]
    map_row = interp_state.bin_to_corner[cid]
    c0 = int(local[int(map_row[0])])
    c1 = int(local[int(map_row[1])])
    c2 = int(local[int(map_row[2])])
    c3 = int(local[int(map_row[3])])
    c4 = int(local[int(map_row[4])])
    c5 = int(local[int(map_row[5])])
    c6 = int(local[int(map_row[6])])
    c7 = int(local[int(map_row[7])])

    out_row[:] = (
        w0 * interp_state.point_values_2d[c0]
        + w1 * interp_state.point_values_2d[c1]
        + w2 * interp_state.point_values_2d[c2]
        + w3 * interp_state.point_values_2d[c3]
        + w4 * interp_state.point_values_2d[c4]
        + w5 * interp_state.point_values_2d[c5]
        + w6 * interp_state.point_values_2d[c6]
        + w7 * interp_state.point_values_2d[c7]
    )


@njit(cache=True)
def _trilinear_from_cell(
    out_row: np.ndarray,
    node_id: int,
    x: float,
    y: float,
    z: float,
    interp_state: CartesianInterpKernelState,
) -> None:
    """Write one Cartesian trilinear interpolation result row for one leaf node."""
    cid = int(node_id)
    u = (x - interp_state.cell_x0[cid]) / interp_state.cell_xden[cid]
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    v = (y - interp_state.cell_y0[cid]) / interp_state.cell_yden[cid]
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    w = (z - interp_state.cell_z0[cid]) / interp_state.cell_zden[cid]
    if w < 0.0:
        w = 0.0
    elif w > 1.0:
        w = 1.0

    w0 = (1.0 - u) * (1.0 - v) * (1.0 - w)
    w1 = u * (1.0 - v) * (1.0 - w)
    w2 = (1.0 - u) * v * (1.0 - w)
    w3 = u * v * (1.0 - w)
    w4 = (1.0 - u) * (1.0 - v) * w
    w5 = u * (1.0 - v) * w
    w6 = (1.0 - u) * v * w
    w7 = u * v * w

    local = interp_state.corners[cid]
    map_row = interp_state.bin_to_corner[cid]
    c0 = int(local[int(map_row[0])])
    c1 = int(local[int(map_row[1])])
    c2 = int(local[int(map_row[2])])
    c3 = int(local[int(map_row[3])])
    c4 = int(local[int(map_row[4])])
    c5 = int(local[int(map_row[5])])
    c6 = int(local[int(map_row[6])])
    c7 = int(local[int(map_row[7])])

    out_row[:] = (
        w0 * interp_state.point_values_2d[c0]
        + w1 * interp_state.point_values_2d[c1]
        + w2 * interp_state.point_values_2d[c2]
        + w3 * interp_state.point_values_2d[c3]
        + w4 * interp_state.point_values_2d[c4]
        + w5 * interp_state.point_values_2d[c5]
        + w6 * interp_state.point_values_2d[c6]
        + w7 * interp_state.point_values_2d[c7]
    )


@njit(cache=True, parallel=True)
def _interp_from_node_ids_rpa(
    queries_rpa: np.ndarray,
    node_ids: np.ndarray,
    fill_values: np.ndarray,
    interp_state: SphericalInterpKernelState,
    ) -> np.ndarray:
    """Evaluate spherical-tree interpolation for spherical queries with known leaf node ids."""
    n_query = queries_rpa.shape[0]
    ncomp = interp_state.point_values_2d.shape[1]
    out = np.empty((n_query, ncomp), dtype=interp_state.point_values_2d.dtype)
    for i in prange(n_query):
        out[i, :] = fill_values
        cid = int(node_ids[i])
        if cid < 0:
            continue
        _trilinear_from_cell_rpa(
            out[i],
            cid,
            queries_rpa[i, 0],
            queries_rpa[i, 1],
            queries_rpa[i, 2] % _TWO_PI,
            interp_state,
        )
    return out


@njit(cache=True, parallel=True)
def _interp_from_node_ids_xyz_cartesian(
    queries_xyz: np.ndarray,
    node_ids: np.ndarray,
    fill_values: np.ndarray,
    interp_state: CartesianInterpKernelState,
    ) -> np.ndarray:
    """Evaluate Cartesian-tree interpolation for Cartesian queries with known leaf node ids.

    Assumes the Cartesian backend cell model (axis-aligned per-cell bounds).
    """
    n_query = queries_xyz.shape[0]
    ncomp = interp_state.point_values_2d.shape[1]
    out = np.empty((n_query, ncomp), dtype=interp_state.point_values_2d.dtype)
    for i in prange(n_query):
        out[i, :] = fill_values
        cid = int(node_ids[i])
        if cid < 0:
            continue
        _trilinear_from_cell(
            out[i],
            cid,
            queries_xyz[i, 0],
            queries_xyz[i, 1],
            queries_xyz[i, 2],
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

    Ray methods additionally split cells into a fixed 6-tet decomposition and
    produce piecewise-linear functions along the ray.

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
            int(self._ds.corners.shape[0]),
        )
        self.value_names: tuple[str, ...] = ()
        self._point_values = self._coerce_point_values(values)
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

    def _coerce_point_values(self, values: list[str] | np.ndarray | None) -> np.ndarray:
        """Resolve requested fields into an array indexed by dataset points."""
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
                return arr
            merged = np.concatenate([arr.reshape(n_points, -1) for arr in arrays], axis=1)
            logger.debug("Using %d fields with merged shape=%s", len(names), tuple(merged.shape))
            return merged

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
                return arr
            merged = np.concatenate([arr.reshape(n_points, -1) for arr in arrays], axis=1)
            logger.debug("Using %d fields with merged shape=%s", len(names), tuple(merged.shape))
            return merged

        arr = np.asarray(values)
        if arr.shape[0] != n_points:
            raise ValueError(f"values length {arr.shape[0]} does not match required n_points={n_points}.")
        self.value_names = ()
        logger.debug("Using explicit value array with shape=%s", tuple(arr.shape))
        return arr

    def prepare_kernel_cache(self) -> None:
        """Pack per-point values and bind them to tree-owned interpolation geometry."""
        flat = self._point_values.reshape(int(self._point_values.shape[0]), -1)
        self._point_values_2d = np.array(flat, dtype=np.float64, order="C")
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
            q_rpa, node_ids_xyz = self.tree._lookup_points_local(q_xyz, coord="xyz")
            _q_rpa_direct, node_ids_rpa = self.tree._lookup_points_local(q_rpa, coord="rpa")
            _interp_from_node_ids_rpa(q_rpa, node_ids_xyz, fill, self._interp_state)
            _interp_from_node_ids_rpa(q_rpa, node_ids_rpa, fill, self._interp_state)
            return
        _q_local, node_ids_xyz = self.tree._lookup_points_local(q_xyz, coord="xyz")
        _interp_from_node_ids_xyz_cartesian(
            q_xyz,
            node_ids_xyz,
            fill,
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
        trailing = self._point_values.shape[1:]
        logger.debug("Interpolating %d query points in %s space", int(n), qs)
        fill = self._fill_value_vector()
        t_after_fill = perf_counter() if debug_timing else 0.0

        if self._tree_coord == "rpa":
            q_local, node_ids = self.tree._lookup_points_local(q_array, coord=qs)
            if debug_timing:
                logger.debug("Interpolation kernel mode: compiled-rpa")
            out2d = _interp_from_node_ids_rpa(q_local, node_ids, fill, self._interp_state)
        else:
            _q_local, node_ids = self.tree._lookup_points_local(q_array, coord="xyz")
            if debug_timing:
                logger.debug("Interpolation kernel mode: compiled-xyz")
            out2d = _interp_from_node_ids_xyz_cartesian(
                q_array,
                node_ids,
                fill,
                self._interp_state,
            )
        t_after_kernel = perf_counter() if debug_timing else 0.0

        cell_ids = np.full(node_ids.shape, -1, dtype=np.int64)
        hits = node_ids >= 0
        if np.any(hits):
            cell_ids[hits] = self.tree._node_value[node_ids[hits]]

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
        node_ids = np.flatnonzero(self.tree._node_value == cid)
        if node_ids.size != 1:
            raise ValueError(f"Invalid cell_id {cid}.")
        return int(np.unique(self.tree._interp_bin_to_corner[int(node_ids[0])]).size)

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
        return self._ds.corners

    @property
    def point_values(self) -> np.ndarray:
        """Return per-node interpolation values in original component shape."""
        return self._point_values

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
        self._point_values = self._coerce_point_values(values)
        self.prepare_kernel_cache()
        # Fail fast when a vector fill value no longer matches component count.
        _ = self._fill_value_vector()
        if warmup:
            self.warmup_kernels()

    def __str__(self) -> str:
        """Return a compact human-readable interpolator description."""
        n_points = int(self._ds.points.shape[0]) if hasattr(self._ds, "points") else -1
        n_cells = int(self._ds.corners.shape[0])
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

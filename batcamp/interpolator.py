#!/usr/bin/env python3
"""Octree interpolator and interpolation kernels."""

from __future__ import annotations

import logging
import math
from time import perf_counter
from typing import Literal
from typing import NamedTuple

from numba import njit
from numba import prange
import numpy as np

from .octree import LookupKernelState
from .octree import Octree
from .cartesian import _lookup_xyz_cell_id_kernel
from .spherical import _lookup_rpa_cell_id_kernel
from .spherical import _xyz_to_rpa_components

logger = logging.getLogger(__name__)

_DEFAULT_SEED_CHUNK_SIZE = 1024
_TWO_PI = 2.0 * math.pi

class SphericalInterpKernelState(NamedTuple):
    """Numba interpolation-kernel arrays with explicit field names."""

    point_values_2d: np.ndarray
    corners: np.ndarray
    bin_to_corner: np.ndarray
    cell_r0: np.ndarray
    cell_rden: np.ndarray
    cell_t0: np.ndarray
    cell_tden: np.ndarray
    cell_p_start: np.ndarray
    cell_p_width: np.ndarray
    cell_pden: np.ndarray
    cell_phi_full: np.ndarray
    cell_phi_tiny: np.ndarray

class CartesianInterpKernelState(NamedTuple):
    """Numba Cartesian interpolation-kernel arrays with explicit field names."""

    point_values_2d: np.ndarray
    corners: np.ndarray
    bin_to_corner: np.ndarray
    cell_x0: np.ndarray
    cell_xden: np.ndarray
    cell_y0: np.ndarray
    cell_yden: np.ndarray
    cell_z0: np.ndarray
    cell_zden: np.ndarray


@njit(cache=True)
def _trilinear_from_cell_rpa(
    out_row: np.ndarray,
    cell_id: int,
    r: float,
    polar: float,
    azimuth: float,
    interp_state: SphericalInterpKernelState,
) -> None:
    """Write one interpolated value row for one spherical query in one cell."""
    cid = int(cell_id)

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
    cell_id: int,
    x: float,
    y: float,
    z: float,
    interp_state: CartesianInterpKernelState,
) -> None:
    """Write one Cartesian trilinear interpolation result row for one cell."""
    cid = int(cell_id)
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
def _interp_batch_xyz(
    queries_xyz: np.ndarray,
    fill_values: np.ndarray,
    interp_state: SphericalInterpKernelState,
    lookup_state: LookupKernelState,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a batch of Cartesian queries and return values plus cell ids."""
    n_query = queries_xyz.shape[0]
    ncomp = interp_state.point_values_2d.shape[1]
    out = np.empty((n_query, ncomp), dtype=interp_state.point_values_2d.dtype)
    cell_ids = np.full(n_query, -1, dtype=np.int64)
    chunk_size = int(_DEFAULT_SEED_CHUNK_SIZE)
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    for chunk_id in prange(n_chunks):
        start = chunk_id * chunk_size
        end = min(n_query, start + chunk_size)
        hint_cid = -1
        for i in range(start, end):
            out[i, :] = fill_values

            x = queries_xyz[i, 0]
            y = queries_xyz[i, 1]
            z = queries_xyz[i, 2]
            r = math.sqrt(x * x + y * y + z * z)
            if r == 0.0:
                polar = 0.0
            else:
                zr = z / r
                if zr < -1.0:
                    zr = -1.0
                elif zr > 1.0:
                    zr = 1.0
                polar = math.acos(zr)
            azimuth = math.atan2(y, x) % _TWO_PI
            cid = _lookup_rpa_cell_id_kernel(
                r,
                polar,
                azimuth,
                lookup_state,
                hint_cid,
            )
            if cid < 0:
                hint_cid = -1
                continue
            cell_ids[i] = cid
            hint_cid = int(cid)
            _trilinear_from_cell_rpa(
                out[i],
                cid,
                r,
                polar,
                azimuth,
                interp_state,
            )
    return out, cell_ids


@njit(cache=True, parallel=True)
def _interp_batch_rpa(
    queries_rpa: np.ndarray,
    fill_values: np.ndarray,
    interp_state: SphericalInterpKernelState,
    lookup_state: LookupKernelState,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a batch of spherical queries and return values plus cell ids."""
    n_query = queries_rpa.shape[0]
    ncomp = interp_state.point_values_2d.shape[1]
    out = np.empty((n_query, ncomp), dtype=interp_state.point_values_2d.dtype)
    cell_ids = np.full(n_query, -1, dtype=np.int64)
    chunk_size = int(_DEFAULT_SEED_CHUNK_SIZE)
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    for chunk_id in prange(n_chunks):
        start = chunk_id * chunk_size
        end = min(n_query, start + chunk_size)
        hint_cid = -1
        for i in range(start, end):
            out[i, :] = fill_values

            r = queries_rpa[i, 0]
            polar = queries_rpa[i, 1]
            azimuth = queries_rpa[i, 2] % _TWO_PI
            cid = _lookup_rpa_cell_id_kernel(
                r,
                polar,
                azimuth,
                lookup_state,
                hint_cid,
            )
            if cid < 0:
                hint_cid = -1
                continue
            cell_ids[i] = cid
            hint_cid = int(cid)
            _trilinear_from_cell_rpa(
                out[i],
                cid,
                r,
                polar,
                azimuth,
                interp_state,
            )
    return out, cell_ids


@njit(cache=True, parallel=True)
def _interp_batch_xyz_cartesian(
    queries_xyz: np.ndarray,
    fill_values: np.ndarray,
    interp_state: CartesianInterpKernelState,
    lookup_state: LookupKernelState,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Cartesian queries for Cartesian trees via compiled kernels.

    Assumes the Cartesian backend cell model (axis-aligned per-cell bounds).
    """
    n_query = queries_xyz.shape[0]
    ncomp = interp_state.point_values_2d.shape[1]
    out = np.empty((n_query, ncomp), dtype=interp_state.point_values_2d.dtype)
    cell_ids = np.full(n_query, -1, dtype=np.int64)
    chunk_size = int(_DEFAULT_SEED_CHUNK_SIZE)
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    for chunk_id in prange(n_chunks):
        start = chunk_id * chunk_size
        end = min(n_query, start + chunk_size)
        hint_cid = -1
        for i in range(start, end):
            out[i, :] = fill_values

            x = queries_xyz[i, 0]
            y = queries_xyz[i, 1]
            z = queries_xyz[i, 2]
            cid = _lookup_xyz_cell_id_kernel(x, y, z, lookup_state, hint_cid)
            if cid < 0:
                hint_cid = -1
                continue
            cell_ids[i] = cid
            hint_cid = int(cid)
            _trilinear_from_cell(
                out[i],
                cid,
                x,
                y,
                z,
                interp_state,
            )
    return out, cell_ids

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
        if tree.ds is None or tree.ds.corners is None:
            logger.error("Octree is not bound to a dataset; cannot build interpolator.")
            raise ValueError("Octree is not bound to a dataset with corners.")
        self.tree = tree
        lookup_geometry = self.tree.lookup_geometry()
        self._ds = tree.ds
        self._corners = lookup_geometry.corners
        self._points = lookup_geometry.points
        self._lookup_state = lookup_geometry.coord_state
        self.fill_value = fill_value

        logger.debug(
            "Initializing OctreeInterpolator: points=%d, cells=%d",
            int(self._points.shape[0]),
            int(self._corners.shape[0]),
        )
        self.value_names: tuple[str, ...] = ()
        self._point_values = self._coerce_point_values(values)
        self._tree_coord = str(self.tree.tree_coord)
        if self._tree_coord == "rpa":
            self._prepare_spherical_points()
            self._prepare_trilinear_cache()
        elif self._tree_coord == "xyz":
            self._prepare_trilinear_cache_xyz()
        else:
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

    def _prepare_spherical_points(self) -> None:
        """Precompute spherical coordinates `(r, theta, phi)` for each node."""
        x = self._points[:, 0]
        y = self._points[:, 1]
        z = self._points[:, 2]
        r = np.sqrt(x * x + y * y + z * z)
        self._node_r = r
        self._node_theta = np.arccos(np.clip(z / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0))
        self._node_phi = np.mod(np.arctan2(y, x), 2.0 * math.pi)

    def _prepare_trilinear_cache(self) -> None:
        """Build per-cell corner mappings used for spherical trilinear interpolation."""
        corners = self._corners
        vr = self._node_r[corners]
        vt = self._node_theta[corners]
        vp = self._node_phi[corners]
        lookup_state = self._lookup_state
        self._cell_r0 = lookup_state.cell_axis0_start
        self._cell_r1 = lookup_state.cell_axis0_start + lookup_state.cell_axis0_width
        self._cell_t0 = lookup_state.cell_axis1_start
        self._cell_t1 = lookup_state.cell_axis1_start + lookup_state.cell_axis1_width
        self._cell_p_start = lookup_state.cell_axis2_start
        self._cell_p_width = lookup_state.cell_axis2_width

        tiny = np.finfo(float).tiny
        self._cell_rden = np.maximum(self._cell_r1 - self._cell_r0, tiny)
        self._cell_tden = np.maximum(self._cell_t1 - self._cell_t0, tiny)
        self._cell_pden = np.maximum(self._cell_p_width, tiny)
        self._cell_phi_full = self._cell_p_width >= (2.0 * math.pi - 1e-10)
        self._cell_phi_tiny = self._cell_p_width <= tiny

        p_rel = np.mod(vp - self._cell_p_start[:, None], 2.0 * math.pi)
        clip_mask = (~self._cell_phi_full)[:, None]
        p_rel = np.where(clip_mask, np.clip(p_rel, 0.0, self._cell_p_width[:, None]), p_rel)

        r_mid = 0.5 * (self._cell_r0 + self._cell_r1)[:, None]
        t_mid = 0.5 * (self._cell_t0 + self._cell_t1)[:, None]
        p_mid = 0.5 * self._cell_p_width[:, None]

        bit_r = (vr >= r_mid).astype(np.int8)
        bit_t = (vt >= t_mid).astype(np.int8)
        bit_p = np.zeros_like(bit_r, dtype=np.int8)
        valid_phi = ~self._cell_phi_tiny
        if np.any(valid_phi):
            bit_p[valid_phi] = (p_rel[valid_phi] >= p_mid[valid_phi]).astype(np.int8)

        bin_id = bit_r + (bit_t << 1) + (bit_p << 2)
        bit_trip = np.stack((bit_r, bit_t, bit_p), axis=2)
        target_bits = np.array(
            [[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)],
            dtype=np.int8,
        )

        n_cells = corners.shape[0]
        bin_to_corner = np.empty((n_cells, 8), dtype=np.int8)
        for k in range(8):
            eq = bin_id == k
            has = np.any(eq, axis=1)
            pick = np.argmax(eq, axis=1).astype(np.int64)
            missing = ~has
            if np.any(missing):
                d = np.sum((bit_trip[missing] - target_bits[k]) ** 2, axis=2)
                pick[missing] = np.argmin(d, axis=1)
            bin_to_corner[:, k] = pick.astype(np.int8)
        self._bin_to_corner = bin_to_corner

    def _prepare_trilinear_cache_xyz(self) -> None:
        """Build per-cell corner mappings for Cartesian trilinear interpolation.

        The Cartesian backend maps query/corner positions through per-cell
        axis-aligned min/max bounds (slab normalization).
        """
        corners = self._corners
        pts = self._points
        lookup_state = self._lookup_state
        vx = pts[corners, 0]
        vy = pts[corners, 1]
        vz = pts[corners, 2]

        self._cell_x0 = lookup_state.cell_axis0_start
        self._cell_x1 = lookup_state.cell_axis0_start + lookup_state.cell_axis0_width
        self._cell_y0 = lookup_state.cell_axis1_start
        self._cell_y1 = lookup_state.cell_axis1_start + lookup_state.cell_axis1_width
        self._cell_z0 = lookup_state.cell_axis2_start
        self._cell_z1 = lookup_state.cell_axis2_start + lookup_state.cell_axis2_width

        tiny = np.finfo(float).tiny
        self._cell_xden = np.maximum(self._cell_x1 - self._cell_x0, tiny)
        self._cell_yden = np.maximum(self._cell_y1 - self._cell_y0, tiny)
        self._cell_zden = np.maximum(self._cell_z1 - self._cell_z0, tiny)

        x_mid = 0.5 * (self._cell_x0 + self._cell_x1)[:, None]
        y_mid = 0.5 * (self._cell_y0 + self._cell_y1)[:, None]
        z_mid = 0.5 * (self._cell_z0 + self._cell_z1)[:, None]

        bit_x = (vx >= x_mid).astype(np.int8)
        bit_y = (vy >= y_mid).astype(np.int8)
        bit_z = (vz >= z_mid).astype(np.int8)
        bin_id = bit_x + (bit_y << 1) + (bit_z << 2)
        bit_trip = np.stack((bit_x, bit_y, bit_z), axis=2)
        target_bits = np.array(
            [[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)],
            dtype=np.int8,
        )

        n_cells = corners.shape[0]
        bin_to_corner = np.empty((n_cells, 8), dtype=np.int8)
        for k in range(8):
            eq = bin_id == k
            has = np.any(eq, axis=1)
            pick = np.argmax(eq, axis=1).astype(np.int64)
            missing = ~has
            if np.any(missing):
                d = np.sum((bit_trip[missing] - target_bits[k]) ** 2, axis=2)
                pick[missing] = np.argmin(d, axis=1)
            bin_to_corner[:, k] = pick.astype(np.int8)
        self._bin_to_corner = bin_to_corner

    def prepare_kernel_cache(self) -> None:
        """Pack arrays used by compiled interpolation code."""
        flat = self._point_values.reshape(int(self._point_values.shape[0]), -1)
        self._point_values_2d = np.array(flat, dtype=np.float64, order="C")
        self._n_value_components = int(self._point_values_2d.shape[1])
        self._bin_to_corner_index = np.array(self._bin_to_corner, dtype=np.int64, order="C")
        if self._tree_coord == "rpa":
            self._interp_state_rpa = SphericalInterpKernelState(
                point_values_2d=self._point_values_2d,
                corners=self._corners,
                bin_to_corner=self._bin_to_corner_index,
                cell_r0=self._cell_r0,
                cell_rden=self._cell_rden,
                cell_t0=self._cell_t0,
                cell_tden=self._cell_tden,
                cell_p_start=self._cell_p_start,
                cell_p_width=self._cell_p_width,
                cell_pden=self._cell_pden,
                cell_phi_full=self._cell_phi_full,
                cell_phi_tiny=self._cell_phi_tiny,
            )
            self._lookup_state_rpa = self._lookup_state
            return
        if self._tree_coord == "xyz":
            self._interp_state_xyz = CartesianInterpKernelState(
                point_values_2d=self._point_values_2d,
                corners=self._corners,
                bin_to_corner=self._bin_to_corner_index,
                cell_x0=self._cell_x0,
                cell_xden=self._cell_xden,
                cell_y0=self._cell_y0,
                cell_yden=self._cell_yden,
                cell_z0=self._cell_z0,
                cell_zden=self._cell_zden,
            )
            self._lookup_state_xyz = self._lookup_state
            return
        raise NotImplementedError(f"Unsupported tree_coord '{self._tree_coord}' for kernel cache setup.")

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
        q_xyz = np.array(self._points[:1], dtype=np.float64, order="C")
        if q_xyz.shape[0] == 0:
            q_xyz = np.zeros((1, 3), dtype=np.float64)
        fill = self._fill_value_vector()
        if self._tree_coord == "rpa":
            r, polar, azimuth = _xyz_to_rpa_components(float(q_xyz[0, 0]), float(q_xyz[0, 1]), float(q_xyz[0, 2]))
            q_rpa = np.array([[r, polar, azimuth]], dtype=np.float64, order="C")
            _interp_batch_xyz(
                q_xyz,
                fill,
                self._interp_state_rpa,
                self._lookup_state_rpa,
            )
            _interp_batch_rpa(
                q_rpa,
                fill,
                self._interp_state_rpa,
                self._lookup_state_rpa,
            )
            return
        if self._tree_coord == "xyz":
            _interp_batch_xyz_cartesian(
                q_xyz,
                fill,
                self._interp_state_xyz,
                self._lookup_state_xyz,
            )
            return
        raise NotImplementedError(f"Unsupported tree_coord '{self._tree_coord}' for kernel warmup.")

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
            batch_kernel = _interp_batch_xyz if qs == "xyz" else _interp_batch_rpa
            if debug_timing:
                logger.debug("Interpolation kernel mode: compiled-rpa")
            out2d, cell_ids = batch_kernel(
                q_array,
                fill,
                self._interp_state_rpa,
                self._lookup_state_rpa,
            )
        else:
            if debug_timing:
                logger.debug("Interpolation kernel mode: compiled-xyz")
            out2d, cell_ids = _interp_batch_xyz_cartesian(
                q_array,
                fill,
                self._interp_state_xyz,
                self._lookup_state_xyz,
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
        n_cells = int(self._bin_to_corner.shape[0])
        if cid < 0 or cid >= n_cells:
            raise ValueError(f"Invalid cell_id {cid}; expected [0, {n_cells - 1}].")
        return int(np.unique(self._bin_to_corner[cid]).size)

    def cell_has_full_trilinear_corner_map(self, cell_id: int) -> bool:
        """Return whether one cell maps to all 8 logical trilinear corners."""
        return self.trilinear_corner_count(int(cell_id)) == 8

    @property
    def n_value_components(self) -> int:
        """Return flattened component count of the interpolated output."""
        return int(self._n_value_components)

    @property
    def xyz_interp_state(self) -> CartesianInterpKernelState | None:
        """Return Cartesian kernel state for xyz trees, otherwise `None`."""
        if self._tree_coord != "xyz":
            return None
        return getattr(self, "_interp_state_xyz", None)

    @property
    def corners(self) -> np.ndarray:
        """Return cell-to-node corner connectivity used by interpolation."""
        return self._corners

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
        n_cells = int(self._corners.shape[0])
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

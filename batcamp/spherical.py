#!/usr/bin/env python3
"""Spherical octree and spherical lookup implementation."""

from __future__ import annotations

import math
from typing import ClassVar
from typing import NamedTuple

from numba import njit
import numpy as np

from .octree import GridIndex
from .octree import GridPath
from .octree import LookupHit
from .octree import Octree

_TWO_PI = 2.0 * math.pi
_LOOKUP_CONTAIN_TOL = 1e-10
_DEFAULT_LOOKUP_MAX_RADIUS = 8
_MISSING_NODE_VALUE = -1
_INTERNAL_NODE_VALUE = -2


@njit(cache=True)
def _xyz_to_rpa_components(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert one Cartesian point to spherical `(r, polar, azimuth)`."""
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
    return r, polar, azimuth


class SphericalLookupKernelState(NamedTuple):
    """Arrays used by compiled spherical lookup code."""

    levels_desc: np.ndarray
    shape_table: np.ndarray
    dtheta_table: np.ndarray
    dphi_table: np.ndarray
    bin_level_offset: np.ndarray
    bin_offsets: np.ndarray
    bin_cell_ids: np.ndarray
    cell_r_min: np.ndarray
    cell_r_max: np.ndarray
    cell_theta_min: np.ndarray
    cell_theta_max: np.ndarray
    cell_phi_start: np.ndarray
    cell_phi_width: np.ndarray
    cell_valid: np.ndarray
    cell_centers: np.ndarray
    r_min: float
    r_max: float
    radial_edges: np.ndarray
    max_radius: int
    leaf_shape: np.ndarray
    tree_depth: int
    cell_i0: np.ndarray
    cell_i1: np.ndarray
    cell_i2: np.ndarray
    node_depth: np.ndarray
    node_i0: np.ndarray
    node_i1: np.ndarray
    node_i2: np.ndarray
    node_value: np.ndarray


@njit(cache=True)
def _lookup_axis_index(q: float, q_min: float, q_span: float, n_fine: int) -> int:
    """Map one scalar coordinate to a finest-grid index, clamped to bounds."""
    pos = ((q - q_min) / q_span) * n_fine
    if pos <= 0.0:
        return 0
    if pos >= n_fine:
        return int(n_fine - 1)
    # On an exact shared boundary, give ownership to the lower-index cell.
    idx = int(math.ceil(pos)) - 1
    if idx < 0:
        return 0
    if idx >= n_fine:
        return int(n_fine - 1)
    return idx


@njit(cache=True)
def _lookup_interval_index(q: float, edges: np.ndarray) -> int:
    """Map one scalar coordinate to the interval index of a sorted edge array."""
    n_bins = int(edges.shape[0]) - 1
    if n_bins <= 0:
        return -1
    if q <= edges[0]:
        return 0
    if q >= edges[n_bins]:
        return int(n_bins - 1)
    lo = 0
    hi = int(edges.shape[0])
    while lo < hi:
        mid = (lo + hi) // 2
        if q <= edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    idx = lo - 1
    if idx < 0:
        return 0
    if idx >= n_bins:
        return int(n_bins - 1)
    return idx


@njit(cache=True)
def _find_node_value(
    depth: int,
    i0: int,
    i1: int,
    i2: int,
    lookup_state: SphericalLookupKernelState,
) -> int:
    """Binary-search one occupied spherical node by `(depth, i0, i1, i2)`."""
    lo = 0
    hi = int(lookup_state.node_depth.shape[0]) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        m_depth = int(lookup_state.node_depth[mid])
        if depth < m_depth:
            hi = mid - 1
            continue
        if depth > m_depth:
            lo = mid + 1
            continue

        m_i0 = int(lookup_state.node_i0[mid])
        if i0 < m_i0:
            hi = mid - 1
            continue
        if i0 > m_i0:
            lo = mid + 1
            continue

        m_i1 = int(lookup_state.node_i1[mid])
        if i1 < m_i1:
            hi = mid - 1
            continue
        if i1 > m_i1:
            lo = mid + 1
            continue

        m_i2 = int(lookup_state.node_i2[mid])
        if i2 < m_i2:
            hi = mid - 1
            continue
        if i2 > m_i2:
            lo = mid + 1
            continue

        return int(lookup_state.node_value[mid])
    return _MISSING_NODE_VALUE
@njit(cache=True)
def _contains_rpa_cell(
    cid: int,
    r: float,
    polar: float,
    azimuth: float,
    lookup_state: SphericalLookupKernelState,
) -> bool:
    """Return whether one spherical point lies inside one cell."""
    if not lookup_state.cell_valid[cid]:
        return False
    tol = _LOOKUP_CONTAIN_TOL
    if r < (lookup_state.cell_r_min[cid] - tol) or r > (lookup_state.cell_r_max[cid] + tol):
        return False
    if polar < (lookup_state.cell_theta_min[cid] - tol) or polar > (lookup_state.cell_theta_max[cid] + tol):
        return False
    if abs(r * math.sin(polar)) <= tol:
        return True
    width = lookup_state.cell_phi_width[cid]
    dphi = (azimuth - lookup_state.cell_phi_start[cid]) % _TWO_PI
    if width >= (_TWO_PI - tol):
        return True
    return dphi <= (width + tol)


@njit(cache=True)
def _lookup_rpa_cell_id_kernel(
    r: float,
    polar: float,
    azimuth: float,
    lookup_state: SphericalLookupKernelState,
    prev_cid: int = -1,
) -> int:
    """Return the containing spherical cell id, or `-1` when not found."""
    if not (math.isfinite(r) and math.isfinite(polar) and math.isfinite(azimuth)):
        return -1
    if polar < 0.0 or polar > math.pi:
        return -1
    azimuth = azimuth % _TWO_PI
    if r < lookup_state.r_min or r > lookup_state.r_max:
        return -1
    if prev_cid >= 0 and _contains_rpa_cell(
        int(prev_cid),
        r,
        polar,
        azimuth,
        lookup_state,
    ):
        return int(prev_cid)

    fine_i0 = _lookup_axis_index(
        r,
        lookup_state.r_min,
        max(lookup_state.r_max - lookup_state.r_min, np.finfo(np.float64).tiny),
        int(lookup_state.leaf_shape[0]),
    )
    if lookup_state.radial_edges.shape[0] > 1:
        fine_i0 = _lookup_interval_index(r, lookup_state.radial_edges)
    fine_i1 = _lookup_axis_index(
        polar,
        0.0,
        math.pi,
        int(lookup_state.leaf_shape[1]),
    )
    fine_i2 = _lookup_axis_index(
        azimuth,
        0.0,
        _TWO_PI,
        int(lookup_state.leaf_shape[2]),
    )

    tree_depth = int(lookup_state.tree_depth)
    if abs(r * math.sin(polar)) <= _LOOKUP_CONTAIN_TOL:
        for depth in range(tree_depth + 1):
            shift = tree_depth - depth
            i0 = fine_i0 >> shift
            i1 = fine_i1 >> shift
            nphi = int(lookup_state.leaf_shape[2]) >> shift
            node_value = _MISSING_NODE_VALUE
            for i2 in range(nphi):
                node_value = _find_node_value(depth, i0, i1, i2, lookup_state)
                if node_value != _MISSING_NODE_VALUE:
                    break
            if node_value == _MISSING_NODE_VALUE:
                return -1
            if node_value >= 0:
                cid = int(node_value)
                if _contains_rpa_cell(cid, r, polar, azimuth, lookup_state):
                    return cid
                return -1
        return -1

    for depth in range(tree_depth + 1):
        shift = tree_depth - depth
        i0 = fine_i0 >> shift
        i1 = fine_i1 >> shift
        i2 = fine_i2 >> shift
        node_value = _find_node_value(depth, i0, i1, i2, lookup_state)
        if node_value == _MISSING_NODE_VALUE:
            return -1
        if node_value >= 0:
            cid = int(node_value)
            if _contains_rpa_cell(cid, r, polar, azimuth, lookup_state):
                return cid
            return -1
    return -1

class _SphericalCellLookup:
    """Cell lookup helper for spherical trees."""

    @staticmethod
    def _cluster_close_values(values: np.ndarray, *, atol: float) -> tuple[np.ndarray, np.ndarray]:
        """Cluster sorted boundary values within one absolute tolerance."""
        ordered = np.sort(np.asarray(values, dtype=float).reshape(-1))
        if ordered.size == 0:
            return ordered, ordered
        clusters: list[list[float]] = [[float(ordered[0])]]
        prev = float(ordered[0])
        for value in ordered[1:]:
            current = float(value)
            if abs(current - prev) <= float(atol):
                clusters[-1].append(current)
                prev = current
                continue
            clusters.append([current])
            prev = current

        centers = np.empty(len(clusters), dtype=float)
        tolerances = np.empty(len(clusters), dtype=float)
        for idx, cluster in enumerate(clusters):
            arr = np.asarray(cluster, dtype=float)
            center = float(np.mean(arr))
            centers[idx] = center
            tolerances[idx] = max(float(np.max(np.abs(arr - center))), float(atol))
        return centers, tolerances

    def _init_lookup_state(
        self,
        tree: Octree,
    ) -> None:
        """Build lookup arrays from a bound spherical tree."""
        if tree.ds is None or tree.ds.corners is None:
            raise ValueError("Lookup requires a bound octree with dataset and corners.")
        ds = tree.ds
        corners = np.array(ds.corners, dtype=np.int64)
        cell_levels = tree.cell_levels
        axis_rho_tol = float(tree.axis_rho_tol)
        if not set(Octree.XYZ_VARS).issubset(set(ds.variables)):
            raise ValueError("Lookup requires X/Y/Z variables.")

        self.tree = tree
        self._axis_rho_tol = axis_rho_tol
        self._corners = np.array(corners, dtype=np.int64)
        self._points = np.column_stack(
            (
                np.array(ds[Octree.X_VAR], dtype=float),
                np.array(ds[Octree.Y_VAR], dtype=float),
                np.array(ds[Octree.Z_VAR], dtype=float),
            )
        )
        self._cell_centers = np.mean(self._points[self._corners], axis=1)
        n_cells = int(self._corners.shape[0])
        if cell_levels is None or int(cell_levels.shape[0]) != n_cells:
            self._cell_level = np.full(n_cells, int(tree.max_level), dtype=np.int64)
        else:
            self._cell_level = np.array(cell_levels, dtype=np.int64)
        self._build_index()

    def _build_index(self) -> None:
        """Build per-level lookup tables, bins, and per-cell bounds."""
        n_cells = self._corners.shape[0]
        valid = self._cell_level >= 0
        if not np.any(valid):
            raise ValueError("Lookup requires at least one valid leaf level.")

        self._max_level = int(self.tree.max_level)
        valid_levels = sorted(set(int(v) for v in self._cell_level[valid].tolist()))
        shape_by_level: dict[int, tuple[int, int, int]] = {}
        dtheta_by_level: dict[int, float] = {}
        dphi_by_level: dict[int, float] = {}
        for level in valid_levels:
            depth = int(level)
            if depth < 0:
                raise ValueError(f"Derived negative level {level}; max_level={self.tree.max_level}.")
            nr = int(self.tree.root_shape[0] * (1 << depth))
            ntheta = int(self.tree.root_shape[1] * (1 << depth))
            nphi = int(self.tree.root_shape[2] * (1 << depth))
            shape_by_level[level] = (nr, ntheta, nphi)
            dtheta_by_level[level] = math.pi / float(ntheta)
            dphi_by_level[level] = _TWO_PI / float(nphi)
        self._shape_by_level = shape_by_level
        self._dtheta_by_level = dtheta_by_level
        self._dphi_by_level = dphi_by_level

        levels_asc = np.array(sorted(self._shape_by_level.keys()), dtype=np.int64)
        self._levels_desc = levels_asc[::-1]
        level_cap = int(np.max(levels_asc)) + 1
        shape_table = np.full((level_cap, 3), -1, dtype=np.int64)
        dtheta_table = np.full(level_cap, np.nan, dtype=float)
        dphi_table = np.full(level_cap, np.nan, dtype=float)
        bin_level_offset = np.full(level_cap, -1, dtype=np.int64)
        running_offset = 0
        for level in levels_asc:
            lvl = int(level)
            shape = self._shape_by_level[lvl]
            shape_table[lvl, 0] = int(shape[0])
            shape_table[lvl, 1] = int(shape[1])
            shape_table[lvl, 2] = int(shape[2])
            dtheta_table[lvl] = float(self._dtheta_by_level[lvl])
            dphi_table[lvl] = float(self._dphi_by_level[lvl])
            bin_level_offset[lvl] = running_offset
            running_offset += int(shape[1]) * int(shape[2])
        self._shape_table = shape_table
        self._dtheta_table = dtheta_table
        self._dphi_table = dphi_table
        self._bin_level_offset = bin_level_offset
        self._leaf_shape = np.asarray(self.tree.leaf_shape, dtype=np.int64)
        self._tree_depth = int(self.tree.max_level)
        points_r = np.linalg.norm(self._points, axis=1)
        cell_r_min = np.min(points_r[self._corners], axis=1)
        cell_r_max = np.max(points_r[self._corners], axis=1)
        theta_points = np.arccos(
            np.clip(self._points[:, 2] / np.maximum(points_r, np.finfo(float).tiny), -1.0, 1.0)
        )
        phi_points = np.mod(np.arctan2(self._points[:, 1], self._points[:, 0]), 2.0 * math.pi)
        rho_points = np.hypot(self._points[:, 0], self._points[:, 1])
        axis_mask = rho_points[self._corners] <= self._axis_rho_tol
        self._cell_r_min = cell_r_min
        self._cell_r_max = cell_r_max
        self._r_min = float(np.min(cell_r_min))
        self._r_max = float(np.max(cell_r_max))
        radial_tol = 1e-7 * max(float(self._r_max - self._r_min), 1.0)
        radial_edges, radial_edge_tol = self._cluster_close_values(
            np.concatenate((self._cell_r_min[valid], self._cell_r_max[valid])),
            atol=radial_tol,
        )
        expected_edges = int(self._leaf_shape[0]) + 1
        if radial_edges.size != expected_edges:
            raise ValueError(
                "Spherical radial edge count does not match leaf_shape: "
                f"edges={int(radial_edges.size)}, expected={expected_edges}."
            )
        self._radial_edges = radial_edges
        self._radial_edge_tol = radial_edge_tol
        self._cell_theta_min = np.min(theta_points[self._corners], axis=1)
        self._cell_theta_max = np.max(theta_points[self._corners], axis=1)

        phi_start = np.empty(n_cells, dtype=float)
        phi_width = np.empty(n_cells, dtype=float)
        phi_corners = phi_points[self._corners]
        for cid in range(n_cells):
            vals = phi_corners[cid, ~axis_mask[cid]]
            if vals.size < 2:
                vals = phi_corners[cid]
            start, width = self._minimal_phi_interval(vals)
            phi_start[cid] = start
            phi_width[cid] = width
        self._cell_phi_start = phi_start
        self._cell_phi_width = phi_width

        self._i0 = np.full(n_cells, -1, dtype=np.int64)
        self._i1 = np.full(n_cells, -1, dtype=np.int64)
        self._i2 = np.full(n_cells, -1, dtype=np.int64)
        self._init_exact_addresses()
        n_bins = int(running_offset)
        bin_lists: list[list[int]] = [[] for _ in range(n_bins)]
        for cid in np.flatnonzero(valid):
            level = int(self._cell_level[cid])
            tt = int(self._i1[cid])
            pp = int(self._i2[cid])
            nphi = int(self._shape_table[level, 2])
            key = int(self._bin_level_offset[level] + tt * nphi + pp)
            bin_lists[key].append(int(cid))

        bin_counts = np.zeros(n_bins, dtype=np.int64)
        for key, ids in enumerate(bin_lists):
            if not ids:
                continue
            arr = np.array(ids, dtype=np.int64)
            order = np.argsort(self._i0[arr], kind="stable")
            sorted_ids = arr[order]
            bin_lists[key] = sorted_ids.tolist()
            bin_counts[key] = int(sorted_ids.size)

        bin_offsets = np.zeros(n_bins + 1, dtype=np.int64)
        if n_bins > 0:
            np.cumsum(bin_counts, out=bin_offsets[1:])
        total_refs = int(bin_offsets[-1])
        bin_cell_ids = np.empty(total_refs, dtype=np.int64)
        for key in range(n_bins):
            start = int(bin_offsets[key])
            end = int(bin_offsets[key + 1])
            if end <= start:
                continue
            ids = bin_lists[key]
            bin_cell_ids[start:end] = np.array(ids, dtype=np.int64)

        self._bin_counts = bin_counts
        self._bin_offsets = bin_offsets
        self._bin_cell_ids = bin_cell_ids
        self._lookup_state = SphericalLookupKernelState(
            levels_desc=self._levels_desc,
            shape_table=self._shape_table,
            dtheta_table=self._dtheta_table,
            dphi_table=self._dphi_table,
            bin_level_offset=self._bin_level_offset,
            bin_offsets=self._bin_offsets,
            bin_cell_ids=self._bin_cell_ids,
            cell_r_min=self._cell_r_min,
            cell_r_max=self._cell_r_max,
            cell_theta_min=self._cell_theta_min,
            cell_theta_max=self._cell_theta_max,
            cell_phi_start=self._cell_phi_start,
            cell_phi_width=self._cell_phi_width,
            cell_valid=(self._cell_level >= 0),
            cell_centers=self._cell_centers,
            r_min=float(self._r_min),
            r_max=float(self._r_max),
            radial_edges=self._radial_edges,
            max_radius=int(_DEFAULT_LOOKUP_MAX_RADIUS),
            leaf_shape=self._leaf_shape,
            tree_depth=int(self._tree_depth),
            cell_i0=self._i0,
            cell_i1=self._i1,
            cell_i2=self._i2,
            node_depth=self._node_depth,
            node_i0=self._node_i0,
            node_i1=self._node_i1,
            node_i2=self._node_i2,
            node_value=self._node_value,
        )

    @staticmethod
    def _wrapped_delta(a: float, b: float) -> float:
        """Return the signed shortest wrapped azimuth difference `a - b`."""
        return float((a - b + math.pi) % _TWO_PI - math.pi)

    def _init_exact_addresses(self) -> None:
        """Derive exact `(level, i0, i1, i2)` addresses from spherical cell bounds."""
        valid_ids = np.flatnonzero(self._cell_level >= 0).astype(np.int64)
        if valid_ids.size == 0:
            raise ValueError("Spherical lookup requires at least one valid cell address.")
        r_tol = 1e-7 * max(float(self._r_max - self._r_min), 1.0)
        theta_tol = 1e-7 * math.pi
        phi_tol = 1e-7 * _TWO_PI
        tree_depth = int(self._tree_depth)
        depths = np.asarray(self._cell_level[valid_ids], dtype=np.int64)
        shifts = np.asarray(tree_depth - depths, dtype=np.int64)
        if np.any(shifts < 0):
            bad = int(valid_ids[np.flatnonzero(shifts < 0)[0]])
            raise ValueError(f"Spherical cell {bad} depth exceeds tree_depth={tree_depth}.")
        width_units = np.left_shift(np.ones_like(shifts, dtype=np.int64), shifts)

        radial_edges = np.asarray(self._radial_edges, dtype=float)
        radial_edge_tol = np.asarray(self._radial_edge_tol, dtype=float)
        r0_search = np.searchsorted(radial_edges, self._cell_r_min[valid_ids], side="left").astype(np.int64)
        r1_search = np.searchsorted(radial_edges, self._cell_r_max[valid_ids], side="left").astype(np.int64)
        r0_next = np.clip(r0_search, 0, radial_edges.size - 1)
        r1_next = np.clip(r1_search, 0, radial_edges.size - 1)
        r0_prev = np.clip(r0_search - 1, 0, radial_edges.size - 1)
        r1_prev = np.clip(r1_search - 1, 0, radial_edges.size - 1)
        r0_use_prev = (r0_search > 0) & (
            np.abs(radial_edges[r0_prev] - self._cell_r_min[valid_ids])
            <= np.abs(radial_edges[r0_next] - self._cell_r_min[valid_ids])
        )
        r1_use_prev = (r1_search > 0) & (
            np.abs(radial_edges[r1_prev] - self._cell_r_max[valid_ids])
            <= np.abs(radial_edges[r1_next] - self._cell_r_max[valid_ids])
        )
        r0_f = np.where(r0_use_prev, r0_prev, r0_next).astype(np.int64)
        r1_f = np.where(r1_use_prev, r1_prev, r1_next).astype(np.int64)
        n_r_fine = int(self._leaf_shape[0])
        if np.any(r0_f < 0) or np.any(r1_f > n_r_fine):
            bad = int(valid_ids[np.flatnonzero((r0_f < 0) | (r1_f > n_r_fine))[0]])
            raise ValueError(f"Spherical cell {bad} radial address is outside the inferred octree grid.")
        r0_ok = np.abs(radial_edges[r0_f] - self._cell_r_min[valid_ids]) <= radial_edge_tol[r0_f]
        r1_ok = np.abs(radial_edges[r1_f] - self._cell_r_max[valid_ids]) <= radial_edge_tol[r1_f]
        if np.any(~r0_ok):
            bad = int(valid_ids[np.flatnonzero(~r0_ok)[0]])
            raise ValueError(f"Spherical cell {bad} r-min does not align with the inferred octree grid.")
        if np.any(~r1_ok):
            bad = int(valid_ids[np.flatnonzero(~r1_ok)[0]])
            raise ValueError(f"Spherical cell {bad} r-max does not align with the inferred octree grid.")

        d_theta_f = math.pi / float(int(self._leaf_shape[1]))
        d_phi_f = _TWO_PI / float(int(self._leaf_shape[2]))
        width_phi = np.asarray(self._cell_phi_width[valid_ids], dtype=float)
        if np.any(width_phi >= (_TWO_PI - phi_tol)):
            bad = int(valid_ids[np.flatnonzero(width_phi >= (_TWO_PI - phi_tol))[0]])
            raise ValueError(f"Spherical cell {bad} spans the full azimuth and has no unique octree address.")

        i1_f = np.rint(self._cell_theta_min[valid_ids] / d_theta_f).astype(np.int64)
        i1_hi = np.rint(self._cell_theta_max[valid_ids] / d_theta_f).astype(np.int64)
        i2_f = np.rint(self._cell_phi_start[valid_ids] / d_phi_f).astype(np.int64)
        i2_hi = np.rint((self._cell_phi_start[valid_ids] + width_phi) / d_phi_f).astype(np.int64)

        n_theta_fine = int(self._leaf_shape[1])
        n_phi_fine = int(self._leaf_shape[2])
        in_bounds = (
            (i1_f >= 0) & (i1_hi <= n_theta_fine)
            & (i2_f >= 0) & (i2_hi <= n_phi_fine)
        )
        if not np.all(in_bounds):
            bad = int(valid_ids[np.flatnonzero(~in_bounds)[0]])
            raise ValueError(f"Spherical cell {bad} address is outside the inferred octree grid.")

        if np.any(~np.isclose(self._cell_theta_min[valid_ids], i1_f * d_theta_f, rtol=0.0, atol=theta_tol)):
            bad = int(valid_ids[np.flatnonzero(~np.isclose(self._cell_theta_min[valid_ids], i1_f * d_theta_f, rtol=0.0, atol=theta_tol))[0]])
            raise ValueError(f"Spherical cell {bad} theta-min does not align with the inferred octree grid.")
        if np.any(~np.isclose(self._cell_theta_max[valid_ids], i1_hi * d_theta_f, rtol=0.0, atol=theta_tol)):
            bad = int(valid_ids[np.flatnonzero(~np.isclose(self._cell_theta_max[valid_ids], i1_hi * d_theta_f, rtol=0.0, atol=theta_tol))[0]])
            raise ValueError(f"Spherical cell {bad} theta-max does not align with the inferred octree grid.")
        phi_delta = np.abs((self._cell_phi_start[valid_ids] - (i2_f * d_phi_f) + math.pi) % _TWO_PI - math.pi)
        if np.any(phi_delta > phi_tol):
            bad = int(valid_ids[np.flatnonzero(phi_delta > phi_tol)[0]])
            raise ValueError(f"Spherical cell {bad} phi-start does not align with the inferred octree grid.")
        if np.any(~np.isclose(width_phi, (i2_hi - i2_f) * d_phi_f, rtol=0.0, atol=phi_tol)):
            bad = int(valid_ids[np.flatnonzero(~np.isclose(width_phi, (i2_hi - i2_f) * d_phi_f, rtol=0.0, atol=phi_tol))[0]])
            raise ValueError(f"Spherical cell {bad} phi-width does not align with the inferred octree grid.")

        spans_ok = (
            ((r1_f - r0_f) == width_units)
            & ((i1_hi - i1_f) == width_units)
            & ((i2_hi - i2_f) == width_units)
        )
        if not np.all(spans_ok):
            bad = int(valid_ids[np.flatnonzero(~spans_ok)[0]])
            raise ValueError(f"Spherical cell {bad} width does not match inferred level {int(self._cell_level[bad])}.")

        aligned = (
            ((r0_f % width_units) == 0)
            & ((i1_f % width_units) == 0)
            & ((i2_f % width_units) == 0)
        )
        if not np.all(aligned):
            bad = int(valid_ids[np.flatnonzero(~aligned)[0]])
            raise ValueError(f"Spherical cell {bad} fine-grid origin is not aligned to its inferred level.")

        i0 = np.right_shift(r0_f, shifts)
        i1 = np.right_shift(i1_f, shifts)
        i2 = np.right_shift(i2_f, shifts)
        self._i0[valid_ids] = i0
        self._i1[valid_ids] = i1
        self._i2[valid_ids] = i2

        leaf_depth = depths
        leaf_i0 = i0
        leaf_i1 = i1
        leaf_i2 = i2
        leaf_value = valid_ids
        leaf_order = np.lexsort((leaf_i2, leaf_i1, leaf_i0, leaf_depth))
        leaf_depth = leaf_depth[leaf_order]
        leaf_i0 = leaf_i0[leaf_order]
        leaf_i1 = leaf_i1[leaf_order]
        leaf_i2 = leaf_i2[leaf_order]
        leaf_value = leaf_value[leaf_order]
        same_leaf = (
            (leaf_depth[1:] == leaf_depth[:-1])
            & (leaf_i0[1:] == leaf_i0[:-1])
            & (leaf_i1[1:] == leaf_i1[:-1])
            & (leaf_i2[1:] == leaf_i2[:-1])
        )
        if np.any(same_leaf):
            dup = int(np.flatnonzero(same_leaf)[0])
            raise ValueError(
                f"Spherical cells overlap at octree address "
                f"{(int(leaf_depth[dup]), int(leaf_i0[dup]), int(leaf_i1[dup]), int(leaf_i2[dup]))}."
            )

        node_depth_parts = [leaf_depth]
        node_i0_parts = [leaf_i0]
        node_i1_parts = [leaf_i1]
        node_i2_parts = [leaf_i2]
        node_value_parts = [leaf_value]
        for parent_depth in range(tree_depth):
            mask = depths > int(parent_depth)
            if not np.any(mask):
                continue
            up = np.asarray(depths[mask] - int(parent_depth), dtype=np.int64)
            parent_nodes = np.column_stack(
                (
                    np.full(int(np.count_nonzero(mask)), int(parent_depth), dtype=np.int64),
                    np.right_shift(i0[mask], up),
                    np.right_shift(i1[mask], up),
                    np.right_shift(i2[mask], up),
                )
            )
            parent_nodes = np.unique(parent_nodes, axis=0)
            node_depth_parts.append(parent_nodes[:, 0].astype(np.int64, copy=False))
            node_i0_parts.append(parent_nodes[:, 1].astype(np.int64, copy=False))
            node_i1_parts.append(parent_nodes[:, 2].astype(np.int64, copy=False))
            node_i2_parts.append(parent_nodes[:, 3].astype(np.int64, copy=False))
            node_value_parts.append(np.full(parent_nodes.shape[0], _INTERNAL_NODE_VALUE, dtype=np.int64))

        node_depth = np.concatenate(node_depth_parts)
        node_i0 = np.concatenate(node_i0_parts)
        node_i1 = np.concatenate(node_i1_parts)
        node_i2 = np.concatenate(node_i2_parts)
        node_value = np.concatenate(node_value_parts)
        node_order = np.lexsort((node_i2, node_i1, node_i0, node_depth))
        node_depth = node_depth[node_order]
        node_i0 = node_i0[node_order]
        node_i1 = node_i1[node_order]
        node_i2 = node_i2[node_order]
        node_value = node_value[node_order]
        same_node = (
            (node_depth[1:] == node_depth[:-1])
            & (node_i0[1:] == node_i0[:-1])
            & (node_i1[1:] == node_i1[:-1])
            & (node_i2[1:] == node_i2[:-1])
        )
        if np.any(same_node):
            dup = int(np.flatnonzero(same_node)[0])
            raise ValueError(
                "Spherical cells overlap across parent/child addresses at "
                f"({int(node_depth[dup])}, {int(node_i0[dup])}, {int(node_i1[dup])}, {int(node_i2[dup])})."
            )
        self._node_depth = node_depth
        self._node_i0 = node_i0
        self._node_i1 = node_i1
        self._node_i2 = node_i2
        self._node_value = node_value

    @staticmethod
    def _path(i0: int, i1: int, i2: int, level: int) -> GridPath:
        """Build the root-to-leaf grid index path for one cell."""
        out: list[GridIndex] = []
        for path_level in range(level + 1):
            shift = level - path_level
            out.append((i0 >> shift, i1 >> shift, i2 >> shift))
        return tuple(out)

    def _cell_bounds_xyz(self, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
        cid = int(cell_id)
        corners = np.asarray(self._corners[cid], dtype=np.int64)
        pts = np.asarray(self._points[corners], dtype=float)
        return np.min(pts, axis=0), np.max(pts, axis=0)

    def _cell_bounds_rpa(self, cell_id: int) -> tuple[np.ndarray, np.ndarray]:
        cid = int(cell_id)
        return (
            np.array([self._cell_r_min[cid], self._cell_theta_min[cid], self._cell_phi_start[cid]], dtype=float),
            np.array(
                [
                    self._cell_r_max[cid],
                    self._cell_theta_max[cid],
                    (self._cell_phi_start[cid] + self._cell_phi_width[cid]) % (2.0 * np.pi),
                ],
                dtype=float,
            ),
        )

    def _domain_bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(self._points, dtype=float)
        return np.min(pts, axis=0), np.max(pts, axis=0)

    def _domain_bounds_rpa(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self._r_min, 0.0, 0.0], dtype=float), np.array([self._r_max, np.pi, 2.0 * np.pi], dtype=float)

    @staticmethod
    def _minimal_phi_interval(values: np.ndarray) -> tuple[float, float]:
        """Return the smallest wrapped azimuth interval covering the samples."""
        vals = np.sort(np.mod(np.array(values, dtype=float), 2.0 * math.pi))
        if vals.size == 0:
            return 0.0, 2.0 * math.pi
        if vals.size == 1:
            return float(vals[0]), 0.0
        wrapped = np.concatenate((vals, vals[:1] + 2.0 * math.pi))
        gaps = np.diff(wrapped)
        k = int(np.argmax(gaps))
        start = float(wrapped[k + 1] % (2.0 * math.pi))
        width = float((2.0 * math.pi) - gaps[k])
        return start, width

    def _candidate_ids(self, level: int, itheta: int, iphi: int, radius: int) -> np.ndarray:
        """Return candidate cell ids from nearby angular bins."""
        if level < 0 or level >= self._shape_table.shape[0]:
            return np.array([], dtype=np.int64)
        ntheta = int(self._shape_table[level, 1])
        nphi = int(self._shape_table[level, 2])
        if ntheta <= 0 or nphi <= 0:
            return np.array([], dtype=np.int64)
        level_offset = int(self._bin_level_offset[level])
        if level_offset < 0:
            return np.array([], dtype=np.int64)
        total = 0
        for dt in range(-radius, radius + 1):
            tt = itheta + dt
            if tt < 0 or tt >= ntheta:
                continue
            for dp in range(-radius, radius + 1):
                pp = (iphi + dp) % nphi
                key = int(level_offset + tt * nphi + pp)
                total += int(self._bin_counts[key])
        if total <= 0:
            return np.array([], dtype=np.int64)
        out = np.empty(total, dtype=np.int64)
        cursor = 0
        for dt in range(-radius, radius + 1):
            tt = itheta + dt
            if tt < 0 or tt >= ntheta:
                continue
            for dp in range(-radius, radius + 1):
                pp = (iphi + dp) % nphi
                key = int(level_offset + tt * nphi + pp)
                count = int(self._bin_counts[key])
                if count <= 0:
                    continue
                start = int(self._bin_offsets[key])
                end = int(self._bin_offsets[key + 1])
                out[cursor : cursor + count] = self._bin_cell_ids[start:end]
                cursor += count
        return out

    def _contains_rpa(
        self,
        cids: np.ndarray,
        r: float,
        polar: float,
        azimuth: float,
    ) -> np.ndarray:
        """Return a boolean mask for which candidate cells contain the point."""
        tol = 1e-10
        if cids.size == 0:
            return np.array([], dtype=np.bool_)
        ok_r = (r >= (self._cell_r_min[cids] - tol)) & (r <= (self._cell_r_max[cids] + tol))
        ok_t = (polar >= (self._cell_theta_min[cids] - tol)) & (
            polar <= (self._cell_theta_max[cids] + tol)
        )
        starts = self._cell_phi_start[cids]
        widths = self._cell_phi_width[cids]
        dphi = np.mod(azimuth - starts, 2.0 * math.pi)
        ok_p = (widths >= (2.0 * math.pi - tol)) | (dphi <= (widths + tol))
        return ok_r & ok_t & ok_p

    def lookup_cell_id(
        self,
        point: np.ndarray,
        *,
        coord: str,
    ) -> int:
        """Return the containing cell id for one point in `xyz` or `rpa`."""
        q = np.array(point, dtype=float).reshape(3)
        resolved = str(coord)
        if resolved == "xyz":
            return self._lookup_xyz_cell_id(float(q[0]), float(q[1]), float(q[2]))
        if resolved == "rpa":
            return self._lookup_rpa_cell_id(float(q[0]), float(q[1]), float(q[2]))
        raise ValueError("coord must be 'xyz' or 'rpa'.")

    def contains_cell(
        self,
        cell_id: int,
        point: np.ndarray,
        *,
        coord: str,
        tol: float = 1e-10,
    ) -> bool:
        """Return whether one point lies inside one cell."""
        q = np.array(point, dtype=float).reshape(3)
        resolved = str(coord)
        if resolved == "xyz":
            return self._contains_xyz_cell(
                int(cell_id),
                float(q[0]),
                float(q[1]),
                float(q[2]),
                tol=float(tol),
            )
        if resolved == "rpa":
            return self._contains_rpa_cell(
                int(cell_id),
                float(q[0]),
                float(q[1]),
                float(q[2]),
                tol=float(tol),
            )
        raise ValueError("coord must be 'xyz' or 'rpa'.")

    def _lookup_rpa_cell_id(self, r: float, polar: float, azimuth: float) -> int:
        """Return the containing spherical cell id, or `-1` when not found."""
        return int(
            _lookup_rpa_cell_id_kernel(
                float(r),
                float(polar),
                float(azimuth),
                self._lookup_state,
            )
        )

    def _contains_rpa_cell(self, cell_id: int, r: float, polar: float, azimuth: float, *, tol: float = 1e-10) -> bool:
        """Return whether one spherical point lies inside one cell."""
        cid = int(cell_id)
        rr = float(r)
        pp = float(polar)
        aa = float(azimuth)
        t = float(tol)
        if rr < (float(self._cell_r_min[cid]) - t) or rr > (float(self._cell_r_max[cid]) + t):
            return False
        if pp < (float(self._cell_theta_min[cid]) - t) or pp > (float(self._cell_theta_max[cid]) + t):
            return False
        if abs(rr * math.sin(pp)) <= t:
            return True
        start = float(self._cell_phi_start[cid])
        width = float(self._cell_phi_width[cid])
        dphi = float((aa - start) % (2.0 * math.pi))
        if width >= (2.0 * math.pi - t):
            return True
        return dphi <= (width + t)

    def _contains_xyz_cell(self, cell_id: int, x: float, y: float, z: float, *, tol: float = 1e-10) -> bool:
        """Return whether one Cartesian point lies inside one cell."""
        r = float(math.sqrt(x * x + y * y + z * z))
        if r == 0.0:
            polar = 0.0
        else:
            polar = float(math.acos(max(-1.0, min(1.0, z / r))))
        azimuth = float(math.atan2(y, x) % (2.0 * math.pi))
        return self._contains_rpa_cell(int(cell_id), r, polar, azimuth, tol=float(tol))

    def cell_step_hint(self, cell_id: int) -> float:
        """Return an initial step-size hint for Python ray tracing."""
        cid = int(cell_id)
        r_span = float(self._cell_r_max[cid] - self._cell_r_min[cid])
        theta_span = float(self._cell_theta_max[cid] - self._cell_theta_min[cid])
        phi_span = float(min(self._cell_phi_width[cid], 2.0 * math.pi))
        length_scale = max(float(self._cell_r_max[cid]), 1.0)
        return float(max(r_span, length_scale * theta_span, length_scale * phi_span, 1e-6))

    def _lookup_xyz_cell_id(self, x: float, y: float, z: float) -> int:
        """Return the containing cell id for `(x, y, z)`, or `-1`."""
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return -1
        r = float(math.sqrt(x * x + y * y + z * z))
        if r == 0.0:
            polar = 0.0
        else:
            polar = float(math.acos(max(-1.0, min(1.0, z / r))))
        azimuth = float(math.atan2(y, x) % (2.0 * math.pi))
        return self._lookup_rpa_cell_id(r, polar, azimuth)

    def hit_from_chosen(self, chosen: int, *, allow_invalid_level: bool = False) -> LookupHit | None:
        """Build a `LookupHit` from an internal cell id."""
        if chosen < 0:
            return None
        center = self._cell_centers[chosen]
        level = int(self._cell_level[chosen])
        if level < 0 and not allow_invalid_level:
            return None
        if level < 0:
            path_level = int(self.tree.max_level)
        else:
            path_level = int(level)
            if path_level < 0:
                raise ValueError(f"Derived negative level {level}; max_level={self.tree.max_level}.")
        cell_i0 = int(self._i0[chosen])
        cell_i1 = int(self._i1[chosen])
        cell_i2 = int(self._i2[chosen])
        return LookupHit(
            cell_id=chosen,
            level=level,
            i0=cell_i0,
            i1=cell_i1,
            i2=cell_i2,
            path=self._path(cell_i0, cell_i1, cell_i2, path_level),
            center_xyz=(float(center[0]), float(center[1]), float(center[2])),
        )



class SphericalOctree(_SphericalCellLookup, Octree):
    """Octree specialization for spherical `(r, polar, azimuth)` datasets."""

    TREE_COORD: ClassVar[str | None] = "rpa"

    def lookup_local(self, xyz: np.ndarray, near_cid: int | None = None) -> "LookupHit | None":
        """Lookup in `xyz`, first trying cells near `near_cid` when provided."""
        q = np.array(xyz, dtype=float)
        x = float(q[0])
        y = float(q[1])
        z = float(q[2])
        self._require_lookup()
        lookup = self
        if near_cid is not None and int(near_cid) >= 0:
            near = int(near_cid)
            if self.contains_cell(near, q, coord="xyz"):
                return self.hit_from_cell_id(near)

            near_level = int(lookup._cell_level[near])
            near_i1 = int(lookup._i1[near])
            near_i2 = int(lookup._i2[near])
            shape_table = lookup._shape_table
            near_shape: np.ndarray | None = None
            if 0 <= near_level < shape_table.shape[0] and int(shape_table[near_level, 0]) > 0:
                near_shape = shape_table[near_level]
            candidate_arrays: list[np.ndarray] = []
            for level in (near_level, near_level - 1, near_level + 1):
                if level < 0 or level >= shape_table.shape[0]:
                    continue
                shape = shape_table[level]
                if int(shape[0]) <= 0:
                    continue
                ntheta = int(shape[1])
                nphi = int(shape[2])
                if near_shape is None:
                    mapped_i1 = near_i1
                    mapped_i2 = near_i2
                else:
                    mapped_i1 = int(
                        np.clip(
                            round(((near_i1 + 0.5) * shape[1] / near_shape[1]) - 0.5),
                            0,
                            ntheta - 1,
                        )
                    )
                    mapped_i2 = int(
                        np.clip(
                            round(((near_i2 + 0.5) * shape[2] / near_shape[2]) - 0.5),
                            0,
                            nphi - 1,
                        )
                    )
                for radius in (0, 1):
                    cands = lookup._candidate_ids(int(level), mapped_i1, mapped_i2, radius)
                    if cands.size > 0:
                        candidate_arrays.append(cands)

            if candidate_arrays:
                candidates = np.unique(np.concatenate(candidate_arrays))
                r, polar, azimuth = _xyz_to_rpa_components(float(q[0]), float(q[1]), float(q[2]))
                inside = lookup._contains_rpa(candidates, r, polar, azimuth)
                if np.any(inside):
                    valid = candidates[inside]
                    d = np.linalg.norm(lookup._cell_centers[valid] - q, axis=1)
                    return self.hit_from_cell_id(int(valid[int(np.argmin(d))]))

        return self.lookup_point(np.array([x, y, z], dtype=float), coord="xyz")

    def build_lookup(
        self,
    ) -> None:
        """Build lookup arrays for this spherical tree."""
        if self.ds is None or self.ds.corners is None:
            raise ValueError("Octree is not bound to a dataset. Call bind(...) before lookup.")
        self._init_lookup_state(self)

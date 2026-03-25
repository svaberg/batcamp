#!/usr/bin/env python3
"""Spherical octree and spherical lookup implementation."""

from __future__ import annotations

import math
from typing import NamedTuple

from numba import njit
import numpy as np

from .octree import GridIndex
from .octree import GridPath
from .octree import LookupHit
from .octree import Octree

_TWO_PI = 2.0 * math.pi
_LOOKUP_CONTAIN_TOL = 1e-10
_MISSING_NODE_VALUE = -1


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
    node_value: np.ndarray
    node_child: np.ndarray
    root_node_ids: np.ndarray
    node_parent: np.ndarray
    cell_node_id: np.ndarray
    node_r_min: np.ndarray
    node_r_max: np.ndarray
    node_theta_min: np.ndarray
    node_theta_max: np.ndarray
    node_phi_start: np.ndarray
    node_phi_width: np.ndarray

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
def _contains_rpa_node(
    node_id: int,
    r: float,
    polar: float,
    azimuth: float,
    lookup_state: SphericalLookupKernelState,
) -> bool:
    """Return whether one spherical point lies inside one occupied node."""
    nid = int(node_id)
    tol = _LOOKUP_CONTAIN_TOL
    if r < (lookup_state.node_r_min[nid] - tol) or r > (lookup_state.node_r_max[nid] + tol):
        return False
    if polar < (lookup_state.node_theta_min[nid] - tol) or polar > (lookup_state.node_theta_max[nid] + tol):
        return False
    if abs(r * math.sin(polar)) <= tol:
        return True
    width = lookup_state.node_phi_width[nid]
    dphi = (azimuth - lookup_state.node_phi_start[nid]) % _TWO_PI
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

    current = -1
    if prev_cid >= 0:
        current = int(lookup_state.cell_node_id[int(prev_cid)])
        while current >= 0:
            if _contains_rpa_node(current, r, polar, azimuth, lookup_state):
                break
            current = int(lookup_state.node_parent[current])
    if current < 0:
        for root_pos in range(int(lookup_state.root_node_ids.shape[0])):
            node_id = int(lookup_state.root_node_ids[root_pos])
            if _contains_rpa_node(node_id, r, polar, azimuth, lookup_state):
                current = node_id
                break
    if current < 0:
        return -1

    while True:
        node_value = int(lookup_state.node_value[current])
        if node_value >= 0:
            cid = int(node_value)
            if _contains_rpa_cell(cid, r, polar, azimuth, lookup_state):
                return cid
            return -1

        found_child = False
        for child_ord in range(8):
            child_id = int(lookup_state.node_child[current, child_ord])
            if child_id < 0:
                continue
            if _contains_rpa_node(child_id, r, polar, azimuth, lookup_state):
                current = child_id
                found_child = True
                break
        if not found_child:
            return -1

class _SphericalCellLookup:
    """Cell lookup helper for spherical trees."""

    def _init_lookup_state(self) -> None:
        """Pack builder-provided spherical tree geometry into lookup state."""
        if self.ds is None or self.ds.corners is None:
            raise ValueError("Lookup requires a bound octree with dataset and corners.")
        required = (
            "_corners",
            "_points",
            "_cell_centers",
            "_cell_r_min",
            "_cell_r_max",
            "_r_min",
            "_r_max",
            "_cell_theta_min",
            "_cell_theta_max",
            "_cell_phi_start",
            "_cell_phi_width",
            "_i0",
            "_i1",
            "_i2",
            "_node_depth",
            "_node_i0",
            "_node_i1",
            "_node_i2",
            "_node_value",
            "_radial_edges",
            "_node_child",
            "_root_node_ids",
            "_node_parent",
            "_cell_node_id",
            "_node_r_min",
            "_node_r_max",
            "_node_theta_min",
            "_node_theta_max",
            "_node_phi_start",
            "_node_phi_width",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError(f"Spherical lookup requires builder-provided geometry state: missing {missing}.")
        self._corners = np.asarray(self._corners, dtype=np.int64)
        self._points = np.asarray(self._points, dtype=np.float64)
        self._cell_centers = np.asarray(self._cell_centers, dtype=np.float64)
        self._cell_r_min = np.asarray(self._cell_r_min, dtype=np.float64)
        self._cell_r_max = np.asarray(self._cell_r_max, dtype=np.float64)
        self._cell_theta_min = np.asarray(self._cell_theta_min, dtype=np.float64)
        self._cell_theta_max = np.asarray(self._cell_theta_max, dtype=np.float64)
        self._cell_phi_start = np.asarray(self._cell_phi_start, dtype=np.float64)
        self._cell_phi_width = np.asarray(self._cell_phi_width, dtype=np.float64)
        self._r_min = float(self._r_min)
        self._r_max = float(self._r_max)
        n_cells = int(self._cell_centers.shape[0])
        if self.cell_levels is None or int(self.cell_levels.shape[0]) != n_cells:
            raise ValueError("Spherical lookup requires exact builder-provided cell_levels.")
        self._cell_level = np.array(self.cell_levels, dtype=np.int64)
        self._i0 = np.asarray(self._i0, dtype=np.int64)
        self._i1 = np.asarray(self._i1, dtype=np.int64)
        self._i2 = np.asarray(self._i2, dtype=np.int64)
        self._node_depth = np.asarray(self._node_depth, dtype=np.int64)
        self._node_i0 = np.asarray(self._node_i0, dtype=np.int64)
        self._node_i1 = np.asarray(self._node_i1, dtype=np.int64)
        self._node_i2 = np.asarray(self._node_i2, dtype=np.int64)
        self._node_value = np.asarray(self._node_value, dtype=np.int64)
        self._node_child = np.asarray(self._node_child, dtype=np.int64)
        self._root_node_ids = np.asarray(self._root_node_ids, dtype=np.int64)
        self._node_parent = np.asarray(self._node_parent, dtype=np.int64)
        self._cell_node_id = np.asarray(self._cell_node_id, dtype=np.int64)
        self._node_r_min = np.asarray(self._node_r_min, dtype=np.float64)
        self._node_r_max = np.asarray(self._node_r_max, dtype=np.float64)
        self._node_theta_min = np.asarray(self._node_theta_min, dtype=np.float64)
        self._node_theta_max = np.asarray(self._node_theta_max, dtype=np.float64)
        self._node_phi_start = np.asarray(self._node_phi_start, dtype=np.float64)
        self._node_phi_width = np.asarray(self._node_phi_width, dtype=np.float64)
        self._radial_edges = np.asarray(self._radial_edges, dtype=np.float64)
        self._lookup_state = SphericalLookupKernelState(
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
            node_value=self._node_value,
            node_child=self._node_child,
            root_node_ids=self._root_node_ids,
            node_parent=self._node_parent,
            cell_node_id=self._cell_node_id,
            node_r_min=self._node_r_min,
            node_r_max=self._node_r_max,
            node_theta_min=self._node_theta_min,
            node_theta_max=self._node_theta_max,
            node_phi_start=self._node_phi_start,
            node_phi_width=self._node_phi_width,
        )

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
            return _SphericalCellLookup._lookup_xyz_cell_id(self, float(q[0]), float(q[1]), float(q[2]))
        if resolved == "rpa":
            return _SphericalCellLookup._lookup_rpa_cell_id(self, float(q[0]), float(q[1]), float(q[2]))
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
            return _SphericalCellLookup._contains_xyz_cell(
                self,
                int(cell_id),
                float(q[0]),
                float(q[1]),
                float(q[2]),
                tol=float(tol),
            )
        if resolved == "rpa":
            return _SphericalCellLookup._contains_rpa_cell(
                self,
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
        return _SphericalCellLookup._contains_rpa_cell(self, int(cell_id), r, polar, azimuth, tol=float(tol))

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
        return _SphericalCellLookup._lookup_rpa_cell_id(self, r, polar, azimuth)

    def hit_from_chosen(self, chosen: int, *, allow_invalid_level: bool = False) -> LookupHit | None:
        """Build a `LookupHit` from an internal cell id."""
        if chosen < 0:
            return None
        center = self._cell_centers[chosen]
        level = int(self._cell_level[chosen])
        if level < 0 and not allow_invalid_level:
            return None
        if level < 0:
            path_level = int(self.max_level)
        else:
            path_level = int(level)
            if path_level < 0:
                raise ValueError(f"Derived negative level {level}; max_level={self.max_level}.")
        cell_i0 = int(self._i0[chosen])
        cell_i1 = int(self._i1[chosen])
        cell_i2 = int(self._i2[chosen])
        return LookupHit(
            cell_id=chosen,
            level=level,
            i0=cell_i0,
            i1=cell_i1,
            i2=cell_i2,
            path=_SphericalCellLookup._path(cell_i0, cell_i1, cell_i2, path_level),
            center_xyz=(float(center[0]), float(center[1]), float(center[2])),
        )

#!/usr/bin/env python3
"""Spherical coordinate support for octree lookup."""

from __future__ import annotations

import math

from numba import njit
import numpy as np

from .constants import XYZ_VARS
from .octree import AXIS0
from .octree import AXIS1
from .octree import AXIS2
from .octree import _bound_xyz_and_leaf_levels
from .octree import START
from .octree import WIDTH

_TWO_PI = 2.0 * math.pi

class _SphericalCoordSupport:
    """Spherical geometry support for octree lookup."""

    def _attach_coord_state(self, ds, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """Derive spherical cell bounds, domain bounds, and axis-2 periodic metadata from the bound dataset."""
        corners, x, y, z, cell_levels = _bound_xyz_and_leaf_levels(self, ds, corners)
        n_cells = int(corners.shape[0])
        valid_ids = np.flatnonzero(cell_levels >= 0)
        shifts = int(self.max_level) - cell_levels[valid_ids]
        width_units = np.left_shift(np.ones_like(shifts, dtype=np.int64), shifts)
        r0_f = np.left_shift(self._cell_ijk[valid_ids, AXIS0], shifts)
        r1_f = r0_f + width_units
        n_r_edges = int(self.leaf_shape[0]) + 1
        radial_sum = np.zeros(n_r_edges, dtype=np.float64)
        radial_count = np.zeros(n_r_edges, dtype=np.int64)
        point_r = np.sqrt(x * x + y * y + z * z)
        cell_r_lo_obs = np.min(point_r[corners], axis=1)
        cell_r_hi_obs = np.max(point_r[corners], axis=1)
        for row, edge_idx in enumerate(r0_f):
            radial_sum[int(edge_idx)] += float(cell_r_lo_obs[valid_ids[row]])
            radial_count[int(edge_idx)] += 1
        for row, edge_idx in enumerate(r1_f):
            radial_sum[int(edge_idx)] += float(cell_r_hi_obs[valid_ids[row]])
            radial_count[int(edge_idx)] += 1
        if np.any(radial_count == 0):
            missing_edge = int(np.flatnonzero(radial_count == 0)[0])
            raise ValueError(f"Spherical lookup could not reconstruct radial edge {missing_edge}.")
        self._radial_edges = radial_sum / radial_count
        r_min = float(self._radial_edges[0])
        r_max = float(self._radial_edges[-1])
        d_theta_f = math.pi / float(int(self.leaf_shape[1]))
        d_phi_f = (2.0 * math.pi) / float(int(self.leaf_shape[2]))
        fine_i1 = np.left_shift(self._cell_ijk[valid_ids, AXIS1], shifts)
        fine_i2 = np.left_shift(self._cell_ijk[valid_ids, AXIS2], shifts)
        n_octree_cells = int(self._cell_depth.shape[0])
        octree_cell_r_min = np.full(n_octree_cells, np.nan, dtype=np.float64)
        octree_cell_r_max = np.full(n_octree_cells, np.nan, dtype=np.float64)
        octree_cell_theta_min = np.full(n_octree_cells, np.nan, dtype=np.float64)
        octree_cell_theta_max = np.full(n_octree_cells, np.nan, dtype=np.float64)
        octree_cell_phi_start = np.full(n_octree_cells, np.nan, dtype=np.float64)
        octree_cell_phi_width = np.full(n_octree_cells, np.nan, dtype=np.float64)
        occupied_ids = np.flatnonzero(self._cell_depth >= 0)
        cell_shift = int(self.max_level) - self._cell_depth[occupied_ids]
        cell_width = np.left_shift(np.ones_like(cell_shift, dtype=np.int64), cell_shift)
        cell_r0_f = np.left_shift(self._cell_ijk[occupied_ids, AXIS0], cell_shift)
        cell_r1_f = cell_r0_f + cell_width
        cell_t0_f = np.left_shift(self._cell_ijk[occupied_ids, AXIS1], cell_shift)
        cell_t1_f = cell_t0_f + cell_width
        cell_p0_f = np.left_shift(self._cell_ijk[occupied_ids, AXIS2], cell_shift)
        octree_cell_r_min[occupied_ids] = self._radial_edges[cell_r0_f]
        octree_cell_r_max[occupied_ids] = self._radial_edges[cell_r1_f]
        octree_cell_theta_min[occupied_ids] = cell_t0_f * d_theta_f
        octree_cell_theta_max[occupied_ids] = cell_t1_f * d_theta_f
        octree_cell_phi_start[occupied_ids] = np.mod(cell_p0_f * d_phi_f, 2.0 * math.pi)
        octree_cell_phi_width[occupied_ids] = cell_width * d_phi_f
        cell_bounds = np.empty((n_octree_cells, 3, 2), dtype=np.float64)
        cell_bounds[:, AXIS0, START] = octree_cell_r_min
        cell_bounds[:, AXIS0, WIDTH] = octree_cell_r_max - octree_cell_r_min
        cell_bounds[:, AXIS1, START] = octree_cell_theta_min
        cell_bounds[:, AXIS1, WIDTH] = octree_cell_theta_max - octree_cell_theta_min
        cell_bounds[:, AXIS2, START] = octree_cell_phi_start
        cell_bounds[:, AXIS2, WIDTH] = octree_cell_phi_width
        domain_bounds = np.empty((3, 2), dtype=np.float64)
        domain_bounds[:, START] = np.array([r_min, 0.0, 0.0], dtype=np.float64)
        domain_bounds[:, WIDTH] = np.array([float(r_max - r_min), float(math.pi), float(_TWO_PI)], dtype=np.float64)
        return cell_bounds, domain_bounds, float(_TWO_PI), True

    def _domain_bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.column_stack(
            (
                self.ds[XYZ_VARS[0]],
                self.ds[XYZ_VARS[1]],
                self.ds[XYZ_VARS[2]],
            )
        )
        return np.min(pts, axis=0), np.max(pts, axis=0)

    def _domain_bounds_rpa(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array(self._domain_bounds[:, START], dtype=float)
        hi = np.array(self._domain_bounds[:, START] + self._domain_bounds[:, WIDTH], dtype=float)
        return lo, hi


@njit(cache=True)
def _xyz_to_rpa_components(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert one Cartesian point to spherical `(r, polar, azimuth)`."""
    r = np.sqrt(x * x + y * y + z * z)
    if r == 0.0:
        polar = 0.0
    else:
        zr = z / r
        if zr < -1.0:
            zr = -1.0
        elif zr > 1.0:
            zr = 1.0
        polar = np.arccos(zr)
    azimuth = np.arctan2(y, x) % _TWO_PI
    return r, polar, azimuth


def _xyz_arrays_to_rpa(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian coordinate arrays to spherical `(r, polar, azimuth)`."""
    r = np.sqrt(x * x + y * y + z * z)
    polar = np.zeros_like(r)
    valid = r > 0.0
    if np.any(valid):
        zr = np.clip(z[valid] / r[valid], -1.0, 1.0)
        polar[valid] = np.arccos(zr)
    azimuth = np.mod(np.arctan2(y, x), _TWO_PI)
    return r, polar, azimuth

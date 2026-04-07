#!/usr/bin/env python3
"""Spherical coordinate support for octree lookup."""

from __future__ import annotations

import math

from numba import njit
import numpy as np

from .octree import AXIS0
from .octree import AXIS1
from .octree import AXIS2
from .octree import START
from .octree import WIDTH

_TWO_PI = 2.0 * math.pi


def _attach_spherical_coord_state(
    tree,
    points: np.ndarray,
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Derive spherical cell bounds, domain bounds, and axis-2 periodic metadata from point/corner geometry."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    cell_levels = tree.cell_levels
    valid_ids = np.flatnonzero(cell_levels >= 0)
    shifts = int(tree.max_level) - cell_levels[valid_ids]
    width_units = np.left_shift(np.ones_like(shifts, dtype=np.int64), shifts)
    r0_f = np.left_shift(tree.cell_ijk[valid_ids, AXIS0], shifts)
    r1_f = r0_f + width_units
    n_r_edges = int(tree.leaf_shape[0]) + 1
    point_r = np.sqrt(x * x + y * y + z * z)
    cell_r_lo_obs = np.min(point_r[corners], axis=1)
    cell_r_hi_obs = np.max(point_r[corners], axis=1)
    radial_sum = (
        np.bincount(r0_f, weights=cell_r_lo_obs[valid_ids], minlength=n_r_edges)
        + np.bincount(r1_f, weights=cell_r_hi_obs[valid_ids], minlength=n_r_edges)
    )
    radial_count = (
        np.bincount(r0_f, minlength=n_r_edges)
        + np.bincount(r1_f, minlength=n_r_edges)
    )
    if np.any(radial_count == 0):
        missing_edge = int(np.flatnonzero(radial_count == 0)[0])
        raise ValueError(f"Spherical lookup could not reconstruct radial edge {missing_edge}.")
    tree.radial_edges = radial_sum / radial_count
    r_min = float(tree.radial_edges[0])
    r_max = float(tree.radial_edges[-1])
    d_polar_f = math.pi / float(int(tree.leaf_shape[1]))
    d_azimuth_f = (2.0 * math.pi) / float(int(tree.leaf_shape[2]))
    n_octree_cells = int(tree.cell_depth.shape[0])
    octree_cell_r_min = np.full(n_octree_cells, np.nan, dtype=np.float64)
    octree_cell_r_max = np.full(n_octree_cells, np.nan, dtype=np.float64)
    octree_cell_polar_min = np.full(n_octree_cells, np.nan, dtype=np.float64)
    octree_cell_polar_max = np.full(n_octree_cells, np.nan, dtype=np.float64)
    octree_cell_azimuth_start = np.full(n_octree_cells, np.nan, dtype=np.float64)
    octree_cell_azimuth_width = np.full(n_octree_cells, np.nan, dtype=np.float64)
    occupied_ids = np.flatnonzero(tree.cell_depth >= 0)
    cell_shift = int(tree.max_level) - tree.cell_depth[occupied_ids]
    cell_width = np.left_shift(np.ones_like(cell_shift, dtype=np.int64), cell_shift)
    cell_r0_f = np.left_shift(tree.cell_ijk[occupied_ids, AXIS0], cell_shift)
    cell_r1_f = cell_r0_f + cell_width
    cell_polar0_f = np.left_shift(tree.cell_ijk[occupied_ids, AXIS1], cell_shift)
    cell_polar1_f = cell_polar0_f + cell_width
    cell_azimuth0_f = np.left_shift(tree.cell_ijk[occupied_ids, AXIS2], cell_shift)
    octree_cell_r_min[occupied_ids] = tree.radial_edges[cell_r0_f]
    octree_cell_r_max[occupied_ids] = tree.radial_edges[cell_r1_f]
    octree_cell_polar_min[occupied_ids] = cell_polar0_f * d_polar_f
    octree_cell_polar_max[occupied_ids] = cell_polar1_f * d_polar_f
    octree_cell_azimuth_start[occupied_ids] = np.mod(cell_azimuth0_f * d_azimuth_f, 2.0 * math.pi)
    octree_cell_azimuth_width[occupied_ids] = cell_width * d_azimuth_f
    cell_bounds = np.empty((n_octree_cells, 3, 2), dtype=np.float64)
    cell_bounds[:, AXIS0, START] = octree_cell_r_min
    cell_bounds[:, AXIS0, WIDTH] = octree_cell_r_max - octree_cell_r_min
    cell_bounds[:, AXIS1, START] = octree_cell_polar_min
    cell_bounds[:, AXIS1, WIDTH] = octree_cell_polar_max - octree_cell_polar_min
    cell_bounds[:, AXIS2, START] = octree_cell_azimuth_start
    cell_bounds[:, AXIS2, WIDTH] = octree_cell_azimuth_width
    domain_bounds = np.empty((3, 2), dtype=np.float64)
    domain_bounds[:, START] = np.array([r_min, 0.0, 0.0], dtype=np.float64)
    domain_bounds[:, WIDTH] = np.array([float(r_max - r_min), float(math.pi), float(_TWO_PI)], dtype=np.float64)
    return cell_bounds, domain_bounds, float(_TWO_PI), True


@njit(cache=True)
def xyz_to_rpa_components(x: float, y: float, z: float) -> tuple[float, float, float]:
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


def xyz_arrays_to_rpa(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian coordinate arrays to spherical `(r, polar, azimuth)`."""
    r = np.sqrt(x * x + y * y + z * z)
    polar = np.zeros_like(r)
    valid = r > 0.0
    if np.any(valid):
        zr = np.clip(z[valid] / r[valid], -1.0, 1.0)
        polar[valid] = np.arccos(zr)
    azimuth = np.mod(np.arctan2(y, x), _TWO_PI)
    return r, polar, azimuth

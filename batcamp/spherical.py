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
_RPA_LOOKUP_TOL = 1.0e-10
_RPA_CELL_BITS = np.array(
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
    used_edge = radial_count > 0
    if not used_edge[0] or not used_edge[-1]:
        missing_edge = int(0 if not used_edge[0] else (n_r_edges - 1))
        raise ValueError(f"Spherical lookup could not reconstruct radial edge {missing_edge}.")
    radial_edges = np.full(n_r_edges, np.nan, dtype=np.float64)
    radial_edges[used_edge] = radial_sum[used_edge] / radial_count[used_edge]
    tree.radial_edges = radial_edges
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
    if np.any(np.isnan(tree.radial_edges[cell_r0_f])) or np.any(np.isnan(tree.radial_edges[cell_r1_f])):
        missing_edge = int(
            np.concatenate(
                (
                    cell_r0_f[np.isnan(tree.radial_edges[cell_r0_f])],
                    cell_r1_f[np.isnan(tree.radial_edges[cell_r1_f])],
                )
            )[0]
        )
        raise ValueError(f"Spherical occupied cell requires unobserved radial edge {missing_edge}.")
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


def cell_volumes_rpa(tree) -> np.ndarray:
    """Return leaf-slot physical spherical cell volumes aligned with `tree.cell_levels`.

    For ``tree_coord="rpa"``, cell volume means the physical 3D measure

        dV = r^2 sin(theta) dr dtheta dphi

    rather than the coordinate-space measure ``dr dtheta dphi``.
    """
    cached = getattr(tree, "_cell_volumes", None)
    if cached is not None:
        return cached

    volumes = np.full(int(tree.cell_count), np.nan, dtype=np.float64)
    valid_leaf = np.asarray(tree.cell_levels, dtype=np.int64) >= 0
    leaf_bounds = np.asarray(tree.cell_bounds[: int(tree.cell_count)], dtype=np.float64)
    r0 = leaf_bounds[:, AXIS0, START]
    dr = leaf_bounds[:, AXIS0, WIDTH]
    r1 = r0 + dr
    theta0 = leaf_bounds[:, AXIS1, START]
    dtheta = leaf_bounds[:, AXIS1, WIDTH]
    theta1 = theta0 + dtheta
    dphi = leaf_bounds[:, AXIS2, WIDTH]

    # Exact physical spherical cell volume:
    # integral dV = dphi * (cos(theta0) - cos(theta1)) * (r1^3 - r0^3) / 3
    volumes[valid_leaf] = (
        dphi[valid_leaf]
        * (np.cos(theta0[valid_leaf]) - np.cos(theta1[valid_leaf]))
        * ((r1[valid_leaf] ** 3 - r0[valid_leaf] ** 3) / 3.0)
    )
    tree._cell_volumes = volumes
    return volumes


def cell_integrals_rpa(tree, point_values: np.ndarray, leaf_ids: np.ndarray) -> np.ndarray:
    """Return exact physical-volume integrals of the spherical trilinear interpolant.

    For ``tree_coord="rpa"``, whole-cell integrals use the physical 3D measure

        dV = r^2 sin(theta) dr dtheta dphi

    rather than the coordinate-space measure ``dr dtheta dphi``.

    If ``f_ijk`` are the 8 corner values in BATSRUS spherical corner order, the
    exact cell integral factorizes as

        integral(cell) = sum_ijk f_ijk * R_i * Theta_j * Phi_k

    where ``R_i``, ``Theta_j``, and ``Phi_k`` are the exact 1D basis integrals
    against ``r^2``, ``sin(theta)``, and ``1``.
    """
    leaf_bounds = np.asarray(tree.cell_bounds[leaf_ids], dtype=np.float64)
    r0 = leaf_bounds[:, AXIS0, START]
    delta_r = leaf_bounds[:, AXIS0, WIDTH]
    theta0 = leaf_bounds[:, AXIS1, START]
    delta_theta = leaf_bounds[:, AXIS1, WIDTH]
    theta1 = theta0 + delta_theta
    delta_phi = leaf_bounds[:, AXIS2, WIDTH]
    corner_values = np.asarray(point_values[tree.corners[leaf_ids]], dtype=np.float64)

    radial_weights = np.empty((leaf_ids.size, 2), dtype=np.float64)
    radial_weights[:, 0] = delta_r * (
        0.5 * r0 * r0 + (r0 * delta_r) / 3.0 + (delta_r * delta_r) / 12.0
    )
    radial_weights[:, 1] = delta_r * (
        0.5 * r0 * r0 + (2.0 * r0 * delta_r) / 3.0 + (delta_r * delta_r) / 4.0
    )

    sin_span = np.sin(theta1) - np.sin(theta0)
    polar_weights = np.empty((leaf_ids.size, 2), dtype=np.float64)
    polar_weights[:, 0] = np.cos(theta0) - (sin_span / delta_theta)
    polar_weights[:, 1] = (sin_span / delta_theta) - np.cos(theta1)

    azimuth_weights = np.empty((leaf_ids.size, 2), dtype=np.float64)
    azimuth_weights[:, 0] = 0.5 * delta_phi
    azimuth_weights[:, 1] = 0.5 * delta_phi

    corner_weights = np.empty((leaf_ids.size, 8), dtype=np.float64)
    for corner_ord in range(8):
        bit_r = int(_RPA_CELL_BITS[corner_ord, AXIS0])
        bit_p = int(_RPA_CELL_BITS[corner_ord, AXIS1])
        bit_a = int(_RPA_CELL_BITS[corner_ord, AXIS2])
        corner_weights[:, corner_ord] = (
            radial_weights[:, bit_r] * polar_weights[:, bit_p] * azimuth_weights[:, bit_a]
        )

    return np.sum(corner_values * corner_weights[:, :, None], axis=1)


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


@njit(cache=True)
def _contains_box_rpa(
    q: np.ndarray,
    bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> bool:
    """Return whether one spherical query lies in one packed box under the spherical lookup tolerance."""
    for axis in range(AXIS2):
        value = float(q[axis])
        start = float(bounds[axis, START])
        width = float(bounds[axis, WIDTH])
        if value < (start - _RPA_LOOKUP_TOL) or value > (start + width + _RPA_LOOKUP_TOL):
            return False
    value = float(q[AXIS2])
    start = float(bounds[AXIS2, START])
    width = float(bounds[AXIS2, WIDTH])
    if axis2_periodic:
        if width >= (float(axis2_period) - _RPA_LOOKUP_TOL):
            return True
        return ((value - start) % float(axis2_period)) <= (width + _RPA_LOOKUP_TOL)
    return value >= (start - _RPA_LOOKUP_TOL) and value <= (start + width + _RPA_LOOKUP_TOL)

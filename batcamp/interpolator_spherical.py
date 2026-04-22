#!/usr/bin/env python3
"""Spherical-specific interpolation and cell/ray integrals."""

from __future__ import annotations

import math

from numba import njit
from numba import prange
import numpy as np

from .trilinear_shared import _accumulate_trilinear
from .trilinear_shared import _axis_fraction
from .trilinear_shared import _periodic_axis_fraction
from .octree import TREE_COORD_AXIS0
from .octree import TREE_COORD_AXIS1
from .octree import TREE_COORD_AXIS2
from .octree import BOUNDS_START_SLOT
from .octree import BOUNDS_WIDTH_SLOT
from .octree_spherical import xyz_arrays_to_rpa
from .octree_spherical import xyz_to_rpa_components

# Map corner ordinal 0..7 to the low/high bit used on each `(r, polar, azimuth)` axis.
# The ordering matches BATSRUS spherical corner order rather than Cartesian cube order.
_CORNER_ORDINAL_TO_AXIS_BITS = np.array(
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


def prepare_queries(q_array: np.ndarray, query_coord: str) -> np.ndarray:
    """Return backend-local spherical queries."""
    if query_coord == "rpa":
        return q_array
    if query_coord == "xyz":
        return np.column_stack(xyz_arrays_to_rpa(q_array[:, 0], q_array[:, 1], q_array[:, 2]))
    raise ValueError("Spherical interpolation supports only query_coord='xyz' or query_coord='rpa'.")


@njit(cache=True)
def _interp_cell(
    out_row: np.ndarray,
    cell_id: int,
    radius: float,
    polar: float,
    azimuth: float,
    cell_bounds: np.ndarray,
    corners: np.ndarray,
    point_values: np.ndarray,
) -> None:
    """Write one interpolated value row for one spherical query in one leaf cell using flat point values."""
    cell_id = int(cell_id)
    frac_r = _axis_fraction(radius, cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_START_SLOT], cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_WIDTH_SLOT])
    frac_p = _axis_fraction(polar, cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_START_SLOT], cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_WIDTH_SLOT])
    frac_a = _periodic_axis_fraction(
        azimuth,
        cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_START_SLOT],
        cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT],
        2.0 * math.pi,
    )

    _accumulate_trilinear(
        out_row,
        cell_id,
        frac_r,
        frac_p,
        frac_a,
        corners,
        point_values,
        _CORNER_ORDINAL_TO_AXIS_BITS,
    )


@njit(cache=True)
def _integrate_straight_segment(
    out_row: np.ndarray,
    cell_id: int,
    segment_span: float,
    x_start: float,
    y_start: float,
    z_start: float,
    x_stop: float,
    y_stop: float,
    z_stop: float,
    cell_bounds: np.ndarray,
    corners: np.ndarray,
    point_values: np.ndarray,
) -> None:
    """Write one RPA-local straight-slab trilinear segment integral row."""
    cell_id = int(cell_id)
    r_start, polar_start, azimuth_start = xyz_to_rpa_components(x_start, y_start, z_start)
    r_stop, polar_stop, azimuth_stop = xyz_to_rpa_components(x_stop, y_stop, z_stop)
    frac_r0 = _axis_fraction(r_start, cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_START_SLOT], cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_WIDTH_SLOT])
    frac_p0 = _axis_fraction(polar_start, cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_START_SLOT], cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_WIDTH_SLOT])
    frac_a0 = _periodic_axis_fraction(
        azimuth_start,
        cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_START_SLOT],
        cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT],
        2.0 * math.pi,
    )
    frac_r1 = _axis_fraction(r_stop, cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_START_SLOT], cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_WIDTH_SLOT])
    frac_p1 = _axis_fraction(polar_stop, cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_START_SLOT], cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_WIDTH_SLOT])
    frac_a1 = _periodic_axis_fraction(
        azimuth_stop,
        cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_START_SLOT],
        cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT],
        2.0 * math.pi,
    )
    d_frac_r = frac_r1 - frac_r0
    d_frac_p = frac_p1 - frac_p0
    d_frac_a = frac_a1 - frac_a0

    cell_corner_ids = corners[cell_id]
    out_row[:] = 0.0
    for corner_ord in range(8):
        bit_r = _CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS0]
        bit_p = _CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS1]
        bit_a = _CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS2]

        c_r = frac_r0 if bit_r else (1.0 - frac_r0)
        c_p = frac_p0 if bit_p else (1.0 - frac_p0)
        c_a = frac_a0 if bit_a else (1.0 - frac_a0)
        d_r = d_frac_r if bit_r else -d_frac_r
        d_p = d_frac_p if bit_p else -d_frac_p
        d_a = d_frac_a if bit_a else -d_frac_a

        w0 = c_r * c_p * c_a
        w1 = d_r * c_p * c_a + c_r * d_p * c_a + c_r * c_p * d_a
        w2 = d_r * d_p * c_a + d_r * c_p * d_a + c_r * d_p * d_a
        w3 = d_r * d_p * d_a
        weight = segment_span * (w0 + 0.5 * w1 + (w2 / 3.0) + 0.25 * w3)

        corner_point_id = int(cell_corner_ids[corner_ord])
        out_row[:] += weight * point_values[corner_point_id]


@njit(cache=True, parallel=True)
def interp_cells(
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
        _interp_cell(
            out[i],
            cell_id,
            queries_rpa[i, 0],
            queries_rpa[i, 1],
            queries_rpa[i, 2] % (2.0 * math.pi),
            cell_bounds,
            corners,
            point_values,
        )
    return out


def cell_integrals(tree, point_values: np.ndarray, leaf_ids: np.ndarray) -> np.ndarray:
    """Return exact physical-volume integrals of the spherical trilinear interpolant."""
    corner_values = np.asarray(point_values[tree.corners[leaf_ids]], dtype=np.float64)
    leaf_bounds = np.asarray(tree.cell_bounds[leaf_ids], dtype=np.float64)
    cell_lower = leaf_bounds[:, :, BOUNDS_START_SLOT]
    cell_upper = cell_lower + leaf_bounds[:, :, BOUNDS_WIDTH_SLOT]
    return _integrate_boxes(leaf_bounds, corner_values, cell_lower, cell_upper)


def _integrate_boxes(
    leaf_bounds: np.ndarray,
    corner_values: np.ndarray,
    sub_lower: np.ndarray,
    sub_upper: np.ndarray,
) -> np.ndarray:
    """Return exact physical-volume trilinear integrals over one clipped sub-box per leaf."""
    cell_lower = leaf_bounds[:, :, BOUNDS_START_SLOT]
    cell_upper = cell_lower + leaf_bounds[:, :, BOUNDS_WIDTH_SLOT]
    r0 = cell_lower[:, TREE_COORD_AXIS0]
    r1 = cell_upper[:, TREE_COORD_AXIS0]
    theta0 = cell_lower[:, TREE_COORD_AXIS1]
    theta1 = cell_upper[:, TREE_COORD_AXIS1]
    azimuth0 = cell_lower[:, TREE_COORD_AXIS2]
    azimuth1 = cell_upper[:, TREE_COORD_AXIS2]
    radius_a = sub_lower[:, TREE_COORD_AXIS0]
    radius_b = sub_upper[:, TREE_COORD_AXIS0]
    polar_a = sub_lower[:, TREE_COORD_AXIS1]
    polar_b = sub_upper[:, TREE_COORD_AXIS1]
    azimuth_a = sub_lower[:, TREE_COORD_AXIS2]
    azimuth_b = sub_upper[:, TREE_COORD_AXIS2]
    delta_r = leaf_bounds[:, TREE_COORD_AXIS0, BOUNDS_WIDTH_SLOT]
    delta_theta = leaf_bounds[:, TREE_COORD_AXIS1, BOUNDS_WIDTH_SLOT]
    delta_azimuth = leaf_bounds[:, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT]

    radial_weights = np.empty((leaf_bounds.shape[0], 2), dtype=np.float64)
    polar_weights = np.empty((leaf_bounds.shape[0], 2), dtype=np.float64)
    azimuth_weights = np.empty((leaf_bounds.shape[0], 2), dtype=np.float64)
    radial_weights[:, 0] = (
        (r1 * radius_b**3 / 3.0 - radius_b**4 / 4.0)
        - (r1 * radius_a**3 / 3.0 - radius_a**4 / 4.0)
    ) / delta_r
    radial_weights[:, 1] = (
        (radius_b**4 / 4.0 - r0 * radius_b**3 / 3.0)
        - (radius_a**4 / 4.0 - r0 * radius_a**3 / 3.0)
    ) / delta_r
    polar_weights[:, 0] = (
        ((polar_b - theta1) * np.cos(polar_b) - np.sin(polar_b))
        - ((polar_a - theta1) * np.cos(polar_a) - np.sin(polar_a))
    ) / delta_theta
    polar_weights[:, 1] = (
        (((theta0 - polar_b) * np.cos(polar_b) + np.sin(polar_b)))
        - (((theta0 - polar_a) * np.cos(polar_a) + np.sin(polar_a)))
    ) / delta_theta
    azimuth_weights[:, 0] = (
        (azimuth1 * azimuth_b - 0.5 * azimuth_b * azimuth_b)
        - (azimuth1 * azimuth_a - 0.5 * azimuth_a * azimuth_a)
    ) / delta_azimuth
    azimuth_weights[:, 1] = (
        (0.5 * azimuth_b * azimuth_b - azimuth0 * azimuth_b)
        - (0.5 * azimuth_a * azimuth_a - azimuth0 * azimuth_a)
    ) / delta_azimuth

    corner_weights = np.empty((leaf_bounds.shape[0], 8), dtype=np.float64)
    for corner_ord in range(8):
        bit_r = int(_CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS0])
        bit_p = int(_CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS1])
        bit_a = int(_CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS2])
        corner_weights[:, corner_ord] = (
            radial_weights[:, bit_r] * polar_weights[:, bit_p] * azimuth_weights[:, bit_a]
        )

    return np.sum(corner_values * corner_weights[:, :, None], axis=1)


def integrate_box(tree, point_values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return the exact physical-volume integral of the spherical trilinear interpolant over one native box."""
    domain_lower, domain_upper = tree.domain_bounds(coord="rpa")
    clipped_lower = np.maximum(np.asarray(lower, dtype=np.float64), domain_lower)
    clipped_upper = np.minimum(np.asarray(upper, dtype=np.float64), domain_upper)
    n_components = int(point_values.shape[1])
    if np.any(clipped_upper <= clipped_lower):
        return np.zeros(n_components, dtype=np.float64)

    leaf_bounds = np.asarray(tree.cell_bounds[: int(tree.cell_count)], dtype=np.float64)
    cell_lower = leaf_bounds[:, :, BOUNDS_START_SLOT]
    cell_upper = cell_lower + leaf_bounds[:, :, BOUNDS_WIDTH_SLOT]
    overlap = np.all(
        (clipped_upper[None, :] > cell_lower) & (clipped_lower[None, :] < cell_upper),
        axis=1,
    )
    leaf_ids = np.flatnonzero(overlap)
    if leaf_ids.size == 0:
        return np.zeros(n_components, dtype=np.float64)

    clipped_sub_lower = np.maximum(cell_lower[leaf_ids], clipped_lower[None, :])
    clipped_sub_upper = np.minimum(cell_upper[leaf_ids], clipped_upper[None, :])
    leaf_bounds = leaf_bounds[leaf_ids]
    corner_values = np.asarray(point_values[tree.corners[leaf_ids]], dtype=np.float64)
    return np.sum(
        _integrate_boxes(leaf_bounds, corner_values, clipped_sub_lower, clipped_sub_upper),
        axis=0,
    )


@njit(cache=True, parallel=True)
def accumulate_trilinear_cells(
    origins_xyz: np.ndarray,
    directions_xyz: np.ndarray,
    cell_counts: np.ndarray,
    cell_ids_scratch: np.ndarray,
    times_scratch: np.ndarray,
    cell_bounds: np.ndarray,
    corners: np.ndarray,
    point_values: np.ndarray,
) -> np.ndarray:
    """Accumulate RPA-local straight-slab trilinear segment integrals."""
    n_rays = int(origins_xyz.shape[0])
    n_components = int(point_values.shape[1])
    out = np.zeros((n_rays, n_components), dtype=point_values.dtype)
    for ray_id in prange(n_rays):
        integral = np.empty(n_components, dtype=point_values.dtype)
        origin_x = float(origins_xyz[ray_id, 0])
        origin_y = float(origins_xyz[ray_id, 1])
        origin_z = float(origins_xyz[ray_id, 2])
        direction_x = float(directions_xyz[ray_id, 0])
        direction_y = float(directions_xyz[ray_id, 1])
        direction_z = float(directions_xyz[ray_id, 2])
        n_cell = int(cell_counts[ray_id])
        for cell_pos in range(n_cell):
            cell_id = int(cell_ids_scratch[ray_id, cell_pos])
            if cell_id < 0:
                continue
            t_start = float(times_scratch[ray_id, cell_pos])
            t_stop = float(times_scratch[ray_id, cell_pos + 1])
            if t_stop <= t_start:
                continue
            _integrate_straight_segment(
                integral,
                cell_id,
                t_stop - t_start,
                origin_x + t_start * direction_x,
                origin_y + t_start * direction_y,
                origin_z + t_start * direction_z,
                origin_x + t_stop * direction_x,
                origin_y + t_stop * direction_y,
                origin_z + t_stop * direction_z,
                cell_bounds,
                corners,
                point_values,
            )
            out[ray_id, :] += integral
    return out

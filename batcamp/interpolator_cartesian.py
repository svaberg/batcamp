#!/usr/bin/env python3
"""Cartesian-specific interpolation and cell/ray integrals."""

from __future__ import annotations

import numpy as np
from numba import njit
from numba import prange

from .shared import TraceScratch
from .shared import TrilinearField
from .trilinear_shared import _accumulate_trilinear
from .trilinear_shared import _axis_fraction
from .octree import TREE_COORD_AXIS0
from .octree import TREE_COORD_AXIS1
from .octree import TREE_COORD_AXIS2
from .octree import BOUNDS_START_SLOT
from .octree import BOUNDS_WIDTH_SLOT

# Map corner ordinal 0..7 to the low/high bit used on each Cartesian axis.
_CORNER_ORDINAL_TO_AXIS_BITS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int8,
)


def prepare_queries(q_array: np.ndarray, query_coord: str) -> np.ndarray:
    """Return backend-local Cartesian queries."""
    if query_coord != "xyz":
        raise ValueError("Cartesian interpolation supports only query_coord='xyz'.")
    return q_array


@njit(cache=True)
def _interp_cell(
    out_row: np.ndarray,
    cell_id: int,
    x: float,
    y: float,
    z: float,
    field: TrilinearField,
) -> None:
    """Write one Cartesian trilinear interpolation result row for one leaf cell using flat point values."""
    cell_id = int(cell_id)
    frac_x = _axis_fraction(
        x,
        field.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_START_SLOT],
        field.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_WIDTH_SLOT],
    )
    frac_y = _axis_fraction(
        y,
        field.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_START_SLOT],
        field.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_WIDTH_SLOT],
    )
    frac_z = _axis_fraction(
        z,
        field.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_START_SLOT],
        field.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT],
    )

    _accumulate_trilinear(
        out_row,
        cell_id,
        frac_x,
        frac_y,
        frac_z,
        field.corners,
        field.point_values,
        _CORNER_ORDINAL_TO_AXIS_BITS,
    )


@njit(cache=True)
def _integrate_segment(
    out_row: np.ndarray,
    cell_id: int,
    segment_span: float,
    x_start: float,
    y_start: float,
    z_start: float,
    x_stop: float,
    y_stop: float,
    z_stop: float,
    field: TrilinearField,
) -> None:
    """Write one exact Cartesian trilinear segment integral row for one leaf cell."""
    cell_id = int(cell_id)
    frac_x0 = _axis_fraction(
        x_start,
        field.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_START_SLOT],
        field.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_WIDTH_SLOT],
    )
    frac_y0 = _axis_fraction(
        y_start,
        field.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_START_SLOT],
        field.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_WIDTH_SLOT],
    )
    frac_z0 = _axis_fraction(
        z_start,
        field.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_START_SLOT],
        field.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT],
    )
    frac_x1 = _axis_fraction(
        x_stop,
        field.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_START_SLOT],
        field.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_WIDTH_SLOT],
    )
    frac_y1 = _axis_fraction(
        y_stop,
        field.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_START_SLOT],
        field.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_WIDTH_SLOT],
    )
    frac_z1 = _axis_fraction(
        z_stop,
        field.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_START_SLOT],
        field.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT],
    )
    d_frac_x = frac_x1 - frac_x0
    d_frac_y = frac_y1 - frac_y0
    d_frac_z = frac_z1 - frac_z0

    out_row[:] = 0.0
    for corner_ord in range(8):
        bit_x = _CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS0]
        bit_y = _CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS1]
        bit_z = _CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS2]

        c_x = frac_x0 if bit_x else (1.0 - frac_x0)
        c_y = frac_y0 if bit_y else (1.0 - frac_y0)
        c_z = frac_z0 if bit_z else (1.0 - frac_z0)
        d_x = d_frac_x if bit_x else -d_frac_x
        d_y = d_frac_y if bit_y else -d_frac_y
        d_z = d_frac_z if bit_z else -d_frac_z

        w0 = c_x * c_y * c_z
        w1 = d_x * c_y * c_z + c_x * d_y * c_z + c_x * c_y * d_z
        w2 = d_x * d_y * c_z + d_x * c_y * d_z + c_x * d_y * d_z
        w3 = d_x * d_y * d_z
        weight = segment_span * (w0 + 0.5 * w1 + (w2 / 3.0) + 0.25 * w3)

        corner_point_id = int(field.corners[cell_id, corner_ord])
        out_row[:] += weight * field.point_values[corner_point_id]


@njit(cache=True, parallel=True)
def interp_cells(
    queries_xyz: np.ndarray,
    cell_ids: np.ndarray,
    fill_values: np.ndarray,
    field: TrilinearField,
) -> np.ndarray:
    """Evaluate Cartesian-tree interpolation for Cartesian queries with known leaf cell ids using flat point values."""
    n_query = queries_xyz.shape[0]
    ncomp = field.point_values.shape[1]
    out = np.empty((n_query, ncomp), dtype=field.point_values.dtype)
    for i in prange(n_query):
        out[i, :] = fill_values
        cell_id = int(cell_ids[i])
        if cell_id < 0:
            continue
        _interp_cell(
            out[i],
            cell_id,
            queries_xyz[i, 0],
            queries_xyz[i, 1],
            queries_xyz[i, 2],
            field,
        )
    return out


def cell_integrals(tree, field: TrilinearField, leaf_ids: np.ndarray) -> np.ndarray:
    """Return exact whole-cell integrals of the Cartesian trilinear interpolant."""
    leaf_bounds = np.asarray(field.cell_bounds[leaf_ids], dtype=np.float64)
    corner_values = np.asarray(field.point_values[field.corners[leaf_ids]], dtype=np.float64)
    cell_lower = leaf_bounds[:, :, BOUNDS_START_SLOT]
    cell_upper = cell_lower + leaf_bounds[:, :, BOUNDS_WIDTH_SLOT]
    return _integrate_boxes(leaf_bounds, corner_values, cell_lower, cell_upper)


def _integrate_boxes(
    leaf_bounds: np.ndarray,
    corner_values: np.ndarray,
    sub_lower: np.ndarray,
    sub_upper: np.ndarray,
) -> np.ndarray:
    """Return exact Cartesian trilinear integrals over one clipped sub-box per leaf."""
    cell_lower = leaf_bounds[:, :, BOUNDS_START_SLOT]
    cell_upper = cell_lower + leaf_bounds[:, :, BOUNDS_WIDTH_SLOT]
    x0 = cell_lower[:, TREE_COORD_AXIS0]
    x1 = cell_upper[:, TREE_COORD_AXIS0]
    y0 = cell_lower[:, TREE_COORD_AXIS1]
    y1 = cell_upper[:, TREE_COORD_AXIS1]
    z0 = cell_lower[:, TREE_COORD_AXIS2]
    z1 = cell_upper[:, TREE_COORD_AXIS2]
    xa = sub_lower[:, TREE_COORD_AXIS0]
    xb = sub_upper[:, TREE_COORD_AXIS0]
    ya = sub_lower[:, TREE_COORD_AXIS1]
    yb = sub_upper[:, TREE_COORD_AXIS1]
    za = sub_lower[:, TREE_COORD_AXIS2]
    zb = sub_upper[:, TREE_COORD_AXIS2]
    delta_x = leaf_bounds[:, TREE_COORD_AXIS0, BOUNDS_WIDTH_SLOT]
    delta_y = leaf_bounds[:, TREE_COORD_AXIS1, BOUNDS_WIDTH_SLOT]
    delta_z = leaf_bounds[:, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT]

    x_weights = np.empty((leaf_bounds.shape[0], 2), dtype=np.float64)
    y_weights = np.empty((leaf_bounds.shape[0], 2), dtype=np.float64)
    z_weights = np.empty((leaf_bounds.shape[0], 2), dtype=np.float64)
    x_weights[:, 0] = ((x1 * xb - 0.5 * xb * xb) - (x1 * xa - 0.5 * xa * xa)) / delta_x
    x_weights[:, 1] = ((0.5 * xb * xb - x0 * xb) - (0.5 * xa * xa - x0 * xa)) / delta_x
    y_weights[:, 0] = ((y1 * yb - 0.5 * yb * yb) - (y1 * ya - 0.5 * ya * ya)) / delta_y
    y_weights[:, 1] = ((0.5 * yb * yb - y0 * yb) - (0.5 * ya * ya - y0 * ya)) / delta_y
    z_weights[:, 0] = ((z1 * zb - 0.5 * zb * zb) - (z1 * za - 0.5 * za * za)) / delta_z
    z_weights[:, 1] = ((0.5 * zb * zb - z0 * zb) - (0.5 * za * za - z0 * za)) / delta_z

    corner_weights = np.empty((leaf_bounds.shape[0], 8), dtype=np.float64)
    for corner_ord in range(8):
        bit_x = int(_CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS0])
        bit_y = int(_CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS1])
        bit_z = int(_CORNER_ORDINAL_TO_AXIS_BITS[corner_ord, TREE_COORD_AXIS2])
        corner_weights[:, corner_ord] = x_weights[:, bit_x] * y_weights[:, bit_y] * z_weights[:, bit_z]

    return np.sum(corner_values * corner_weights[:, :, None], axis=1)


def integrate_box(tree, field: TrilinearField, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return the exact integral of the Cartesian trilinear interpolant over one native box."""
    domain_lower, domain_upper = tree.domain_bounds(coord="xyz")
    clipped_lower = np.maximum(np.asarray(lower, dtype=np.float64), domain_lower)
    clipped_upper = np.minimum(np.asarray(upper, dtype=np.float64), domain_upper)
    n_components = int(field.point_values.shape[1])
    if np.any(clipped_upper <= clipped_lower):
        return np.zeros(n_components, dtype=np.float64)

    leaf_bounds = np.asarray(field.cell_bounds[: int(tree.cell_count)], dtype=np.float64)
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
    corner_values = np.asarray(field.point_values[field.corners[leaf_ids]], dtype=np.float64)
    return np.sum(
        _integrate_boxes(leaf_bounds, corner_values, clipped_sub_lower, clipped_sub_upper),
        axis=0,
    )


@njit(cache=True, parallel=True)
def accumulate_midpoint_cells(
    origins_xyz: np.ndarray,
    directions_xyz: np.ndarray,
    trace: TraceScratch,
    field: TrilinearField,
) -> np.ndarray:
    """Accumulate midpoint-sampled known-cell Cartesian segments into one `(n_rays, n_components)` array."""
    n_rays = int(origins_xyz.shape[0])
    n_components = int(field.point_values.shape[1])
    out = np.zeros((n_rays, n_components), dtype=field.point_values.dtype)
    for ray_id in prange(n_rays):
        sample = np.empty(n_components, dtype=field.point_values.dtype)
        origin_x = float(origins_xyz[ray_id, 0])
        origin_y = float(origins_xyz[ray_id, 1])
        origin_z = float(origins_xyz[ray_id, 2])
        direction_x = float(directions_xyz[ray_id, 0])
        direction_y = float(directions_xyz[ray_id, 1])
        direction_z = float(directions_xyz[ray_id, 2])
        n_cell = int(trace.cell_counts[ray_id])
        for cell_pos in range(n_cell):
            cell_id = int(trace.cell_ids[ray_id, cell_pos])
            if cell_id < 0:
                continue
            t_start = float(trace.times[ray_id, cell_pos])
            t_stop = float(trace.times[ray_id, cell_pos + 1])
            segment_length = t_stop - t_start
            if segment_length <= 0.0:
                continue
            midpoint_t = 0.5 * (t_start + t_stop)
            _interp_cell(
                sample,
                cell_id,
                origin_x + midpoint_t * direction_x,
                origin_y + midpoint_t * direction_y,
                origin_z + midpoint_t * direction_z,
                field,
            )
            for component_id in range(n_components):
                out[ray_id, component_id] += segment_length * sample[component_id]
    return out


@njit(cache=True, parallel=True)
def accumulate_trilinear_cells(
    origins_xyz: np.ndarray,
    directions_xyz: np.ndarray,
    trace: TraceScratch,
    field: TrilinearField,
) -> np.ndarray:
    """Accumulate exact Cartesian trilinear segment integrals into one `(n_rays, n_components)` array."""
    n_rays = int(origins_xyz.shape[0])
    n_components = int(field.point_values.shape[1])
    out = np.zeros((n_rays, n_components), dtype=field.point_values.dtype)
    for ray_id in prange(n_rays):
        integral = np.empty(n_components, dtype=field.point_values.dtype)
        origin_x = float(origins_xyz[ray_id, 0])
        origin_y = float(origins_xyz[ray_id, 1])
        origin_z = float(origins_xyz[ray_id, 2])
        direction_x = float(directions_xyz[ray_id, 0])
        direction_y = float(directions_xyz[ray_id, 1])
        direction_z = float(directions_xyz[ray_id, 2])
        n_cell = int(trace.cell_counts[ray_id])
        for cell_pos in range(n_cell):
            cell_id = int(trace.cell_ids[ray_id, cell_pos])
            if cell_id < 0:
                continue
            t_start = float(trace.times[ray_id, cell_pos])
            t_stop = float(trace.times[ray_id, cell_pos + 1])
            if t_stop <= t_start:
                continue
            _integrate_segment(
                integral,
                cell_id,
                t_stop - t_start,
                origin_x + t_start * direction_x,
                origin_y + t_start * direction_y,
                origin_z + t_start * direction_z,
                origin_x + t_stop * direction_x,
                origin_y + t_stop * direction_y,
                origin_z + t_stop * direction_z,
                field,
            )
            out[ray_id, :] += integral
    return out

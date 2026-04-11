#!/usr/bin/env python3
"""Cartesian coordinate support for octree lookup.

Important modeling assumption for ``tree_coord="xyz"``:
leaf cells are treated as axis-aligned Cartesian slabs from
per-cell corner min/max values. Lookup/interpolation kernels in the
Cartesian backend are exact under this axis-aligned representation.
"""

from __future__ import annotations
import numpy as np

from .octree import START
from .octree import WIDTH


def _attach_cartesian_coord_state(
    tree,
    points: np.ndarray,
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Derive Cartesian cell bounds, domain bounds, and axis-2 periodic metadata from point/corner geometry."""
    xyz_min = np.min(points, axis=0).astype(np.float64, copy=False)
    xyz_max = np.max(points, axis=0).astype(np.float64, copy=False)
    leaf_shape = np.asarray(tree.leaf_shape, dtype=np.int64)
    max_level = int(tree.max_level)
    n_cells = int(tree.cell_depth.shape[0])
    cell_bounds = np.empty((n_cells, 3, 2), dtype=np.float64)
    cell_bounds.fill(np.nan)
    leaf_ids = np.flatnonzero(tree.cell_levels >= 0)
    leaf_corners = corners[leaf_ids]
    leaf_shift = max_level - tree.cell_levels[leaf_ids]
    leaf_width = np.left_shift(np.ones_like(leaf_shift, dtype=np.int64), leaf_shift)
    leaf_start_f = np.left_shift(tree.cell_ijk[leaf_ids], leaf_shift[:, None])

    axis_edges = [np.full(int(leaf_shape[axis]) + 1, np.nan, dtype=np.float64) for axis in range(3)]
    axis_edges[0][leaf_start_f[:, 0]] = points[leaf_corners[:, 0], 0]
    axis_edges[0][leaf_start_f[:, 0] + leaf_width] = points[leaf_corners[:, 1], 0]
    axis_edges[1][leaf_start_f[:, 1]] = points[leaf_corners[:, 0], 1]
    axis_edges[1][leaf_start_f[:, 1] + leaf_width] = points[leaf_corners[:, 2], 1]
    axis_edges[2][leaf_start_f[:, 2]] = points[leaf_corners[:, 0], 2]
    axis_edges[2][leaf_start_f[:, 2] + leaf_width] = points[leaf_corners[:, 4], 2]

    occupied_ids = np.flatnonzero(tree.cell_depth >= 0)
    cell_shift = max_level - tree.cell_depth[occupied_ids]
    cell_width = np.left_shift(np.ones_like(cell_shift, dtype=np.int64), cell_shift)
    cell_start_f = np.left_shift(tree.cell_ijk[occupied_ids], cell_shift[:, None])
    for axis in range(3):
        edges = axis_edges[axis]
        start_index = cell_start_f[:, axis]
        stop_index = start_index + cell_width
        cell_start = edges[start_index]
        cell_stop = edges[stop_index]
        if np.any(np.isnan(cell_start)) or np.any(np.isnan(cell_stop)):
            raise ValueError(
                "Cartesian coord state requires separable axis-aligned slabs with resolvable line coordinates."
            )
        cell_bounds[occupied_ids, axis, START] = cell_start
        cell_bounds[occupied_ids, axis, WIDTH] = cell_stop - cell_start

    domain_bounds = np.empty((3, 2), dtype=np.float64)
    domain_bounds[:, START] = xyz_min
    domain_bounds[:, WIDTH] = xyz_max - xyz_min
    return cell_bounds, domain_bounds, 0.0, False

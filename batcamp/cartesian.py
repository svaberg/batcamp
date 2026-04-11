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
    leaf_shape = np.asarray(tree.leaf_shape, dtype=np.int64)
    axis_edges = [np.unique(points[:, axis]).astype(np.float64, copy=False) for axis in range(3)]
    xyz_min = np.array([edges[0] for edges in axis_edges], dtype=np.float64)
    xyz_max = np.array([edges[-1] for edges in axis_edges], dtype=np.float64)
    n_cells = int(tree.cell_depth.shape[0])
    cell_bounds = np.empty((n_cells, 3, 2), dtype=np.float64)
    cell_bounds.fill(np.nan)
    occupied_ids = np.flatnonzero(tree.cell_depth >= 0)
    if all(edges.size == int(leaf_shape[axis]) + 1 for axis, edges in enumerate(axis_edges)):
        cell_shift = int(tree.max_level) - tree.cell_depth[occupied_ids]
        cell_width = np.left_shift(np.ones_like(cell_shift, dtype=np.int64), cell_shift)
        cell_start_f = np.left_shift(tree.cell_ijk[occupied_ids], cell_shift[:, None])
        for axis, edges in enumerate(axis_edges):
            start_index = cell_start_f[:, axis]
            stop_index = start_index + cell_width
            cell_start = edges[start_index]
            cell_stop = edges[stop_index]
            cell_bounds[occupied_ids, axis, START] = cell_start
            cell_bounds[occupied_ids, axis, WIDTH] = cell_stop - cell_start
    else:
        leaf_ids = np.flatnonzero(tree.cell_levels >= 0)
        leaf_xyz = points[corners[leaf_ids]]
        leaf_min = np.min(leaf_xyz, axis=1)
        leaf_max = np.max(leaf_xyz, axis=1)
        cell_bounds[leaf_ids, :, START] = leaf_min
        cell_bounds[leaf_ids, :, WIDTH] = leaf_max - leaf_min
        depth_order = occupied_ids[np.argsort(tree.cell_depth[occupied_ids])[::-1]]
        for cell_id in depth_order:
            child_ids = tree.cell_child[cell_id]
            child_ids = child_ids[child_ids >= 0]
            if child_ids.size == 0:
                continue
            child_start = cell_bounds[child_ids, :, START]
            child_stop = child_start + cell_bounds[child_ids, :, WIDTH]
            cell_start = np.min(child_start, axis=0)
            cell_stop = np.max(child_stop, axis=0)
            cell_bounds[cell_id, :, START] = cell_start
            cell_bounds[cell_id, :, WIDTH] = cell_stop - cell_start

    domain_bounds = np.empty((3, 2), dtype=np.float64)
    domain_bounds[:, START] = xyz_min
    domain_bounds[:, WIDTH] = xyz_max - xyz_min
    return cell_bounds, domain_bounds, 0.0, False

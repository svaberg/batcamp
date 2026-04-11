#!/usr/bin/env python3
"""Cartesian coordinate support for octree lookup.

Important modeling assumption for ``tree_coord="xyz"``:
leaf cells are treated as axis-aligned Cartesian slabs from
per-cell corner min/max values. Lookup/interpolation kernels in the
Cartesian backend are exact under this axis-aligned representation.
"""

from __future__ import annotations
import numpy as np
from numba import njit
from numba import prange

from .octree import AXIS0
from .octree import AXIS1
from .octree import AXIS2
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


@njit(cache=True)
def _contains_box_xyz(
    q: np.ndarray,
    bounds: np.ndarray,
    domain_bounds: np.ndarray,
) -> bool:
    """Return whether one Cartesian query lies in one exact half-open slab box."""
    for axis in range(3):
        value = float(q[axis])
        start = float(bounds[axis, START])
        stop = start + float(bounds[axis, WIDTH])
        domain_start = float(domain_bounds[axis, START])
        if start == domain_start:
            if value < start or value > stop:
                return False
        else:
            if value <= start or value > stop:
                return False
    return True


@njit(cache=True, parallel=True)
def _find_cells_xyz(
    queries: np.ndarray,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
) -> np.ndarray:
    """Resolve Cartesian queries to containing cell ids using one exact slab rule."""
    n_query = int(queries.shape[0])
    cell_ids = np.full(n_query, -1, dtype=np.int64)
    chunk_size = 1024
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    for chunk_id in prange(n_chunks):
        start = chunk_id * chunk_size
        end = min(n_query, start + chunk_size)
        hint_cell_id = -1
        for i in range(start, end):
            q = queries[i]
            if not (np.isfinite(q[AXIS0]) and np.isfinite(q[AXIS1]) and np.isfinite(q[AXIS2])):
                cell_id = -1
            elif not _contains_box_xyz(q, domain_bounds, domain_bounds):
                cell_id = -1
            else:
                current = int(hint_cell_id)
                while current >= 0 and not _contains_box_xyz(q, cell_bounds[current], domain_bounds):
                    current = int(cell_parent[current])

                if current < 0:
                    for root_pos in range(int(root_cell_ids.shape[0])):
                        root_cell_id = int(root_cell_ids[root_pos])
                        if _contains_box_xyz(q, cell_bounds[root_cell_id], domain_bounds):
                            current = root_cell_id
                            break
                if current < 0:
                    cell_id = -1
                else:
                    while np.any(cell_child[current] >= 0):
                        next_cell_id = -1
                        for child_ord in range(8):
                            child_id = int(cell_child[current, child_ord])
                            if child_id < 0:
                                continue
                            if _contains_box_xyz(q, cell_bounds[child_id], domain_bounds):
                                next_cell_id = child_id
                                break
                        if next_cell_id < 0:
                            current = -1
                            break
                        current = next_cell_id
                    cell_id = int(current)
            cell_ids[i] = cell_id
            hint_cell_id = int(cell_id) if cell_id >= 0 else -1
    return cell_ids

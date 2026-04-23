#!/usr/bin/env python3
"""Cartesian-specific octree geometry and lookup helpers."""

from __future__ import annotations

import numpy as np
from numba import njit
from numba import prange

from .octree import TREE_COORD_AXIS0
from .octree import TREE_COORD_AXIS1
from .octree import TREE_COORD_AXIS2
from .octree import BOUNDS_START_SLOT
from .octree import BOUNDS_WIDTH_SLOT
from .shared import LookupTree


def attach_state(
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
        cell_bounds[occupied_ids, axis, BOUNDS_START_SLOT] = cell_start
        cell_bounds[occupied_ids, axis, BOUNDS_WIDTH_SLOT] = cell_stop - cell_start

    domain_bounds = np.empty((3, 2), dtype=np.float64)
    domain_bounds[:, BOUNDS_START_SLOT] = xyz_min
    domain_bounds[:, BOUNDS_WIDTH_SLOT] = xyz_max - xyz_min
    return cell_bounds, domain_bounds, 0.0, False


def cell_volumes(tree) -> np.ndarray:
    """Return leaf-slot Cartesian cell volumes aligned with `tree.cell_levels`."""
    volumes = np.full(int(tree.cell_count), np.nan, dtype=np.float64)
    valid_leaf = np.asarray(tree.cell_levels, dtype=np.int64) >= 0
    leaf_widths = np.asarray(tree.cell_bounds[: int(tree.cell_count), :, BOUNDS_WIDTH_SLOT], dtype=np.float64)
    delta_x = leaf_widths[:, TREE_COORD_AXIS0]
    delta_y = leaf_widths[:, TREE_COORD_AXIS1]
    delta_z = leaf_widths[:, TREE_COORD_AXIS2]
    volumes[valid_leaf] = delta_x[valid_leaf] * delta_y[valid_leaf] * delta_z[valid_leaf]
    return volumes


def lookup_points(tree, queries: np.ndarray, coord: str) -> np.ndarray:
    """Resolve one batch of Cartesian queries to leaf cell ids."""
    if coord != "xyz":
        raise ValueError("Cartesian lookup supports only coord='xyz'.")
    return find_cells(queries, tree.lookup_tree)


@njit(cache=True)
def _contains_box(
    query_point: np.ndarray,
    bounds: np.ndarray,
    domain_bounds: np.ndarray,
) -> bool:
    """Return whether one Cartesian query lies in one exact half-open slab box."""
    for axis in range(3):
        value = float(query_point[axis])
        start = float(bounds[axis, BOUNDS_START_SLOT])
        stop = start + float(bounds[axis, BOUNDS_WIDTH_SLOT])
        domain_start = float(domain_bounds[axis, BOUNDS_START_SLOT])
        if start == domain_start:
            if value < start or value > stop:
                return False
        else:
            if value <= start or value > stop:
                return False
    return True


@njit(cache=True, parallel=True)
def find_cells(
    queries: np.ndarray,
    lookup_tree: LookupTree,
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
            query_point = queries[i]
            if not (
                np.isfinite(query_point[TREE_COORD_AXIS0])
                and np.isfinite(query_point[TREE_COORD_AXIS1])
                and np.isfinite(query_point[TREE_COORD_AXIS2])
            ):
                cell_id = -1
            elif not _contains_box(query_point, lookup_tree.domain_bounds, lookup_tree.domain_bounds):
                cell_id = -1
            else:
                current = int(hint_cell_id)
                while current >= 0 and not _contains_box(
                    query_point,
                    lookup_tree.cell_bounds[current],
                    lookup_tree.domain_bounds,
                ):
                    current = int(lookup_tree.cell_parent[current])

                if current < 0:
                    for root_pos in range(int(lookup_tree.root_cell_ids.shape[0])):
                        root_cell_id = int(lookup_tree.root_cell_ids[root_pos])
                        if _contains_box(query_point, lookup_tree.cell_bounds[root_cell_id], lookup_tree.domain_bounds):
                            current = root_cell_id
                            break
                if current < 0:
                    cell_id = -1
                else:
                    while np.any(lookup_tree.cell_child[current] >= 0):
                        next_cell_id = -1
                        for child_ord in range(8):
                            child_id = int(lookup_tree.cell_child[current, child_ord])
                            if child_id < 0:
                                continue
                            if _contains_box(
                                query_point,
                                lookup_tree.cell_bounds[child_id],
                                lookup_tree.domain_bounds,
                            ):
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

#!/usr/bin/env python3
"""Core octree data structures and shared lookup/interpolation utilities."""

from __future__ import annotations

from functools import cached_property
import logging
from pathlib import Path
import time

import numpy as np
from batread import Dataset
from numba import njit
from numba import prange

from .builder import _build_octree_state
from .builder import _warn_if_blocks_aux_mismatch
from .shared import XYZ_VARS
from .shared import SUPPORTED_TREE_COORDS
from .persistence import OctreeState
from .shared import GridShape
from .shared import LevelCountTable
from .shared import LookupTree
from .shared import TreeCoord
from .shared import TraversalTree
from .shared import timed_info_decorator

logger = logging.getLogger(__name__)

TREE_COORD_AXIS0 = 0  # Packed bounds axis index for the first tree coordinate.
TREE_COORD_AXIS1 = 1  # Packed bounds axis index for the second tree coordinate.
TREE_COORD_AXIS2 = 2  # Packed bounds axis index for the third tree coordinate.
BOUNDS_START_SLOT = 0  # Packed bounds slot index for interval start.
BOUNDS_WIDTH_SLOT = 1  # Packed bounds slot index for interval width.

# Map child ordinal 0..7 to its `(i, j, k)` low/high bits within a parent cell.
_CHILD_ORDINAL_TO_IJK_BITS = np.array(
    [[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(8)],
    dtype=np.int64,
)
# Map face id 0..5 to the tree-coordinate axis normal to that face.
_FACE_ID_TO_AXIS = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
# Map face id 0..5 to side 0/1, where 0 is the lower face and 1 is the upper face.
_FACE_ID_TO_SIDE = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
# Map face id 0..5 to the two tangential axes that lie in that face.
_FACE_ID_TO_TANGENTIAL_AXES = np.array(
    [
        [1, 2],
        [1, 2],
        [0, 2],
        [0, 2],
        [0, 1],
        [0, 1],
    ],
    dtype=np.int8,
)
# Map tree-coordinate axis 0/1/2 to the child-bit shift used in Morton-style child ordinals.
_AXIS_TO_CHILD_BIT_SHIFT = np.array([2, 1, 0], dtype=np.int8)


@njit(cache=True)
def _contains_box(
    query_point: np.ndarray,
    bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    tol: float,
) -> bool:
    """Return whether one query lies inside one packed axis-0/1/2 box."""
    for axis in range(TREE_COORD_AXIS2):
        value = float(query_point[axis])
        start = float(bounds[axis, BOUNDS_START_SLOT])
        width = float(bounds[axis, BOUNDS_WIDTH_SLOT])
        if value < (start - tol) or value > (start + width + tol):
            return False
    value = float(query_point[TREE_COORD_AXIS2])
    start = float(bounds[TREE_COORD_AXIS2, BOUNDS_START_SLOT])
    width = float(bounds[TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT])
    if axis2_periodic:
        if width >= (float(axis2_period) - tol):
            return True
        return ((value - start) % float(axis2_period)) <= (width + tol)
    return value >= (start - tol) and value <= (start + width + tol)


@njit(cache=True, parallel=True)
def _find_cells(
    queries: np.ndarray,
    lookup_tree: LookupTree,
) -> np.ndarray:
    """Resolve a batch of same-coordinate queries to containing cell ids."""
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
            elif not _contains_box(
                query_point,
                lookup_tree.domain_bounds,
                lookup_tree.axis2_period,
                lookup_tree.axis2_periodic,
                tol=0.0,
            ):
                cell_id = -1
            else:
                current = int(hint_cell_id)
                while current >= 0 and not _contains_box(
                    query_point,
                    lookup_tree.cell_bounds[current],
                    lookup_tree.axis2_period,
                    lookup_tree.axis2_periodic,
                    1.0e-10,
                ):
                    current = int(lookup_tree.cell_parent[current])

                if current < 0:
                    for root_pos in range(int(lookup_tree.root_cell_ids.shape[0])):
                        root_cell_id = int(lookup_tree.root_cell_ids[root_pos])
                        if _contains_box(
                            query_point,
                            lookup_tree.cell_bounds[root_cell_id],
                            lookup_tree.axis2_period,
                            lookup_tree.axis2_periodic,
                            1.0e-10,
                        ):
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
                                lookup_tree.axis2_period,
                                lookup_tree.axis2_periodic,
                                1.0e-10,
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


def _cell_row_order(cell_depth: np.ndarray, cell_ijk: np.ndarray) -> np.ndarray:
    """Return sorted `(depth, axis0, axis1, axis2)` row order for packed cell addresses."""
    return np.lexsort(np.column_stack((cell_depth, cell_ijk))[:, ::-1].T)


def _pack_cell_keys(cell_depth: np.ndarray, cell_ijk: np.ndarray, axis_bases: np.ndarray) -> np.ndarray:
    """Pack `(depth, axis0, axis1, axis2)` rows into sortable integer keys."""
    depth = cell_depth.astype(np.uint64, copy=False)
    axis0, axis1, axis2 = cell_ijk.astype(np.uint64, copy=False).T
    base0, base1, base2 = axis_bases.astype(np.uint64, copy=False)
    key = depth * base0 + axis0
    key = key * base1 + axis1
    key = key * base2 + axis2
    return key


def _unpack_cell_keys(keys: np.ndarray, axis_bases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack sortable integer keys into `(depth, cell_ijk)` arrays."""
    key = keys.astype(np.uint64, copy=True)
    base0, base1, base2 = axis_bases.astype(np.uint64, copy=False)
    axis2 = (key % base2).astype(np.int64)
    key //= base2
    axis1 = (key % base1).astype(np.int64)
    key //= base1
    axis0 = (key % base0).astype(np.int64)
    depth = (key // base0).astype(np.int64)
    return depth, np.column_stack((axis0, axis1, axis2))


@timed_info_decorator
def _rebuild_cells(
    depths: np.ndarray,
    cell_ijk: np.ndarray,
    leaf_value: np.ndarray,
    *,
    n_leaf_slots: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build rebuilt octree cell arrays from exact leaf addresses."""
    leaf_ijk_raw = cell_ijk
    if leaf_ijk_raw.ndim != 2 or leaf_ijk_raw.shape[1] != 3:
        raise ValueError("cell_ijk must have shape (n_cells, 3).")

    axis_bases = np.max(leaf_ijk_raw, axis=0).astype(np.uint64) + 1
    leaf_keys = _pack_cell_keys(depths, leaf_ijk_raw, axis_bases)
    leaf_order = np.argsort(leaf_keys)
    sorted_leaf_keys = leaf_keys[leaf_order]

    same_leaf = sorted_leaf_keys[1:] == sorted_leaf_keys[:-1]
    if np.any(same_leaf):
        dup = int(np.flatnonzero(same_leaf)[0])
        dup_depth, dup_ijk = _unpack_cell_keys(sorted_leaf_keys[dup:dup + 1], axis_bases)
        raise ValueError(
            "Cells overlap at octree address "
            f"{(int(dup_depth[0]), *(int(v) for v in dup_ijk[0]))}."
        )

    frontier_depth = depths
    frontier_ijk = leaf_ijk_raw
    parent_key_parts: list[np.ndarray] = []
    while True:
        mask = frontier_depth > 0
        if not np.any(mask):
            break
        frontier_depth = frontier_depth[mask] - 1
        frontier_ijk = np.right_shift(frontier_ijk[mask], 1)
        frontier_keys = np.unique(_pack_cell_keys(frontier_depth, frontier_ijk, axis_bases))
        parent_key_parts.append(frontier_keys)
        frontier_depth, frontier_ijk = _unpack_cell_keys(frontier_keys, axis_bases)

    if parent_key_parts:
        internal_keys = np.unique(np.concatenate(parent_key_parts))
        parent_pos = np.searchsorted(sorted_leaf_keys, internal_keys)
        hits = parent_pos < sorted_leaf_keys.size
        hits[hits] = sorted_leaf_keys[parent_pos[hits]] == internal_keys[hits]
        if np.any(hits):
            dup_depth, dup_ijk = _unpack_cell_keys(
                internal_keys[np.flatnonzero(hits)[:1]],
                axis_bases,
            )
            raise ValueError(
                "Cells overlap across parent/child addresses at "
                f"{(int(dup_depth[0]), *(int(v) for v in dup_ijk[0]))}."
            )
        internal_depth, internal_ijk = _unpack_cell_keys(internal_keys, axis_bases)
    else:
        internal_keys = np.empty(0, dtype=np.uint64)
        internal_depth = np.empty(0, dtype=np.int64)
        internal_ijk = np.empty((0, 3), dtype=np.int64)

    leaf_slots = int(np.max(leaf_value)) + 1 if n_leaf_slots is None and leaf_value.size else int(n_leaf_slots or 0)
    n_cells = leaf_slots + int(internal_depth.shape[0])
    cell_depth = np.full(n_cells, -1, dtype=np.int64)
    cell_ijk_out = np.full((n_cells, 3), -1, dtype=np.int64)
    cell_depth[leaf_value] = depths
    cell_ijk_out[leaf_value] = leaf_ijk_raw
    start = leaf_slots
    stop = start + int(internal_depth.shape[0])
    cell_depth[start:stop] = internal_depth
    cell_ijk_out[start:stop] = internal_ijk
    return cell_depth, cell_ijk_out, internal_keys, internal_depth, internal_ijk


@timed_info_decorator
def _build_cell_topology(
    depths: np.ndarray,
    leaf_ijk: np.ndarray,
    leaf_value: np.ndarray,
    internal_keys: np.ndarray,
    internal_depth: np.ndarray,
    internal_ijk: np.ndarray,
    leaf_slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse 8-child references for occupied rebuilt cells."""
    n_cells = int(leaf_slots) + int(internal_depth.shape[0])
    cell_child = np.full((n_cells, 8), -1, dtype=np.int64)
    cell_parent = np.full(n_cells, -1, dtype=np.int64)
    axis_bases = np.max(leaf_ijk, axis=0).astype(np.uint64) + 1

    leaf_mask = depths > 0
    leaf_child_ids = leaf_value[leaf_mask]
    leaf_parent_keys = _pack_cell_keys(
        depths[leaf_mask] - 1,
        np.right_shift(leaf_ijk[leaf_mask], 1),
        axis_bases,
    )
    leaf_child_ord = (
        ((leaf_ijk[leaf_mask, TREE_COORD_AXIS0] & 1) << 2)
        | ((leaf_ijk[leaf_mask, TREE_COORD_AXIS1] & 1) << 1)
        | (leaf_ijk[leaf_mask, TREE_COORD_AXIS2] & 1)
    ).astype(np.int64)

    internal_mask = internal_depth > 0
    internal_child_ids = int(leaf_slots) + np.flatnonzero(internal_mask).astype(np.int64)
    internal_parent_keys = _pack_cell_keys(
        internal_depth[internal_mask] - 1,
        np.right_shift(internal_ijk[internal_mask], 1),
        axis_bases,
    )
    internal_child_ord = (
        ((internal_ijk[internal_mask, TREE_COORD_AXIS0] & 1) << 2)
        | ((internal_ijk[internal_mask, TREE_COORD_AXIS1] & 1) << 1)
        | (internal_ijk[internal_mask, TREE_COORD_AXIS2] & 1)
    ).astype(np.int64)

    child_ids = np.concatenate((leaf_child_ids, internal_child_ids))
    parent_keys = np.concatenate((leaf_parent_keys, internal_parent_keys))
    child_ord = np.concatenate((leaf_child_ord, internal_child_ord))
    parent_pos = np.searchsorted(internal_keys, parent_keys)
    hits = parent_pos < internal_keys.size
    hits[hits] = internal_keys[parent_pos[hits]] == parent_keys[hits]
    parent_ids = int(leaf_slots) + parent_pos[hits]
    child_ids = child_ids[hits]
    child_ord = child_ord[hits]
    cell_child[parent_ids, child_ord] = child_ids
    cell_parent[child_ids] = parent_ids
    root_cell_ids = np.concatenate(
        (
            leaf_value[depths == 0],
            int(leaf_slots) + np.flatnonzero(internal_depth == 0).astype(np.int64),
        )
    )
    return cell_child, root_cell_ids, cell_parent


def _rebuild_cell_state(
    cell_levels: np.ndarray,
    cell_ijk: np.ndarray,
    tree_coord: str,
) -> tuple[np.ndarray, ...]:
    """Rebuild exact occupied cells from leaf addresses."""
    if cell_ijk.ndim != 2 or cell_ijk.shape[1] != 3:
        raise ValueError("Octree leaf cell_ijk must have shape (n_cells, 3).")
    if cell_levels.shape[0] != cell_ijk.shape[0]:
        raise ValueError("Octree leaf level/index arrays must have matching shapes.")
    valid_ids = np.flatnonzero(cell_levels >= 0).astype(np.int64)
    if valid_ids.size == 0:
        raise ValueError("Octree state requires at least one valid leaf cell.")
    depths = cell_levels[valid_ids]
    leaf_ijk_valid = cell_ijk[valid_ids]
    cell_depth, cell_ijk_rt, internal_keys, internal_depth, internal_ijk = _rebuild_cells(
        depths,
        leaf_ijk_valid,
        valid_ids,
        n_leaf_slots=int(cell_levels.shape[0]),
    )
    cell_child, root_cell_ids, cell_parent = _build_cell_topology(
        depths,
        leaf_ijk_valid,
        valid_ids,
        internal_keys,
        internal_depth,
        internal_ijk,
        int(cell_levels.shape[0]),
    )
    return cell_depth, cell_ijk_rt, cell_child, root_cell_ids, cell_parent


@njit(cache=True)
def _child_ord_from_cell_ijk(cell_ijk: np.ndarray, cell_id: int) -> int:
    """Return one runtime cell's child ordinal inside its parent."""
    return (
        ((int(cell_ijk[cell_id, TREE_COORD_AXIS0]) & 1) << 2)
        | ((int(cell_ijk[cell_id, TREE_COORD_AXIS1]) & 1) << 1)
        | (int(cell_ijk[cell_id, TREE_COORD_AXIS2]) & 1)
    )


@njit(cache=True)
def _neighbor_child_for_subface(
    cell_child: np.ndarray,
    neighbor_id: int,
    face_id: int,
    subface_id: int,
) -> int:
    """Return the patchwise neighboring runtime cell for one face subface."""
    if neighbor_id < 0:
        return -1
    has_child = False
    for child_ord in range(8):
        if int(cell_child[neighbor_id, child_ord]) >= 0:
            has_child = True
            break
    if not has_child:
        return int(neighbor_id)

    axis = int(_FACE_ID_TO_AXIS[face_id])
    side = int(_FACE_ID_TO_SIDE[face_id])
    tangential_axes = _FACE_ID_TO_TANGENTIAL_AXES[face_id]
    child_bits = np.zeros(3, dtype=np.int64)
    child_bits[axis] = 1 if side == 0 else 0
    child_bits[int(tangential_axes[0])] = (int(subface_id) >> 1) & 1
    child_bits[int(tangential_axes[1])] = int(subface_id) & 1
    child_ord = (
        ((int(child_bits[TREE_COORD_AXIS0]) & 1) << 2)
        | ((int(child_bits[TREE_COORD_AXIS1]) & 1) << 1)
        | (int(child_bits[TREE_COORD_AXIS2]) & 1)
    )
    return int(cell_child[neighbor_id, child_ord])


@njit(cache=True)
def _build_cell_neighbor_graph(
    cell_depth: np.ndarray,
    cell_ijk: np.ndarray,
    cell_child: np.ndarray,
    cell_parent: np.ndarray,
    root_cell_ids: np.ndarray,
    root_shape: np.ndarray,
    axis2_periodic: bool,
) -> np.ndarray:
    """Build one runtime-cell face/subface neighbor table for every occupied cell."""
    n_cells = int(cell_depth.shape[0])
    next_cell = np.full((n_cells, 6, 4), -1, dtype=np.int32)

    max_depth = -1
    n_valid = 0
    for cell_id in range(n_cells):
        depth = int(cell_depth[cell_id])
        if depth < 0:
            continue
        n_valid += 1
        if depth > max_depth:
            max_depth = depth
    if n_valid == 0:
        return next_cell

    depth_count = np.zeros(max_depth + 1, dtype=np.int64)
    for cell_id in range(n_cells):
        depth = int(cell_depth[cell_id])
        if depth >= 0:
            depth_count[depth] += 1

    depth_offset = np.zeros(max_depth + 2, dtype=np.int64)
    for depth in range(max_depth + 1):
        depth_offset[depth + 1] = depth_offset[depth] + depth_count[depth]

    depth_cursor = np.empty(max_depth + 1, dtype=np.int64)
    for depth in range(max_depth + 1):
        depth_cursor[depth] = depth_offset[depth]
    cells_by_depth = np.empty(n_valid, dtype=np.int64)
    for cell_id in range(n_cells):
        depth = int(cell_depth[cell_id])
        if depth < 0:
            continue
        pos = int(depth_cursor[depth])
        cells_by_depth[pos] = cell_id
        depth_cursor[depth] = pos + 1

    root_lookup = np.full(
        (int(root_shape[TREE_COORD_AXIS0]), int(root_shape[TREE_COORD_AXIS1]), int(root_shape[TREE_COORD_AXIS2])),
        -1,
        dtype=np.int64,
    )
    for root_pos in range(int(root_cell_ids.shape[0])):
        root_id = int(root_cell_ids[root_pos])
        if root_id < 0:
            continue
        root_ijk = cell_ijk[root_id]
        root_lookup[
            int(root_ijk[TREE_COORD_AXIS0]),
            int(root_ijk[TREE_COORD_AXIS1]),
            int(root_ijk[TREE_COORD_AXIS2]),
        ] = root_id

    for pos in range(int(depth_offset[0]), int(depth_offset[1])):
        cell_id = int(cells_by_depth[pos])
        ijk = cell_ijk[cell_id]
        for face_id in range(6):
            axis = int(_FACE_ID_TO_AXIS[face_id])
            side = int(_FACE_ID_TO_SIDE[face_id])
            neighbor_ijk0 = int(ijk[TREE_COORD_AXIS0])
            neighbor_ijk1 = int(ijk[TREE_COORD_AXIS1])
            neighbor_ijk2 = int(ijk[TREE_COORD_AXIS2])
            if axis == TREE_COORD_AXIS0:
                neighbor_ijk0 += -1 if side == 0 else 1
            elif axis == TREE_COORD_AXIS1:
                neighbor_ijk1 += -1 if side == 0 else 1
            else:
                neighbor_ijk2 += -1 if side == 0 else 1
                if axis2_periodic:
                    if neighbor_ijk2 < 0:
                        neighbor_ijk2 += int(root_shape[TREE_COORD_AXIS2])
                    elif neighbor_ijk2 >= int(root_shape[TREE_COORD_AXIS2]):
                        neighbor_ijk2 -= int(root_shape[TREE_COORD_AXIS2])
            if (
                neighbor_ijk0 < 0
                or neighbor_ijk0 >= int(root_shape[TREE_COORD_AXIS0])
                or neighbor_ijk1 < 0
                or neighbor_ijk1 >= int(root_shape[TREE_COORD_AXIS1])
                or neighbor_ijk2 < 0
                or neighbor_ijk2 >= int(root_shape[TREE_COORD_AXIS2])
            ):
                neighbor_id = -1
            else:
                neighbor_id = int(root_lookup[neighbor_ijk0, neighbor_ijk1, neighbor_ijk2])
            for subface_id in range(4):
                next_cell[cell_id, face_id, subface_id] = np.int32(
                    _neighbor_child_for_subface(cell_child, neighbor_id, face_id, subface_id)
                )

    for depth in range(1, max_depth + 1):
        start = int(depth_offset[depth])
        stop = int(depth_offset[depth + 1])
        for pos in range(start, stop):
            cell_id = int(cells_by_depth[pos])
            parent_id = int(cell_parent[cell_id])
            child_ord = _child_ord_from_cell_ijk(cell_ijk, cell_id)
            for face_id in range(6):
                axis = int(_FACE_ID_TO_AXIS[face_id])
                side = int(_FACE_ID_TO_SIDE[face_id])
                shift = int(_AXIS_TO_CHILD_BIT_SHIFT[axis])
                side_bit = (child_ord >> shift) & 1

                if (side == 0 and side_bit == 1) or (side == 1 and side_bit == 0):
                    base_neighbor_id = int(cell_child[parent_id, child_ord ^ (1 << shift)])
                else:
                    tangential_axes = _FACE_ID_TO_TANGENTIAL_AXES[face_id]
                    first_bit = (child_ord >> int(_AXIS_TO_CHILD_BIT_SHIFT[int(tangential_axes[0])])) & 1
                    second_bit = (child_ord >> int(_AXIS_TO_CHILD_BIT_SHIFT[int(tangential_axes[1])])) & 1
                    parent_subface_id = 2 * first_bit + second_bit
                    base_neighbor_id = int(next_cell[parent_id, face_id, parent_subface_id])

                for subface_id in range(4):
                    next_cell[cell_id, face_id, subface_id] = np.int32(
                        _neighbor_child_for_subface(cell_child, base_neighbor_id, face_id, subface_id)
                    )

    return next_cell


def _contains_boxes_vectorized(
    query_points: np.ndarray,
    bounds: np.ndarray,
    *,
    axis2_period: float,
    axis2_periodic: bool,
    tol: float,
) -> np.ndarray:
    """Return whether each query lies inside its corresponding packed box."""
    inside = np.ones(query_points.shape[0], dtype=bool)
    for axis in range(TREE_COORD_AXIS2):
        value = query_points[:, axis]
        start = bounds[:, axis, BOUNDS_START_SLOT]
        width = bounds[:, axis, BOUNDS_WIDTH_SLOT]
        inside &= (value >= (start - tol)) & (value <= (start + width + tol))
    value = query_points[:, TREE_COORD_AXIS2]
    start = bounds[:, TREE_COORD_AXIS2, BOUNDS_START_SLOT]
    width = bounds[:, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT]
    if axis2_periodic:
        full_wrap = width >= (float(axis2_period) - tol)
        inside &= full_wrap | ((((value - start) % float(axis2_period)) <= (width + tol)))
    else:
        inside &= (value >= (start - tol)) & (value <= (start + width + tol))
    return inside


def _validate_parent_child_center_containment(
    *,
    cell_bounds: np.ndarray,
    cell_parent: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> None:
    """Require reconstructed child centers to lie inside reconstructed parent bounds."""
    child_ids = np.flatnonzero(np.asarray(cell_parent, dtype=np.int64) >= 0).astype(np.int64)
    if child_ids.size == 0:
        return
    parent_ids = np.asarray(cell_parent[child_ids], dtype=np.int64)
    child_bounds = np.asarray(cell_bounds[child_ids], dtype=np.float64)
    parent_bounds = np.asarray(cell_bounds[parent_ids], dtype=np.float64)
    child_centers = np.empty((child_ids.size, 3), dtype=np.float64)
    child_centers[:, :TREE_COORD_AXIS2] = (
        child_bounds[:, :TREE_COORD_AXIS2, BOUNDS_START_SLOT]
        + 0.5 * child_bounds[:, :TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT]
    )
    child_centers[:, TREE_COORD_AXIS2] = (
        child_bounds[:, TREE_COORD_AXIS2, BOUNDS_START_SLOT]
        + 0.5 * child_bounds[:, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT]
    )
    if axis2_periodic:
        child_centers[:, TREE_COORD_AXIS2] = np.mod(child_centers[:, TREE_COORD_AXIS2], float(axis2_period))
    inside = _contains_boxes_vectorized(
        child_centers,
        parent_bounds,
        axis2_period=axis2_period,
        axis2_periodic=axis2_periodic,
        tol=1.0e-10,
    )
    if np.all(inside):
        return
    bad_pos = np.flatnonzero(~inside)
    first_pos = int(bad_pos[0])
    first_child = int(child_ids[first_pos])
    first_parent = int(parent_ids[first_pos])
    for pos in bad_pos[:3]:
        child_id = int(child_ids[int(pos)])
        parent_id = int(parent_ids[int(pos)])
        logger.debug(
            "Parent/child center containment child %d parent %d: center=%s child_bounds=%s parent_bounds=%s",
            child_id,
            parent_id,
            np.array2string(child_centers[int(pos)], precision=17),
            np.array2string(child_bounds[int(pos)], precision=17),
            np.array2string(parent_bounds[int(pos)], precision=17),
        )
    raise ValueError(
        "Parent/child center containment failed for "
        f"{int(bad_pos.size)} reconstructed cells; "
        f"first_child={first_child} first_parent={first_parent}."
    )


class Octree:
    """Adaptive octree summary plus bound lookup entrypoints.

    `level_counts` rows are
    `(level, leaf_count, fine_equivalent_count)`.
    """

    def __init__(
        self,
        points: np.ndarray,
        corners: np.ndarray,
        *,
        tree_coord: TreeCoord | None = None,
        build_axis_tol: float = 1e-12,
        build_level_rtol: float = 1e-4,
        build_level_atol: float = 1e-9,
    ) -> None:
        """Build an octree from explicit point coordinates and cell corners.

        `tree_coord` optionally fixes the coordinate system. The `build_*`
        arguments only affect geometric inference during construction.
        """
        state = _build_octree_state(
            points,
            corners,
            tree_coord=tree_coord,
            axis_tol=build_axis_tol,
            level_rtol=build_level_rtol,
            level_atol=build_level_atol,
            cell_levels=None,
        )
        logger.info("_init_from_state: coord=%s", state.tree_coord)
        logger.debug("_init_from_state...")
        t0 = time.perf_counter()
        self._init_from_state(
            root_shape=state.root_shape,
            tree_coord=state.tree_coord,
            cell_levels=state.cell_levels,
            cell_ijk=state.cell_ijk,
            points=points,
            corners=corners,
        )
        logger.info("_init_from_state complete in %.2fs", float(time.perf_counter() - t0))

    def _init_from_state(
        self,
        *,
        root_shape: GridShape,
        tree_coord: TreeCoord,
        cell_levels: np.ndarray,
        cell_ijk: np.ndarray,
        points: np.ndarray,
        corners: np.ndarray,
    ) -> None:
        """Materialize one octree from exact leaf addresses and explicit geometry."""
        leaf_levels: np.ndarray
        leaf_ijk: np.ndarray
        resolved_tree_coord = str(tree_coord)
        if resolved_tree_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported tree_coord '{resolved_tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        self._root_shape = tuple(int(v) for v in root_shape)
        self._tree_coord = resolved_tree_coord
        leaf_levels = np.asarray(cell_levels, dtype=np.int64)
        leaf_ijk = np.asarray(cell_ijk, dtype=np.int64)
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n_points, 3).")
        self._points = points
        corner_rows = np.asarray(corners, dtype=np.int64)
        if corner_rows.ndim != 2 or corner_rows.shape != (leaf_levels.shape[0], 8):
            raise ValueError("corners must have shape (n_cells, 8) matching cell_levels.")
        max_level = int(np.max(leaf_levels))
        logger.info("_rebuild_cell_state: coord=%s max_level=%d", self._tree_coord, max_level)
        logger.debug("_rebuild_cell_state...")
        t0 = time.perf_counter()
        (
            self._cell_depth,
            self._cell_ijk,
            self._cell_child,
            self._root_cell_ids,
            self._cell_parent,
        ) = _rebuild_cell_state(
            leaf_levels,
            leaf_ijk,
            self._tree_coord,
        )
        logger.info("_rebuild_cell_state complete in %.2fs", float(time.perf_counter() - t0))
        logger.info("_build_cell_neighbor_graph...")
        logger.debug("_build_cell_neighbor_graph...")
        t0 = time.perf_counter()
        self._cell_neighbor = _build_cell_neighbor_graph(
            self._cell_depth,
            self._cell_ijk,
            self._cell_child,
            self._cell_parent,
            self._root_cell_ids,
            np.asarray(self._root_shape, dtype=np.int64),
            bool(self._tree_coord == "rpa"),
        )
        logger.info("_build_cell_neighbor_graph complete in %.2fs", float(time.perf_counter() - t0))
        self._leaf_slot_count = int(leaf_levels.shape[0])
        if self._tree_coord == "xyz":
            from . import octree_cartesian as octree_backend
        else:
            from . import octree_spherical as octree_backend
        self._octree_module = octree_backend
        logger.info("%s: coord=%s", self._octree_module.attach_state.__name__, self._tree_coord)
        logger.debug("%s...", self._octree_module.attach_state.__name__)
        t0 = time.perf_counter()
        cell_bounds, domain_bounds, axis2_period, axis2_periodic = self._octree_module.attach_state(
            self,
            points,
            corner_rows,
        )
        logger.info(
            "%s complete in %.2fs",
            self._octree_module.attach_state.__name__,
            float(time.perf_counter() - t0),
        )
        self._corners = corner_rows
        self._cell_bounds = cell_bounds
        self._domain_bounds = domain_bounds
        self._axis2_period = float(axis2_period)
        self._axis2_periodic = bool(axis2_periodic)
        _validate_parent_child_center_containment(
            cell_bounds=self._cell_bounds,
            cell_parent=self._cell_parent,
            axis2_period=self._axis2_period,
            axis2_periodic=self._axis2_periodic,
        )

    @property
    def root_shape(self) -> GridShape:
        """Return root-grid shape."""
        return self._root_shape

    @property
    def tree_coord(self) -> TreeCoord:
        """Return tree coordinate system."""
        return self._tree_coord

    @property
    def leaf_shape(self) -> GridShape:
        """Return finest leaf-grid shape."""
        scale = 1 << self.max_level
        return tuple(int(v) * scale for v in self._root_shape)

    @property
    def level_counts(self) -> LevelCountTable:
        """Return `(level, leaf_count, fine_equivalent_count)` rows."""
        valid_levels = self.cell_levels[self.cell_levels >= 0]
        max_level = int(np.max(valid_levels))
        return tuple(
            (
                int(level),
                int(np.count_nonzero(valid_levels == level)),
                int(np.count_nonzero(valid_levels == level) * (8 ** int(max_level - level))),
            )
            for level in sorted(set(int(v) for v in valid_levels.tolist()))
        )

    @property
    def min_level(self) -> int:
        """Return minimum occupied refinement level."""
        valid_levels = self.cell_levels[self.cell_levels >= 0]
        return int(np.min(valid_levels))

    @property
    def max_level(self) -> int:
        """Return maximum occupied refinement level."""
        valid_levels = self.cell_levels[self.cell_levels >= 0]
        return int(np.max(valid_levels))

    @property
    def corners(self) -> np.ndarray:
        """Return leaf-row corner point ids in Tecplot/BATSRUS brick order."""
        return self._corners

    def normalize_leaf_cell_ids(self, cell_id: int | np.ndarray) -> np.ndarray:
        """Return validated leaf-slot ids as one `int64` array."""
        leaf_ids = np.asarray(cell_id, dtype=np.int64)
        if np.any(leaf_ids < 0) or np.any(leaf_ids >= self._leaf_slot_count):
            raise ValueError("Only valid leaf cell ids are supported.")
        if np.any(self._cell_depth[leaf_ids] < 0):
            raise ValueError("Only valid leaf cell ids are supported.")
        return leaf_ids

    def cell_points(self, cell_id: int | np.ndarray) -> np.ndarray:
        """Return explicit leaf corner point coordinates with shape `(..., 8, 3)`."""
        leaf_ids = self.normalize_leaf_cell_ids(cell_id)
        if leaf_ids.ndim == 0:
            return self._points[self._corners[int(leaf_ids)]]
        return self._points[self._corners[leaf_ids]]

    @property
    def cell_bounds(self) -> np.ndarray:
        """Return packed `(n_cells, 3, 2)` start/width bounds for rebuilt cells."""
        return self._cell_bounds

    @cached_property
    def cell_volumes(self) -> np.ndarray:
        """Return leaf-slot cell volumes aligned with `cell_levels`."""
        return self._octree_module.cell_volumes(self)

    def native_axis_slabs(self, axis: int) -> np.ndarray:
        """Return consecutive native-axis slab bounds induced by occupied leaf cells."""
        resolved_axis = int(axis)
        if resolved_axis < 0 or resolved_axis > 2:
            raise ValueError("axis must be one of 0, 1, or 2.")

        occupied_leaf_ids = np.flatnonzero(self.cell_levels >= 0)
        leaf_bounds = np.asarray(
            self.cell_bounds[occupied_leaf_ids, resolved_axis, :],
            dtype=np.float64,
        )
        slab_edges = np.unique(
            np.concatenate(
                (
                    leaf_bounds[:, BOUNDS_START_SLOT],
                    leaf_bounds[:, BOUNDS_START_SLOT] + leaf_bounds[:, BOUNDS_WIDTH_SLOT],
                )
            )
        )
        return np.column_stack((slab_edges[:-1], slab_edges[1:]))

    @property
    def packed_domain_bounds(self) -> np.ndarray:
        """Return packed `(3, 2)` start/width domain bounds in tree coordinates."""
        return self._domain_bounds

    @cached_property
    def lookup_tree(self) -> LookupTree:
        """Return the shared lookup-tree bundle for point-ownership kernels."""
        return LookupTree(
            self._cell_child,
            self._root_cell_ids,
            self._cell_parent,
            self._cell_bounds,
            self._domain_bounds,
            self._axis2_period,
            self._axis2_periodic,
        )

    @property
    def cell_levels(self) -> np.ndarray:
        """Return exact persisted leaf-slot levels, including unused slots as `-1`."""
        return self._cell_depth[: self._leaf_slot_count]

    @property
    def cell_depth(self) -> np.ndarray:
        """Return rebuilt runtime cell depths."""
        return self._cell_depth

    @property
    def cell_child(self) -> np.ndarray:
        """Return rebuilt runtime 8-child references, with `-1` for missing children."""
        return self._cell_child

    @property
    def root_cell_ids(self) -> np.ndarray:
        """Return rebuilt runtime root cell ids."""
        return self._root_cell_ids

    @property
    def cell_parent(self) -> np.ndarray:
        """Return rebuilt runtime parent references, with `-1` for roots."""
        return self._cell_parent

    @property
    def cell_ijk(self) -> np.ndarray:
        """Return rebuilt runtime cell addresses."""
        return self._cell_ijk

    @property
    def cell_neighbor(self) -> np.ndarray:
        """Return rebuilt runtime face/subface neighbor references for all occupied cells."""
        return self._cell_neighbor

    @cached_property
    def traversal_tree(self) -> TraversalTree:
        """Return the shared traversal-tree bundle for ray-tracing kernels."""
        return TraversalTree(
            self._root_cell_ids,
            self._cell_child,
            self._cell_bounds,
            self._domain_bounds,
            self._cell_neighbor,
            self._cell_depth,
        )

    @property
    def domain_bounds_packed(self) -> np.ndarray:
        """Return packed `(3, 2)` start/width domain bounds for the tree coordinate system."""
        return self._domain_bounds

    @property
    def radial_edges(self) -> np.ndarray:
        """Return spherical radial edge locations, with `NaN` where no occupied cell uses that fine edge."""
        return self._radial_edges

    @radial_edges.setter
    def radial_edges(self, radial_edges: np.ndarray) -> None:
        """Store spherical radial edge locations, with `NaN` where no occupied cell uses that fine edge."""
        self._radial_edges = radial_edges

    def save(self, path: str | Path) -> None:
        """Save this tree to a compressed `.npz` file."""
        state = OctreeState.from_tree(self)
        out_path = Path(path)
        state.save_npz(out_path)
        logger.info("Saved octree to %s", str(out_path))

    @classmethod
    def from_ds(
        cls,
        ds: Dataset,
        *,
        tree_coord: TreeCoord | None = None,
        build_axis_tol: float = 1e-12,
        build_level_rtol: float = 1e-4,
        build_level_atol: float = 1e-9,
    ) -> "Octree":
        """Build an octree from a dataset by extracting explicit points and corners.

        `tree_coord` optionally fixes the coordinate system. The `build_*`
        arguments only affect geometric inference during construction.
        """
        if ds.corners is None:
            raise ValueError("Dataset has no corners; cannot build octree.")
        logger.debug("from_ds...")
        t0 = time.perf_counter()
        points = np.column_stack(tuple(np.asarray(ds[name], dtype=np.float64) for name in XYZ_VARS))
        corners = np.asarray(ds.corners, dtype=np.int64)
        _warn_if_blocks_aux_mismatch(ds, int(corners.shape[0]))
        logger.info(
            "from_ds: n_points=%d n_cells=%d",
            int(points.shape[0]),
            int(corners.shape[0]),
        )
        tree = cls(
            points,
            corners,
            tree_coord=tree_coord,
            build_axis_tol=build_axis_tol,
            build_level_rtol=build_level_rtol,
            build_level_atol=build_level_atol,
        )
        logger.info("from_ds complete in %.2fs", float(time.perf_counter() - t0))
        return tree

    @classmethod
    def from_state(
        cls,
        state: "OctreeState",
        *,
        points: np.ndarray,
        corners: np.ndarray,
    ) -> "Octree":
        """Instantiate one tree from exact saved state and explicit point/corner geometry."""
        tree = cls.__new__(cls)
        cls._init_from_state(
            tree,
            root_shape=tuple(int(v) for v in state.root_shape),
            tree_coord=state.tree_coord,
            cell_levels=state.cell_levels,
            cell_ijk=state.cell_ijk,
            points=points,
            corners=corners,
        )
        return tree

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        points: np.ndarray,
        corners: np.ndarray,
    ) -> "Octree":
        """Load a tree from `.npz` and bind it to explicit point/corner geometry."""
        in_path = Path(path)
        state = OctreeState.load_npz(in_path)
        tree = cls.from_state(state, points=points, corners=corners)
        logger.info("Loaded octree from %s", str(in_path))
        return tree

    def __str__(self) -> str:
        """Return a compact human-readable tree summary."""
        leaf_levels = self.cell_levels
        n_leaf_cells = int(np.count_nonzero(leaf_levels >= 0))
        n_runtime_cells = int(np.count_nonzero(self._cell_depth >= 0))
        return (
            "Octree("
            f"tree_coord={self.tree_coord}, "
            f"root_shape={self.root_shape}, "
            f"leaf_shape={self.leaf_shape}, "
            f"leaf_cells={n_leaf_cells}, "
            f"runtime_cells={n_runtime_cells}, "
            f"levels={self.min_level}..{self.max_level}"
            ")"
        )

    @property
    def cell_count(self) -> int:
        """Return number of exact persisted leaf rows."""
        return int(self.cell_levels.shape[0])

    def lookup_points(self, points: np.ndarray, *, coord: TreeCoord) -> np.ndarray:
        """Resolve one batch of query points to leaf cell ids, with `-1` for misses."""
        query_points = np.array(points, dtype=np.float64, order="C")
        if query_points.ndim == 0 or query_points.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3).")
        shape = (1,) if query_points.ndim == 1 else query_points.shape[:-1]
        query_points = query_points.reshape(-1, 3)
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        cell_ids = self._octree_module.lookup_points(self, query_points, resolved_coord)
        return cell_ids.reshape(shape)

    def domain_bounds(self, *, coord: TreeCoord = "xyz") -> tuple[np.ndarray, np.ndarray]:
        """Return global `(lo, hi)` bounds in the tree's own coordinate system."""
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        if resolved_coord != self.tree_coord:
            raise ValueError(f"domain_bounds only supports coord={self.tree_coord!r} for this tree.")
        lo = np.array(self._domain_bounds[:, BOUNDS_START_SLOT], dtype=float)
        hi = np.array(
            self._domain_bounds[:, BOUNDS_START_SLOT] + self._domain_bounds[:, BOUNDS_WIDTH_SLOT],
            dtype=float,
        )
        return lo, hi

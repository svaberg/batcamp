#!/usr/bin/env python3
"""Core octree data structures and shared lookup/interpolation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import NamedTuple
from typing import TYPE_CHECKING

import numpy as np
from batread import Dataset
from numba import njit
from numba import prange

from .constants import DEFAULT_TREE_COORD
from .constants import SUPPORTED_TREE_COORDS
from .constants import XYZ_VARS
from .shared_types import GridIndex
from .shared_types import GridPath
from .shared_types import GridShape
from .shared_types import LevelCountTable
from .shared_types import TreeCoord

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .face_neighbors import OctreeFaceNeighbors

AXIS0 = 0  # Packed bounds axis index for the first tree coordinate.
AXIS1 = 1  # Packed bounds axis index for the second tree coordinate.
AXIS2 = 2  # Packed bounds axis index for the third tree coordinate.
START = 0  # Packed bounds slot index for interval start.
WIDTH = 1  # Packed bounds slot index for interval width.

class SphericalInterpKernelState(NamedTuple):
    """Packed spherical interpolation arrays for compiled flat trilinear kernels."""

    point_values: np.ndarray  # Shape `(n_points, n_components)`.

    corners: np.ndarray  # Shape `(n_cells, 8)`. Corner point ids for leaf cells; non-leaf rows are `-1`.
    bin_to_corner: np.ndarray  # Shape `(n_cells, 8)`. Logical trilinear corner order -> row-local corner index.

    cell_a_full: np.ndarray  # Shape `(n_cells,)`. `True` where the azimuth span is effectively the full circle.
    cell_a_tiny: np.ndarray  # Shape `(n_cells,)`. `True` where the azimuth span is effectively zero.


class CartesianInterpKernelState(NamedTuple):
    """Packed Cartesian interpolation arrays for compiled flat trilinear kernels."""

    point_values: np.ndarray  # Shape `(n_points, n_components)`.

    corners: np.ndarray  # Shape `(n_cells, 8)`. Corner point ids for leaf cells; non-leaf rows are `-1`.
    bin_to_corner: np.ndarray  # Shape `(n_cells, 8)`. Logical trilinear corner order -> row-local corner index.


_TRILINEAR_TARGET_BITS = np.array(
    [[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)],
    dtype=np.int8,
)

def _coord_state_inputs(
    tree: "Octree",
    ds: Dataset,
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the bound dataset arrays and exact leaf levels needed for coordinate state."""
    missing = [
        name
        for name in (
            "_i0",
            "_i1",
            "_i2",
            "_cell_depth",
            "_cell_i0",
            "_cell_i1",
            "_cell_i2",
            "_cell_child",
            "_root_cell_ids",
            "_cell_parent",
        )
        if not hasattr(tree, name)
    ]
    if missing:
        raise ValueError(f"Lookup requires exact tree state: missing {missing}.")
    x, y, z = (np.asarray(ds[name], dtype=np.float64) for name in XYZ_VARS)
    cell_levels = tree.cell_levels
    if cell_levels is None or int(cell_levels.shape[0]) != int(corners.shape[0]):
        raise ValueError("Lookup requires exact cell_levels.")
    return corners, x, y, z, cell_levels


def _build_interp_bin_to_corner(
    axis0: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    *,
    axis0_start: np.ndarray,
    axis0_width: np.ndarray,
    axis1_start: np.ndarray,
    axis1_width: np.ndarray,
    axis2_start: np.ndarray,
    axis2_width: np.ndarray,
    axis2_periodic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build logical trilinear corner indices from per-corner axis coordinates."""
    tiny = np.finfo(float).tiny

    axis0_mid = axis0_start[:, None] + 0.5 * axis0_width[:, None]
    axis1_mid = axis1_start[:, None] + 0.5 * axis1_width[:, None]
    bit0 = (axis0 >= axis0_mid).astype(np.int8)
    bit1 = (axis1 >= axis1_mid).astype(np.int8)
    if axis2_periodic:
        axis2_full = axis2_width >= (2.0 * np.pi - 1.0e-10)
        axis2_tiny = axis2_width <= tiny
        axis2_rel = np.mod(axis2 - axis2_start[:, None], 2.0 * np.pi)
        axis2_rel = np.where(
            (~axis2_full)[:, None],
            np.clip(axis2_rel, 0.0, axis2_width[:, None]),
            axis2_rel,
        )
        axis2_mid = 0.5 * axis2_width[:, None]
        bit2 = np.zeros_like(bit0, dtype=np.int8)
        valid_axis2 = ~axis2_tiny
        if np.any(valid_axis2):
            bit2[valid_axis2] = (axis2_rel[valid_axis2] >= axis2_mid[valid_axis2]).astype(np.int8)
    else:
        axis2_full = np.zeros(axis2_width.shape[0], dtype=bool)
        axis2_tiny = np.zeros(axis2_width.shape[0], dtype=bool)
        axis2_mid = axis2_start[:, None] + 0.5 * axis2_width[:, None]
        bit2 = (axis2 >= axis2_mid).astype(np.int8)

    bin_id = bit0 + (bit1 << 1) + (bit2 << 2)
    bit_trip = np.stack((bit0, bit1, bit2), axis=2)
    n_cells = int(axis0.shape[0])
    bin_to_corner = np.empty((n_cells, 8), dtype=np.int64)
    for k in range(8):
        eq = bin_id == k
        has = np.any(eq, axis=1)
        pick = np.argmax(eq, axis=1).astype(np.int64)
        missing = ~has
        if np.any(missing):
            d = np.sum((bit_trip[missing] - _TRILINEAR_TARGET_BITS[k]) ** 2, axis=2)
            pick[missing] = np.argmin(d, axis=1)
        bin_to_corner[:, k] = pick
    return (
        bin_to_corner,
        axis2_full,
        axis2_tiny,
    )


@njit(cache=True)
def _contains_lookup_interval(
    value: float,
    start: float,
    width: float,
    periodic: bool,
    period: float,
    tol: float,
) -> bool:
    """Return whether one coordinate lies inside one start/width interval."""
    if periodic:
        if width >= (period - tol):
            return True
        delta = (value - start) % period
        return delta <= (width + tol)
    return value >= (start - tol) and value <= (start + width + tol)


@njit(cache=True)
def _contains_lookup_coords(
    q0: float,
    q1: float,
    q2: float,
    axis0_start: float,
    axis0_width: float,
    axis1_start: float,
    axis1_width: float,
    axis2_start: float,
    axis2_width: float,
    axis2_period: float,
    axis2_periodic: bool,
    tol: float,
) -> bool:
    """Return whether one query lies inside one axis-0/1/2 box."""
    if not _contains_lookup_interval(
        q0,
        axis0_start,
        axis0_width,
        periodic=False,
        period=0.0,
        tol=tol,
    ):
        return False
    if not _contains_lookup_interval(
        q1,
        axis1_start,
        axis1_width,
        periodic=False,
        period=0.0,
        tol=tol,
    ):
        return False
    return _contains_lookup_interval(
        q2,
        axis2_start,
        axis2_width,
        periodic=bool(axis2_periodic),
        period=float(axis2_period),
        tol=tol,
    )


@njit(cache=True)
def _contains_lookup_cell(
    cid: int,
    q0: float,
    q1: float,
    q2: float,
    cell_is_leaf: np.ndarray,
    cell_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    tol: float,
) -> bool:
    """Return whether one query lies inside one leaf cell geometry."""
    if not cell_is_leaf[cid]:
        return False
    return _contains_octree_cell(cid, q0, q1, q2, cell_bounds, axis2_period, axis2_periodic, tol)


@njit(cache=True)
def _contains_octree_cell(
    cell_id: int,
    q0: float,
    q1: float,
    q2: float,
    cell_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    tol: float,
) -> bool:
    """Return whether one query lies inside one occupied octree cell."""
    cid = int(cell_id)
    return _contains_lookup_coords(
        q0,
        q1,
        q2,
        float(cell_bounds[cid, AXIS0, START]),
        float(cell_bounds[cid, AXIS0, WIDTH]),
        float(cell_bounds[cid, AXIS1, START]),
        float(cell_bounds[cid, AXIS1, WIDTH]),
        float(cell_bounds[cid, AXIS2, START]),
        float(cell_bounds[cid, AXIS2, WIDTH]),
        axis2_period,
        axis2_periodic,
        tol=tol,
    )


@njit(cache=True)
def _contains_lookup_domain(
    q0: float,
    q1: float,
    q2: float,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    tol: float = 0.0,
) -> bool:
    """Return whether one query lies inside the global lookup domain."""
    return _contains_lookup_coords(
        q0,
        q1,
        q2,
        domain_bounds[AXIS0, START],
        domain_bounds[AXIS0, WIDTH],
        domain_bounds[AXIS1, START],
        domain_bounds[AXIS1, WIDTH],
        domain_bounds[AXIS2, START],
        domain_bounds[AXIS2, WIDTH],
        axis2_period,
        axis2_periodic,
        tol=tol,
    )


@njit(cache=True)
def _lookup_hint_cell(
    prev_cell_id: int,
    q0: float,
    q1: float,
    q2: float,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    tol: float,
) -> int:
    """Return the nearest ancestor hint cell containing one query, or `-1`."""
    current = int(prev_cell_id)
    while current >= 0:
        if _contains_octree_cell(current, q0, q1, q2, cell_bounds, axis2_period, axis2_periodic, tol):
            return current
        current = int(cell_parent[current])
    return -1


@njit(cache=True)
def _lookup_descend_to_leaf_cell(
    q0: float,
    q1: float,
    q2: float,
    start_cell_id: int,
    cell_is_leaf: np.ndarray,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    tol: float,
) -> int:
    """Descend one sparse tree from a containing cell hint, or from the roots."""
    current = int(start_cell_id)
    if current < 0:
        for root_pos in range(int(root_cell_ids.shape[0])):
            cell_id = int(root_cell_ids[root_pos])
            if _contains_octree_cell(cell_id, q0, q1, q2, cell_bounds, axis2_period, axis2_periodic, tol):
                current = cell_id
                break
    if current < 0:
        return -1

    while True:
        if cell_is_leaf[current]:
            return int(current)

        found_child = False
        for child_ord in range(8):
            child_id = int(cell_child[current, child_ord])
            if child_id < 0:
                continue
            if _contains_octree_cell(child_id, q0, q1, q2, cell_bounds, axis2_period, axis2_periodic, tol):
                current = child_id
                found_child = True
                break
        if not found_child:
            return -1


@njit(cache=True)
def _lookup_leaf_cell_id_kernel(
    q0: float,
    q1: float,
    q2: float,
    cell_is_leaf: np.ndarray,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    prev_cell_id: int = -1,
    tol: float = 1.0e-10,
) -> int:
    """Resolve one query to a leaf cell id by sparse-tree descent."""
    if not (np.isfinite(q0) and np.isfinite(q1) and np.isfinite(q2)):
        return -1
    if prev_cell_id >= 0 and _contains_octree_cell(
        int(prev_cell_id), q0, q1, q2, cell_bounds, axis2_period, axis2_periodic, tol
    ):
        if cell_is_leaf[int(prev_cell_id)]:
            return int(prev_cell_id)
    if not _contains_lookup_domain(q0, q1, q2, domain_bounds, axis2_period, axis2_periodic):
        return -1
    current = _lookup_hint_cell(
        prev_cell_id,
        q0,
        q1,
        q2,
        cell_parent,
        cell_bounds,
        axis2_period,
        axis2_periodic,
        tol,
    )
    return _lookup_descend_to_leaf_cell(
        q0,
        q1,
        q2,
        current,
        cell_is_leaf,
        cell_child,
        root_cell_ids,
        cell_bounds,
        axis2_period,
        axis2_periodic,
        tol,
    )


@njit(cache=True, parallel=True)
def _lookup_leaf_cell_ids_kernel(
    queries: np.ndarray,
    cell_is_leaf: np.ndarray,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
) -> np.ndarray:
    """Resolve a batch of same-coordinate queries to leaf cell ids."""
    n_query = int(queries.shape[0])
    cell_ids = np.full(n_query, -1, dtype=np.int64)
    chunk_size = 1024
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    for chunk_id in prange(n_chunks):
        start = chunk_id * chunk_size
        end = min(n_query, start + chunk_size)
        hint_cell_id = -1
        for i in range(start, end):
            cell_id = _lookup_leaf_cell_id_kernel(
                queries[i, 0],
                queries[i, 1],
                queries[i, 2],
                cell_is_leaf,
                cell_child,
                root_cell_ids,
                cell_parent,
                cell_bounds,
                domain_bounds,
                axis2_period,
                axis2_periodic,
                hint_cell_id,
            )
            cell_ids[i] = cell_id
            hint_cell_id = int(cell_id) if cell_id >= 0 else -1
    return cell_ids


@njit(cache=True)
def _lookup_cell_id_kernel(
    q0: float,
    q1: float,
    q2: float,
    cell_is_leaf: np.ndarray,
    cell_child: np.ndarray,
    root_cell_ids: np.ndarray,
    cell_parent: np.ndarray,
    cell_bounds: np.ndarray,
    domain_bounds: np.ndarray,
    axis2_period: float,
    axis2_periodic: bool,
    prev_cid: int = -1,
    tol: float = 1.0e-10,
) -> int:
    """Resolve one query to a cell id by sparse-tree descent."""
    cell_id = _lookup_leaf_cell_id_kernel(
        q0,
        q1,
        q2,
        cell_is_leaf,
        cell_child,
        root_cell_ids,
        cell_parent,
        cell_bounds,
        domain_bounds,
        axis2_period,
        axis2_periodic,
        int(prev_cid),
        tol,
    )
    return int(cell_id)


def _level_metadata_from_leaves(
    root_shape: GridShape,
    cell_levels: np.ndarray,
) -> tuple[GridShape, bool, LevelCountTable, int, int]:
    """Rebuild tree summary metadata from exact leaf levels."""
    levels = np.asarray(cell_levels, dtype=np.int64)
    valid = levels >= 0
    valid_levels = levels[valid]
    if valid_levels.size == 0:
        raise ValueError("Octree state requires at least one valid cell level.")
    min_level = int(np.min(valid_levels))
    max_level = int(np.max(valid_levels))
    scale = 1 << max_level
    leaf_shape = tuple(int(v) * scale for v in root_shape)
    level_counts = tuple(
        (
            int(level),
            int(np.count_nonzero(valid_levels == level)),
            int(np.count_nonzero(valid_levels == level) * (8 ** int(max_level - level))),
        )
        for level in sorted(set(int(v) for v in valid_levels.tolist()))
    )
    weighted_cells = int(sum(item[2] for item in level_counts))
    is_full = int(np.count_nonzero(valid)) == int(levels.size) and weighted_cells == int(np.prod(leaf_shape))
    return leaf_shape, bool(is_full), level_counts, min_level, max_level


def _build_cell_arrays(
    depths: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    leaf_value: np.ndarray,
    *,
    tree_depth: int,
    label: str,
    n_leaf_slots: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build rebuilt octree cell arrays from exact leaf addresses."""
    leaf_depth_raw = np.asarray(depths, dtype=np.int64)
    leaf_i0_raw = np.asarray(i0, dtype=np.int64)
    leaf_i1_raw = np.asarray(i1, dtype=np.int64)
    leaf_i2_raw = np.asarray(i2, dtype=np.int64)
    leaf_value_arr = np.asarray(leaf_value, dtype=np.int64)

    leaf_order = np.lexsort((leaf_i2_raw, leaf_i1_raw, leaf_i0_raw, leaf_depth_raw))
    leaf_depth = leaf_depth_raw[leaf_order]
    leaf_i0 = leaf_i0_raw[leaf_order]
    leaf_i1 = leaf_i1_raw[leaf_order]
    leaf_i2 = leaf_i2_raw[leaf_order]

    same_leaf = (
        (leaf_depth[1:] == leaf_depth[:-1])
        & (leaf_i0[1:] == leaf_i0[:-1])
        & (leaf_i1[1:] == leaf_i1[:-1])
        & (leaf_i2[1:] == leaf_i2[:-1])
    )
    if np.any(same_leaf):
        dup = int(np.flatnonzero(same_leaf)[0])
        raise ValueError(
            f"{label} cells overlap at octree address "
            f"{(int(leaf_depth[dup]), int(leaf_i0[dup]), int(leaf_i1[dup]), int(leaf_i2[dup]))}."
        )

    parent_depth_parts: list[np.ndarray] = []
    parent_i0_parts: list[np.ndarray] = []
    parent_i1_parts: list[np.ndarray] = []
    parent_i2_parts: list[np.ndarray] = []
    for parent_depth in range(int(tree_depth)):
        mask = depths > int(parent_depth)
        if not np.any(mask):
            continue
        up = np.asarray(depths[mask] - int(parent_depth), dtype=np.int64)
        parent_cells = np.column_stack(
            (
                np.full(int(np.count_nonzero(mask)), int(parent_depth), dtype=np.int64),
                np.right_shift(i0[mask], up),
                np.right_shift(i1[mask], up),
                np.right_shift(i2[mask], up),
            )
        )
        parent_cells = np.unique(parent_cells, axis=0)
        parent_depth_parts.append(parent_cells[:, 0].astype(np.int64, copy=False))
        parent_i0_parts.append(parent_cells[:, 1].astype(np.int64, copy=False))
        parent_i1_parts.append(parent_cells[:, 2].astype(np.int64, copy=False))
        parent_i2_parts.append(parent_cells[:, 3].astype(np.int64, copy=False))

    if parent_depth_parts:
        internal_depth = np.concatenate(parent_depth_parts)
        internal_i0 = np.concatenate(parent_i0_parts)
        internal_i1 = np.concatenate(parent_i1_parts)
        internal_i2 = np.concatenate(parent_i2_parts)
        internal_order = np.lexsort((internal_i2, internal_i1, internal_i0, internal_depth))
        internal_depth = internal_depth[internal_order]
        internal_i0 = internal_i0[internal_order]
        internal_i1 = internal_i1[internal_order]
        internal_i2 = internal_i2[internal_order]
    else:
        internal_depth = np.empty(0, dtype=np.int64)
        internal_i0 = np.empty(0, dtype=np.int64)
        internal_i1 = np.empty(0, dtype=np.int64)
        internal_i2 = np.empty(0, dtype=np.int64)

    all_depth = np.concatenate((leaf_depth, internal_depth))
    all_i0 = np.concatenate((leaf_i0, internal_i0))
    all_i1 = np.concatenate((leaf_i1, internal_i1))
    all_i2 = np.concatenate((leaf_i2, internal_i2))
    all_order = np.lexsort((all_i2, all_i1, all_i0, all_depth))
    all_depth = all_depth[all_order]
    all_i0 = all_i0[all_order]
    all_i1 = all_i1[all_order]
    all_i2 = all_i2[all_order]

    same_cell = (
        (all_depth[1:] == all_depth[:-1])
        & (all_i0[1:] == all_i0[:-1])
        & (all_i1[1:] == all_i1[:-1])
        & (all_i2[1:] == all_i2[:-1])
    )
    if np.any(same_cell):
        dup = int(np.flatnonzero(same_cell)[0])
        raise ValueError(
            f"{label} cells overlap across parent/child addresses at "
            f"({int(all_depth[dup])}, {int(all_i0[dup])}, {int(all_i1[dup])}, {int(all_i2[dup])})."
        )

    leaf_slots = int(np.max(leaf_value_arr)) + 1 if n_leaf_slots is None and leaf_value_arr.size else int(n_leaf_slots or 0)
    n_cells = leaf_slots + int(internal_depth.shape[0])
    cell_depth = np.full(n_cells, -1, dtype=np.int64)
    cell_i0 = np.full(n_cells, -1, dtype=np.int64)
    cell_i1 = np.full(n_cells, -1, dtype=np.int64)
    cell_i2 = np.full(n_cells, -1, dtype=np.int64)
    cell_depth[leaf_value_arr] = leaf_depth_raw
    cell_i0[leaf_value_arr] = leaf_i0_raw
    cell_i1[leaf_value_arr] = leaf_i1_raw
    cell_i2[leaf_value_arr] = leaf_i2_raw
    start = leaf_slots
    stop = start + int(internal_depth.shape[0])
    cell_depth[start:stop] = internal_depth
    cell_i0[start:stop] = internal_i0
    cell_i1[start:stop] = internal_i1
    cell_i2[start:stop] = internal_i2
    return cell_depth, cell_i0, cell_i1, cell_i2


def _build_cell_children(
    cell_depth: np.ndarray,
    cell_i0: np.ndarray,
    cell_i1: np.ndarray,
    cell_i2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse 8-child references for occupied rebuilt cells."""
    n_cells = int(np.asarray(cell_depth, dtype=np.int64).shape[0])
    cell_child = np.full((n_cells, 8), -1, dtype=np.int64)
    cell_parent = np.full(n_cells, -1, dtype=np.int64)
    occupied = np.flatnonzero(cell_depth >= 0).astype(np.int64)
    key_to_cell = {
        (int(cell_depth[idx]), int(cell_i0[idx]), int(cell_i1[idx]), int(cell_i2[idx])): int(idx)
        for idx in occupied
    }
    for idx in occupied:
        depth = int(cell_depth[idx])
        i0 = int(cell_i0[idx])
        i1 = int(cell_i1[idx])
        i2 = int(cell_i2[idx])
        for child_ord in range(8):
            b0 = (child_ord >> 2) & 1
            b1 = (child_ord >> 1) & 1
            b2 = child_ord & 1
            child_key = (depth + 1, 2 * i0 + b0, 2 * i1 + b1, 2 * i2 + b2)
            child_idx = key_to_cell.get(child_key)
            if child_idx is not None:
                cell_child[idx, child_ord] = int(child_idx)
                cell_parent[int(child_idx)] = int(idx)
    root_cell_ids = np.flatnonzero(np.asarray(cell_depth, dtype=np.int64) == 0).astype(np.int64)
    return cell_child, root_cell_ids, cell_parent


def _cell_state_from_leaves(
    cell_levels: np.ndarray,
    cell_i0: np.ndarray,
    cell_i1: np.ndarray,
    cell_i2: np.ndarray,
    *,
    max_level: int,
) -> tuple[np.ndarray, ...]:
    """Rebuild exact occupied cells from leaf addresses."""

    levels = np.asarray(cell_levels, dtype=np.int64)
    i0_all = np.asarray(cell_i0, dtype=np.int64)
    i1_all = np.asarray(cell_i1, dtype=np.int64)
    i2_all = np.asarray(cell_i2, dtype=np.int64)
    if not (levels.shape == i0_all.shape == i1_all.shape == i2_all.shape):
        raise ValueError("Octree leaf level/index arrays must have matching shapes.")
    valid_ids = np.flatnonzero(levels >= 0).astype(np.int64)
    if valid_ids.size == 0:
        raise ValueError("Octree state requires at least one valid leaf cell.")
    depths = np.asarray(levels[valid_ids], dtype=np.int64)
    leaf_i0 = np.asarray(i0_all[valid_ids], dtype=np.int64)
    leaf_i1 = np.asarray(i1_all[valid_ids], dtype=np.int64)
    leaf_i2 = np.asarray(i2_all[valid_ids], dtype=np.int64)
    cell_depth, cell_i0_rt, cell_i1_rt, cell_i2_rt = _build_cell_arrays(
        depths,
        leaf_i0,
        leaf_i1,
        leaf_i2,
        valid_ids,
        tree_depth=int(max_level),
        label="Restored",
        n_leaf_slots=int(levels.shape[0]),
    )
    cell_child, root_cell_ids, cell_parent = _build_cell_children(
        cell_depth,
        cell_i0_rt,
        cell_i1_rt,
        cell_i2_rt,
    )
    return cell_depth, cell_i0_rt, cell_i1_rt, cell_i2_rt, cell_child, root_cell_ids, cell_parent


def _normalize_octree_inputs(
    *,
    root_shape: GridShape,
    tree_coord: TreeCoord,
    cell_levels: np.ndarray,
    cell_i0: np.ndarray,
    cell_i1: np.ndarray,
    cell_i2: np.ndarray,
) -> tuple[GridShape, TreeCoord, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize constructor inputs to the exact persisted leaf-state arrays."""
    resolved_tree_coord = str(tree_coord)
    if resolved_tree_coord not in SUPPORTED_TREE_COORDS:
        raise ValueError(
            f"Unsupported tree_coord '{resolved_tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
        )
    return (
        tuple(int(v) for v in root_shape),
        resolved_tree_coord,
        np.asarray(cell_levels, dtype=np.int64),
        np.asarray(cell_i0, dtype=np.int64),
        np.asarray(cell_i1, dtype=np.int64),
        np.asarray(cell_i2, dtype=np.int64),
    )


def _normalize_bound_dataset(ds: Dataset) -> np.ndarray:
    """Validate one bound dataset and return packed corners."""
    if ds.corners is None:
        raise ValueError("Dataset has no corners; cannot bind octree lookup.")
    if not set(XYZ_VARS).issubset(set(ds.variables)):
        raise ValueError("Dataset must provide X/Y/Z variables to bind octree lookup.")
    return np.asarray(ds.corners, dtype=np.int64)


def _build_interpolation_geometry(
    tree: "Octree",
    ds: Dataset,
    corners_all: np.ndarray,
    cell_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the leaf-cell trilinear interpolation arrays from the bound dataset."""
    leaf_cell_ids = np.flatnonzero(tree.cell_levels >= 0).astype(np.int64)
    corners = corners_all[leaf_cell_ids]
    leaf_bounds = cell_bounds[leaf_cell_ids]
    axis0_start = leaf_bounds[:, AXIS0, START]
    axis0_width = leaf_bounds[:, AXIS0, WIDTH]
    axis1_start = leaf_bounds[:, AXIS1, START]
    axis1_width = leaf_bounds[:, AXIS1, WIDTH]
    axis2_start = leaf_bounds[:, AXIS2, START]
    axis2_width = leaf_bounds[:, AXIS2, WIDTH]
    x = np.asarray(ds[XYZ_VARS[0]], dtype=np.float64)
    y = np.asarray(ds[XYZ_VARS[1]], dtype=np.float64)
    z = np.asarray(ds[XYZ_VARS[2]], dtype=np.float64)
    if str(tree.tree_coord) == "xyz":
        axis0 = x[corners]
        axis1 = y[corners]
        axis2 = z[corners]
        axis2_periodic = False
    else:
        from .spherical import _xyz_arrays_to_rpa

        point_r, point_p, point_a = _xyz_arrays_to_rpa(x, y, z)
        axis0 = point_r[corners]
        axis1 = point_p[corners]
        axis2 = point_a[corners]
        axis2_periodic = True
    leaf_bin_to_corner, leaf_axis2_full, leaf_axis2_tiny = _build_interp_bin_to_corner(
        axis0,
        axis1,
        axis2,
        axis0_start=axis0_start,
        axis0_width=axis0_width,
        axis1_start=axis1_start,
        axis1_width=axis1_width,
        axis2_start=axis2_start,
        axis2_width=axis2_width,
        axis2_periodic=axis2_periodic,
    )
    n_cells = int(tree._cell_depth.shape[0])
    interp_corners = np.full((n_cells, 8), -1, dtype=np.int64)
    interp_corners[leaf_cell_ids] = corners
    interp_bin_to_corner = np.zeros((n_cells, 8), dtype=np.int64)
    interp_bin_to_corner[leaf_cell_ids] = leaf_bin_to_corner
    interp_axis2_full = np.zeros(n_cells, dtype=np.bool_)
    interp_axis2_full[leaf_cell_ids] = leaf_axis2_full
    interp_axis2_tiny = np.zeros(n_cells, dtype=np.bool_)
    interp_axis2_tiny[leaf_cell_ids] = leaf_axis2_tiny
    return (
        interp_corners,
        interp_bin_to_corner,
        interp_axis2_full,
        interp_axis2_tiny,
    )


class Octree:
    """Adaptive octree summary plus bound lookup entrypoints.

    `level_counts` rows are
    `(level, leaf_count, fine_equivalent_count)`.
    """

    def __init__(
        self,
        *,
        root_shape: GridShape,
        tree_coord: TreeCoord = DEFAULT_TREE_COORD,
        cell_levels: np.ndarray,
        cell_i0: np.ndarray,
        cell_i1: np.ndarray,
        cell_i2: np.ndarray,
        ds: Dataset,
    ) -> None:
        """Build one octree directly from exact leaf addresses."""
        (
            self.root_shape,
            self.tree_coord,
            self.cell_levels,
            self._i0,
            self._i1,
            self._i2,
        ) = _normalize_octree_inputs(
            root_shape=root_shape,
            tree_coord=tree_coord,
            cell_levels=cell_levels,
            cell_i0=cell_i0,
            cell_i1=cell_i1,
            cell_i2=cell_i2,
        )

        self.leaf_shape, self.is_full, self.level_counts, self.min_level, self.max_level = _level_metadata_from_leaves(
            self.root_shape,
            self.cell_levels,
        )
        (
            self._cell_depth,
            self._cell_i0,
            self._cell_i1,
            self._cell_i2,
            self._cell_child,
            self._root_cell_ids,
            self._cell_parent,
        ) = _cell_state_from_leaves(
            self.cell_levels,
            self._i0,
            self._i1,
            self._i2,
            max_level=self.max_level,
        )
        corners = _normalize_bound_dataset(ds)
        cell_bounds, domain_bounds, axis2_period, axis2_periodic = self._coord_support(str(self.tree_coord))._attach_coord_state(
            self, ds, corners
        )
        (
            interp_corners,
            interp_bin_to_corner,
            interp_axis2_full,
            interp_axis2_tiny,
        ) = _build_interpolation_geometry(self, ds, corners, cell_bounds)
        self._cell_is_leaf = (self._cell_depth >= 0) & np.all(self._cell_child < 0, axis=1)
        self.ds = ds
        self._corners = corners
        self._cell_bounds = cell_bounds
        self._domain_bounds = domain_bounds
        self._axis2_period = float(axis2_period)
        self._axis2_periodic = bool(axis2_periodic)
        self._interp_corners = interp_corners
        self._interp_bin_to_corner = interp_bin_to_corner
        self._interp_axis2_full = interp_axis2_full
        self._interp_axis2_tiny = interp_axis2_tiny

    @property
    def levels(self) -> tuple[int, ...]:
        """Return the sorted refinement levels present in this tree."""
        return tuple(int(level) for level, _count, _expected in self.level_counts)

    @property
    def is_uniform(self) -> bool:
        """Return `True` when all cells are at one refinement level."""
        return int(self.min_level) == int(self.max_level)

    @property
    def depth(self) -> int:
        """Return the maximum root-relative level.

        `depth` is kept as a read-only alias for `max_level` so the tree has
        one refinement coordinate system internally.
        """
        return int(self.max_level)

    def save(self, path: str | Path) -> None:
        """Save this tree to a compressed `.npz` file."""
        from .persistence import OctreeState

        state = OctreeState.from_tree(self)
        out_path = Path(path)
        state.save_npz(out_path)
        logger.info("Saved octree to %s", str(out_path))

    @classmethod
    def from_state(
        cls,
        state: "OctreeState",
        *,
        ds: Dataset,
    ) -> "Octree":
        """Instantiate one tree from exact saved state."""
        if str(state.tree_coord) not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported tree_coord '{state.tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        return cls(
            root_shape=tuple(int(v) for v in state.root_shape),
            tree_coord=state.tree_coord,
            cell_levels=state.cell_levels,
            cell_i0=state.cell_i0,
            cell_i1=state.cell_i1,
            cell_i2=state.cell_i2,
            ds=ds,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        ds: Dataset,
    ) -> "Octree":
        """Load a tree from `.npz` and bind it to the given dataset."""
        from .persistence import OctreeState

        in_path = Path(path)
        state = OctreeState.load_npz(in_path)
        tree = cls.from_state(state, ds=ds)
        logger.info("Loaded octree from %s", str(in_path))
        return tree

    def __str__(self) -> str:
        """Return human-readable summary text."""
        leaf_levels = ", ".join(
            f"L{level}:{count} (fine-equiv {expected})"
            for level, count, expected in self.level_counts
        )
        shape_kind = "uniform" if self.is_uniform else "adaptive"
        return (
            f"Octree ({shape_kind}): "
            f"tree_coord={self.tree_coord}, "
            f"finest_leaf_grid={self.leaf_shape}, root_grid={self.root_shape}, "
            f"max_level={self.max_level}, full={self.is_full}, "
            f"levels={self.min_level}..{self.max_level}; leaf_levels[{leaf_levels}]"
        )

    @staticmethod
    def _coord_support(tree_coord: str) -> type:
        """Return the geometry-specific support class for one tree coordinate."""
        if tree_coord == "xyz":
            from .cartesian import _CartesianCoordSupport

            return _CartesianCoordSupport
        if tree_coord == "rpa":
            from .spherical import _SphericalCoordSupport

            return _SphericalCoordSupport
        raise ValueError(
            f"Unsupported tree_coord '{tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
        )

    @staticmethod
    def _path(i0: int, i1: int, i2: int, level: int) -> GridPath:
        """Construct the root-to-leaf grid-index path for one cell."""
        out: list[GridIndex] = []
        for path_level in range(level + 1):
            shift = level - path_level
            out.append((i0 >> shift, i1 >> shift, i2 >> shift))
        return tuple(out)

    def _hit_from_chosen(self, chosen: int, *, allow_invalid_level: bool = False) -> "LookupHit | None":
        """Build one `LookupHit` from an internal cell id."""
        if chosen < 0:
            return None
        level = int(self.cell_levels[chosen])
        if level < 0 and not allow_invalid_level:
            return None
        if level < 0:
            path_level = int(self.max_level)
        else:
            path_level = int(level)
            if path_level < 0:
                raise ValueError(f"Derived negative level {level}; max_level={self.max_level}.")
        cell_i0 = int(self._i0[chosen])
        cell_i1 = int(self._i1[chosen])
        cell_i2 = int(self._i2[chosen])
        return LookupHit(
            cell_id=int(chosen),
            level=level,
            i0=cell_i0,
            i1=cell_i1,
            i2=cell_i2,
            path=self._path(cell_i0, cell_i1, cell_i2, path_level),
        )

    @property
    def cell_count(self) -> int:
        """Return number of leaf cells in the exact tree state."""
        return int(self.cell_levels.shape[0])

    def lookup_points(self, points: np.ndarray, *, coord: TreeCoord) -> np.ndarray:
        """Resolve one batch of query points to leaf cell ids, with `-1` for misses."""
        q = np.array(points, dtype=np.float64, order="C")
        if q.ndim == 1:
            if q.size != 3:
                raise ValueError("points must have shape (..., 3).")
            shape = (1,)
            q = q.reshape(1, 3)
        else:
            if q.shape[-1] != 3:
                raise ValueError("points must have shape (..., 3).")
            shape = q.shape[:-1]
            q = q.reshape(-1, 3)
        _q_local, cell_ids = self._lookup_points_local(q, coord=coord)
        return cell_ids.reshape(shape)

    def _lookup_points_local(self, points: np.ndarray, *, coord: TreeCoord) -> tuple[np.ndarray, np.ndarray]:
        """Return tree-coordinate queries plus resolved cell ids for one batch."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        if str(self.tree_coord) == "xyz":
            if resolved_coord != "xyz":
                raise ValueError("Cartesian lookup supports only coord='xyz'.")
            return points, _lookup_leaf_cell_ids_kernel(
                points,
                self._cell_is_leaf,
                self._cell_child,
                self._root_cell_ids,
                self._cell_parent,
                self._cell_bounds,
                self._domain_bounds,
                self._axis2_period,
                self._axis2_periodic,
            )
        if resolved_coord == "rpa":
            return points, _lookup_leaf_cell_ids_kernel(
                points,
                self._cell_is_leaf,
                self._cell_child,
                self._root_cell_ids,
                self._cell_parent,
                self._cell_bounds,
                self._domain_bounds,
                self._axis2_period,
                self._axis2_periodic,
            )
        from .spherical import _xyz_arrays_to_rpa

        axis0, axis1, axis2 = _xyz_arrays_to_rpa(points[:, 0], points[:, 1], points[:, 2])
        q_local = np.column_stack((axis0, axis1, axis2))
        return q_local, _lookup_leaf_cell_ids_kernel(
            q_local,
            self._cell_is_leaf,
            self._cell_child,
            self._root_cell_ids,
            self._cell_parent,
            self._cell_bounds,
            self._domain_bounds,
            self._axis2_period,
            self._axis2_periodic,
        )

    def _interp_state_from_values(
        self,
        point_values: np.ndarray,
    ) -> CartesianInterpKernelState | SphericalInterpKernelState:
        """Pack one interpolation state for the given per-point values."""
        if str(self.tree_coord) == "xyz":
            return CartesianInterpKernelState(
                point_values=point_values,
                corners=self._interp_corners,
                bin_to_corner=self._interp_bin_to_corner,
            )
        return SphericalInterpKernelState(
            point_values=point_values,
            corners=self._interp_corners,
            bin_to_corner=self._interp_bin_to_corner,
            cell_a_full=self._interp_axis2_full,
            cell_a_tiny=self._interp_axis2_tiny,
        )

    def _frontier_cells(self, max_level: int) -> tuple[np.ndarray, ...]:
        """Return unique frontier cells by truncating leaves to one level cutoff."""
        if self.cell_levels is None:
            raise ValueError("Octree has no cell_levels; cannot build frontier nodes.")
        required = ("_i0", "_i1", "_i2")
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError(f"Octree frontier nodes require exact leaf addresses: missing {missing}.")

        levels_all = self.cell_levels
        valid = levels_all >= 0
        if not np.any(valid):
            raise ValueError("Octree contains no valid cells (all levels are < 0).")

        cell_ids = np.flatnonzero(valid)
        i0_all = self._i0
        i1_all = self._i1
        i2_all = self._i2
        if not (levels_all.shape == i0_all.shape == i1_all.shape == i2_all.shape):
            raise ValueError("Cell level/index arrays must have matching shapes.")

        levels_valid = levels_all[valid]
        active_levels = np.minimum(levels_valid, int(max_level))
        shift = levels_valid - active_levels
        active_i0 = np.right_shift(i0_all[valid], shift)
        active_i1 = np.right_shift(i1_all[valid], shift)
        active_i2 = np.right_shift(i2_all[valid], shift)

        keys = np.column_stack((active_levels, active_i0, active_i1, active_i2))
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

        frontier_cell_ids = np.full(unique_keys.shape[0], -1, dtype=np.int64)
        leaf_to_frontier_cell_id = np.full(levels_all.shape[0], -1, dtype=np.int64)
        for row, frontier_cell_id in enumerate(inverse):
            cid = int(frontier_cell_id)
            if frontier_cell_ids[cid] < 0:
                frontier_cell_ids[cid] = int(cell_ids[row])
            leaf_to_frontier_cell_id[int(cell_ids[row])] = cid

        levels = unique_keys[:, 0]
        i0 = unique_keys[:, 1]
        i1 = unique_keys[:, 2]
        i2 = unique_keys[:, 3]
        return levels, i0, i1, i2, frontier_cell_ids, leaf_to_frontier_cell_id

    def face_neighbors(self, *, max_level: int | None = None) -> "OctreeFaceNeighbors":
        """Return the lazily built face-neighbor graph for one level cutoff."""
        if self.cell_levels is None:
            raise ValueError("Octree has no cell_levels; cannot build face neighbors.")
        valid_levels = self.cell_levels[self.cell_levels >= 0]
        if valid_levels.size == 0:
            raise ValueError("Octree contains no valid cell levels (all < 0).")

        target_max_level = int(np.max(valid_levels) if max_level is None else max_level)
        cache = getattr(self, "_face_neighbors_by_max_level", None)
        if cache is None:
            cache = {}
            self._face_neighbors_by_max_level = cache
        face_neighbors = cache.get(target_max_level)
        if face_neighbors is None:
            from .face_neighbors import _build_face_neighbors_from_frontier

            frontier_cells = self._frontier_cells(target_max_level)
            face_neighbors = _build_face_neighbors_from_frontier(
                root_shape=self.root_shape,
                tree_coord=self.tree_coord,
                target_max_level=target_max_level,
                frontier_cells=frontier_cells,
            )
            cache[int(face_neighbors.max_level)] = face_neighbors
        return face_neighbors

    def domain_bounds(self, *, coord: TreeCoord = "xyz") -> tuple[np.ndarray, np.ndarray]:
        """Return global `(lo, hi)` bounds for the bound tree in requested coord."""
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )

        backend = self._coord_support(str(self.tree_coord))
        if resolved_coord == "xyz":
            return backend._domain_bounds_xyz(self)
        return backend._domain_bounds_rpa(self)

    def lookup_point(
        self,
        point: np.ndarray,
        *,
        coord: TreeCoord,
    ) -> "LookupHit | None":
        """Find which cell contains one point, or return `None` if not found."""
        chosen = int(self.lookup_points(np.asarray(point, dtype=float).reshape(1, 3), coord=coord)[0])
        return self._hit_from_chosen(chosen)

    def contains_cell(
        self,
        cell_id: int,
        point: np.ndarray,
        *,
        coord: TreeCoord,
        tol: float = 1e-10,
    ) -> bool:
        """Return whether one point lies inside one cell."""
        q = np.array(point, dtype=float).reshape(3)
        backend = self._coord_support(str(self.tree_coord))
        if str(self.tree_coord) == "xyz":
            if str(coord) != "xyz":
                raise ValueError("Cartesian lookup supports only coord='xyz'.")
            return backend._contains_xyz_cell(
                self,
                int(cell_id),
                float(q[0]),
                float(q[1]),
                float(q[2]),
                tol=float(tol),
            )
        if str(coord) == "xyz":
            return backend._contains_xyz_cell(
                self,
                int(cell_id),
                float(q[0]),
                float(q[1]),
                float(q[2]),
                tol=float(tol),
            )
        return backend._contains_rpa_cell(
            self,
            int(cell_id),
            float(q[0]),
            float(q[1]),
            float(q[2]),
            tol=float(tol),
        )

@dataclass(frozen=True)
class LookupHit:
    """Resolved lookup metadata for one query point."""

    cell_id: int
    level: int
    i0: int
    i1: int
    i2: int
    path: GridPath

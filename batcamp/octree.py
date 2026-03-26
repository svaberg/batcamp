#!/usr/bin/env python3
"""Core octree data structures and shared lookup/ray utilities."""

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


class BoundGeometryState(NamedTuple):
    """Bound dataset arrays and packed coordinate geometry owned by one octree."""

    points: np.ndarray
    corners: np.ndarray
    coord_state: "LookupKernelState"


class LookupKernelState(NamedTuple):
    """Shared packed lookup arrays for compiled tree descent and containment."""

    cell_axis0_start: np.ndarray
    cell_axis0_width: np.ndarray
    cell_axis1_start: np.ndarray
    cell_axis1_width: np.ndarray
    cell_axis2_start: np.ndarray
    cell_axis2_width: np.ndarray
    cell_valid: np.ndarray
    domain_axis0_start: float
    domain_axis0_width: float
    domain_axis1_start: float
    domain_axis1_width: float
    domain_axis2_start: float
    domain_axis2_width: float
    axis2_period: float
    axis2_periodic: bool
    node_value: np.ndarray
    node_child: np.ndarray
    root_node_ids: np.ndarray
    node_parent: np.ndarray
    cell_node_id: np.ndarray
    node_axis0_start: np.ndarray
    node_axis0_width: np.ndarray
    node_axis1_start: np.ndarray
    node_axis1_width: np.ndarray
    node_axis2_start: np.ndarray
    node_axis2_width: np.ndarray


class SphericalInterpKernelState(NamedTuple):
    """Packed spherical interpolation arrays for compiled trilinear kernels."""

    point_values_2d: np.ndarray
    corners: np.ndarray
    bin_to_corner: np.ndarray
    cell_r0: np.ndarray
    cell_rden: np.ndarray
    cell_t0: np.ndarray
    cell_tden: np.ndarray
    cell_p_start: np.ndarray
    cell_p_width: np.ndarray
    cell_pden: np.ndarray
    cell_phi_full: np.ndarray
    cell_phi_tiny: np.ndarray


class CartesianInterpKernelState(NamedTuple):
    """Packed Cartesian interpolation arrays for compiled trilinear kernels."""

    point_values_2d: np.ndarray
    corners: np.ndarray
    bin_to_corner: np.ndarray
    cell_x0: np.ndarray
    cell_xden: np.ndarray
    cell_y0: np.ndarray
    cell_yden: np.ndarray
    cell_z0: np.ndarray
    cell_zden: np.ndarray


_TRILINEAR_TARGET_BITS = np.array(
    [[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)],
    dtype=np.int8,
)

def _coord_state_inputs(tree: "Octree") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the bound dataset arrays and exact leaf levels needed for coordinate state."""
    missing = [
        name
        for name in (
            "_i0",
            "_i1",
            "_i2",
            "_node_depth",
            "_node_i0",
            "_node_i1",
            "_node_i2",
            "_node_value",
            "_node_child",
            "_root_node_ids",
            "_node_parent",
            "_cell_node_id",
        )
        if not hasattr(tree, name)
    ]
    if missing:
        raise ValueError(f"Lookup requires exact tree state: missing {missing}.")
    corners = np.asarray(tree.ds.corners, dtype=np.int64)
    x, y, z = (np.asarray(tree.ds[name], dtype=np.float64) for name in XYZ_VARS)
    cell_levels = tree.cell_levels
    if cell_levels is None or int(cell_levels.shape[0]) != int(corners.shape[0]):
        raise ValueError("Lookup requires exact cell_levels.")
    return corners, x, y, z, cell_levels


def _pack_coord_state(
    tree: "Octree",
    *,
    cell_axis0_start: np.ndarray,
    cell_axis0_width: np.ndarray,
    cell_axis1_start: np.ndarray,
    cell_axis1_width: np.ndarray,
    cell_axis2_start: np.ndarray,
    cell_axis2_width: np.ndarray,
    cell_valid: np.ndarray,
    domain_axis0_start: float,
    domain_axis0_width: float,
    domain_axis1_start: float,
    domain_axis1_width: float,
    domain_axis2_start: float,
    domain_axis2_width: float,
    axis2_period: float,
    axis2_periodic: bool,
    node_axis0_start: np.ndarray,
    node_axis0_width: np.ndarray,
    node_axis1_start: np.ndarray,
    node_axis1_width: np.ndarray,
    node_axis2_start: np.ndarray,
    node_axis2_width: np.ndarray,
) -> LookupKernelState:
    """Pack one shared lookup state from topology plus coordinate-specific geometry."""
    return LookupKernelState(
        cell_axis0_start=cell_axis0_start,
        cell_axis0_width=cell_axis0_width,
        cell_axis1_start=cell_axis1_start,
        cell_axis1_width=cell_axis1_width,
        cell_axis2_start=cell_axis2_start,
        cell_axis2_width=cell_axis2_width,
        cell_valid=cell_valid,
        domain_axis0_start=domain_axis0_start,
        domain_axis0_width=domain_axis0_width,
        domain_axis1_start=domain_axis1_start,
        domain_axis1_width=domain_axis1_width,
        domain_axis2_start=domain_axis2_start,
        domain_axis2_width=domain_axis2_width,
        axis2_period=axis2_period,
        axis2_periodic=axis2_periodic,
        node_value=tree._node_value,
        node_child=tree._node_child,
        root_node_ids=tree._root_node_ids,
        node_parent=tree._node_parent,
        cell_node_id=tree._cell_node_id,
        node_axis0_start=node_axis0_start,
        node_axis0_width=node_axis0_width,
        node_axis1_start=node_axis1_start,
        node_axis1_width=node_axis1_width,
        node_axis2_start=node_axis2_start,
        node_axis2_width=node_axis2_width,
    )


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
    axis0_den = np.maximum(axis0_width, tiny)
    axis1_den = np.maximum(axis1_width, tiny)
    axis2_den = np.maximum(axis2_width, tiny)

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
    return bin_to_corner, axis0_den, axis1_den, axis2_den, axis2_full, axis2_tiny


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
def _contains_lookup_cell(
    cid: int,
    q0: float,
    q1: float,
    q2: float,
    lookup_state: LookupKernelState,
    tol: float,
) -> bool:
    """Return whether one query lies inside one leaf cell geometry."""
    if not lookup_state.cell_valid[cid]:
        return False
    if not _contains_lookup_interval(
        q0,
        float(lookup_state.cell_axis0_start[cid]),
        float(lookup_state.cell_axis0_width[cid]),
        periodic=False,
        period=0.0,
        tol=tol,
    ):
        return False
    if not _contains_lookup_interval(
        q1,
        float(lookup_state.cell_axis1_start[cid]),
        float(lookup_state.cell_axis1_width[cid]),
        periodic=False,
        period=0.0,
        tol=tol,
    ):
        return False
    return _contains_lookup_interval(
        q2,
        float(lookup_state.cell_axis2_start[cid]),
        float(lookup_state.cell_axis2_width[cid]),
        periodic=bool(lookup_state.axis2_periodic),
        period=float(lookup_state.axis2_period),
        tol=tol,
    )


@njit(cache=True)
def _contains_lookup_node(
    node_id: int,
    q0: float,
    q1: float,
    q2: float,
    lookup_state: LookupKernelState,
    tol: float,
) -> bool:
    """Return whether one query lies inside one occupied node geometry."""
    nid = int(node_id)
    if not _contains_lookup_interval(
        q0,
        float(lookup_state.node_axis0_start[nid]),
        float(lookup_state.node_axis0_width[nid]),
        periodic=False,
        period=0.0,
        tol=tol,
    ):
        return False
    if not _contains_lookup_interval(
        q1,
        float(lookup_state.node_axis1_start[nid]),
        float(lookup_state.node_axis1_width[nid]),
        periodic=False,
        period=0.0,
        tol=tol,
    ):
        return False
    return _contains_lookup_interval(
        q2,
        float(lookup_state.node_axis2_start[nid]),
        float(lookup_state.node_axis2_width[nid]),
        periodic=bool(lookup_state.axis2_periodic),
        period=float(lookup_state.axis2_period),
        tol=tol,
    )


@njit(cache=True)
def _contains_lookup_domain(
    q0: float,
    q1: float,
    q2: float,
    lookup_state: LookupKernelState,
    tol: float = 0.0,
) -> bool:
    """Return whether one query lies inside the global lookup domain."""
    if not _contains_lookup_interval(
        q0,
        float(lookup_state.domain_axis0_start),
        float(lookup_state.domain_axis0_width),
        periodic=False,
        period=0.0,
        tol=tol,
    ):
        return False
    if not _contains_lookup_interval(
        q1,
        float(lookup_state.domain_axis1_start),
        float(lookup_state.domain_axis1_width),
        periodic=False,
        period=0.0,
        tol=tol,
    ):
        return False
    return _contains_lookup_interval(
        q2,
        float(lookup_state.domain_axis2_start),
        float(lookup_state.domain_axis2_width),
        periodic=bool(lookup_state.axis2_periodic),
        period=float(lookup_state.axis2_period),
        tol=tol,
    )


@njit(cache=True)
def _lookup_hint_node(
    prev_cid: int,
    q0: float,
    q1: float,
    q2: float,
    lookup_state: LookupKernelState,
    tol: float,
) -> int:
    """Return the nearest ancestor hint node containing one query, or `-1`."""
    if prev_cid < 0:
        return -1
    current = int(lookup_state.cell_node_id[int(prev_cid)])
    while current >= 0:
        if _contains_lookup_node(current, q0, q1, q2, lookup_state, tol):
            return current
        current = int(lookup_state.node_parent[current])
    return -1


@njit(cache=True)
def _lookup_descend_to_leaf(
    q0: float,
    q1: float,
    q2: float,
    start_node_id: int,
    lookup_state: LookupKernelState,
    tol: float,
) -> int:
    """Descend one sparse tree from a containing node hint, or from the roots."""
    current = int(start_node_id)
    if current < 0:
        for root_pos in range(int(lookup_state.root_node_ids.shape[0])):
            node_id = int(lookup_state.root_node_ids[root_pos])
            if _contains_lookup_node(node_id, q0, q1, q2, lookup_state, tol):
                current = node_id
                break
    if current < 0:
        return -1

    while True:
        value = int(lookup_state.node_value[current])
        if value >= 0:
            cid = int(value)
            if _contains_lookup_cell(cid, q0, q1, q2, lookup_state, tol):
                return cid
            return -1

        found_child = False
        for child_ord in range(8):
            child_id = int(lookup_state.node_child[current, child_ord])
            if child_id < 0:
                continue
            if _contains_lookup_node(child_id, q0, q1, q2, lookup_state, tol):
                current = child_id
                found_child = True
                break
        if not found_child:
            return -1


@njit(cache=True)
def _lookup_cell_id_kernel(
    q0: float,
    q1: float,
    q2: float,
    lookup_state: LookupKernelState,
    prev_cid: int = -1,
    tol: float = 1.0e-10,
) -> int:
    """Resolve one query to a cell id by sparse-tree descent."""
    if not (np.isfinite(q0) and np.isfinite(q1) and np.isfinite(q2)):
        return -1
    if prev_cid >= 0 and _contains_lookup_cell(int(prev_cid), q0, q1, q2, lookup_state, tol):
        return int(prev_cid)
    if not _contains_lookup_domain(q0, q1, q2, lookup_state):
        return -1
    current = _lookup_hint_node(
        int(prev_cid),
        q0,
        q1,
        q2,
        lookup_state,
        tol,
    )
    return _lookup_descend_to_leaf(
        q0,
        q1,
        q2,
        current,
        lookup_state,
        tol,
    )


@njit(cache=True, parallel=True)
def _lookup_cell_ids_kernel(queries: np.ndarray, lookup_state: LookupKernelState) -> np.ndarray:
    """Resolve a batch of same-coordinate queries to cell ids."""
    n_query = int(queries.shape[0])
    cell_ids = np.full(n_query, -1, dtype=np.int64)
    chunk_size = 1024
    n_chunks = (n_query + chunk_size - 1) // chunk_size
    for chunk_id in prange(n_chunks):
        start = chunk_id * chunk_size
        end = min(n_query, start + chunk_size)
        hint_cid = -1
        for i in range(start, end):
            cid = _lookup_cell_id_kernel(
                queries[i, 0],
                queries[i, 1],
                queries[i, 2],
                lookup_state,
                hint_cid,
            )
            cell_ids[i] = cid
            hint_cid = int(cid) if cid >= 0 else -1
    return cell_ids


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


def _build_node_arrays(
    depths: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    leaf_value: np.ndarray,
    *,
    tree_depth: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sorted occupied-node arrays from exact leaf addresses."""
    leaf_depth = np.asarray(depths, dtype=np.int64)
    leaf_i0 = np.asarray(i0, dtype=np.int64)
    leaf_i1 = np.asarray(i1, dtype=np.int64)
    leaf_i2 = np.asarray(i2, dtype=np.int64)
    leaf_value_arr = np.asarray(leaf_value, dtype=np.int64)

    leaf_order = np.lexsort((leaf_i2, leaf_i1, leaf_i0, leaf_depth))
    leaf_depth = leaf_depth[leaf_order]
    leaf_i0 = leaf_i0[leaf_order]
    leaf_i1 = leaf_i1[leaf_order]
    leaf_i2 = leaf_i2[leaf_order]
    leaf_value_arr = leaf_value_arr[leaf_order]

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

    node_depth_parts = [leaf_depth]
    node_i0_parts = [leaf_i0]
    node_i1_parts = [leaf_i1]
    node_i2_parts = [leaf_i2]
    node_value_parts = [leaf_value_arr]
    for parent_depth in range(int(tree_depth)):
        mask = depths > int(parent_depth)
        if not np.any(mask):
            continue
        up = np.asarray(depths[mask] - int(parent_depth), dtype=np.int64)
        parent_nodes = np.column_stack(
            (
                np.full(int(np.count_nonzero(mask)), int(parent_depth), dtype=np.int64),
                np.right_shift(i0[mask], up),
                np.right_shift(i1[mask], up),
                np.right_shift(i2[mask], up),
            )
        )
        parent_nodes = np.unique(parent_nodes, axis=0)
        node_depth_parts.append(parent_nodes[:, 0].astype(np.int64, copy=False))
        node_i0_parts.append(parent_nodes[:, 1].astype(np.int64, copy=False))
        node_i1_parts.append(parent_nodes[:, 2].astype(np.int64, copy=False))
        node_i2_parts.append(parent_nodes[:, 3].astype(np.int64, copy=False))
        node_value_parts.append(np.full(parent_nodes.shape[0], -2, dtype=np.int64))

    node_depth = np.concatenate(node_depth_parts)
    node_i0 = np.concatenate(node_i0_parts)
    node_i1 = np.concatenate(node_i1_parts)
    node_i2 = np.concatenate(node_i2_parts)
    node_value = np.concatenate(node_value_parts)
    node_order = np.lexsort((node_i2, node_i1, node_i0, node_depth))
    node_depth = node_depth[node_order]
    node_i0 = node_i0[node_order]
    node_i1 = node_i1[node_order]
    node_i2 = node_i2[node_order]
    node_value = node_value[node_order]

    same_node = (
        (node_depth[1:] == node_depth[:-1])
        & (node_i0[1:] == node_i0[:-1])
        & (node_i1[1:] == node_i1[:-1])
        & (node_i2[1:] == node_i2[:-1])
    )
    if np.any(same_node):
        dup = int(np.flatnonzero(same_node)[0])
        raise ValueError(
            f"{label} cells overlap across parent/child addresses at "
            f"({int(node_depth[dup])}, {int(node_i0[dup])}, {int(node_i1[dup])}, {int(node_i2[dup])})."
        )

    return node_depth, node_i0, node_i1, node_i2, node_value


def _build_child_table(
    node_depth: np.ndarray,
    node_i0: np.ndarray,
    node_i1: np.ndarray,
    node_i2: np.ndarray,
    node_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse 8-child references for occupied nodes."""
    n_nodes = int(np.asarray(node_depth, dtype=np.int64).shape[0])
    node_child = np.full((n_nodes, 8), -1, dtype=np.int64)
    node_parent = np.full(n_nodes, -1, dtype=np.int64)
    key_to_node = {
        (int(node_depth[idx]), int(node_i0[idx]), int(node_i1[idx]), int(node_i2[idx])): int(idx)
        for idx in range(n_nodes)
    }
    for idx in range(n_nodes):
        if int(node_value[idx]) >= 0:
            continue
        depth = int(node_depth[idx])
        i0 = int(node_i0[idx])
        i1 = int(node_i1[idx])
        i2 = int(node_i2[idx])
        for child_ord in range(8):
            b0 = (child_ord >> 2) & 1
            b1 = (child_ord >> 1) & 1
            b2 = child_ord & 1
            child_key = (depth + 1, 2 * i0 + b0, 2 * i1 + b1, 2 * i2 + b2)
            child_idx = key_to_node.get(child_key)
            if child_idx is not None:
                node_child[idx, child_ord] = int(child_idx)
                node_parent[int(child_idx)] = int(idx)
    root_node_ids = np.flatnonzero(np.asarray(node_depth, dtype=np.int64) == 0).astype(np.int64)
    return node_child, root_node_ids, node_parent


def _node_state_from_leaves(
    cell_levels: np.ndarray,
    cell_i0: np.ndarray,
    cell_i1: np.ndarray,
    cell_i2: np.ndarray,
    *,
    max_level: int,
) -> tuple[np.ndarray, ...]:
    """Rebuild exact occupied nonleaves and leaf-node maps from leaf addresses."""

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
    node_depth, node_i0, node_i1, node_i2, node_value = _build_node_arrays(
        depths,
        leaf_i0,
        leaf_i1,
        leaf_i2,
        valid_ids,
        tree_depth=int(max_level),
        label="Restored",
    )
    node_child, root_node_ids, node_parent = _build_child_table(
        node_depth,
        node_i0,
        node_i1,
        node_i2,
        node_value,
    )
    cell_node_id = np.full(levels.shape[0], -1, dtype=np.int64)
    leaf_mask = node_value >= 0
    cell_node_id[node_value[leaf_mask]] = np.flatnonzero(leaf_mask).astype(np.int64)
    return node_depth, node_i0, node_i1, node_i2, node_value, node_child, root_node_ids, node_parent, cell_node_id


class Octree:
    """Adaptive octree summary plus bound lookup/ray-query entrypoints.

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
        resolved_tree_coord = str(tree_coord)
        if resolved_tree_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported tree_coord '{resolved_tree_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        self.root_shape = tuple(int(v) for v in root_shape)
        self.tree_coord = resolved_tree_coord
        self.cell_levels = np.asarray(cell_levels, dtype=np.int64)

        self.leaf_shape, self.is_full, self.level_counts, self.min_level, self.max_level = _level_metadata_from_leaves(
            self.root_shape,
            self.cell_levels,
        )
        self._i0 = np.asarray(cell_i0, dtype=np.int64)
        self._i1 = np.asarray(cell_i1, dtype=np.int64)
        self._i2 = np.asarray(cell_i2, dtype=np.int64)
        (
            self._node_depth,
            self._node_i0,
            self._node_i1,
            self._node_i2,
            self._node_value,
            self._node_child,
            self._root_node_ids,
            self._node_parent,
            self._cell_node_id,
        ) = _node_state_from_leaves(
            self.cell_levels,
            self._i0,
            self._i1,
            self._i2,
            max_level=self.max_level,
        )
        self._bind(ds)

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

    def _bind(self, ds: Dataset) -> None:
        """Attach a dataset to this tree so lookup and ray methods can run."""
        if ds.corners is None:
            raise ValueError("Dataset has no corners; cannot bind octree lookup.")
        if not set(XYZ_VARS).issubset(set(ds.variables)):
            raise ValueError("Dataset must provide X/Y/Z variables to bind octree lookup.")
        self.ds = ds
        self._coord_support(str(self.tree_coord))._attach_coord_state(self)
        self._prepare_interpolation_geometry()

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

    def lookup_geometry(self) -> BoundGeometryState:
        """Return bound dataset arrays plus packed coordinate geometry."""
        return BoundGeometryState(
            points=np.column_stack(
                (
                    self.ds[XYZ_VARS[0]],
                    self.ds[XYZ_VARS[1]],
                    self.ds[XYZ_VARS[2]],
                )
            ),
            corners=self.ds.corners,
            coord_state=self._coord_state,
        )

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

        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        if str(self.tree_coord) == "xyz":
            if resolved_coord != "xyz":
                raise ValueError("Cartesian lookup supports only coord='xyz'.")
            cell_ids = _lookup_cell_ids_kernel(q, self._coord_state)
        elif resolved_coord == "xyz":
            from .spherical import _lookup_xyz_cell_ids_for_rpa_tree_kernel

            cell_ids = _lookup_xyz_cell_ids_for_rpa_tree_kernel(q, self._coord_state)
        else:
            cell_ids = _lookup_cell_ids_kernel(q, self._coord_state)
        return cell_ids.reshape(shape)

    def _query_points_in_tree_coords(self, points: np.ndarray, *, coord: TreeCoord) -> np.ndarray:
        """Return one query batch expressed in this tree's own coordinate system."""
        q = np.array(points, dtype=np.float64, order="C")
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")
        resolved_coord = str(coord)
        if resolved_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported lookup coord '{resolved_coord}'; expected one of {SUPPORTED_TREE_COORDS}."
            )
        if str(self.tree_coord) == "xyz":
            if resolved_coord != "xyz":
                raise ValueError("Cartesian lookup supports only coord='xyz'.")
            return q
        if resolved_coord == "rpa":
            return q
        from .spherical import _xyz_array_to_rpa

        return _xyz_array_to_rpa(q)

    def _prepare_interpolation_geometry(self) -> None:
        """Build the per-cell trilinear corner map from the bound dataset."""
        corners = np.asarray(self.ds.corners, dtype=np.int64)
        cell_axis0_start = self._coord_state.cell_axis0_start
        cell_axis0_width = self._coord_state.cell_axis0_width
        cell_axis1_start = self._coord_state.cell_axis1_start
        cell_axis1_width = self._coord_state.cell_axis1_width
        cell_axis2_start = self._coord_state.cell_axis2_start
        cell_axis2_width = self._coord_state.cell_axis2_width
        x = np.asarray(self.ds[XYZ_VARS[0]], dtype=np.float64)
        y = np.asarray(self.ds[XYZ_VARS[1]], dtype=np.float64)
        z = np.asarray(self.ds[XYZ_VARS[2]], dtype=np.float64)
        if str(self.tree_coord) == "xyz":
            axis0 = x[corners]
            axis1 = y[corners]
            axis2 = z[corners]
            axis2_periodic = False
        else:
            from .spherical import _xyz_arrays_to_rpa

            point_r, point_theta, point_phi = _xyz_arrays_to_rpa(x, y, z)
            axis0 = point_r[corners]
            axis1 = point_theta[corners]
            axis2 = point_phi[corners]
            axis2_periodic = True
        (
            self._interp_bin_to_corner,
            self._interp_axis0_den,
            self._interp_axis1_den,
            self._interp_axis2_den,
            self._interp_axis2_full,
            self._interp_axis2_tiny,
        ) = _build_interp_bin_to_corner(
            axis0,
            axis1,
            axis2,
            axis0_start=cell_axis0_start,
            axis0_width=cell_axis0_width,
            axis1_start=cell_axis1_start,
            axis1_width=cell_axis1_width,
            axis2_start=cell_axis2_start,
            axis2_width=cell_axis2_width,
            axis2_periodic=axis2_periodic,
        )

    def _interp_state_from_values(
        self,
        point_values_2d: np.ndarray,
    ) -> CartesianInterpKernelState | SphericalInterpKernelState:
        """Pack one interpolation state for the given per-point values."""
        corners = np.asarray(self.ds.corners, dtype=np.int64)
        if str(self.tree_coord) == "xyz":
            return CartesianInterpKernelState(
                point_values_2d=point_values_2d,
                corners=corners,
                bin_to_corner=self._interp_bin_to_corner,
                cell_x0=self._coord_state.cell_axis0_start,
                cell_xden=self._interp_axis0_den,
                cell_y0=self._coord_state.cell_axis1_start,
                cell_yden=self._interp_axis1_den,
                cell_z0=self._coord_state.cell_axis2_start,
                cell_zden=self._interp_axis2_den,
            )
        return SphericalInterpKernelState(
            point_values_2d=point_values_2d,
            corners=corners,
            bin_to_corner=self._interp_bin_to_corner,
            cell_r0=self._coord_state.cell_axis0_start,
            cell_rden=self._interp_axis0_den,
            cell_t0=self._coord_state.cell_axis1_start,
            cell_tden=self._interp_axis1_den,
            cell_p_start=self._coord_state.cell_axis2_start,
            cell_p_width=self._coord_state.cell_axis2_width,
            cell_pden=self._interp_axis2_den,
            cell_phi_full=self._interp_axis2_full,
            cell_phi_tiny=self._interp_axis2_tiny,
        )

    def _frontier_nodes(self, max_level: int) -> tuple[np.ndarray, ...]:
        """Return unique frontier nodes by truncating leaves to one level cutoff."""
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

        node_cell_ids = np.full(unique_keys.shape[0], -1, dtype=np.int64)
        cell_to_node_id = np.full(levels_all.shape[0], -1, dtype=np.int64)
        for row, node_id in enumerate(inverse):
            nid = int(node_id)
            if node_cell_ids[nid] < 0:
                node_cell_ids[nid] = int(cell_ids[row])
            cell_to_node_id[int(cell_ids[row])] = nid

        levels = unique_keys[:, 0]
        i0 = unique_keys[:, 1]
        i1 = unique_keys[:, 2]
        i2 = unique_keys[:, 3]
        return levels, i0, i1, i2, node_cell_ids, cell_to_node_id

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

            frontier_nodes = self._frontier_nodes(target_max_level)
            face_neighbors = _build_face_neighbors_from_frontier(
                root_shape=self.root_shape,
                tree_coord=self.tree_coord,
                target_max_level=target_max_level,
                frontier_nodes=frontier_nodes,
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

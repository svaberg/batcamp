#!/usr/bin/env python3
"""Topological face-neighbor construction for octree leaf/frontier cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from numba import njit
import numpy as np

from .octree import Octree

FACE_COUNT = 6
_NEG_POS = np.array([-1, 1], dtype=np.int64)


class TopologicalKernelState(NamedTuple):
    """Arrays consumed directly by numba traversal kernels."""

    face_offsets: np.ndarray
    face_neighbors: np.ndarray
    node_cell_ids: np.ndarray
    cell_to_node_id: np.ndarray


@dataclass(frozen=True)
class TopologicalNeighborhood:
    """Face-neighbor graph for one octree at one level cutoff."""

    levels: np.ndarray
    i0: np.ndarray
    i1: np.ndarray
    i2: np.ndarray
    face_counts: np.ndarray
    face_offsets: np.ndarray
    face_neighbors: np.ndarray
    node_cell_ids: np.ndarray
    cell_to_node_id: np.ndarray
    min_level: int
    max_level: int
    periodic_i2: bool

    @property
    def node_count(self) -> int:
        """Return number of frontier nodes in this graph."""
        return int(self.levels.shape[0])

    def face_neighbor_ids(self, node_id: int, face: int) -> np.ndarray:
        """Return neighbor node ids for one `(node_id, face)` pair."""
        nid = int(node_id)
        f = int(face)
        if nid < 0 or nid >= self.node_count:
            raise ValueError(f"Invalid node_id {nid}; expected [0, {self.node_count - 1}].")
        if f < 0 or f >= FACE_COUNT:
            raise ValueError(f"Invalid face {f}; expected [0, {FACE_COUNT - 1}].")
        slot = nid * FACE_COUNT + f
        start = int(self.face_offsets[slot])
        end = int(self.face_offsets[slot + 1])
        return self.face_neighbors[start:end]

    @property
    def kernel_state(self) -> TopologicalKernelState:
        """Return compact topology arrays for numba kernels."""
        return TopologicalKernelState(
            face_offsets=np.asarray(self.face_offsets, dtype=np.int64),
            face_neighbors=np.asarray(self.face_neighbors, dtype=np.int64),
            node_cell_ids=np.asarray(self.node_cell_ids, dtype=np.int64),
            cell_to_node_id=np.asarray(self.cell_to_node_id, dtype=np.int64),
        )


@njit(cache=True)
def _find_node_id(
    levels: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    q_level: int,
    q_i0: int,
    q_i1: int,
    q_i2: int,
) -> int:
    """Binary-search one frontier node by `(level, i0, i1, i2)` key."""
    lo = 0
    hi = int(levels.shape[0]) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        m_level = int(levels[mid])
        if q_level < m_level:
            hi = mid - 1
            continue
        if q_level > m_level:
            lo = mid + 1
            continue

        m_i0 = int(i0[mid])
        if q_i0 < m_i0:
            hi = mid - 1
            continue
        if q_i0 > m_i0:
            lo = mid + 1
            continue

        m_i1 = int(i1[mid])
        if q_i1 < m_i1:
            hi = mid - 1
            continue
        if q_i1 > m_i1:
            lo = mid + 1
            continue

        m_i2 = int(i2[mid])
        if q_i2 < m_i2:
            hi = mid - 1
            continue
        if q_i2 > m_i2:
            lo = mid + 1
            continue

        return int(mid)
    return -1


@njit(cache=True)
def _normalize_i2(i2_value: int, n2: int, periodic_i2: bool) -> tuple[int, bool]:
    """Normalize axis-2 index according to periodic flag and bounds."""
    if periodic_i2:
        if n2 <= 0:
            return 0, False
        wrapped = i2_value % n2
        if wrapped < 0:
            wrapped += n2
        return wrapped, True
    if i2_value < 0 or i2_value >= n2:
        return 0, False
    return i2_value, True


@njit(cache=True)
def _emit_face_neighbors(
    node_id: int,
    face: int,
    levels: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    min_level: int,
    max_level: int,
    level_shapes: np.ndarray,
    periodic_i2: bool,
    neighbor_out: np.ndarray,
    out_offset: int,
    do_write: bool,
) -> int:
    """Enumerate neighbors for one node face; count or write based on `do_write`."""
    level_node = int(levels[node_id])
    i0_node = int(i0[node_id])
    i1_node = int(i1[node_id])
    i2_node = int(i2[node_id])

    axis = face // 2
    sign = int(_NEG_POS[face % 2])
    count = 0

    for level_cand in range(int(min_level), int(max_level) + 1):
        shape_idx = int(level_cand - min_level)
        n0 = int(level_shapes[shape_idx, 0])
        n1 = int(level_shapes[shape_idx, 1])
        n2 = int(level_shapes[shape_idx, 2])

        if level_cand <= level_node:
            scale = 1 << int(level_node - level_cand)
            c0 = 0
            c1 = 0
            c2 = 0
            valid = True

            if axis == 0:
                if sign < 0:
                    if (i0_node % scale) != 0:
                        valid = False
                    c0 = i0_node // scale - 1
                else:
                    if ((i0_node + 1) % scale) != 0:
                        valid = False
                    c0 = (i0_node + 1) // scale
                c1 = i1_node // scale
                c2 = i2_node // scale
            elif axis == 1:
                if sign < 0:
                    if (i1_node % scale) != 0:
                        valid = False
                    c1 = i1_node // scale - 1
                else:
                    if ((i1_node + 1) % scale) != 0:
                        valid = False
                    c1 = (i1_node + 1) // scale
                c0 = i0_node // scale
                c2 = i2_node // scale
            else:
                if sign < 0:
                    if (i2_node % scale) != 0:
                        valid = False
                    c2 = i2_node // scale - 1
                else:
                    if ((i2_node + 1) % scale) != 0:
                        valid = False
                    c2 = (i2_node + 1) // scale
                c0 = i0_node // scale
                c1 = i1_node // scale

            if not valid:
                continue
            if c0 < 0 or c0 >= n0 or c1 < 0 or c1 >= n1:
                continue
            c2_norm, ok_i2 = _normalize_i2(c2, n2, periodic_i2)
            if not ok_i2:
                continue
            nbr = _find_node_id(levels, i0, i1, i2, level_cand, c0, c1, c2_norm)
            if nbr >= 0 and nbr != node_id:
                if do_write:
                    neighbor_out[out_offset + count] = nbr
                count += 1
            continue

        scale = 1 << int(level_cand - level_node)
        if axis == 0:
            if sign < 0:
                c0 = i0_node * scale - 1
            else:
                c0 = (i0_node + 1) * scale
            if c0 < 0 or c0 >= n0:
                continue
            base1 = i1_node * scale
            base2 = i2_node * scale
            for o1 in range(scale):
                c1 = base1 + o1
                if c1 < 0 or c1 >= n1:
                    continue
                for o2 in range(scale):
                    c2 = base2 + o2
                    c2_norm, ok_i2 = _normalize_i2(c2, n2, periodic_i2)
                    if not ok_i2:
                        continue
                    nbr = _find_node_id(levels, i0, i1, i2, level_cand, c0, c1, c2_norm)
                    if nbr >= 0 and nbr != node_id:
                        if do_write:
                            neighbor_out[out_offset + count] = nbr
                        count += 1
        elif axis == 1:
            if sign < 0:
                c1 = i1_node * scale - 1
            else:
                c1 = (i1_node + 1) * scale
            if c1 < 0 or c1 >= n1:
                continue
            base0 = i0_node * scale
            base2 = i2_node * scale
            for o0 in range(scale):
                c0 = base0 + o0
                if c0 < 0 or c0 >= n0:
                    continue
                for o2 in range(scale):
                    c2 = base2 + o2
                    c2_norm, ok_i2 = _normalize_i2(c2, n2, periodic_i2)
                    if not ok_i2:
                        continue
                    nbr = _find_node_id(levels, i0, i1, i2, level_cand, c0, c1, c2_norm)
                    if nbr >= 0 and nbr != node_id:
                        if do_write:
                            neighbor_out[out_offset + count] = nbr
                        count += 1
        else:
            if sign < 0:
                c2 = i2_node * scale - 1
            else:
                c2 = (i2_node + 1) * scale
            c2_norm, ok_i2 = _normalize_i2(c2, n2, periodic_i2)
            if not ok_i2:
                continue
            base0 = i0_node * scale
            base1 = i1_node * scale
            for o0 in range(scale):
                c0 = base0 + o0
                if c0 < 0 or c0 >= n0:
                    continue
                for o1 in range(scale):
                    c1 = base1 + o1
                    if c1 < 0 or c1 >= n1:
                        continue
                    nbr = _find_node_id(levels, i0, i1, i2, level_cand, c0, c1, c2_norm)
                    if nbr >= 0 and nbr != node_id:
                        if do_write:
                            neighbor_out[out_offset + count] = nbr
                        count += 1
    return count


@njit(cache=True)
def build_topological_neighborhood_kernel(
    levels: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    min_level: int,
    max_level: int,
    level_shapes: np.ndarray,
    periodic_i2: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build face-neighbor adjacency for sorted frontier nodes."""
    n_nodes = int(levels.shape[0])
    empty = np.empty(0, dtype=np.int64)
    face_counts = np.zeros((n_nodes, FACE_COUNT), dtype=np.int64)
    for node_id in range(n_nodes):
        for face in range(FACE_COUNT):
            face_counts[node_id, face] = _emit_face_neighbors(
                node_id,
                face,
                levels,
                i0,
                i1,
                i2,
                min_level,
                max_level,
                level_shapes,
                periodic_i2,
                empty,
                0,
                False,
            )

    face_offsets = np.zeros(n_nodes * FACE_COUNT + 1, dtype=np.int64)
    running = 0
    for node_id in range(n_nodes):
        for face in range(FACE_COUNT):
            slot = node_id * FACE_COUNT + face
            face_offsets[slot] = running
            running += int(face_counts[node_id, face])
    face_offsets[n_nodes * FACE_COUNT] = running

    face_neighbors = np.empty(running, dtype=np.int64)
    for node_id in range(n_nodes):
        for face in range(FACE_COUNT):
            slot = node_id * FACE_COUNT + face
            start = int(face_offsets[slot])
            _emit_face_neighbors(
                node_id,
                face,
                levels,
                i0,
                i1,
                i2,
                min_level,
                max_level,
                level_shapes,
                periodic_i2,
                face_neighbors,
                start,
                True,
            )
    return face_counts, face_offsets, face_neighbors


def _level_shapes_for_cutoff(tree: Octree, min_level: int, max_level: int) -> np.ndarray:
    """Return `(n0, n1, n2)` for every level in `[min_level, max_level]`."""
    if max_level < min_level:
        raise ValueError(f"Invalid level bounds: min_level={min_level}, max_level={max_level}.")
    out = np.empty((max_level - min_level + 1, 3), dtype=np.int64)
    root0 = int(tree.root_shape[0])
    root1 = int(tree.root_shape[1])
    root2 = int(tree.root_shape[2])
    for level in range(min_level, max_level + 1):
        if int(level) < 0:
            raise ValueError(f"Derived negative level={level}; max_level={tree.max_level}.")
        scale = 1 << int(level)
        row = level - min_level
        out[row, 0] = root0 * scale
        out[row, 1] = root1 * scale
        out[row, 2] = root2 * scale
    return out


def _cell_local_indices_from_bounds(tree: Octree, cell_ids: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, ...]:
    """Derive exact per-level `(i0, i1, i2)` starts from cell bounds."""
    tree._require_lookup()
    lookup_state = tree.lookup_state
    n = int(cell_ids.shape[0])
    i0 = np.full(n, -1, dtype=np.int64)
    i1 = np.full(n, -1, dtype=np.int64)
    i2 = np.full(n, -1, dtype=np.int64)

    if str(tree.tree_coord) == "xyz":
        xyz_min = np.asarray(lookup_state.xyz_min, dtype=float)
        xyz_max = np.asarray(lookup_state.xyz_max, dtype=float)
        domain_span = np.maximum(xyz_max - xyz_min, np.finfo(float).tiny)
        cell_min0 = np.asarray(lookup_state.cell_x_min, dtype=float)[cell_ids]
        cell_min1 = np.asarray(lookup_state.cell_y_min, dtype=float)[cell_ids]
        cell_min2 = np.asarray(lookup_state.cell_z_min, dtype=float)[cell_ids]
        offset0 = cell_min0 - float(xyz_min[0])
        offset1 = cell_min1 - float(xyz_min[1])
        offset2 = cell_min2 - float(xyz_min[2])
    else:
        raise ValueError("Exact bound-derived topology indices are only available for Cartesian trees.")

    for level in sorted(set(int(v) for v in levels.tolist())):
        mask = levels == int(level)
        if not np.any(mask):
            continue
        if int(level) < 0:
            raise ValueError(f"Derived negative level={level}; max_level={tree.max_level}.")
        scale = 1 << int(level)
        n0 = int(tree.root_shape[0]) * scale
        n1 = int(tree.root_shape[1]) * scale
        n2 = int(tree.root_shape[2]) * scale
        d0 = float(domain_span[0] / max(n0, 1))
        d1 = float(domain_span[1] / max(n1, 1))
        d2 = float(domain_span[2] / max(n2, 1))
        i0[mask] = np.rint(offset0[mask] / d0).astype(np.int64)
        i1[mask] = np.rint(offset1[mask] / d1).astype(np.int64)
        i2[mask] = np.rint(offset2[mask] / d2).astype(np.int64)
    return i0, i1, i2


def _frontier_nodes_from_octree(tree: Octree, max_level: int) -> tuple[np.ndarray, ...]:
    """Build unique frontier nodes by truncating leaf levels to `max_level`."""
    tree._require_lookup()
    cell_levels = tree.cell_levels
    if cell_levels is None:
        raise ValueError("Octree has no cell_levels; cannot build topological neighborhood.")

    levels_all = np.asarray(cell_levels, dtype=np.int64)

    valid = levels_all >= 0
    if not np.any(valid):
        raise ValueError("Octree contains no valid cells (all levels are < 0).")
    cell_ids = np.flatnonzero(valid).astype(np.int64)
    if str(tree.tree_coord) == "xyz" and getattr(tree, "_lookup_state", None) is not None:
        i0_valid, i1_valid, i2_valid = _cell_local_indices_from_bounds(tree, cell_ids, levels_all[valid])
    else:
        if not hasattr(tree, "_i0") or not hasattr(tree, "_i1") or not hasattr(tree, "_i2"):
            raise ValueError("Lookup indices (_i0/_i1/_i2) are unavailable; build lookup before topology.")
        i0_all = np.asarray(getattr(tree, "_i0"), dtype=np.int64)
        i1_all = np.asarray(getattr(tree, "_i1"), dtype=np.int64)
        i2_all = np.asarray(getattr(tree, "_i2"), dtype=np.int64)
        if not (levels_all.shape == i0_all.shape == i1_all.shape == i2_all.shape):
            raise ValueError("Cell level/index arrays must have matching shapes.")
        i0_valid = i0_all[valid]
        i1_valid = i1_all[valid]
        i2_valid = i2_all[valid]

    levels_valid = levels_all[valid]
    active_levels = np.minimum(levels_valid, int(max_level))
    shift = levels_valid - active_levels
    active_i0 = np.right_shift(i0_valid, shift)
    active_i1 = np.right_shift(i1_valid, shift)
    active_i2 = np.right_shift(i2_valid, shift)

    keys = np.column_stack((active_levels, active_i0, active_i1, active_i2)).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    node_cell_ids = np.full(unique_keys.shape[0], -1, dtype=np.int64)
    cell_to_node_id = np.full(levels_all.shape[0], -1, dtype=np.int64)
    for row, node_id in enumerate(inverse):
        nid = int(node_id)
        if node_cell_ids[nid] < 0:
            node_cell_ids[nid] = int(cell_ids[row])
        cell_to_node_id[int(cell_ids[row])] = nid

    levels = np.asarray(unique_keys[:, 0], dtype=np.int64)
    i0 = np.asarray(unique_keys[:, 1], dtype=np.int64)
    i1 = np.asarray(unique_keys[:, 2], dtype=np.int64)
    i2 = np.asarray(unique_keys[:, 3], dtype=np.int64)
    return levels, i0, i1, i2, node_cell_ids, cell_to_node_id


def build_topological_neighborhood(tree: Octree, *, max_level: int | None = None) -> TopologicalNeighborhood:
    """Build topological face-neighbor graph from one octree."""
    if tree.cell_levels is None:
        raise ValueError("Octree has no cell_levels; cannot build topological neighborhood.")
    valid_levels = np.asarray(tree.cell_levels, dtype=np.int64)
    valid_levels = valid_levels[valid_levels >= 0]
    if valid_levels.size == 0:
        raise ValueError("Octree contains no valid cell levels (all < 0).")

    min_tree_level = int(np.min(valid_levels))
    max_tree_level = int(np.max(valid_levels))
    target_max_level = int(max_tree_level if max_level is None else max_level)
    if target_max_level < min_tree_level or target_max_level > max_tree_level:
        raise ValueError(
            f"max_level={target_max_level} is outside [{min_tree_level}, {max_tree_level}] for this tree."
        )

    levels, i0, i1, i2, node_cell_ids, cell_to_node_id = _frontier_nodes_from_octree(tree, target_max_level)
    min_level = int(np.min(levels))
    level_shapes = _level_shapes_for_cutoff(tree, min_level, target_max_level)
    periodic_i2 = str(tree.tree_coord) == "rpa"

    face_counts, face_offsets, face_neighbors = build_topological_neighborhood_kernel(
        levels,
        i0,
        i1,
        i2,
        min_level,
        target_max_level,
        level_shapes,
        periodic_i2,
    )
    return TopologicalNeighborhood(
        levels=levels,
        i0=i0,
        i1=i1,
        i2=i2,
        face_counts=face_counts,
        face_offsets=face_offsets,
        face_neighbors=face_neighbors,
        node_cell_ids=node_cell_ids,
        cell_to_node_id=cell_to_node_id,
        min_level=min_level,
        max_level=target_max_level,
        periodic_i2=periodic_i2,
    )

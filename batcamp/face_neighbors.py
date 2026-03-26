#!/usr/bin/env python3
"""Face-neighbor construction for octree leaf/frontier cells."""

from __future__ import annotations

from dataclasses import dataclass

from numba import njit
import numpy as np

from .octree import Octree
from .shared_types import GridShape
from .shared_types import TreeCoord

FACE_COUNT = 6
_NEG_POS = np.array([-1, 1], dtype=np.int64)


@dataclass(frozen=True)
class OctreeFaceNeighbors:
    """Face-neighbor graph for one octree at one level cutoff."""

    levels: np.ndarray
    cell_ijk: np.ndarray
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

@njit(cache=True)
def _find_node_id(
    levels: np.ndarray,
    cell_ijk: np.ndarray,
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

        m_i0 = int(cell_ijk[mid, 0])
        if q_i0 < m_i0:
            hi = mid - 1
            continue
        if q_i0 > m_i0:
            lo = mid + 1
            continue

        m_i1 = int(cell_ijk[mid, 1])
        if q_i1 < m_i1:
            hi = mid - 1
            continue
        if q_i1 > m_i1:
            lo = mid + 1
            continue

        m_i2 = int(cell_ijk[mid, 2])
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
    cell_ijk: np.ndarray,
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
    i0_node = int(cell_ijk[node_id, 0])
    i1_node = int(cell_ijk[node_id, 1])
    i2_node = int(cell_ijk[node_id, 2])

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
            nbr = _find_node_id(levels, cell_ijk, level_cand, c0, c1, c2_norm)
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
                    nbr = _find_node_id(levels, cell_ijk, level_cand, c0, c1, c2_norm)
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
                    nbr = _find_node_id(levels, cell_ijk, level_cand, c0, c1, c2_norm)
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
                    nbr = _find_node_id(levels, cell_ijk, level_cand, c0, c1, c2_norm)
                    if nbr >= 0 and nbr != node_id:
                        if do_write:
                            neighbor_out[out_offset + count] = nbr
                        count += 1
    return count


@njit(cache=True)
def build_face_neighbors_kernel(
    levels: np.ndarray,
    cell_ijk: np.ndarray,
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
                cell_ijk,
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
                cell_ijk,
                min_level,
                max_level,
                level_shapes,
                periodic_i2,
                face_neighbors,
                start,
                True,
            )
    return face_counts, face_offsets, face_neighbors

def _collect_frontier_cells(
    tree: Octree,
    *,
    max_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect frontier cells by traversing the rebuilt cell tree and stopping at `max_level`."""
    if max_level < 0:
        raise ValueError(f"Invalid max_level={max_level}.")

    cell_depth = tree._cell_depth
    cell_ijk = tree._cell_ijk
    cell_child = tree._cell_child
    root_cell_ids = tree._root_cell_ids

    frontier_cell_ids_list: list[int] = []
    stack = [int(cell_id) for cell_id in root_cell_ids[::-1].tolist()]
    while stack:
        cell_id = stack.pop()
        children = cell_child[cell_id]
        has_child = bool(np.any(children >= 0))
        if int(cell_depth[cell_id]) >= max_level or not has_child:
            frontier_cell_ids_list.append(int(cell_id))
            continue
        for child_id in children[::-1]:
            if int(child_id) >= 0:
                stack.append(int(child_id))

    frontier_cell_ids = np.asarray(frontier_cell_ids_list, dtype=np.int64)
    order = np.lexsort(
        (
            cell_ijk[frontier_cell_ids, 2],
            cell_ijk[frontier_cell_ids, 1],
            cell_ijk[frontier_cell_ids, 0],
            cell_depth[frontier_cell_ids],
        )
    )
    frontier_cell_ids = frontier_cell_ids[order]
    levels = cell_depth[frontier_cell_ids]
    frontier_cell_ijk = cell_ijk[frontier_cell_ids]

    leaf_row_count = int(tree.cell_levels.shape[0])
    cell_to_node_id = np.full(leaf_row_count, -1, dtype=np.int64)
    for node_id, frontier_cell_id in enumerate(frontier_cell_ids.tolist()):
        stack = [int(frontier_cell_id)]
        while stack:
            cell_id = stack.pop()
            children = cell_child[cell_id]
            has_child = False
            for child_id in children:
                if int(child_id) >= 0:
                    has_child = True
                    stack.append(int(child_id))
            if not has_child and cell_id < leaf_row_count:
                cell_to_node_id[cell_id] = int(node_id)

    return levels, frontier_cell_ijk, frontier_cell_ids, cell_to_node_id


def _build_face_neighbors(
    *,
    tree: Octree,
    target_max_level: int,
) -> OctreeFaceNeighbors:
    """Build one face-neighbor graph by traversing the rebuilt cell tree to one cutoff."""
    levels, cell_ijk, node_cell_ids, cell_to_node_id = _collect_frontier_cells(
        tree,
        max_level=target_max_level,
    )
    min_level = int(np.min(levels))
    level_shapes = np.empty((target_max_level - min_level + 1, 3), dtype=np.int64)
    root0 = int(tree.root_shape[0])
    root1 = int(tree.root_shape[1])
    root2 = int(tree.root_shape[2])
    for level in range(min_level, target_max_level + 1):
        scale = 1 << int(level)
        row = level - min_level
        level_shapes[row, 0] = root0 * scale
        level_shapes[row, 1] = root1 * scale
        level_shapes[row, 2] = root2 * scale
    periodic_i2 = str(tree.tree_coord) == "rpa"

    face_counts, face_offsets, face_neighbors = build_face_neighbors_kernel(
        levels,
        cell_ijk,
        min_level,
        target_max_level,
        level_shapes,
        periodic_i2,
    )
    return OctreeFaceNeighbors(
        levels=levels,
        cell_ijk=cell_ijk,
        face_counts=face_counts,
        face_offsets=face_offsets,
        face_neighbors=face_neighbors,
        node_cell_ids=node_cell_ids,
        cell_to_node_id=cell_to_node_id,
        min_level=min_level,
        max_level=target_max_level,
        periodic_i2=periodic_i2,
    )

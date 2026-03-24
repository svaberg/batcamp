#!/usr/bin/env python3
"""Octree builder and related utility functions."""

from __future__ import annotations

from collections import Counter
import logging
import re
from typing import TypeAlias

import numpy as np
from batread.dataset import Dataset

from .octree import TreeCoord
from .octree import DEFAULT_AXIS_RHO_TOL
from .octree import DEFAULT_TREE_COORD
from .octree import GridShape
from .octree import LevelCountTable
from .octree import Octree
from .octree import SUPPORTED_TREE_COORDS
from .octree import octree_class_for_coord

LevelShapeStatsRow: TypeAlias = tuple[int, int, float, float, int]
"""Tuple `(n_axis1, n_axis2, d_axis1, d_axis2, n_cells_at_level)`."""

LevelShapeStatsMap: TypeAlias = dict[int, LevelShapeStatsRow]
"""Map `level -> LevelShapeStatsRow`."""

BlockAux: TypeAlias = tuple[int, GridShape]
"""Parsed BLOCKS aux tuple `(n_blocks, cells_per_block_xyz)`."""

logger = logging.getLogger(__name__)


def _median_positive(values: np.ndarray, *, tiny: float = 1e-12) -> float:
    """Compute the median of positive values above `tiny`."""
    pos = np.asarray(values, dtype=float)
    pos = pos[pos > float(tiny)]
    if pos.size == 0:
        raise ValueError("No positive values available to infer spacing.")
    return float(np.median(pos))


def point_refinement_levels(
    n_points: int,
    corners: np.ndarray,
    cell_levels: np.ndarray,
) -> np.ndarray:
    """Assign each point the finest valid level among its neighboring cells."""
    out = np.full(n_points, -1, dtype=np.int64)
    for cell_id, nodes in enumerate(corners):
        level = int(cell_levels[cell_id])
        if level < 0:
            continue
        out[nodes] = np.maximum(out[nodes], level)
    return out


def format_histogram(levels: np.ndarray) -> str:
    """Format level histogram text as `level:count` pairs."""
    counts = Counter(int(v) for v in levels.tolist())
    return ", ".join(f"{lvl}:{counts[lvl]}" for lvl in sorted(counts))


def valid_cell_fraction(levels: np.ndarray) -> tuple[int, int, float]:
    """Compute valid-level fraction statistics for `levels >= 0`."""
    total = int(levels.size)
    valid = int(np.count_nonzero(levels >= 0))
    frac = float(valid / total) if total > 0 else 0.0
    return valid, total, frac


def _resolve_cell_levels(
    *,
    inferred_levels: np.ndarray | None,
    cell_levels: np.ndarray | None,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, int, int]:
    """Finalize level array with shared validation and min/max extraction."""
    if cell_levels is None:
        if inferred_levels is None:
            raise ValueError("inferred_levels is required when cell_levels is omitted.")
        levels = np.asarray(inferred_levels, dtype=np.int64)
    else:
        levels = np.asarray(cell_levels, dtype=np.int64)
    if levels.shape != tuple(expected_shape):
        raise ValueError(
            "cell_levels shape does not match inferred corner-cell shape: "
            f"levels={levels.shape}, inferred={expected_shape}."
        )
    valid_levels = levels[levels >= 0]
    if valid_levels.size == 0:
        raise ValueError("No valid (>=0) levels available to infer octree.")
    min_level = int(np.min(valid_levels))
    max_level = int(np.max(valid_levels))
    return levels, min_level, max_level


def _parse_blocks_aux(text: str | None) -> BlockAux | None:
    """Parse `BLOCKS` aux string as `(n_blocks, cells_per_block_xyz)`."""
    if text is None:
        return None
    match = re.search(r"(\d+)\s+(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", str(text))
    if not match:
        return None
    n_blocks = int(match.group(1))
    block_cells = (int(match.group(2)), int(match.group(3)), int(match.group(4)))
    return n_blocks, block_cells


def _blocks_match_cell_count(
    n_cells: int,
    block_count: int,
    block_cells_xyz: GridShape,
) -> bool:
    """Return `True` when BLOCKS metadata matches dataset cell count."""
    if block_count <= 0 or any(int(v) <= 0 for v in block_cells_xyz):
        return False
    cells_per_block = int(np.prod(np.asarray(block_cells_xyz, dtype=np.int64)))
    if cells_per_block <= 0:
        return False
    if int(n_cells) <= 0 or (int(n_cells) % cells_per_block) != 0:
        return False
    inferred_blocks = int(n_cells) // cells_per_block
    return inferred_blocks == int(block_count)


def _warn_if_blocks_aux_mismatch(ds: Dataset, n_cells: int) -> None:
    """Warn when `ds.aux['BLOCKS']` exists but conflicts with dataset cell count."""
    aux = getattr(ds, "aux", None)
    if not hasattr(aux, "get"):
        return
    raw = aux.get("BLOCKS")
    if raw is None:
        return
    raw_text = str(raw)
    parsed = _parse_blocks_aux(raw_text)
    if parsed is None:
        logger.warning(
            "Dataset aux BLOCKS='%s' is not parseable; ignoring BLOCKS metadata.",
            raw_text,
        )
        return
    block_count, block_cells_xyz = parsed
    if not _blocks_match_cell_count(n_cells, block_count, block_cells_xyz):
        cells_per_block = int(np.prod(np.asarray(block_cells_xyz, dtype=np.int64)))
        logger.warning(
            "Dataset aux BLOCKS='%s' does not match dataset cells (n_cells=%d, cells_per_block=%d); "
            "ignoring BLOCKS metadata.",
            raw_text,
            int(n_cells),
            cells_per_block,
        )


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


class OctreeBuilder:
    """Build octrees from dataset cell connectivity."""

    def __init__(
        self,
        *,
        level_rtol: float = 1e-4,
        level_atol: float = 1e-9,
    ) -> None:
        """Configure tolerances used for dyadic level inference."""
        self.level_rtol = float(level_rtol)
        self.level_atol = float(level_atol)
        from .builder_cartesian import CartesianOctreeBuilder
        from .builder_spherical import SphericalOctreeBuilder

        self._rpa_builder = SphericalOctreeBuilder(level_rtol=level_rtol, level_atol=level_atol)
        self._xyz_builder = CartesianOctreeBuilder(level_rtol=level_rtol, level_atol=level_atol)

    @staticmethod
    def _twos_factor(n: int) -> int:
        """Compute the exponent of the largest power of two dividing `n`."""
        k = 0
        while n > 0 and (n % 2 == 0):
            n //= 2
            k += 1
        return k

    @staticmethod
    def _full_tree_counts(leaf_shape: GridShape) -> tuple[LevelCountTable, GridShape, int]:
        """Compute full-tree counts, root shape, and depth from finest leaf shape."""
        depth = min(
            OctreeBuilder._twos_factor(leaf_shape[0]),
            OctreeBuilder._twos_factor(leaf_shape[1]),
            OctreeBuilder._twos_factor(leaf_shape[2]),
        )
        root_shape = (
            leaf_shape[0] >> depth,
            leaf_shape[1] >> depth,
            leaf_shape[2] >> depth,
        )
        base = int(root_shape[0] * root_shape[1] * root_shape[2])
        counts = tuple((level, base * (8**level), base * (8**level)) for level in range(depth + 1))
        return counts, root_shape, depth

    def build(
        self,
        ds: Dataset,
        *,
        tree_coord: TreeCoord = DEFAULT_TREE_COORD,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> Octree:
        """Build and bind an octree from a dataset.

        The returned tree is ready for lookup and ray methods.
        """
        return self._build(
            ds,
            tree_coord=tree_coord,
            axis_rho_tol=axis_rho_tol,
            cell_levels=None,
            bind=True,
        )

    def _build(
        self,
        ds: Dataset,
        *,
        tree_coord: TreeCoord = DEFAULT_TREE_COORD,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
        cell_levels: np.ndarray | None = None,
        bind: bool = True,
    ) -> Octree:
        """Internal build path with optional level and bind overrides."""
        if tree_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported tree_coord '{tree_coord}'; "
                f"expected one of {SUPPORTED_TREE_COORDS}."
            )
        if ds.corners is None:
            raise ValueError("Dataset has no corners; cannot build octree.")
        corners_arr = np.asarray(ds.corners, dtype=np.int64)

        tree_cls = octree_class_for_coord(tree_coord)
        if tree_coord == "rpa":
            level_shapes, levels, min_level, max_level, leaf_shape, weighted_cells = self._rpa_builder.infer_tree_geometry(
                ds,
                corners_arr,
                cell_levels=cell_levels,
                axis_rho_tol=axis_rho_tol,
            )
        else:
            level_shapes, levels, min_level, max_level, leaf_shape = self._xyz_builder.infer_tree_geometry(
                ds,
                corners_arr,
                cell_levels=cell_levels,
            )
            weighted_cells = int(
                sum(int(level_shapes[level][4]) * (8 ** int(max_level - level)) for level in level_shapes)
            )

        _warn_if_blocks_aux_mismatch(ds, int(corners_arr.shape[0]))
        _counts_full, root_shape, _depth = self._full_tree_counts(leaf_shape)
        level_offset = int(_depth) - int(max_level)
        if level_offset < 0:
            raise ValueError(
                f"Inferred level offset is negative: depth={_depth}, max_level={max_level}."
            )
        levels = np.asarray(levels, dtype=np.int64)
        levels_abs = np.array(levels, copy=True)
        levels_abs[levels_abs >= 0] += int(level_offset)
        level_counts = tuple(
            (
                int(level + level_offset),
                int(level_shapes[level][4]),
                int(level_shapes[level][4] * (8 ** int(max_level - level))),
            )
            for level in sorted(level_shapes)
        )
        is_full = (
            int(np.count_nonzero(levels_abs >= 0)) == int(levels_abs.size)
            and int(sum(item[2] for item in level_counts)) == int(np.prod(leaf_shape))
            and int(weighted_cells) == int(np.prod(leaf_shape))
        )
        tree = tree_cls(
            leaf_shape=leaf_shape,
            root_shape=root_shape,
            is_full=bool(is_full),
            level_counts=level_counts,
            min_level=int(min_level + level_offset),
            max_level=int(max_level + level_offset),
            tree_coord=tree_coord,
            cell_levels=levels_abs,
            axis_rho_tol=float(axis_rho_tol),
        )
        if tree_coord == "rpa":
            self._rpa_builder.populate_tree_state(tree, ds, corners_arr)
        else:
            self._xyz_builder.populate_tree_state(tree, ds, corners_arr)
        if bind:
            tree.bind(ds, axis_rho_tol=axis_rho_tol)
        return tree

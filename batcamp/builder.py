#!/usr/bin/env python3
"""Octree builder and related utility functions."""

from __future__ import annotations

from collections import Counter
from typing import TypeAlias

import numpy as np
from starwinds_readplt.dataset import Dataset

from .octree import TreeCoord
from .octree import DEFAULT_AXIS_RHO_TOL
from .octree import DEFAULT_TREE_COORD
from .octree import GridShape
from .octree import LevelCountTable
from .octree import Octree
from .octree import SUPPORTED_TREE_COORDS
from .octree import octree_class_for_coord
from .builder_cartesian import CartesianOctreeBuilder
from .builder_spherical import SphericalOctreeBuilder

LevelShapeStatsRow: TypeAlias = tuple[int, int, float, float, int]
"""Tuple `(n_axis1, n_axis2, d_axis1, d_axis2, n_cells_at_level)`."""

LevelShapeStatsMap: TypeAlias = dict[int, LevelShapeStatsRow]
"""Map `level -> LevelShapeStatsRow`."""


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

    @staticmethod
    def _infer_leaf_shape_from_levels(
        level_shapes: LevelShapeStatsMap,
    ) -> tuple[GridShape, int, int]:
        """Infer finest leaf shape from per-level shape/count statistics."""
        max_level = max(level_shapes)
        n_axis1_f = int(level_shapes[max_level][0])
        n_axis2_f = int(level_shapes[max_level][1])
        weighted_cells = 0
        for level, (_n_axis1, _n_axis2, _d_axis1, _d_axis2, count) in level_shapes.items():
            weighted_cells += int(count) * (8 ** int(max_level - level))

        denom = int(n_axis1_f * n_axis2_f)
        if denom <= 0:
            raise ValueError("Invalid finest angular denominator while inferring n_axis0.")
        n_axis0_float = weighted_cells / float(denom)
        n_axis0 = int(round(n_axis0_float))
        if not np.isclose(n_axis0_float, float(n_axis0), rtol=0.0, atol=1e-9):
            raise ValueError(
                "Could not infer integer finest n_axis0 from weighted cell counts: "
                f"weighted={weighted_cells}, n_axis1={n_axis1_f}, n_axis2={n_axis2_f}."
            )
        return (n_axis0, n_axis1_f, n_axis2_f), int(weighted_cells), max_level

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
        return self._build_with_overrides(
            ds,
            tree_coord=tree_coord,
            axis_rho_tol=axis_rho_tol,
            cell_levels=None,
            bind=True,
        )

    def _build_with_overrides(
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
            level_shapes, levels, min_level, max_level = self._rpa_builder.infer_level_shapes(
                ds,
                corners_arr,
                cell_levels=cell_levels,
                axis_rho_tol=axis_rho_tol,
            )
        else:
            level_shapes, levels, min_level, max_level = self._xyz_builder.infer_level_shapes(
                ds,
                corners_arr,
                cell_levels=cell_levels,
            )

        leaf_shape, weighted_cells, _max_level = self._infer_leaf_shape_from_levels(level_shapes)
        _counts_full, root_shape, _depth = self._full_tree_counts(leaf_shape)
        level_counts = tuple(
            (
                int(level),
                int(level_shapes[level][4]),
                int(level_shapes[level][4] * (8 ** int(max_level - level))),
            )
            for level in sorted(level_shapes)
        )
        is_full = (
            int(np.count_nonzero(levels >= 0)) == int(levels.size)
            and int(sum(item[2] for item in level_counts)) == int(np.prod(leaf_shape))
            and int(weighted_cells) == int(np.prod(leaf_shape))
        )
        tree = tree_cls(
            leaf_shape=leaf_shape,
            root_shape=root_shape,
            is_full=bool(is_full),
            level_counts=level_counts,
            min_level=min_level,
            max_level=max_level,
            tree_coord=tree_coord,
            cell_levels=levels,
        )
        if bind:
            tree.bind(ds, axis_rho_tol=axis_rho_tol)
        return tree

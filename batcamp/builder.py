#!/usr/bin/env python3
"""Octree builder and related utility functions."""

from __future__ import annotations

from collections import Counter
import logging
import re
from typing import TypeAlias

import numpy as np
from batread import Dataset

from .constants import DEFAULT_TREE_COORD
from .constants import SUPPORTED_TREE_COORDS
from .constants import XYZ_VARS
from .octree import Octree
from .shared_types import GridShape
from .shared_types import TreeCoord

LevelShapeStatsRow: TypeAlias = tuple[int, int, float, float, int]
"""Tuple `(n_axis1, n_axis2, d_axis1, d_axis2, n_cells_at_level)`."""

LevelShapeStatsMap: TypeAlias = dict[int, LevelShapeStatsRow]
"""Map `level -> LevelShapeStatsRow`."""

logger = logging.getLogger(__name__)

DEFAULT_AXIS_RHO_TOL = 1e-12
"""Default polar-axis radius tolerance used only during spherical builder inference."""

DEFAULT_MIN_VALID_CELL_FRACTION = 0.5
"""Default minimum fraction of valid inferred cell levels accepted by builder utilities."""


def infer_tree_coord_from_geometry(ds: Dataset, *, sample_size: int = 2048) -> TreeCoord:
    """Guess whether the mesh is Cartesian (`xyz`) or spherical-like (`rpa`)."""
    corners = getattr(ds, "corners", None)
    if corners is None:
        raise ValueError("Dataset has no cell connectivity (corners).")
    corners_arr = np.asarray(corners, dtype=np.int64)
    if corners_arr.ndim != 2 or corners_arr.shape[0] == 0:
        return "rpa"

    if corners_arr.shape[0] > int(sample_size):
        idx = np.linspace(0, corners_arr.shape[0] - 1, int(sample_size), dtype=np.int64)
        sample = corners_arr[idx]
    else:
        sample = corners_arr

    x = np.asarray(ds[XYZ_VARS[0]], dtype=float)
    y = np.asarray(ds[XYZ_VARS[1]], dtype=float)
    z = np.asarray(ds[XYZ_VARS[2]], dtype=float)
    xr = np.round(x[sample], 12)
    yr = np.round(y[sample], 12)
    zr = np.round(z[sample], 12)

    ux = np.array([np.unique(row).size for row in xr], dtype=np.int64)
    uy = np.array([np.unique(row).size for row in yr], dtype=np.int64)
    uz = np.array([np.unique(row).size for row in zr], dtype=np.int64)
    axis_like = (ux <= 2) & (uy <= 2) & (uz <= 2)
    frac_axis_like = float(np.mean(axis_like)) if axis_like.size > 0 else 0.0
    return "xyz" if frac_axis_like >= 0.98 else "rpa"


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
) -> tuple[np.ndarray, int]:
    """Finalize level array with shared validation and max-level extraction."""
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
    max_level = int(np.max(valid_levels))
    return levels, max_level


def _warn_if_blocks_aux_mismatch(ds: Dataset, n_cells: int) -> None:
    """Warn when `ds.aux['BLOCKS']` exists but conflicts with dataset cell count."""
    aux = getattr(ds, "aux", None)
    if not hasattr(aux, "get"):
        return
    raw = aux.get("BLOCKS")
    if raw is None:
        return
    raw_text = str(raw)
    match = re.search(r"(\d+)\s+(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", raw_text)
    if match is None:
        logger.warning(
            "Dataset aux BLOCKS='%s' is not parseable; ignoring BLOCKS metadata.",
            raw_text,
        )
        return
    block_count = int(match.group(1))
    block_cells_xyz = (int(match.group(2)), int(match.group(3)), int(match.group(4)))
    cells_per_block = int(np.prod(np.asarray(block_cells_xyz, dtype=np.int64)))
    block_match = (
        block_count > 0
        and all(int(v) > 0 for v in block_cells_xyz)
        and cells_per_block > 0
        and int(n_cells) > 0
        and (int(n_cells) % cells_per_block) == 0
        and (int(n_cells) // cells_per_block) == block_count
    )
    if not block_match:
        logger.warning(
            "Dataset aux BLOCKS='%s' does not match dataset cells (n_cells=%d, cells_per_block=%d); "
            "ignoring BLOCKS metadata.",
            raw_text,
            int(n_cells),
            cells_per_block,
        )

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

    def build(
        self,
        ds: Dataset,
        *,
        tree_coord: TreeCoord | None = None,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    ) -> Octree:
        """Build and bind an octree from a dataset.

        The returned tree is ready for lookup.
        """
        if ds.corners is None:
            raise ValueError("Dataset has no corners; cannot build octree.")
        resolved_tree_coord = infer_tree_coord_from_geometry(ds) if tree_coord is None else tree_coord
        return self._build(
            ds,
            tree_coord=resolved_tree_coord,
            axis_rho_tol=axis_rho_tol,
            cell_levels=None,
        )

    def _build(
        self,
        ds: Dataset,
        *,
        tree_coord: TreeCoord = DEFAULT_TREE_COORD,
        axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
        cell_levels: np.ndarray | None = None,
    ) -> Octree:
        """Internal build path with optional exact cell-level override."""
        from .persistence import OctreeState

        if tree_coord not in SUPPORTED_TREE_COORDS:
            raise ValueError(
                f"Unsupported tree_coord '{tree_coord}'; "
                f"expected one of {SUPPORTED_TREE_COORDS}."
            )
        if ds.corners is None:
            raise ValueError("Dataset has no corners; cannot build octree.")
        corners_arr = np.asarray(ds.corners, dtype=np.int64)

        if tree_coord == "rpa":
            level_shapes, levels, max_level = self._rpa_builder.infer_level_shapes(
                ds,
                corners_arr,
                cell_levels=cell_levels,
                axis_rho_tol=axis_rho_tol,
            )
            leaf_shape = self._rpa_builder.infer_leaf_shape(level_shapes)
        else:
            _level_shapes, levels, max_level = self._xyz_builder.infer_level_shapes(
                ds,
                corners_arr,
                cell_levels=cell_levels,
            )
            leaf_shape = self._xyz_builder.infer_leaf_shape(
                ds,
                corners_arr,
                levels,
                max_level=max_level,
            )

        _warn_if_blocks_aux_mismatch(ds, int(corners_arr.shape[0]))
        depth = min(int(np.log2(v & -v)) for v in leaf_shape)
        root_shape = (
            leaf_shape[0] >> depth,
            leaf_shape[1] >> depth,
            leaf_shape[2] >> depth,
        )
        level_offset = int(depth) - int(max_level)
        if level_offset < 0:
            raise ValueError(
                f"Inferred level offset is negative: depth={depth}, max_level={max_level}."
            )
        levels = np.asarray(levels, dtype=np.int64)
        levels_abs = np.array(levels, copy=True)
        levels_abs[levels_abs >= 0] += int(level_offset)
        if tree_coord == "rpa":
            state_payload = self._rpa_builder.populate_tree_state(
                leaf_shape=leaf_shape,
                max_level=int(max_level + level_offset),
                cell_levels=levels_abs,
                axis_rho_tol=float(axis_rho_tol),
                ds=ds,
                corners=corners_arr,
            )
        else:
            state_payload = self._xyz_builder.populate_tree_state(
                leaf_shape=leaf_shape,
                max_level=int(max_level + level_offset),
                cell_levels=levels_abs,
                ds=ds,
                corners=corners_arr,
            )
        state = OctreeState(
            tree_coord=tree_coord,
            root_shape=tuple(int(v) for v in root_shape),
            **state_payload,
        )
        return Octree.from_state(
            state,
            ds=ds,
        )

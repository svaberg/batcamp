#!/usr/bin/env python3
"""Octree builder and related utility functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import TypeAlias

import numpy as np
from batread import Dataset

from .shared_types import TreeCoord

if TYPE_CHECKING:
    from .persistence import OctreeState

LevelShapeStatsRow: TypeAlias = tuple[int, int, float, float, int]
"""Tuple `(n_axis1, n_axis2, d_axis1, d_axis2, n_cells_at_level)`."""

LevelShapeStatsMap: TypeAlias = dict[int, LevelShapeStatsRow]
"""Map `level -> LevelShapeStatsRow`."""

logger = logging.getLogger(__name__)

DEFAULT_AXIS_TOL = 1e-12
"""Default polar-axis radius tolerance used only during spherical builder inference."""

DEFAULT_MIN_VALID_CELL_FRACTION = 0.5
"""Default minimum fraction of valid inferred cell levels accepted by builder utilities."""


def infer_tree_coord_from_geometry(
    points: np.ndarray,
    corners: np.ndarray,
    *,
    sample_size: int = 2048,
) -> TreeCoord:
    """Guess whether point/corner geometry is Cartesian (`xyz`) or spherical-like (`rpa`)."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3).")
    corners_arr = np.asarray(corners, dtype=np.int64)
    if corners_arr.ndim != 2 or corners_arr.shape[0] == 0:
        return "rpa"

    if corners_arr.shape[0] > int(sample_size):
        idx = np.linspace(0, corners_arr.shape[0] - 1, int(sample_size), dtype=np.int64)
        sample = corners_arr[idx]
    else:
        sample = corners_arr

    xr = np.round(points[sample, 0], 12)
    yr = np.round(points[sample, 1], 12)
    zr = np.round(points[sample, 2], 12)

    ux = np.array([np.unique(row).size for row in xr], dtype=np.int64)
    uy = np.array([np.unique(row).size for row in yr], dtype=np.int64)
    uz = np.array([np.unique(row).size for row in zr], dtype=np.int64)
    axis_like = (ux <= 2) & (uy <= 2) & (uz <= 2)
    frac_axis_like = float(np.mean(axis_like)) if axis_like.size > 0 else 0.0
    return "xyz" if frac_axis_like >= 0.98 else "rpa"


def median_positive(values: np.ndarray, *, tiny: float = 1e-12) -> float:
    """Compute the median of positive values above `tiny`."""
    pos = np.asarray(values, dtype=float)
    pos = pos[pos > float(tiny)]
    if pos.size == 0:
        raise ValueError("No positive values available to infer spacing.")
    return float(np.median(pos))


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
    """Warn when BLOCKS metadata is malformed or disagrees with the dataset cell count."""
    raw = ds.aux.get("BLOCKS")
    if raw is None:
        return
    raw_text = str(raw)
    # Parse `nblock nx x ny x nz` by normalizing both `x` and `X` separators to spaces.
    tokens = raw_text.replace("x", " ").replace("X", " ").split()
    if len(tokens) != 4:
        logger.warning(
            "Dataset aux BLOCKS='%s' is not parseable; ignoring BLOCKS metadata.",
            raw_text,
        )
        return
    try:
        block_count, nx, ny, nz = (int(token) for token in tokens)
    except ValueError:
        logger.warning(
            "Dataset aux BLOCKS='%s' is not parseable; ignoring BLOCKS metadata.",
            raw_text,
        )
        return
    block_cells_xyz = (nx, ny, nz)
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


def _build_octree_state(
    points: np.ndarray,
    corners: np.ndarray,
    *,
    tree_coord: TreeCoord | None = None,
    axis_tol: float = DEFAULT_AXIS_TOL,
    level_rtol: float = 1e-4,
    level_atol: float = 1e-9,
    cell_levels: np.ndarray | None = None,
) -> "OctreeState":
    """Infer exact octree state from explicit points/corners."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3).")
    corners_arr = np.asarray(corners, dtype=np.int64)
    if corners_arr.ndim != 2 or corners_arr.shape[1] != 8:
        raise ValueError("corners must have shape (n_cells, 8).")

    resolved_tree_coord = infer_tree_coord_from_geometry(points, corners_arr) if tree_coord is None else tree_coord
    logger.info("resolve tree coord: coord=%s", resolved_tree_coord)

    if resolved_tree_coord == "rpa":
        from .builder_spherical import infer_levels
        from .builder_spherical import populate_tree_state

        levels, max_level, leaf_shape = infer_levels(
            points,
            corners_arr,
            cell_levels=cell_levels,
            axis_tol=axis_tol,
            level_rtol=level_rtol,
            level_atol=level_atol,
        )
    elif resolved_tree_coord == "xyz":
        from .builder_cartesian import infer_levels
        from .builder_cartesian import populate_tree_state

        levels, max_level, leaf_shape = infer_levels(
            points,
            corners_arr,
            cell_levels=cell_levels,
            level_rtol=level_rtol,
            level_atol=level_atol,
        )
    else:
        raise ValueError(
            f"Unsupported tree_coord '{resolved_tree_coord}'; expected 'rpa' or 'xyz'."
        )
    logger.info("infer levels: coord=%s max_level=%d", resolved_tree_coord, int(max_level))
    logger.info("infer leaf shape: coord=%s leaf_shape=%s", resolved_tree_coord, leaf_shape)

    depth: int | None = None
    for axis_size in leaf_shape:
        axis_depth = 0
        value = int(axis_size)
        while value > 0 and (value % 2) == 0:
            axis_depth += 1
            value //= 2
        if depth is None or axis_depth < depth:
            depth = axis_depth
    if depth is None:
        raise ValueError(f"Invalid leaf_shape={leaf_shape}.")
    root_shape = (
        int(leaf_shape[0]) >> depth,
        int(leaf_shape[1]) >> depth,
        int(leaf_shape[2]) >> depth,
    )
    level_offset = int(depth) - int(max_level)
    if level_offset < 0:
        raise ValueError(
            f"Inferred level offset is negative: depth={depth}, max_level={max_level}."
        )
    levels_arr = np.asarray(levels, dtype=np.int64)
    levels_abs = np.array(levels_arr, copy=True)
    levels_abs[levels_abs >= 0] += int(level_offset)
    logger.info(
        "normalize levels: coord=%s root_shape=%s depth=%d max_level=%d",
        resolved_tree_coord,
        root_shape,
        int(depth),
        int(max_level + level_offset),
    )

    if resolved_tree_coord == "rpa":
        built_state = populate_tree_state(
            leaf_shape=leaf_shape,
            max_level=int(max_level + level_offset),
            cell_levels=levels_abs,
            axis_tol=axis_tol,
            points=points,
            corners=corners_arr,
        )
    else:
        built_state = populate_tree_state(
            leaf_shape=leaf_shape,
            max_level=int(max_level + level_offset),
            cell_levels=levels_abs,
            points=points,
            corners=corners_arr,
        )
    logger.info("populate tree state: coord=%s", resolved_tree_coord)
    from .persistence import OctreeState
    cell_levels = np.asarray(built_state["cell_levels"], dtype=np.int64)
    cell_ijk = np.asarray(built_state["cell_ijk"], dtype=np.int64)

    return OctreeState(
        tree_coord=resolved_tree_coord,
        root_shape=tuple(int(v) for v in root_shape),
        cell_levels=cell_levels,
        cell_ijk=cell_ijk,
    )

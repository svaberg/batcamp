#!/usr/bin/env python3
"""Octree builder and related utility functions."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import logging
import re
import time
from typing import TypeAlias

import numpy as np
from batread import Dataset

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


@contextmanager
def timed_stage(stage: str):
    """Log one start/finish INFO pair around one builder stage."""
    logger.debug("%s...", stage, stacklevel=2)
    t0 = time.perf_counter()
    yield
    logger.info("%s complete in %.2fs", stage, float(time.perf_counter() - t0), stacklevel=2)


def xyz_points_from_ds(ds: Dataset) -> np.ndarray:
    """Extract one explicit `(n_points, 3)` XYZ point array from a dataset."""
    return np.column_stack(tuple(np.asarray(ds[name], dtype=np.float64) for name in XYZ_VARS))


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

def root_shape_and_depth(leaf_shape: GridShape) -> tuple[GridShape, int]:
    """Return dyadic root shape and tree depth implied by one finest leaf shape."""
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
    return (
        int(leaf_shape[0]) >> depth,
        int(leaf_shape[1]) >> depth,
        int(leaf_shape[2]) >> depth,
    ), depth

def build_octree(
    points: np.ndarray,
    corners: np.ndarray,
    *,
    tree_coord: TreeCoord | None = None,
    axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    level_rtol: float = 1e-4,
    level_atol: float = 1e-9,
) -> Octree:
    """Build one octree from explicit points and corners."""
    with timed_stage("build_octree"):
        with timed_stage("prepare explicit geometry"):
            points = np.asarray(points, dtype=np.float64)
            corners_arr = np.asarray(corners, dtype=np.int64)
        logger.info(
            "prepare explicit geometry: n_points=%d n_cells=%d",
            int(points.shape[0]),
            int(corners_arr.shape[0]),
        )
        tree = Octree(
            points,
            corners_arr,
            tree_coord=tree_coord,
            axis_rho_tol=axis_rho_tol,
            level_rtol=level_rtol,
            level_atol=level_atol,
        )
    logger.info(
        "build_octree: coord=%s n_points=%d n_cells=%d",
        str(tree.tree_coord),
        int(points.shape[0]),
        int(corners_arr.shape[0]),
    )
    return tree


def _build_octree_state(
    points: np.ndarray,
    corners: np.ndarray,
    *,
    tree_coord: TreeCoord | None = None,
    axis_rho_tol: float = DEFAULT_AXIS_RHO_TOL,
    level_rtol: float = 1e-4,
    level_atol: float = 1e-9,
    cell_levels: np.ndarray | None = None,
) -> "OctreeState":
    """Infer exact octree state from explicit points/corners."""
    from .builder_cartesian import cartesian_cell_geometry
    from .builder_cartesian import _infer_leaf_shape_from_geometry
    from .builder_cartesian import _infer_levels_from_geometry
    from .builder_cartesian import _populate_tree_state_from_geometry
    from .builder_spherical import infer_leaf_shape as infer_rpa_leaf_shape
    from .builder_spherical import infer_level_shapes as infer_rpa_level_shapes
    from .builder_spherical import populate_tree_state as populate_rpa_tree_state
    from .persistence import OctreeState

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3).")
    corners_arr = np.asarray(corners, dtype=np.int64)
    if corners_arr.ndim != 2 or corners_arr.shape[1] != 8:
        raise ValueError("corners must have shape (n_cells, 8).")
    with timed_stage("resolve tree coord"):
        resolved_tree_coord = infer_tree_coord_from_geometry(points, corners_arr) if tree_coord is None else tree_coord
    if resolved_tree_coord not in SUPPORTED_TREE_COORDS:
        raise ValueError(
            f"Unsupported tree_coord '{resolved_tree_coord}'; "
            f"expected one of {SUPPORTED_TREE_COORDS}."
        )
    logger.info("resolve tree coord: coord=%s", resolved_tree_coord)

    with timed_stage("infer levels"):
        if resolved_tree_coord == "rpa":
            level_shapes, levels, max_level = infer_rpa_level_shapes(
                points,
                corners_arr,
                cell_levels=cell_levels,
                axis_rho_tol=axis_rho_tol,
                level_rtol=level_rtol,
                level_atol=level_atol,
            )
        else:
            cell_min, cell_max, cell_span = cartesian_cell_geometry(
                points,
                corners_arr,
            )
            levels, max_level = _infer_levels_from_geometry(
                cell_span,
                cell_levels=cell_levels,
                level_rtol=level_rtol,
                level_atol=level_atol,
            )
    logger.info("infer levels: coord=%s max_level=%d", resolved_tree_coord, int(max_level))

    with timed_stage("infer leaf shape"):
        if resolved_tree_coord == "rpa":
            leaf_shape = infer_rpa_leaf_shape(level_shapes)
        else:
            leaf_shape = _infer_leaf_shape_from_geometry(
                cell_min,
                cell_max,
                cell_span,
                levels,
                max_level=max_level,
            )
    logger.info("infer leaf shape: coord=%s leaf_shape=%s", resolved_tree_coord, leaf_shape)

    with timed_stage("normalize levels"):
        root_shape, depth = root_shape_and_depth(leaf_shape)
        level_offset = int(depth) - int(max_level)
        if level_offset < 0:
            raise ValueError(
                f"Inferred level offset is negative: depth={depth}, max_level={max_level}."
            )
        levels = np.asarray(levels, dtype=np.int64)
        levels_abs = np.array(levels, copy=True)
        levels_abs[levels_abs >= 0] += int(level_offset)
    logger.info(
        "normalize levels: coord=%s root_shape=%s depth=%d max_level=%d",
        resolved_tree_coord,
        root_shape,
        int(depth),
        int(max_level + level_offset),
    )

    with timed_stage("populate tree state"):
        if resolved_tree_coord == "rpa":
            state_payload = populate_rpa_tree_state(
                leaf_shape=leaf_shape,
                max_level=int(max_level + level_offset),
                cell_levels=levels_abs,
                axis_rho_tol=float(axis_rho_tol),
                points=points,
                corners=corners_arr,
            )
        else:
            state_payload = _populate_tree_state_from_geometry(
                leaf_shape=leaf_shape,
                max_level=int(max_level + level_offset),
                cell_levels=levels_abs,
                cell_min=cell_min,
                cell_max=cell_max,
            )
    logger.info("populate tree state: coord=%s", resolved_tree_coord)
    return OctreeState(
        tree_coord=resolved_tree_coord,
        root_shape=tuple(int(v) for v in root_shape),
        **state_payload,
    )

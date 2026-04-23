#!/usr/bin/env python3
"""Shared constants, type aliases, and lightweight helpers.

This module is the central home for small cross-file declarations that do not
deserve their own tiny modules. Shared kernel-facing `NamedTuple` types should
also live here once they are introduced.
"""

from __future__ import annotations

from functools import wraps
import logging
import time
from typing import Literal
from typing import NamedTuple
from typing import TypeAlias

import numpy as np
from numba import njit

__all__ = [
    "DEFAULT_TREE_COORD",
    "GridIndex",
    "GridPath",
    "GridShape",
    "LevelCountRow",
    "LevelCountTable",
    "LookupTree",
    "resolved_active_side",
    "TraceScratch",
    "SUPPORTED_TREE_COORDS",
    "TraversalTree",
    "TrilinearField",
    "TreeCoord",
    "quick_subface_slot",
    "XYZ_VARS",
    "timed_info_decorator",
]

# Supported octree coordinate-system tags used throughout the package.
SUPPORTED_TREE_COORDS = ("rpa", "xyz")

# Default coordinate-system tag when one must be implied.
DEFAULT_TREE_COORD = "xyz"

# Canonical dataset variable names for Cartesian point coordinates.
XYZ_VARS = ("X [R]", "Y [R]", "Z [R]")

TreeCoord: TypeAlias = Literal["rpa", "xyz"]
"""Coordinate-system tag for octree state and lookup."""

GridShape: TypeAlias = tuple[int, int, int]
"""Grid extents `(n_axis0, n_axis1, n_axis2)`."""

GridIndex: TypeAlias = tuple[int, int, int]
"""Discrete cell/bin index triplet `(i_axis0, i_axis1, i_axis2)`."""

GridPath: TypeAlias = tuple[GridIndex, ...]
"""Root-to-leaf sequence of `GridIndex` entries."""

LevelCountRow: TypeAlias = tuple[int, int, int]
"""Tuple meaning `(level, leaf_count, fine_equivalent_count)`."""

LevelCountTable: TypeAlias = tuple[LevelCountRow, ...]
"""Sorted collection of per-level count rows."""


class TrilinearField(NamedTuple):
    """Trilinear field samples tied to one octree's leaf geometry."""

    cell_bounds: np.ndarray
    corners: np.ndarray
    point_values: np.ndarray


class LookupTree(NamedTuple):
    """Tree geometry/topology needed by point-ownership lookup kernels."""

    cell_child: np.ndarray
    root_cell_ids: np.ndarray
    cell_parent: np.ndarray
    cell_bounds: np.ndarray
    domain_bounds: np.ndarray
    axis2_period: float
    axis2_periodic: bool


class TraversalTree(NamedTuple):
    """Tree geometry/topology needed by ray-traversal kernels."""

    root_cell_ids: np.ndarray
    cell_child: np.ndarray
    cell_bounds: np.ndarray
    domain_bounds: np.ndarray
    cell_neighbor: np.ndarray
    cell_depth: np.ndarray


class TraceScratch(NamedTuple):
    """Per-chunk traced segment scratch buffers for direct accumulation kernels."""

    cell_counts: np.ndarray
    cell_ids: np.ndarray
    times: np.ndarray


@njit(cache=True)
def quick_subface_slot(
    cell_neighbor: np.ndarray,
    cell_id: int,
    face_id: int,
) -> tuple[int, bool]:
    """Return an immediately-resolved subface slot when one neighbor row is uniform."""
    first_neighbor = -1
    first_slot = -1
    for subface_id in range(4):
        neighbor_id = int(cell_neighbor[int(cell_id), int(face_id), subface_id])
        if neighbor_id < 0:
            continue
        if first_neighbor < 0:
            first_neighbor = neighbor_id
            first_slot = int(subface_id)
            continue
        if neighbor_id != first_neighbor:
            return -1, False
    if first_neighbor < 0:
        return 0, True
    return int(first_slot), True


@njit(cache=True)
def resolved_active_side(
    face_id_to_side: np.ndarray,
    active_face_id: int,
    active_face_order: np.ndarray,
    current_face_order: int,
    implicit_active_side: int,
) -> int:
    """Return the tangential side chosen by explicit or implicit active-face ownership."""
    if active_face_id >= 0:
        active_side = int(face_id_to_side[int(active_face_id)])
        if int(active_face_order[int(active_face_id)]) < int(current_face_order):
            active_side = 1 - active_side
        return int(active_side)
    return int(implicit_active_side)


def timed_info_decorator(func):
    """Decorate one function with a fixed elapsed-time INFO log line."""
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapped(*args, **kwargs):
        logger.debug("%s...", func.__name__)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        logger.info("%s complete in %.2fs", func.__name__, float(time.perf_counter() - t0))
        return result

    return wrapped

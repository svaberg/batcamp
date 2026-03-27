#!/usr/bin/env python3
"""Cartesian coordinate support for octree lookup.

Important modeling assumption for ``tree_coord="xyz"``:
leaf cells are treated as axis-aligned Cartesian slabs from
per-cell corner min/max values. Lookup/interpolation kernels in the
Cartesian backend are exact under this axis-aligned representation.
"""

from __future__ import annotations
import numpy as np

from .octree import AXIS0
from .octree import AXIS1
from .octree import AXIS2
from .octree import START
from .octree import WIDTH


def _attach_cartesian_coord_state(tree, points: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Derive Cartesian cell bounds, domain bounds, and axis-2 periodic metadata from point/corner geometry."""
    xyz_min = np.min(points, axis=0).astype(np.float64, copy=False)
    xyz_max = np.max(points, axis=0).astype(np.float64, copy=False)
    xyz_span = np.maximum(xyz_max - xyz_min, np.finfo(np.float64).tiny)
    fine_step = xyz_span / np.asarray(tree.leaf_shape, dtype=np.float64)
    n_cells = int(tree._cell_depth.shape[0])
    occupied_ids = np.flatnonzero(tree._cell_depth >= 0)
    cell_shift = int(tree.max_level) - tree._cell_depth[occupied_ids]
    cell_width = np.left_shift(np.ones_like(cell_shift, dtype=np.int64), cell_shift)
    cell_start_f = np.left_shift(tree._cell_ijk[occupied_ids], cell_shift[:, None])
    cell_bounds = np.empty((n_cells, 3, 2), dtype=np.float64)
    cell_bounds.fill(np.nan)
    cell_bounds[occupied_ids, :, START] = xyz_min + cell_start_f * fine_step
    cell_bounds[occupied_ids, :, WIDTH] = cell_width[:, None] * fine_step
    domain_bounds = np.empty((3, 2), dtype=np.float64)
    domain_bounds[:, START] = xyz_min
    domain_bounds[:, WIDTH] = xyz_max - xyz_min
    return cell_bounds, domain_bounds, 0.0, False

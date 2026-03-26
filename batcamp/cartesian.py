#!/usr/bin/env python3
"""Cartesian coordinate support for octree lookup.

Important modeling assumption for ``tree_coord="xyz"``:
leaf cells are treated as axis-aligned Cartesian slabs from
per-cell corner min/max values. Lookup/interpolation kernels in the
Cartesian backend are exact under this axis-aligned representation.
"""

from __future__ import annotations
import numpy as np

from .constants import XYZ_VARS
from .octree import AXIS0
from .octree import AXIS1
from .octree import AXIS2
from .octree import START
from .octree import WIDTH
def _attach_cartesian_coord_state(tree, ds, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Derive Cartesian cell bounds, domain bounds, and axis-2 periodic metadata from the bound dataset."""
    x = np.asarray(ds[XYZ_VARS[0]], dtype=np.float64)
    y = np.asarray(ds[XYZ_VARS[1]], dtype=np.float64)
    z = np.asarray(ds[XYZ_VARS[2]], dtype=np.float64)
    cell_levels = tree.cell_levels
    cell_x = x[corners]
    cell_y = y[corners]
    cell_z = z[corners]
    cell_x_min = np.min(cell_x, axis=1)
    cell_x_max = np.max(cell_x, axis=1)
    cell_y_min = np.min(cell_y, axis=1)
    cell_y_max = np.max(cell_y, axis=1)
    cell_z_min = np.min(cell_z, axis=1)
    cell_z_max = np.max(cell_z, axis=1)

    leaf_cell_ids = np.flatnonzero(cell_levels >= 0)
    n_cells = int(tree._cell_depth.shape[0])
    octree_cell_x_min = np.full(n_cells, np.inf, dtype=np.float64)
    octree_cell_x_max = np.full(n_cells, -np.inf, dtype=np.float64)
    octree_cell_y_min = np.full(n_cells, np.inf, dtype=np.float64)
    octree_cell_y_max = np.full(n_cells, -np.inf, dtype=np.float64)
    octree_cell_z_min = np.full(n_cells, np.inf, dtype=np.float64)
    octree_cell_z_max = np.full(n_cells, -np.inf, dtype=np.float64)
    octree_cell_x_min[leaf_cell_ids] = cell_x_min[leaf_cell_ids]
    octree_cell_x_max[leaf_cell_ids] = cell_x_max[leaf_cell_ids]
    octree_cell_y_min[leaf_cell_ids] = cell_y_min[leaf_cell_ids]
    octree_cell_y_max[leaf_cell_ids] = cell_y_max[leaf_cell_ids]
    octree_cell_z_min[leaf_cell_ids] = cell_z_min[leaf_cell_ids]
    octree_cell_z_max[leaf_cell_ids] = cell_z_max[leaf_cell_ids]
    occupied_ids = np.flatnonzero(tree._cell_depth >= 0)
    occupied_ids = occupied_ids[np.argsort(tree._cell_depth[occupied_ids])[::-1]]
    for cell_id in occupied_ids:
        parent = int(tree._cell_parent[cell_id])
        if parent < 0:
            continue
        octree_cell_x_min[parent] = min(octree_cell_x_min[parent], octree_cell_x_min[cell_id])
        octree_cell_x_max[parent] = max(octree_cell_x_max[parent], octree_cell_x_max[cell_id])
        octree_cell_y_min[parent] = min(octree_cell_y_min[parent], octree_cell_y_min[cell_id])
        octree_cell_y_max[parent] = max(octree_cell_y_max[parent], octree_cell_y_max[cell_id])
        octree_cell_z_min[parent] = min(octree_cell_z_min[parent], octree_cell_z_min[cell_id])
        octree_cell_z_max[parent] = max(octree_cell_z_max[parent], octree_cell_z_max[cell_id])
    root_ids = tree._root_cell_ids
    xyz_min = np.array(
        [
            float(np.min(octree_cell_x_min[root_ids])),
            float(np.min(octree_cell_y_min[root_ids])),
            float(np.min(octree_cell_z_min[root_ids])),
        ],
        dtype=np.float64,
    )
    xyz_max = np.array(
        [
            float(np.max(octree_cell_x_max[root_ids])),
            float(np.max(octree_cell_y_max[root_ids])),
            float(np.max(octree_cell_z_max[root_ids])),
        ],
        dtype=np.float64,
    )
    cell_bounds = np.empty((n_cells, 3, 2), dtype=np.float64)
    cell_bounds[:, AXIS0, START] = octree_cell_x_min
    cell_bounds[:, AXIS0, WIDTH] = octree_cell_x_max - octree_cell_x_min
    cell_bounds[:, AXIS1, START] = octree_cell_y_min
    cell_bounds[:, AXIS1, WIDTH] = octree_cell_y_max - octree_cell_y_min
    cell_bounds[:, AXIS2, START] = octree_cell_z_min
    cell_bounds[:, AXIS2, WIDTH] = octree_cell_z_max - octree_cell_z_min
    domain_bounds = np.empty((3, 2), dtype=np.float64)
    domain_bounds[:, START] = xyz_min
    domain_bounds[:, WIDTH] = xyz_max - xyz_min
    return cell_bounds, domain_bounds, 0.0, False


def _cartesian_domain_bounds_xyz(tree) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array(tree._domain_bounds[:, START], dtype=float)
    hi = np.array(tree._domain_bounds[:, START] + tree._domain_bounds[:, WIDTH], dtype=float)
    return lo, hi


def _cartesian_domain_bounds_rpa(tree) -> tuple[np.ndarray, np.ndarray]:
    pts = np.column_stack(
        (
            tree.ds[XYZ_VARS[0]],
            tree.ds[XYZ_VARS[1]],
            tree.ds[XYZ_VARS[2]],
        )
    )
    r = np.linalg.norm(pts, axis=1)
    theta = np.arccos(np.clip(pts[:, 2] / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0))
    phi = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * np.pi)
    return (
        np.array([float(np.min(r)), float(np.min(theta)), float(np.min(phi))], dtype=float),
        np.array([float(np.max(r)), float(np.max(theta)), float(np.max(phi))], dtype=float),
    )

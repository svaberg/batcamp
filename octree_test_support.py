from __future__ import annotations

import math

import numpy as np

from batcamp import Octree
from batcamp.constants import XYZ_VARS
from batcamp.octree import AXIS0
from batcamp.octree import AXIS1
from batcamp.octree import AXIS2
from batcamp.octree import START
from batcamp.octree import WIDTH


def cell_bounds(tree: Octree, cell_id: int, *, coord: str = "xyz") -> tuple[np.ndarray, np.ndarray]:
    cell_id = int(cell_id)
    corners = tree.corners[cell_id]
    pts = np.column_stack(
        (
            tree.ds[XYZ_VARS[0]][corners],
            tree.ds[XYZ_VARS[1]][corners],
            tree.ds[XYZ_VARS[2]][corners],
        )
    )
    if coord == "xyz":
        return np.min(pts, axis=0), np.max(pts, axis=0)
    if coord == "rpa" and str(tree.tree_coord) == "rpa":
        lo = np.array(
            [
                tree.cell_bounds[cell_id, AXIS0, START],
                tree.cell_bounds[cell_id, AXIS1, START],
                tree.cell_bounds[cell_id, AXIS2, START],
            ],
            dtype=float,
        )
        hi = np.array(
            [
                tree.cell_bounds[cell_id, AXIS0, START] + tree.cell_bounds[cell_id, AXIS0, WIDTH],
                tree.cell_bounds[cell_id, AXIS1, START] + tree.cell_bounds[cell_id, AXIS1, WIDTH],
                (tree.cell_bounds[cell_id, AXIS2, START] + tree.cell_bounds[cell_id, AXIS2, WIDTH])
                % (2.0 * math.pi),
            ],
            dtype=float,
        )
        return lo, hi
    if coord == "rpa":
        r = np.linalg.norm(pts, axis=1)
        theta = np.arccos(np.clip(pts[:, 2] / np.maximum(r, np.finfo(float).tiny), -1.0, 1.0))
        phi = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * math.pi)
        return (
            np.array([float(np.min(r)), float(np.min(theta)), float(np.min(phi))], dtype=float),
            np.array([float(np.max(r)), float(np.max(theta)), float(np.max(phi))], dtype=float),
        )
    raise ValueError(f"Unsupported coord {coord!r}.")

from __future__ import annotations

import math

import numpy as np

from batcamp import Octree
from batcamp.constants import XYZ_VARS


def cell_bounds(tree: Octree, cell_id: int, *, coord: str = "xyz") -> tuple[np.ndarray, np.ndarray]:
    cid = int(cell_id)
    corners = tree.ds.corners[cid]
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
        lo = np.array([tree._cell_r_min[cid], tree._cell_theta_min[cid], tree._cell_phi_start[cid]], dtype=float)
        hi = np.array(
            [
                tree._cell_r_max[cid],
                tree._cell_theta_max[cid],
                (tree._cell_phi_start[cid] + tree._cell_phi_width[cid]) % (2.0 * math.pi),
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

from __future__ import annotations

import math

import numpy as np

from batcamp import Octree
from batcamp.octree import AXIS0
from batcamp.octree import AXIS1
from batcamp.octree import AXIS2
from batcamp.octree import START
from batcamp.octree import WIDTH


def cell_bounds(tree: Octree, cell_id: int, *, coord: str = "xyz") -> tuple[np.ndarray, np.ndarray]:
    cell_id = int(cell_id)
    if coord == "xyz" and str(tree.tree_coord) == "xyz":
        lo = np.array(tree.cell_bounds[cell_id, :, START], dtype=float)
        hi = np.array(tree.cell_bounds[cell_id, :, START] + tree.cell_bounds[cell_id, :, WIDTH], dtype=float)
        return lo, hi
    if coord == "xyz" and str(tree.tree_coord) == "rpa":
        r0 = float(tree.cell_bounds[cell_id, AXIS0, START])
        r1 = float(tree.cell_bounds[cell_id, AXIS0, START] + tree.cell_bounds[cell_id, AXIS0, WIDTH])
        polar0 = float(tree.cell_bounds[cell_id, AXIS1, START])
        polar1 = float(tree.cell_bounds[cell_id, AXIS1, START] + tree.cell_bounds[cell_id, AXIS1, WIDTH])
        azimuth0 = float(tree.cell_bounds[cell_id, AXIS2, START])
        azimuth_width = float(tree.cell_bounds[cell_id, AXIS2, WIDTH] % (2.0 * math.pi))
        if np.isclose(azimuth_width, 0.0, atol=1e-12):
            azimuth_width = 2.0 * math.pi
        r = 0.5 * (r0 + r1)
        polar = 0.5 * (polar0 + polar1)
        azimuth = (azimuth0 + 0.5 * azimuth_width) % (2.0 * math.pi)
        sin_polar = math.sin(polar)
        xyz_center = np.array(
            [
                r * sin_polar * math.cos(azimuth),
                r * sin_polar * math.sin(azimuth),
                r * math.cos(polar),
            ],
            dtype=float,
        )
        return xyz_center, xyz_center
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
    raise ValueError(f"Unsupported coord {coord!r} for tree_coord={tree.tree_coord!r}.")

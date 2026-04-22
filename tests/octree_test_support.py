from __future__ import annotations

import math

import numpy as np

from batcamp import Octree
from batcamp.octree import TREE_COORD_AXIS0
from batcamp.octree import TREE_COORD_AXIS1
from batcamp.octree import TREE_COORD_AXIS2
from batcamp.octree import BOUNDS_START_SLOT
from batcamp.octree import BOUNDS_WIDTH_SLOT


def cell_bounds(tree: Octree, cell_id: int, *, coord: str = "xyz") -> tuple[np.ndarray, np.ndarray]:
    cell_id = int(cell_id)
    if coord == "xyz" and str(tree.tree_coord) == "xyz":
        lo = np.array(tree.cell_bounds[cell_id, :, BOUNDS_START_SLOT], dtype=float)
        hi = np.array(tree.cell_bounds[cell_id, :, BOUNDS_START_SLOT] + tree.cell_bounds[cell_id, :, BOUNDS_WIDTH_SLOT], dtype=float)
        return lo, hi
    if coord == "xyz" and str(tree.tree_coord) == "rpa":
        r0 = float(tree.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_START_SLOT])
        r1 = float(tree.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_START_SLOT] + tree.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_WIDTH_SLOT])
        polar0 = float(tree.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_START_SLOT])
        polar1 = float(tree.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_START_SLOT] + tree.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_WIDTH_SLOT])
        azimuth0 = float(tree.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_START_SLOT])
        azimuth_width = float(tree.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT] % (2.0 * math.pi))
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
                tree.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_START_SLOT],
                tree.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_START_SLOT],
                tree.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_START_SLOT],
            ],
            dtype=float,
        )
        hi = np.array(
            [
                tree.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_START_SLOT] + tree.cell_bounds[cell_id, TREE_COORD_AXIS0, BOUNDS_WIDTH_SLOT],
                tree.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_START_SLOT] + tree.cell_bounds[cell_id, TREE_COORD_AXIS1, BOUNDS_WIDTH_SLOT],
                (tree.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_START_SLOT] + tree.cell_bounds[cell_id, TREE_COORD_AXIS2, BOUNDS_WIDTH_SLOT])
                % (2.0 * math.pi),
            ],
            dtype=float,
        )
        return lo, hi
    raise ValueError(f"Unsupported coord {coord!r} for tree_coord={tree.tree_coord!r}.")


def find_shared_face_pair(
    tree: Octree,
    cell_ids: np.ndarray,
    *,
    face_axis: int,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return one pair of candidate cells that share one full interior face orthogonal to `face_axis`."""
    cached_bounds = {int(cell_id): cell_bounds(tree, int(cell_id), coord="rpa") for cell_id in cell_ids.tolist()}
    axes = [0, 1, 2]
    axes.remove(int(face_axis))
    for left_id in cell_ids.tolist():
        lo_left, hi_left = cached_bounds[int(left_id)]
        for right_id in cell_ids.tolist():
            if int(left_id) == int(right_id):
                continue
            lo_right, hi_right = cached_bounds[int(right_id)]
            if not np.allclose(lo_left[axes], lo_right[axes], atol=1e-12, rtol=0.0):
                continue
            if not np.allclose(hi_left[axes], hi_right[axes], atol=1e-12, rtol=0.0):
                continue
            if not np.isclose(
                float(hi_left[int(face_axis)]),
                float(lo_right[int(face_axis)]),
                atol=1e-12,
                rtol=0.0,
            ):
                continue
            return int(left_id), int(right_id), lo_left, hi_left
    raise AssertionError(f"No shared face pair found for axis {int(face_axis)}.")

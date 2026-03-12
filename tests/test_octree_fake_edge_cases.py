from __future__ import annotations

import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeInterpolator
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh


def _build_regular_fake_dataset(
    *,
    nr: int = 1,
    ntheta: int = 2,
    nphi: int = 4,
) -> _FakeDataset:
    """Private test helper: build a regular spherical dataset for edge-case tests."""
    points, corners = _build_spherical_hex_mesh(
        nr=nr,
        ntheta=ntheta,
        nphi=nphi,
        r_min=1.0,
        r_max=2.0,
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    scalar = 2.0 * x - 1.0 * y + 0.5 * z + 7.0
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Scalar": scalar,
        },
    )


def _build_axis_only_fake_dataset() -> _FakeDataset:
    """Private test helper: build an invalid dataset with all corners on axis."""
    z = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    points = np.column_stack((x, y, z))
    corners = np.arange(8, dtype=np.int64).reshape(1, 8)
    scalar = 0.25 * z + 1.0
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Scalar": scalar,
        },
    )


def test_fake_axis_only_dataset_is_rejected() -> None:
    """Axis-only cells should fail octree build because no valid phi levels exist."""
    ds = _build_axis_only_fake_dataset()
    with pytest.raises(ValueError, match="No valid \\(>=0\\) levels"):
        Octree.from_dataset(ds, tree_coord="rpa")


def test_fake_lookup_rejects_invalid_queries() -> None:
    """Lookup should return None for non-finite or invalid-angle queries."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")

    assert tree.lookup_point(np.array([float("nan"), 0.0, 0.0], dtype=float), coord="xyz") is None
    assert tree.lookup_point(np.array([float("inf"), 0.0, 0.0], dtype=float), coord="xyz") is None
    assert tree.lookup_point(np.array([1.5, -1e-6, 0.0], dtype=float), coord="rpa") is None
    assert tree.lookup_point(np.array([1.5, math.pi + 1e-6, 0.0], dtype=float), coord="rpa") is None
    assert tree.lookup_point(np.array([float("nan"), 1.0, 0.0], dtype=float), coord="rpa") is None


def test_fake_interpolator_fill_for_invalid_points() -> None:
    """Interpolator should emit fill value and cell_id=-1 for invalid queries."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree, fill_value=-123.0)

    invalid = np.array(
        [
            [float("nan"), 0.0, 0.0],
            [float("inf"), 0.0, 0.0],
            [-float("inf"), 0.0, 0.0],
        ]
    )
    q = invalid

    vals, cids = interp(q, return_cell_ids=True)
    assert np.all(cids == -1)
    assert np.allclose(vals, -123.0, atol=0.0, rtol=0.0)


def test_fake_bind_without_corners_is_rejected() -> None:
    """Binding a tree to a dataset with missing corners should fail clearly."""
    ds = _build_regular_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    ds_no_corners = _FakeDataset(ds.points, None, ds._variables)

    with pytest.raises(ValueError, match="Dataset has no corners"):
        tree.bind(ds_no_corners)

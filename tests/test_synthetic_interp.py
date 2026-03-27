from __future__ import annotations

import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh
from octree_test_support import cell_bounds


def _build_uniform_spherical_hex_dataset(
    *,
    nr: int = 2,
    npolar: int = 4,
    nazimuth: int = 8,
) -> tuple[_FakeDataset, np.ndarray, tuple[float, float, float, float]]:
    """Private test helper: build a synthetic full-sphere hexahedral dataset."""
    points, corners = _build_spherical_hex_mesh(
        nr=nr,
        npolar=npolar,
        nazimuth=nazimuth,
        r_min=1.0,
        r_max=3.0,
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r_nodes = np.sqrt(x * x + y * y + z * z)
    polar_nodes = np.arccos(np.clip(z / np.maximum(r_nodes, np.finfo(float).tiny), -1.0, 1.0))
    azimuth_nodes = np.mod(np.arctan2(y, x), 2.0 * math.pi)

    a, b, c, d = (1.7, -0.45, 0.3, 2.1)
    linear_field = a * r_nodes + b * polar_nodes + c * azimuth_nodes + d
    linear_field2 = 2.0 * linear_field + 1.0
    linear_const = np.full_like(linear_field, 5.0)

    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "LinField": linear_field,
            "LinField2": linear_field2,
            "LinFieldConst": linear_const,
        },
    )
    return ds, linear_field, (a, b, c, d)


@pytest.fixture(scope="module")
def synthetic_context() -> tuple[_FakeDataset, Octree, np.ndarray, tuple[float, float, float, float]]:
    """Return synthetic dataset, built tree, linear nodal field and coefficients."""
    ds, linear_field, coeffs = _build_uniform_spherical_hex_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    return ds, tree, linear_field, coeffs


def _sample_inside_cells(tree: Octree, cell_ids: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Private test helper: sample one interior xyz/rpa point per selected cell."""
    xyz_list: list[np.ndarray] = []
    rpa_list: list[np.ndarray] = []
    for cell_id in cell_ids.tolist():
        lo, hi = cell_bounds(tree, int(cell_id), coord="rpa")
        r0 = float(lo[0])
        r1 = float(hi[0])
        t0 = float(lo[1])
        t1 = float(hi[1])
        p0 = float(lo[2])
        pw = float((hi[2] - lo[2]) % (2.0 * math.pi))
        if np.isclose(pw, 0.0, atol=1e-12):
            pw = 2.0 * math.pi

        u = float(rng.uniform(0.15, 0.85))
        v = float(rng.uniform(0.15, 0.85))
        w = float(rng.uniform(0.15, 0.85))
        r = r0 + u * (r1 - r0)
        polar = t0 + v * (t1 - t0)
        azimuth = (p0 + w * pw) % (2.0 * math.pi)

        st = math.sin(polar)
        xyz = np.array([r * st * math.cos(azimuth), r * st * math.sin(azimuth), r * math.cos(polar)])
        xyz_list.append(xyz)
        rpa_list.append(np.array([r, polar, azimuth]))
    return np.array(xyz_list), np.array(rpa_list)


def _midpoints_xyz(tree: Octree, cell_ids: np.ndarray) -> np.ndarray:
    xyz_list: list[np.ndarray] = []
    for cell_id in cell_ids.tolist():
        lo, hi = cell_bounds(tree, int(cell_id), coord="rpa")
        r = 0.5 * (float(lo[0]) + float(hi[0]))
        polar = 0.5 * (float(lo[1]) + float(hi[1]))
        azimuth0 = float(lo[2])
        azimuth_width = float((hi[2] - lo[2]) % (2.0 * math.pi))
        if np.isclose(azimuth_width, 0.0, atol=1e-12):
            azimuth_width = 2.0 * math.pi
        azimuth = (azimuth0 + 0.5 * azimuth_width) % (2.0 * math.pi)
        st = math.sin(polar)
        xyz_list.append(np.array([r * st * math.cos(azimuth), r * st * math.sin(azimuth), r * math.cos(polar)]))
    return np.asarray(xyz_list, dtype=float)


def _interpolation_valid_cells(
    tree: Octree,
    *,
    interp: OctreeInterpolator | None = None,
) -> np.ndarray:
    """Private test helper: return cells suitable for stable interpolation checks."""
    n_cells = int(tree.cell_count)
    lo = np.empty((n_cells, 3), dtype=float)
    hi = np.empty((n_cells, 3), dtype=float)
    for cell_id in range(n_cells):
        lo[cell_id], hi[cell_id] = cell_bounds(tree, cell_id, coord="rpa")
    azimuth_end = hi[:, 2]
    ids = np.flatnonzero(
        (lo[:, 1] > 1e-6)
        & (hi[:, 1] < (math.pi - 1e-6))
        & (azimuth_end < (2.0 * math.pi - 1e-8))
    )
    if interp is None:
        return ids
    good = [int(cell_id) for cell_id in ids.tolist() if np.unique(interp.tree.corners[int(cell_id)]).size == 8]
    return np.array(good, dtype=np.int64)


def test_lookup_hits_cell_midpoints(synthetic_context) -> None:
    """Lookup of each synthetic cell midpoint should return the corresponding cell id."""
    _ds, tree, _field, _coeffs = synthetic_context
    queries = _midpoints_xyz(tree, np.arange(tree.cell_count, dtype=np.int64))
    for cell_id in range(queries.shape[0]):
        q = queries[cell_id]
        assert int(tree.lookup_points(q, coord="xyz")[0]) == int(cell_id)


def test_lookup_xyz_rpa_match_random_points(synthetic_context) -> None:
    """xyz and rpa lookups should agree on synthetic interior random points."""
    _ds, tree, _field, _coeffs = synthetic_context
    rng = np.random.default_rng(11)
    valid = _interpolation_valid_cells(tree)
    choose = rng.choice(valid, size=min(120, valid.size), replace=False)
    xyz, rpa = _sample_inside_cells(tree, choose, rng)

    for i in range(xyz.shape[0]):
        cell_id_xyz = int(tree.lookup_points(xyz[i], coord="xyz")[0])
        cell_id_rpa = int(tree.lookup_points(rpa[i], coord="rpa")[0])
        assert cell_id_xyz >= 0
        assert cell_id_rpa >= 0
        assert cell_id_xyz == cell_id_rpa


def test_interp_matches_linear_field_xyz(synthetic_context) -> None:
    """xyz interpolation should reconstruct an exactly linear spherical field."""
    ds, tree, _field, coeffs = synthetic_context
    a, b, c, d = coeffs
    interp = OctreeInterpolator(tree, np.asarray(ds["LinField"]))
    rng = np.random.default_rng(22)
    valid = _interpolation_valid_cells(tree, interp=interp)
    assert valid.size > 0
    choose = rng.choice(valid, size=min(200, valid.size), replace=False)
    xyz, rpa = _sample_inside_cells(tree, choose, rng)
    vals, cell_ids = interp(xyz, return_cell_ids=True)
    expected = a * rpa[:, 0] + b * rpa[:, 1] + c * rpa[:, 2] + d

    assert np.array_equal(cell_ids, choose)
    assert np.allclose(vals, expected, atol=1e-9, rtol=0.0)


def test_interp_matches_linear_field_rpa_wrap(synthetic_context) -> None:
    """rpa interpolation should normalize wrapped azimuth and match linear field."""
    ds, tree, _field, coeffs = synthetic_context
    a, b, c, d = coeffs
    interp = OctreeInterpolator(tree, np.asarray(ds["LinField"]))
    rng = np.random.default_rng(33)
    valid = _interpolation_valid_cells(tree, interp=interp)
    assert valid.size > 0
    choose = rng.choice(valid, size=min(200, valid.size), replace=False)
    _xyz, rpa = _sample_inside_cells(tree, choose, rng)
    wrapped = np.array(rpa, copy=True)
    wrapped[:, 2] += 2.0 * math.pi
    vals, cell_ids = interp(wrapped, query_coord="rpa", return_cell_ids=True)
    expected = a * rpa[:, 0] + b * rpa[:, 1] + c * rpa[:, 2] + d

    assert np.array_equal(cell_ids, choose)
    assert np.allclose(vals, expected, atol=1e-9, rtol=0.0)


def test_vector_interp_shape_and_values(synthetic_context) -> None:
    """Vector-valued interpolation should preserve trailing dims and nodal exactness."""
    ds, tree, _linear_field, coeffs = synthetic_context
    a, b, c, d = coeffs
    interp = OctreeInterpolator(
        tree,
        np.column_stack((np.asarray(ds["LinField"]), np.asarray(ds["LinField2"]), np.asarray(ds["LinFieldConst"]))),
    )

    rng = np.random.default_rng(44)
    valid = _interpolation_valid_cells(tree, interp=interp)
    assert valid.size > 0
    choose = rng.choice(valid, size=min(120, valid.size), replace=False)
    xyz, rpa = _sample_inside_cells(tree, choose, rng)
    vals = interp(xyz)

    scalar = a * rpa[:, 0] + b * rpa[:, 1] + c * rpa[:, 2] + d
    expected = np.column_stack((scalar, 2.0 * scalar + 1.0, np.full_like(scalar, 5.0)))

    assert vals.shape == expected.shape
    assert np.allclose(vals, expected, atol=1e-9, rtol=0.0)


def test_outside_points_use_fill_and_negative_cell_id(synthetic_context) -> None:
    """Outside-domain synthetic points should return fill value and cell_id=-1."""
    ds, tree, _field, _coeffs = synthetic_context
    fill = -999.0
    interp = OctreeInterpolator(tree, np.asarray(ds["LinField"]), fill_value=fill)

    inside = _midpoints_xyz(tree, np.array([0], dtype=np.int64))
    outside = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    q = np.vstack((inside, outside))

    vals, cell_ids = interp(q, return_cell_ids=True)
    assert int(cell_ids[0]) >= 0
    assert int(cell_ids[1]) == -1
    assert int(cell_ids[2]) == -1
    assert np.isclose(float(vals[1]), fill)
    assert np.isclose(float(vals[2]), fill)

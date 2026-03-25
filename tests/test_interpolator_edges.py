from __future__ import annotations

import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS
from batcamp.octree import _lookup_cell_id_kernel
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh
from octree_test_support import cell_bounds


def _build_fake_dataset(
    *,
    nr: int = 1,
    ntheta: int = 2,
    nphi: int = 4,
) -> _FakeDataset:
    """Private test helper: build a small regular spherical hexahedral dataset."""
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
    scalar = 3.0 * x - 2.0 * y + 0.5 * z + 1.0
    scalar2 = 2.0 * scalar + 3.0
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Scalar": scalar,
            "Scalar2": scalar2,
        },
    )


def _build_fake_cartesian_dataset() -> _FakeDataset:
    """Private test helper: build a small regular Cartesian hexahedral dataset."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([-0.5, 0.5], dtype=float),
        z_edges=np.array([-0.25, 0.75], dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    scalar = 2.5 * x - 1.25 * y + 0.75 * z + 3.0
    scalar2 = -0.5 * scalar + 2.0
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Scalar": scalar,
            "Scalar2": scalar2,
        },
    )



def _first_resolvable_rpa(tree: Octree) -> tuple[float, float, float]:
    """Private test helper: return one interior spherical point resolved by lookup."""
    for cid in range(int(tree.cell_count)):
        lo, hi = cell_bounds(tree, int(cid), coord="rpa")
        r = 0.5 * (float(lo[0]) + float(hi[0]))
        polar = 0.5 * (float(lo[1]) + float(hi[1]))
        width = float((hi[2] - lo[2]) % (2.0 * math.pi))
        if np.isclose(width, 0.0, atol=1e-12):
            width = 2.0 * math.pi
        azimuth = (float(lo[2]) + 0.4 * width) % (2.0 * math.pi)
        hit = tree.lookup_point(np.array([r, polar, azimuth], dtype=float), coord="rpa")
        if hit is not None:
            return r, polar, azimuth
    raise AssertionError("No resolvable interior rpa point found in fake dataset.")

def test_constructor_rejects_query_coord_kw() -> None:
    """Constructor no longer accepts query_coord; it is call-time only."""
    ds = _build_fake_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    with pytest.raises(TypeError, match="unexpected keyword argument 'query_coord'"):
        OctreeInterpolator(tree, ["Scalar"], query_coord="bad")

def test_constructor_rejects_non_list_values() -> None:
    """Constructor should enforce tree-first values contracts."""
    ds = _build_fake_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    bad_values = np.ones(ds.points.shape[0] - 1)
    with pytest.raises(ValueError, match="values length"):
        OctreeInterpolator(tree, bad_values)
    with pytest.raises(ValueError, match="single-string values are not supported"):
        OctreeInterpolator(tree, "Scalar")

def test_default_octree_build_selects_spherical() -> None:
    """Default tree inference should select spherical lookup for non-axis-aligned cells."""
    ds = _build_fake_dataset()
    assert OctreeBuilder().build(ds).tree_coord == "rpa"

def test_default_octree_build_selects_cartesian() -> None:
    """Default tree inference should select Cartesian lookup for axis-aligned cells."""
    ds = _build_fake_cartesian_dataset()
    assert OctreeBuilder().build(ds).tree_coord == "xyz"

def test_interpolator_does_not_stash_cartesian_lookup_state() -> None:
    """Interpolator should not keep a separate Cartesian lookup-state copy."""
    ds = _build_fake_cartesian_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="xyz")

    interp = OctreeInterpolator(tree, ["Scalar"])
    assert not hasattr(interp, "_lookup_state_xyz")

def test_interpolator_does_not_stash_spherical_lookup_state() -> None:
    """Interpolator should not keep a separate spherical lookup-state copy."""
    ds = _build_fake_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="rpa")

    interp = OctreeInterpolator(tree, ["Scalar"])
    assert not hasattr(interp, "_lookup_state_rpa")

def test_cartesian_lookup_reuses_ancestor_from_previous_cell() -> None:
    """Cartesian lookup should climb ancestors from `prev_cid` instead of requiring root restart."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
        y_edges=np.array([-1.0, 0.0, 1.0], dtype=float),
        z_edges=np.array([-1.0, 0.0, 1.0], dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Scalar": x + y + z,
        },
    )
    tree = OctreeBuilder().build(ds, tree_coord="xyz")
    lookup_state = tree._coord_state

    q0 = np.array([0.5, -0.5, -0.5], dtype=float)
    q1 = np.array([1.5, -0.5, -0.5], dtype=float)
    hit0 = tree.lookup_point(q0, coord="xyz")
    hit1 = tree.lookup_point(q1, coord="xyz")
    assert hit0 is not None
    assert hit1 is not None
    assert int(hit0.cell_id) != int(hit1.cell_id)

    state_no_roots = lookup_state._replace(root_node_ids=np.empty((0,), dtype=np.int64))
    cid = _lookup_cell_id_kernel(float(q1[0]), float(q1[1]), float(q1[2]), state_no_roots, int(hit0.cell_id))
    assert int(cid) == int(hit1.cell_id)

def test_spherical_lookup_reuses_ancestor_from_previous_cell() -> None:
    """Spherical lookup should climb ancestors from `prev_cid` instead of requiring root restart."""
    ds = _build_fake_dataset(nr=2, ntheta=4, nphi=8)
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    lookup_state = tree._coord_state

    lo0, hi0 = cell_bounds(tree, 0, coord="rpa")
    lo1, hi1 = cell_bounds(tree, 1, coord="rpa")
    q0 = np.array([
        0.5 * (float(lo0[0]) + float(hi0[0])),
        0.5 * (float(lo0[1]) + float(hi0[1])),
        (float(lo0[2]) + 0.4 * float((hi0[2] - lo0[2]) % (2.0 * math.pi) or 2.0 * math.pi)) % (2.0 * math.pi),
    ], dtype=float)
    q1 = np.array([
        0.5 * (float(lo1[0]) + float(hi1[0])),
        0.5 * (float(lo1[1]) + float(hi1[1])),
        (float(lo1[2]) + 0.4 * float((hi1[2] - lo1[2]) % (2.0 * math.pi) or 2.0 * math.pi)) % (2.0 * math.pi),
    ], dtype=float)
    hit0 = tree.lookup_point(q0, coord="rpa")
    hit1 = tree.lookup_point(q1, coord="rpa")
    assert hit0 is not None
    assert hit1 is not None
    assert int(hit0.cell_id) != int(hit1.cell_id)

    state_no_roots = lookup_state._replace(root_node_ids=np.empty((0,), dtype=np.int64))
    cid = _lookup_cell_id_kernel(float(q1[0]), float(q1[1]), float(q1[2]), state_no_roots, int(hit0.cell_id))
    assert int(cid) == int(hit1.cell_id)

def test_call_rejects_invalid_query_coord() -> None:
    """Runtime call should reject invalid query_coord override."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(OctreeBuilder().build(ds, tree_coord="rpa"), ["Scalar"])
    with pytest.raises(ValueError, match="query_(coord|space) must be 'xyz' or 'rpa'"):
        interp(np.array([[1.0, 0.0, 0.0]]), query_coord="bad")

def test_supports_xyz_and_rpa_query_coords() -> None:
    """Interpolator should support both xyz and rpa query coordinates on spherical trees."""
    ds = _build_fake_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    interp = OctreeInterpolator(tree, ["Scalar"])
    r, polar, azimuth = _first_resolvable_rpa(interp.tree)
    q_rpa = np.array([[r, polar, azimuth]], dtype=float)
    q = np.array(
        [[
            r * math.sin(polar) * math.cos(azimuth),
            r * math.sin(polar) * math.sin(azimuth),
            r * math.cos(polar),
        ]],
        dtype=float,
    )
    vals_a, cids_a = interp(q, query_coord="xyz", return_cell_ids=True)
    vals_b, cids_b = interp(q_rpa, query_coord="rpa", return_cell_ids=True)
    assert np.array_equal(cids_a, cids_b)
    assert np.allclose(vals_a, vals_b, atol=0.0, rtol=0.0, equal_nan=True)

def test_prepare_queries_validation() -> None:
    """`prepare_queries` should enforce valid tuple/shape conventions."""
    with pytest.raises(ValueError, match="Tuple input must have exactly 3 arrays"):
        OctreeInterpolator.prepare_queries((np.array([1.0]), np.array([2.0])))
    with pytest.raises(ValueError, match="1D xi must have length 3"):
        OctreeInterpolator.prepare_queries(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="xi must have shape"):
        OctreeInterpolator.prepare_queries(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="Call with xi or with x1, x2, x3"):
        OctreeInterpolator.prepare_queries(np.array([1.0]), np.array([2.0]))

def test_outside_queries_use_fill_and_minus_one() -> None:
    """Outside-domain queries should return fill values and `cell_id=-1`."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(OctreeBuilder().build(ds, tree_coord="rpa"), ["Scalar"], fill_value=-77.0)
    q = np.array(
        [
            [1e6, 0.0, 0.0],
            [-1e6, 0.0, 0.0],
        ]
    )
    vals, cids = interp(q, return_cell_ids=True)
    assert np.all(cids == -1)
    assert np.allclose(vals, -77.0, atol=0.0, rtol=0.0)

def test_vector_fill_applies_outside_domain() -> None:
    """Vector-valued fill should broadcast correctly for outside-domain queries."""
    ds = _build_fake_dataset()
    fill = np.array([-5.0, 8.0])
    interp = OctreeInterpolator(OctreeBuilder().build(ds, tree_coord="rpa"), ["Scalar", "Scalar2"], fill_value=fill)
    q = np.array([[1e6, 0.0, 0.0]])
    vals, cids = interp(q, return_cell_ids=True)
    assert vals.shape == (1, 2)
    assert int(cids[0]) == -1
    assert np.allclose(vals[0], fill, atol=0.0, rtol=0.0)

def test_rpa_wrap_equivalence() -> None:
    """`rpa` interpolation should treat azimuth `phi` and `phi + 2pi` equivalently."""
    ds = _build_fake_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    interp = OctreeInterpolator(tree, ["Scalar"])
    r, polar, azimuth = _first_resolvable_rpa(tree)
    q0 = np.array([[r, polar, azimuth]])
    q1 = np.array([[r, polar, azimuth + 2.0 * math.pi]])
    v0, c0 = interp(q0, query_coord="rpa", return_cell_ids=True)
    v1, c1 = interp(q1, query_coord="rpa", return_cell_ids=True)
    assert np.array_equal(c0, c1)
    assert np.allclose(v0, v1, atol=1e-12, rtol=0.0)

def test_invalid_level_cells_treated_as_misses() -> None:
    """Lookup and interpolation should both treat level<0 cells as invalid."""
    ds = _build_fake_cartesian_dataset()
    levels = np.array([-1, 0], dtype=np.int64)
    tree = OctreeBuilder()._build(
        ds,
        tree_coord="xyz",
        cell_levels=levels,
    )

    q_invalid = np.array([0.5, 0.0, 0.25], dtype=float)
    q_valid = np.array([1.5, 0.0, 0.25], dtype=float)

    assert tree.lookup_point(q_invalid, coord="xyz") is None
    assert tree.lookup_point(q_valid, coord="xyz") is not None

    interp = OctreeInterpolator(tree, ["Scalar"], fill_value=-123.0)
    vals, cids = interp(np.vstack((q_invalid, q_valid)), return_cell_ids=True)

    assert int(cids[0]) == -1
    assert int(cids[1]) >= 0
    assert np.isclose(float(vals[0]), -123.0, atol=0.0, rtol=0.0)

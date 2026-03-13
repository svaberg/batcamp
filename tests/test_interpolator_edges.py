from __future__ import annotations

import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp import OctreeInterpolator
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh


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
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
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
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Scalar": scalar,
            "Scalar2": scalar2,
        },
    )



def _first_resolvable_rpa(tree: Octree) -> tuple[float, float, float]:
    """Private test helper: return one interior spherical point resolved by lookup."""
    for cid in range(int(tree.cell_count)):
        lo, hi = tree.cell_bounds(int(cid), coord="rpa")
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
    with pytest.raises(TypeError, match="unexpected keyword argument 'query_coord'"):
        OctreeInterpolator(ds, ["Scalar"], query_coord="bad")

def test_constructor_rejects_missing_corners() -> None:
    """Constructor should fail when dataset has no corner connectivity."""
    ds = _build_fake_dataset()
    ds_bad = _FakeDataset(ds.points, None, ds._variables)
    with pytest.raises(ValueError, match="Dataset has no cell connectivity"):
        OctreeInterpolator(ds_bad, ["Scalar"])

def test_constructor_rejects_non_list_values() -> None:
    """Constructor should enforce `values=None` or `values=list[str]`."""
    ds = _build_fake_dataset()
    bad_values = np.ones(ds.points.shape[0] - 1)
    with pytest.raises(ValueError, match="values must be None or"):
        OctreeInterpolator(ds, bad_values)
    with pytest.raises(ValueError, match="single-string values are not supported"):
        OctreeInterpolator(ds, "Scalar")

def test_auto_tree_coord_selects_spherical() -> None:
    """Auto coord-system should select spherical lookup for non-axis-aligned cells."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"], tree=None)
    assert interp.tree.tree_coord == "rpa"

def test_auto_tree_coord_selects_cartesian() -> None:
    """Auto coord-system should select Cartesian lookup for axis-aligned cells."""
    ds = _build_fake_cartesian_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"], tree=None)
    assert interp.tree.tree_coord == "xyz"

def test_reuses_prebuilt_cartesian_lookup() -> None:
    """Interpolator should reuse the same prebuilt tree/lookup object."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    lookup_before = tree.lookup

    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    assert tree.lookup is lookup_before
    assert interp.lookup is lookup_before

def test_reuses_prebuilt_spherical_lookup() -> None:
    """Interpolator should reuse the same prebuilt tree/lookup object."""
    ds = _build_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    lookup_before = tree.lookup

    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    assert tree.lookup is lookup_before
    assert interp.lookup is lookup_before

def test_call_rejects_invalid_query_coord() -> None:
    """Runtime call should reject invalid query_coord override."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"])
    with pytest.raises(ValueError, match="query_(coord|space) must be 'xyz' or 'rpa'"):
        interp(np.array([[1.0, 0.0, 0.0]]), query_coord="bad")

def test_supports_query_and_tree_coord_names() -> None:
    """New API names should work directly for constructor and call override."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"], tree_coord="rpa")
    r, polar, azimuth = _first_resolvable_rpa(interp.tree)
    q = np.array(
        [[
            r * math.sin(polar) * math.cos(azimuth),
            r * math.sin(polar) * math.sin(azimuth),
            r * math.cos(polar),
        ]],
        dtype=float,
    )
    vals_a, cids_a = interp(q, return_cell_ids=True)
    vals_b, cids_b = interp(q, return_cell_ids=True)
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
    interp = OctreeInterpolator(ds, ["Scalar"], fill_value=-77.0)
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
    interp = OctreeInterpolator(ds, ["Scalar", "Scalar2"], fill_value=fill)
    q = np.array([[1e6, 0.0, 0.0]])
    vals, cids = interp(q, return_cell_ids=True)
    assert vals.shape == (1, 2)
    assert int(cids[0]) == -1
    assert np.allclose(vals[0], fill, atol=0.0, rtol=0.0)

def test_rpa_wrap_equivalence() -> None:
    """`rpa` interpolation should treat azimuth `phi` and `phi + 2pi` equivalently."""
    ds = _build_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
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
        bind=False,
    )
    tree.bind(ds)

    q_invalid = np.array([0.5, 0.0, 0.25], dtype=float)
    q_valid = np.array([1.5, 0.0, 0.25], dtype=float)

    assert tree.lookup_point(q_invalid, coord="xyz") is None
    assert tree.lookup_point(q_valid, coord="xyz") is not None

    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree, fill_value=-123.0)
    vals, cids = interp(np.vstack((q_invalid, q_valid)), return_cell_ids=True)

    assert int(cids[0]) == -1
    assert int(cids[1]) >= 0
    assert np.isclose(float(vals[0]), -123.0, atol=0.0, rtol=0.0)


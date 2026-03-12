from __future__ import annotations

import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp import OctreeInterpolator
from batcamp import OctreeRayInterpolator
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


def _first_resolvable_center(tree: Octree) -> np.ndarray:
    """Private test helper: return first cell center that resolves via lookup."""
    for c in np.array(tree.cell_centers, dtype=float):
        hit = tree.lookup_point(np.array(c, dtype=float), coord="xyz")
        if hit is not None:
            return np.array(c, dtype=float)
    raise AssertionError("No resolvable center found in fake dataset.")


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


def test_interpolator_constructor_rejects_query_coord_keyword() -> None:
    """Constructor no longer accepts query_coord; it is call-time only."""
    ds = _build_fake_dataset()
    with pytest.raises(TypeError, match="unexpected keyword argument 'query_coord'"):
        OctreeInterpolator(ds, ["Scalar"], query_coord="bad")


def test_interpolator_constructor_rejects_missing_corners() -> None:
    """Constructor should fail when dataset has no corner connectivity."""
    ds = _build_fake_dataset()
    ds_bad = _FakeDataset(ds.points, None, ds._variables)
    with pytest.raises(ValueError, match="Dataset has no cell connectivity"):
        OctreeInterpolator(ds_bad, ["Scalar"])


def test_interpolator_constructor_rejects_non_list_values() -> None:
    """Constructor should enforce `values=None` or `values=list[str]`."""
    ds = _build_fake_dataset()
    bad_values = np.ones(ds.points.shape[0] - 1)
    with pytest.raises(ValueError, match="values must be None or"):
        OctreeInterpolator(ds, bad_values)
    with pytest.raises(ValueError, match="single-string values are not supported"):
        OctreeInterpolator(ds, "Scalar")


def test_interpolator_auto_tree_coord_selects_spherical_for_curvilinear_cells() -> None:
    """Auto coord-system should select spherical lookup for non-axis-aligned cells."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"], tree=None)
    assert interp.tree.tree_coord == "rpa"


def test_interpolator_auto_tree_coord_selects_cartesian_for_axis_aligned_cells() -> None:
    """Auto coord-system should select Cartesian lookup for axis-aligned cells."""
    ds = _build_fake_cartesian_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"], tree=None)
    assert interp.tree.tree_coord == "xyz"


def test_interpolator_reuses_prebuilt_cartesian_lookup_for_same_dataset() -> None:
    """Interpolator should reuse the same prebuilt tree/lookup object."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    lookup_before = tree.lookup

    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    assert tree.lookup is lookup_before
    assert interp.lookup is lookup_before


def test_interpolator_reuses_prebuilt_spherical_lookup_for_same_dataset() -> None:
    """Interpolator should reuse the same prebuilt tree/lookup object."""
    ds = _build_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    lookup_before = tree.lookup

    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    assert tree.lookup is lookup_before
    assert interp.lookup is lookup_before


def test_interpolator_call_rejects_invalid_query_coord_override() -> None:
    """Runtime call should reject invalid query_coord override."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"])
    with pytest.raises(ValueError, match="query_(coord|space) must be 'xyz' or 'rpa'"):
        interp(np.array([[1.0, 0.0, 0.0]]), query_coord="bad")


def test_interpolator_supports_query_coord_and_tree_coord_names() -> None:
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


def test_prepare_queries_validation_errors() -> None:
    """`prepare_queries` should enforce valid tuple/shape conventions."""
    with pytest.raises(ValueError, match="Tuple input must have exactly 3 arrays"):
        OctreeInterpolator.prepare_queries((np.array([1.0]), np.array([2.0])))
    with pytest.raises(ValueError, match="1D xi must have length 3"):
        OctreeInterpolator.prepare_queries(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="xi must have shape"):
        OctreeInterpolator.prepare_queries(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="Call with xi or with x1, x2, x3"):
        OctreeInterpolator.prepare_queries(np.array([1.0]), np.array([2.0]))


def test_sample_ray_xyz_rejects_bad_arguments() -> None:
    """Ray sampling should reject non-positive sample count and zero direction."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"])
    with pytest.raises(ValueError, match="n_samples must be positive"):
        OctreeRayInterpolator(interp).sample(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0, 1.0, 0)
    with pytest.raises(ValueError, match="direction_xyz must be finite and non-zero"):
        OctreeRayInterpolator(interp).sample(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.0, 1.0, 10)


def test_ray_linear_pieces_rejects_zero_direction() -> None:
    """Piecewise linear ray decomposition should reject zero direction."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"])
    with pytest.raises(ValueError, match="direction_xyz must be finite and non-zero"):
        OctreeRayInterpolator(interp).linear_pieces(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.0, 1.0)


def test_integrate_field_along_rays_rejects_bad_arguments() -> None:
    """Bulk ray integration should validate origin shape, chunk size and interval."""
    ds = _build_fake_dataset()
    interp = OctreeInterpolator(ds, ["Scalar"])
    ray = OctreeRayInterpolator(interp)
    with pytest.raises(ValueError, match="origins_xyz must have shape"):
        ray.integrate_field_along_rays(np.array([1.0, 2.0]), np.array([1.0, 0.0, 0.0]), 0.0, 1.0)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ray.integrate_field_along_rays(np.array([[1.0, 0.0, 0.0]]), np.array([1.0, 0.0, 0.0]), 0.0, 1.0, chunk_size=0)
    with pytest.raises(ValueError, match="t_end must be greater than t_start"):
        ray.integrate_field_along_rays(np.array([[1.0, 0.0, 0.0]]), np.array([1.0, 0.0, 0.0]), 1.0, 1.0)


def test_integrate_field_along_rays_matches_linear_piece_integral() -> None:
    """Bulk integral should match per-ray linear-piece integration on axis-aligned rays."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = tree.domain_bounds(coord="xyz")
    xmin = float(dmin[0])
    xmax = float(dmax[0])
    yc = 0.5 * float(dmin[1] + dmax[1])
    zc = 0.5 * float(dmin[2] + dmax[2])
    y_span = 0.2 * float(dmax[1] - dmin[1])
    z_span = 0.2 * float(dmax[2] - dmin[2])

    origins = np.array(
        [
            [xmin, yc - y_span, zc - z_span],
            [xmin, yc, zc],
            [xmin, yc + y_span, zc + z_span],
        ],
        dtype=float,
    )
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    t0 = 0.0
    t1 = xmax - xmin

    bulk = np.asarray(
        ray.integrate_field_along_rays(origins, direction, t0, t1, chunk_size=2),
        dtype=float,
    )

    expected = np.empty(origins.shape[0], dtype=float)
    for i, origin in enumerate(origins):
        pieces = ray.linear_pieces(origin, direction, t0, t1)
        col = 0.0
        for seg in pieces:
            a = float(seg.slope)
            b = float(seg.intercept)
            ta = float(seg.t_start)
            tb = float(seg.t_end)
            col += 0.5 * a * (tb * tb - ta * ta) + b * (tb - ta)
        expected[i] = col

    assert np.allclose(bulk, expected, atol=1e-6, rtol=1e-9)


def test_cartesian_boundary_start_outward_ray_has_no_long_interior_path() -> None:
    """Boundary-start outward rays should not trace/integrate as interior paths."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = tree.domain_bounds(coord="xyz")
    origin = np.array(
        [
            float(dmin[0]),
            0.5 * float(dmin[1] + dmax[1]),
            0.5 * float(dmin[2] + dmax[2]),
        ],
        dtype=float,
    )
    direction = np.array([-1.0, 1.0e-6, 0.0], dtype=float)
    t0 = 0.0
    t1 = float(dmax[0] - dmin[0])

    segments = ray.linear_pieces(origin, direction, t0, t1)
    total_length = sum(float(seg.t_end) - float(seg.t_start) for seg in segments)
    assert total_length <= 1.0e-6

    origins = origin.reshape(1, 3)
    exact = np.asarray(ray.integrate_field_along_rays(origins, direction, t0, t1), dtype=float)
    midpoint = np.asarray(ray.integrate_field_along_rays_midpoint(origins, direction, t0, t1), dtype=float)
    assert np.all(np.isfinite(exact))
    assert np.all(np.isfinite(midpoint))
    assert abs(float(exact[0])) <= 1.0e-6
    assert abs(float(midpoint[0])) <= 1.0e-6


def test_cartesian_outside_start_inward_ray_traces_and_integrates() -> None:
    """Outside-start inward rays should enter the domain and produce valid segments/integrals."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = tree.domain_bounds(coord="xyz")
    origin = np.array(
        [
            float(dmin[0]) - 1.0,
            0.5 * float(dmin[1] + dmax[1]),
            0.5 * float(dmin[2] + dmax[2]),
        ],
        dtype=float,
    )
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    t0 = 0.0
    t1 = float(dmax[0] - dmin[0]) + 2.0

    segments = ray.linear_pieces(origin, direction, t0, t1)
    assert len(segments) > 0
    assert float(segments[0].t_start) > 0.0
    assert all(int(seg.cell_id) >= 0 for seg in segments)

    origins = origin.reshape(1, 3)
    exact = np.asarray(ray.integrate_field_along_rays(origins, direction, t0, t1), dtype=float)
    midpoint = np.asarray(ray.integrate_field_along_rays_midpoint(origins, direction, t0, t1), dtype=float)
    assert np.all(np.isfinite(exact))
    assert np.all(np.isfinite(midpoint))
    assert float(exact[0]) > 0.0
    assert float(midpoint[0]) > 0.0


def test_spherical_outside_start_inward_ray_traces_and_integrates() -> None:
    """Outside-start inward rays should trace/integrate on spherical trees too."""
    ds = _build_fake_dataset()
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = tree.domain_bounds(coord="xyz")
    origin = np.array(
        [
            float(dmax[0]) + 1.0,
            0.2,
            -0.15,
        ],
        dtype=float,
    )
    direction = np.array([-1.0, 0.0, 0.0], dtype=float)
    t0 = 0.0
    t1 = float(dmax[0] - dmin[0]) + 3.0

    segments = ray.linear_pieces(origin, direction, t0, t1)
    assert len(segments) > 0
    assert float(segments[0].t_start) > 0.0
    assert all(int(seg.cell_id) >= 0 for seg in segments)

    _t_vals, vals, cell_ids, _segments = ray.sample(origin, direction, t0, t1, 96)
    cids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    v = np.asarray(vals, dtype=float).reshape(-1)
    assert np.any(cids >= 0)
    assert np.any(np.isfinite(v[cids >= 0]))


def test_adaptive_midpoint_rule_outputs_consistent_offsets() -> None:
    """Adaptive midpoint packing should return monotone offsets and matching lengths."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    origins = np.array(
        [
            [0.0, -0.2, 0.1],
            [0.0, 0.0, 0.2],
            [0.0, 0.2, 0.3],
        ],
        dtype=float,
    )
    mids, weights, offsets = ray.adaptive_midpoint_rule(
        origins,
        np.array([1.0, 0.1, -0.05], dtype=float),
        0.0,
        2.0,
        chunk_size=2,
    )

    assert mids.ndim == 2 and mids.shape[1] == 3
    assert weights.ndim == 1
    assert mids.shape[0] == weights.shape[0]
    assert offsets.shape == (origins.shape[0] + 1,)
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == int(weights.shape[0])
    assert np.all(np.diff(offsets) >= 0)
    assert np.all(weights >= 0.0)


def test_midpoint_integrator_matches_exact_for_linear_field() -> None:
    """Midpoint quadrature should match exact integral for globally linear fields."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    origins = np.array(
        [
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.2, 0.4],
        ],
        dtype=float,
    )
    direction = np.array([1.0, 0.2, -0.1], dtype=float)
    t0 = 0.0
    t1 = 1.0

    exact = np.asarray(
        ray.integrate_field_along_rays(
            origins,
            direction,
            t0,
            t1,
            chunk_size=2,
        ),
        dtype=float,
    )
    midpoint = np.asarray(
        ray.integrate_field_along_rays_midpoint(
            origins,
            direction,
            t0,
            t1,
            chunk_size=2,
        ),
        dtype=float,
    )
    assert np.allclose(midpoint, exact, atol=1e-8, rtol=1e-9)


def test_vector_ray_integrals_preserve_shape_when_all_rays_miss() -> None:
    """Vector ray integration should keep `(n_rays, n_components)` shape on all misses."""
    ds = _build_fake_cartesian_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Scalar", "Scalar2"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    origins = np.array(
        [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
        ],
        dtype=float,
    )
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    exact = np.asarray(ray.integrate_field_along_rays(origins, direction, 0.0, 1.0), dtype=float)
    midpoint = np.asarray(ray.integrate_field_along_rays_midpoint(origins, direction, 0.0, 1.0), dtype=float)

    assert exact.shape == (origins.shape[0], 2)
    assert midpoint.shape == (origins.shape[0], 2)
    assert np.all(np.isnan(exact))
    assert np.all(np.isnan(midpoint))


def test_interpolator_outside_queries_use_fill_value_and_minus_one_cell_id() -> None:
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


def test_interpolator_vector_fill_value_is_applied_outside_domain() -> None:
    """Vector-valued fill should broadcast correctly for outside-domain queries."""
    ds = _build_fake_dataset()
    fill = np.array([-5.0, 8.0])
    interp = OctreeInterpolator(ds, ["Scalar", "Scalar2"], fill_value=fill)
    q = np.array([[1e6, 0.0, 0.0]])
    vals, cids = interp(q, return_cell_ids=True)
    assert vals.shape == (1, 2)
    assert int(cids[0]) == -1
    assert np.allclose(vals[0], fill, atol=0.0, rtol=0.0)


def test_interpolator_rpa_wrap_equivalence_on_resolvable_point() -> None:
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


def test_invalid_level_cells_are_consistently_treated_as_misses() -> None:
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

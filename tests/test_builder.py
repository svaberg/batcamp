from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp.builder import DEFAULT_AXIS_TOL
from batcamp.builder import _build_octree_state
from batcamp.builder import _resolve_cell_levels
import batcamp.builder_cartesian as cartesian_builder
import batcamp.builder_spherical as spherical_builder
from batcamp.shared import XYZ_VARS
from batcamp.octree import _rebuild_cells
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh
from tests.octree_test_support import cell_bounds


def _tree_from_state_build(
    points: np.ndarray,
    corners: np.ndarray,
    *,
    tree_coord: str | None,
    cell_levels: np.ndarray | None = None,
) -> Octree:
    state = _build_octree_state(
        points,
        corners,
        tree_coord=tree_coord,
        cell_levels=cell_levels,
    )
    return Octree.from_state(state, points=points, corners=corners)


def _build_regular_dataset(
    *,
    nr: int = 2,
    npolar: int = 4,
    nazimuth: int = 8,
) -> _FakeDataset:
    """Private test helper: build a small regular spherical hexahedral dataset."""
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
    scalar = 1.5 * x - 0.7 * y + 0.2 * z + 3.0
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Scalar": scalar,
        },
    )


def _build_irregular_spherical_dataset() -> _FakeDataset:
    """Build one spherical dataset with a deliberately non-dyadic azimuth corner."""
    ds = _build_regular_dataset()
    points = np.array(ds.points, dtype=float, copy=True)
    movable = np.flatnonzero((np.abs(points[:, 0]) > 1e-6) & (np.abs(points[:, 1]) > 1e-6))
    idx = int(movable[0])
    angle = 0.01
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = float(points[idx, 0])
    y = float(points[idx, 1])
    points[idx, 0] = cos_a * x - sin_a * y
    points[idx, 1] = sin_a * x + cos_a * y

    x_var = points[:, 0]
    y_var = points[:, 1]
    z_var = points[:, 2]
    scalar = 1.5 * x_var - 0.7 * y_var + 0.2 * z_var + 3.0
    return _FakeDataset(
        points=points,
        corners=np.array(ds.corners, dtype=np.int64, copy=True),
        variables={
            XYZ_VARS[0]: x_var,
            XYZ_VARS[1]: y_var,
            XYZ_VARS[2]: z_var,
            "Scalar": scalar,
        },
    )


def _build_regular_xyz_dataset(
    *,
    nx: int = 4,
    ny: int = 3,
    nz: int = 2,
) -> _FakeDataset:
    """Private test helper: build a small regular Cartesian hexahedral dataset."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.linspace(-2.0, 2.0, nx + 1),
        y_edges=np.linspace(-1.5, 1.5, ny + 1),
        z_edges=np.linspace(-1.0, 1.0, nz + 1),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    scalar = 1.2 * x - 0.3 * y + 0.8 * z + 0.5
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Scalar": scalar,
        },
    )


def _build_disjoint_xyz_dataset() -> _FakeDataset:
    """Private test helper: build two Cartesian cells separated by an empty gap."""
    cube0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    cube1 = cube0 + np.array([10.0, 0.0, 0.0], dtype=float)
    points = np.vstack((cube0, cube1))
    corners = np.array(
        [
            [0, 1, 3, 2, 4, 5, 7, 6],
            [8, 9, 11, 10, 12, 13, 15, 14],
        ],
        dtype=np.int64,
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    scalar = x + 2.0 * y + 3.0 * z
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Scalar": scalar,
        },
    )


def _build_adaptive_xyz_dataset() -> tuple[_FakeDataset, np.ndarray]:
    """Build one simple adaptive Cartesian dataset with one coarse and eight fine leaves."""
    x_edges = np.array([0.0, 1.0, 1.5, 2.0], dtype=float)
    y_edges = np.array([0.0, 0.5, 1.0], dtype=float)
    z_edges = np.array([0.0, 0.5, 1.0], dtype=float)
    points, _corners = _build_cartesian_hex_mesh(
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
    )
    node_index = np.arange(points.shape[0], dtype=np.int64).reshape(x_edges.size, y_edges.size, z_edges.size)
    corners = [
        [
            int(node_index[0, 0, 0]),
            int(node_index[1, 0, 0]),
            int(node_index[1, 2, 0]),
            int(node_index[0, 2, 0]),
            int(node_index[0, 0, 2]),
            int(node_index[1, 0, 2]),
            int(node_index[1, 2, 2]),
            int(node_index[0, 2, 2]),
        ]
    ]
    levels = [0]
    for ix in (1, 2):
        for iy in (0, 1):
            for iz in (0, 1):
                corners.append(
                    [
                        int(node_index[ix, iy, iz]),
                        int(node_index[ix + 1, iy, iz]),
                        int(node_index[ix + 1, iy + 1, iz]),
                        int(node_index[ix, iy + 1, iz]),
                        int(node_index[ix, iy, iz + 1]),
                        int(node_index[ix + 1, iy, iz + 1]),
                        int(node_index[ix + 1, iy + 1, iz + 1]),
                        int(node_index[ix, iy + 1, iz + 1]),
                    ]
                )
                levels.append(1)

    corners_arr = np.array(corners, dtype=np.int64)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    scalar = x + 2.0 * y + 3.0 * z
    ds = _FakeDataset(
        points=points,
        corners=corners_arr,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Scalar": scalar,
        },
    )
    return ds, np.array(levels, dtype=np.int64)


def _build_disjoint_spherical_shell_dataset() -> _FakeDataset:
    """Private test helper: build two spherical shell layers with a radial gap."""
    polar_edges = np.array([0.0, 0.5 * math.pi, math.pi], dtype=float)
    azimuth_edges = np.array([0.0, math.pi, 2.0 * math.pi], dtype=float)
    shell_edges = ((1.0, 2.0), (4.0, 5.0))

    points: list[tuple[float, float, float]] = []
    corners: list[list[int]] = []

    for r0, r1 in shell_edges:
        node_index = -np.ones((2, polar_edges.size, azimuth_edges.size), dtype=np.int64)
        for ir, rr in enumerate((r0, r1)):
            for ipolar, polar in enumerate(polar_edges):
                st = math.sin(float(polar))
                ct = math.cos(float(polar))
                for iazimuth, azimuth in enumerate(azimuth_edges):
                    x = float(rr) * st * math.cos(float(azimuth))
                    y = float(rr) * st * math.sin(float(azimuth))
                    z = float(rr) * ct
                    node_index[ir, ipolar, iazimuth] = len(points)
                    points.append((x, y, z))

        for ipolar in range(polar_edges.size - 1):
            for iazimuth in range(azimuth_edges.size - 1):
                corners.append(
                    [
                        int(node_index[0, ipolar + 1, iazimuth]),
                        int(node_index[1, ipolar + 1, iazimuth]),
                        int(node_index[1, ipolar + 1, iazimuth + 1]),
                        int(node_index[0, ipolar + 1, iazimuth + 1]),
                        int(node_index[0, ipolar, iazimuth]),
                        int(node_index[1, ipolar, iazimuth]),
                        int(node_index[1, ipolar, iazimuth + 1]),
                        int(node_index[0, ipolar, iazimuth + 1]),
                    ]
                )

    points_arr = np.array(points, dtype=float)
    x = points_arr[:, 0]
    y = points_arr[:, 1]
    z = points_arr[:, 2]
    scalar = x - y + z
    return _FakeDataset(
        points=points_arr,
        corners=np.array(corners, dtype=np.int64),
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Scalar": scalar,
        },
    )


def _make_cartesian_tree(
    *,
    leaf_shape: tuple[int, int, int],
    root_shape: tuple[int, int, int],
    max_level: int,
    cell_levels: np.ndarray | None,
) -> SimpleNamespace:
    levels = None if cell_levels is None else np.asarray(cell_levels, dtype=np.int64)
    return SimpleNamespace(
        leaf_shape=leaf_shape,
        root_shape=root_shape,
        max_level=int(max_level),
        cell_levels=levels,
    )


def _make_spherical_tree(
    *,
    leaf_shape: tuple[int, int, int],
    root_shape: tuple[int, int, int],
    max_level: int,
    cell_levels: np.ndarray | None,
) -> SimpleNamespace:
    levels = None if cell_levels is None else np.asarray(cell_levels, dtype=np.int64)
    return SimpleNamespace(
        leaf_shape=leaf_shape,
        root_shape=root_shape,
        max_level=int(max_level),
        cell_levels=levels,
    )


def _expected_rpa_cell_volumes(bounds: np.ndarray) -> np.ndarray:
    """Return exact physical spherical cell volumes for packed `(r, polar, azimuth)` bounds."""
    packed = np.asarray(bounds, dtype=float)
    r0 = packed[..., 0, 0]
    dr = packed[..., 0, 1]
    r1 = r0 + dr
    theta0 = packed[..., 1, 0]
    dtheta = packed[..., 1, 1]
    theta1 = theta0 + dtheta
    dphi = packed[..., 2, 1]
    return dphi * (np.cos(theta0) - np.cos(theta1)) * ((r1**3 - r0**3) / 3.0)


def _expected_rpa_radial_integrals(bounds: np.ndarray) -> np.ndarray:
    """Return exact physical-volume integrals of `f(r, polar, azimuth)=r` over spherical cells."""
    packed = np.asarray(bounds, dtype=float)
    r0 = packed[..., 0, 0]
    dr = packed[..., 0, 1]
    r1 = r0 + dr
    theta0 = packed[..., 1, 0]
    dtheta = packed[..., 1, 1]
    theta1 = theta0 + dtheta
    dphi = packed[..., 2, 1]
    return dphi * (np.cos(theta0) - np.cos(theta1)) * ((r1**4 - r0**4) / 4.0)


def _expected_rpa_polar_integrals(bounds: np.ndarray) -> np.ndarray:
    """Return exact physical-volume integrals of `f(r, polar, azimuth)=polar` over spherical cells."""
    packed = np.asarray(bounds, dtype=float)
    r0 = packed[..., 0, 0]
    dr = packed[..., 0, 1]
    r1 = r0 + dr
    theta0 = packed[..., 1, 0]
    dtheta = packed[..., 1, 1]
    theta1 = theta0 + dtheta
    dphi = packed[..., 2, 1]
    radial_factor = (r1**3 - r0**3) / 3.0
    polar_antiderivative = np.sin(theta1) - theta1 * np.cos(theta1) - (
        np.sin(theta0) - theta0 * np.cos(theta0)
    )
    return dphi * radial_factor * polar_antiderivative


def _expected_rpa_unwrapped_azimuth_integrals(bounds: np.ndarray) -> np.ndarray:
    """Return exact physical-volume integrals of `f(r, polar, azimuth)=azimuth` over spherical cells."""
    packed = np.asarray(bounds, dtype=float)
    r0 = packed[..., 0, 0]
    dr = packed[..., 0, 1]
    r1 = r0 + dr
    theta0 = packed[..., 1, 0]
    dtheta = packed[..., 1, 1]
    theta1 = theta0 + dtheta
    phi0 = packed[..., 2, 0]
    dphi = packed[..., 2, 1]
    phi1 = phi0 + dphi
    return ((r1**3 - r0**3) / 3.0) * (np.cos(theta0) - np.cos(theta1)) * ((phi1**2 - phi0**2) / 2.0)


def _rpa_pole_cell_ids(tree: Octree) -> np.ndarray:
    """Return ids of leaf cells that touch either polar cap."""
    bounds = np.asarray(tree.cell_bounds[: int(tree.cell_count)], dtype=float)
    theta0 = bounds[:, 1, 0]
    theta1 = theta0 + bounds[:, 1, 1]
    return np.flatnonzero(np.isclose(theta0, 0.0, atol=1.0e-12) | np.isclose(theta1, math.pi, atol=1.0e-12))


def _rpa_azimuth_seam_cell_ids(tree: Octree) -> np.ndarray:
    """Return ids of leaf cells that touch the azimuth seam at `0 == 2pi`."""
    bounds = np.asarray(tree.cell_bounds[: int(tree.cell_count)], dtype=float)
    phi0 = bounds[:, 2, 0]
    phi1 = phi0 + bounds[:, 2, 1]
    return np.flatnonzero(np.isclose(phi0, 0.0, atol=1.0e-12) | np.isclose(phi1, 2.0 * math.pi, atol=1.0e-12))


def _expected_rpa_box_volume(lower: np.ndarray, upper: np.ndarray) -> float:
    """Return exact physical volume of one spherical native box."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float((hi[2] - lo[2]) * (np.cos(lo[1]) - np.cos(hi[1])) * ((hi[0] ** 3 - lo[0] ** 3) / 3.0))


def _expected_rpa_box_radial_integral(lower: np.ndarray, upper: np.ndarray) -> float:
    """Return exact physical-volume integral of `f(r, polar, azimuth)=r` over one spherical native box."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float((hi[2] - lo[2]) * (np.cos(lo[1]) - np.cos(hi[1])) * ((hi[0] ** 4 - lo[0] ** 4) / 4.0))


def _expected_rpa_box_polar_integral(lower: np.ndarray, upper: np.ndarray) -> float:
    """Return exact physical-volume integral of `f(r, polar, azimuth)=polar` over one spherical native box."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    radial_factor = (hi[0] ** 3 - lo[0] ** 3) / 3.0
    polar_factor = (np.sin(hi[1]) - hi[1] * np.cos(hi[1])) - (np.sin(lo[1]) - lo[1] * np.cos(lo[1]))
    return float((hi[2] - lo[2]) * radial_factor * polar_factor)


@pytest.fixture(scope="module")
def cartesian_octree_context() -> tuple[_FakeDataset, Octree, OctreeInterpolator]:
    """Build one reusable Cartesian octree/interpolator context for xyz-path tests."""
    ds = _build_regular_xyz_dataset()
    tree = Octree.from_ds(ds, tree_coord="xyz")
    assert isinstance(tree, Octree)
    interp = OctreeInterpolator(tree, np.asarray(ds["Scalar"]))
    return ds, tree, interp


@pytest.fixture(scope="module")
def spherical_octree_context() -> tuple[_FakeDataset, Octree, OctreeInterpolator]:
    """Build one reusable spherical octree/interpolator context for rpa-path tests."""
    ds = _build_regular_dataset()
    tree = Octree.from_ds(ds, tree_coord="rpa")
    assert isinstance(tree, Octree)
    interp = OctreeInterpolator(tree, np.asarray(ds["Scalar"]))
    return ds, tree, interp


def test_xyz_lookup_hits_cell_midpoints(cartesian_octree_context) -> None:
    """Cartesian lookup should resolve each cell midpoint to its own cell id."""
    _ds, tree, _interp = cartesian_octree_context
    for cell_id in range(tree.cell_count):
        lo, hi = cell_bounds(tree, cell_id, coord="xyz")
        q = 0.5 * (np.asarray(lo, dtype=float) + np.asarray(hi, dtype=float))
        assert int(tree.lookup_points(q, coord="xyz")[0]) == int(cell_id)


def test_xyz_interp_matches_linear_field(cartesian_octree_context) -> None:
    """Cartesian trilinear interpolation should reconstruct the synthetic linear xyz field."""
    _ds, tree, interp = cartesian_octree_context
    rng = np.random.default_rng(42)
    n_cells = int(tree.cell_count)
    choose = rng.choice(n_cells, size=min(20, n_cells), replace=False)

    q = np.empty((choose.size, 3), dtype=float)
    expected = np.empty(choose.size, dtype=float)
    for i, cell_id in enumerate(choose.tolist()):
        lo, hi = cell_bounds(tree, int(cell_id), coord="xyz")
        u, v, w = rng.uniform(0.1, 0.9, size=3)
        x = float(lo[0] + u * (hi[0] - lo[0]))
        y = float(lo[1] + v * (hi[1] - lo[1]))
        z = float(lo[2] + w * (hi[2] - lo[2]))
        q[i] = (x, y, z)
        expected[i] = 1.2 * x - 0.3 * y + 0.8 * z + 0.5

    values, cell_ids = interp(q, return_cell_ids=True)
    assert np.array_equal(np.array(cell_ids, dtype=np.int64), np.array(choose, dtype=np.int64))
    assert np.allclose(np.array(values, dtype=float), expected, atol=1e-12, rtol=0.0)


def test_xyz_cell_volumes_match_regular_mesh(cartesian_octree_context) -> None:
    """Regular Cartesian leaf volumes should be the product of the slab widths."""
    _ds, tree, _interp = cartesian_octree_context
    np.testing.assert_allclose(tree.cell_volumes, np.ones(int(tree.cell_count), dtype=float), atol=1.0e-12, rtol=0.0)


def test_xyz_native_axis_slabs_match_leaf_x_breaks(cartesian_octree_context) -> None:
    """Cartesian native-axis slabs should recover the occupied x-interval partition."""
    _ds, tree, _interp = cartesian_octree_context
    expected = np.array(
        [
            [-2.0, -1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(tree.native_axis_slabs(0), expected, atol=1.0e-12, rtol=0.0)


def test_xyz_cell_volumes_match_adaptive_mesh() -> None:
    """Adaptive Cartesian leaf volumes should stay aligned with the persisted leaf rows."""
    ds, levels = _build_adaptive_xyz_dataset()
    tree = _tree_from_state_build(
        np.asarray(ds.points, dtype=float),
        np.asarray(ds.corners, dtype=np.int64),
        tree_coord="xyz",
        cell_levels=levels,
    )
    expected = np.where(levels == 0, 1.0, 0.125)
    np.testing.assert_allclose(tree.cell_volumes, expected, atol=1.0e-12, rtol=0.0)


def test_xyz_cell_integrals_match_constant_field(cartesian_octree_context) -> None:
    """Whole-cell trilinear integrals should reduce to constant times volume for constant data."""
    ds, tree, _interp = cartesian_octree_context
    const_value = 3.25
    interp = OctreeInterpolator(tree, np.full(np.asarray(ds.points).shape[0], const_value, dtype=float))
    np.testing.assert_allclose(
        interp.cell_integrals(),
        const_value * tree.cell_volumes,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_xyz_cell_integrals_match_linear_vector_subset(cartesian_octree_context) -> None:
    """Whole-cell xyz integrals should be exact for linear data and preserve vector components."""
    ds, tree, _interp = cartesian_octree_context
    scalar = np.asarray(ds["Scalar"], dtype=float)
    interp = OctreeInterpolator(tree, np.column_stack((scalar, 2.0 * scalar + 1.0)))
    subset = np.array([0, int(tree.cell_count // 2), int(tree.cell_count - 1)], dtype=np.int64)

    expected = np.empty((subset.size, 2), dtype=float)
    for pos, cell_id in enumerate(subset.tolist()):
        lo, hi = cell_bounds(tree, int(cell_id), coord="xyz")
        center = 0.5 * (np.asarray(lo, dtype=float) + np.asarray(hi, dtype=float))
        volume = float(tree.cell_volumes[int(cell_id)])
        scalar_center = 1.2 * float(center[0]) - 0.3 * float(center[1]) + 0.8 * float(center[2]) + 0.5
        expected[pos, 0] = volume * scalar_center
        expected[pos, 1] = volume * (2.0 * scalar_center + 1.0)

    np.testing.assert_allclose(interp.cell_integrals(subset), expected, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(interp.cell_integrals(int(subset[0])), expected[0], atol=1.0e-12, rtol=0.0)


def test_xyz_integrate_box_matches_constant_field(cartesian_octree_context) -> None:
    """Cartesian box integrals should reduce to constant times box volume."""
    ds, tree, _interp = cartesian_octree_context
    const_value = -1.75
    interp = OctreeInterpolator(tree, np.full(np.asarray(ds.points).shape[0], const_value, dtype=float))
    lower = np.array([-1.25, -0.8, -0.45], dtype=float)
    upper = np.array([1.5, 0.9, 0.7], dtype=float)
    expected = const_value * float(np.prod(upper - lower))
    np.testing.assert_allclose(interp.integrate_box(lower, upper), expected, atol=1.0e-12, rtol=0.0)


def test_xyz_integrate_box_matches_linear_vector_field(cartesian_octree_context) -> None:
    """Cartesian box integrals should be exact for linear vector-valued data."""
    ds, tree, _interp = cartesian_octree_context
    scalar = np.asarray(ds["Scalar"], dtype=float)
    interp = OctreeInterpolator(tree, np.column_stack((scalar, 2.0 * scalar + 1.0)))
    lower = np.array([-1.3, -0.6, -0.3], dtype=float)
    upper = np.array([1.1, 1.0, 0.95], dtype=float)
    center = 0.5 * (lower + upper)
    volume = float(np.prod(upper - lower))
    scalar_center = 1.2 * float(center[0]) - 0.3 * float(center[1]) + 0.8 * float(center[2]) + 0.5
    expected = np.array([volume * scalar_center, volume * (2.0 * scalar_center + 1.0)], dtype=float)
    np.testing.assert_allclose(interp.integrate_box(lower, upper), expected, atol=1.0e-12, rtol=0.0)


def test_rpa_cell_volumes_match_physical_spherical_formula(spherical_octree_context) -> None:
    """Spherical leaf volumes should match the exact physical shell-sector formula."""
    _ds, tree, _interp = spherical_octree_context
    expected = _expected_rpa_cell_volumes(tree.cell_bounds[: int(tree.cell_count)])
    np.testing.assert_allclose(tree.cell_volumes, expected, atol=1.0e-12, rtol=0.0)


def test_rpa_native_axis_slabs_match_leaf_radial_breaks(spherical_octree_context) -> None:
    """Spherical native-axis slabs should recover the occupied radial partition."""
    _ds, tree, _interp = spherical_octree_context
    expected = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(tree.native_axis_slabs(0), expected, atol=1.0e-12, rtol=0.0)


def test_rpa_cell_integrals_match_constant_field(spherical_octree_context) -> None:
    """Whole-cell spherical integrals should reduce to constant times physical volume."""
    ds, tree, _interp = spherical_octree_context
    const_value = 2.75
    interp = OctreeInterpolator(tree, np.full(np.asarray(ds.points).shape[0], const_value, dtype=float))
    np.testing.assert_allclose(
        interp.cell_integrals(),
        const_value * tree.cell_volumes,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_rpa_cell_integrals_match_radial_and_polar_vector_subset(spherical_octree_context) -> None:
    """Whole-cell spherical integrals should be exact for fields linear in `r` and `polar`."""
    ds, tree, _interp = spherical_octree_context
    points = np.asarray(ds.points, dtype=float)
    radii = np.linalg.norm(points, axis=1)
    polar = np.arccos(np.clip(points[:, 2] / radii, -1.0, 1.0))
    interp = OctreeInterpolator(tree, np.column_stack((radii, polar)))
    subset = np.array([0, int(tree.cell_count // 2), int(tree.cell_count - 1)], dtype=np.int64)

    subset_bounds = tree.cell_bounds[subset]
    expected = np.column_stack(
        (
            _expected_rpa_radial_integrals(subset_bounds),
            _expected_rpa_polar_integrals(subset_bounds),
        )
    )

    np.testing.assert_allclose(interp.cell_integrals(subset), expected, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(interp.cell_integrals(int(subset[0])), expected[0], atol=1.0e-12, rtol=0.0)


def test_rpa_cell_integrals_match_polar_field_on_pole_cells(spherical_octree_context) -> None:
    """Pole-adjacent spherical cells should integrate the `polar` nodal field exactly."""
    ds, tree, _interp = spherical_octree_context
    points = np.asarray(ds.points, dtype=float)
    radii = np.linalg.norm(points, axis=1)
    polar = np.arccos(np.clip(points[:, 2] / radii, -1.0, 1.0))
    interp = OctreeInterpolator(tree, polar)
    pole_ids = _rpa_pole_cell_ids(tree)

    assert pole_ids.size > 0
    expected = _expected_rpa_polar_integrals(tree.cell_bounds[pole_ids])
    np.testing.assert_allclose(interp.cell_integrals(pole_ids), expected, atol=1.0e-12, rtol=0.0)


def test_rpa_cell_integrals_match_unwrapped_azimuth_field_on_seam_cells(
    spherical_octree_context,
) -> None:
    """Azimuth-seam cells should integrate an unwrapped azimuth nodal field exactly."""
    _ds, tree, _interp = spherical_octree_context
    n_phi_edge = int(tree.leaf_shape[2]) + 1
    azimuth_nodes = (np.arange(np.asarray(tree._points).shape[0], dtype=np.float64) % n_phi_edge) * (
        2.0 * math.pi / float(int(tree.leaf_shape[2]))
    )
    interp = OctreeInterpolator(tree, azimuth_nodes)
    seam_ids = _rpa_azimuth_seam_cell_ids(tree)

    assert seam_ids.size > 0
    expected = _expected_rpa_unwrapped_azimuth_integrals(tree.cell_bounds[seam_ids])
    np.testing.assert_allclose(interp.cell_integrals(seam_ids), expected, atol=1.0e-12, rtol=0.0)


def test_rpa_integrate_box_matches_constant_field(spherical_octree_context) -> None:
    """Spherical box integrals should reduce to constant times physical box volume."""
    ds, tree, _interp = spherical_octree_context
    const_value = 4.25
    interp = OctreeInterpolator(tree, np.full(np.asarray(ds.points).shape[0], const_value, dtype=float))
    lower = np.array([1.2, 0.0, 0.0], dtype=float)
    upper = np.array([2.6, 0.5 * math.pi, 2.0 * math.pi], dtype=float)
    expected = const_value * _expected_rpa_box_volume(lower, upper)
    np.testing.assert_allclose(interp.integrate_box(lower, upper), expected, atol=1.0e-12, rtol=0.0)


def test_rpa_integrate_box_matches_radial_and_polar_vector_field(spherical_octree_context) -> None:
    """Spherical box integrals should be exact for vector-valued `r` and `polar` fields."""
    ds, tree, _interp = spherical_octree_context
    points = np.asarray(ds.points, dtype=float)
    radii = np.linalg.norm(points, axis=1)
    polar = np.arccos(np.clip(points[:, 2] / radii, -1.0, 1.0))
    interp = OctreeInterpolator(tree, np.column_stack((radii, polar)))
    lower = np.array([1.25, 0.25, 0.1], dtype=float)
    upper = np.array([2.75, 2.4, 5.2], dtype=float)
    expected = np.array(
        [
            _expected_rpa_box_radial_integral(lower, upper),
            _expected_rpa_box_polar_integral(lower, upper),
        ],
        dtype=float,
    )
    np.testing.assert_allclose(interp.integrate_box(lower, upper), expected, atol=1.0e-12, rtol=0.0)


def test_build_rejects_missing_corners() -> None:
    """Builder should fail fast when dataset has no corners."""
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    ds = _FakeDataset(
        points=points,
        corners=None,
        variables={XYZ_VARS[0]: points[:, 0], XYZ_VARS[1]: points[:, 1], XYZ_VARS[2]: points[:, 2]},
    )
    with pytest.raises(ValueError, match="Dataset has no corners"):
        Octree.from_ds(ds)


def test_build_rejects_unknown_tree_coord() -> None:
    """Builder should reject unsupported coordinate-system identifiers."""
    ds = _build_regular_dataset()
    with pytest.raises(ValueError, match="Unsupported tree_coord"):
        Octree.from_ds(ds, tree_coord="foo")


def test_build_xyz_returns_cartesian_tree() -> None:
    """Builder should construct Cartesian octree when tree_coord='xyz'."""
    ds = _build_regular_xyz_dataset()
    tree = Octree.from_ds(ds, tree_coord="xyz")
    assert isinstance(tree, Octree)
    assert tree.tree_coord == "xyz"


def test_build_default_returns_cartesian_tree() -> None:
    """Default build path should return the Cartesian octree specialization."""
    ds = _build_regular_xyz_dataset()
    tree = Octree.from_ds(ds)
    assert isinstance(tree, Octree)
    assert tree.tree_coord == "xyz"


def test_build_rejects_bad_point_shape() -> None:
    """Explicit point arrays must have shape `(n_points, 3)`."""
    points = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    corners = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="points must have shape"):
        Octree(points, corners, tree_coord="rpa")


def test_compute_azimuth_spans_reject_bad_corner_rank() -> None:
    """Azimuth-span computation should reject non-2D corner arrays."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    corners = np.array([0, 1, 2], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={XYZ_VARS[0]: points[:, 0], XYZ_VARS[1]: points[:, 1], XYZ_VARS[2]: points[:, 2]},
    )
    with pytest.raises(ValueError, match="Expected 2D corner array"):
        spherical_builder.compute_azimuth_spans_and_levels(np.asarray(ds.points, dtype=float), corners=corners)


def test_compute_azimuth_spans_reject_too_few_corners() -> None:
    """Azimuth-span computation should reject cells with fewer than 3 corners."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    corners = np.array([[0, 1]], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={XYZ_VARS[0]: points[:, 0], XYZ_VARS[1]: points[:, 1], XYZ_VARS[2]: points[:, 2]},
    )
    with pytest.raises(ValueError, match="Need at least 3 corners per cell"):
        spherical_builder.compute_azimuth_spans_and_levels(np.asarray(ds.points, dtype=float), corners=corners)


def test_infer_levels_marks_non_dyadic_span_invalid() -> None:
    """Non-dyadic azimuth spans should map to level -1."""
    levels, _expected, _coarse = spherical_builder.infer_level_expectation(np.array([1.0, 0.5, 0.3]))
    assert np.array_equal(levels, np.array([0, 1, -1], dtype=np.int64))


def test_build_tree_rejects_all_invalid_levels() -> None:
    """Tree construction should fail when all provided levels are invalid."""
    ds = _build_regular_dataset()
    azimuth_span, _azimuth_center, _cell_levels, _expected, _coarse = (
        spherical_builder.compute_azimuth_spans_and_levels(
            np.asarray(ds.points, dtype=float),
            corners=np.asarray(ds.corners, dtype=np.int64),
        )
    )
    all_invalid = np.full(azimuth_span.shape, -1, dtype=np.int64)
    with pytest.raises(ValueError, match="No valid \\(>=0\\) levels available to infer octree"):
        _tree_from_state_build(
            np.asarray(ds.points, dtype=float),
            np.asarray(ds.corners, dtype=np.int64),
            tree_coord="rpa",
            cell_levels=all_invalid,
        )


def test_warns_on_incompatible_blocks_aux_without_block_tree(caplog: pytest.LogCaptureFixture) -> None:
    """Incompatible BLOCKS aux metadata should be ignored with a warning."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "7 3x5x9"
    with caplog.at_level(logging.WARNING, logger="batcamp.builder"):
        tree = Octree.from_ds(ds, tree_coord="rpa")
    assert tree.level_counts
    assert tree.leaf_shape[0] > 0
    assert any("BLOCKS" in rec.getMessage() and "does not match" in rec.getMessage() for rec in caplog.records)


def test_no_warning_on_compatible_blocks_aux(caplog: pytest.LogCaptureFixture) -> None:
    """Compatible BLOCKS aux metadata should not emit a warning."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "1 2x4x8"
    with caplog.at_level(logging.WARNING, logger="batcamp.builder"):
        tree = Octree.from_ds(ds, tree_coord="rpa")
    assert tree.level_counts
    assert not any("BLOCKS" in rec.getMessage() for rec in caplog.records)


def test_warns_on_blocks_count_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    """BLOCKS count mismatch should emit a warning."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "5 1x1x1"
    with caplog.at_level(logging.WARNING, logger="batcamp.builder"):
        tree = Octree.from_ds(ds, tree_coord="rpa")
    assert tree.level_counts
    assert any("BLOCKS" in rec.getMessage() and "does not match" in rec.getMessage() for rec in caplog.records)


def test_warns_on_impossible_blocks_count(caplog: pytest.LogCaptureFixture) -> None:
    """Impossible BLOCKS counts should emit a mismatch warning."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "1000 1x1x1"
    with caplog.at_level(logging.WARNING, logger="batcamp.builder"):
        tree = Octree.from_ds(ds, tree_coord="rpa")
    assert tree.level_counts
    assert any("BLOCKS" in rec.getMessage() and "does not match" in rec.getMessage() for rec in caplog.records)


def test_warns_on_unparseable_blocks_aux(caplog: pytest.LogCaptureFixture) -> None:
    """Unparseable BLOCKS aux metadata should be ignored with a warning."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "garbage"
    with caplog.at_level(logging.WARNING, logger="batcamp.builder"):
        tree = Octree.from_ds(ds, tree_coord="rpa")
    assert tree.level_counts
    assert any("BLOCKS" in rec.getMessage() and "not parseable" in rec.getMessage() for rec in caplog.records)


def test_resolve_cell_levels_requires_inferred_levels_when_cell_levels_missing() -> None:
    """Level resolution should fail when neither inferred nor explicit levels are provided."""
    with pytest.raises(ValueError, match="inferred_levels is required"):
        _resolve_cell_levels(
            inferred_levels=None,
            cell_levels=None,
            expected_shape=(2,),
        )


def test_resolve_cell_levels_rejects_shape_mismatch() -> None:
    """Level resolution should reject arrays that do not match the expected cell shape."""
    with pytest.raises(ValueError, match="cell_levels shape does not match"):
        _resolve_cell_levels(
            inferred_levels=np.array([0, 1], dtype=np.int64),
            cell_levels=None,
            expected_shape=(1,),
        )


def test_rebuild_cells_rejects_duplicate_leaf_addresses() -> None:
    """Sparse-cell construction should reject duplicate leaf octree addresses."""
    with pytest.raises(ValueError, match="overlap at octree address"):
        _rebuild_cells(
            np.array([0, 0], dtype=np.int64),
            np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        )


def test_rebuild_cells_rejects_parent_child_overlap() -> None:
    """Sparse-cell construction should reject explicit parent/child double occupancy."""
    with pytest.raises(ValueError, match="overlap across parent/child addresses"):
        _rebuild_cells(
            np.array([0, 1], dtype=np.int64),
            np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        )


def test_no_public_depth_for_level_helper() -> None:
    """Depth conversion is internal; no public depth-for-level helper is exposed."""
    tree = Octree.from_ds(_build_regular_dataset(), tree_coord="rpa")
    assert not hasattr(tree, "depth_for_level")


def test_regular_spherical_tree_uses_absolute_levels() -> None:
    """Uniform spherical trees should store root-relative absolute levels."""
    tree = Octree.from_ds(_build_regular_dataset(), tree_coord="rpa")
    assert tree.min_level == tree.max_level
    assert tree.cell_levels is not None
    assert np.all(tree.cell_levels == tree.max_level)


def test_build_materializes_exact_tree_state_on_ready_tree() -> None:
    """Builder should attach exact tree indices on the ready bound tree."""
    xyz_ds = _build_regular_xyz_dataset()
    xyz_tree = _tree_from_state_build(
        np.asarray(xyz_ds.points, dtype=float),
        np.asarray(xyz_ds.corners, dtype=np.int64),
        tree_coord="xyz",
    )
    assert xyz_tree.cell_levels is not None
    assert np.asarray(
        xyz_tree._cell_ijk[: xyz_tree.cell_levels.shape[0]],
        dtype=np.int64,
    ).shape == (xyz_tree.cell_levels.shape[0], 3)
    assert np.asarray(xyz_tree._cell_depth, dtype=np.int64).ndim == 1
    assert np.asarray(xyz_tree._cell_child, dtype=np.int64).shape[1] == 8
    assert np.asarray(xyz_tree._root_cell_ids, dtype=np.int64).ndim == 1
    assert not hasattr(xyz_tree, "_radial_edges")

    rpa_ds = _build_regular_dataset()
    rpa_tree = _tree_from_state_build(
        np.asarray(rpa_ds.points, dtype=float),
        np.asarray(rpa_ds.corners, dtype=np.int64),
        tree_coord="rpa",
    )
    assert rpa_tree.cell_levels is not None
    assert np.asarray(
        rpa_tree._cell_ijk[: rpa_tree.cell_levels.shape[0]],
        dtype=np.int64,
    ).shape == (rpa_tree.cell_levels.shape[0], 3)
    assert np.asarray(rpa_tree._cell_depth, dtype=np.int64).ndim == 1
    assert np.asarray(rpa_tree._cell_child, dtype=np.int64).shape[1] == 8
    assert np.asarray(rpa_tree._root_cell_ids, dtype=np.int64).ndim == 1
    assert hasattr(rpa_tree, "_radial_edges")


def test_spherical_lookup_rejects_non_exact_geometry() -> None:
    """Irregular spherical geometry should fail in the builder."""
    regular = _build_regular_dataset()
    _azimuth_span, _azimuth_center, cell_levels, _expected, _coarse = (
        spherical_builder.compute_azimuth_spans_and_levels(
            np.asarray(regular.points, dtype=float),
            corners=np.asarray(regular.corners, dtype=np.int64),
        )
    )
    irregular = _build_irregular_spherical_dataset()
    with pytest.raises(ValueError, match="Spherical cell .* inferred octree grid|no unique octree address"):
        _tree_from_state_build(
            np.asarray(irregular.points, dtype=float),
            np.asarray(irregular.corners, dtype=np.int64),
            tree_coord="rpa",
            cell_levels=cell_levels,
        )


def test_adaptive_cartesian_tree_preserves_root_relative_levels() -> None:
    """Adaptive Cartesian builds should keep supplied root-relative levels unchanged."""
    ds, cell_levels = _build_adaptive_xyz_dataset()
    tree = _tree_from_state_build(
        np.asarray(ds.points, dtype=float),
        np.asarray(ds.corners, dtype=np.int64),
        tree_coord="xyz",
        cell_levels=cell_levels,
    )
    assert tree.max_level == 1
    assert tree.min_level == 0
    assert tree.cell_levels is not None
    assert np.array_equal(tree.cell_levels, cell_levels)


def test_cartesian_level_shapes_reject_inconsistent_dx() -> None:
    """Cartesian level-shape inference should fail on one nonuniform same-level row."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0, 2.0, 2.3], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        z_edges=np.array([0.0, 1.0], dtype=float),
    )
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: points[:, 0],
            XYZ_VARS[1]: points[:, 1],
            XYZ_VARS[2]: points[:, 2],
            "Scalar": points[:, 0],
        },
    )
    cell_min, cell_max, cell_span = cartesian_builder.cell_geometry(
        np.asarray(ds.points, dtype=float),
        np.asarray(corners, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="inconsistent dx"):
        cartesian_builder.infer_level_shapes(
            cell_min,
            cell_max,
            cell_span,
            np.zeros(3, dtype=np.int64),
        )


def test_build_rejects_inconsistent_cartesian_geometry() -> None:
    """Cartesian build should fail with the higher-level builder message on inconsistent same-level geometry."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0, 2.0, 2.3], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        z_edges=np.array([0.0, 1.0], dtype=float),
    )
    with pytest.raises(ValueError, match="Could not build a Cartesian octree"):
        _tree_from_state_build(
            np.asarray(points, dtype=float),
            np.asarray(corners, dtype=np.int64),
            tree_coord="xyz",
            cell_levels=np.zeros(3, dtype=np.int64),
        )


def test_cartesian_infer_leaf_shape_rejects_missing_max_level() -> None:
    """Cartesian finest-shape inference should fail when the requested max level is absent."""
    ds = _build_regular_xyz_dataset(nx=2, ny=2, nz=2)
    cell_min, cell_max, cell_span = cartesian_builder.cell_geometry(
        np.asarray(ds.points, dtype=float),
        np.asarray(ds.corners, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="No cells found at max_level=1"):
        cartesian_builder.infer_leaf_shape(
            cell_min,
            cell_max,
            cell_span,
            np.zeros(int(np.asarray(ds.corners).shape[0]), dtype=np.int64),
            max_level=1,
        )


def test_cartesian_tree_state_requires_at_least_one_valid_level() -> None:
    """Cartesian exact-state construction should fail when all cell levels are invalid."""
    ds = _build_regular_xyz_dataset(nx=2, ny=1, nz=1)
    tree = _make_cartesian_tree(
        leaf_shape=(2, 1, 1),
        root_shape=(2, 1, 1),
        max_level=0,
        cell_levels=np.full(int(np.asarray(ds.corners).shape[0]), -1, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="at least one valid cell level"):
        cartesian_builder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            points=np.asarray(ds.points, dtype=float),
            corners=np.asarray(ds.corners, dtype=np.int64),
        )


def test_cartesian_tree_state_rejects_depth_above_tree_depth() -> None:
    """Cartesian exact-state construction should reject cells deeper than tree.max_level."""
    ds = _build_regular_xyz_dataset(nx=2, ny=1, nz=1)
    tree = _make_cartesian_tree(
        leaf_shape=(2, 1, 1),
        root_shape=(2, 1, 1),
        max_level=0,
        cell_levels=np.ones(int(np.asarray(ds.corners).shape[0]), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="depth exceeds tree_depth=0"):
        cartesian_builder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            points=np.asarray(ds.points, dtype=float),
            corners=np.asarray(ds.corners, dtype=np.int64),
        )


def test_cartesian_tree_state_rejects_width_mismatch() -> None:
    """Cartesian exact-state construction should reject cells whose width contradicts the supplied level."""
    ds = _build_regular_xyz_dataset(nx=4, ny=2, nz=2)
    tree = _make_cartesian_tree(
        leaf_shape=(4, 2, 2),
        root_shape=(2, 1, 1),
        max_level=1,
        cell_levels=np.zeros(int(np.asarray(ds.corners).shape[0]), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="width does not match inferred level 0"):
        cartesian_builder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            points=np.asarray(ds.points, dtype=float),
            corners=np.asarray(ds.corners, dtype=np.int64),
        )


def test_spherical_level_shapes_require_valid_levels() -> None:
    """Spherical angular-shape inference should fail when all levels are invalid."""
    ds = _build_regular_dataset()
    corners = np.asarray(ds.corners, dtype=np.int64)
    azimuth_span, _azimuth_center, _levels, _expected, _coarse = spherical_builder.compute_azimuth_spans_and_levels(
        np.asarray(ds.points, dtype=float),
        corners=corners,
    )
    with pytest.raises(ValueError, match="No valid \\(>=0\\) cell levels available for tree inference"):
        spherical_builder.infer_level_angular_shapes(
            np.asarray(ds.points, dtype=float),
            corners,
            azimuth_span,
            np.full(azimuth_span.shape, -1, dtype=np.int64),
        )


def test_spherical_infer_leaf_shape_rejects_noninteger_radial_count() -> None:
    """Spherical finest-shape inference should fail when weighted cells imply a noninteger radial count."""
    with pytest.raises(ValueError, match="Could not infer integer finest n_axis0"):
        spherical_builder.infer_leaf_shape(
            {
                0: (2, 4, math.pi / 2.0, math.pi / 2.0, 1),
                1: (4, 8, math.pi / 4.0, math.pi / 4.0, 1),
            }
        )


def test_spherical_tree_state_requires_cell_levels() -> None:
    """Spherical exact-state construction should fail without cell levels."""
    ds = _build_regular_dataset()
    tree = _make_spherical_tree(
        leaf_shape=(2, 4, 8),
        root_shape=(1, 2, 4),
        max_level=1,
        cell_levels=None,
    )
    with pytest.raises(ValueError, match="Spherical tree state requires cell_levels"):
        spherical_builder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            points=np.asarray(ds.points, dtype=float),
            corners=np.asarray(ds.corners, dtype=np.int64),
            axis_tol=DEFAULT_AXIS_TOL,
        )


def test_spherical_tree_state_requires_at_least_one_valid_level() -> None:
    """Spherical exact-state construction should fail when all cell levels are invalid."""
    ds = _build_regular_dataset()
    tree = _make_spherical_tree(
        leaf_shape=(2, 4, 8),
        root_shape=(1, 2, 4),
        max_level=1,
        cell_levels=np.full(int(np.asarray(ds.corners).shape[0]), -1, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="at least one valid cell level"):
        spherical_builder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            points=np.asarray(ds.points, dtype=float),
            corners=np.asarray(ds.corners, dtype=np.int64),
            axis_tol=DEFAULT_AXIS_TOL,
        )


def test_spherical_tree_state_rejects_depth_above_tree_depth() -> None:
    """Spherical exact-state construction should reject cells deeper than tree.max_level."""
    ds = _build_regular_dataset()
    tree = _make_spherical_tree(
        leaf_shape=(2, 4, 8),
        root_shape=(2, 4, 8),
        max_level=0,
        cell_levels=np.ones(int(np.asarray(ds.corners).shape[0]), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="depth exceeds tree_depth=0"):
        spherical_builder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            points=np.asarray(ds.points, dtype=float),
            corners=np.asarray(ds.corners, dtype=np.int64),
            axis_tol=DEFAULT_AXIS_TOL,
        )


def test_spherical_tree_state_rejects_width_mismatch() -> None:
    """Spherical exact-state construction should reject cells whose width contradicts the supplied level."""
    ds = _build_regular_dataset()
    tree = _make_spherical_tree(
        leaf_shape=(2, 4, 8),
        root_shape=(1, 2, 4),
        max_level=1,
        cell_levels=np.zeros(int(np.asarray(ds.corners).shape[0]), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="width does not match inferred level 0"):
        spherical_builder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            points=np.asarray(ds.points, dtype=float),
            corners=np.asarray(ds.corners, dtype=np.int64),
            axis_tol=DEFAULT_AXIS_TOL,
        )


def test_build_returns_bound_tree() -> None:
    """Builder should return a tree with lookup geometry ready for use."""
    ds = _build_regular_dataset()
    _azimuth_span, _azimuth_center, cell_levels, _expected, _coarse = (
        spherical_builder.compute_azimuth_spans_and_levels(
            np.asarray(ds.points, dtype=float),
            corners=np.asarray(ds.corners, dtype=np.int64),
        )
    )
    tree = _tree_from_state_build(
        np.asarray(ds.points, dtype=float),
        np.asarray(ds.corners, dtype=np.int64),
        tree_coord="rpa",
        cell_levels=cell_levels,
    )
    assert int(tree.lookup_points(np.array([1.0, 0.0, 0.0], dtype=float), coord="xyz")[0]) >= 0


def test_build_stores_tree_coord() -> None:
    """Builder should store requested coordinate-system metadata in the tree."""
    ds = _build_regular_xyz_dataset()
    tree = _tree_from_state_build(
        np.asarray(ds.points, dtype=float),
        np.asarray(ds.corners, dtype=np.int64),
        tree_coord="xyz",
    )
    assert isinstance(tree, Octree)
    assert tree.tree_coord == "xyz"


def test_build_rejects_inconsistent_corners_for_spherical_inference() -> None:
    """Spherical build should fail when `ds.corners` is internally inconsistent."""
    ds = _build_regular_dataset(nr=1, npolar=2, nazimuth=2)
    corners_full = np.array(ds.corners, copy=True)
    ds.corners = np.array(corners_full[:2], copy=True)

    with pytest.raises(ValueError, match="Could not build a spherical octree"):
        _tree_from_state_build(
            np.asarray(ds.points, dtype=float),
            np.asarray(ds.corners, dtype=np.int64),
            tree_coord="rpa",
            cell_levels=None,
        )


def test_build_rejects_forced_xyz_on_spherical_geometry() -> None:
    """Forcing Cartesian build on spherical geometry should fail with the Cartesian builder message."""
    ds = _build_regular_dataset()
    with pytest.raises(ValueError, match="Could not build a Cartesian octree"):
        _tree_from_state_build(
            np.asarray(ds.points, dtype=float),
            np.asarray(ds.corners, dtype=np.int64),
            tree_coord="xyz",
        )


def test_build_rejects_forced_rpa_on_cartesian_geometry() -> None:
    """Forcing spherical build on Cartesian geometry should fail with the spherical builder message."""
    ds = _build_regular_xyz_dataset()
    with pytest.raises(ValueError, match="Could not build a spherical octree"):
        _tree_from_state_build(
            np.asarray(ds.points, dtype=float),
            np.asarray(ds.corners, dtype=np.int64),
            tree_coord="rpa",
        )


def test_lookup_runs_for_xyz() -> None:
    """Lookup APIs should run when the tree is tagged as Cartesian."""
    ds = _build_regular_xyz_dataset()
    tree = Octree.from_ds(ds, tree_coord="xyz")
    assert int(tree.lookup_points(np.array([1.0, 0.0, 0.0], dtype=float), coord="xyz")[0]) >= 0
    assert not hasattr(tree, "lookup_rpa")


def test_lookup_gap_none_for_disjoint_cartesian_cells() -> None:
    """Cartesian lookup should return miss for points in an uncovered bbox gap."""
    ds = _build_disjoint_xyz_dataset()
    tree = Octree.from_ds(ds, tree_coord="xyz")
    q_gap = np.array([5.0, 0.5, 0.5], dtype=float)
    assert int(tree.lookup_points(q_gap, coord="xyz")[0]) < 0


def test_lookup_gap_none_for_disjoint_spherical_shells() -> None:
    """Gappy spherical shells should be rejected by the builder."""
    ds = _build_disjoint_spherical_shell_dataset()
    with pytest.raises(
        ValueError,
        match=(
            "(radial edge count does not match leaf_shape|"
            "Cells overlap at octree address)"
        ),
    ):
        Octree.from_ds(ds, tree_coord="rpa")

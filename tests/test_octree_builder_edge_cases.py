from __future__ import annotations

import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import CartesianOctree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayTracer
from batcamp import OctreeBuilder
from batcamp.builder_spherical import SphericalOctreeBuilder
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh


def _build_regular_dataset(
    *,
    nr: int = 2,
    ntheta: int = 4,
    nphi: int = 8,
) -> _FakeDataset:
    """Private test helper: build a small regular spherical hexahedral dataset."""
    points, corners = _build_spherical_hex_mesh(
        nr=nr,
        ntheta=ntheta,
        nphi=nphi,
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
            "X [R]": x,
            "Y [R]": y,
            "Z [R]": z,
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
            "X [R]": x,
            "Y [R]": y,
            "Z [R]": z,
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
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
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
            "X [R]": x,
            "Y [R]": y,
            "Z [R]": z,
            "Scalar": scalar,
        },
    )


def _build_disjoint_spherical_shell_dataset() -> _FakeDataset:
    """Private test helper: build two spherical shell layers with a radial gap."""
    theta_edges = np.array([0.0, 0.5 * math.pi, math.pi], dtype=float)
    phi_edges = np.array([0.0, math.pi, 2.0 * math.pi], dtype=float)
    shell_edges = ((1.0, 2.0), (4.0, 5.0))

    points: list[tuple[float, float, float]] = []
    corners: list[list[int]] = []

    for r0, r1 in shell_edges:
        node_index = -np.ones((2, theta_edges.size, phi_edges.size), dtype=np.int64)
        for ir, rr in enumerate((r0, r1)):
            for it, theta in enumerate(theta_edges):
                st = math.sin(float(theta))
                ct = math.cos(float(theta))
                for ip, phi in enumerate(phi_edges):
                    x = float(rr) * st * math.cos(float(phi))
                    y = float(rr) * st * math.sin(float(phi))
                    z = float(rr) * ct
                    node_index[ir, it, ip] = len(points)
                    points.append((x, y, z))

        for it in range(theta_edges.size - 1):
            for ip in range(phi_edges.size - 1):
                corners.append(
                    [
                        int(node_index[0, it, ip]),
                        int(node_index[1, it, ip]),
                        int(node_index[0, it + 1, ip]),
                        int(node_index[1, it + 1, ip]),
                        int(node_index[0, it, ip + 1]),
                        int(node_index[1, it, ip + 1]),
                        int(node_index[0, it + 1, ip + 1]),
                        int(node_index[1, it + 1, ip + 1]),
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
            "X [R]": x,
            "Y [R]": y,
            "Z [R]": z,
            "Scalar": scalar,
        },
    )


@pytest.fixture(scope="module")
def cartesian_octree_context() -> tuple[_FakeDataset, CartesianOctree, OctreeInterpolator]:
    """Build one reusable Cartesian octree/interpolator context for xyz-path tests."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="xyz")
    assert isinstance(tree, CartesianOctree)
    interp = OctreeInterpolator(ds, ["Scalar"], tree=tree)
    return ds, tree, interp


def test_cartesian_fixture_builds_xyz_tree(cartesian_octree_context) -> None:
    """Fixture should provide a bound Cartesian tree and xyz interpolator."""
    _ds, tree, interp = cartesian_octree_context
    assert isinstance(tree, CartesianOctree)
    assert tree.tree_coord == "xyz"
    assert interp.tree is tree


def test_cartesian_lookup_hits_cell_centers(cartesian_octree_context) -> None:
    """Cartesian lookup should resolve each cell center to its own cell id."""
    _ds, tree, _interp = cartesian_octree_context
    centers = np.array(tree.cell_centers, dtype=float)
    for cid in range(centers.shape[0]):
        q = centers[cid]
        hit = tree.lookup_point(q, coord="xyz")
        assert hit is not None
        assert int(hit.cell_id) == int(cid)


def test_cartesian_interpolation_matches_linear_xyz_field(cartesian_octree_context) -> None:
    """Cartesian trilinear interpolation should reconstruct the synthetic linear xyz field."""
    _ds, tree, interp = cartesian_octree_context
    rng = np.random.default_rng(42)
    n_cells = int(tree.cell_count)
    choose = rng.choice(n_cells, size=min(20, n_cells), replace=False)

    q = np.empty((choose.size, 3), dtype=float)
    expected = np.empty(choose.size, dtype=float)
    for i, cid in enumerate(choose.tolist()):
        lo, hi = tree.cell_bounds(int(cid), coord="xyz")
        u, v, w = rng.uniform(0.1, 0.9, size=3)
        x = float(lo[0] + u * (hi[0] - lo[0]))
        y = float(lo[1] + v * (hi[1] - lo[1]))
        z = float(lo[2] + w * (hi[2] - lo[2]))
        q[i] = (x, y, z)
        expected[i] = 1.2 * x - 0.3 * y + 0.8 * z + 0.5

    values, cell_ids = interp(q, return_cell_ids=True)
    assert np.array_equal(np.array(cell_ids, dtype=np.int64), np.array(choose, dtype=np.int64))
    assert np.allclose(np.array(values, dtype=float), expected, atol=1e-12, rtol=0.0)


def test_builder_build_rejects_missing_corners() -> None:
    """Builder should fail fast when dataset has no corners."""
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    ds = _FakeDataset(
        points=points,
        corners=None,
        variables={"X [R]": points[:, 0], "Y [R]": points[:, 1], "Z [R]": points[:, 2]},
    )
    with pytest.raises(ValueError, match="Dataset has no corners"):
        OctreeBuilder().build(ds)


def test_builder_build_rejects_unknown_tree_coord() -> None:
    """Builder should reject unsupported coordinate-system identifiers."""
    ds = _build_regular_dataset()
    with pytest.raises(ValueError, match="Unsupported tree_coord"):
        OctreeBuilder().build(ds, tree_coord="foo")


def test_builder_build_xyz_returns_cartesian_octree() -> None:
    """Builder should construct Cartesian octree when tree_coord='xyz'."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="xyz")
    assert isinstance(tree, CartesianOctree)
    assert tree.tree_coord == "xyz"


def test_builder_build_default_returns_cartesian_octree_subclass() -> None:
    """Default build path should return the Cartesian octree specialization."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds)
    assert isinstance(tree, CartesianOctree)


def test_builder_compute_phi_levels_rejects_missing_phi_source() -> None:
    """Phi-level computation should reject datasets lacking phi source fields."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    corners = np.array([[0, 1, 2]], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={"X [R]": points[:, 0], "Z [R]": points[:, 2]},
    )
    with pytest.raises(ValueError, match="Could not determine phi"):
        SphericalOctreeBuilder().compute_phi_levels(ds)


def test_builder_compute_phi_levels_rejects_bad_corner_rank() -> None:
    """Phi-level computation should reject non-2D corner arrays."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    corners = np.array([0, 1, 2], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={"X [R]": points[:, 0], "Y [R]": points[:, 1], "Z [R]": points[:, 2]},
    )
    with pytest.raises(ValueError, match="Expected 2D corner array"):
        SphericalOctreeBuilder().compute_phi_levels(ds)


def test_builder_compute_phi_levels_rejects_too_few_corners_per_cell() -> None:
    """Phi-level computation should reject cells with fewer than 3 corners."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    corners = np.array([[0, 1]], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={"X [R]": points[:, 0], "Y [R]": points[:, 1], "Z [R]": points[:, 2]},
    )
    with pytest.raises(ValueError, match="Need at least 3 corners per cell"):
        SphericalOctreeBuilder().compute_phi_levels(ds)


def test_builder_infer_levels_marks_non_dyadic_span_invalid() -> None:
    """Non-dyadic delta-phi spans should map to level -1."""
    levels = SphericalOctreeBuilder().infer_levels_from_delta_phi(np.array([1.0, 0.5, 0.3]))
    assert np.array_equal(levels, np.array([0, 1, -1], dtype=np.int64))


def test_builder_build_tree_rejects_all_invalid_levels() -> None:
    """Tree construction should fail when all provided levels are invalid."""
    ds = _build_regular_dataset()
    builder = OctreeBuilder()
    delta_phi, _center_phi, _cell_levels, _expected, _coarse = SphericalOctreeBuilder().compute_phi_levels(ds)
    all_invalid = np.full(delta_phi.shape, -1, dtype=np.int64)
    with pytest.raises(ValueError, match="No valid \\(>=0\\) levels available to infer octree"):
        builder._build_with_overrides(ds, tree_coord="rpa", cell_levels=all_invalid, bind=False)


def test_builder_handles_incompatible_blocks_aux_without_block_tree() -> None:
    """Incompatible BLOCKS aux metadata should be ignored by the octree builder."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "7 3x5x9"
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    assert tree.level_counts
    assert tree.leaf_shape[0] > 0


def test_octree_no_public_depth_for_level_helper() -> None:
    """Depth conversion is internal; no public depth-for-level helper is exposed."""
    tree = OctreeBuilder().build(_build_regular_dataset(), tree_coord="rpa")
    assert not hasattr(tree, "depth_for_level")


def test_octree_trace_ray_returns_empty_for_non_increasing_interval() -> None:
    """Ray trace should return empty when `t_end <= t_start`."""
    tree = OctreeBuilder().build(_build_regular_dataset(), tree_coord="rpa")
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    assert OctreeRayTracer(tree).trace(origin, direction, 1.0, 1.0) == []
    assert OctreeRayTracer(tree).trace(origin, direction, 2.0, 1.0) == []


def test_builder_build_bind_false_returns_unbound_tree_until_bind() -> None:
    """Builder with `bind=False` should return unbound tree requiring explicit bind."""
    ds = _build_regular_dataset()
    _delta_phi, _center_phi, cell_levels, _expected, _coarse = SphericalOctreeBuilder().compute_phi_levels(ds)
    tree = OctreeBuilder()._build_with_overrides(
        ds,
        tree_coord="rpa",
        cell_levels=cell_levels,
        bind=False,
    )
    with pytest.raises(ValueError, match="not bound to a dataset"):
        tree.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), coord="xyz")
    tree.bind(ds)
    assert tree.ds is ds


def test_builder_build_bind_false_stores_tree_coord_metadata() -> None:
    """Builder should store requested coordinate-system metadata in the tree."""
    ds = _build_regular_dataset()
    _delta_phi, _center_phi, cell_levels, _expected, _coarse = SphericalOctreeBuilder().compute_phi_levels(ds)
    tree = OctreeBuilder()._build_with_overrides(
        ds,
        tree_coord="xyz",
        cell_levels=cell_levels,
        bind=False,
    )
    assert isinstance(tree, CartesianOctree)
    assert tree.tree_coord == "xyz"


def test_builder_build_rejects_inconsistent_dataset_corners_for_spherical_inference() -> None:
    """Spherical build should fail when `ds.corners` is internally inconsistent."""
    ds = _build_regular_dataset(nr=1, ntheta=2, nphi=2)
    corners_full = np.array(ds.corners, copy=True)
    ds.corners = np.array(corners_full[:2], copy=True)

    with pytest.raises(ValueError, match="Could not infer integer finest n_axis0"):
        OctreeBuilder()._build_with_overrides(
            ds,
            tree_coord="rpa",
            cell_levels=None,
            bind=False,
        )


def test_lookup_runs_for_xyz_tree_coord() -> None:
    """Lookup APIs should run when the tree is tagged as Cartesian."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="xyz")
    hit_xyz = tree.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), coord="xyz")
    assert hit_xyz is not None
    assert not hasattr(tree, "lookup_rpa")


def test_lookup_gap_returns_none_for_disjoint_cartesian_cells() -> None:
    """Cartesian lookup should return miss for points in an uncovered bbox gap."""
    ds = _build_disjoint_xyz_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="xyz")
    q_gap = np.array([5.0, 0.5, 0.5], dtype=float)
    hit = tree.lookup_point(q_gap, coord="xyz")
    assert hit is None


def test_lookup_gap_returns_none_for_disjoint_spherical_shells() -> None:
    """Spherical lookup should return miss for points in a radial gap."""
    ds = _build_disjoint_spherical_shell_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    q_gap = np.array([3.0, 0.5 * math.pi, 0.5 * math.pi], dtype=float)
    hit = tree.lookup_point(q_gap, coord="rpa")
    assert hit is None

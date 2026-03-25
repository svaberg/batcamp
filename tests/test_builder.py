from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeBuilder
from batcamp.builder import _build_node_arrays
from batcamp.builder import _resolve_cell_levels
from batcamp.builder_cartesian import CartesianOctreeBuilder
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
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
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
            Octree.X_VAR: x_var,
            Octree.Y_VAR: y_var,
            Octree.Z_VAR: z_var,
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
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
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
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
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
            int(node_index[0, 2, 0]),
            int(node_index[1, 2, 0]),
            int(node_index[0, 0, 2]),
            int(node_index[1, 0, 2]),
            int(node_index[0, 2, 2]),
            int(node_index[1, 2, 2]),
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
                        int(node_index[ix, iy + 1, iz]),
                        int(node_index[ix + 1, iy + 1, iz]),
                        int(node_index[ix, iy, iz + 1]),
                        int(node_index[ix + 1, iy, iz + 1]),
                        int(node_index[ix, iy + 1, iz + 1]),
                        int(node_index[ix + 1, iy + 1, iz + 1]),
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
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Scalar": scalar,
        },
    )
    return ds, np.array(levels, dtype=np.int64)


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
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Scalar": scalar,
        },
    )


def _make_cartesian_tree(
    *,
    leaf_shape: tuple[int, int, int],
    root_shape: tuple[int, int, int],
    max_level: int,
    cell_levels: np.ndarray | None,
) -> Octree:
    levels = None if cell_levels is None else np.asarray(cell_levels, dtype=np.int64)
    valid = np.empty((0,), dtype=np.int64) if levels is None else levels[levels >= 0]
    min_level = int(np.min(valid)) if valid.size > 0 else 0
    count = int(valid.size)
    return Octree(
        leaf_shape=leaf_shape,
        root_shape=root_shape,
        is_full=False,
        level_counts=((min_level, count, count),),
        min_level=min_level,
        max_level=int(max_level),
        tree_coord="xyz",
        cell_levels=levels,
    )


def _make_spherical_tree(
    *,
    leaf_shape: tuple[int, int, int],
    root_shape: tuple[int, int, int],
    max_level: int,
    cell_levels: np.ndarray | None,
) -> Octree:
    levels = None if cell_levels is None else np.asarray(cell_levels, dtype=np.int64)
    valid = np.empty((0,), dtype=np.int64) if levels is None else levels[levels >= 0]
    min_level = int(np.min(valid)) if valid.size > 0 else 0
    count = int(valid.size)
    return Octree(
        leaf_shape=leaf_shape,
        root_shape=root_shape,
        is_full=False,
        level_counts=((min_level, count, count),),
        min_level=min_level,
        max_level=int(max_level),
        tree_coord="rpa",
        cell_levels=levels,
    )


@pytest.fixture(scope="module")
def cartesian_octree_context() -> tuple[_FakeDataset, Octree, OctreeInterpolator]:
    """Build one reusable Cartesian octree/interpolator context for xyz-path tests."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="xyz")
    assert isinstance(tree, Octree)
    interp = OctreeInterpolator(tree, ["Scalar"])
    return ds, tree, interp


def test_xyz_fixture_builds_tree(cartesian_octree_context) -> None:
    """Fixture should provide a bound Cartesian tree and xyz interpolator."""
    _ds, tree, interp = cartesian_octree_context
    assert isinstance(tree, Octree)
    assert tree.tree_coord == "xyz"
    assert interp.tree is tree


def test_xyz_lookup_hits_cell_centers(cartesian_octree_context) -> None:
    """Cartesian lookup should resolve each cell center to its own cell id."""
    _ds, tree, _interp = cartesian_octree_context
    centers = np.array(tree.cell_centers, dtype=float)
    for cid in range(centers.shape[0]):
        q = centers[cid]
        hit = tree.lookup_point(q, coord="xyz")
        assert hit is not None
        assert int(hit.cell_id) == int(cid)


def test_xyz_interp_matches_linear_field(cartesian_octree_context) -> None:
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


def test_xyz_lookup_reports_exact_adaptive_paths() -> None:
    """Adaptive Cartesian lookup should report exact discrete addresses and root-leaf paths."""
    ds, levels = _build_adaptive_xyz_dataset()
    tree = OctreeBuilder()._build(ds, tree_coord="xyz", cell_levels=levels, bind=True)

    coarse_hit = tree.lookup_point(np.array([0.25, 0.25, 0.25], dtype=float), coord="xyz")
    assert coarse_hit is not None
    assert coarse_hit.level == 0
    assert (coarse_hit.i0, coarse_hit.i1, coarse_hit.i2) == (0, 0, 0)
    assert coarse_hit.path == ((0, 0, 0),)

    fine_hit = tree.lookup_point(np.array([1.75, 0.75, 0.75], dtype=float), coord="xyz")
    assert fine_hit is not None
    assert fine_hit.level == 1
    assert (fine_hit.i0, fine_hit.i1, fine_hit.i2) == (3, 1, 1)
    assert fine_hit.path == ((1, 0, 0), (3, 1, 1))


def test_build_rejects_missing_corners() -> None:
    """Builder should fail fast when dataset has no corners."""
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    ds = _FakeDataset(
        points=points,
        corners=None,
        variables={Octree.X_VAR: points[:, 0], Octree.Y_VAR: points[:, 1], Octree.Z_VAR: points[:, 2]},
    )
    with pytest.raises(ValueError, match="Dataset has no corners"):
        OctreeBuilder().build(ds)


def test_build_rejects_unknown_tree_coord() -> None:
    """Builder should reject unsupported coordinate-system identifiers."""
    ds = _build_regular_dataset()
    with pytest.raises(ValueError, match="Unsupported tree_coord"):
        OctreeBuilder().build(ds, tree_coord="foo")


def test_build_xyz_returns_cartesian_tree() -> None:
    """Builder should construct Cartesian octree when tree_coord='xyz'."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="xyz")
    assert isinstance(tree, Octree)
    assert tree.tree_coord == "xyz"


def test_build_default_returns_cartesian_tree() -> None:
    """Default build path should return the Cartesian octree specialization."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds)
    assert isinstance(tree, Octree)
    assert tree.tree_coord == "xyz"


def test_compute_phi_levels_rejects_missing_phi_source() -> None:
    """Phi-level computation should reject datasets lacking phi source fields."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    corners = np.array([[0, 1, 2]], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={Octree.X_VAR: points[:, 0], Octree.Z_VAR: points[:, 2]},
    )
    with pytest.raises(ValueError, match="Could not determine phi"):
        SphericalOctreeBuilder.compute_delta_phi_and_levels(ds)


def test_compute_phi_levels_rejects_bad_corner_rank() -> None:
    """Phi-level computation should reject non-2D corner arrays."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    corners = np.array([0, 1, 2], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={Octree.X_VAR: points[:, 0], Octree.Y_VAR: points[:, 1], Octree.Z_VAR: points[:, 2]},
    )
    with pytest.raises(ValueError, match="Expected 2D corner array"):
        SphericalOctreeBuilder.compute_delta_phi_and_levels(ds)


def test_compute_phi_levels_rejects_too_few_corners() -> None:
    """Phi-level computation should reject cells with fewer than 3 corners."""
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    corners = np.array([[0, 1]], dtype=np.int64)
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={Octree.X_VAR: points[:, 0], Octree.Y_VAR: points[:, 1], Octree.Z_VAR: points[:, 2]},
    )
    with pytest.raises(ValueError, match="Need at least 3 corners per cell"):
        SphericalOctreeBuilder.compute_delta_phi_and_levels(ds)


def test_infer_levels_marks_non_dyadic_span_invalid() -> None:
    """Non-dyadic delta-phi spans should map to level -1."""
    levels = SphericalOctreeBuilder.infer_levels_from_span(np.array([1.0, 0.5, 0.3]))
    assert np.array_equal(levels, np.array([0, 1, -1], dtype=np.int64))


def test_build_tree_rejects_all_invalid_levels() -> None:
    """Tree construction should fail when all provided levels are invalid."""
    ds = _build_regular_dataset()
    builder = OctreeBuilder()
    delta_phi, _center_phi, _cell_levels, _expected, _coarse = SphericalOctreeBuilder.compute_delta_phi_and_levels(ds)
    all_invalid = np.full(delta_phi.shape, -1, dtype=np.int64)
    with pytest.raises(ValueError, match="No valid \\(>=0\\) levels available to infer octree"):
        builder._build(ds, tree_coord="rpa", cell_levels=all_invalid, bind=False)


def test_warns_on_incompatible_blocks_aux_without_block_tree(caplog: pytest.LogCaptureFixture) -> None:
    """Incompatible BLOCKS aux metadata should be ignored with a warning."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "7 3x5x9"
    with caplog.at_level(logging.WARNING, logger="batcamp.builder"):
        tree = OctreeBuilder().build(ds, tree_coord="rpa")
    assert tree.level_counts
    assert tree.leaf_shape[0] > 0
    assert any("BLOCKS" in rec.getMessage() and "does not match" in rec.getMessage() for rec in caplog.records)


def test_no_warning_on_compatible_blocks_aux(caplog: pytest.LogCaptureFixture) -> None:
    """Compatible BLOCKS aux metadata should not emit a warning."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "1 2x4x8"
    with caplog.at_level(logging.WARNING, logger="batcamp.builder"):
        tree = OctreeBuilder().build(ds, tree_coord="rpa")
    assert tree.level_counts
    assert not any("BLOCKS" in rec.getMessage() for rec in caplog.records)


def test_warns_on_blocks_count_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    """BLOCKS count mismatch should emit a warning."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "5 1x1x1"
    with caplog.at_level(logging.WARNING, logger="batcamp.builder"):
        tree = OctreeBuilder().build(ds, tree_coord="rpa")
    assert tree.level_counts
    assert any("BLOCKS" in rec.getMessage() and "does not match" in rec.getMessage() for rec in caplog.records)


def test_warns_on_impossible_blocks_count(caplog: pytest.LogCaptureFixture) -> None:
    """Impossible BLOCKS counts should emit a mismatch warning."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "1000 1x1x1"
    with caplog.at_level(logging.WARNING, logger="batcamp.builder"):
        tree = OctreeBuilder().build(ds, tree_coord="rpa")
    assert tree.level_counts
    assert any("BLOCKS" in rec.getMessage() and "does not match" in rec.getMessage() for rec in caplog.records)


def test_warns_on_unparseable_blocks_aux(caplog: pytest.LogCaptureFixture) -> None:
    """Unparseable BLOCKS aux metadata should be ignored with a warning."""
    ds = _build_regular_dataset()
    ds.aux["BLOCKS"] = "garbage"
    with caplog.at_level(logging.WARNING, logger="batcamp.builder"):
        tree = OctreeBuilder().build(ds, tree_coord="rpa")
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


def test_build_node_arrays_rejects_duplicate_leaf_addresses() -> None:
    """Sparse-node construction should reject duplicate leaf octree addresses."""
    with pytest.raises(ValueError, match="overlap at octree address"):
        _build_node_arrays(
            np.array([0, 0], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            tree_depth=0,
            label="Cartesian",
        )


def test_build_node_arrays_rejects_parent_child_overlap() -> None:
    """Sparse-node construction should reject explicit parent/child double occupancy."""
    with pytest.raises(ValueError, match="overlap across parent/child addresses"):
        _build_node_arrays(
            np.array([0, 1], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            tree_depth=1,
            label="Cartesian",
        )


def test_no_public_depth_for_level_helper() -> None:
    """Depth conversion is internal; no public depth-for-level helper is exposed."""
    tree = OctreeBuilder().build(_build_regular_dataset(), tree_coord="rpa")
    assert not hasattr(tree, "depth_for_level")


def test_regular_spherical_tree_uses_absolute_levels() -> None:
    """Uniform spherical trees should store root-relative absolute levels."""
    tree = OctreeBuilder().build(_build_regular_dataset(), tree_coord="rpa")
    assert tree.max_level == tree.depth
    assert tree.min_level == tree.max_level
    assert tree.cell_levels is not None
    assert np.all(tree.cell_levels == tree.max_level)


def test_build_materializes_exact_tree_state_before_lookup() -> None:
    """Builder should attach exact tree indices before lookup is ever built."""
    xyz_tree = OctreeBuilder()._build(_build_regular_xyz_dataset(), tree_coord="xyz", bind=False)
    assert xyz_tree.cell_levels is not None
    assert np.asarray(xyz_tree._i0, dtype=np.int64).shape == xyz_tree.cell_levels.shape
    assert np.asarray(xyz_tree._node_depth, dtype=np.int64).ndim == 1
    assert np.asarray(xyz_tree._node_child, dtype=np.int64).shape[1] == 8
    assert np.asarray(xyz_tree._root_node_ids, dtype=np.int64).ndim == 1
    assert not hasattr(xyz_tree, "_radial_edges")

    rpa_tree = OctreeBuilder()._build(_build_regular_dataset(), tree_coord="rpa", bind=False)
    assert rpa_tree.cell_levels is not None
    assert np.asarray(rpa_tree._i0, dtype=np.int64).shape == rpa_tree.cell_levels.shape
    assert np.asarray(rpa_tree._node_depth, dtype=np.int64).ndim == 1
    assert np.asarray(rpa_tree._node_child, dtype=np.int64).shape[1] == 8
    assert np.asarray(rpa_tree._root_node_ids, dtype=np.int64).ndim == 1
    assert not hasattr(rpa_tree, "_radial_edges")


def test_regular_spherical_lookup_materializes_exact_indices() -> None:
    """Regular spherical grids should expose exact root-relative cell indices."""
    tree = OctreeBuilder().build(_build_regular_dataset(), tree_coord="rpa")
    first = tree.hit_from_cell_id(0)
    last = tree.hit_from_cell_id(int(tree.cell_count) - 1)
    assert (first.level, first.i0, first.i1, first.i2) == (1, 0, 0, 0)
    assert first.path == ((0, 0, 0), (0, 0, 0))
    assert (last.level, last.i0, last.i1, last.i2) == (1, 1, 3, 7)
    assert last.path == ((0, 1, 3), (1, 3, 7))


def test_spherical_lookup_rejects_non_exact_geometry() -> None:
    """Irregular spherical geometry should fail in the builder."""
    regular = _build_regular_dataset()
    _delta_phi, _center_phi, cell_levels, _expected, _coarse = SphericalOctreeBuilder.compute_delta_phi_and_levels(regular)
    with pytest.raises(ValueError, match="Spherical cell .* inferred octree grid|no unique octree address"):
        OctreeBuilder()._build(
            _build_irregular_spherical_dataset(),
            tree_coord="rpa",
            cell_levels=cell_levels,
            bind=True,
        )


def test_adaptive_cartesian_tree_preserves_root_relative_levels() -> None:
    """Adaptive Cartesian builds should keep supplied root-relative levels unchanged."""
    ds, cell_levels = _build_adaptive_xyz_dataset()
    tree = OctreeBuilder()._build(ds, tree_coord="xyz", cell_levels=cell_levels, bind=False)
    assert tree.max_level == tree.depth == 1
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
            Octree.X_VAR: points[:, 0],
            Octree.Y_VAR: points[:, 1],
            Octree.Z_VAR: points[:, 2],
            "Scalar": points[:, 0],
        },
    )
    with pytest.raises(ValueError, match="inconsistent dx"):
        CartesianOctreeBuilder.infer_xyz_level_shapes(ds, np.asarray(corners, dtype=np.int64), np.zeros(3, dtype=np.int64))


def test_cartesian_infer_leaf_shape_rejects_missing_max_level() -> None:
    """Cartesian finest-shape inference should fail when the requested max level is absent."""
    ds = _build_regular_xyz_dataset(nx=2, ny=2, nz=2)
    with pytest.raises(ValueError, match="No cells found at max_level=1"):
        CartesianOctreeBuilder.infer_leaf_shape(
            ds,
            np.asarray(ds.corners, dtype=np.int64),
            np.zeros(int(np.asarray(ds.corners).shape[0]), dtype=np.int64),
            max_level=1,
        )


def test_cartesian_tree_state_requires_cell_levels() -> None:
    """Cartesian exact-state construction should fail without cell levels."""
    ds = _build_regular_xyz_dataset(nx=2, ny=2, nz=2)
    tree = _make_cartesian_tree(
        leaf_shape=(2, 2, 2),
        root_shape=(1, 1, 1),
        max_level=1,
        cell_levels=None,
    )
    with pytest.raises(ValueError, match="Cartesian tree state requires cell_levels"):
        CartesianOctreeBuilder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            ds=ds,
            corners=np.asarray(ds.corners, dtype=np.int64),
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
        CartesianOctreeBuilder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            ds=ds,
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
        CartesianOctreeBuilder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            ds=ds,
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
        CartesianOctreeBuilder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            ds=ds,
            corners=np.asarray(ds.corners, dtype=np.int64),
        )


def test_spherical_level_shapes_require_valid_levels() -> None:
    """Spherical angular-shape inference should fail when all levels are invalid."""
    ds = _build_regular_dataset()
    corners = np.asarray(ds.corners, dtype=np.int64)
    delta_phi, _center_phi, _levels, _expected, _coarse = SphericalOctreeBuilder.compute_delta_phi_and_levels(ds)
    with pytest.raises(ValueError, match="No valid \\(>=0\\) cell levels available for tree inference"):
        SphericalOctreeBuilder.infer_level_angular_shapes(
            ds,
            corners,
            delta_phi,
            np.full(delta_phi.shape, -1, dtype=np.int64),
        )


def test_spherical_infer_leaf_shape_rejects_noninteger_radial_count() -> None:
    """Spherical finest-shape inference should fail when weighted cells imply a noninteger radial count."""
    with pytest.raises(ValueError, match="Could not infer integer finest n_axis0"):
        SphericalOctreeBuilder.infer_leaf_shape(
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
        SphericalOctreeBuilder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            axis_rho_tol=tree.axis_rho_tol,
            ds=ds,
            corners=np.asarray(ds.corners, dtype=np.int64),
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
        SphericalOctreeBuilder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            axis_rho_tol=tree.axis_rho_tol,
            ds=ds,
            corners=np.asarray(ds.corners, dtype=np.int64),
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
        SphericalOctreeBuilder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            axis_rho_tol=tree.axis_rho_tol,
            ds=ds,
            corners=np.asarray(ds.corners, dtype=np.int64),
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
        SphericalOctreeBuilder.populate_tree_state(
            leaf_shape=tree.leaf_shape,
            max_level=tree.max_level,
            cell_levels=tree.cell_levels,
            axis_rho_tol=tree.axis_rho_tol,
            ds=ds,
            corners=np.asarray(ds.corners, dtype=np.int64),
        )


def test_build_bind_false_returns_unbound_until_bind() -> None:
    """Builder with `bind=False` should return unbound tree requiring explicit bind."""
    ds = _build_regular_dataset()
    _delta_phi, _center_phi, cell_levels, _expected, _coarse = SphericalOctreeBuilder.compute_delta_phi_and_levels(ds)
    tree = OctreeBuilder()._build(
        ds,
        tree_coord="rpa",
        cell_levels=cell_levels,
        bind=False,
    )
    with pytest.raises(ValueError, match="not bound to a dataset"):
        tree.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), coord="xyz")
    tree.bind(ds)
    assert tree.ds is ds


def test_build_bind_false_stores_tree_coord() -> None:
    """Builder should store requested coordinate-system metadata in the tree."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder()._build(ds, tree_coord="xyz", bind=False)
    assert isinstance(tree, Octree)
    assert tree.tree_coord == "xyz"


def test_build_rejects_inconsistent_corners_for_spherical_inference() -> None:
    """Spherical build should fail when `ds.corners` is internally inconsistent."""
    ds = _build_regular_dataset(nr=1, ntheta=2, nphi=2)
    corners_full = np.array(ds.corners, copy=True)
    ds.corners = np.array(corners_full[:2], copy=True)

    with pytest.raises(ValueError, match="Could not infer integer finest n_axis0"):
        OctreeBuilder()._build(
            ds,
            tree_coord="rpa",
            cell_levels=None,
            bind=False,
        )


def test_lookup_runs_for_xyz() -> None:
    """Lookup APIs should run when the tree is tagged as Cartesian."""
    ds = _build_regular_xyz_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="xyz")
    hit_xyz = tree.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), coord="xyz")
    assert hit_xyz is not None
    assert not hasattr(tree, "lookup_rpa")


def test_lookup_gap_none_for_disjoint_cartesian_cells() -> None:
    """Cartesian lookup should return miss for points in an uncovered bbox gap."""
    ds = _build_disjoint_xyz_dataset()
    tree = OctreeBuilder().build(ds, tree_coord="xyz")
    q_gap = np.array([5.0, 0.5, 0.5], dtype=float)
    hit = tree.lookup_point(q_gap, coord="xyz")
    assert hit is None


def test_lookup_gap_none_for_disjoint_spherical_shells() -> None:
    """Gappy spherical shells should be rejected by the builder."""
    ds = _build_disjoint_spherical_shell_dataset()
    with pytest.raises(ValueError, match="radial edge count does not match leaf_shape"):
        OctreeBuilder().build(ds, tree_coord="rpa")

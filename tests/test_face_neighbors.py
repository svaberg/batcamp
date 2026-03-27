from __future__ import annotations

import time

import numpy as np
import pytest
from batread import Dataset

from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp.constants import XYZ_VARS
from batcamp.face_neighbors import OctreeFaceNeighbors
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh
from sample_data_helper import data_file


def _build_cartesian_uniform_tree() -> Octree:
    """Private test helper: small uniform Cartesian tree with 8 cells."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        z_edges=np.array([0.0, 1.0, 2.0], dtype=float),
    )
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: points[:, 0],
            XYZ_VARS[1]: points[:, 1],
            XYZ_VARS[2]: points[:, 2],
        },
    )
    return OctreeBuilder().from_ds(ds, tree_coord="xyz")


def _build_spherical_uniform_tree() -> Octree:
    """Private test helper: small uniform spherical tree with periodic azimuth."""
    points, corners = _build_spherical_hex_mesh(
        nr=1,
        npolar=2,
        nazimuth=4,
        r_min=1.0,
        r_max=2.0,
    )
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: points[:, 0],
            XYZ_VARS[1]: points[:, 1],
            XYZ_VARS[2]: points[:, 2],
        },
    )
    return OctreeBuilder().from_ds(ds, tree_coord="rpa")


def _build_two_level_topology_tree() -> Octree:
    """Private test helper: synthetic 2-level Cartesian leaf frontier."""
    coarse = [(i0, i1, i2) for i0 in (0, 1) for i1 in (0, 1) for i2 in (0, 1) if (i0, i1, i2) != (0, 0, 0)]
    fine = [(i0, i1, i2) for i0 in (0, 1) for i1 in (0, 1) for i2 in (0, 1)]

    levels = np.concatenate(
        (
            np.zeros(len(coarse), dtype=np.int64),
            np.ones(len(fine), dtype=np.int64),
        )
    )
    i0 = np.concatenate(
        (
            np.array([idx[0] for idx in coarse], dtype=np.int64),
            np.array([idx[0] for idx in fine], dtype=np.int64),
        )
    )
    i1 = np.concatenate(
        (
            np.array([idx[1] for idx in coarse], dtype=np.int64),
            np.array([idx[1] for idx in fine], dtype=np.int64),
        )
    )
    i2 = np.concatenate(
        (
            np.array([idx[2] for idx in coarse], dtype=np.int64),
            np.array([idx[2] for idx in fine], dtype=np.int64),
        )
    )
    n_cells = int(levels.shape[0])
    cell_points: list[np.ndarray] = []
    cell_corners = np.empty((n_cells, 8), dtype=np.int64)
    for cell_id in range(n_cells):
        x0 = 2.0 * float(cell_id)
        pts = np.array(
            [
                [x0 + 0.0, 0.0, 0.0],
                [x0 + 1.0, 0.0, 0.0],
                [x0 + 0.0, 1.0, 0.0],
                [x0 + 1.0, 1.0, 0.0],
                [x0 + 0.0, 0.0, 1.0],
                [x0 + 1.0, 0.0, 1.0],
                [x0 + 0.0, 1.0, 1.0],
                [x0 + 1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        start = 8 * cell_id
        cell_points.append(pts)
        cell_corners[cell_id] = np.arange(start, start + 8, dtype=np.int64)
    points = np.vstack(cell_points)
    tree = Octree(
        root_shape=(2, 2, 2),
        tree_coord="xyz",
        cell_levels=levels,
        cell_ijk=np.column_stack((i0, i1, i2)),
        points=points,
        corners=cell_corners,
    )
    return tree


def _assert_basic_face_neighbor_invariants(topo: OctreeFaceNeighbors) -> None:
    """Private test helper: validate basic face-neighbor graph invariants."""
    assert topo.node_count > 0
    assert topo.face_counts.shape == (topo.node_count, 6)
    assert topo.face_offsets.shape == (topo.node_count * 6 + 1,)
    assert int(topo.face_offsets[0]) == 0
    assert int(topo.face_offsets[-1]) == int(topo.face_neighbors.size)
    assert np.all(np.diff(topo.face_offsets) >= 0)
    np.testing.assert_array_equal(np.diff(topo.face_offsets), topo.face_counts.reshape(-1))

    if topo.face_neighbors.size > 0:
        assert np.all(topo.face_neighbors >= 0)
        assert np.all(topo.face_neighbors < topo.node_count)

    for node_id in range(topo.node_count):
        for face in range(6):
            neighbors = topo.face_neighbor_ids(node_id, face)
            if neighbors.size == 0:
                continue
            assert not np.any(neighbors == node_id)
            assert np.unique(neighbors).size == neighbors.size


def _build_face_neighbors_timed(
    tree: Octree,
    *,
    label: str,
    max_level: int | None = None,
) -> tuple[OctreeFaceNeighbors, float]:
    """Private test helper: build face neighbors and emit a small timing line."""
    t0 = time.perf_counter()
    topo = tree.face_neighbors(max_level=max_level)
    elapsed_s = time.perf_counter() - t0
    print(
        f"face-neighbors {label}: {elapsed_s * 1.0e3:.2f} ms, "
        f"nodes={topo.node_count}, face_refs={int(topo.face_neighbors.size)}"
    )
    return topo, elapsed_s

def _time_call(func, /, *args, **kwargs):
    """Private test helper: time one function call and return result plus seconds."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_s = time.perf_counter() - t0
    return result, elapsed_s

@pytest.fixture(scope="session", autouse=True)
def _warm_topology_numba() -> None:
    """Private test helper: compile face-neighbors kernels before timing assertions/prints."""
    _build_cartesian_uniform_tree().face_neighbors()
    _build_spherical_uniform_tree().face_neighbors()


@pytest.mark.parametrize(
    "tree_builder",
    [_build_cartesian_uniform_tree, _build_spherical_uniform_tree],
    ids=["cartesian_uniform", "spherical_uniform"],
)
def test_uniform_topology_basic_invariants(tree_builder, record_property) -> None:
    tree = tree_builder()
    topo, elapsed_s = _build_face_neighbors_timed(tree, label=tree.tree_coord)
    record_property("build_topology_ms", round(elapsed_s * 1.0e3, 3))
    _assert_basic_face_neighbor_invariants(topo)
    assert np.all(topo.face_counts <= 1)


def test_octree_face_neighbors_method_reuses_cached_graph() -> None:
    tree = _build_two_level_topology_tree()
    full = tree.face_neighbors()
    repeat = tree.face_neighbors()
    coarse = tree.face_neighbors(max_level=0)

    assert full is repeat
    assert int(full.max_level) == int(tree.max_level)
    assert int(coarse.max_level) == 0


def test_cartesian_uniform_topology_neighbors_are_bidirectional(record_property) -> None:
    tree = _build_cartesian_uniform_tree()
    topo, elapsed_s = _build_face_neighbors_timed(tree, label="cartesian_bidirectional")
    record_property("build_topology_ms", round(elapsed_s * 1.0e3, 3))

    assert topo.node_count == int(tree.cell_count)
    np.testing.assert_array_equal(
        topo.face_counts.sum(axis=1),
        np.full(topo.node_count, 3, dtype=np.int64),
    )

    for node_id in range(topo.node_count):
        for face in range(6):
            opposite = face ^ 1
            neighbors = topo.face_neighbor_ids(node_id, face)
            for nbr in neighbors:
                back = topo.face_neighbor_ids(int(nbr), opposite)
                assert np.any(back == node_id)


def test_spherical_uniform_topology_wraps_azimuth_faces(record_property) -> None:
    tree = _build_spherical_uniform_tree()
    topo, elapsed_s = _build_face_neighbors_timed(tree, label="spherical_wrap")
    record_property("build_topology_ms", round(elapsed_s * 1.0e3, 3))

    assert topo.periodic_i2
    np.testing.assert_array_equal(
        topo.face_counts.sum(axis=1),
        np.full(topo.node_count, 3, dtype=np.int64),
    )

    i2_max = int(np.max(topo.cell_ijk[:, 2]))
    nodes_on_zero = np.flatnonzero(topo.cell_ijk[:, 2] == 0)
    assert nodes_on_zero.size > 0
    for node_id in nodes_on_zero:
        left = topo.face_neighbor_ids(int(node_id), 4)
        assert left.size == 1
        assert int(topo.cell_ijk[int(left[0]), 2]) == i2_max


def test_max_level_cutoff_reduces_frontier_size(record_property) -> None:
    tree = _build_two_level_topology_tree()

    topo_full, elapsed_full_s = _build_face_neighbors_timed(tree, label="cutoff_full", max_level=1)
    topo_coarse, elapsed_coarse_s = _build_face_neighbors_timed(tree, label="cutoff_coarse", max_level=0)
    record_property("build_topology_full_ms", round(elapsed_full_s * 1.0e3, 3))
    record_property("build_topology_coarse_ms", round(elapsed_coarse_s * 1.0e3, 3))

    assert topo_full.max_level == 1
    assert topo_coarse.max_level == 0
    assert topo_coarse.node_count <= topo_full.node_count
    assert topo_coarse.node_count < topo_full.node_count
    assert topo_full.node_count == 15
    assert topo_coarse.node_count == 8

    _assert_basic_face_neighbor_invariants(topo_full)
    _assert_basic_face_neighbor_invariants(topo_coarse)
    assert np.all(topo_full.face_counts <= 4)
    assert np.all(topo_coarse.face_counts <= 1)
    assert np.all(topo_full.face_counts.sum(axis=1) <= 24)


_TOPOLOGY_SAMPLE_CASES = [
    pytest.param(
        "3d__var_1_n00000000.plt",
        "rpa",
        id="provided_example_file",
    ),
    pytest.param(
        "3d__var_4_n00044000.plt",
        "rpa",
        id="pooch_sc",
        marks=[pytest.mark.pooch, pytest.mark.slow],
    ),
    pytest.param(
        "3d__var_4_n00005000.plt",
        "xyz",
        id="pooch_ih",
        marks=[pytest.mark.pooch, pytest.mark.slow],
    ),
]

@pytest.mark.parametrize("file_name,tree_coord", _TOPOLOGY_SAMPLE_CASES)
def test_topological_neighborhood_on_sample_files(file_name: str, tree_coord: str, record_property) -> None:
    path, resolve_path_s = _time_call(data_file, file_name)
    ds, read_dataset_s = _time_call(Dataset.from_file, str(path))
    tree, build_tree_s = _time_call(OctreeBuilder().from_ds, ds, tree_coord=tree_coord)
    topo, build_topology_s = _build_face_neighbors_timed(tree, label=file_name)
    _unused, validate_invariants_s = _time_call(_assert_basic_face_neighbor_invariants, topo)

    print(
        f"sample {file_name}: "
        f"resolve={resolve_path_s * 1.0e3:.2f} ms, "
        f"read={read_dataset_s * 1.0e3:.2f} ms, "
        f"tree={build_tree_s * 1.0e3:.2f} ms, "
        f"topology={build_topology_s * 1.0e3:.2f} ms, "
        f"validate={validate_invariants_s * 1.0e3:.2f} ms"
    )
    record_property("resolve_path_ms", round(resolve_path_s * 1.0e3, 3))
    record_property("read_dataset_ms", round(read_dataset_s * 1.0e3, 3))
    record_property("build_tree_ms", round(build_tree_s * 1.0e3, 3))
    record_property("build_topology_ms", round(build_topology_s * 1.0e3, 3))
    record_property("validate_invariants_ms", round(validate_invariants_s * 1.0e3, 3))
    assert topo.node_count == int(tree.cell_count)
    assert topo.max_level == int(tree.max_level)
    assert topo.periodic_i2 == (tree_coord == "rpa")
    assert int(topo.face_neighbors.size) > 0

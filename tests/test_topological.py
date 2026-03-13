from __future__ import annotations

import numpy as np
import pytest
from starwinds_readplt.dataset import Dataset

from batcamp import Octree
from batcamp.topological import build_topological_neighborhood
from batcamp.topological import TopologicalNeighborhood
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
            Octree.X_VAR: points[:, 0],
            Octree.Y_VAR: points[:, 1],
            Octree.Z_VAR: points[:, 2],
        },
    )
    return Octree.from_dataset(ds, tree_coord="xyz")


def _build_spherical_uniform_tree() -> Octree:
    """Private test helper: small uniform spherical tree with periodic azimuth."""
    points, corners = _build_spherical_hex_mesh(
        nr=1,
        ntheta=2,
        nphi=4,
        r_min=1.0,
        r_max=2.0,
    )
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            Octree.X_VAR: points[:, 0],
            Octree.Y_VAR: points[:, 1],
            Octree.Z_VAR: points[:, 2],
        },
    )
    return Octree.from_dataset(ds, tree_coord="rpa")


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

    tree = Octree(
        leaf_shape=(4, 4, 4),
        root_shape=(2, 2, 2),
        is_full=False,
        level_counts=((0, len(coarse), len(coarse) * 8), (1, len(fine), len(fine))),
        min_level=0,
        max_level=1,
        tree_coord="xyz",
        cell_levels=levels,
    )
    tree._i0 = i0
    tree._i1 = i1
    tree._i2 = i2
    tree._lookup_ready = True
    return tree


def _assert_basic_topology_invariants(topo: TopologicalNeighborhood) -> None:
    """Private test helper: validate basic neighborhood graph invariants."""
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


@pytest.mark.parametrize(
    "tree_builder",
    [_build_cartesian_uniform_tree, _build_spherical_uniform_tree],
    ids=["cartesian_uniform", "spherical_uniform"],
)
def test_uniform_topology_basic_invariants(tree_builder) -> None:
    tree = tree_builder()
    topo = build_topological_neighborhood(tree)
    _assert_basic_topology_invariants(topo)
    assert np.all(topo.face_counts <= 1)


def test_cartesian_uniform_topology_neighbors_are_bidirectional() -> None:
    tree = _build_cartesian_uniform_tree()
    topo = build_topological_neighborhood(tree)

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


def test_spherical_uniform_topology_wraps_azimuth_faces() -> None:
    tree = _build_spherical_uniform_tree()
    topo = build_topological_neighborhood(tree)

    assert topo.periodic_i2
    np.testing.assert_array_equal(
        topo.face_counts.sum(axis=1),
        np.full(topo.node_count, 3, dtype=np.int64),
    )

    i2_max = int(np.max(topo.i2))
    nodes_on_zero = np.flatnonzero(topo.i2 == 0)
    assert nodes_on_zero.size > 0
    for node_id in nodes_on_zero:
        left = topo.face_neighbor_ids(int(node_id), 4)
        assert left.size == 1
        assert int(topo.i2[int(left[0])]) == i2_max


def test_max_level_cutoff_reduces_frontier_size() -> None:
    tree = _build_two_level_topology_tree()

    topo_full = build_topological_neighborhood(tree, max_level=1)
    topo_coarse = build_topological_neighborhood(tree, max_level=0)

    assert topo_full.max_level == 1
    assert topo_coarse.max_level == 0
    assert topo_coarse.node_count <= topo_full.node_count
    assert topo_coarse.node_count < topo_full.node_count
    assert topo_full.node_count == 15
    assert topo_coarse.node_count == 8

    _assert_basic_topology_invariants(topo_full)
    _assert_basic_topology_invariants(topo_coarse)
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
def test_topological_neighborhood_on_sample_files(file_name: str, tree_coord: str) -> None:
    ds = Dataset.from_file(str(data_file(file_name)))
    tree = Octree.from_dataset(ds, tree_coord=tree_coord)
    topo = build_topological_neighborhood(tree)

    _assert_basic_topology_invariants(topo)
    assert topo.node_count == int(tree.cell_count)
    assert topo.max_level == int(tree.max_level)
    assert topo.periodic_i2 == (tree_coord == "rpa")

    # Real meshes should have at least some connected faces.
    assert int(topo.face_neighbors.size) > 0

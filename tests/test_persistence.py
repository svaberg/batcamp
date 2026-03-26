from __future__ import annotations

import numpy as np
import pytest

from batcamp import OCTREE_FILE_VERSION
from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp.constants import XYZ_VARS
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh


@pytest.fixture(scope="module")
def tree_dataset_pair() -> tuple[Octree, _FakeDataset]:
    """Return one synthetic spherical octree and source dataset for persistence tests."""
    points, corners = _build_spherical_hex_mesh(
        nr=2,
        ntheta=4,
        nphi=8,
        r_min=1.0,
        r_max=3.0,
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
    tree = OctreeBuilder().build(ds, tree_coord="rpa")
    return tree, ds


def test_save_load_roundtrip_preserves_core_arrays(tree_dataset_pair, tmp_path) -> None:
    """Round-trip save/load preserves core octree metadata arrays."""
    tree, ds = tree_dataset_pair
    path = tmp_path / "persist" / "tree_roundtrip.npz"
    tree.save(path)

    loaded = Octree.load(path, ds=ds)
    assert isinstance(loaded, Octree)
    assert loaded.leaf_shape == tree.leaf_shape
    assert loaded.root_shape == tree.root_shape
    assert loaded.level_counts == tree.level_counts
    assert loaded.tree_coord == tree.tree_coord

    assert loaded.cell_levels is not None and tree.cell_levels is not None
    assert np.array_equal(loaded.cell_levels, tree.cell_levels)
    assert np.array_equal(
        np.asarray(loaded._cell_ijk[: loaded.cell_levels.shape[0]], dtype=np.int64),
        np.asarray(tree._cell_ijk[: tree.cell_levels.shape[0]], dtype=np.int64),
    )
    assert np.array_equal(np.asarray(loaded._cell_depth, dtype=np.int64), np.asarray(tree._cell_depth, dtype=np.int64))
    assert np.array_equal(np.asarray(loaded._cell_ijk, dtype=np.int64), np.asarray(tree._cell_ijk, dtype=np.int64))
    assert np.array_equal(np.asarray(loaded._cell_child, dtype=np.int64), np.asarray(tree._cell_child, dtype=np.int64))
    assert np.array_equal(np.asarray(loaded._root_cell_ids, dtype=np.int64), np.asarray(tree._root_cell_ids, dtype=np.int64))

    q_xyz = np.array([1.0, 0.0, 0.0], dtype=float)
    assert int(tree.lookup_points(q_xyz, coord="xyz")[0]) == int(loaded.lookup_points(q_xyz, coord="xyz")[0])
    assert np.array_equal(np.asarray(loaded._radial_edges, dtype=float), np.asarray(tree._radial_edges, dtype=float))
    assert np.allclose(
        np.asarray(loaded.cell_bounds[:, 0, 0], dtype=float),
        np.asarray(tree.cell_bounds[:, 0, 0], dtype=float),
    )
    assert np.allclose(
        np.asarray(loaded.cell_bounds[:, 0, 1], dtype=float),
        np.asarray(tree.cell_bounds[:, 0, 1], dtype=float),
    )
    assert np.allclose(
        np.asarray(loaded.cell_bounds[:, 1, 0], dtype=float),
        np.asarray(tree.cell_bounds[:, 1, 0], dtype=float),
    )
    assert np.allclose(
        np.asarray(loaded.cell_bounds[:, 1, 1], dtype=float),
        np.asarray(tree.cell_bounds[:, 1, 1], dtype=float),
    )
    assert np.allclose(
        np.asarray(loaded.cell_bounds[:, 2, 0], dtype=float),
        np.asarray(tree.cell_bounds[:, 2, 0], dtype=float),
    )
    assert np.allclose(
        np.asarray(loaded.cell_bounds[:, 2, 1], dtype=float),
        np.asarray(tree.cell_bounds[:, 2, 1], dtype=float),
    )
    assert float(loaded._domain_bounds[0, 0]) == pytest.approx(float(tree._domain_bounds[0, 0]))
    assert float(loaded._domain_bounds[0, 1]) == pytest.approx(float(tree._domain_bounds[0, 1]))


@pytest.mark.slow
def test_load_requires_dataset_binding(tree_dataset_pair, tmp_path) -> None:
    """Loading requires dataset so lookup geometry is always dataset-bound."""
    tree, ds = tree_dataset_pair
    path = tmp_path / "tree_bound_load.npz"
    tree.save(path)

    loaded = Octree.load(path, ds=ds)
    assert loaded.ds is ds
    assert int(loaded.lookup_points(np.array([1.0, 0.0, 0.0], dtype=float), coord="xyz")[0]) >= 0


@pytest.mark.slow
def test_persistence_omits_legacy_corners_payload(tree_dataset_pair, tmp_path) -> None:
    """Persistence file should contain only minimal saved state."""
    tree, ds = tree_dataset_pair
    path = tmp_path / "tree_no_corner_payload.npz"
    tree.save(path)
    loaded = Octree.load(path, ds=ds)
    with np.load(path, allow_pickle=False) as data:
        assert "corners" not in data.files
        assert "has_corners" not in data.files
        assert set(data.files) == {"version", "tree_coord", "root_shape", "cell_levels", "cell_ijk"}
    assert int(loaded.lookup_points(np.array([1.0, 0.0, 0.0], dtype=float), coord="xyz")[0]) >= 0


def test_load_rejects_unsupported_file_version(tree_dataset_pair, tmp_path) -> None:
    """Loader rejects persisted files with unknown serialization version."""
    tree, _ds = tree_dataset_pair
    good_path = tmp_path / "tree_good.npz"
    bad_path = tmp_path / "tree_bad_version.npz"
    tree.save(good_path)

    payload: dict[str, np.ndarray] = {}
    with np.load(good_path, allow_pickle=False) as data:
        for key in data.files:
            payload[key] = np.array(data[key], copy=True)
    payload["version"] = np.array(int(OCTREE_FILE_VERSION) + 100, dtype=np.int64)
    np.savez_compressed(bad_path, **payload)

    with pytest.raises(ValueError, match="Unsupported octree file version"):
        Octree.load(bad_path, ds=tree_dataset_pair[1])


def test_load_rejects_unsupported_tree_coord(tree_dataset_pair, tmp_path) -> None:
    """Loader should reject serialized metadata with unknown coordinate-system tags."""
    tree, _ds = tree_dataset_pair
    good_path = tmp_path / "tree_good_coords.npz"
    bad_path = tmp_path / "tree_bad_coords.npz"
    tree.save(good_path)

    payload: dict[str, np.ndarray] = {}
    with np.load(good_path, allow_pickle=False) as data:
        for key in data.files:
            payload[key] = np.array(data[key], copy=True)
    payload["tree_coord"] = np.array("bad_coords")
    np.savez_compressed(bad_path, **payload)

    with pytest.raises(ValueError, match="Unsupported tree_coord"):
        Octree.load(bad_path, ds=tree_dataset_pair[1])

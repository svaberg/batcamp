from __future__ import annotations

import numpy as np
import pytest
from batread.dataset import Dataset

from sample_data_helper import data_file
from batcamp import Octree
from batcamp.octree import OCTREE_FILE_VERSION
from batcamp.spherical import SphericalOctree


@pytest.fixture(scope="module")
def tree_dataset_pair() -> tuple[Octree, Dataset]:
    """Return one built octree and source dataset for persistence tests."""
    input_file = data_file("3d__var_2_n00060005.plt")
    assert input_file.exists(), f"Missing sample file: {input_file}"
    ds = Dataset.from_file(str(input_file))
    tree = Octree.from_dataset(ds, tree_coord="rpa")
    return tree, ds


def test_save_load_roundtrip_preserves_core_arrays(tree_dataset_pair, tmp_path) -> None:
    """Round-trip save/load preserves core octree metadata arrays."""
    tree, ds = tree_dataset_pair
    path = tmp_path / "persist" / "tree_roundtrip.npz"
    tree.save(path)

    loaded = Octree.load(path, ds=ds)
    assert isinstance(loaded, SphericalOctree)
    assert loaded.leaf_shape == tree.leaf_shape
    assert loaded.root_shape == tree.root_shape
    assert loaded.level_counts == tree.level_counts
    assert loaded.depth == tree.depth
    assert loaded.tree_coord == tree.tree_coord

    assert loaded.cell_levels is not None and tree.cell_levels is not None
    assert np.array_equal(loaded.cell_levels, tree.cell_levels)

    q_xyz = np.array([1.0, 0.0, 0.0], dtype=float)
    hit_tree = tree.lookup_point(q_xyz, coord="xyz")
    hit_loaded = loaded.lookup_point(q_xyz, coord="xyz")
    assert hit_tree is not None
    assert hit_loaded is not None
    assert int(hit_tree.cell_id) == int(hit_loaded.cell_id)


@pytest.mark.slow
def test_load_requires_dataset_binding(tree_dataset_pair, tmp_path) -> None:
    """Loading requires dataset so lookup geometry is always dataset-bound."""
    tree, ds = tree_dataset_pair
    path = tmp_path / "tree_bound_load.npz"
    tree.save(path)

    loaded = Octree.load(path, ds=ds)
    assert loaded.ds is ds
    hit = loaded.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), coord="xyz")
    assert hit is not None


@pytest.mark.slow
def test_persistence_omits_legacy_corners_payload(tree_dataset_pair, tmp_path) -> None:
    """Persistence file should not contain legacy corners payload keys."""
    tree, ds = tree_dataset_pair
    path = tmp_path / "tree_no_corner_payload.npz"
    tree.save(path)
    loaded = Octree.load(path, ds=ds)
    with np.load(path, allow_pickle=False) as data:
        assert "corners" not in data.files
        assert "has_corners" not in data.files
    hit = loaded.lookup_point(np.array([1.0, 0.0, 0.0], dtype=float), coord="xyz")
    assert hit is not None


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

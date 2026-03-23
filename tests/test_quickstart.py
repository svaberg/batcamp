import numpy as np
import pytest
from batread import Dataset

from batcamp import Octree, OctreeInterpolator
from batcamp.interpolator import _infer_tree_coord_from_geometry
from sample_data_helper import data_file


_CASES = [
    pytest.param("3d__var_2_n00060005.plt", "rpa", id="local_rpa"),
    pytest.param("3d__var_2_n00006003.plt", "xyz", id="local_xyz"),
]


@pytest.mark.parametrize("name,tree_coord", _CASES)
def test_infer_tree_coord_from_geometry(name: str, tree_coord: str) -> None:
    """Inference contract: geometry-based coord inference matches expected."""
    ds = Dataset.from_file(str(data_file(name)))
    assert str(_infer_tree_coord_from_geometry(ds)) == tree_coord


@pytest.mark.parametrize("name,tree_coord", _CASES)
def test_tree_build_uses_expected_coord(name: str, tree_coord: str) -> None:
    """Tree contract: correct tree_coord builds; wrong tree_coord fails."""
    ds = Dataset.from_file(str(data_file(name)))
    wrong_tree_coord = "xyz" if tree_coord == "rpa" else "rpa"
    assert str(Octree.from_dataset(ds, tree_coord=tree_coord).tree_coord) == tree_coord
    with pytest.raises(ValueError):
        Octree.from_dataset(ds, tree_coord=wrong_tree_coord)


@pytest.mark.parametrize("name,tree_coord", _CASES)
def test_tree_build_default_matches_expected(name: str, tree_coord: str) -> None:
    """Tree contract: default `Octree.from_dataset(ds)` resolves correct tree type."""
    ds = Dataset.from_file(str(data_file(name)))
    assert str(Octree.from_dataset(ds).tree_coord) == tree_coord


@pytest.mark.parametrize("name,tree_coord", _CASES)
def test_explicit_tree_equals_auto_tree(name: str, tree_coord: str) -> None:
    """Interpolator contract: explicit prebuilt tree and auto-tree must match."""
    ds = Dataset.from_file(str(data_file(name)))
    tree = Octree.from_dataset(ds, tree_coord=tree_coord)
    queries = np.asarray(tree.cell_centers[:16], dtype=float)

    interp_explicit = OctreeInterpolator(ds, ["Rho [g/cm^3]"], tree=tree)
    interp_auto = OctreeInterpolator(ds, ["Rho [g/cm^3]"])
    vals_explicit, cell_ids_explicit = interp_explicit(queries, return_cell_ids=True)
    vals_auto, cell_ids_auto = interp_auto(queries, return_cell_ids=True)

    np.testing.assert_array_equal(np.asarray(cell_ids_explicit), np.asarray(cell_ids_auto))
    np.testing.assert_allclose(
        np.asarray(vals_explicit),
        np.asarray(vals_auto),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )

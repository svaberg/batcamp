import numpy as np
import pytest
from batread import Dataset

from batcamp import Octree, OctreeInterpolator, build_octree_from_ds
from batcamp.builder import infer_tree_coord_from_geometry
from octree_test_support import cell_bounds
from sample_data_helper import data_file


_CASES = [
    pytest.param("3d__var_1_n00000000.plt", "rpa", id="example_file"),
    pytest.param(
        "3d__var_4_n00044000.plt",
        "rpa",
        id="pooch_sc",
        marks=pytest.mark.pooch,
    ),
    pytest.param(
        "3d__var_4_n00005000.plt",
        "xyz",
        id="pooch_ih",
        marks=pytest.mark.pooch,
    ),
]


def _midpoint_queries_xyz(tree: Octree, n_query: int) -> np.ndarray:
    queries = []
    for cell_id in range(min(int(n_query), int(tree.cell_count))):
        lo, hi = cell_bounds(tree, cell_id, coord="xyz")
        queries.append(0.5 * (np.asarray(lo, dtype=float) + np.asarray(hi, dtype=float)))
    return np.asarray(queries, dtype=float)


@pytest.mark.parametrize("name,tree_coord", _CASES)
def test_infer_tree_coord_from_geometry(name: str, tree_coord: str) -> None:
    """Inference contract: geometry-based coord inference matches expected."""
    ds = Dataset.from_file(str(data_file(name)))
    points = np.column_stack((np.asarray(ds["X [R]"]), np.asarray(ds["Y [R]"]), np.asarray(ds["Z [R]"])))
    assert str(infer_tree_coord_from_geometry(points, np.asarray(ds.corners, dtype=np.int64))) == tree_coord


@pytest.mark.parametrize("name,tree_coord", _CASES)
def test_tree_build_uses_expected_coord(name: str, tree_coord: str) -> None:
    """Tree contract: correct tree_coord builds; wrong tree_coord fails."""
    ds = Dataset.from_file(str(data_file(name)))
    wrong_tree_coord = "xyz" if tree_coord == "rpa" else "rpa"
    assert str(build_octree_from_ds(ds, tree_coord=tree_coord).tree_coord) == tree_coord
    with pytest.raises(ValueError):
        build_octree_from_ds(ds, tree_coord=wrong_tree_coord)


@pytest.mark.parametrize("name,tree_coord", _CASES)
def test_tree_build_default_matches_expected(name: str, tree_coord: str) -> None:
    """Tree contract: default `build_octree_from_ds(ds)` resolves correct tree type."""
    ds = Dataset.from_file(str(data_file(name)))
    assert str(build_octree_from_ds(ds).tree_coord) == tree_coord


@pytest.mark.parametrize("name,tree_coord", _CASES)
def test_explicit_tree_equals_auto_tree(name: str, tree_coord: str) -> None:
    """Interpolator contract: explicit and inferred trees should interpolate identically."""
    ds = Dataset.from_file(str(data_file(name)))
    tree_explicit = build_octree_from_ds(ds, tree_coord=tree_coord)
    tree_auto = build_octree_from_ds(ds)
    queries = _midpoint_queries_xyz(tree_explicit, 16)

    values = np.asarray(ds["Rho [g/cm^3]"])
    interp_explicit = OctreeInterpolator(tree_explicit, values)
    interp_auto = OctreeInterpolator(tree_auto, values)
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

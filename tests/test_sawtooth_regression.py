from __future__ import annotations

import math

import numpy as np
import pytest
from batread import Dataset

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp.shared import XYZ_VARS
from tests.octree_test_support import find_shared_face_pair
from sample_data_helper import local_data_file


@pytest.fixture(scope="module")
def sc_context() -> tuple[Dataset, Octree]:
    """Return one real spherical sample dataset with its spherical tree."""
    ds = Dataset.from_file(str(local_data_file("3d__var_2_n00060005.plt")))
    points = ds[XYZ_VARS]
    tree = Octree(points, ds.corners, tree_coord="rpa")
    return ds, tree


def test_sample_rpa_array_queries_match_split_component_queries(sc_context) -> None:
    """One `(…, 3)` rpa query array should match the split-component rpa call on the SC sample."""
    ds, tree = sc_context
    interp = OctreeInterpolator(tree, np.asarray(ds["Rho [g/cm^3]"]))
    polar = np.linspace(0.0, math.pi, 33, dtype=float)
    azimuth = np.linspace(0.0, 2.0 * math.pi, 65, endpoint=False, dtype=float)
    aa, pp = np.meshgrid(azimuth, polar, indexing="xy")
    rr = np.full_like(aa, 20.0)
    q = np.stack((rr, pp, aa), axis=-1)

    vals_array, cells_array = interp(q, query_coord="rpa", return_cell_ids=True)
    vals_split, cells_split = interp(rr, pp, aa, query_coord="rpa", return_cell_ids=True)

    assert np.array_equal(cells_array, cells_split)
    assert np.allclose(vals_array, vals_split, atol=1e-12, rtol=0.0)


def test_sample_pole_cut_is_azimuth_invariant_for_z_field(sc_context) -> None:
    """Varying azimuth at the north pole should not change one single-valued geometric field."""
    ds, tree = sc_context
    interp = OctreeInterpolator(tree, np.asarray(ds[XYZ_VARS[2]]))
    azimuth = np.linspace(0.0, 2.0 * math.pi, 65, endpoint=False, dtype=float)
    values, cell_ids = interp(
        np.full_like(azimuth, 20.0),
        np.zeros_like(azimuth),
        azimuth,
        query_coord="rpa",
        return_cell_ids=True,
    )

    assert np.all(cell_ids >= 0)
    assert np.allclose(values, values[0], atol=1e-12, rtol=0.0)


def test_sample_off_pole_cut_is_azimuth_invariant_for_z_field(sc_context) -> None:
    """Varying azimuth away from the pole should not change the interpolated `Z` field at fixed `(r, polar)`."""
    ds, tree = sc_context
    interp = OctreeInterpolator(tree, np.asarray(ds[XYZ_VARS[2]]))
    azimuth = np.linspace(0.0, 2.0 * math.pi, 65, endpoint=False, dtype=float)
    values, cell_ids = interp(
        np.full_like(azimuth, 20.0),
        np.full_like(azimuth, 0.8),
        azimuth,
        query_coord="rpa",
        return_cell_ids=True,
    )

    assert np.all(cell_ids >= 0)
    assert np.allclose(values, values[0], atol=1e-12, rtol=0.0)


def test_sample_density_is_continuous_across_shared_azimuth_face(sc_context) -> None:
    """Density interpolation should stay continuous across one shared azimuth face in the SC sample."""
    ds, tree = sc_context
    interp = OctreeInterpolator(tree, np.asarray(ds["Rho [g/cm^3]"]))
    bounds = tree.cell_bounds
    leaf_ids = np.flatnonzero(tree.cell_levels >= 0)
    lo = bounds[leaf_ids, :, 0]
    hi = bounds[leaf_ids, :, 0] + bounds[leaf_ids, :, 1]
    valid = leaf_ids[
        (lo[:, 1] > 1.0e-6)
        & (hi[:, 1] < (math.pi - 1.0e-6))
        & (hi[:, 2] < (2.0 * math.pi - 1.0e-8))
        & np.array([np.unique(tree.corners[int(cell_id)]).size == 8 for cell_id in leaf_ids.tolist()], dtype=bool)
    ]
    left_id, right_id, lo_left, hi_left = find_shared_face_pair(tree, valid, face_axis=2)
    r = 0.5 * (float(lo_left[0]) + float(hi_left[0]))
    polar = 0.5 * (float(lo_left[1]) + float(hi_left[1]))
    face = float(hi_left[2])
    eps = 1.0e-8 * float(hi_left[2] - lo_left[2])
    q = np.array([[r, polar, face - eps], [r, polar, face + eps]], dtype=float)
    vals, cell_ids = interp(q, query_coord="rpa", return_cell_ids=True)

    assert np.array_equal(cell_ids, np.array([left_id, right_id], dtype=np.int64))
    assert np.allclose(vals[0], vals[1], atol=1e-12, rtol=1.0e-8)

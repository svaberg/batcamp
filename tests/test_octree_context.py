from __future__ import annotations

import numpy as np
import pytest

from batcamp import DEFAULT_MIN_VALID_CELL_FRACTION
from batcamp import format_histogram
from batcamp import valid_cell_fraction


@pytest.fixture(scope="module")
def octree_context(difflevels_rpa_context: dict[str, object]) -> dict[str, object]:
    """Expose shared session-scoped difflevels context in this test module."""
    return difflevels_rpa_context


def _xyz_to_rpa_numpy(q_xyz: np.ndarray) -> np.ndarray:
    """Private test helper: convert one xyz point to one rpa point."""
    q = np.asarray(q_xyz, dtype=float).reshape(3)
    r = float(np.linalg.norm(q))
    zr = float(q[2] / max(r, float(np.finfo(float).tiny)))
    polar = float(np.arccos(np.clip(zr, -1.0, 1.0)))
    azimuth = float(np.mod(np.arctan2(q[1], q[0]), 2.0 * np.pi))
    return np.array([r, polar, azimuth], dtype=float)


def test_phi_level_arrays_shapes(octree_context: dict[str, object]) -> None:
    """Per-cell level arrays have expected sizes and finite coarse spacing."""
    corners = octree_context["corners"]
    delta_phi = octree_context["delta_phi"]
    center_phi = octree_context["center_phi"]
    cell_levels = octree_context["cell_levels"]
    expected = octree_context["expected"]
    coarse = octree_context["coarse"]

    assert delta_phi.shape[0] == corners.shape[0]
    assert center_phi.shape[0] == corners.shape[0]
    assert cell_levels.shape[0] == corners.shape[0]
    assert expected.shape[0] == corners.shape[0]
    assert np.isfinite(coarse)


def test_valid_fraction_and_histograms(octree_context: dict[str, object]) -> None:
    """Valid-level fraction and histogram utilities behave on the sample file."""
    corners = octree_context["corners"]
    ds = octree_context["ds"]
    cell_levels = octree_context["cell_levels"]
    point_levels = octree_context["point_levels"]

    valid, total, frac_valid = valid_cell_fraction(cell_levels)
    assert total == corners.shape[0]
    assert valid > 0
    assert frac_valid >= DEFAULT_MIN_VALID_CELL_FRACTION

    assert point_levels.shape[0] == ds.points.shape[0]
    cell_hist = format_histogram(cell_levels)
    point_hist = format_histogram(point_levels)
    assert cell_hist
    assert point_hist


def test_tree_caches_metadata(octree_context: dict[str, object]) -> None:
    """Tree stores lookup-level metadata needed for downstream construction."""
    cell_levels = octree_context["cell_levels"]
    tree = octree_context["tree"]

    assert tree.max_level >= tree.min_level
    assert tree.max_level > tree.min_level
    assert tree.cell_levels is not None
    assert tree.cell_levels.shape == cell_levels.shape


def test_lookup_xyz_rpa_agree(octree_context: dict[str, object]) -> None:
    """Cartesian and spherical lookup coords resolve the same leaf cell."""
    tree = octree_context["tree"]

    q_xyz = np.array([1.0, 0.0, 0.0], dtype=float)
    hit_xyz = tree.lookup_point(q_xyz, coord="xyz")
    assert hit_xyz is not None

    hit_rpa = tree.lookup_point(_xyz_to_rpa_numpy(q_xyz), coord="rpa")
    assert hit_rpa is not None
    assert hit_xyz.cell_id == hit_rpa.cell_id

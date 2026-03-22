from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pooch
import pytest
from batread.dataset import Dataset

from batcamp import Octree
from batcamp import OctreeInterpolator

_G2211_URL = "https://zenodo.org/records/7110555/files/run-Sun-G2211.tar.gz"
_G2211_SHA256 = "c31a32aab08cc20d5b643bba734fd7220e6b369e691f55f88a3a08cc5b2a2136"


def _fetch_g2211_member(member: str) -> Path:
    """Fetch one named member from the G2211 archive with pooch."""
    extracted = pooch.retrieve(
        url=_G2211_URL,
        known_hash=_G2211_SHA256,
        progressbar=False,
        processor=pooch.Untar(members=[member]),
    )
    if isinstance(extracted, (list, tuple)):
        extracted = extracted[0]
    return Path(extracted)


def _rpa_to_xyz(q_rpa: np.ndarray) -> np.ndarray:
    """Convert one `(r, polar, azimuth)` point to Cartesian coordinates."""
    q = np.asarray(q_rpa, dtype=float).reshape(3)
    sin_polar = math.sin(float(q[1]))
    return np.array(
        [
            float(q[0]) * sin_polar * math.cos(float(q[2])),
            float(q[0]) * sin_polar * math.sin(float(q[2])),
            float(q[0]) * math.cos(float(q[1])),
        ],
        dtype=float,
    )


def _midpoint_rpa(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Return a midpoint inside one wrapped `(r, polar, azimuth)` box."""
    phi_width = float((float(hi[2]) - float(lo[2])) % (2.0 * math.pi))
    if phi_width == 0.0 and not np.isclose(float(lo[2]), float(hi[2]), rtol=0.0, atol=0.0):
        phi_width = 2.0 * math.pi
    return np.array(
        [
            0.5 * (float(lo[0]) + float(hi[0])),
            0.5 * (float(lo[1]) + float(hi[1])),
            float((float(lo[2]) + 0.5 * phi_width) % (2.0 * math.pi)),
        ],
        dtype=float,
    )


def _find_query_in_cell(tree: Octree, cid: int, *, coord: str) -> np.ndarray:
    """Construct one interior query point for a given cell and coord system."""
    lo, hi = tree.cell_bounds(cid, coord=coord)
    if coord == "xyz":
        return 0.5 * (np.asarray(lo, dtype=float) + np.asarray(hi, dtype=float))
    return _rpa_to_xyz(_midpoint_rpa(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)))


def _first_working_query(tree: Octree, *, coord: str) -> tuple[int, np.ndarray]:
    """Return one cell id and query point that resolve back to that cell."""
    assert tree.cell_levels is not None
    levels = np.asarray(tree.cell_levels, dtype=np.int64)
    candidate_ids = np.flatnonzero(levels == int(tree.min_level))
    if candidate_ids.size == 0:
        candidate_ids = np.arange(tree.cell_count, dtype=np.int64)

    for cid in candidate_ids.tolist():
        q_xyz = _find_query_in_cell(tree, int(cid), coord=coord)
        hit = tree.lookup_point(q_xyz, coord="xyz")
        if hit is not None and int(hit.cell_id) == int(cid):
            return int(cid), q_xyz

    raise AssertionError(f"Could not find a stable {coord} query point in fetched dataset.")


@pytest.mark.pooch
def test_remote_sc_rpa_build_and_query() -> None:
    """Remote SC pooch sample should build as spherical and support one query."""
    path = _fetch_g2211_member("run-Sun-G2211/SC/IO2/3d__var_4_n00044000.plt")
    ds = Dataset.from_file(str(path))
    tree = Octree.from_dataset(ds)
    assert tree.tree_coord == "rpa"

    cid, q_xyz = _first_working_query(tree, coord="rpa")
    hit = tree.lookup_point(q_xyz, coord="xyz")
    assert hit is not None
    assert int(hit.cell_id) == cid

    interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"], tree=tree)
    vals, cell_ids = interp(q_xyz[None, :], return_cell_ids=True)
    assert int(cell_ids[0]) == cid
    assert np.isfinite(vals).any()


@pytest.mark.pooch
def test_remote_ih_xyz_build_and_query() -> None:
    """Remote IH pooch sample should build as Cartesian and support one query."""
    path = _fetch_g2211_member("run-Sun-G2211/IH/IO2/3d__var_4_n00005000.plt")
    ds = Dataset.from_file(str(path))
    tree = Octree.from_dataset(ds)
    assert tree.tree_coord == "xyz"

    cid, q_xyz = _first_working_query(tree, coord="xyz")
    hit = tree.lookup_point(q_xyz, coord="xyz")
    assert hit is not None
    assert int(hit.cell_id) == cid

    interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"], tree=tree)
    vals, cell_ids = interp(q_xyz[None, :], return_cell_ids=True)
    assert int(cell_ids[0]) == cid
    assert np.isfinite(vals).any()

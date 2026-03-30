from __future__ import annotations

import numpy as np

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh


_PLANE_RAMP = (2, 4, 8, 16, 32, 64)


def _linear_scalar(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return one globally linear scalar field with nontrivial slopes in all axes."""
    return 0.7 * np.asarray(x, dtype=float) - 1.2 * np.asarray(y, dtype=float) + 0.5 * np.asarray(z, dtype=float) + 2.0


def _build_uniform_cartesian_linear_dataset() -> _FakeDataset:
    """Build one full Cartesian mesh carrying one exact linear field."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.linspace(-1.0, 1.0, 17, dtype=float),
        y_edges=np.linspace(-0.75, 0.75, 13, dtype=float),
        z_edges=np.linspace(-0.5, 0.5, 9, dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Scalar": _linear_scalar(x, y, z),
            "XCoord": np.asarray(x, dtype=float),
            "YCoord": np.asarray(y, dtype=float),
            "ZCoord": np.asarray(z, dtype=float),
        },
    )


def _build_adaptive_cartesian_linear_dataset() -> _FakeDataset:
    """Build one adaptive Cartesian mesh carrying the same exact linear field."""
    x_edges = np.array([0.0, 1.0, 1.5, 2.0], dtype=float)
    y_edges = np.array([0.0, 0.5, 1.0], dtype=float)
    z_edges = np.array([0.0, 0.5, 1.0], dtype=float)
    points, _unused = _build_cartesian_hex_mesh(
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
    )
    node_index = np.arange(points.shape[0], dtype=np.int64).reshape(x_edges.size, y_edges.size, z_edges.size)

    corners = [
        [
            int(node_index[0, 0, 0]),
            int(node_index[1, 0, 0]),
            int(node_index[1, 2, 0]),
            int(node_index[0, 2, 0]),
            int(node_index[0, 0, 2]),
            int(node_index[1, 0, 2]),
            int(node_index[1, 2, 2]),
            int(node_index[0, 2, 2]),
        ]
    ]
    for ix in (1, 2):
        for iy in (0, 1):
            for iz in (0, 1):
                corners.append(
                    [
                        int(node_index[ix, iy, iz]),
                        int(node_index[ix + 1, iy, iz]),
                        int(node_index[ix + 1, iy + 1, iz]),
                        int(node_index[ix, iy + 1, iz]),
                        int(node_index[ix, iy, iz + 1]),
                        int(node_index[ix + 1, iy, iz + 1]),
                        int(node_index[ix + 1, iy + 1, iz + 1]),
                        int(node_index[ix, iy + 1, iz + 1]),
                    ]
                )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return _FakeDataset(
        points=points,
        corners=np.array(corners, dtype=np.int64),
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Scalar": _linear_scalar(x, y, z),
            "XCoord": np.asarray(x, dtype=float),
            "YCoord": np.asarray(y, dtype=float),
            "ZCoord": np.asarray(z, dtype=float),
        },
    )


def _xyz_points(ds: _FakeDataset) -> np.ndarray:
    """Return dataset point coordinates as one dense `(n, 3)` array."""
    return np.column_stack(
        [
            np.asarray(ds[XYZ_VARS[0]], dtype=float),
            np.asarray(ds[XYZ_VARS[1]], dtype=float),
            np.asarray(ds[XYZ_VARS[2]], dtype=float),
        ]
    )


def _xy_plane_queries(xyz: np.ndarray, *, resolution: int) -> np.ndarray:
    """Return one interior `xy` plane of query points."""
    dmin = np.min(xyz, axis=0)
    dmax = np.max(xyz, axis=0)
    span = dmax - dmin
    eps = 1e-6 * np.maximum(span, 1.0)
    x = np.linspace(float(dmin[0] + eps[0]), float(dmax[0] - eps[0]), int(resolution), dtype=float)
    y = np.linspace(float(dmin[1] + eps[1]), float(dmax[1] - eps[1]), int(resolution), dtype=float)
    xg, yg = np.meshgrid(x, y, indexing="xy")
    zg = np.full_like(xg, 0.5 * float(dmin[2] + dmax[2]))
    return np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))


def _random_interior_queries(xyz: np.ndarray, *, n_query: int, seed: int) -> np.ndarray:
    """Return random interior Cartesian queries away from the outer box faces."""
    rng = np.random.default_rng(int(seed))
    dmin = np.min(xyz, axis=0)
    dmax = np.max(xyz, axis=0)
    span = dmax - dmin
    eps = 1e-6 * np.maximum(span, 1.0)
    lo = dmin + eps
    hi = dmax - eps
    return rng.uniform(lo, hi, size=(int(n_query), 3))


def _assert_plane_ramp_matches_exact_field(
    ds: _FakeDataset,
    *,
    field_name: str,
    expected_fn,
) -> None:
    """Assert plane-resampling stays exact over one resolution ramp."""
    xyz = _xyz_points(ds)
    interp = OctreeInterpolator(Octree.from_ds(ds, tree_coord="xyz"), np.asarray(ds[field_name]))

    for resolution in _PLANE_RAMP:
        query = _xy_plane_queries(xyz, resolution=resolution)
        values, cell_ids = interp(
            query,
            query_coord="xyz",
            log_outside_domain=False,
            return_cell_ids=True,
        )
        values = np.asarray(values, dtype=float)
        expected = np.asarray(expected_fn(query), dtype=float)

        assert np.all(cell_ids >= 0)
        assert np.isfinite(values).all()
        assert np.allclose(values, expected, atol=1e-10, rtol=0.0)


def test_uniform_cartesian_plane_ramp_matches_exact_linear_field() -> None:
    """Uniform synthetic plane ramp should reproduce the exact linear field."""
    _assert_plane_ramp_matches_exact_field(
        _build_uniform_cartesian_linear_dataset(),
        field_name="Scalar",
        expected_fn=lambda q: _linear_scalar(q[:, 0], q[:, 1], q[:, 2]),
    )


def test_adaptive_cartesian_plane_ramp_matches_exact_linear_field() -> None:
    """Adaptive synthetic plane ramp should reproduce the exact linear field."""
    _assert_plane_ramp_matches_exact_field(
        _build_adaptive_cartesian_linear_dataset(),
        field_name="Scalar",
        expected_fn=lambda q: _linear_scalar(q[:, 0], q[:, 1], q[:, 2]),
    )


def test_uniform_cartesian_plane_ramp_matches_x_coordinate_field() -> None:
    """Uniform synthetic plane ramp should reproduce the exact `x` coordinate field."""
    _assert_plane_ramp_matches_exact_field(
        _build_uniform_cartesian_linear_dataset(),
        field_name="XCoord",
        expected_fn=lambda q: q[:, 0],
    )


def test_adaptive_cartesian_plane_ramp_matches_x_coordinate_field() -> None:
    """Adaptive synthetic plane ramp should reproduce the exact `x` coordinate field."""
    _assert_plane_ramp_matches_exact_field(
        _build_adaptive_cartesian_linear_dataset(),
        field_name="XCoord",
        expected_fn=lambda q: q[:, 0],
    )


def test_uniform_cartesian_interp_matches_xyz_coordinate_fields() -> None:
    """Uniform Cartesian interpolation should preserve axis order for `(x, y, z)` fields."""
    ds = _build_uniform_cartesian_linear_dataset()
    xyz = _xyz_points(ds)
    interp = OctreeInterpolator(
        Octree.from_ds(ds, tree_coord="xyz"),
        np.column_stack((np.asarray(ds["XCoord"]), np.asarray(ds["YCoord"]), np.asarray(ds["ZCoord"]))),
    )
    query = _random_interior_queries(xyz, n_query=512, seed=123)
    values, cell_ids = interp(query, query_coord="xyz", log_outside_domain=False, return_cell_ids=True)
    assert np.all(cell_ids >= 0)
    assert np.allclose(values, query, atol=1e-10, rtol=0.0)


def test_adaptive_cartesian_interp_matches_xyz_coordinate_fields() -> None:
    """Adaptive Cartesian interpolation should preserve axis order for `(x, y, z)` fields."""
    ds = _build_adaptive_cartesian_linear_dataset()
    xyz = _xyz_points(ds)
    interp = OctreeInterpolator(
        Octree.from_ds(ds, tree_coord="xyz"),
        np.column_stack((np.asarray(ds["XCoord"]), np.asarray(ds["YCoord"]), np.asarray(ds["ZCoord"]))),
    )
    query = _random_interior_queries(xyz, n_query=512, seed=456)
    values, cell_ids = interp(query, query_coord="xyz", log_outside_domain=False, return_cell_ids=True)
    assert np.all(cell_ids >= 0)
    assert np.allclose(values, query, atol=1e-10, rtol=0.0)

from __future__ import annotations

import numpy as np

from batcamp import Octree
from batcamp import build_octree_from_ds
from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh


def _build_ring_pattern_dataset(*, nxy: int = 24) -> _FakeDataset:
    """Build one thin Cartesian slab whose midplane resamples to concentric rings."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.linspace(-1.0, 1.0, int(nxy) + 1, dtype=float),
        y_edges=np.linspace(-1.0, 1.0, int(nxy) + 1, dtype=float),
        z_edges=np.array([-0.25, 0.25], dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    radius = np.sqrt(x * x + y * y)
    ring = 0.05 + np.exp(-((radius - 0.55) ** 2) / (2.0 * 0.08 * 0.08))
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Pattern": ring,
        },
    )


def _build_checkerboard_pattern_dataset(*, nxy: int = 24) -> _FakeDataset:
    """Build one thin Cartesian slab with alternating bright/dim squares on the midplane."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.linspace(-1.0, 1.0, int(nxy) + 1, dtype=float),
        y_edges=np.linspace(-1.0, 1.0, int(nxy) + 1, dtype=float),
        z_edges=np.array([-0.25, 0.25], dtype=float),
    )
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    board = 1.0 + 0.75 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Pattern": board,
        },
    )


def _build_adaptive_slab_points_and_corners() -> tuple[np.ndarray, np.ndarray]:
    """Build one dyadic slab with six coarse octants and one refined northeast column."""
    x_edges = np.linspace(-1.0, 1.0, 9, dtype=float)
    y_edges = np.linspace(-1.0, 1.0, 9, dtype=float)
    z_edges = np.linspace(-0.25, 0.25, 9, dtype=float)
    points, _unused = _build_cartesian_hex_mesh(x_edges=x_edges, y_edges=y_edges, z_edges=z_edges)
    node_index = np.arange(points.shape[0], dtype=np.int64).reshape(x_edges.size, y_edges.size, z_edges.size)

    corners: list[list[int]] = []

    def add_cell(ix0: int, ix1: int, iy0: int, iy1: int, iz0: int, iz1: int) -> None:
        corners.append(
            [
                int(node_index[ix0, iy0, iz0]),
                int(node_index[ix1, iy0, iz0]),
                int(node_index[ix0, iy1, iz0]),
                int(node_index[ix1, iy1, iz0]),
                int(node_index[ix0, iy0, iz1]),
                int(node_index[ix1, iy0, iz1]),
                int(node_index[ix0, iy1, iz1]),
                int(node_index[ix1, iy1, iz1]),
            ]
        )

    for ix0, ix1 in ((0, 4), (4, 8)):
        for iy0, iy1 in ((0, 4), (4, 8)):
            for iz0, iz1 in ((0, 4), (4, 8)):
                if ix0 == 4 and iy0 == 4:
                    continue
                add_cell(ix0, ix1, iy0, iy1, iz0, iz1)
    for ix in range(4, 8):
        for iy in range(4, 8):
            for iz in range(0, 8):
                add_cell(ix, ix + 1, iy, iy + 1, iz, iz + 1)
    return points, np.array(corners, dtype=np.int64)


def _build_adaptive_ring_pattern_dataset() -> _FakeDataset:
    """Build one adaptive slab whose refined northeast patch contains a ring."""
    points, corners = _build_adaptive_slab_points_and_corners()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    radius = np.sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5))
    ring = 0.05 + np.exp(-((radius - 0.25) ** 2) / (2.0 * 0.07 * 0.07))
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Pattern": ring,
        },
    )


def _build_adaptive_checkerboard_pattern_dataset() -> _FakeDataset:
    """Build one adaptive slab with one 2x2 checkerboard in the refined northeast patch."""
    points, corners = _build_adaptive_slab_points_and_corners()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ix = np.floor(np.clip((x - 0.0) / 0.5, 0.0, 1.999999)).astype(np.int64)
    iy = np.floor(np.clip((y - 0.0) / 0.5, 0.0, 1.999999)).astype(np.int64)
    in_patch = (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
    parity = (ix + iy) % 2
    board = np.where(in_patch, np.where(parity == 0, 1.8, 0.2), 1.0)
    return _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: x,
            XYZ_VARS[1]: y,
            XYZ_VARS[2]: z,
            "Pattern": board,
        },
    )


def _resample_xy_plane(
    interp: OctreeInterpolator,
    *,
    nxy: int = 64,
    z_value: float = 0.03125,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample one scalar field onto one regular `xy` plane inside the Cartesian slab."""
    dmin, dmax = interp.tree.domain_bounds(coord="xyz")
    x = np.linspace(float(dmin[0]), float(dmax[0]), int(nxy), dtype=float)
    y = np.linspace(float(dmin[1]), float(dmax[1]), int(nxy), dtype=float)
    xg, yg = np.meshgrid(x, y, indexing="xy")
    zg = np.full_like(xg, float(z_value))
    query = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))
    img = np.asarray(
        interp(query, query_coord="xyz", log_outside_domain=False),
        dtype=float,
    ).reshape(int(nxy), int(nxy))
    return xg, yg, img


def _assert_ring_pattern(
    xg: np.ndarray,
    yg: np.ndarray,
    img: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    ring_radius: float,
    ring_width: float,
    inner_radius: float | None = None,
) -> None:
    """Assert one resampled scalar image preserves one annular peak."""
    radius = np.sqrt((xg - float(center_x)) ** 2 + (yg - float(center_y)) ** 2)
    finite = np.isfinite(img)
    if inner_radius is None:
        inner_radius = max(0.18, 0.5 * float(ring_radius))
    inner = img[finite & (radius <= float(inner_radius))]
    ring = img[finite & (np.abs(radius - float(ring_radius)) <= float(ring_width))]
    outer = img[finite & (radius >= 0.82)]

    assert np.any(finite)
    assert ring.size > 0
    assert inner.size > 0
    assert outer.size > 0
    assert float(np.max(img[finite])) > float(np.min(img[finite]))
    assert float(np.mean(ring)) > 2.0 * float(np.mean(inner))
    assert float(np.mean(ring)) > 2.0 * float(np.mean(outer))


def _assert_checkerboard_pattern(xg: np.ndarray, yg: np.ndarray, img: np.ndarray) -> None:
    """Assert one resampled scalar image preserves alternating checkerboard tiles."""
    finite = np.isfinite(img)
    assert np.any(finite)
    assert float(np.max(img[finite])) > float(np.min(img[finite]))

    edges = np.linspace(-1.0, 1.0, 5, dtype=float)
    tile_means = np.empty((4, 4), dtype=float)
    for iy in range(4):
        for ix in range(4):
            x_mask = (xg >= edges[ix]) & (xg <= edges[ix + 1] if ix == 3 else xg < edges[ix + 1])
            y_mask = (yg >= edges[iy]) & (yg <= edges[iy + 1] if iy == 3 else yg < edges[iy + 1])
            mask = finite & x_mask & y_mask
            assert np.any(mask)
            tile_means[iy, ix] = float(np.mean(img[mask]))

    parity = np.indices((4, 4)).sum(axis=0) % 2
    bright = tile_means[parity == 0]
    dim = tile_means[parity == 1]
    centered = tile_means - float(np.mean(tile_means))

    assert float(np.mean(bright)) > float(np.mean(dim)) + 0.15
    assert np.all(centered[:, :-1] * centered[:, 1:] < 0.0)
    assert np.all(centered[:-1, :] * centered[1:, :] < 0.0)


def _assert_checkerboard_patch(
    xg: np.ndarray,
    yg: np.ndarray,
    img: np.ndarray,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    n_tiles: int,
) -> None:
    """Assert one local checker patch alternates over its tile means."""
    finite = np.isfinite(img)
    edges_x = np.linspace(float(x0), float(x1), int(n_tiles) + 1, dtype=float)
    edges_y = np.linspace(float(y0), float(y1), int(n_tiles) + 1, dtype=float)
    tile_means = np.empty((int(n_tiles), int(n_tiles)), dtype=float)
    for iy in range(int(n_tiles)):
        for ix in range(int(n_tiles)):
            x_mask = (xg >= edges_x[ix]) & (xg <= edges_x[ix + 1] if ix == int(n_tiles) - 1 else xg < edges_x[ix + 1])
            y_mask = (yg >= edges_y[iy]) & (yg <= edges_y[iy + 1] if iy == int(n_tiles) - 1 else yg < edges_y[iy + 1])
            mask = finite & x_mask & y_mask
            assert np.any(mask)
            tile_means[iy, ix] = float(np.mean(img[mask]))

    parity = np.indices((int(n_tiles), int(n_tiles))).sum(axis=0) % 2
    bright = tile_means[parity == 0]
    dim = tile_means[parity == 1]
    centered = tile_means - float(np.mean(tile_means))

    assert float(np.mean(bright)) > float(np.mean(dim)) + 0.2
    assert np.all(centered[:, :-1] * centered[:, 1:] < 0.0)
    assert np.all(centered[:-1, :] * centered[1:, :] < 0.0)


def test_xy_plane_resample_preserves_ring_pattern() -> None:
    """Midplane resample of a synthetic ring field should peak on an annulus, not at the center."""
    ds = _build_ring_pattern_dataset()
    interp = OctreeInterpolator(build_octree_from_ds(ds, tree_coord="xyz"), np.asarray(ds["Pattern"]))
    xg, yg, img = _resample_xy_plane(interp)
    _assert_ring_pattern(xg, yg, img, center_x=0.0, center_y=0.0, ring_radius=0.55, ring_width=0.08)


def test_xy_plane_resample_preserves_adaptive_ring_pattern() -> None:
    """Adaptive builder path should still preserve a ring in the refined patch."""
    ds = _build_adaptive_ring_pattern_dataset()
    interp = OctreeInterpolator(build_octree_from_ds(ds, tree_coord="xyz"), np.asarray(ds["Pattern"]))
    xg, yg, img = _resample_xy_plane(interp)
    _assert_ring_pattern(xg, yg, img, center_x=0.5, center_y=0.5, ring_radius=0.25, ring_width=0.07, inner_radius=0.10)


def test_xy_plane_resample_preserves_checkerboard_pattern() -> None:
    """Midplane resample of a synthetic checkerboard field should alternate tile brightness."""
    ds = _build_checkerboard_pattern_dataset()
    interp = OctreeInterpolator(build_octree_from_ds(ds, tree_coord="xyz"), np.asarray(ds["Pattern"]))
    xg, yg, img = _resample_xy_plane(interp)
    _assert_checkerboard_pattern(xg, yg, img)


def test_xy_plane_resample_preserves_adaptive_checkerboard_pattern() -> None:
    """Adaptive builder path should still preserve checkerboard alternation."""
    ds = _build_adaptive_checkerboard_pattern_dataset()
    interp = OctreeInterpolator(build_octree_from_ds(ds, tree_coord="xyz"), np.asarray(ds["Pattern"]))
    xg, yg, img = _resample_xy_plane(interp)
    _assert_checkerboard_patch(xg, yg, img, x0=0.0, x1=1.0, y0=0.0, y1=1.0, n_tiles=2)

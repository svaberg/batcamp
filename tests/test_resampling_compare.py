from __future__ import annotations

import numpy as np

from batcamp import Octree
from batcamp import OctreeInterpolator
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
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
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
            Octree.X_VAR: x,
            Octree.Y_VAR: y,
            Octree.Z_VAR: z,
            "Pattern": board,
        },
    )


def _resample_xy_plane(
    interp: OctreeInterpolator,
    *,
    nxy: int = 64,
    z_value: float = 0.0,
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


def test_xy_plane_resample_preserves_ring_pattern() -> None:
    """Midplane resample of a synthetic ring field should peak on an annulus, not at the center."""
    ds = _build_ring_pattern_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Pattern"], tree=tree)
    xg, yg, img = _resample_xy_plane(interp)

    radius = np.sqrt(xg * xg + yg * yg)
    finite = np.isfinite(img)
    inner = img[finite & (radius <= 0.18)]
    ring = img[finite & (np.abs(radius - 0.55) <= 0.08)]
    outer = img[finite & (radius >= 0.82)]

    assert np.any(finite)
    assert ring.size > 0
    assert inner.size > 0
    assert outer.size > 0
    assert float(np.max(img[finite])) > float(np.min(img[finite]))
    assert float(np.mean(ring)) > 2.0 * float(np.mean(inner))
    assert float(np.mean(ring)) > 2.0 * float(np.mean(outer))


def test_xy_plane_resample_preserves_checkerboard_pattern() -> None:
    """Midplane resample of a synthetic checkerboard field should alternate tile brightness."""
    ds = _build_checkerboard_pattern_dataset()
    tree = Octree.from_dataset(ds, tree_coord="xyz")
    interp = OctreeInterpolator(ds, ["Pattern"], tree=tree)
    xg, yg, img = _resample_xy_plane(interp)

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

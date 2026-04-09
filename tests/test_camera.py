from __future__ import annotations

import numpy as np
import pytest

from batcamp import camera_rays


def test_parallel_camera_rays_shape_and_direction_contract() -> None:
    """Parallel camera rays should fill one `(ny, nx, 3)` origin/direction grid."""
    origins, directions = camera_rays(
        origin=[-5.0, 0.0, 0.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0],
        nx=4,
        ny=3,
        width=6.0,
        height=4.0,
    )

    assert origins.shape == (3, 4, 3)
    assert directions.shape == (3, 4, 3)
    np.testing.assert_allclose(origins.mean(axis=(0, 1)), np.array([-5.0, 0.0, 0.0]), atol=1e-15)
    np.testing.assert_allclose(origins[..., 0], -5.0)
    np.testing.assert_allclose(directions, np.broadcast_to(np.array([1.0, 0.0, 0.0]), directions.shape))
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1.0)


def test_pinhole_camera_rays_use_single_origin_and_center_ray() -> None:
    """Pinhole camera rays should share one origin and point through the target plane."""
    origins, directions = camera_rays(
        origin=[-5.0, 0.0, 0.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0],
        nx=5,
        ny=3,
        width=6.0,
        height=4.0,
        projection="pinhole",
    )

    assert origins.shape == (3, 5, 3)
    assert directions.shape == (3, 5, 3)
    np.testing.assert_allclose(origins, np.broadcast_to(np.array([-5.0, 0.0, 0.0]), origins.shape))
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1.0)
    np.testing.assert_allclose(directions[1, 2], np.array([1.0, 0.0, 0.0]))


def test_camera_rays_reject_invalid_projection() -> None:
    """Unknown projection names should fail loudly."""
    with pytest.raises(ValueError):
        camera_rays(
            origin=[-5.0, 0.0, 0.0],
            target=[0.0, 0.0, 0.0],
            up=[0.0, 0.0, 1.0],
            nx=4,
            ny=3,
            width=6.0,
            height=4.0,
            projection="fishbowl",
        )


def test_camera_rays_reject_parallel_up_vector() -> None:
    """The up vector must not be parallel to the viewing direction."""
    with pytest.raises(ValueError):
        camera_rays(
            origin=[-5.0, 0.0, 0.0],
            target=[0.0, 0.0, 0.0],
            up=[1.0, 0.0, 0.0],
            nx=4,
            ny=3,
            width=6.0,
            height=4.0,
        )

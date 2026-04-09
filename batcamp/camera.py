#!/usr/bin/env python3
"""Camera-style ray generation helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np

Projection = Literal["parallel", "pinhole"]

__all__ = ["camera_rays"]


def _vector3(name: str, value: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    """Return one finite 3-vector."""
    vec = np.asarray(value, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} must contain only finite values.")
    return vec


def _positive_int(name: str, value: int) -> int:
    """Return one positive integer parameter."""
    ivalue = int(value)
    if ivalue <= 0 or ivalue != value:
        raise ValueError(f"{name} must be a positive integer.")
    return ivalue


def _positive_float(name: str, value: float) -> float:
    """Return one positive finite float parameter."""
    fvalue = float(value)
    if not np.isfinite(fvalue) or fvalue <= 0.0:
        raise ValueError(f"{name} must be a positive finite float.")
    return fvalue


def _camera_basis(
    origin: np.ndarray | list[float] | tuple[float, float, float],
    target: np.ndarray | list[float] | tuple[float, float, float],
    up: np.ndarray | list[float] | tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one orthonormal camera frame."""
    origin_xyz = _vector3("origin", origin)
    target_xyz = _vector3("target", target)
    up_guess = _vector3("up", up)

    forward = target_xyz - origin_xyz
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 0.0:
        raise ValueError("origin and target must differ.")
    forward /= forward_norm

    right = np.cross(forward, up_guess)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 0.0:
        raise ValueError("up must not be parallel to target - origin.")
    right /= right_norm

    up_unit = np.cross(right, forward)
    up_unit /= float(np.linalg.norm(up_unit))
    return origin_xyz, target_xyz, forward, right, up_unit


def _pixel_center_offsets(nx: int, ny: int, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Return image-plane pixel-center offsets with shape `(ny, nx)`."""
    dx = width / float(nx)
    dy = height / float(ny)
    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx - 0.5 * width
    y = (np.arange(ny, dtype=np.float64) + 0.5) * dy - 0.5 * height
    return np.meshgrid(x, y, indexing="xy")


def camera_rays(
    *,
    origin: np.ndarray | list[float] | tuple[float, float, float],
    target: np.ndarray | list[float] | tuple[float, float, float],
    up: np.ndarray | list[float] | tuple[float, float, float],
    nx: int,
    ny: int,
    width: float,
    height: float,
    projection: Projection = "parallel",
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera ray origins and directions with shape `(ny, nx, 3)`.

    `projection="parallel"` uses one detector plane centered on `origin` with
    all directions parallel to `target - origin`.

    `projection="pinhole"` uses one point camera at `origin` and one image
    plane centered on `target`.
    """
    ncol = _positive_int("nx", nx)
    nrow = _positive_int("ny", ny)
    plane_width = _positive_float("width", width)
    plane_height = _positive_float("height", height)
    resolved_projection = str(projection)
    if resolved_projection not in {"parallel", "pinhole"}:
        raise ValueError("projection must be 'parallel' or 'pinhole'.")

    origin_xyz, target_xyz, forward, right, up_unit = _camera_basis(origin, target, up)
    grid_x, grid_y = _pixel_center_offsets(ncol, nrow, plane_width, plane_height)
    offsets = grid_x[..., None] * right + grid_y[..., None] * up_unit

    if resolved_projection == "parallel":
        origins = origin_xyz + offsets
        directions = np.broadcast_to(forward, origins.shape).copy()
        return np.array(origins, dtype=np.float64, order="C"), np.array(directions, dtype=np.float64, order="C")

    plane_points = target_xyz + offsets
    origins = np.broadcast_to(origin_xyz, plane_points.shape).copy()
    directions = plane_points - origin_xyz
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return np.array(origins, dtype=np.float64, order="C"), np.array(directions, dtype=np.float64, order="C")

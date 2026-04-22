#!/usr/bin/env python3
"""Shared trilinear interpolation helpers for coordinate-specific backends."""

from __future__ import annotations

import numpy as np
from numba import njit

from .octree import TREE_COORD_AXIS0
from .octree import TREE_COORD_AXIS1
from .octree import TREE_COORD_AXIS2

_TINY = np.finfo(np.float64).tiny

@njit(cache=True)
def _clamp_unit_interval(value: float) -> float:
    """Clamp one interpolation fraction onto the unit interval."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@njit(cache=True)
def _axis_fraction(value: float, start: float, width: float) -> float:
    """Return one clamped affine cell fraction along a non-periodic axis."""
    return _clamp_unit_interval((value - start) / max(width, _TINY))


@njit(cache=True)
def _periodic_axis_fraction(value: float, start: float, width: float, period: float) -> float:
    """Return one clamped affine cell fraction along a periodic axis."""
    if width <= _TINY:
        return 0.0
    wrapped = (value - start) % period
    if width < (period - 1.0e-10) and wrapped > width:
        wrapped = width
    return _clamp_unit_interval(wrapped / max(width, _TINY))


@njit(cache=True)
def _accumulate_trilinear(
    out_row: np.ndarray,
    cell_id: int,
    frac_axis0: float,
    frac_axis1: float,
    frac_axis2: float,
    corners: np.ndarray,
    point_values: np.ndarray,
    bits: np.ndarray,
) -> None:
    """Write one trilinear interpolation row from one 8-corner low/high bit table."""
    cell_id = int(cell_id)
    frac_axis0_lo = 1.0 - frac_axis0
    frac_axis1_lo = 1.0 - frac_axis1
    frac_axis2_lo = 1.0 - frac_axis2
    cell_corner_ids = corners[cell_id]
    out_row[:] = 0.0
    for corner_ord in range(8):
        bit0 = bits[corner_ord, TREE_COORD_AXIS0]
        bit1 = bits[corner_ord, TREE_COORD_AXIS1]
        bit2 = bits[corner_ord, TREE_COORD_AXIS2]
        weight = frac_axis0 if bit0 else frac_axis0_lo
        weight *= frac_axis1 if bit1 else frac_axis1_lo
        weight *= frac_axis2 if bit2 else frac_axis2_lo
        corner_point_id = int(cell_corner_ids[corner_ord])
        out_row[:] += weight * point_values[corner_point_id]

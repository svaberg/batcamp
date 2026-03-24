"""Octree package exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .builder import OctreeBuilder
from .builder import format_histogram
from .builder import point_refinement_levels
from .builder import valid_cell_fraction
from .octree import DEFAULT_AXIS_RHO_TOL
from .octree import DEFAULT_MIN_VALID_CELL_FRACTION
from .octree import DEFAULT_TREE_COORD
from .octree import OCTREE_FILE_VERSION
from .octree import LookupHit
from .octree import Octree

__all__ = [
    "OctreeBuilder",
    "format_histogram",
    "point_refinement_levels",
    "valid_cell_fraction",
    "DEFAULT_AXIS_RHO_TOL",
    "DEFAULT_TREE_COORD",
    "DEFAULT_MIN_VALID_CELL_FRACTION",
    "OCTREE_FILE_VERSION",
    "LookupHit",
    "Octree",
    "OctreeInterpolator",
    "OctreeRayTracer",
    "OctreeRayInterpolator",
    "FlatCamera",
    "FovCamera",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "OctreeInterpolator": (".interpolator", "OctreeInterpolator"),
    "OctreeRayTracer": (".ray", "OctreeRayTracer"),
    "OctreeRayInterpolator": (".ray", "OctreeRayInterpolator"),
    "FlatCamera": (".ray", "FlatCamera"),
    "FovCamera": (".ray", "FovCamera"),
}


def __getattr__(name: str) -> Any:
    """Lazily import heavy exports only when they are requested."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

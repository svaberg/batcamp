from __future__ import annotations

import itertools

import numpy as np
import pytest

from batcamp import Octree
from batcamp._xyz_refined_event_walk import _cell_exit_event_xyz
from batcamp._xyz_refined_event_walk import _event_subface_id
from batcamp._xyz_refined_event_walk import trace_xyz_refined_event_path
from batcamp._xyz_refined_event_walk import walk_event_faces_xyz
from batcamp.octree import _FACE_AXIS
from batcamp.octree import _FACE_SIDE
from batcamp.octree import _FACE_TANGENTIAL_AXES
from fake_dataset import build_cartesian_hex_mesh

# Distinct situations covered by this toy suite:
# - pure same-level interior face / edge / corner events
# - every first-event topology present in the coarse/fine toy tree
# - neighbor-table shape/range invariants in both toy trees
# - boundary iff `-1` neighbor invariants in both toy trees
# - same-level face-neighbor `ijk` invariants in both toy trees
# - coarse/fine face-patch compatibility invariants in both toy trees
# - every whole center-origin trace in the regular toy tree
# - every whole center-origin trace in the coarse/fine toy tree
# - clipped whole-trace suffix/interior windows in both toy trees
# - coarse-to-fine face crossings through all four face subpatches
# - fine-to-coarse face crossings through all four face subpatches
# - coarse-to-fine edge and corner events
# - fine-to-coarse edge and corner events
# - miss, clipped-away, and clipped-in trace intervals

_SIGN_CASES = tuple(
    (sx, sy, sz)
    for sx in (-1, 0, 1)
    for sy in (-1, 0, 1)
    for sz in (-1, 0, 1)
    if (sx, sy, sz) != (0, 0, 0)
)
_MULTIFACE_SIGN_CASES = tuple(sign_triplet for sign_triplet in _SIGN_CASES if sum(int(value != 0) for value in sign_triplet) > 1)
_REFINED_OCTANTS = tuple(itertools.product((0, 1), repeat=3))
_INTERIOR_OCTANT_RELATIVE_ORIGINS = tuple(itertools.product((0.125, 0.875), repeat=3))
_REGULAR_CELL_IDS = tuple(range(27))
_REFINED_CELL_IDS = tuple(range(15))
_LOOKUP_ORACLE_MAX_ULP = 16


def _build_xyz_regular_tree_from_edges(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
) -> Octree:
    """Return one regular Cartesian toy tree from explicit edge arrays."""
    points, corners = build_cartesian_hex_mesh(
        x_edges=np.asarray(x_edges, dtype=float),
        y_edges=np.asarray(y_edges, dtype=float),
        z_edges=np.asarray(z_edges, dtype=float),
    )
    return Octree(points, corners, tree_coord="xyz")


def _build_xyz_regular_tree() -> Octree:
    """Return one dyadic-friendly 3x3x3 Cartesian tree with one true interior leaf."""
    return _build_xyz_regular_tree_from_edges(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
    )


def _build_xyz_regular_tree_nondyadic() -> Octree:
    """Return one non-dyadic 3x3x3 Cartesian tree."""
    return _build_xyz_regular_tree_from_edges(
        np.array([0.0, 0.1, 0.2, 0.3], dtype=float),
        np.array([0.0, 0.15, 0.3, 0.45], dtype=float),
        np.array([0.0, 0.12, 0.24, 0.36], dtype=float),
    )


def _midpoint_edges(total_width: float) -> np.ndarray:
    """Return one non-dyadic five-edge split built by repeated midpoint arithmetic."""
    edge0 = 0.0
    edge4 = float(total_width)
    edge2 = 0.5 * (edge0 + edge4)
    edge1 = 0.5 * (edge0 + edge2)
    edge3 = 0.5 * (edge2 + edge4)
    return np.array([edge0, edge1, edge2, edge3, edge4], dtype=float)


def _half_interval(bit: int) -> tuple[int, int]:
    """Return one coarse half-interval index pair."""
    return (0, 2) if int(bit) == 0 else (2, 4)


def _quarter_intervals(bit: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the two quarter-interval index pairs inside one selected half."""
    return ((0, 1), (1, 2)) if int(bit) == 0 else ((2, 3), (3, 4))


def _build_xyz_coarse_fine_tree_from_edges(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    refined_half: tuple[int, int, int] = (1, 0, 0),
) -> Octree:
    """Return one Cartesian coarse/fine toy tree from explicit edge arrays."""
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)
    z_edges = np.asarray(z_edges, dtype=float)

    node_index = -np.ones((x_edges.size, y_edges.size, z_edges.size), dtype=np.int64)
    xyz_list: list[tuple[float, float, float]] = []
    node_id = 0
    for ix, x in enumerate(x_edges):
        for iy, y in enumerate(y_edges):
            for iz, z in enumerate(z_edges):
                xyz_list.append((float(x), float(y), float(z)))
                node_index[ix, iy, iz] = node_id
                node_id += 1

    def cell(ix0: int, ix1: int, iy0: int, iy1: int, iz0: int, iz1: int) -> list[int]:
        return [
            int(node_index[ix0, iy0, iz0]),
            int(node_index[ix1, iy0, iz0]),
            int(node_index[ix1, iy1, iz0]),
            int(node_index[ix0, iy1, iz0]),
            int(node_index[ix0, iy0, iz1]),
            int(node_index[ix1, iy0, iz1]),
            int(node_index[ix1, iy1, iz1]),
            int(node_index[ix0, iy1, iz1]),
        ]

    coarse_rows: list[list[int]] = []
    for half_x, half_y, half_z in itertools.product((0, 1), repeat=3):
        if (half_x, half_y, half_z) == tuple(int(v) for v in refined_half):
            continue
        (ix0, ix1) = _half_interval(half_x)
        (iy0, iy1) = _half_interval(half_y)
        (iz0, iz1) = _half_interval(half_z)
        coarse_rows.append(cell(ix0, ix1, iy0, iy1, iz0, iz1))

    refined_rows: list[list[int]] = []
    x_pairs = _quarter_intervals(refined_half[0])
    y_pairs = _quarter_intervals(refined_half[1])
    z_pairs = _quarter_intervals(refined_half[2])
    for (ix0, ix1), (iy0, iy1), (iz0, iz1) in itertools.product(x_pairs, y_pairs, z_pairs):
        refined_rows.append(cell(ix0, ix1, iy0, iy1, iz0, iz1))

    corners = np.array(coarse_rows + refined_rows, dtype=np.int64)
    return Octree(np.array(xyz_list, dtype=float), corners, tree_coord="xyz")


def _build_xyz_coarse_fine_tree(refined_half: tuple[int, int, int] = (1, 0, 0)) -> Octree:
    """Return one dyadic-friendly Cartesian tree with one selected half-octant refined."""
    return _build_xyz_coarse_fine_tree_from_edges(
        np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float),
        np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float),
        np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float),
        refined_half=refined_half,
    )


def _build_xyz_coarse_fine_tree_nondyadic(refined_half: tuple[int, int, int] = (1, 0, 0)) -> Octree:
    """Return one non-dyadic Cartesian tree with one selected half-octant refined."""
    return _build_xyz_coarse_fine_tree_from_edges(
        _midpoint_edges(0.3),
        _midpoint_edges(0.35),
        _midpoint_edges(0.45),
        refined_half=refined_half,
    )


_REGULAR_TREE_BUILDERS = (
    pytest.param(_build_xyz_regular_tree, id="dyadic"),
    pytest.param(_build_xyz_regular_tree_nondyadic, id="nondyadic"),
)
_REFINED_TREE_BUILDERS = (
    pytest.param(_build_xyz_coarse_fine_tree, id="dyadic"),
    pytest.param(_build_xyz_coarse_fine_tree_nondyadic, id="nondyadic"),
)
_REGULAR_WHOLE_TRACE_TREE_BUILDERS = _REGULAR_TREE_BUILDERS
_REFINED_WHOLE_TRACE_TREE_BUILDERS = _REFINED_TREE_BUILDERS
_DYADIC_REGULAR_TREE_BUILDERS = (pytest.param(_build_xyz_regular_tree, id="dyadic"),)
_DYADIC_REFINED_TREE_BUILDERS = (pytest.param(_build_xyz_coarse_fine_tree, id="dyadic"),)


def _leaf_center(tree: Octree, cell_id: int) -> np.ndarray:
    """Return one leaf center from packed Cartesian bounds."""
    bounds = tree.cell_bounds[int(cell_id)]
    return np.array(bounds[:, 0] + 0.5 * bounds[:, 1], dtype=float)


def _center_direction(tree: Octree, cell_id: int, sign_triplet: tuple[int, int, int]) -> np.ndarray:
    """Return one center-to-face/edge/corner direction with first event at t=1 in float arithmetic."""
    origin = _leaf_center(tree, cell_id)
    bounds = np.asarray(tree.cell_bounds[int(cell_id)], dtype=float)
    direction = np.zeros(3, dtype=float)
    for axis, sign in enumerate(sign_triplet):
        if sign > 0:
            face_value = float(bounds[axis, 0] + bounds[axis, 1])
        elif sign < 0:
            face_value = float(bounds[axis, 0])
        else:
            continue
        direction[axis] = face_value - float(origin[axis])
    return direction


def _active_faces_from_signs(sign_triplet: tuple[int, int, int]) -> tuple[int, ...]:
    """Return the exact Cartesian face ids implied by one sign triple."""
    faces: list[int] = []
    for axis, sign in enumerate(sign_triplet):
        if sign > 0:
            faces.append(2 * axis + 1)
        elif sign < 0:
            faces.append(2 * axis)
    return tuple(faces)


def _lookup_after_first_event(tree: Octree, origin: np.ndarray, direction: np.ndarray) -> int:
    """Return the independent post-event owner from one exact just-after-event probe."""
    probe_xyz = origin + 1.25 * direction
    return int(tree.lookup_points(probe_xyz[None, :], coord="xyz")[0])


def _center_origin_and_direction(
    tree: Octree,
    start_cell_id: int,
    sign_triplet: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return one center-origin toy ray for one leaf and one sign pattern."""
    origin = _leaf_center(tree, start_cell_id)
    direction = _center_direction(tree, start_cell_id, sign_triplet)
    return origin, direction


def _relative_origin_and_direction(
    tree: Octree,
    start_cell_id: int,
    relative_origin_xyz: tuple[float, float, float],
    sign_triplet: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return one interior-origin toy ray with the requested first-event face set at `t=1`."""
    bounds = np.asarray(tree.cell_bounds[int(start_cell_id)], dtype=float)
    origin = bounds[:, 0] + np.asarray(relative_origin_xyz, dtype=float) * bounds[:, 1]
    direction = np.zeros(3, dtype=float)
    for axis, sign in enumerate(sign_triplet):
        if sign > 0:
            face_value = float(bounds[axis, 0] + bounds[axis, 1])
        elif sign < 0:
            face_value = float(bounds[axis, 0])
        else:
            continue
        direction[axis] = face_value - float(origin[axis])
    return origin, direction


def _leaf_ids(tree: Octree) -> np.ndarray:
    """Return the valid leaf-slot ids for one toy tree."""
    return np.flatnonzero(tree.cell_levels >= 0).astype(np.int64)


def _face_patch_bounds(
    tree: Octree,
    cell_id: int,
    face_id: int,
    subface_id: int | None,
) -> np.ndarray:
    """Return one Cartesian face patch as exact `(3, 2)` start/stop bounds."""
    bounds = np.array(tree.cell_bounds[int(cell_id)], dtype=float)
    patch = np.empty((3, 2), dtype=float)
    patch[:, 0] = bounds[:, 0]
    patch[:, 1] = bounds[:, 0] + bounds[:, 1]

    axis = int(_FACE_AXIS[int(face_id)])
    side = int(_FACE_SIDE[int(face_id)])
    face_value = patch[axis, side]
    patch[axis, 0] = face_value
    patch[axis, 1] = face_value

    if subface_id is None:
        return patch

    for bit_shift, tangential_axis in zip((1, 0), _FACE_TANGENTIAL_AXES[int(face_id)]):
        tangential_axis = int(tangential_axis)
        start = patch[tangential_axis, 0]
        stop = patch[tangential_axis, 1]
        middle = 0.5 * (start + stop)
        if (int(subface_id) >> bit_shift) & 1:
            patch[tangential_axis, 0] = middle
        else:
            patch[tangential_axis, 1] = middle
    return patch


def _is_domain_boundary_face(tree: Octree, cell_id: int, face_id: int) -> bool:
    """Return whether one leaf face lies on the Cartesian domain boundary."""
    domain_lo, domain_hi = tree.domain_bounds(coord="xyz")
    bounds = np.asarray(tree.cell_bounds[int(cell_id)], dtype=float)
    axis = int(_FACE_AXIS[int(face_id)])
    side = int(_FACE_SIDE[int(face_id)])
    face_value = float(bounds[axis, 0]) if side == 0 else float(bounds[axis, 0] + bounds[axis, 1])
    boundary_value = float(domain_lo[axis]) if side == 0 else float(domain_hi[axis])
    return face_value == boundary_value


def _patch_is_contained_in(container: np.ndarray, patch: np.ndarray) -> bool:
    """Return whether one face patch lies inside another exact Cartesian face patch."""
    return bool(np.all(container[:, 0] <= patch[:, 0]) and np.all(patch[:, 1] <= container[:, 1]))


def _assert_neighbor_table_shape_and_range(tree: Octree) -> None:
    """Check the leaf neighbor table shape and leaf-id range contract."""
    leaf_ids = _leaf_ids(tree)
    leaf_id_set = set(int(cell_id) for cell_id in leaf_ids)
    assert tree.cell_neighbor.shape[1:] == (6, 4)
    for cell_id in leaf_ids:
        for face_id in range(6):
            for subface_id in range(4):
                neighbor_id = int(tree.cell_neighbor[int(cell_id), face_id, subface_id])
                assert neighbor_id == -1 or neighbor_id in leaf_id_set


def _assert_boundary_iff_missing_neighbor(tree: Octree) -> None:
    """Check that only true Cartesian boundary face patches carry `-1` neighbors."""
    for cell_id in _leaf_ids(tree):
        for face_id in range(6):
            boundary_face = _is_domain_boundary_face(tree, int(cell_id), face_id)
            for subface_id in range(4):
                neighbor_id = int(tree.cell_neighbor[int(cell_id), face_id, subface_id])
                if boundary_face:
                    assert neighbor_id == -1
                else:
                    assert neighbor_id >= 0


def _assert_same_level_face_neighbors_match_ijk(tree: Octree) -> None:
    """Check exact `ijk` offsets for same-level leaf neighbors."""
    for cell_id in _leaf_ids(tree):
        for face_id in range(6):
            axis = int(_FACE_AXIS[int(face_id)])
            side = int(_FACE_SIDE[int(face_id)])
            for subface_id in range(4):
                neighbor_id = int(tree.cell_neighbor[int(cell_id), face_id, subface_id])
                if neighbor_id < 0:
                    continue
                if int(tree.cell_levels[neighbor_id]) != int(tree.cell_levels[int(cell_id)]):
                    continue
                delta = np.array(tree.cell_ijk[neighbor_id] - tree.cell_ijk[int(cell_id)], dtype=np.int64)
                expected = np.zeros(3, dtype=np.int64)
                expected[axis] = -1 if side == 0 else 1
                np.testing.assert_array_equal(delta, expected)


def _assert_face_patch_compatibility(tree: Octree) -> None:
    """Check that every nonboundary face patch matches the neighbor geometry exactly."""
    for cell_id in _leaf_ids(tree):
        for face_id in range(6):
            opposite_face = int(face_id) ^ 1
            for subface_id in range(4):
                neighbor_id = int(tree.cell_neighbor[int(cell_id), face_id, subface_id])
                if neighbor_id < 0:
                    continue
                patch = _face_patch_bounds(tree, int(cell_id), face_id, subface_id)
                neighbor_full_face = _face_patch_bounds(tree, neighbor_id, opposite_face, None)
                neighbor_subpatches = tuple(
                    _face_patch_bounds(tree, neighbor_id, opposite_face, neighbor_subface_id)
                    for neighbor_subface_id in range(4)
                )
                assert _patch_is_contained_in(neighbor_full_face, patch)
                if int(tree.cell_levels[neighbor_id]) > int(tree.cell_levels[int(cell_id)]):
                    assert np.array_equal(patch, neighbor_full_face)
                    continue
                assert any(_patch_is_contained_in(candidate, patch) for candidate in neighbor_subpatches)


def _format_segments(cell_ids: np.ndarray, times: np.ndarray) -> str:
    """Return one readable packed trace listing."""
    if cell_ids.size == 0 and times.size == 0:
        return "  <empty>"
    if times.size != cell_ids.size + 1:
        return f"  <invalid packed trace: {cell_ids.size=} {times.size=}>"
    lines: list[str] = []
    for segment_id, (cell_id, t_start, t_stop) in enumerate(zip(cell_ids, times[:-1], times[1:])):
        lines.append(f"  {segment_id}: cell {int(cell_id)} [{float(t_start)!r}, {float(t_stop)!r}]")
    return "\n".join(lines)


def _trace_debug_report(
    tree: Octree,
    start_cell_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    actual_cell_ids: np.ndarray,
    actual_times: np.ndarray,
    expected_cell_ids: np.ndarray | None = None,
    expected_times: np.ndarray | None = None,
    positive_cell_ids: np.ndarray | None = None,
    positive_times: np.ndarray | None = None,
    oracle_cell_ids: np.ndarray | None = None,
    oracle_times: np.ndarray | None = None,
    extra_lines: tuple[str, ...] = (),
) -> str:
    """Return one full ray report for assertion failures."""
    domain_lo, domain_hi = tree.domain_bounds(coord="xyz")
    lines = [
        "ray trace failure",
        f"start_cell_id={int(start_cell_id)}",
        f"origin_xyz={tuple(float(value) for value in np.asarray(origin_xyz, dtype=float))}",
        f"direction_xyz={tuple(float(value) for value in np.asarray(direction_xyz, dtype=float))}",
        f"t_min={float(t_min)!r}",
        f"t_max={float(t_max)!r}",
        f"domain_lo={tuple(float(value) for value in np.asarray(domain_lo, dtype=float))}",
        f"domain_hi={tuple(float(value) for value in np.asarray(domain_hi, dtype=float))}",
    ]
    lines.extend(extra_lines)
    if expected_cell_ids is not None or expected_times is not None:
        lines.append("expected raw trace:")
        lines.append(
            _format_segments(
                np.asarray(expected_cell_ids if expected_cell_ids is not None else np.empty(0, dtype=np.int64), dtype=np.int64),
                np.asarray(expected_times if expected_times is not None else np.empty(0, dtype=float), dtype=float),
            )
        )
    lines.append("actual raw trace:")
    lines.append(_format_segments(np.asarray(actual_cell_ids, dtype=np.int64), np.asarray(actual_times, dtype=float)))
    if positive_cell_ids is not None or positive_times is not None:
        lines.append("actual positive trace:")
        lines.append(
            _format_segments(
                np.asarray(positive_cell_ids if positive_cell_ids is not None else np.empty(0, dtype=np.int64), dtype=np.int64),
                np.asarray(positive_times if positive_times is not None else np.empty(0, dtype=float), dtype=float),
            )
        )
    if oracle_cell_ids is not None or oracle_times is not None:
        lines.append("oracle positive trace:")
        lines.append(
            _format_segments(
                np.asarray(oracle_cell_ids if oracle_cell_ids is not None else np.empty(0, dtype=np.int64), dtype=np.int64),
                np.asarray(oracle_times if oracle_times is not None else np.empty(0, dtype=float), dtype=float),
            )
        )
    return "\n".join(lines)


def _times_match_with_ulp_slack(actual_times: np.ndarray, expected_times: np.ndarray, *, max_ulp: int) -> bool:
    """Return whether one nondecreasing time array matches within a fixed ULP count."""
    actual = np.asarray(actual_times, dtype=np.float64)
    expected = np.asarray(expected_times, dtype=np.float64)
    if actual.shape != expected.shape:
        return False
    if np.array_equal(actual, expected):
        return True
    if np.any(~np.isfinite(actual)) or np.any(~np.isfinite(expected)):
        return False
    lower = np.minimum(actual, expected)
    upper = np.maximum(actual, expected)
    stepped = np.array(lower, copy=True)
    for _ in range(int(max_ulp)):
        stepped = np.nextafter(stepped, np.inf)
    return bool(np.all(stepped >= upper))


def _normalize_positive_trace_for_lookup_oracle(
    cell_ids: np.ndarray,
    times: np.ndarray,
    *,
    max_ulp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop oracle-scale pseudo-segments and merge equal owners across ULP-sized gaps."""
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    times = np.asarray(times, dtype=np.float64)
    if cell_ids.size == 0:
        return cell_ids, times

    normalized_cell_ids: list[int] = []
    normalized_times: list[float] = [float(times[0])]
    for cell_id, t_start, t_stop in zip(cell_ids, times[:-1], times[1:]):
        t_start = float(t_start)
        t_stop = float(t_stop)
        if _times_match_with_ulp_slack(np.array([t_start]), np.array([t_stop]), max_ulp=max_ulp):
            continue
        cell_id = int(cell_id)
        if (
            normalized_cell_ids
            and normalized_cell_ids[-1] == cell_id
            and _times_match_with_ulp_slack(
                np.array([normalized_times[-1]]),
                np.array([t_start]),
                max_ulp=max_ulp,
            )
        ):
            normalized_times[-1] = t_stop
            continue
        if not normalized_cell_ids:
            normalized_times[0] = t_start
        normalized_cell_ids.append(cell_id)
        normalized_times.append(t_stop)
    return np.asarray(normalized_cell_ids, dtype=np.int64), np.asarray(normalized_times, dtype=np.float64)


def _assert_positive_trace_interval_invariants(
    tree: Octree,
    start_cell_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    actual_cell_ids: np.ndarray,
    actual_times: np.ndarray,
    positive_cell_ids: np.ndarray,
    positive_times: np.ndarray,
) -> None:
    """Check that one positive trace covers the clipped domain interval without gaps."""
    interval = _independent_domain_interval(tree, origin_xyz, direction_xyz)
    if interval is None:
        if positive_cell_ids.size == 0 and positive_times.size == 0:
            return
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin_xyz,
                direction_xyz,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=actual_cell_ids,
                actual_times=actual_times,
                positive_cell_ids=positive_cell_ids,
                positive_times=positive_times,
                extra_lines=("positive trace exists outside the analytical domain interval",),
            )
        )

    domain_enter, domain_exit = interval
    start_t = max(float(t_min), float(domain_enter))
    stop_t = min(float(t_max), float(domain_exit))
    if not start_t < stop_t:
        if positive_cell_ids.size == 0 and positive_times.size == 0:
            return
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin_xyz,
                direction_xyz,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=actual_cell_ids,
                actual_times=actual_times,
                positive_cell_ids=positive_cell_ids,
                positive_times=positive_times,
                extra_lines=("positive trace exists outside the analytical clipped interval",),
            )
        )

    if positive_times.size != positive_cell_ids.size + 1:
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin_xyz,
                direction_xyz,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=actual_cell_ids,
                actual_times=actual_times,
                positive_cell_ids=positive_cell_ids,
                positive_times=positive_times,
                extra_lines=(f"invalid positive packed trace: {positive_cell_ids.size=} {positive_times.size=}",),
            )
        )
    if positive_times.size == 0:
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin_xyz,
                direction_xyz,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=actual_cell_ids,
                actual_times=actual_times,
                positive_cell_ids=positive_cell_ids,
                positive_times=positive_times,
                extra_lines=("analytical clipped interval is nonempty but the positive trace is empty",),
            )
        )
    if not np.all(positive_times[1:] > positive_times[:-1]):
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin_xyz,
                direction_xyz,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=actual_cell_ids,
                actual_times=actual_times,
                positive_cell_ids=positive_cell_ids,
                positive_times=positive_times,
                extra_lines=("positive trace times are not strictly increasing",),
            )
        )
    if not _times_match_with_ulp_slack(
        np.array([positive_times[0]], dtype=np.float64),
        np.array([start_t], dtype=np.float64),
        max_ulp=_LOOKUP_ORACLE_MAX_ULP,
    ):
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin_xyz,
                direction_xyz,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=actual_cell_ids,
                actual_times=actual_times,
                positive_cell_ids=positive_cell_ids,
                positive_times=positive_times,
                extra_lines=(
                    f"positive trace start disagrees with analytical clipped start: {float(positive_times[0])!r} vs {float(start_t)!r}",
                ),
            )
        )
    if not _times_match_with_ulp_slack(
        np.array([positive_times[-1]], dtype=np.float64),
        np.array([stop_t], dtype=np.float64),
        max_ulp=_LOOKUP_ORACLE_MAX_ULP,
    ):
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin_xyz,
                direction_xyz,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=actual_cell_ids,
                actual_times=actual_times,
                positive_cell_ids=positive_cell_ids,
                positive_times=positive_times,
                extra_lines=(
                    f"positive trace end disagrees with analytical clipped stop: {float(positive_times[-1])!r} vs {float(stop_t)!r}",
                ),
            )
        )

    positive_length = float(np.sum(np.diff(np.asarray(positive_times, dtype=np.float64)), dtype=np.float64))
    expected_length = float(stop_t - start_t)
    if not _times_match_with_ulp_slack(
        np.array([positive_length], dtype=np.float64),
        np.array([expected_length], dtype=np.float64),
        max_ulp=_LOOKUP_ORACLE_MAX_ULP,
    ):
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin_xyz,
                direction_xyz,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=actual_cell_ids,
                actual_times=actual_times,
                positive_cell_ids=positive_cell_ids,
                positive_times=positive_times,
                extra_lines=(
                    f"positive trace length disagrees with analytical clipped length: {positive_length!r} vs {expected_length!r}",
                ),
            )
        )


def _assert_positive_trace_midpoints_match_owners(
    tree: Octree,
    start_cell_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    actual_cell_ids: np.ndarray,
    actual_times: np.ndarray,
    positive_cell_ids: np.ndarray,
    positive_times: np.ndarray,
) -> None:
    """Check that every positive segment owns its interior midpoint on the ray."""
    for cell_id, t_start, t_stop in zip(positive_cell_ids, positive_times[:-1], positive_times[1:]):
        midpoint_xyz = np.asarray(origin_xyz, dtype=float) + (0.5 * (float(t_start) + float(t_stop))) * np.asarray(direction_xyz, dtype=float)
        owner = int(tree.lookup_points(midpoint_xyz[None, :], coord="xyz")[0])
        if owner != int(cell_id):
            raise AssertionError(
                _trace_debug_report(
                    tree,
                    start_cell_id,
                    origin_xyz,
                    direction_xyz,
                    t_min=float(t_min),
                    t_max=float(t_max),
                    actual_cell_ids=actual_cell_ids,
                    actual_times=actual_times,
                    positive_cell_ids=positive_cell_ids,
                    positive_times=positive_times,
                    extra_lines=(
                        f"segment midpoint owner mismatch: cell {int(cell_id)} midpoint_owner={owner}",
                        f"midpoint_xyz={tuple(float(value) for value in midpoint_xyz)}",
                    ),
                )
            )


def _assert_trace(
    tree: Octree,
    start_cell_id: int,
    origin_xyz: tuple[float, float, float],
    direction_xyz: tuple[float, float, float],
    expected_cell_ids: tuple[int, ...],
    expected_times: tuple[float, ...],
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> None:
    """Trace one toy ray and match exact packed cells and times."""
    origin = np.array(origin_xyz, dtype=float)
    direction = np.array(direction_xyz, dtype=float)
    cell_ids, times = trace_xyz_refined_event_path(
        tree,
        start_cell_id,
        origin,
        direction,
        t_min=float(t_min),
        t_max=float(t_max),
    )
    expected_cell_ids_array = np.array(expected_cell_ids, dtype=np.int64)
    expected_times_array = np.array(expected_times, dtype=float)
    if not np.array_equal(cell_ids, expected_cell_ids_array) or not np.array_equal(times, expected_times_array):
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin,
                direction,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=cell_ids,
                actual_times=times,
                expected_cell_ids=expected_cell_ids_array,
                expected_times=expected_times_array,
            )
        )


def _positive_trace(cell_ids: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop zero-length event hops and merge contiguous equal owners."""
    positive_cell_ids: list[int] = []
    positive_times: list[float] = []
    for cell_id, t_start, t_stop in zip(cell_ids, times[:-1], times[1:]):
        t_start = float(t_start)
        t_stop = float(t_stop)
        if t_stop == t_start:
            continue
        cell_id = int(cell_id)
        if positive_cell_ids and positive_cell_ids[-1] == cell_id and positive_times[-1] == t_start:
            positive_times[-1] = t_stop
            continue
        if not positive_times:
            positive_times.append(t_start)
        positive_cell_ids.append(cell_id)
        positive_times.append(t_stop)
    return np.array(positive_cell_ids, dtype=np.int64), np.array(positive_times, dtype=float)


def _independent_domain_interval(
    tree: Octree,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
) -> tuple[float, float] | None:
    """Return the clipped Cartesian domain interval without using the toy walker."""
    domain_lo, domain_hi = tree.domain_bounds(coord="xyz")
    t_enter = -np.inf
    t_exit = np.inf
    for axis in range(3):
        direction_value = float(direction_xyz[axis])
        origin_value = float(origin_xyz[axis])
        lo_value = float(domain_lo[axis])
        hi_value = float(domain_hi[axis])
        if direction_value == 0.0:
            if origin_value < lo_value or origin_value > hi_value:
                return None
            continue
        t0 = (lo_value - origin_value) / direction_value
        t1 = (hi_value - origin_value) / direction_value
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 > t_enter:
            t_enter = t0
        if t1 < t_exit:
            t_exit = t1
        if t_enter > t_exit:
            return None
    return float(t_enter), float(t_exit)


def _independent_positive_trace(
    tree: Octree,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the positive-length ownership trace from independent lookup probes."""
    interval = _independent_domain_interval(tree, origin_xyz, direction_xyz)
    if interval is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    domain_enter, domain_exit = interval
    start_t = max(float(t_min), float(domain_enter))
    stop_t = min(float(t_max), float(domain_exit))
    if not start_t < stop_t:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    break_times = {float(start_t), float(stop_t)}
    for axis in range(3):
        direction_value = float(direction_xyz[axis])
        if direction_value == 0.0:
            continue
        starts = np.array(tree.cell_bounds[:, axis, 0], dtype=float)
        stops = starts + np.array(tree.cell_bounds[:, axis, 1], dtype=float)
        for plane_value in np.unique(np.concatenate((starts, stops))):
            t_plane = (float(plane_value) - float(origin_xyz[axis])) / direction_value
            if start_t < t_plane < stop_t:
                break_times.add(float(t_plane))

    sorted_times = np.array(sorted(break_times), dtype=float)
    positive_cell_ids: list[int] = []
    positive_times: list[float] = [float(sorted_times[0])]
    for t_start, t_stop in zip(sorted_times[:-1], sorted_times[1:]):
        midpoint_xyz = np.asarray(origin_xyz, dtype=float) + (0.5 * (float(t_start) + float(t_stop))) * np.asarray(direction_xyz, dtype=float)
        cell_id = int(tree.lookup_points(midpoint_xyz[None, :], coord="xyz")[0])
        if cell_id < 0:
            raise AssertionError("Independent midpoint probe fell outside the Cartesian toy tree.")
        if positive_cell_ids and positive_cell_ids[-1] == cell_id:
            positive_times[-1] = float(t_stop)
            continue
        positive_cell_ids.append(cell_id)
        positive_times.append(float(t_stop))
    return np.array(positive_cell_ids, dtype=np.int64), np.array(positive_times, dtype=float)


def _assert_full_trace_matches_lookup_oracle(
    tree: Octree,
    start_cell_id: int,
    origin_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> None:
    """Check one whole toy trace against an independent positive-length lookup oracle."""
    origin = np.asarray(origin_xyz, dtype=float)
    direction = np.asarray(direction_xyz, dtype=float)
    cell_ids, times = trace_xyz_refined_event_path(
        tree,
        int(start_cell_id),
        origin,
        direction,
        t_min=float(t_min),
        t_max=float(t_max),
    )
    if not (times.size == cell_ids.size + 1 or (cell_ids.size == 0 and times.size == 0)):
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin,
                direction,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=cell_ids,
                actual_times=times,
                extra_lines=(f"invalid packed trace shape: {cell_ids.size=} {times.size=}",),
            )
        )
    if times.size and not np.all(times[1:] >= times[:-1]):
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin,
                direction,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=cell_ids,
                actual_times=times,
                extra_lines=("times are not monotone nondecreasing",),
            )
        )

    positive_cell_ids, positive_times = _positive_trace(cell_ids, times)
    oracle_cell_ids, oracle_times = _independent_positive_trace(
        tree,
        origin,
        direction,
        t_min=float(t_min),
        t_max=float(t_max),
    )
    positive_cell_ids, positive_times = _normalize_positive_trace_for_lookup_oracle(
        positive_cell_ids,
        positive_times,
        max_ulp=_LOOKUP_ORACLE_MAX_ULP,
    )
    oracle_cell_ids, oracle_times = _normalize_positive_trace_for_lookup_oracle(
        oracle_cell_ids,
        oracle_times,
        max_ulp=_LOOKUP_ORACLE_MAX_ULP,
    )
    _assert_positive_trace_interval_invariants(
        tree,
        start_cell_id,
        origin,
        direction,
        t_min=float(t_min),
        t_max=float(t_max),
        actual_cell_ids=cell_ids,
        actual_times=times,
        positive_cell_ids=positive_cell_ids,
        positive_times=positive_times,
    )
    _assert_positive_trace_midpoints_match_owners(
        tree,
        start_cell_id,
        origin,
        direction,
        t_min=float(t_min),
        t_max=float(t_max),
        actual_cell_ids=cell_ids,
        actual_times=times,
        positive_cell_ids=positive_cell_ids,
        positive_times=positive_times,
    )
    if not (
        np.array_equal(positive_cell_ids, oracle_cell_ids)
        and _times_match_with_ulp_slack(positive_times, oracle_times, max_ulp=_LOOKUP_ORACLE_MAX_ULP)
    ):
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin,
                direction,
                t_min=float(t_min),
                t_max=float(t_max),
                actual_cell_ids=cell_ids,
                actual_times=times,
                positive_cell_ids=positive_cell_ids,
                positive_times=positive_times,
                oracle_cell_ids=oracle_cell_ids,
                oracle_times=oracle_times,
                extra_lines=(
                    f"whole-trace comparison uses oracle-side positive-trace normalization and {_LOOKUP_ORACLE_MAX_ULP}-ULP time slack",
                ),
            )
        )


def _assert_first_event_matches_lookup(
    tree: Octree,
    start_cell_id: int,
    sign_triplet: tuple[int, int, int],
    *,
    min_path_length: int,
    max_path_length: int | None = None,
) -> None:
    """Check one first face/edge/corner event against independent post-event ownership."""
    origin, direction = _center_origin_and_direction(tree, start_cell_id, sign_triplet)
    t_exit, active_faces = _cell_exit_event_xyz(tree.cell_bounds, start_cell_id, origin, direction, 0.0)
    event_xyz = origin + t_exit * direction
    path = walk_event_faces_xyz(tree, start_cell_id, active_faces, event_xyz, direction)
    full_cell_ids, full_times = trace_xyz_refined_event_path(tree, start_cell_id, origin, direction)
    expected_faces = _active_faces_from_signs(sign_triplet)
    expected_owner = _lookup_after_first_event(tree, origin, direction)

    if max_path_length is None:
        max_path_length = len(active_faces)
    if not (
        t_exit == 1.0
        and active_faces == expected_faces
        and min_path_length <= len(path) <= max_path_length
        and path[-1] == expected_owner
    ):
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin,
                direction,
                t_min=0.0,
                t_max=np.inf,
                actual_cell_ids=full_cell_ids,
                actual_times=full_times,
                extra_lines=(
                    f"sign_triplet={sign_triplet}",
                    f"t_exit={float(t_exit)!r}",
                    f"active_faces={active_faces}",
                    f"expected_active_faces={expected_faces}",
                    f"path={path}",
                    f"expected_post_event_owner={expected_owner}",
                    f"min_path_length={int(min_path_length)}",
                    f"max_path_length={int(max_path_length)}",
                ),
            )
        )


def _assert_multiface_event_is_order_invariant(
    tree: Octree,
    start_cell_id: int,
    sign_triplet: tuple[int, int, int],
) -> None:
    """Check that one edge/corner event reaches the same final owner for every face order."""
    origin, direction = _center_origin_and_direction(tree, start_cell_id, sign_triplet)
    t_exit, active_faces = _cell_exit_event_xyz(tree.cell_bounds, start_cell_id, origin, direction, 0.0)
    event_xyz = origin + t_exit * direction
    full_cell_ids, full_times = trace_xyz_refined_event_path(tree, start_cell_id, origin, direction)
    finals = {
        int(walk_event_faces_xyz(tree, start_cell_id, permutation, event_xyz, direction)[-1])
        for permutation in itertools.permutations(active_faces)
    }
    expected_owner = _lookup_after_first_event(tree, origin, direction)
    if finals != {expected_owner}:
        raise AssertionError(
            _trace_debug_report(
                tree,
                start_cell_id,
                origin,
                direction,
                t_min=0.0,
                t_max=np.inf,
                actual_cell_ids=full_cell_ids,
                actual_times=full_times,
                extra_lines=(
                    f"sign_triplet={sign_triplet}",
                    f"t_exit={float(t_exit)!r}",
                    f"active_faces={active_faces}",
                    f"permutation_final_owners={tuple(sorted(finals))}",
                    f"expected_post_event_owner={expected_owner}",
                ),
            )
        )


def _assert_center_trace_matches_lookup_oracle(
    tree: Octree,
    start_cell_id: int,
    sign_triplet: tuple[int, int, int],
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> None:
    """Check one center-origin whole trace against the independent interval-ownership oracle."""
    origin, direction = _center_origin_and_direction(tree, start_cell_id, sign_triplet)
    _assert_full_trace_matches_lookup_oracle(tree, start_cell_id, origin, direction, t_min=t_min, t_max=t_max)


def _assert_relative_origin_trace_matches_lookup_oracle(
    tree: Octree,
    start_cell_id: int,
    relative_origin_xyz: tuple[float, float, float],
    sign_triplet: tuple[int, int, int],
    *,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> None:
    """Check one off-center whole trace against the normalized independent oracle."""
    origin, direction = _relative_origin_and_direction(tree, start_cell_id, relative_origin_xyz, sign_triplet)
    _assert_full_trace_matches_lookup_oracle(tree, start_cell_id, origin, direction, t_min=t_min, t_max=t_max)


def _regular_cell_id(ix: int, iy: int, iz: int) -> int:
    """Return one known flattened cell id for the 3x3x3 regular tree."""
    return ix * 9 + iy * 3 + iz


@pytest.mark.parametrize("tree_builder", _REGULAR_TREE_BUILDERS)
def test_regular_neighbor_table_invariants_hold(tree_builder) -> None:
    """The regular toy neighbor graph should satisfy the structural Cartesian invariants."""
    tree = tree_builder()
    _assert_neighbor_table_shape_and_range(tree)
    _assert_boundary_iff_missing_neighbor(tree)
    _assert_same_level_face_neighbors_match_ijk(tree)
    _assert_face_patch_compatibility(tree)


@pytest.mark.parametrize("tree_builder", _REGULAR_TREE_BUILDERS)
@pytest.mark.parametrize("sign_triplet", _SIGN_CASES)
def test_regular_interior_first_event_matches_lookup(tree_builder, sign_triplet: tuple[int, int, int]) -> None:
    """Every same-level interior first event should match independent point lookup after the event."""
    tree = tree_builder()
    start_cell_id = _regular_cell_id(1, 1, 1)
    _assert_first_event_matches_lookup(
        tree,
        start_cell_id,
        sign_triplet,
        min_path_length=len(_active_faces_from_signs(sign_triplet)),
        max_path_length=len(_active_faces_from_signs(sign_triplet)),
    )


@pytest.mark.parametrize("tree_builder", _REGULAR_TREE_BUILDERS)
@pytest.mark.parametrize("sign_triplet", _MULTIFACE_SIGN_CASES)
def test_regular_interior_multiface_events_are_order_invariant(tree_builder, sign_triplet: tuple[int, int, int]) -> None:
    """Same-level interior edge and corner events should not depend on face order."""
    tree = tree_builder()
    _assert_multiface_event_is_order_invariant(tree, _regular_cell_id(1, 1, 1), sign_triplet)


@pytest.mark.parametrize("tree_builder", _REGULAR_WHOLE_TRACE_TREE_BUILDERS)
@pytest.mark.parametrize("start_cell_id", _REGULAR_CELL_IDS)
@pytest.mark.parametrize("sign_triplet", _SIGN_CASES)
def test_regular_whole_center_traces_match_lookup_oracle(
    tree_builder,
    start_cell_id: int,
    sign_triplet: tuple[int, int, int],
) -> None:
    """Every regular center-origin whole trace should match independent interval ownership."""
    tree = tree_builder()
    _assert_center_trace_matches_lookup_oracle(tree, start_cell_id, sign_triplet)


@pytest.mark.parametrize("tree_builder", _REGULAR_WHOLE_TRACE_TREE_BUILDERS)
@pytest.mark.parametrize("start_cell_id", _REGULAR_CELL_IDS)
@pytest.mark.parametrize("sign_triplet", _SIGN_CASES)
def test_regular_clipped_center_traces_match_lookup_oracle(
    tree_builder,
    start_cell_id: int,
    sign_triplet: tuple[int, int, int],
) -> None:
    """Regular center-origin traces should still match the oracle after clipping."""
    tree = tree_builder()
    _assert_center_trace_matches_lookup_oracle(tree, start_cell_id, sign_triplet, t_min=0.5, t_max=2.5)


@pytest.mark.parametrize("tree_builder", _REGULAR_TREE_BUILDERS)
def test_regular_off_center_whole_traces_match_lookup_oracle(tree_builder) -> None:
    """Regular off-center rays from all eight interior octants should match the oracle."""
    tree = tree_builder()
    for start_cell_id in _REGULAR_CELL_IDS:
        for relative_origin_xyz in _INTERIOR_OCTANT_RELATIVE_ORIGINS:
            for sign_triplet in _SIGN_CASES:
                _assert_relative_origin_trace_matches_lookup_oracle(
                    tree,
                    start_cell_id,
                    relative_origin_xyz,
                    sign_triplet,
                )


@pytest.mark.parametrize("tree_builder", _REFINED_TREE_BUILDERS)
@pytest.mark.parametrize("refined_half", _REFINED_OCTANTS)
def test_refined_neighbor_table_invariants_hold(tree_builder, refined_half: tuple[int, int, int]) -> None:
    """Every refined-octant toy neighbor graph should satisfy the structural Cartesian invariants."""
    tree = tree_builder(refined_half)
    _assert_neighbor_table_shape_and_range(tree)
    _assert_boundary_iff_missing_neighbor(tree)
    _assert_same_level_face_neighbors_match_ijk(tree)
    _assert_face_patch_compatibility(tree)


@pytest.mark.parametrize("tree_builder", _REFINED_TREE_BUILDERS)
@pytest.mark.parametrize("refined_half", _REFINED_OCTANTS)
@pytest.mark.parametrize("start_cell_id", _REFINED_CELL_IDS)
@pytest.mark.parametrize("sign_triplet", _SIGN_CASES)
def test_refined_leaf_first_event_matches_lookup(
    tree_builder,
    refined_half: tuple[int, int, int],
    start_cell_id: int,
    sign_triplet: tuple[int, int, int],
) -> None:
    """Every coarse/fine leaf-center first event should match independent point lookup after the event."""
    tree = tree_builder(refined_half)
    _assert_first_event_matches_lookup(tree, start_cell_id, sign_triplet, min_path_length=1)


@pytest.mark.parametrize("tree_builder", _REFINED_TREE_BUILDERS)
@pytest.mark.parametrize("refined_half", _REFINED_OCTANTS)
@pytest.mark.parametrize("start_cell_id", _REFINED_CELL_IDS)
@pytest.mark.parametrize("sign_triplet", _MULTIFACE_SIGN_CASES)
def test_refined_multiface_events_are_order_invariant(
    tree_builder,
    refined_half: tuple[int, int, int],
    start_cell_id: int,
    sign_triplet: tuple[int, int, int],
) -> None:
    """Every refined multi-face first event should end in the same cell for any face order."""
    tree = tree_builder(refined_half)
    _assert_multiface_event_is_order_invariant(tree, start_cell_id, sign_triplet)


@pytest.mark.parametrize("tree_builder", _REFINED_WHOLE_TRACE_TREE_BUILDERS)
@pytest.mark.parametrize("refined_half", _REFINED_OCTANTS)
@pytest.mark.parametrize("start_cell_id", _REFINED_CELL_IDS)
@pytest.mark.parametrize("sign_triplet", _SIGN_CASES)
def test_refined_whole_center_traces_match_lookup_oracle(
    tree_builder,
    refined_half: tuple[int, int, int],
    start_cell_id: int,
    sign_triplet: tuple[int, int, int],
) -> None:
    """Refined whole traces should match normalized independent interval ownership."""
    tree = tree_builder(refined_half)
    _assert_center_trace_matches_lookup_oracle(tree, start_cell_id, sign_triplet)


@pytest.mark.parametrize("tree_builder", _REFINED_WHOLE_TRACE_TREE_BUILDERS)
@pytest.mark.parametrize("refined_half", _REFINED_OCTANTS)
@pytest.mark.parametrize("start_cell_id", _REFINED_CELL_IDS)
@pytest.mark.parametrize("sign_triplet", _SIGN_CASES)
def test_refined_clipped_center_traces_match_lookup_oracle(
    tree_builder,
    refined_half: tuple[int, int, int],
    start_cell_id: int,
    sign_triplet: tuple[int, int, int],
) -> None:
    """Refined traces should still match normalized midpoint-probe ownership after clipping."""
    tree = tree_builder(refined_half)
    _assert_center_trace_matches_lookup_oracle(tree, start_cell_id, sign_triplet, t_min=0.5, t_max=2.5)


@pytest.mark.parametrize("tree_builder", _REFINED_TREE_BUILDERS)
@pytest.mark.parametrize("refined_half", _REFINED_OCTANTS)
def test_refined_off_center_whole_traces_match_lookup_oracle(
    tree_builder,
    refined_half: tuple[int, int, int],
) -> None:
    """Refined off-center rays from all eight interior octants should match the oracle."""
    tree = tree_builder(refined_half)
    for start_cell_id in _REFINED_CELL_IDS:
        for relative_origin_xyz in _INTERIOR_OCTANT_RELATIVE_ORIGINS:
            for sign_triplet in _SIGN_CASES:
                _assert_relative_origin_trace_matches_lookup_oracle(
                    tree,
                    start_cell_id,
                    relative_origin_xyz,
                    sign_triplet,
                )


@pytest.mark.parametrize(
    ("direction_xyz", "expected_cell_ids", "expected_times"),
    (
        ((0.5, 0.0, 0.0), (_regular_cell_id(1, 1, 1), _regular_cell_id(2, 1, 1)), (0.0, 1.0, 3.0)),
        ((0.0, 0.5, 0.0), (_regular_cell_id(1, 1, 1), _regular_cell_id(1, 2, 1)), (0.0, 1.0, 3.0)),
        ((0.0, 0.0, 0.5), (_regular_cell_id(1, 1, 1), _regular_cell_id(1, 1, 2)), (0.0, 1.0, 3.0)),
        ((0.5, 0.5, 0.0), (_regular_cell_id(1, 1, 1), _regular_cell_id(2, 1, 1), _regular_cell_id(2, 2, 1)), (0.0, 1.0, 1.0, 3.0)),
        ((0.5, 0.0, 0.5), (_regular_cell_id(1, 1, 1), _regular_cell_id(2, 1, 1), _regular_cell_id(2, 1, 2)), (0.0, 1.0, 1.0, 3.0)),
        ((0.0, 0.5, 0.5), (_regular_cell_id(1, 1, 1), _regular_cell_id(1, 2, 1), _regular_cell_id(1, 2, 2)), (0.0, 1.0, 1.0, 3.0)),
        ((0.5, 0.5, 0.5), (_regular_cell_id(1, 1, 1), _regular_cell_id(2, 1, 1), _regular_cell_id(2, 2, 1), _regular_cell_id(2, 2, 2)), (0.0, 1.0, 1.0, 1.0, 3.0)),
    ),
)
def test_regular_representative_paths_are_exact(
    direction_xyz: tuple[float, float, float],
    expected_cell_ids: tuple[int, ...],
    expected_times: tuple[float, ...],
) -> None:
    """Representative same-level face, edge, and corner traces should be exact."""
    tree = _build_xyz_regular_tree()
    start_cell_id = _regular_cell_id(1, 1, 1)
    _assert_trace(tree, start_cell_id, (1.5, 1.5, 1.5), direction_xyz, expected_cell_ids, expected_times)


@pytest.mark.parametrize(
    ("origin_xyz", "start_cell_id", "expected_cell_ids"),
    (
        ((0.25, 0.125, 0.125), 0, (0, 7, 11)),
        ((0.25, 0.125, 0.375), 0, (0, 8, 12)),
        ((0.25, 0.375, 0.125), 0, (0, 9, 13)),
        ((0.25, 0.375, 0.375), 0, (0, 10, 14)),
    ),
)
def test_refined_coarse_to_fine_face_subpatch_paths_are_exact(
    origin_xyz: tuple[float, float, float],
    start_cell_id: int,
    expected_cell_ids: tuple[int, ...],
) -> None:
    """Each coarse face subpatch should choose the correct refined x-neighbor chain."""
    tree = _build_xyz_coarse_fine_tree()
    _assert_trace(tree, start_cell_id, origin_xyz, (0.25, 0.0, 0.0), expected_cell_ids, (0.0, 1.0, 2.0, 3.0))


@pytest.mark.parametrize(
    ("origin_xyz", "start_cell_id", "expected_cell_ids"),
    (
        ((0.875, 0.125, 0.125), 11, (11, 7, 0)),
        ((0.875, 0.125, 0.375), 12, (12, 8, 0)),
        ((0.875, 0.375, 0.125), 13, (13, 9, 0)),
        ((0.875, 0.375, 0.375), 14, (14, 10, 0)),
    ),
)
def test_refined_fine_to_coarse_face_subpatch_paths_are_exact(
    origin_xyz: tuple[float, float, float],
    start_cell_id: int,
    expected_cell_ids: tuple[int, ...],
) -> None:
    """Each refined face subpatch should choose the correct fine-to-coarse x-chain."""
    tree = _build_xyz_coarse_fine_tree()
    _assert_trace(tree, start_cell_id, origin_xyz, (-0.125, 0.0, 0.0), expected_cell_ids, (0.0, 1.0, 3.0, 7.0))


def test_event_subface_id_uses_destination_slabs_not_source_midpoints() -> None:
    """A crossed refined face must be classified by destination child slabs, not source-face arithmetic midpoints."""
    domain_bounds = np.array(
        (
            (0.0, 1.0),
            (0.0, 2.0),
            (0.0, 1.0),
        ),
        dtype=float,
    )
    cell_bounds = np.array(
        (
            (
                (0.0, 0.5),
                (0.0, 1.0),
                (0.0, 0.5),
            ),
            (
                (0.5, 0.25),
                (0.0, 0.49),
                (0.0, 0.25),
            ),
            (
                (0.5, 0.25),
                (0.0, 0.49),
                (0.25, 0.25),
            ),
            (
                (0.5, 0.25),
                (0.49, 0.51),
                (0.0, 0.25),
            ),
            (
                (0.5, 0.25),
                (0.49, 0.51),
                (0.25, 0.25),
            ),
        ),
        dtype=float,
    )
    cell_neighbor = -np.ones((5, 6, 4), dtype=np.int64)
    cell_neighbor[0, 1] = np.array((1, 2, 3, 4), dtype=np.int64)
    event_xyz = np.array((0.5, 0.5, 0.25), dtype=float)
    subface_id = _event_subface_id(
        cell_neighbor,
        domain_bounds,
        cell_bounds,
        0,
        1,
        (1,),
        event_xyz,
        np.array((1.0, 0.0, 0.0), dtype=float),
    )
    assert subface_id == 2


@pytest.mark.parametrize(
    ("origin_xyz", "direction_xyz", "start_cell_id", "expected_cell_ids", "expected_times"),
    (
        ((0.25, 0.25, 0.25), (0.25, 0.25, 0.0), 0, (0, 9, 5), (0.0, 1.0, 1.0, 3.0)),
        ((0.25, 0.25, 0.25), (0.25, 0.0, 0.25), 0, (0, 8, 4), (0.0, 1.0, 1.0, 3.0)),
        ((0.25, 0.25, 0.25), (0.25, 0.25, 0.25), 0, (0, 10, 5, 6), (0.0, 1.0, 1.0, 1.0, 3.0)),
        ((0.625, 0.125, 0.125), (-0.125, 0.125, 0.0), 7, (7, 0, 2), (0.0, 1.0, 3.0, 5.0)),
        ((0.625, 0.125, 0.125), (-0.125, 0.0, 0.125), 7, (7, 0, 1), (0.0, 1.0, 3.0, 5.0)),
        ((0.625, 0.125, 0.125), (-0.125, 0.125, 0.125), 7, (7, 0, 2, 3), (0.0, 1.0, 3.0, 3.0, 5.0)),
    ),
)
def test_refined_multiface_representative_paths_are_exact(
    origin_xyz: tuple[float, float, float],
    direction_xyz: tuple[float, float, float],
    start_cell_id: int,
    expected_cell_ids: tuple[int, ...],
    expected_times: tuple[float, ...],
) -> None:
    """Representative coarse/fine edge and corner traces should be exact."""
    tree = _build_xyz_coarse_fine_tree()
    _assert_trace(tree, start_cell_id, origin_xyz, direction_xyz, expected_cell_ids, expected_times)


@pytest.mark.parametrize("tree_builder", _REFINED_TREE_BUILDERS)
def test_refined_toy_returns_empty_on_cartesian_miss(tree_builder) -> None:
    """A Cartesian miss should return empty packed arrays."""
    tree = tree_builder()
    cell_ids, times = trace_xyz_refined_event_path(
        tree,
        0,
        np.array((-0.25, 1.25, 0.125), dtype=float),
        np.array((0.25, 0.0, 0.0), dtype=float),
    )
    assert cell_ids.size == 0
    assert times.size == 0


@pytest.mark.parametrize("tree_builder", _REFINED_TREE_BUILDERS)
def test_refined_toy_returns_empty_when_clipping_removes_the_interval(tree_builder) -> None:
    """A fully clipped interval should return no traced cells."""
    tree = tree_builder()
    cell_ids, times = trace_xyz_refined_event_path(
        tree,
        0,
        np.array((0.25, 0.125, 0.125), dtype=float),
        np.array((0.25, 0.0, 0.0), dtype=float),
        t_min=3.0,
        t_max=4.0,
    )
    assert cell_ids.size == 0
    assert times.size == 0


@pytest.mark.parametrize("tree_builder", _REFINED_TREE_BUILDERS)
def test_refined_toy_clips_inside_one_valid_trace_window(tree_builder) -> None:
    """A clipped interval should keep the exact interior suffix of one valid trace."""
    tree = tree_builder()
    _assert_full_trace_matches_lookup_oracle(
        tree,
        0,
        np.array((0.25, 0.125, 0.125), dtype=float),
        np.array((0.25, 0.0, 0.0), dtype=float),
        t_min=1.5,
        t_max=2.5,
    )

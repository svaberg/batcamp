from __future__ import annotations

import time

import numpy as np
import pytest
from batread import Dataset

from batcamp import Octree
from batcamp import OctreeBuilder
from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS
from batcamp.ray import _candidate_face_neighbor_nodes_after_exit
from batcamp.face_neighbors import build_face_neighbors
from batcamp.face_neighbors import OctreeFaceNeighbors
from fake_dataset import FakeDataset as _FakeDataset
from fake_dataset import build_cartesian_hex_mesh as _build_cartesian_hex_mesh
from fake_dataset import build_spherical_hex_mesh as _build_spherical_hex_mesh
from sample_data_helper import data_file


def _build_cartesian_uniform_tree() -> Octree:
    """Private test helper: small uniform Cartesian tree with 8 cells."""
    points, corners = _build_cartesian_hex_mesh(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        z_edges=np.array([0.0, 1.0, 2.0], dtype=float),
    )
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: points[:, 0],
            XYZ_VARS[1]: points[:, 1],
            XYZ_VARS[2]: points[:, 2],
        },
    )
    return OctreeBuilder().build(ds, tree_coord="xyz")


def _build_spherical_uniform_tree() -> Octree:
    """Private test helper: small uniform spherical tree with periodic azimuth."""
    points, corners = _build_spherical_hex_mesh(
        nr=1,
        ntheta=2,
        nphi=4,
        r_min=1.0,
        r_max=2.0,
    )
    ds = _FakeDataset(
        points=points,
        corners=corners,
        variables={
            XYZ_VARS[0]: points[:, 0],
            XYZ_VARS[1]: points[:, 1],
            XYZ_VARS[2]: points[:, 2],
        },
    )
    return OctreeBuilder().build(ds, tree_coord="rpa")


def _build_two_level_topology_tree() -> Octree:
    """Private test helper: synthetic 2-level Cartesian leaf frontier."""
    coarse = [(i0, i1, i2) for i0 in (0, 1) for i1 in (0, 1) for i2 in (0, 1) if (i0, i1, i2) != (0, 0, 0)]
    fine = [(i0, i1, i2) for i0 in (0, 1) for i1 in (0, 1) for i2 in (0, 1)]

    levels = np.concatenate(
        (
            np.zeros(len(coarse), dtype=np.int64),
            np.ones(len(fine), dtype=np.int64),
        )
    )
    i0 = np.concatenate(
        (
            np.array([idx[0] for idx in coarse], dtype=np.int64),
            np.array([idx[0] for idx in fine], dtype=np.int64),
        )
    )
    i1 = np.concatenate(
        (
            np.array([idx[1] for idx in coarse], dtype=np.int64),
            np.array([idx[1] for idx in fine], dtype=np.int64),
        )
    )
    i2 = np.concatenate(
        (
            np.array([idx[2] for idx in coarse], dtype=np.int64),
            np.array([idx[2] for idx in fine], dtype=np.int64),
        )
    )
    n_cells = int(levels.shape[0])
    cell_points: list[np.ndarray] = []
    cell_corners = np.empty((n_cells, 8), dtype=np.int64)
    for cell_id in range(n_cells):
        x0 = 2.0 * float(cell_id)
        pts = np.array(
            [
                [x0 + 0.0, 0.0, 0.0],
                [x0 + 1.0, 0.0, 0.0],
                [x0 + 0.0, 1.0, 0.0],
                [x0 + 1.0, 1.0, 0.0],
                [x0 + 0.0, 0.0, 1.0],
                [x0 + 1.0, 0.0, 1.0],
                [x0 + 0.0, 1.0, 1.0],
                [x0 + 1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        start = 8 * cell_id
        cell_points.append(pts)
        cell_corners[cell_id] = np.arange(start, start + 8, dtype=np.int64)
    points = np.vstack(cell_points)
    ds = _FakeDataset(
        points=points,
        corners=cell_corners,
        variables={
            XYZ_VARS[0]: points[:, 0],
            XYZ_VARS[1]: points[:, 1],
            XYZ_VARS[2]: points[:, 2],
        },
    )

    tree = Octree(
        root_shape=(2, 2, 2),
        tree_coord="xyz",
        cell_levels=levels,
        cell_i0=i0,
        cell_i1=i1,
        cell_i2=i2,
        ds=ds,
    )
    return tree


def _assert_basic_face_neighbor_invariants(topo: OctreeFaceNeighbors) -> None:
    """Private test helper: validate basic face-neighbor graph invariants."""
    assert topo.node_count > 0
    assert topo.face_counts.shape == (topo.node_count, 6)
    assert topo.face_offsets.shape == (topo.node_count * 6 + 1,)
    assert int(topo.face_offsets[0]) == 0
    assert int(topo.face_offsets[-1]) == int(topo.face_neighbors.size)
    assert np.all(np.diff(topo.face_offsets) >= 0)
    np.testing.assert_array_equal(np.diff(topo.face_offsets), topo.face_counts.reshape(-1))

    if topo.face_neighbors.size > 0:
        assert np.all(topo.face_neighbors >= 0)
        assert np.all(topo.face_neighbors < topo.node_count)

    for node_id in range(topo.node_count):
        for face in range(6):
            neighbors = topo.face_neighbor_ids(node_id, face)
            if neighbors.size == 0:
                continue
            assert not np.any(neighbors == node_id)
            assert np.unique(neighbors).size == neighbors.size


def _build_face_neighbors_timed(
    tree: Octree,
    *,
    label: str,
    max_level: int | None = None,
) -> tuple[OctreeFaceNeighbors, float]:
    """Private test helper: build face neighbors and emit a small timing line."""
    t0 = time.perf_counter()
    topo = build_face_neighbors(tree, max_level=max_level)
    elapsed_s = time.perf_counter() - t0
    print(
        f"face-neighbors {label}: {elapsed_s * 1.0e3:.2f} ms, "
        f"nodes={topo.node_count}, face_refs={int(topo.face_neighbors.size)}"
    )
    return topo, elapsed_s


def _candidate_nodes_for_face_mask(topo: OctreeFaceNeighbors, node_id: int, face_mask: int) -> np.ndarray:
    """Private test helper: return face-neighbors candidates for one face-mask exit."""
    kernel_state = topo.kernel_state
    work0 = np.full(max(64, topo.node_count + 1), -1, dtype=np.int64)
    work1 = np.full(max(64, topo.node_count + 1), -1, dtype=np.int64)
    n_candidates, active = _candidate_face_neighbor_nodes_after_exit(
        int(node_id),
        int(face_mask),
        kernel_state,
        work0,
        work1,
    )
    nodes = work0 if active == 0 else work1
    return np.asarray(nodes[:n_candidates], dtype=np.int64)


def _oracle_interpolator(tree: Octree) -> OctreeInterpolator:
    """Private test helper: build the regular interpolator used as oracle."""
    ds = tree.ds
    assert ds is not None
    return OctreeInterpolator(tree, [XYZ_VARS[0]])


def _oracle_ray_cell_ids(
    interp: OctreeInterpolator,
    origin: np.ndarray,
    direction: np.ndarray,
    t_start: float,
    t_end: float,
    *,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Private test helper: sample oracle cell ids along one ray."""
    d = np.asarray(direction, dtype=float).reshape(3)
    d = d / np.linalg.norm(d)
    t_samples = np.linspace(float(t_start), float(t_end), int(n_samples), dtype=float)
    points = np.asarray(origin, dtype=float).reshape(1, 3) + t_samples[:, None] * d[None, :]
    _values, cell_ids = interp(
        points,
        query_coord="xyz",
        return_cell_ids=True,
        log_outside_domain=False,
    )
    return t_samples, np.asarray(cell_ids, dtype=np.int64).reshape(-1)


def _oracle_cell_id(
    interp: OctreeInterpolator,
    origin: np.ndarray,
    direction: np.ndarray,
    t_value: float,
) -> int:
    """Private test helper: resolve one oracle cell id on one ray."""
    d = np.asarray(direction, dtype=float).reshape(3)
    d = d / np.linalg.norm(d)
    point = np.asarray(origin, dtype=float).reshape(1, 3) + float(t_value) * d.reshape(1, 3)
    _values, cell_ids = interp(
        point,
        query_coord="xyz",
        return_cell_ids=True,
        log_outside_domain=False,
    )
    return int(np.asarray(cell_ids, dtype=np.int64).reshape(-1)[0])


def _refine_oracle_interval(
    interp: OctreeInterpolator,
    origin: np.ndarray,
    direction: np.ndarray,
    t0: float,
    cell0: int,
    t1: float,
    cell1: int,
    *,
    depth: int,
) -> list[int]:
    """Private test helper: recover skipped cell ids between two oracle samples."""
    if int(cell0) == int(cell1) or depth <= 0:
        return [int(cell0), int(cell1)]
    tm = 0.5 * (float(t0) + float(t1))
    cellm = _oracle_cell_id(interp, origin, direction, tm)
    left = _refine_oracle_interval(interp, origin, direction, t0, int(cell0), tm, cellm, depth=depth - 1)
    right = _refine_oracle_interval(interp, origin, direction, tm, cellm, t1, int(cell1), depth=depth - 1)
    return left[:-1] + right


def _oracle_cell_sequence(
    interp: OctreeInterpolator,
    origin: np.ndarray,
    direction: np.ndarray,
    t_start: float,
    t_end: float,
    *,
    n_samples: int,
    refine_depth: int = 10,
) -> np.ndarray:
    """Private test helper: oracle cell sequence with recursive refinement on changed intervals."""
    t_samples, cell_ids = _oracle_ray_cell_ids(
        interp,
        origin,
        direction,
        t_start,
        t_end,
        n_samples=n_samples,
    )
    seq: list[int] = []
    for idx in range(t_samples.size - 1):
        interval = _refine_oracle_interval(
            interp,
            origin,
            direction,
            float(t_samples[idx]),
            int(cell_ids[idx]),
            float(t_samples[idx + 1]),
            int(cell_ids[idx + 1]),
            depth=int(refine_depth),
        )
        if not seq:
            seq.extend(interval)
            continue
        seq.extend(interval[1:])
    if t_samples.size == 1:
        seq = [int(cell_ids[0])]
    ids = np.asarray(seq, dtype=np.int64)
    keep = np.ones(ids.shape, dtype=bool)
    keep[1:] = ids[1:] != ids[:-1]
    return ids[keep]


def _valid_cell_sequence(cell_ids: np.ndarray) -> np.ndarray:
    """Private test helper: compress consecutive valid oracle cell ids."""
    ids = np.asarray(cell_ids, dtype=np.int64)
    valid = ids[ids >= 0]
    if valid.size == 0:
        return np.empty(0, dtype=np.int64)
    keep = np.ones(valid.shape, dtype=bool)
    keep[1:] = valid[1:] != valid[:-1]
    return valid[keep]


def _assert_no_internal_gaps(cell_ids: np.ndarray) -> None:
    """Private test helper: chosen oracle rays must stay inside once they enter."""
    ids = np.asarray(cell_ids, dtype=np.int64)
    valid_idx = np.flatnonzero(ids >= 0)
    assert valid_idx.size > 0
    first = int(valid_idx[0])
    last = int(valid_idx[-1])
    assert np.all(ids[first : last + 1] >= 0)


def _node_extents(topo: OctreeFaceNeighbors, node_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Private test helper: node extent in the face-neighbors finest-grid index space."""
    level = int(topo.levels[node_id])
    scale = 1 << int(topo.max_level - level)
    lo = np.array(
        [
            int(topo.i0[node_id]) * scale,
            int(topo.i1[node_id]) * scale,
            int(topo.i2[node_id]) * scale,
        ],
        dtype=np.int64,
    )
    hi = lo + scale
    return lo, hi


def _positive_overlap(lo0: int, hi0: int, lo1: int, hi1: int) -> bool:
    """Private test helper: true when intervals overlap with positive width."""
    return max(int(lo0), int(lo1)) < min(int(hi0), int(hi1))


def _touch_or_overlap(lo0: int, hi0: int, lo1: int, hi1: int) -> bool:
    """Private test helper: true when intervals touch or overlap."""
    return max(int(lo0), int(lo1)) <= min(int(hi0), int(hi1))


def _face_masks_between_nodes(tree: Octree, topo: OctreeFaceNeighbors, node_a: int, node_b: int) -> np.ndarray:
    """Private test helper: admissible face masks on `node_a` that can lead to `node_b`."""
    lo_a, hi_a = _node_extents(topo, node_a)
    lo_b, hi_b = _node_extents(topo, node_b)
    period = int(tree.root_shape[2]) * (1 << int(topo.max_level))
    shifts = (0,) if not topo.periodic_i2 else (-period, 0, period)

    matches: list[int] = []
    for shift in shifts:
        face_mask = 0
        valid = True
        for axis in range(3):
            blo = int(lo_b[axis])
            bhi = int(hi_b[axis])
            if axis == 2:
                blo += int(shift)
                bhi += int(shift)
            alo = int(lo_a[axis])
            ahi = int(hi_a[axis])
            if not _touch_or_overlap(alo, ahi, blo, bhi):
                valid = False
                break
            if ahi == blo:
                face_mask |= 1 << (2 * axis + 1)
                continue
            if alo == bhi:
                face_mask |= 1 << (2 * axis)
                continue
            if not _positive_overlap(alo, ahi, blo, bhi):
                valid = False
                break
        if valid and face_mask != 0:
            matches.append(int(face_mask))

    unique = np.unique(np.asarray(matches, dtype=np.int64))
    assert unique.size > 0
    return unique


def _assert_oracle_sequence_is_adjacent(topo: OctreeFaceNeighbors, cell_ids: np.ndarray) -> None:
    """Private test helper: each oracle cell transition must follow one graph edge set."""
    seq = _valid_cell_sequence(cell_ids)
    assert seq.size > 0
    for cell_a, cell_b in zip(seq[:-1], seq[1:], strict=True):
        node_a = int(topo.cell_to_node_id[int(cell_a)])
        node_b = int(topo.cell_to_node_id[int(cell_b)])
        assert node_a >= 0
        assert node_b >= 0
        found = False
        for face in range(6):
            if np.any(topo.face_neighbor_ids(node_a, face) == node_b):
                found = True
                break
        assert found


def _assert_oracle_sequence_is_admissible_by_transition_rule(
    tree: Octree,
    topo: OctreeFaceNeighbors,
    cell_ids: np.ndarray,
) -> None:
    """Private test helper: each oracle next cell must be in the face-neighbors candidate set."""
    seq = _valid_cell_sequence(cell_ids)
    assert seq.size > 0
    for cell_a, cell_b in zip(seq[:-1], seq[1:], strict=True):
        node_a = int(topo.cell_to_node_id[int(cell_a)])
        node_b = int(topo.cell_to_node_id[int(cell_b)])
        face_masks = _face_masks_between_nodes(tree, topo, node_a, node_b)
        found = False
        for face_mask in face_masks:
            candidates = _candidate_nodes_for_face_mask(topo, node_a, int(face_mask))
            if np.any(candidates == node_b):
                found = True
                break
        assert found


def _time_call(func, /, *args, **kwargs):
    """Private test helper: time one function call and return result plus seconds."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_s = time.perf_counter() - t0
    return result, elapsed_s


def _oracle_sample_count(tree: Octree, *, minimum: int = 2049) -> int:
    """Private test helper: dense-enough sample count for one oracle ray."""
    along_x = int(tree.leaf_shape[0])
    return max(int(minimum), 8 * along_x + 1)


def _cartesian_face_ray() -> tuple[np.ndarray, np.ndarray, float]:
    """Private test helper: face-only Cartesian ray through the small uniform tree."""
    origin = np.array([-0.25, 0.5, 0.5], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    return origin, direction, 2.5


def _spherical_shell_face_ray() -> tuple[np.ndarray, np.ndarray, float]:
    """Private test helper: off-axis ray through the spherical shell with no cavity gap."""
    origin = np.array([-2.5, 1.25, 0.0], dtype=float)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    return origin, direction, 5.0


def _axis_sample_ray(
    tree: Octree,
    axis: str,
    *,
    n_plane: int = 4,
    i_first: int = 1,
    i_second: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Private test helper: build one off-center axis ray through the tree domain."""
    xyz_min, xyz_max = tree.domain_bounds(coord="xyz")
    xyz_min = np.asarray(xyz_min, dtype=float)
    xyz_max = np.asarray(xyz_max, dtype=float)
    grids = [np.linspace(float(xyz_min[k]), float(xyz_max[k]), int(n_plane), dtype=float) for k in range(3)]
    spans = xyz_max - xyz_min

    if axis == "x":
        origin = np.array(
            [
                float(xyz_min[0] - 1.0e-6 * max(1.0, float(spans[0]))),
                float(grids[1][int(i_first)]),
                float(grids[2][int(i_second)]),
            ],
            dtype=float,
        )
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
        t_end = float((xyz_max[0] - origin[0]) * 0.999999)
    elif axis == "y":
        origin = np.array(
            [
                float(grids[0][int(i_first)]),
                float(xyz_min[1] - 1.0e-6 * max(1.0, float(spans[1]))),
                float(grids[2][int(i_second)]),
            ],
            dtype=float,
        )
        direction = np.array([0.0, 1.0, 0.0], dtype=float)
        t_end = float((xyz_max[1] - origin[1]) * 0.999999)
    elif axis == "z":
        origin = np.array(
            [
                float(grids[0][int(i_first)]),
                float(grids[1][int(i_second)]),
                float(xyz_min[2] - 1.0e-6 * max(1.0, float(spans[2]))),
            ],
            dtype=float,
        )
        direction = np.array([0.0, 0.0, 1.0], dtype=float)
        t_end = float((xyz_max[2] - origin[2]) * 0.999999)
    else:
        raise ValueError(f"Unsupported axis '{axis}'.")
    return origin, direction, t_end


def _sample_tree_and_ray(file_name: str, tree_coord: str, *, axis: str = "x") -> tuple[Octree, np.ndarray, np.ndarray, float]:
    """Private test helper: load one real file and choose one representative axis ray."""
    ds = Dataset.from_file(str(data_file(file_name)))
    tree = OctreeBuilder().build(ds, tree_coord=tree_coord)
    origin, direction, t_end = _axis_sample_ray(tree, axis)
    return tree, origin, direction, t_end


@pytest.fixture(scope="session", autouse=True)
def _warm_topology_numba() -> None:
    """Private test helper: compile face-neighbors kernels before timing assertions/prints."""
    build_face_neighbors(_build_cartesian_uniform_tree())
    build_face_neighbors(_build_spherical_uniform_tree())


@pytest.mark.parametrize(
    "tree_builder",
    [_build_cartesian_uniform_tree, _build_spherical_uniform_tree],
    ids=["cartesian_uniform", "spherical_uniform"],
)
def test_uniform_topology_basic_invariants(tree_builder, record_property) -> None:
    tree = tree_builder()
    topo, elapsed_s = _build_face_neighbors_timed(tree, label=tree.tree_coord)
    record_property("build_topology_ms", round(elapsed_s * 1.0e3, 3))
    _assert_basic_face_neighbor_invariants(topo)
    assert np.all(topo.face_counts <= 1)


def test_octree_face_neighbors_method_reuses_cached_graph() -> None:
    tree = _build_two_level_topology_tree()
    full = tree.face_neighbors()
    repeat = tree.face_neighbors()
    coarse = tree.face_neighbors(max_level=0)

    assert full is repeat
    assert int(full.max_level) == int(tree.max_level)
    assert int(coarse.max_level) == 0


def test_cartesian_uniform_topology_neighbors_are_bidirectional(record_property) -> None:
    tree = _build_cartesian_uniform_tree()
    topo, elapsed_s = _build_face_neighbors_timed(tree, label="cartesian_bidirectional")
    record_property("build_topology_ms", round(elapsed_s * 1.0e3, 3))

    assert topo.node_count == int(tree.cell_count)
    np.testing.assert_array_equal(
        topo.face_counts.sum(axis=1),
        np.full(topo.node_count, 3, dtype=np.int64),
    )

    for node_id in range(topo.node_count):
        for face in range(6):
            opposite = face ^ 1
            neighbors = topo.face_neighbor_ids(node_id, face)
            for nbr in neighbors:
                back = topo.face_neighbor_ids(int(nbr), opposite)
                assert np.any(back == node_id)


def test_spherical_uniform_topology_wraps_azimuth_faces(record_property) -> None:
    tree = _build_spherical_uniform_tree()
    topo, elapsed_s = _build_face_neighbors_timed(tree, label="spherical_wrap")
    record_property("build_topology_ms", round(elapsed_s * 1.0e3, 3))

    assert topo.periodic_i2
    np.testing.assert_array_equal(
        topo.face_counts.sum(axis=1),
        np.full(topo.node_count, 3, dtype=np.int64),
    )

    i2_max = int(np.max(topo.i2))
    nodes_on_zero = np.flatnonzero(topo.i2 == 0)
    assert nodes_on_zero.size > 0
    for node_id in nodes_on_zero:
        left = topo.face_neighbor_ids(int(node_id), 4)
        assert left.size == 1
        assert int(topo.i2[int(left[0])]) == i2_max


def test_cartesian_corner_exit_candidates_cover_faces_edges_and_corner() -> None:
    tree = _build_cartesian_uniform_tree()
    topo = build_face_neighbors(tree)

    node_id = 0
    face_mask = (1 << 1) | (1 << 3) | (1 << 5)
    candidates = _candidate_nodes_for_face_mask(topo, node_id, face_mask)

    expected = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    np.testing.assert_array_equal(np.sort(candidates), expected)


def test_spherical_periodic_multiface_candidates_cover_edge_and_corner() -> None:
    tree = _build_spherical_uniform_tree()
    topo = build_face_neighbors(tree)

    node_id = 0
    face_mask = (1 << 3) | (1 << 4)
    candidates = _candidate_nodes_for_face_mask(topo, node_id, face_mask)

    expected = np.array([3, 4, 7], dtype=np.int64)
    np.testing.assert_array_equal(np.sort(candidates), expected)


def test_max_level_cutoff_reduces_frontier_size(record_property) -> None:
    tree = _build_two_level_topology_tree()

    topo_full, elapsed_full_s = _build_face_neighbors_timed(tree, label="cutoff_full", max_level=1)
    topo_coarse, elapsed_coarse_s = _build_face_neighbors_timed(tree, label="cutoff_coarse", max_level=0)
    record_property("build_topology_full_ms", round(elapsed_full_s * 1.0e3, 3))
    record_property("build_topology_coarse_ms", round(elapsed_coarse_s * 1.0e3, 3))

    assert topo_full.max_level == 1
    assert topo_coarse.max_level == 0
    assert topo_coarse.node_count <= topo_full.node_count
    assert topo_coarse.node_count < topo_full.node_count
    assert topo_full.node_count == 15
    assert topo_coarse.node_count == 8

    _assert_basic_face_neighbor_invariants(topo_full)
    _assert_basic_face_neighbor_invariants(topo_coarse)
    assert np.all(topo_full.face_counts <= 4)
    assert np.all(topo_coarse.face_counts <= 1)
    assert np.all(topo_full.face_counts.sum(axis=1) <= 24)


def test_lookup_oracle_cartesian_face_ray_has_no_internal_gaps() -> None:
    tree = _build_cartesian_uniform_tree()
    interp = _oracle_interpolator(tree)
    origin, direction, t_end = _cartesian_face_ray()
    _t, cell_ids = _oracle_ray_cell_ids(interp, origin, direction, 0.0, t_end, n_samples=513)
    _assert_no_internal_gaps(cell_ids)
    assert _valid_cell_sequence(cell_ids).size == 2


def test_lookup_oracle_spherical_face_ray_has_no_internal_gaps() -> None:
    tree = _build_spherical_uniform_tree()
    interp = _oracle_interpolator(tree)
    origin, direction, t_end = _spherical_shell_face_ray()
    _t, cell_ids = _oracle_ray_cell_ids(interp, origin, direction, 0.0, t_end, n_samples=1025)
    _assert_no_internal_gaps(cell_ids)
    assert _valid_cell_sequence(cell_ids).size >= 2


def test_cartesian_lookup_oracle_transitions_are_adjacent_in_topology() -> None:
    tree = _build_cartesian_uniform_tree()
    topo = build_face_neighbors(tree)
    interp = _oracle_interpolator(tree)
    origin, direction, t_end = _cartesian_face_ray()
    cell_ids = _oracle_cell_sequence(interp, origin, direction, 0.0, t_end, n_samples=129, refine_depth=8)
    _assert_oracle_sequence_is_adjacent(topo, cell_ids)


def test_spherical_lookup_oracle_transitions_are_adjacent_in_topology() -> None:
    tree = _build_spherical_uniform_tree()
    topo = build_face_neighbors(tree)
    interp = _oracle_interpolator(tree)
    origin, direction, t_end = _spherical_shell_face_ray()
    cell_ids = _oracle_cell_sequence(interp, origin, direction, 0.0, t_end, n_samples=257, refine_depth=8)
    _assert_oracle_sequence_is_adjacent(topo, cell_ids)


def test_cartesian_lookup_oracle_transitions_are_admissible_by_face_mask() -> None:
    tree = _build_cartesian_uniform_tree()
    topo = build_face_neighbors(tree)
    interp = _oracle_interpolator(tree)
    origin, direction, t_end = _cartesian_face_ray()
    cell_ids = _oracle_cell_sequence(interp, origin, direction, 0.0, t_end, n_samples=129, refine_depth=8)
    _assert_oracle_sequence_is_admissible_by_transition_rule(tree, topo, cell_ids)


def test_spherical_lookup_oracle_transitions_are_admissible_by_face_mask() -> None:
    tree = _build_spherical_uniform_tree()
    topo = build_face_neighbors(tree)
    interp = _oracle_interpolator(tree)
    origin, direction, t_end = _spherical_shell_face_ray()
    cell_ids = _oracle_cell_sequence(interp, origin, direction, 0.0, t_end, n_samples=257, refine_depth=8)
    _assert_oracle_sequence_is_admissible_by_transition_rule(tree, topo, cell_ids)


_TOPOLOGY_SAMPLE_CASES = [
    pytest.param(
        "3d__var_1_n00000000.plt",
        "rpa",
        id="provided_example_file",
    ),
    pytest.param(
        "3d__var_4_n00044000.plt",
        "rpa",
        id="pooch_sc",
        marks=[pytest.mark.pooch, pytest.mark.slow],
    ),
    pytest.param(
        "3d__var_4_n00005000.plt",
        "xyz",
        id="pooch_ih",
        marks=[pytest.mark.pooch, pytest.mark.slow],
    ),
]


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_example_lookup_oracle_ray_has_no_internal_gaps_on_axis_rays(axis: str) -> None:
    tree, origin, direction, t_end = _sample_tree_and_ray("3d__var_1_n00000000.plt", "rpa", axis=axis)
    interp = _oracle_interpolator(tree)
    _t, cell_ids = _oracle_ray_cell_ids(
        interp,
        origin,
        direction,
        0.0,
        t_end,
        n_samples=_oracle_sample_count(tree, minimum=1025),
    )
    _assert_no_internal_gaps(cell_ids)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_example_lookup_oracle_transitions_are_adjacent_in_topology_on_axis_rays(axis: str) -> None:
    tree, origin, direction, t_end = _sample_tree_and_ray("3d__var_1_n00000000.plt", "rpa", axis=axis)
    topo = build_face_neighbors(tree)
    interp = _oracle_interpolator(tree)
    cell_ids = _oracle_cell_sequence(
        interp,
        origin,
        direction,
        0.0,
        t_end,
        n_samples=_oracle_sample_count(tree, minimum=513),
        refine_depth=10,
    )
    _assert_oracle_sequence_is_adjacent(topo, cell_ids)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_example_lookup_oracle_transitions_are_admissible_by_face_mask_on_axis_rays(axis: str) -> None:
    tree, origin, direction, t_end = _sample_tree_and_ray("3d__var_1_n00000000.plt", "rpa", axis=axis)
    topo = build_face_neighbors(tree)
    interp = _oracle_interpolator(tree)
    cell_ids = _oracle_cell_sequence(
        interp,
        origin,
        direction,
        0.0,
        t_end,
        n_samples=_oracle_sample_count(tree, minimum=513),
        refine_depth=10,
    )
    _assert_oracle_sequence_is_admissible_by_transition_rule(tree, topo, cell_ids)


@pytest.mark.parametrize("file_name,tree_coord", _TOPOLOGY_SAMPLE_CASES)
def test_sample_lookup_oracle_ray_has_no_internal_gaps(file_name: str, tree_coord: str) -> None:
    tree, origin, direction, t_end = _sample_tree_and_ray(file_name, tree_coord)
    interp = _oracle_interpolator(tree)
    _t, cell_ids = _oracle_ray_cell_ids(
        interp,
        origin,
        direction,
        0.0,
        t_end,
        n_samples=_oracle_sample_count(tree),
    )
    _assert_no_internal_gaps(cell_ids)


@pytest.mark.parametrize("file_name,tree_coord", _TOPOLOGY_SAMPLE_CASES)
def test_sample_lookup_oracle_transitions_are_adjacent_in_topology(file_name: str, tree_coord: str) -> None:
    tree, origin, direction, t_end = _sample_tree_and_ray(file_name, tree_coord)
    topo = build_face_neighbors(tree)
    interp = _oracle_interpolator(tree)
    cell_ids = _oracle_cell_sequence(
        interp,
        origin,
        direction,
        0.0,
        t_end,
        n_samples=_oracle_sample_count(tree),
        refine_depth=10,
    )
    _assert_oracle_sequence_is_adjacent(topo, cell_ids)


@pytest.mark.parametrize("file_name,tree_coord", _TOPOLOGY_SAMPLE_CASES)
def test_sample_lookup_oracle_transitions_are_admissible_by_face_mask(file_name: str, tree_coord: str) -> None:
    tree, origin, direction, t_end = _sample_tree_and_ray(file_name, tree_coord)
    topo = build_face_neighbors(tree)
    interp = _oracle_interpolator(tree)
    cell_ids = _oracle_cell_sequence(
        interp,
        origin,
        direction,
        0.0,
        t_end,
        n_samples=_oracle_sample_count(tree),
        refine_depth=10,
    )
    _assert_oracle_sequence_is_admissible_by_transition_rule(tree, topo, cell_ids)


@pytest.mark.parametrize("file_name,tree_coord", _TOPOLOGY_SAMPLE_CASES)
def test_topological_neighborhood_on_sample_files(file_name: str, tree_coord: str, record_property) -> None:
    path, resolve_path_s = _time_call(data_file, file_name)
    ds, read_dataset_s = _time_call(Dataset.from_file, str(path))
    tree, build_tree_s = _time_call(OctreeBuilder().build, ds, tree_coord=tree_coord)
    topo, build_topology_s = _build_face_neighbors_timed(tree, label=file_name)
    _unused, validate_invariants_s = _time_call(_assert_basic_face_neighbor_invariants, topo)

    print(
        f"sample {file_name}: "
        f"resolve={resolve_path_s * 1.0e3:.2f} ms, "
        f"read={read_dataset_s * 1.0e3:.2f} ms, "
        f"tree={build_tree_s * 1.0e3:.2f} ms, "
        f"topology={build_topology_s * 1.0e3:.2f} ms, "
        f"validate={validate_invariants_s * 1.0e3:.2f} ms"
    )
    record_property("resolve_path_ms", round(resolve_path_s * 1.0e3, 3))
    record_property("read_dataset_ms", round(read_dataset_s * 1.0e3, 3))
    record_property("build_tree_ms", round(build_tree_s * 1.0e3, 3))
    record_property("build_topology_ms", round(build_topology_s * 1.0e3, 3))
    record_property("validate_invariants_ms", round(validate_invariants_s * 1.0e3, 3))
    assert topo.node_count == int(tree.cell_count)
    assert topo.max_level == int(tree.max_level)
    assert topo.periodic_i2 == (tree_coord == "rpa")
    assert int(topo.face_neighbors.size) > 0

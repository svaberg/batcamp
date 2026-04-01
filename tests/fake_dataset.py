from __future__ import annotations

import numpy as np


class FakeDataset:
    """Minimal in-memory dataset object with the `Dataset` methods used in tests."""

    def __init__(
        self,
        points: np.ndarray,
        corners: np.ndarray | None,
        variables: dict[str, np.ndarray],
        *,
        aux: dict[str, str] | None = None,
    ) -> None:
        """Store geometry and variables using names that match `Dataset`."""
        self.points = points
        self.corners = corners
        self._variables = variables
        self.variables = list(variables.keys())
        self.aux = {} if aux is None else dict(aux)

    def __getitem__(self, name: str) -> np.ndarray:
        """Return one variable array by name."""
        return self._variables[name]


def build_spherical_hex_mesh(
    *,
    nr: int,
    npolar: int,
    nazimuth: int,
    r_min: float,
    r_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular spherical hexahedral mesh and return points/corners."""
    r_edges = np.linspace(float(r_min), float(r_max), int(nr) + 1)
    polar_edges = np.linspace(0.0, np.pi, int(npolar) + 1)
    azimuth_edges = np.linspace(0.0, 2.0 * np.pi, int(nazimuth) + 1)

    node_index = -np.ones((int(nr) + 1, int(npolar) + 1, int(nazimuth) + 1), dtype=np.int64)
    xyz_list: list[tuple[float, float, float]] = []
    node_id = 0
    for ir in range(int(nr) + 1):
        r = float(r_edges[ir])
        for ipolar in range(int(npolar) + 1):
            polar = float(polar_edges[ipolar])
            st = np.sin(polar)
            ct = np.cos(polar)
            for iazimuth in range(int(nazimuth) + 1):
                azimuth = float(azimuth_edges[iazimuth])
                xyz_list.append((r * st * np.cos(azimuth), r * st * np.sin(azimuth), r * ct))
                node_index[ir, ipolar, iazimuth] = node_id
                node_id += 1

    corners: list[list[int]] = []
    for ir in range(int(nr)):
        for ipolar in range(int(npolar)):
            for iazimuth in range(int(nazimuth)):
                corners.append(
                    [
                        int(node_index[ir, ipolar + 1, iazimuth]),
                        int(node_index[ir + 1, ipolar + 1, iazimuth]),
                        int(node_index[ir + 1, ipolar + 1, iazimuth + 1]),
                        int(node_index[ir, ipolar + 1, iazimuth + 1]),
                        int(node_index[ir, ipolar, iazimuth]),
                        int(node_index[ir + 1, ipolar, iazimuth]),
                        int(node_index[ir + 1, ipolar, iazimuth + 1]),
                        int(node_index[ir, ipolar, iazimuth + 1]),
                    ]
                )
    return np.array(xyz_list, dtype=float), np.array(corners, dtype=np.int64)


def build_cartesian_hex_mesh(
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular Cartesian hexahedral mesh and return points/corners."""
    x_arr = np.asarray(x_edges, dtype=float)
    y_arr = np.asarray(y_edges, dtype=float)
    z_arr = np.asarray(z_edges, dtype=float)

    node_index = -np.ones((x_arr.size, y_arr.size, z_arr.size), dtype=np.int64)
    xyz_list: list[tuple[float, float, float]] = []
    node_id = 0
    for ix, x in enumerate(x_arr):
        for iy, y in enumerate(y_arr):
            for iz, z in enumerate(z_arr):
                xyz_list.append((float(x), float(y), float(z)))
                node_index[ix, iy, iz] = node_id
                node_id += 1

    corners: list[list[int]] = []
    for ix in range(x_arr.size - 1):
        for iy in range(y_arr.size - 1):
            for iz in range(z_arr.size - 1):
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
    return np.array(xyz_list, dtype=float), np.array(corners, dtype=np.int64)

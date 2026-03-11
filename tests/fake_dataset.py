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

    def variable(self, name: str) -> np.ndarray:
        """Return one variable array by name."""
        return self._variables[name]


def build_spherical_hex_mesh(
    *,
    nr: int,
    ntheta: int,
    nphi: int,
    r_min: float,
    r_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular spherical hexahedral mesh and return points/corners."""
    r_edges = np.linspace(float(r_min), float(r_max), int(nr) + 1)
    theta_edges = np.linspace(0.0, np.pi, int(ntheta) + 1)
    phi_edges = np.linspace(0.0, 2.0 * np.pi, int(nphi) + 1)

    node_index = -np.ones((int(nr) + 1, int(ntheta) + 1, int(nphi) + 1), dtype=np.int64)
    xyz_list: list[tuple[float, float, float]] = []
    node_id = 0
    for ir in range(int(nr) + 1):
        r = float(r_edges[ir])
        for it in range(int(ntheta) + 1):
            theta = float(theta_edges[it])
            st = np.sin(theta)
            ct = np.cos(theta)
            for ip in range(int(nphi) + 1):
                phi = float(phi_edges[ip])
                xyz_list.append((r * st * np.cos(phi), r * st * np.sin(phi), r * ct))
                node_index[ir, it, ip] = node_id
                node_id += 1

    corners: list[list[int]] = []
    for ir in range(int(nr)):
        for it in range(int(ntheta)):
            for ip in range(int(nphi)):
                corners.append(
                    [
                        int(node_index[ir, it, ip]),
                        int(node_index[ir + 1, it, ip]),
                        int(node_index[ir, it + 1, ip]),
                        int(node_index[ir + 1, it + 1, ip]),
                        int(node_index[ir, it, ip + 1]),
                        int(node_index[ir + 1, it, ip + 1]),
                        int(node_index[ir, it + 1, ip + 1]),
                        int(node_index[ir + 1, it + 1, ip + 1]),
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
                        int(node_index[ix, iy + 1, iz]),
                        int(node_index[ix + 1, iy + 1, iz]),
                        int(node_index[ix, iy, iz + 1]),
                        int(node_index[ix + 1, iy, iz + 1]),
                        int(node_index[ix, iy + 1, iz + 1]),
                        int(node_index[ix + 1, iy + 1, iz + 1]),
                    ]
                )
    return np.array(xyz_list, dtype=float), np.array(corners, dtype=np.int64)

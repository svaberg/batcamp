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

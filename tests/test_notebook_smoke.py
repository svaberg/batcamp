from __future__ import annotations

import json
from pathlib import Path

import pytest


_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

_CASES = [
    pytest.param(
        "quick_start.ipynb",
        ("OctreeInterpolator(ds",),
        ("tree_coord=",),
        id="quick_start_default_first",
    ),
    pytest.param(
        "octree_resampling_plane_sphere.ipynb",
        ("tree_coord='xyz'",),
        (),
        id="cartesian_resampling_notebook",
    ),
    pytest.param(
        "octree.ipynb",
        ("tree_coord='rpa'",),
        (),
        id="octree_notebook",
    ),
    pytest.param(
        "octree_ray_demo.ipynb",
        ("OctreeRayTracer",),
        (),
        id="ray_notebook",
    ),
]


@pytest.mark.parametrize("notebook_name,required_tokens,forbidden_tokens", _CASES)
def test_notebook_smoke_has_parseable_code_and_expected_api_usage(
    notebook_name: str,
    required_tokens: tuple[str, ...],
    forbidden_tokens: tuple[str, ...],
) -> None:
    """Smoke-check selected notebooks so API drift is caught in CI."""
    notebook_path = _EXAMPLES_DIR / notebook_name
    assert notebook_path.exists(), f"Notebook missing: {notebook_path}"

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    assert code_cells, f"No code cells found in {notebook_path}"

    source = "\n\n".join(code_cells)
    compile(source, str(notebook_path), "exec")

    for token in required_tokens:
        assert token in source, f"Expected token {token!r} in {notebook_path}"
    for token in forbidden_tokens:
        assert token not in source, f"Unexpected token {token!r} in {notebook_path}"

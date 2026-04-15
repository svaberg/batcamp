from __future__ import annotations

from pathlib import Path

import pytest


_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
_CELL_TIMEOUT_SECONDS = 1200

_NOTEBOOK_CASES = [
    pytest.param(
        "ray_image.ipynb",
        id="ray_image_exec",
    ),
    pytest.param(
        "quick_start.ipynb",
        id="quick_start_notebook_exec",
        marks=pytest.mark.pooch,
    ),
]


@pytest.mark.parametrize("notebook_name", _NOTEBOOK_CASES)
def test_execute_examples(notebook_name: str) -> None:
    """Execute example notebooks end-to-end to catch runtime/API regressions."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    notebook_path = _EXAMPLES_DIR / notebook_name
    assert notebook_path.exists(), f"Notebook missing: {notebook_path}"

    with notebook_path.open(encoding="utf-8") as fh:
        notebook = nbformat.read(fh, as_version=4)

    client = nbclient.NotebookClient(
        notebook,
        timeout=_CELL_TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(_EXAMPLES_DIR)}},
        allow_errors=False,
        interrupt_on_timeout=True,
    )
    client.execute()

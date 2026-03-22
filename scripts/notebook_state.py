#!/usr/bin/env python3
"""Set notebooks to the commit state required by the current branch policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

_CELL_TIMEOUT_SECONDS = 1200
_SKIP_TAG = "skip-execution"


def _read_notebook(path: Path):
    with path.open(encoding="utf-8") as fh:
        return nbformat.read(fh, as_version=4)


def _write_notebook(path: Path, notebook) -> None:
    with path.open("w", encoding="utf-8") as fh:
        nbformat.write(notebook, fh)


def _run_notebook(path: Path) -> None:
    notebook = _read_notebook(path)
    client = NotebookClient(
        notebook,
        timeout=_CELL_TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
        interrupt_on_timeout=True,
        skip_cells_with_tag=_SKIP_TAG,
    )
    client.execute()
    _write_notebook(path, notebook)


def _strip_notebook(path: Path) -> None:
    notebook = _read_notebook(path)
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []
    _write_notebook(path, notebook)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "strip"))
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    action = _run_notebook if args.mode == "run" else _strip_notebook
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.suffix != ".ipynb":
            raise ValueError(f"Expected a notebook path, got {path}.")
        if not path.exists():
            raise FileNotFoundError(path)
        action(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path


def data_file(name: str) -> Path:
    """Return absolute path to a file in this repo's example_data folder."""
    root = Path(__file__).resolve().parents[1]
    path = root / "example_data" / name
    if not path.exists():
        raise FileNotFoundError(f"Missing example data file: {path}")
    return path

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_standalone_benchmarks_import_checkout_package() -> None:
    """Benchmark modules imported from `examples/` should use this checkout."""
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    code = """
import pathlib
import benchmark_grid_vs_ray
import benchmark_ray_step_costs
import benchmark_xy_plane
import benchmark_random_points
import batcamp
print(pathlib.Path(batcamp.__file__).resolve())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=examples_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    imported_path = Path(result.stdout.strip())
    assert imported_path == repo_root / "batcamp" / "__init__.py"

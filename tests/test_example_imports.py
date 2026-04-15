from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_BENCHMARK_MODULES = [
    "examples.benchmark_grid_vs_ray",
    "examples.benchmark_ray_step_costs",
    "examples.benchmark_xy_plane",
    "examples.benchmark_random_points",
]


@pytest.mark.parametrize("module_name", _BENCHMARK_MODULES)
def test_benchmark_modules_run_from_repo_root(module_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=repo_root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

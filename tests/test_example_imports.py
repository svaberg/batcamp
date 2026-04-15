from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_BENCHMARK_SCRIPTS = [
    "benchmark_grid_vs_ray.py",
    "benchmark_ray_step_costs.py",
    "benchmark_xy_plane.py",
    "benchmark_random_points.py",
]


def _env_without_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


@pytest.mark.parametrize("script_name", _BENCHMARK_SCRIPTS)
def test_benchmark_scripts_run_from_examples(script_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"

    subprocess.run(
        [sys.executable, script_name, "--help"],
        cwd=examples_dir,
        env=_env_without_pythonpath(),
        check=True,
        text=True,
        capture_output=True,
    )


def test_examples_import_repo_editable_install() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    code = """
import pathlib
import batcamp

print(pathlib.Path(batcamp.__file__).resolve())
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=examples_dir,
        env=_env_without_pythonpath(),
        check=True,
        text=True,
        capture_output=True,
    )

    assert Path(result.stdout.strip()) == repo_root / "batcamp" / "__init__.py"

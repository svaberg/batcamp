from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def _load_benchmark_grid_vs_ray_module():
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "benchmark_grid_vs_ray_for_test",
            examples_dir / "benchmark_grid_vs_ray.py",
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


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


def test_grid_vs_ray_grid_sum_streams_pixel_chunks(monkeypatch) -> None:
    module = _load_benchmark_grid_vs_ray_module()
    monkeypatch.setattr(module, "_GRID_SUM_PIXEL_CHUNK_SIZE", 3)
    calls: list[int] = []

    class LinearInterp:
        def __call__(self, query, *, query_coord: str, log_outside_domain: bool):
            assert query_coord == "xyz"
            assert not log_outside_domain
            calls.append(int(query.shape[0]))
            return query[:, 0] + 2.0 * query[:, 1] + 3.0 * query[:, 2]

    bounds = (0.0, 1.0, 10.0, 14.0, 20.0, 24.0)
    image = module._grid_sum_image(LinearInterp(), n_plane=4, nx_sum=5, bounds=bounds)

    y = module._plane_axis_points(bounds[2], bounds[3], 4)
    z = module._plane_axis_points(bounds[4], bounds[5], 4)
    yg, zg = np.meshgrid(y, z, indexing="xy")
    expected = 0.5 + 2.0 * yg + 3.0 * zg

    np.testing.assert_allclose(image, expected, atol=1.0e-12, rtol=0.0)
    assert calls == [15, 15, 15, 15, 15, 5]

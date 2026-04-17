from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from batcamp import spherical_crossing_trace
from sample_data_helper import local_data_file


_BENCHMARK_SCRIPTS = [
    "benchmark_grid_vs_ray.py",
    "benchmark_xy_plane.py",
    "benchmark_random_points.py",
    "benchmark_star_movie.py",
]

_BENCHMARK_TEST_MAX_SECONDS = 10.0
_BENCHMARK_TEST_EXPECTED_IMAGE_RESOLUTION = 256
_EXAMPLE_LOGGERS = (
    "resample.progress",
    "batcamp.builder",
    "batcamp.octree",
    "batcamp.interpolator",
    "batcamp.ray",
)


def _env_without_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


@pytest.fixture(autouse=True)
def _restore_example_logger_state():
    saved = {}
    for name in _EXAMPLE_LOGGERS:
        logger = logging.getLogger(name)
        saved[name] = (
            list(logger.handlers),
            int(logger.level),
            bool(logger.propagate),
        )
    try:
        yield
    finally:
        for name, (handlers, level, propagate) in saved.items():
            logger = logging.getLogger(name)
            for handler in list(logger.handlers):
                if handler not in handlers:
                    handler.close()
            logger.handlers = list(handlers)
            logger.setLevel(level)
            logger.propagate = propagate


def _load_example_module(script_name: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(examples_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            examples_dir / script_name,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(examples_dir))


def _load_benchmark_grid_vs_ray_module():
    return _load_example_module("benchmark_grid_vs_ray.py", "benchmark_grid_vs_ray_for_test")


def _example_benchmark_output_root(repo_root: Path) -> Path:
    out_root = repo_root / "artifacts" / "test_example_scripts"
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def _example_benchmark_progress(module, out_root: Path, log_name: str):
    log_path = out_root / log_name
    log_path.write_text("", encoding="utf-8")
    module._configure_progress_logging(log_path=log_path)
    module._configure_builder_logging(log_path=log_path)
    return module._ProgressReporter(log_path=log_path), log_path


def _resolve_local_data_file(_repo_root: Path, name: str) -> Path:
    """Resolve one bundled sample file without falling back to pooch."""
    return local_data_file(name)


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


def test_grid_vs_ray_benchmark_runs_one_example_case() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_example_module("benchmark_grid_vs_ray.py", "benchmark_grid_vs_ray_run_for_test")
    module.resolve_data_file = _resolve_local_data_file
    out_root = _example_benchmark_output_root(repo_root)
    expected_path = out_root / (
        "benchmark_grid_vs_ray_example_trilinear_"
        f"resample_grid_vs_ray_{_BENCHMARK_TEST_EXPECTED_IMAGE_RESOLUTION}x"
        f"{_BENCHMARK_TEST_EXPECTED_IMAGE_RESOLUTION}.png"
    )
    expected_path.unlink(missing_ok=True)
    progress, log_path = _example_benchmark_progress(module, out_root, "benchmark_grid_vs_ray.log")

    module._run_case(
        case=module.DatasetCase("example", "3d__var_1_n00000000.plt"),
        repo_root=repo_root,
        out_root=out_root,
        progress=progress,
        resolutions=module._resolution_ramp(4, 1024),
        max_seconds_per_image=_BENCHMARK_TEST_MAX_SECONDS,
        nx_sum=_BENCHMARK_TEST_EXPECTED_IMAGE_RESOLUTION,
        ray_method="trilinear",
        variable="Rho [g/cm^3]",
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert expected_path.exists()
    assert "[example] done -> benchmark_grid_vs_ray_example_<method>_*" in log_text


def test_xy_plane_benchmark_runs_one_example_case() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_example_module("benchmark_xy_plane.py", "benchmark_xy_plane_run_for_test")
    module.resolve_data_file = _resolve_local_data_file
    out_root = _example_benchmark_output_root(repo_root)
    expected_path = out_root / (
        "benchmark_xy_plane_example_"
        f"xy_plane_{_BENCHMARK_TEST_EXPECTED_IMAGE_RESOLUTION}x"
        f"{_BENCHMARK_TEST_EXPECTED_IMAGE_RESOLUTION}.png"
    )
    expected_path.unlink(missing_ok=True)
    progress, log_path = _example_benchmark_progress(module, out_root, "benchmark_xy_plane.log")

    module._run_case(
        case=module.DatasetCase("example", "3d__var_1_n00000000.plt"),
        repo_root=repo_root,
        out_root=out_root,
        progress=progress,
        resolutions=module._resolution_ramp(2, 1024),
        max_seconds_per_image=_BENCHMARK_TEST_MAX_SECONDS,
        z_plane=0.0,
        variable="Rho [g/cm^3]",
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert expected_path.exists()
    assert "[example] done -> benchmark_xy_plane_example_*" in log_text


def test_random_points_benchmark_runs_one_example_case() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_example_module("benchmark_random_points.py", "benchmark_random_points_run_for_test")
    module.resolve_data_file = _resolve_local_data_file
    out_root = _example_benchmark_output_root(repo_root)
    expected_path = out_root / "benchmark_random_points_example_timing_report.md"
    expected_path.unlink(missing_ok=True)
    progress, log_path = _example_benchmark_progress(module, out_root, "benchmark_random_points.log")

    module._run_case(
        case=module.DatasetCase("example", "3d__var_1_n00000000.plt"),
        case_index=0,
        repo_root=repo_root,
        out_root=out_root,
        progress=progress,
        query_counts=module._resolution_ramp(1024, 131072),
        max_seconds_per_query=_BENCHMARK_TEST_MAX_SECONDS,
        variable="Rho [g/cm^3]",
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert expected_path.exists()
    assert "[example] done -> benchmark_random_points_example_*" in log_text


@pytest.mark.pooch
def test_star_movie_benchmark_runs_one_example_case() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_example_module("benchmark_star_movie.py", "benchmark_star_movie_run_for_test")
    out_root = _example_benchmark_output_root(repo_root)
    frame_dir = out_root / "benchmark_star_movie_sc_frames"
    movie_path = out_root / "benchmark_star_movie_sc.mp4"
    report_path = out_root / "benchmark_star_movie_sc_timing_report.md"
    for path in frame_dir.glob("frame_*.png"):
        path.unlink()
    movie_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    progress, log_path = _example_benchmark_progress(module, out_root, "benchmark_star_movie.log")

    module._run_case(
        case=module.DatasetCase("sc", "3d__var_4_n00044000.plt"),
        repo_root=repo_root,
        out_root=out_root,
        progress=progress,
        n_frames=4,
        nx=64,
        ny=64,
        fps=8,
        distance_multiplier=2.0,
        view_width_multiplier=1.5,
        elevation_deg=15.0,
        variable="Rho [g/cm^3]",
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert (frame_dir / "frame_0000.png").exists()
    assert movie_path.exists()
    assert report_path.exists()
    assert "[sc] done -> benchmark_star_movie_sc_*" in log_text


@pytest.mark.pooch
def test_star_movie_view_angles_render_for_all_available_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_example_module("benchmark_star_movie.py", "benchmark_star_movie_angles_for_test")

    for case in module.MOVIE_CASES:
        data_path = module.resolve_data_file(repo_root, case.file_name)
        ds = module.Dataset.from_file(str(data_path))
        tree = module._build_octree(ds)
        interp = module.OctreeInterpolator(tree, np.asarray(ds["Rho [g/cm^3]"], dtype=float))
        tracer = module.OctreeRayTracer(tree)
        domain_radius = module._domain_radius(tree)

        for frame in range(96):
            azimuth_deg = module._frame_azimuth_deg(frame, 96)
            origin = module._orbit_origin(
                radius=2.0 * domain_radius,
                azimuth_deg=azimuth_deg,
                elevation_deg=module.DEFAULT_ELEVATION_DEG,
            )
            try:
                image, counts = module._render_frame(
                    tracer,
                    interp,
                    origin=origin,
                    target=np.zeros(3, dtype=float),
                    up=np.array([0.0, 0.0, 1.0], dtype=float),
                    nx=32,
                    ny=32,
                    width=1.5 * domain_radius,
                    height=1.5 * domain_radius,
                )
            except ValueError as exc:
                raise AssertionError(
                    f"star movie angle failed for {case.label} at azimuth_deg={azimuth_deg:.3f}"
                ) from exc
            assert image.shape == (32, 32), case.label
            assert counts.shape == (32, 32), case.label


@pytest.mark.parametrize(
    ("case_label", "file_name", "azimuth_deg"),
    [
        pytest.param("local_example", "3d__var_1_n00000000.plt", 1.875, id="local_example_1.875"),
        pytest.param("local_rpa", "3d__var_2_n00060005.plt", 5.625, id="local_rpa_5.625"),
        pytest.param(
            "sc",
            "3d__var_4_n00044000.plt",
            5.625,
            id="sc_5.625",
            marks=pytest.mark.pooch,
        ),
    ],
)
def test_star_movie_diagnostic_angles_render_successfully(
    case_label: str, file_name: str, azimuth_deg: float
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_example_module("benchmark_star_movie.py", "benchmark_star_movie_diagnostic_for_test")
    case = module.DatasetCase(case_label, file_name)
    if case_label.startswith("local_"):
        data_path = local_data_file(case.file_name)
    else:
        data_path = module.resolve_data_file(repo_root, case.file_name)
    ds = module.Dataset.from_file(str(data_path))
    tree = module._build_octree(ds)
    interp = module.OctreeInterpolator(tree, np.asarray(ds["Rho [g/cm^3]"], dtype=float))
    tracer = module.OctreeRayTracer(tree)
    domain_radius = module._domain_radius(tree)
    distance_multiplier, view_width_multiplier = module._case_camera_settings(case)
    origin = module._orbit_origin(
        radius=distance_multiplier * domain_radius,
        azimuth_deg=float(azimuth_deg),
        elevation_deg=module.DEFAULT_ELEVATION_DEG,
    )

    image, counts = module._render_frame(
        tracer,
        interp,
        origin=origin,
        target=np.zeros(3, dtype=float),
        up=np.array([0.0, 0.0, 1.0], dtype=float),
        nx=16,
        ny=16,
        width=view_width_multiplier * domain_radius,
        height=view_width_multiplier * domain_radius,
    )

    assert image.shape == (16, 16), case.label
    assert counts.shape == (16, 16), case.label
    assert np.count_nonzero(np.isfinite(image) & (image > 0.0)) > 0, case.label


def test_star_movie_local_example_first_frame_has_no_start_owner_drops() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_example_module("benchmark_star_movie.py", "benchmark_star_movie_local_example_regression_for_test")
    case = module.DatasetCase("local_example", "3d__var_1_n00000000.plt")
    data_path = local_data_file(case.file_name)
    ds = module.Dataset.from_file(str(data_path))
    tree = module._build_octree(ds)
    interp = module.OctreeInterpolator(tree, np.asarray(ds["Rho [g/cm^3]"], dtype=float))
    tracer = module.OctreeRayTracer(tree)
    domain_radius = module._domain_radius(tree)
    distance_multiplier, view_width_multiplier = module._case_camera_settings(case)
    azimuth_deg = module._frame_azimuth_deg(0, 96)
    origin = module._orbit_origin(
        radius=distance_multiplier * domain_radius,
        azimuth_deg=float(azimuth_deg),
        elevation_deg=module.DEFAULT_ELEVATION_DEG,
    )
    width = view_width_multiplier * domain_radius
    origins, directions = module.camera_rays(
        origin=origin,
        target=np.zeros(3, dtype=float),
        up=np.array([0.0, 0.0, 1.0], dtype=float),
        nx=256,
        ny=256,
        width=width,
        height=width,
        projection="parallel",
    )

    # This exact benchmark launch used to drop 977 rays with TRACE_START_NO_ROOT_OWNER.
    _cell_counts, _time_counts, _cell_buffer, _time_buffer, ray_status = tracer._trace_chunk_to_scratch(
        origins.reshape(-1, 3),
        directions.reshape(-1, 3),
        t_min=0.0,
        t_max=np.inf,
    )
    no_root_owner = int(np.count_nonzero(ray_status == spherical_crossing_trace.TRACE_START_NO_ROOT_OWNER))
    assert no_root_owner == 0

    image, counts = module._render_frame(
        tracer,
        interp,
        origin=origin,
        target=np.zeros(3, dtype=float),
        up=np.array([0.0, 0.0, 1.0], dtype=float),
        nx=256,
        ny=256,
        width=width,
        height=width,
    )

    assert image.shape == (256, 256)
    assert counts.shape == (256, 256)
    assert np.count_nonzero(np.isfinite(image) & (image > 0.0)) > 0


def test_grid_vs_ray_rejects_resolution_too_small_for_symlog_height_colors() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"

    result = subprocess.run(
        [
            sys.executable,
            "benchmark_grid_vs_ray.py",
            "--min-resolution",
            "2",
            "--max-resolution",
            "2",
        ],
        cwd=examples_dir,
        env=_env_without_pythonpath(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "min-resolution must be at least 4 for symlog height comparison colors" in result.stderr


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


def test_grid_vs_ray_sample_mask_indices_bounds_scatter_points() -> None:
    module = _load_benchmark_grid_vs_ray_module()
    mask = np.ones((4, 5), dtype=bool)

    iz, iy = module._sample_mask_indices(mask, 6)

    assert iz.size == 6
    assert iy.size == 6
    np.testing.assert_array_equal(iz, np.array([0, 0, 1, 2, 3, 3], dtype=np.int64))
    np.testing.assert_array_equal(iy, np.array([0, 3, 2, 1, 0, 4], dtype=np.int64))


def test_grid_vs_ray_height_color_norm_uses_symlog() -> None:
    module = _load_benchmark_grid_vs_ray_module()

    norm = module._height_color_norm(np.array([-1.0, -1.0e-4, 0.0, 1.0e-4, 10.0], dtype=float))

    assert isinstance(norm, module.SymLogNorm)
    assert norm.linthresh == 1.0e-3
    assert norm.vmin == -1.0
    assert norm.vmax == 10.0


def test_grid_vs_ray_height_color_norm_rejects_degenerate_height() -> None:
    module = _load_benchmark_grid_vs_ray_module()

    with pytest.raises(ValueError, match="non-degenerate"):
        module._height_color_norm(np.array([0.0, 0.0], dtype=float))


def test_grid_vs_ray_style_horizontal_colorbar_top_enables_top_ticks_and_minors() -> None:
    module = _load_benchmark_grid_vs_ray_module()
    fig = module.plt.figure()
    try:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        cax = fig.add_axes([0.1, 0.92, 0.8, 0.04])
        mappable = module.plt.cm.ScalarMappable(norm=module.Normalize(vmin=1.0, vmax=10.0), cmap="viridis")
        mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")

        module._style_horizontal_colorbar_top(cbar, "test label", labelsize=7, tick_labelsize=6)
        fig.canvas.draw()

        assert cbar.ax.xaxis.get_label_position() == "top"
        assert cbar.ax.xaxis.get_ticks_position() == "top"
        assert cbar.ax.xaxis.label.get_text() == "test label"
        assert len(cbar.ax.xaxis.get_minorticklines()) > 0
    finally:
        module.plt.close(fig)


def test_grid_vs_ray_style_horizontal_colorbar_top_uses_sparse_symlog_ticks() -> None:
    module = _load_benchmark_grid_vs_ray_module()
    fig = module.plt.figure()
    try:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        cax = fig.add_axes([0.1, 0.92, 0.8, 0.04])
        mappable = module.plt.cm.ScalarMappable(
            norm=module.SymLogNorm(linthresh=1.0e-3, vmin=-1.0, vmax=10.0),
            cmap="cividis",
        )
        mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")

        module._style_horizontal_colorbar_top(cbar, "height over surface (R-1)", labelsize=7, tick_labelsize=6)
        fig.canvas.draw()

        np.testing.assert_allclose(cbar.get_ticks(), np.array([-1.0, 0.0, 10.0], dtype=float))
        assert [tick.get_text() for tick in cbar.ax.xaxis.get_majorticklabels()] == ["-1", "0", "10"]
        assert all(tick.get_text() == "" for tick in cbar.ax.xaxis.get_minorticklabels())
    finally:
        module.plt.close(fig)


def test_grid_vs_ray_discrepancy_rows_bounds_category_rows() -> None:
    module = _load_benchmark_grid_vs_ray_module()
    img0 = np.ones((4, 4), dtype=float)
    img1 = np.full((4, 4), np.nan, dtype=float)
    pixel_y, pixel_z = np.meshgrid(np.arange(4, dtype=float), np.arange(4, dtype=float), indexing="xy")
    pixel_r = np.arange(16, dtype=float).reshape(4, 4)

    rows = module._discrepancy_rows(
        img0,
        img1,
        pixel_y=pixel_y,
        pixel_z=pixel_z,
        pixel_r=pixel_r,
        max_category_rows=3,
    )

    assert len(rows) == 3
    assert {row["kind"] for row in rows} == {"grid_pos_ray_nan"}
    assert [row["r"] for row in rows] == [15.0, 14.0, 13.0]

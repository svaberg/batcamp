from __future__ import annotations

from pathlib import Path

import numpy as np
from batread.dataset import Dataset

from batcamp import Octree
from batcamp import OctreeInterpolator
from batcamp import OctreeRayInterpolator
from resampling_compare import _equality_deviation
from resampling_compare import _grid_sum_image
from resampling_compare import _pixel_plane_coordinates
from resampling_compare import _ray_image
from resampling_compare import _ray_segment_counts
from resampling_compare import _ray_setup
from resampling_compare import _save_four_panel_figure


def test_resampling_compare_outputs_nonblank_images(tmp_path: Path) -> None:
    """Compare-script smoke test: small local sample should produce visible signal."""
    sample_path = Path(__file__).resolve().parents[1] / "sample_data" / "3d__var_2_n00006003.plt"
    ds = Dataset.from_file(str(sample_path))
    tree = Octree.from_dataset(ds)
    interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"], tree=tree)
    ray = OctreeRayInterpolator(interp)

    dmin, dmax = interp.tree.domain_bounds(coord="xyz")
    bounds = (
        float(dmin[0]),
        float(dmax[0]),
        float(dmin[1]),
        float(dmax[1]),
        float(dmin[2]),
        float(dmax[2]),
    )
    n_plane = 16
    nx_sum = 32
    chunk_size = 1024

    img0 = _grid_sum_image(interp, n_plane=n_plane, nx_sum=nx_sum, bounds=bounds)
    origins, direction, t_end = _ray_setup(n_plane=n_plane, bounds=bounds)
    img1 = _ray_image(
        ray,
        origins=origins,
        direction=direction,
        t_end=t_end,
        n_plane=n_plane,
        chunk_size=chunk_size,
    )

    finite0 = np.isfinite(img0)
    finite1 = np.isfinite(img1)
    positive0 = finite0 & (img0 > 0.0)
    positive1 = finite1 & (img1 > 0.0)

    assert np.any(finite0)
    assert np.any(finite1)
    assert np.any(positive0)
    assert np.any(positive1)
    assert float(np.max(img0[positive0])) > float(np.min(img0[positive0]))
    assert float(np.max(img1[positive1])) > float(np.min(img1[positive1]))

    eq = _equality_deviation(img0, img1)
    ray_seg_counts = _ray_segment_counts(
        ray,
        origins=origins,
        direction=direction,
        t_end=t_end,
        n_plane=n_plane,
        chunk_size=chunk_size,
    )
    _pixel_y, _pixel_z, pixel_r = _pixel_plane_coordinates(n_plane=n_plane, bounds=bounds)

    out_path = tmp_path / "resampling_compare_16x16.png"
    _save_four_panel_figure(
        out_path,
        dataset_label=sample_path.name,
        n_plane=n_plane,
        img0=img0,
        img1=img1,
        pixel_r=pixel_r,
        ray_segment_counts=ray_seg_counts,
        grid_segment_count=nx_sum - 1,
        time0=0.0,
        time1=0.0,
        nx_sum=nx_sum,
        eq_abs_l1=float(eq[2]),
        eq_abs_rmse=float(eq[3]),
        eq_log_l1=float(eq[4]),
        eq_log_rmse=float(eq[5]),
        eq_pos_overlap=int(eq[1]),
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0

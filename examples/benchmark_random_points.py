#!/usr/bin/env python3
"""Random-point resampling benchmark.

This script builds one octree interpolator and compares its random-point
resampling runtime and output against the SciPy ND interpolators on the same
datasets. It writes plots and timing reports under `artifacts/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from batread.dataset import Dataset
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import Delaunay

from batcamp import OctreeInterpolator
from batcamp.constants import XYZ_VARS
from benchmark_helpers import _configure_progress_logging
from benchmark_helpers import _configure_builder_logging
from benchmark_helpers import _build_octree
from benchmark_helpers import _ProgressReporter
from benchmark_helpers import _resolution_ramp
from benchmark_helpers import _time_call
from benchmark_helpers import DatasetCase
from benchmark_helpers import resolve_data_file

_RNG_SEED = 0
_MAX_LINEAR_REFERENCE_POINTS = 20_000


def _linear_reference_note(linear_reference_mode: str, *, n_points_total: int) -> str | None:
    """Return one explicit note when SciPy linear is not part of the comparison."""
    if linear_reference_mode.startswith("skipped"):
        return (
            "SciPy linear not run: apples-to-apples comparison requires the full dataset, "
            f"and this case has {int(n_points_total)} points "
            f"(full-data LinearND is limited to {int(_MAX_LINEAR_REFERENCE_POINTS)} points)."
        )
    return None


def _xyz_points(ds: Dataset) -> np.ndarray:
    """Return dataset point coordinates as one dense `(n_points, 3)` array."""
    return np.column_stack(
        [
            np.asarray(ds[XYZ_VARS[0]], dtype=float),
            np.asarray(ds[XYZ_VARS[1]], dtype=float),
            np.asarray(ds[XYZ_VARS[2]], dtype=float),
        ]
    )


def _random_queries(
    interp: OctreeInterpolator,
    xyz: np.ndarray,
    *,
    n_query: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Cartesian query points uniformly from the dataset xyz bounding box."""
    lo = np.min(np.asarray(xyz, dtype=float), axis=0)
    hi = np.max(np.asarray(xyz, dtype=float), axis=0)
    span = hi - lo
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(span <= 0.0):
        raise ValueError("Invalid xyz domain bounds for random sampling.")

    out = np.empty((int(n_query), 3), dtype=float)
    filled = 0
    while filled < int(n_query):
        remaining = int(n_query) - filled
        batch = max(4096, 2 * remaining)
        candidates = lo[None, :] + rng.random((batch, 3), dtype=float) * span[None, :]
        _vals, cell_ids = interp(
            candidates,
            query_coord="xyz",
            return_cell_ids=True,
            log_outside_domain=False,
        )
        keep = np.asarray(cell_ids, dtype=np.int64) >= 0
        accepted = candidates[keep]
        n_take = min(remaining, int(accepted.shape[0]))
        if n_take > 0:
            out[filled : filled + n_take] = accepted[:n_take]
            filled += n_take
    return out


def _full_linear_reference(
    xyz: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Return the full SciPy-linear reference point set, or skip large datasets."""
    n_points = int(xyz.shape[0])
    if n_points <= _MAX_LINEAR_REFERENCE_POINTS:
        return np.asarray(xyz, dtype=float), np.asarray(values, dtype=float), f"full ({n_points})"
    return (
        None,
        None,
        f"skipped ({n_points} > {_MAX_LINEAR_REFERENCE_POINTS}; full-data only)",
    )


def _comparison_metrics(reference_vals: np.ndarray, candidate_vals: np.ndarray) -> dict[str, float | int]:
    """Return overlap and error metrics relative to one reference result."""
    reference_finite = np.isfinite(reference_vals)
    candidate_finite = np.isfinite(candidate_vals)
    overlap = reference_finite & candidate_finite

    out: dict[str, float | int] = {
        "reference_finite": int(np.count_nonzero(reference_finite)),
        "candidate_finite": int(np.count_nonzero(candidate_finite)),
        "finite_overlap": int(np.count_nonzero(overlap)),
        "positive_overlap": 0,
        "abs_mae": np.nan,
        "abs_rmse": np.nan,
        "log10_mae": np.nan,
        "log10_rmse": np.nan,
    }
    if not np.any(overlap):
        return out

    diff = np.asarray(candidate_vals[overlap] - reference_vals[overlap], dtype=float)
    out["abs_mae"] = float(np.mean(np.abs(diff)))
    out["abs_rmse"] = float(np.sqrt(np.mean(diff * diff)))

    positive = overlap & (reference_vals > 0.0) & (candidate_vals > 0.0)
    out["positive_overlap"] = int(np.count_nonzero(positive))
    if not np.any(positive):
        return out

    log_diff = np.log10(candidate_vals[positive]) - np.log10(reference_vals[positive])
    out["log10_mae"] = float(np.mean(np.abs(log_diff)))
    out["log10_rmse"] = float(np.sqrt(np.mean(log_diff * log_diff)))
    return out


def _fastest_label(
    *,
    octree_query_s: float,
    nearest_query_s: float,
    linear_query_s: float | None = None,
) -> str:
    """Return the fastest method label for one query count."""
    timings = [
        ("octree", float(octree_query_s)),
        ("nearest", float(nearest_query_s)),
    ]
    if linear_query_s is not None:
        timings.append(("linear", float(linear_query_s)))
    return min(timings, key=lambda item: item[1])[0]


def _build_summary(
    *,
    octree_tree_s: float,
    octree_interp_s: float,
    nearest_build_s: float,
    linear_build_total: float | None,
) -> str:
    """Return one short build summary line for the log."""
    octree_total = float(octree_tree_s + octree_interp_s)
    if linear_build_total is None:
        return f"octree total={octree_total:.2f}s nearest={nearest_build_s:.2f}s linear not run"
    return (
        f"octree total={octree_total:.2f}s "
        f"linear total={linear_build_total:.2f}s "
        f"nearest={nearest_build_s:.2f}s"
    )


def _query_summary(
    *,
    query_count: int,
    octree_query_s: float,
    nearest_query_s: float,
    linear_query_s: float | None = None,
    linear_finite: int | None = None,
) -> str:
    """Return one readable per-query timing summary."""
    fastest = _fastest_label(
        octree_query_s=octree_query_s,
        nearest_query_s=nearest_query_s,
        linear_query_s=linear_query_s,
    )
    parts = [f"n={int(query_count)}", f"octree={octree_query_s:.4f}s"]
    if linear_query_s is not None:
        parts.append(f"linear={linear_query_s:.4f}s")
    parts.append(f"nearest={nearest_query_s:.4f}s")
    parts.append(f"fastest={fastest}")
    if linear_finite is not None:
        parts.append(f"linear_finite={int(linear_finite)}/{int(query_count)}")
    return " ".join(parts)


def _write_report(
    rows: list[dict[str, float | int]],
    out_path: Path,
    *,
    dataset_label: str,
    file_name: str,
    variable: str,
    tree_coord: str,
    octree_tree_s: float,
    octree_interp_s: float,
    linear_enabled: bool,
    linear_reference_mode: str,
    linear_reference_note: str | None,
    linear_delaunay_s: float | None,
    linear_interp_s: float | None,
    nearest_build_s: float,
    cold_query_count: int,
    octree_cold_s: float,
    linear_cold_s: float | None,
    nearest_cold_s: float,
) -> None:
    """Write one readable markdown summary with build and query timings."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    octree_build_total = octree_tree_s + octree_interp_s
    linear_build_total = None if linear_delaunay_s is None or linear_interp_s is None else linear_delaunay_s + linear_interp_s

    lines = [
        f"# {dataset_label}",
        "",
        f"- file: `{file_name}`",
        f"- variable: `{variable}`",
        f"- tree_coord: `{tree_coord}`",
        (
            "- methods: octree, scipy linear, scipy nearest"
            if linear_enabled
            else "- methods: octree, scipy nearest"
        ),
        (
            f"- note: {linear_reference_note}"
            if linear_reference_note is not None
            else "- note: all plotted methods use the full dataset"
        ),
        "",
        "## Build",
        "",
        "| method | seconds |",
        "|---|---:|",
        f"| octree tree | {octree_tree_s:.6f} |",
        f"| octree interpolator | {octree_interp_s:.6f} |",
        f"| octree total | {octree_build_total:.6f} |",
        (
            f"| scipy linear Delaunay | {linear_delaunay_s:.6f} |"
            if linear_delaunay_s is not None
            else "| scipy linear Delaunay | not run |"
        ),
        (
            f"| scipy linear total | {linear_build_total:.6f} |"
            if linear_build_total is not None
            else "| scipy linear total | not run |"
        ),
        f"| scipy nearest | {nearest_build_s:.6f} |",
        "",
    ]
    if linear_enabled:
        lines.extend(
            [
                "## Cold Start",
                "",
                f"- first query batch after build: `{int(cold_query_count)}` random points",
                "",
                "| method | seconds |",
                "|---|---:|",
                f"| octree | {octree_cold_s:.6f} |",
                f"| scipy linear | {float(linear_cold_s):.6f} |",
                f"| scipy nearest | {nearest_cold_s:.6f} |",
                "",
                "## Steady-State Query Runtime",
                "",
                "| n_query | octree_s | linear_s | nearest_s | fastest | linear_finite |",
                "|---:|---:|---:|---:|---|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                f"{int(row['query_count'])} | "
                f"{float(row['octree_query_s']):.6f} | "
                f"{float(row['linear_query_s']):.6f} | "
                f"{float(row['nearest_query_s']):.6f} | "
                f"{_fastest_label(octree_query_s=float(row['octree_query_s']), linear_query_s=float(row['linear_query_s']), nearest_query_s=float(row['nearest_query_s']))} | "
                f"{int(row['linear_finite'])}/{int(row['query_count'])} |"
            )
    else:
        lines.extend(
            [
                "## Cold Start",
                "",
                f"- first query batch after build: `{int(cold_query_count)}` random points",
                "",
                "| method | seconds |",
                "|---|---:|",
                f"| octree | {octree_cold_s:.6f} |",
                f"| scipy nearest | {nearest_cold_s:.6f} |",
                "",
                "## Steady-State Query Runtime",
                "",
                "| n_query | octree_s | nearest_s | fastest |",
                "|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                f"{int(row['query_count'])} | "
                f"{float(row['octree_query_s']):.6f} | "
                f"{float(row['nearest_query_s']):.6f} | "
                f"{_fastest_label(octree_query_s=float(row['octree_query_s']), nearest_query_s=float(row['nearest_query_s']))} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_runtime_plot(
    rows: list[dict[str, float | int]],
    out_path: Path,
    *,
    dataset_label: str,
    linear_label: str | None,
    cold_query_count: int,
    octree_cold_s: float,
    linear_cold_s: float | None,
    nearest_cold_s: float,
) -> None:
    """Save one runtime figure with cold-start and steady-state query timings."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    query_counts = np.asarray([int(row["query_count"]) for row in rows], dtype=float)
    octree_query = np.asarray([float(row["octree_query_s"]) for row in rows], dtype=float)
    nearest_query = np.asarray([float(row["nearest_query_s"]) for row in rows], dtype=float)
    fig, (ax_cold, ax_steady) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
        gridspec_kw={"width_ratios": [1.0, 2.2]},
        constrained_layout=True,
    )

    cold_labels = ["octree"]
    cold_values = [float(octree_cold_s)]
    cold_colors = ["C0"]
    if linear_label is not None:
        cold_labels.append(linear_label)
        cold_values.append(float(linear_cold_s))
        cold_colors.append("C1")
        linear_query = np.asarray([float(row["linear_query_s"]) for row in rows], dtype=float)
    cold_labels.append("scipy nearest")
    cold_values.append(float(nearest_cold_s))
    cold_colors.append("C2")

    cold_x = np.arange(len(cold_labels), dtype=float)
    ax_cold.bar(cold_x, np.asarray(cold_values, dtype=float), color=cold_colors)
    ax_cold.set_xticks(cold_x)
    ax_cold.set_xticklabels(cold_labels, rotation=15, ha="right")
    ax_cold.set_yscale("log")
    ax_cold.set_ylabel("Runtime [s]")
    ax_cold.set_title(f"Cold start ({cold_query_count} queries)")
    ax_cold.grid(True, axis="y", which="both", alpha=0.25)

    ax_steady.plot(query_counts, octree_query, "o-", label="octree")
    if linear_label is not None:
        ax_steady.plot(query_counts, linear_query, "o-", label=linear_label)
    ax_steady.plot(query_counts, nearest_query, "o-", label="scipy nearest")
    ax_steady.set_xscale("log")
    ax_steady.set_yscale("log")
    ax_steady.set_xlabel("Random query count")
    ax_steady.set_ylabel("Runtime [s]")
    ax_steady.set_title("Steady state")
    ax_steady.grid(True, which="both", alpha=0.25)
    ax_steady.legend()

    fig.suptitle(f"{dataset_label}: random-point resampling runtime")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_accuracy_plot(
    rows: list[dict[str, float | int]],
    out_path: Path,
    *,
    dataset_label: str,
    linear_label: str | None,
) -> None:
    """Save pairwise error and overlap curves for the active methods."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    query_counts = np.asarray([int(row["query_count"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.3))
    pair_specs = [
        ("octree vs nearest", "octree_vs_nearest", "C2"),
    ]
    if linear_label is not None:
        pair_specs = [
            (f"octree vs {linear_label}", "octree_vs_linear", "C0"),
            (f"nearest vs {linear_label}", "nearest_vs_linear", "C1"),
            ("octree vs nearest", "octree_vs_nearest", "C2"),
        ]

    for label, prefix, color in pair_specs:
        abs_mae = np.asarray([float(row[f"{prefix}_abs_mae"]) for row in rows], dtype=float)
        abs_rmse = np.asarray([float(row[f"{prefix}_abs_rmse"]) for row in rows], dtype=float)
        log_mae = np.asarray([float(row[f"{prefix}_log10_mae"]) for row in rows], dtype=float)
        log_rmse = np.asarray([float(row[f"{prefix}_log10_rmse"]) for row in rows], dtype=float)
        finite_overlap = np.asarray([int(row[f"{prefix}_finite_overlap"]) for row in rows], dtype=float)
        positive_overlap = np.asarray([int(row[f"{prefix}_positive_overlap"]) for row in rows], dtype=float)

        axes[0].plot(query_counts, abs_mae, "o-", color=color, label=f"{label} MAE")
        axes[0].plot(query_counts, abs_rmse, "o--", color=color, label=f"{label} RMSE")
        axes[1].plot(query_counts, log_mae, "o-", color=color, label=f"{label} MAE")
        axes[1].plot(query_counts, log_rmse, "o--", color=color, label=f"{label} RMSE")
        axes[2].plot(query_counts, finite_overlap, "o-", color=color, label=f"{label} finite")
        axes[2].plot(query_counts, positive_overlap, "o--", color=color, label=f"{label} positive")

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Random query count")
    axes[0].set_ylabel("Absolute error")
    axes[0].set_title("Absolute error")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Random query count")
    axes[1].set_ylabel("log10 error")
    axes[1].set_title("Log error")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    axes[2].set_xscale("log")
    axes[2].set_xlabel("Random query count")
    axes[2].set_ylabel("Overlap count")
    axes[2].set_title("Overlap counts")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend()

    fig.suptitle(f"{dataset_label}: pairwise method comparison")
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parity_axis(
    ax: plt.Axes,
    *,
    reference_vals: np.ndarray,
    candidate_vals: np.ndarray,
    title: str,
    ylabel: str,
    xlabel: str,
) -> None:
    """Draw one log-log parity scatter for positive finite overlap."""
    positive = np.isfinite(reference_vals) & np.isfinite(candidate_vals) & (reference_vals > 0.0) & (candidate_vals > 0.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not np.any(positive):
        ax.text(0.5, 0.5, "No positive finite overlap", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, which="both", alpha=0.25)
        return

    x = np.asarray(reference_vals[positive], dtype=float)
    y = np.asarray(candidate_vals[positive], dtype=float)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    ax.scatter(x, y, s=4, alpha=0.35, rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)


def _save_parity_scatter(
    octree_vals: np.ndarray,
    nearest_vals: np.ndarray,
    out_path: Path,
    *,
    dataset_label: str,
    query_count: int,
    linear_vals: np.ndarray | None,
    linear_label: str | None,
) -> None:
    """Save parity scatters for all active method pairings."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_specs: list[tuple[np.ndarray, np.ndarray, str, str, str]] = [
        (
            octree_vals,
            nearest_vals,
            "scipy nearest vs octree",
            "SciPy nearest",
            "Octree",
        )
    ]
    if linear_vals is not None and linear_label is not None:
        panel_specs = [
            (
                linear_vals,
                octree_vals,
                f"octree vs {linear_label}",
                "Octree",
                linear_label,
            ),
            (
                linear_vals,
                nearest_vals,
                f"scipy nearest vs {linear_label}",
                "SciPy nearest",
                linear_label,
            ),
            (
                octree_vals,
                nearest_vals,
                "scipy nearest vs octree",
                "SciPy nearest",
                "Octree",
            ),
        ]

    fig, axes = plt.subplots(1, len(panel_specs), figsize=(6.0 * len(panel_specs), 5.7))
    if len(panel_specs) == 1:
        axes = [axes]
    for ax, (reference_vals, candidate_vals, title, ylabel, xlabel) in zip(axes, panel_specs, strict=True):
        _parity_axis(
            ax,
            reference_vals=reference_vals,
            candidate_vals=candidate_vals,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
        )
    fig.suptitle(f"{dataset_label}: parity at {query_count} random queries")
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.93))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare octree random-point resampling against SciPy linear and nearest ND interpolation."
    )
    parser.add_argument(
        "--min-query-count",
        type=int,
        default=1024,
        help="Smallest random query count (default: 1024).",
    )
    parser.add_argument(
        "--max-query-count",
        type=int,
        default=131072,
        help="Largest random query count (default: 131072).",
    )
    parser.add_argument(
        "--max-seconds-per-query",
        type=float,
        default=0.5,
        help="Stop increasing query count once any method exceeds this query time.",
    )
    parser.add_argument(
        "--variable",
        default="Rho [g/cm^3]",
        help="Dataset variable to resample.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/resample_random_points",
        help="Output directory for timing reports and plots.",
    )
    args = parser.parse_args()

    query_counts = _resolution_ramp(int(args.min_query_count), int(args.max_query_count))
    max_seconds_per_query = float(args.max_seconds_per_query)
    if max_seconds_per_query <= 0.0:
        raise ValueError("max_seconds_per_query must be positive.")

    repo_root = Path(__file__).resolve().parent
    out_root = (repo_root / args.output_dir).resolve()
    progress_log_path = out_root / "progress.log"
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_log_path.write_text("", encoding="utf-8")
    _configure_progress_logging(log_path=progress_log_path)
    _configure_builder_logging(log_path=progress_log_path)
    progress = _ProgressReporter(log_path=progress_log_path)

    cases = [
        DatasetCase("example", "3d__var_1_n00000000.plt"),
        DatasetCase("ih", "3d__var_4_n00005000.plt"),
        DatasetCase("sc", "3d__var_4_n00044000.plt"),
    ]

    progress.note(f"output_dir={out_root}")
    progress.note(f"query_counts={query_counts}")

    for case_index, case in enumerate(cases):
        case_dir = out_root / case.label
        case_dir.mkdir(parents=True, exist_ok=True)

        progress.note(f"[{case.label}] file={case.file_name}")
        progress.note(f"[{case.label}] artifacts={case_dir}")
        progress.start(f"[{case.label}] resolve data file")
        data_path, resolve_s = _time_call(resolve_data_file, repo_root, case.file_name)
        progress.complete(f"[{case.label}] resolve data file", resolve_s, detail=f"-> {data_path}")
        progress.start(f"[{case.label}] read dataset")
        ds, read_s = _time_call(Dataset.from_file, str(data_path))
        progress.complete(f"[{case.label}] read dataset", read_s)

        xyz = _xyz_points(ds)
        values = np.asarray(ds[args.variable], dtype=float)
        linear_xyz, linear_values, linear_reference_mode = _full_linear_reference(xyz, values)
        linear_enabled = linear_xyz is not None and linear_values is not None
        linear_reference_note = _linear_reference_note(linear_reference_mode, n_points_total=int(xyz.shape[0]))
        linear_label = "scipy linear" if linear_enabled else None
        if linear_reference_note is None:
            progress.note(f"[{case.label}] scipy linear reference={linear_reference_mode}")
        else:
            progress.note(f"[{case.label}] {linear_reference_note}")

        progress.start(f"[{case.label}] prepare octree")
        tree, octree_tree_s = _time_call(_build_octree, ds)
        progress.complete(
            f"[{case.label}] prepare octree",
            octree_tree_s,
            detail=f"coord={tree.tree_coord}",
        )
        progress.start(f"[{case.label}] build octree interpolator")
        interp, octree_interp_s = _time_call(OctreeInterpolator, tree, np.asarray(ds[args.variable], dtype=float))
        progress.complete(f"[{case.label}] build octree interpolator", octree_interp_s)

        if linear_enabled:
            progress.start(f"[{case.label}] build scipy Delaunay")
            tri, linear_delaunay_s = _time_call(Delaunay, linear_xyz)
            progress.complete(f"[{case.label}] build scipy Delaunay", linear_delaunay_s, detail=linear_reference_mode)
            progress.start(f"[{case.label}] build scipy LinearND")
            linear_interp, linear_interp_s = _time_call(LinearNDInterpolator, tri, linear_values, fill_value=np.nan)
            progress.complete(f"[{case.label}] build scipy LinearND", linear_interp_s, detail=linear_reference_mode)
        else:
            linear_delaunay_s = None
            linear_interp_s = None
            linear_interp = None

        progress.start(f"[{case.label}] build scipy NearestND")
        nearest_interp, nearest_build_s = _time_call(NearestNDInterpolator, xyz, values)
        progress.complete(f"[{case.label}] build scipy NearestND", nearest_build_s)
        progress.note(
            f"[{case.label}] build summary: "
            f"{_build_summary(
                octree_tree_s=float(octree_tree_s),
                octree_interp_s=float(octree_interp_s),
                nearest_build_s=float(nearest_build_s),
                linear_build_total=None if linear_delaunay_s is None or linear_interp_s is None else float(linear_delaunay_s + linear_interp_s),
            )}"
        )

        rng = np.random.default_rng(_RNG_SEED + case_index)
        progress.start(f"[{case.label}] sample random queries")
        queries, query_sample_s = _time_call(
            _random_queries,
            interp,
            xyz,
            n_query=int(query_counts[-1]),
            rng=rng,
        )
        progress.complete(
            f"[{case.label}] sample random queries",
            query_sample_s,
            detail=f"max_n_query={int(query_counts[-1])}",
        )
        warm_count = min(int(query_counts[0]), 256)
        warm_q = np.asarray(queries[:warm_count], dtype=float)
        progress.start(f"[{case.label}] cold start query check")
        _, octree_warm_s = _time_call(
            interp,
            warm_q,
            query_coord="xyz",
            log_outside_domain=False,
        )
        linear_warm_s = 0.0
        if linear_interp is not None:
            _, linear_warm_s = _time_call(linear_interp, warm_q)
        _, nearest_warm_s = _time_call(nearest_interp, warm_q)
        if linear_interp is not None:
            cold_detail = (
                f"n_query={warm_count} octree={octree_warm_s:.4f}s "
                f"linear={linear_warm_s:.4f}s nearest={nearest_warm_s:.4f}s"
            )
        else:
            cold_detail = f"n_query={warm_count} octree={octree_warm_s:.4f}s nearest={nearest_warm_s:.4f}s"
        progress.complete(
            f"[{case.label}] cold start query check",
            float(octree_warm_s + linear_warm_s + nearest_warm_s),
            detail=cold_detail,
        )

        rows: list[dict[str, float | int]] = []
        largest_linear_vals: np.ndarray | None = None
        largest_octree_vals = np.array([], dtype=float)
        largest_nearest_vals = np.array([], dtype=float)
        largest_count = 0

        for n_query in query_counts:
            query_message = f"[{case.label}] query"
            t_step = time.perf_counter()
            q = np.asarray(queries[: int(n_query)], dtype=float)

            t0 = time.perf_counter()
            octree_vals = np.asarray(
                interp(q, query_coord="xyz", log_outside_domain=False),
                dtype=float,
            ).reshape(-1)
            octree_query_s = float(time.perf_counter() - t0)

            t0 = time.perf_counter()
            nearest_vals = np.asarray(nearest_interp(q), dtype=float).reshape(-1)
            nearest_query_s = float(time.perf_counter() - t0)
            oct_near_metrics = _comparison_metrics(octree_vals, nearest_vals)
            largest_octree_vals = octree_vals
            largest_nearest_vals = nearest_vals
            largest_count = int(n_query)

            if linear_interp is not None:
                t0 = time.perf_counter()
                linear_vals = np.asarray(linear_interp(q), dtype=float).reshape(-1)
                linear_query_s = float(time.perf_counter() - t0)
                oct_metrics = _comparison_metrics(linear_vals, octree_vals)
                near_metrics = _comparison_metrics(linear_vals, nearest_vals)
                row = {
                    "query_count": int(n_query),
                    "octree_query_s": octree_query_s,
                    "linear_query_s": linear_query_s,
                    "nearest_query_s": nearest_query_s,
                    "octree_finite": int(oct_metrics["candidate_finite"]),
                    "linear_finite": int(oct_metrics["reference_finite"]),
                    "nearest_finite": int(near_metrics["candidate_finite"]),
                    "octree_vs_linear_finite_overlap": int(oct_metrics["finite_overlap"]),
                    "octree_vs_linear_positive_overlap": int(oct_metrics["positive_overlap"]),
                    "octree_vs_linear_abs_mae": float(oct_metrics["abs_mae"]),
                    "octree_vs_linear_abs_rmse": float(oct_metrics["abs_rmse"]),
                    "octree_vs_linear_log10_mae": float(oct_metrics["log10_mae"]),
                    "octree_vs_linear_log10_rmse": float(oct_metrics["log10_rmse"]),
                    "nearest_vs_linear_finite_overlap": int(near_metrics["finite_overlap"]),
                    "nearest_vs_linear_positive_overlap": int(near_metrics["positive_overlap"]),
                    "nearest_vs_linear_abs_mae": float(near_metrics["abs_mae"]),
                    "nearest_vs_linear_abs_rmse": float(near_metrics["abs_rmse"]),
                    "nearest_vs_linear_log10_mae": float(near_metrics["log10_mae"]),
                    "nearest_vs_linear_log10_rmse": float(near_metrics["log10_rmse"]),
                    "octree_vs_nearest_finite_overlap": int(oct_near_metrics["finite_overlap"]),
                    "octree_vs_nearest_positive_overlap": int(oct_near_metrics["positive_overlap"]),
                    "octree_vs_nearest_abs_mae": float(oct_near_metrics["abs_mae"]),
                    "octree_vs_nearest_abs_rmse": float(oct_near_metrics["abs_rmse"]),
                    "octree_vs_nearest_log10_mae": float(oct_near_metrics["log10_mae"]),
                    "octree_vs_nearest_log10_rmse": float(oct_near_metrics["log10_rmse"]),
                }
                largest_linear_vals = linear_vals
                run_detail = _query_summary(
                    query_count=int(n_query),
                    octree_query_s=octree_query_s,
                    linear_query_s=linear_query_s,
                    nearest_query_s=nearest_query_s,
                    linear_finite=int(row["linear_finite"]),
                )
            else:
                linear_query_s = float("nan")
                row = {
                    "query_count": int(n_query),
                    "octree_query_s": octree_query_s,
                    "linear_query_s": linear_query_s,
                    "nearest_query_s": nearest_query_s,
                    "octree_finite": int(np.count_nonzero(np.isfinite(octree_vals))),
                    "linear_finite": 0,
                    "nearest_finite": int(np.count_nonzero(np.isfinite(nearest_vals))),
                    "octree_vs_linear_finite_overlap": 0,
                    "octree_vs_linear_positive_overlap": 0,
                    "octree_vs_linear_abs_mae": float("nan"),
                    "octree_vs_linear_abs_rmse": float("nan"),
                    "octree_vs_linear_log10_mae": float("nan"),
                    "octree_vs_linear_log10_rmse": float("nan"),
                    "nearest_vs_linear_finite_overlap": 0,
                    "nearest_vs_linear_positive_overlap": 0,
                    "nearest_vs_linear_abs_mae": float("nan"),
                    "nearest_vs_linear_abs_rmse": float("nan"),
                    "nearest_vs_linear_log10_mae": float("nan"),
                    "nearest_vs_linear_log10_rmse": float("nan"),
                    "octree_vs_nearest_finite_overlap": int(oct_near_metrics["finite_overlap"]),
                    "octree_vs_nearest_positive_overlap": int(oct_near_metrics["positive_overlap"]),
                    "octree_vs_nearest_abs_mae": float(oct_near_metrics["abs_mae"]),
                    "octree_vs_nearest_abs_rmse": float(oct_near_metrics["abs_rmse"]),
                    "octree_vs_nearest_log10_mae": float(oct_near_metrics["log10_mae"]),
                    "octree_vs_nearest_log10_rmse": float(oct_near_metrics["log10_rmse"]),
                }
                run_detail = (
                    _query_summary(
                        query_count=int(n_query),
                        octree_query_s=octree_query_s,
                        nearest_query_s=nearest_query_s,
                    )
                    + " linear not run"
                )
            rows.append(row)

            _write_report(
                rows,
                case_dir / "timing_report.md",
                dataset_label=case.label,
                file_name=case.file_name,
                variable=str(args.variable),
                tree_coord=str(tree.tree_coord),
                octree_tree_s=float(octree_tree_s),
                octree_interp_s=float(octree_interp_s),
                linear_enabled=linear_enabled,
                linear_reference_mode=linear_reference_mode,
                linear_reference_note=linear_reference_note,
                linear_delaunay_s=None if linear_delaunay_s is None else float(linear_delaunay_s),
                linear_interp_s=None if linear_interp_s is None else float(linear_interp_s),
                nearest_build_s=float(nearest_build_s),
                cold_query_count=int(warm_count),
                octree_cold_s=float(octree_warm_s),
                linear_cold_s=None if linear_interp is None else float(linear_warm_s),
                nearest_cold_s=float(nearest_warm_s),
            )
            _save_runtime_plot(
                rows,
                case_dir / "runtime_vs_queries.png",
                dataset_label=case.label,
                linear_label=linear_label,
                cold_query_count=int(warm_count),
                octree_cold_s=float(octree_warm_s),
                linear_cold_s=None if linear_interp is None else float(linear_warm_s),
                nearest_cold_s=float(nearest_warm_s),
            )
            _save_accuracy_plot(
                rows,
                case_dir / "accuracy_vs_queries.png",
                dataset_label=case.label,
                linear_label=linear_label,
            )
            _save_parity_scatter(
                largest_octree_vals,
                largest_nearest_vals,
                case_dir / "parity_scatter.png",
                dataset_label=case.label,
                query_count=largest_count,
                linear_vals=largest_linear_vals,
                linear_label=linear_label,
            )
            progress.note(f"{query_message} {n_query} ({float(time.perf_counter() - t_step):.2f}s) {run_detail}")

            active_query_times = [octree_query_s, nearest_query_s]
            if linear_interp is not None:
                active_query_times.append(linear_query_s)
            if max(active_query_times) > max_seconds_per_query:
                if linear_interp is not None:
                    detail = (
                        f"octree={octree_query_s:.2f}s linear={linear_query_s:.2f}s nearest={nearest_query_s:.2f}s "
                        f"> {max_seconds_per_query:.2f}s"
                    )
                else:
                    detail = (
                        f"octree={octree_query_s:.2f}s nearest={nearest_query_s:.2f}s "
                        f"> {max_seconds_per_query:.2f}s"
                    )
                progress.note(f"[{case.label}] stop at n_query={n_query}: {detail}")
                break

        progress.note(f"[{case.label}] done -> {case_dir}")


if __name__ == "__main__":
    main()

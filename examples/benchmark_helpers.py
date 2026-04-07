#!/usr/bin/env python3
"""Shared helpers for the benchmark scripts in `examples/`."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import tarfile
import time

import pooch
from batread.dataset import Dataset

from batcamp import Octree


_G2211_URL = "https://zenodo.org/records/7110555/files/run-Sun-G2211.tar.gz"
_G2211_SHA256 = "c31a32aab08cc20d5b643bba734fd7220e6b369e691f55f88a3a08cc5b2a2136"


@dataclass(frozen=True)
class DatasetCase:
    label: str
    file_name: str


class _ProgressReporter:
    """Simple progress logger for script stages."""

    def __init__(self, *, log_path: Path | None = None) -> None:
        self._log_path = log_path
        self._logger = logging.getLogger("resample.progress")

    def note(self, message: str) -> None:
        """Write one ordinary progress line."""
        self._logger.info(message)

    def start(self, message: str) -> None:
        """Start one timed stage."""
        self._logger.info("%s...", message)

    def complete(self, message: str, seconds: float, *, detail: str | None = None) -> None:
        """Finish one timed stage."""
        line = f"{message} complete ({seconds:.2f}s)"
        if detail:
            line = f"{line} {detail}"
        self._logger.info(line)


def _configure_progress_logging(*, log_path: Path) -> None:
    """Route script progress logs to stdout and the per-run progress log."""
    progress_logger = logging.getLogger("resample.progress")
    for handler in list(progress_logger.handlers):
        progress_logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    progress_logger.addHandler(stream_handler)
    progress_logger.addHandler(file_handler)
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False


def _configure_builder_logging(*, log_path: Path) -> None:
    """Route batcamp build/materialize logs to stdout and the per-run progress log."""
    formatter = logging.Formatter("  [%(filename)s:%(funcName)s:%(lineno)d] %(message)s")
    for logger_name in ("batcamp.builder", "batcamp.octree"):
        logger_obj = logging.getLogger(logger_name)
        for handler in list(logger_obj.handlers):
            logger_obj.removeHandler(handler)
            handler.close()
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(stream_handler)
        logger_obj.addHandler(file_handler)
        logger_obj.setLevel(logging.INFO)
        logger_obj.propagate = False


def _unique_match(paths: list[Path], *, name: str) -> Path:
    """Return one matched path by name, otherwise raise."""
    if not paths:
        raise FileNotFoundError(name)
    if len(paths) > 1:
        raise FileNotFoundError(f"Expected unique match for {name}, found {len(paths)}: {paths}")
    return paths[0]


def _find_in_sample_data(root: Path, name: str) -> Path:
    """Find one file by basename under sample_data."""
    return _unique_match(sorted(root.rglob(name)), name=name)


def _fetch_from_g2211_archive(name: str) -> Path:
    """Fetch one named file from the Zenodo G2211 archive."""
    archive_path = Path(
        pooch.retrieve(
            url=_G2211_URL,
            known_hash=_G2211_SHA256,
            progressbar=False,
        )
    )
    with tarfile.open(archive_path, "r:gz") as tar:
        member_names = sorted(m.name for m in tar.getmembers() if m.isfile() and Path(m.name).name == name)
    member = _unique_match([Path(m) for m in member_names], name=name).as_posix()
    extracted = pooch.retrieve(
        url=_G2211_URL,
        known_hash=_G2211_SHA256,
        progressbar=False,
        processor=pooch.Untar(members=[member]),
    )
    if isinstance(extracted, (list, tuple)):
        extracted = extracted[0]
    return Path(extracted)


def resolve_data_file(repo_root: Path, name: str) -> Path:
    """Resolve data file from sample_data first, then pooch fallback."""
    try:
        return _find_in_sample_data(repo_root / "sample_data", name)
    except FileNotFoundError:
        return _fetch_from_g2211_archive(name)


def _build_octree(ds: Dataset) -> Octree:
    """Build one octree directly from one dataset."""
    return Octree.from_ds(ds)


def _resolution_ramp(min_resolution: int, max_resolution: int) -> list[int]:
    """Return the doubled resolution ramp `min, 2*min, ...` up to `max`."""
    if int(min_resolution) <= 0:
        raise ValueError("min_resolution must be positive.")
    if int(max_resolution) < int(min_resolution):
        raise ValueError("max_resolution must be >= min_resolution.")

    out: list[int] = []
    n = int(min_resolution)
    while n <= int(max_resolution):
        out.append(n)
        n *= 2
    return out


def _time_call(fn, /, *args, **kwargs):
    """Run one callable and return `(result, seconds)`."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, float(time.perf_counter() - t0)

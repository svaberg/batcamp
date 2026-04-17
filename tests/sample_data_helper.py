from __future__ import annotations

from pathlib import Path
import tarfile

import pooch


_G2211_URL = "https://zenodo.org/records/7110555/files/run-Sun-G2211.tar.gz"
_G2211_SHA256 = "c31a32aab08cc20d5b643bba734fd7220e6b369e691f55f88a3a08cc5b2a2136"


def _unique_match(paths: list[Path], *, name: str) -> Path:
    """Private test helper: return one match for `name` or raise."""
    if not paths:
        raise FileNotFoundError(name)
    if len(paths) > 1:
        raise FileNotFoundError(f"Expected unique match for {name}, found {len(paths)} entries: {paths}")
    return paths[0]


def _find_in_sample_data(root: Path, name: str) -> Path:
    """Private test helper: find one file by basename under `sample_data`."""
    return _unique_match(sorted(root.rglob(name)), name=name)


def _fetch_from_g2211_archive(name: str) -> Path:
    """Private test helper: fetch one named file from the Zenodo G2211 archive."""
    archive_path = Path(
        pooch.retrieve(
            url=_G2211_URL,
            known_hash=_G2211_SHA256,
            progressbar=False,
        )
    )
    with tarfile.open(archive_path, "r:gz") as tar:
        member_names = sorted(
            m.name for m in tar.getmembers() if m.isfile() and Path(m.name).name == name
        )
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


def local_data_file(name: str) -> Path:
    """Resolve one bundled sample file by basename from this repo only."""
    root = Path(__file__).resolve().parents[1]
    return _find_in_sample_data(root / "sample_data", name)


def data_file(name: str) -> Path:
    """Resolve one sample file by basename.

    Resolution order:
    1. Find uniquely in this repo's `sample_data` tree.
    2. Fallback to Zenodo G2211 archive via `pooch`, then extract that one member.
    """
    root = Path(__file__).resolve().parents[1]
    sample_data_root = root / "sample_data"
    try:
        return _find_in_sample_data(sample_data_root, name)
    except FileNotFoundError:
        return _fetch_from_g2211_archive(name)

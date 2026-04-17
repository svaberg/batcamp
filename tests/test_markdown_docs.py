from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pytest


_DOC_CASES = (
    pytest.param("README.md", ((512, 512, 2), (512, 512)), id="readme"),
    pytest.param("pypi.md", ((512, 512, 2), (512, 512)), id="pypi"),
)


def _markdown_python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def _no_show(*args, **kwargs) -> None:
    del args, kwargs


def _runnable_python_block(doc_name: str, block: str) -> str:
    """Replace doc placeholders with one concrete in-repo sample path for tests."""
    if doc_name == "pypi.md":
        return block.replace('Dataset.from_file("MY_FILE")', 'Dataset.from_file("sample_data/3d__var_1_n00000000.plt")')
    return block


@pytest.mark.parametrize(("doc_name", "expected_shapes"), _DOC_CASES)
def test_markdown_python_blocks_execute(
    doc_name: str,
    expected_shapes: tuple[tuple[int, ...], ...],
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    blocks = _markdown_python_blocks(repo_root / doc_name)
    output_root = repo_root / "artifacts" / "test_markdown_docs"

    assert len(blocks) == len(expected_shapes)

    monkeypatch.chdir(repo_root)
    plt.switch_backend("Agg")
    plt.close("all")
    output_root.mkdir(parents=True, exist_ok=True)

    doc_stem = Path(doc_name).stem
    for path in output_root.glob(f"{doc_stem}_block_*.png"):
        path.unlink()

    observed_shapes: list[tuple[int, ...]] = []
    saved_paths: list[Path] = []
    namespace = {"__name__": "__main__"}
    state = {"block_index": 0}

    def _save_show(*args, **kwargs) -> None:
        del args, kwargs
        figure_numbers = tuple(plt.get_fignums())
        for figure_order, figure_number in enumerate(figure_numbers, start=1):
            figure = plt.figure(figure_number)
            suffix = "" if len(figure_numbers) == 1 else f"_fig_{figure_order:02d}"
            out_path = output_root / f"{doc_stem}_block_{state['block_index']:02d}{suffix}.png"
            figure.savefig(out_path, dpi=150)
            saved_paths.append(out_path)
        plt.close("all")

    monkeypatch.setattr(plt, "show", _save_show)

    for block_index, block in enumerate(blocks, start=1):
        state["block_index"] = block_index
        before_ids = {name: id(namespace[name]) for name in ("rho_and_ti", "image") if name in namespace}
        runnable = _runnable_python_block(doc_name, block)
        exec(compile(runnable, f"{doc_name}:block:{block_index}", "exec"), namespace)
        if "rho_and_ti" in namespace:
            if before_ids.get("rho_and_ti") != id(namespace["rho_and_ti"]):
                observed_shapes.append(tuple(namespace["rho_and_ti"].shape))
        if "image" in namespace:
            if before_ids.get("image") != id(namespace["image"]):
                observed_shapes.append(tuple(namespace["image"].shape))

    assert tuple(observed_shapes) == expected_shapes
    assert len(saved_paths) == len(expected_shapes)
    assert all(path.exists() for path in saved_paths)

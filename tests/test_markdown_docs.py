from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pytest


_DOC_CASES = (
    pytest.param("README.md", ((512, 512, 2), (512, 512)), id="readme"),
    pytest.param("pypi.md", ((512, 512, 2),), id="pypi"),
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

    assert len(blocks) == len(expected_shapes)

    monkeypatch.chdir(repo_root)
    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", _no_show)

    observed_shapes: list[tuple[int, ...]] = []

    for block_index, block in enumerate(blocks, start=1):
        namespace = {"__name__": "__main__"}
        runnable = _runnable_python_block(doc_name, block)
        exec(compile(runnable, f"{doc_name}:block:{block_index}", "exec"), namespace)
        if "rho_and_ti" in namespace:
            observed_shapes.append(tuple(namespace["rho_and_ti"].shape))
        if "image" in namespace:
            observed_shapes.append(tuple(namespace["image"].shape))

    assert tuple(observed_shapes) == expected_shapes

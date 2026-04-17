from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt


def _readme_python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def _no_show(*args, **kwargs) -> None:
    del args, kwargs


def test_readme_python_blocks_execute(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    blocks = _readme_python_blocks(repo_root / "README.md")

    assert len(blocks) == 2

    monkeypatch.chdir(repo_root)
    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", _no_show)

    expected_shapes = [(512, 512, 2), (512, 512)]
    observed_shapes: list[tuple[int, ...]] = []

    for block_index, block in enumerate(blocks, start=1):
        namespace = {"__name__": "__main__"}
        exec(compile(block, f"README.md:block:{block_index}", "exec"), namespace)
        if "rho_and_ti" in namespace:
            observed_shapes.append(tuple(namespace["rho_and_ti"].shape))
        if "image" in namespace:
            observed_shapes.append(tuple(namespace["image"].shape))

    assert observed_shapes == expected_shapes

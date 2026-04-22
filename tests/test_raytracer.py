from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RAYTRACER_PATH = _REPO_ROOT / "batcamp" / "raytracer.py"


@dataclass(frozen=True)
class _FloatLiteral:
    lineno: int
    col_offset: int
    source: str


@dataclass(frozen=True)
class _ImportedName:
    lineno: int
    col_offset: int
    name: str
    source: str


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _float_literals(path: Path) -> list[_FloatLiteral]:
    source = _source(path)
    tree = ast.parse(source, filename=str(path))
    literals: list[_FloatLiteral] = []
    seen: set[tuple[int, int, str]] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, float):
            continue
        literal_source = ast.get_source_segment(source, node) or repr(node.value)
        key = (int(node.lineno), int(node.col_offset), literal_source)
        if key in seen:
            continue
        seen.add(key)
        literals.append(
            _FloatLiteral(
                lineno=int(node.lineno),
                col_offset=int(node.col_offset),
                source=literal_source,
            )
        )

    return sorted(literals, key=lambda item: (item.lineno, item.col_offset, item.source))


def _looks_like_constant_name(name: str) -> bool:
    stripped = name.lstrip("_")
    if not stripped:
        return False
    if stripped != stripped.upper():
        return False
    return any(char.isalpha() for char in stripped)


def _constant_like_imports(path: Path) -> list[_ImportedName]:
    source = _source(path)
    tree = ast.parse(source, filename=str(path))
    imports: list[_ImportedName] = []
    seen: set[tuple[int, int, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases = node.names
        elif isinstance(node, ast.ImportFrom):
            aliases = node.names
        else:
            continue

        for alias in aliases:
            bound_name = alias.asname or alias.name.split(".")[-1]
            if not _looks_like_constant_name(bound_name):
                continue
            source_text = ast.get_source_segment(source, node) or bound_name
            key = (int(node.lineno), int(node.col_offset), bound_name)
            if key in seen:
                continue
            seen.add(key)
            imports.append(
                _ImportedName(
                    lineno=int(node.lineno),
                    col_offset=int(node.col_offset),
                    name=bound_name,
                    source=source_text,
                )
            )

    return sorted(imports, key=lambda item: (item.lineno, item.col_offset, item.name))


def test_raytracer_module_contains_no_float_literals() -> None:
    """Guard against inline float literals in the raytracer module."""
    literals = _float_literals(_RAYTRACER_PATH)
    if not literals:
        return

    detail_lines = [
        f"batcamp/raytracer.py:{item.lineno}:{item.col_offset + 1}: {item.source}"
        for item in literals[:50]
    ]
    if len(literals) > 50:
        detail_lines.append(f"... and {len(literals) - 50} more")
    details = "\n".join(detail_lines)
    raise AssertionError(
        f"Found {len(literals)} float literals in batcamp/raytracer.py.\n"
        "Remove them or replace them with named non-inline definitions.\n"
        f"{details}"
    )


def test_raytracer_module_imports_no_constant_like_names() -> None:
    """Guard against importing names that look like constants into the raytracer module."""
    imports = _constant_like_imports(_RAYTRACER_PATH)
    if not imports:
        return

    detail_lines = [
        f"batcamp/raytracer.py:{item.lineno}:{item.col_offset + 1}: {item.name}: {item.source}"
        for item in imports
    ]
    details = "\n".join(detail_lines)
    raise AssertionError(
        f"Found {len(imports)} constant-like imports in batcamp/raytracer.py.\n"
        "Import modules or ordinary names instead of constant-like bindings.\n"
        f"{details}"
    )

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "batcamp"
_ALLOWED_PRIVATE_BASES = {"self", "cls"}
_PRIVATE_ATTR_FUNCS = {"getattr", "hasattr", "setattr", "delattr"}


@dataclass(frozen=True)
class _Violation:
    path: Path
    lineno: int
    kind: str
    name: str
    source: str


def _is_private_name(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _is_allowed_private_base(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in _ALLOWED_PRIVATE_BASES


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _violations_for_file(path: Path) -> list[_Violation]:
    source = _source(path)
    tree = ast.parse(source, filename=str(path))
    violations: list[_Violation] = []
    seen: set[tuple[int, str, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if not _is_private_name(node.attr):
                continue
            if _is_allowed_private_base(node.value):
                continue
            key = (node.lineno, "attribute", node.attr)
            if key in seen:
                continue
            seen.add(key)
            violations.append(
                _Violation(
                    path=path,
                    lineno=int(node.lineno),
                    kind="attribute",
                    name=node.attr,
                    source=ast.get_source_segment(source, node) or node.attr,
                )
            )
            continue

        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in _PRIVATE_ATTR_FUNCS:
            continue
        if len(node.args) < 2:
            continue
        target = node.args[0]
        name_arg = node.args[1]
        if _is_allowed_private_base(target):
            continue
        if not isinstance(name_arg, ast.Constant):
            continue
        if not isinstance(name_arg.value, str):
            continue
        if not _is_private_name(name_arg.value):
            continue
        key = (node.lineno, node.func.id, name_arg.value)
        if key in seen:
            continue
        seen.add(key)
        violations.append(
            _Violation(
                path=path,
                lineno=int(node.lineno),
                kind=node.func.id,
                name=name_arg.value,
                source=ast.get_source_segment(source, node) or name_arg.value,
            )
        )

    return sorted(violations, key=lambda item: (str(item.path), int(item.lineno), item.kind, item.name))


def _all_package_violations() -> list[_Violation]:
    violations: list[_Violation] = []
    for path in sorted(_PACKAGE_ROOT.glob("*.py")):
        violations.extend(_violations_for_file(path))
    return violations


def test_package_modules_do_not_use_foreign_private_names() -> None:
    """Package policy: modules must not reach into another object's private names."""
    violations = _all_package_violations()
    if not violations:
        return

    details = "\n".join(
        f"{item.path.relative_to(_REPO_ROOT)}:{item.lineno}: {item.kind} {item.name}: {item.source}"
        for item in violations
    )
    raise AssertionError(
        "Found foreign private-name access in package modules.\n"
        "Promote a truly shared dependency to a non-private name instead.\n"
        f"{details}"
    )

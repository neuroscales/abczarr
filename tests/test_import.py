"""Import smoke tests.

These guard against a broken core submodule taking down the whole package
at import time -- the failure mode that motivated this file, where a
renamed class in ``_core.path`` left ``abc.path`` (and therefore every
``import abczarr...``) raising, so even unrelated metadata/OME tests could
not be collected.

They install no optional backend, so they also assert that importing the
package never eagerly imports a driver (zarr-python, tensorstore, ...).
"""

from __future__ import annotations

import importlib

import pytest


def test_import_package() -> None:
    """The top-level package imports with no optional backend installed."""
    import abczarr  # noqa: F401


def test_public_names_resolve() -> None:
    """Every name advertised in ``__all__`` is actually reachable."""
    import abczarr

    for name in abczarr.__all__:
        assert hasattr(abczarr, name), name


@pytest.mark.parametrize(
    "module",
    [
        "abczarr.abc",
        "abczarr.abc.node",
        "abczarr.abc.array",
        "abczarr.abc.group",
        "abczarr.abc.store",
        "abczarr.abc.path",
        "abczarr.abc.errors",
        "abczarr.api",
        "abczarr.config",
        "abczarr.registry",
        "abczarr.metadata",
        "abczarr.ome",
        "abczarr.schemas",
    ],
)
def test_submodule_imports(module: str) -> None:
    """Each core submodule imports on its own."""
    importlib.import_module(module)


def test_store_path_classes_exist() -> None:
    """The store/path lattice that broke stays wired to ``_core.path``."""
    from abczarr.abc import path as spath

    for name in spath.__all__:
        assert hasattr(spath, name), name


# ---------------------------------------------------------------------------
# Python 3.8 floor: no modern hint notation in a position 3.8 evaluates.
#
# The package supports Python 3.8, but does not use
# ``from __future__ import annotations`` (its attrs machinery reads real
# annotation objects, not strings). So a PEP 604 union (``X | Y``) or a PEP
# 585 subscript (``list[int]``) in an *evaluated* annotation -- a function
# signature, or a module/class-level variable annotation -- raises on 3.8.
# This runs on every interpreter, so a regression is caught even off the 3.8
# CI leg. Use ``tx.Optional`` / ``tx.List`` / ... instead.
# ---------------------------------------------------------------------------

import ast  # noqa: E402
import pathlib  # noqa: E402

_PEP585 = {"list", "dict", "tuple", "set", "frozenset", "type"}
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "abczarr"


def _modern_hint_offenders(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if "from __future__ import annotations" in text:
        return []  # annotations are strings here; nothing is evaluated
    tree = ast.parse(text)
    hits: list[str] = []

    def scan(node: ast.AST, where: str) -> None:
        for n in ast.walk(node):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
                hits.append(f"{path.name}:{n.lineno} PEP604 union in {where}")
            if (
                isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name)
                and n.value.id in _PEP585
            ):
                hits.append(
                    f"{path.name}:{n.lineno} PEP585 "
                    f"{n.value.id}[...] in {where}"
                )

    depth = 0

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.AST) -> None:
            args = node.args
            for a in (
                list(args.args)
                + list(getattr(args, "posonlyargs", []))
                + list(args.kwonlyargs)
                + [args.vararg, args.kwarg]
            ):
                if a is not None and a.annotation is not None:
                    scan(a.annotation, "arg annotation")
            if node.returns is not None:
                scan(node.returns, "return annotation")
            nonlocal depth
            depth += 1
            self.generic_visit(node)
            depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if depth == 0:  # module/class level is evaluated; locals are not
                scan(node.annotation, "variable annotation")
            self.generic_visit(node)

    Visitor().visit(tree)
    return hits


def test_no_modern_hint_notation_on_evaluated_positions() -> None:
    offenders = []
    for path in _SRC.rglob("*.py"):
        offenders.extend(_modern_hint_offenders(path))
    assert not offenders, (
        "modern hint notation breaks the Python 3.8 floor; use tx.* aliases:\n"
        + "\n".join(offenders)
    )

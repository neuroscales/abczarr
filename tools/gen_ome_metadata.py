#!/usr/bin/env python3
"""Generate the OME-Zarr metadata version tree from the ``v0_1`` template.

The packages under ``src/abczarr/ome/metadata/`` restate almost the same
class surface once per NGFF version.  Adjacent versions are near-identical, so
keeping five hand-written copies in step is error-prone.  This tool keeps a
single hand-written source of truth -- ``v0_1`` -- and derives ``v0_2`` through
``v0_5`` from it by applying a small, explicit *forward delta table*
(``DELTAS`` below).

Why codegen rather than sharing classes at runtime:

* **Docs are static.**  mkdocstrings/griffe reads the class statements from
  disk, so every version has to stay a real committed ``.py`` file with real
  ``class`` statements.  A class synthesised at import time would be invisible
  to the API reference.
* **The conversion engine resolves classes by qualified name.**
  ``abczarr.ome.metadata.base`` converts between versions by finding the class
  with the same ``__qualname__`` in the sibling target-version package and
  reads the version from ``__module__``.  Each version therefore needs its own
  distinct class object, in its own module, with an identical dotted
  ``__qualname__``.
* **``register_subclass`` registers into every ``Metadata`` base in the MRO.**
  De-duplicating by having one version subclass another would pollute the
  discriminated-dispatch registry across versions.

So each version stays a separate, self-contained package of real classes; this
tool just writes four of them from the fifth.

``v0_6dev4`` is deliberately **out of scope**: it is a preview rewrite
(coordinate systems, a large transform model, scenes) rather than a clean
increment over ``v0_5``, so it stays hand-written and is never touched here.

Usage
-----
``python tools/gen_ome_metadata.py --check``
    Regenerate every derived version into a temporary directory and assert it
    is structurally equivalent to the committed tree (same classes, fields,
    requirement levels, ``register_subclass`` keys, module-level type aliases
    and ``__all__``).  Exits non-zero on any drift.  Whitespace, formatting,
    imports and docstrings are normalised away for this comparison.

``python tools/gen_ome_metadata.py --write``
    Overwrite the committed ``v0_2``..``v0_5`` packages with freshly generated
    files, then run ``ruff format`` and ``ruff check --fix`` over them.

This script uses ``ast`` + ``ast.unparse`` and therefore needs Python >= 3.9.
That is fine: it is a dev-only tool.  Its *outputs* are written in the same
Python-3.8-safe syntax as the ``v0_1`` template.
"""

# stdlib
import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# A module set: module name -> its parsed AST.
Modules = Dict[str, ast.Module]

# --------------------------------------------------------------------------
#   Where the tree lives
# --------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = _ROOT / "src" / "abczarr" / "ome" / "metadata"

#: NGFF versions this tool manages, oldest to newest.  ``v0_1`` is the
#: hand-written template; the rest are generated from it.  ``v0_6dev4`` is a
#: preview rewrite and is intentionally excluded from both generation and the
#: drift check.
TEMPLATE = "v0_1"
GENERATED = ["v0_2", "v0_3", "v0_4", "v0_5"]
VERSIONS = [TEMPLATE] + GENERATED

#: The NGFF version string each package declares.
VERSION_STRING = {
    "v0_1": "0.1",
    "v0_2": "0.2",
    "v0_3": "0.3",
    "v0_4": "0.4",
    "v0_5": "0.5",
}

#: The delta-editable modules present in every version.
_BASE_MODULES = ["images", "labels", "ome", "omero", "plates", "wells"]

_GENERATED_HEADER = (
    f"# Generated from {TEMPLATE} by tools/gen_ome_metadata.py"
    " -- do not edit\n"
)
_TEMPLATE_HEADER = (
    "# Hand-written source of truth. tools/gen_ome_metadata.py generates\n"
    "# v0_2..v0_5 from this package; edit here, then regenerate.\n"
)


# ==========================================================================
#
#                         THE FORWARD DELTA TABLE
#
# ==========================================================================
#
# Each op is a small callable-free record applied, in order, to the running
# module set to move it one version forward.  ``module`` names a delta-editable
# module; ``cls`` is the dotted class path within it (``("Multiscale",)`` for a
# top-level class, ``("ImageLabel", "Color")`` for a nested one).
#
# Every delta below was verified against a real ``diff`` of the committed
# adjacent version packages.


class SetAnn:
    """Replace the annotation of an existing field."""

    def __init__(
        self,
        module: str,
        cls: Tuple[str, ...],
        field: str,
        annotation: str,
    ) -> None:
        self.module, self.cls, self.field = module, cls, field
        self.annotation = annotation

    def apply(self, modules: Modules) -> None:
        node = _class_node(modules[self.module], self.cls)
        field = _field_node(node, self.field)
        field.annotation = _parse_expr(self.annotation)


class AddField:
    """Insert a new annotation-only field into a class body."""

    def __init__(
        self,
        module: str,
        cls: Tuple[str, ...],
        field: str,
        annotation: str,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> None:
        self.module, self.cls, self.field = module, cls, field
        self.annotation = annotation
        self.after, self.before = after, before

    def apply(self, modules: Modules) -> None:
        node = _class_node(modules[self.module], self.cls)
        new = ast.AnnAssign(
            target=ast.Name(id=self.field, ctx=ast.Store()),
            annotation=_parse_expr(self.annotation),
            value=None,
            simple=1,
        )
        index = self._insert_index(node)
        node.body.insert(index, new)

    def _insert_index(self, node: ast.ClassDef) -> int:
        if self.after is not None:
            return node.body.index(_field_node(node, self.after)) + 1
        if self.before is not None:
            return node.body.index(_field_node(node, self.before))
        return len(node.body)


class DelField:
    """Remove a field from a class body."""

    def __init__(
        self, module: str, cls: Tuple[str, ...], field: str
    ) -> None:
        self.module, self.cls, self.field = module, cls, field

    def apply(self, modules: Modules) -> None:
        node = _class_node(modules[self.module], self.cls)
        node.body.remove(_field_node(node, self.field))


class DelAssign:
    """Remove a module-level ``name = ...`` statement (a type alias)."""

    def __init__(self, module: str, name: str) -> None:
        self.module, self.name = module, name

    def apply(self, modules: Modules) -> None:
        body = modules[self.module].body
        body.remove(_assign_node(modules[self.module], self.name))


class AddImport:
    """Add a ``from ... import ...`` line just after the existing imports."""

    def __init__(self, module: str, source: str) -> None:
        self.module, self.source = module, source

    def apply(self, modules: Modules) -> None:
        body = modules[self.module].body
        stmt = ast.parse(self.source).body[0]
        last_import = 0
        for i, node in enumerate(body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last_import = i + 1
        body.insert(last_import, stmt)


class AddModule:
    """Introduce a brand-new module, authored with the template's version
    token so it is version-substituted like every other docstring."""

    def __init__(self, name: str, source: str) -> None:
        self.name, self.source = name, source

    def apply(self, modules: Modules) -> None:
        modules[self.name] = ast.parse(self.source)


# -- New modules introduced at v0.4 ----------------------------------------
# Authored with the ``v0_1`` cross-reference token (the canonical placeholder
# every managed module uses); the emitter substitutes it per output version.

_AXES_SOURCE = '''\
"""An axis of a multiscale pyramid: its name, type, and unit."""

__all__ = [
    "Axis", "SpaceAxis", "TimeAxis", "ChannelAxis",
    "AxisType", "SpaceUnit", "TimeUnit", "Unit",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import NotRecommended, Recommended, Required

# locals
from ..base import OMEMetadata

# typing
AxisType = tx.Literal["space", "time", "channel"]

SpaceUnit = tx.Literal[
    'angstrom', 'attometer', 'centimeter', 'decimeter', 'exameter',
    'femtometer', 'foot', 'gigameter', 'hectometer', 'inch', 'kilometer',
    'megameter', 'meter', 'micrometer', 'mile', 'millimeter', 'nanometer',
    'parsec', 'petameter', 'picometer', 'terameter', 'yard', 'yoctometer',
    'yottameter', 'zeptometer', 'zettameter'
]

TimeUnit = tx.Literal[
    'attosecond', 'centisecond', 'day', 'decisecond', 'exasecond',
    'femtosecond', 'gigasecond', 'hectosecond', 'hour', 'kilosecond',
    'megasecond', 'microsecond', 'millisecond', 'minute', 'nanosecond',
    'petasecond', 'picosecond', 'second', 'terasecond', 'yoctosecond',
    'yottasecond', 'zeptosecond', 'zettasecond'
]

Unit = tx.Union[SpaceUnit, TimeUnit]


@autodefine
class Axis(OMEMetadata):
    """One dimension of a
    [Multiscale][abczarr.ome.metadata.v0_1.images.Multiscale] pyramid.

    `name` is the axis's label, such as `"x"` or `"channel"`. Its
    position in a [Multiscale][abczarr.ome.metadata.v0_1.images.Multiscale]'s
    `axes` list is its position in every array shape and every
    coordinate transformation the pyramid carries. `type` says what
    kind of axis it is: `"space"`, `"time"`, or `"channel"`. `unit`
    is its physical unit. Constructing with `type="space"` gives back
    a [SpaceAxis][abczarr.ome.metadata.v0_1.axes.SpaceAxis], and
    likewise for `"time"` and `"channel"`, each restricting `unit` to
    the units that type allows.
    """

    name: Required[str] = field(factory=False)
    type: Recommended[tx.Union[AxisType, str]]
    unit: Recommended[tx.Union[Unit, str]]


@register_subclass(type="space")
class SpaceAxis(Axis):
    """A spatial axis (`x`, `y`, or `z`), with a length unit."""

    type: Recommended[tx.Literal["space"]]
    unit: Recommended[SpaceUnit]


@register_subclass(type="time")
class TimeAxis(Axis):
    """A time axis, with a duration unit."""

    type: Recommended[tx.Literal["time"]]
    unit: Recommended[TimeUnit]


@register_subclass(type="channel")
class ChannelAxis(Axis):
    """A channel axis. It carries no physical unit."""

    type: Recommended[tx.Literal["channel"]]
    unit: NotRecommended[Unit]
'''

_TRANSFORMATIONS_SOURCE = '''\
"""Coordinate transformations: how a resolution level maps to
physical space.
"""

__all__ = [
    "CoordinateTransformation",
    "Translation",
    "Scale",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import Required

# locals
from ..base import OMEMetadata


@autodefine
class CoordinateTransformation(OMEMetadata):
    """A transformation from array indices to physical coordinates.

    Build [Scale][abczarr.ome.metadata.v0_1.transformations.Scale] or
    [Translation][abczarr.ome.metadata.v0_1.transformations.Translation]
    directly rather than this base class. Constructing with
    `type="scale"` or `type="translation"` returns the matching one.
    """

    type: Required[str] = field(factory=False)


@register_subclass(type="translation")
@autodefine
class Translation(CoordinateTransformation):
    """An offset, one value per axis, in the axes' physical units."""

    type: Required[tx.Literal["translation"]]
    translation: Required[tx.List[float]]


@register_subclass(type="scale")
@autodefine
class Scale(CoordinateTransformation):
    """A per-axis scale factor from array indices to physical units.

    For a resolution level, this is the physical size of one array
    element along each axis. It's what turns a pixel index into a
    micrometer, and what makes coarser levels of a pyramid line up
    with the finest one.
    """

    type: Required[tx.Literal["scale"]]
    scale: Required[tx.List[float]]
'''


#: The forward delta table.  ``DELTAS[v]`` moves the running module set from
#: the previous version to ``v``.
DELTAS = {
    # 0.1 -> 0.2
    #   * OMEImageLabel.image_labels: List[ImageLabel] -> a single ImageLabel
    #   * OMEBioformats2Raw.plate: Optional -> Required
    "v0_2": [
        SetAnn("ome", ("OMEImageLabel",), "image_labels",
               "Required[ImageLabel]"),
        SetAnn("ome", ("OMEBioformats2Raw",), "plate", "Required[Plate]"),
    ],
    # 0.2 -> 0.3
    #   * Multiscale.axes added, a bare-string axis literal
    #   * version promoted Recommended -> Required in every carrier
    "v0_3": [
        AddField("images", ("Multiscale",), "axes",
                 "Required[tx.List[Axis]]", before="datasets"),
        SetAnn("images", ("Multiscale",), "version", "Required[Version]"),
        SetAnn("labels", ("ImageLabel",), "version", "Required[Version]"),
        SetAnn("omero", ("Omero",), "version", "Required[Version]"),
        SetAnn("plates", ("Plate",), "version", "Required[Version]"),
        SetAnn("wells", ("Well",), "version", "Required[Version]"),
    ],
    # 0.3 -> 0.4
    #   * new modules axes.py and transformations.py
    #   * Multiscale.axes becomes a typed List[Axis] (the module-level
    #     bare-string alias is dropped and Axis is imported instead)
    #   * per-dataset and per-multiscale coordinateTransformations added
    "v0_4": [
        AddModule("axes", _AXES_SOURCE),
        AddModule("transformations", _TRANSFORMATIONS_SOURCE),
        AddImport("images", "from .axes import Axis"),
        AddImport(
            "images",
            "from .transformations import "
            "CoordinateTransformation, Scale, Translation",
        ),
        DelAssign("images", "SpaceAxis"),
        DelAssign("images", "TimeAxis"),
        DelAssign("images", "ChannelAxis"),
        DelAssign("images", "Axis"),
        AddField(
            "images", ("Dataset",), "coordinateTransformations",
            "Required[tx.Union[tx.Tuple[Scale],"
            " tx.Tuple[Scale, Translation]]]",
            after="path",
        ),
        AddField(
            "images", ("Multiscale",), "coordinateTransformations",
            "Optional[tx.List[CoordinateTransformation]]",
            after="datasets",
        ),
    ],
    # 0.4 -> 0.5
    #   * the per-object `version` field is dropped everywhere it was a
    #     carrier (ome.OME keeps its own -- that is the discriminator).
    "v0_5": [
        DelField("images", ("Multiscale",), "version"),
        DelField("labels", ("ImageLabel",), "version"),
        DelField("omero", ("Omero",), "version"),
        DelField("plates", ("Plate",), "version"),
        DelField("wells", ("Well",), "version"),
    ],
}


# ==========================================================================
#
#                              AST HELPERS
#
# ==========================================================================


def _parse_expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _class_node(module: ast.Module, path: Tuple[str, ...]) -> ast.ClassDef:
    body: List[ast.stmt] = module.body
    node: Optional[ast.ClassDef] = None
    for name in path:
        node = next(
            (
                n
                for n in body
                if isinstance(n, ast.ClassDef) and n.name == name
            ),
            None,
        )
        if node is None:
            raise KeyError("no class {!r}".format(".".join(path)))
        body = node.body
    assert node is not None
    return node


def _field_node(class_node: ast.ClassDef, name: str) -> ast.AnnAssign:
    """The immediate ``name: ...`` field of a class (not one nested inside a
    sub-class of it)."""
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and _target_name(node) == name:
            return node
    raise KeyError(
        f"class {class_node.name!r} has no field {name!r}"
    )


def _assign_node(module: ast.Module, name: str) -> ast.Assign:
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node
    raise KeyError(f"module has no assignment {name!r}")


def _target_name(node: ast.AnnAssign) -> Optional[str]:
    return node.target.id if isinstance(node.target, ast.Name) else None


# ==========================================================================
#
#                              GENERATION
#
# ==========================================================================


def _template_modules() -> Modules:
    """Parse the delta-editable ``v0_1`` modules into ASTs."""
    modules: Modules = {}
    for name in _BASE_MODULES:
        source = (METADATA_DIR / TEMPLATE / (name + ".py")).read_text()
        modules[name] = ast.parse(source)
    return modules


def _build() -> Dict[str, Modules]:
    """Return ``{version: {module_name: ast.Module}}`` for every version by
    folding the forward deltas over the template."""
    modules = _template_modules()
    built = {TEMPLATE: _copy_modules(modules)}
    for version in GENERATED:
        for op in DELTAS[version]:
            op.apply(modules)
        built[version] = _copy_modules(modules)
    return built


def _copy_modules(modules: Modules) -> Modules:
    # Re-parse from the unparsed source: a cheap deep copy that also proves
    # each intermediate stays valid Python.
    return {
        name: ast.parse(ast.unparse(node)) for name, node in modules.items()
    }


def _substitute_version(source: str, version: str) -> str:
    """Rewrite the ``v0_1`` cross-reference token in docstrings to *version*.

    The token ``abczarr.ome.metadata.v0_1`` appears only inside docstrings
    (imports are relative), so a plain string replacement is unambiguous.
    """
    return source.replace(
        "abczarr.ome.metadata." + TEMPLATE,
        "abczarr.ome.metadata." + version,
    )


def _module_source(node: ast.Module, version: str, header: str) -> str:
    body = _substitute_version(ast.unparse(node), version)
    return header + "\n" + body + "\n"


def _version_source(version: str) -> str:
    ver = VERSION_STRING[version]
    return (
        '__all__ = ["Version", "VERSION"]\n\n'
        "import typing_extensions as tx\n\n"
        f'Version = tx.Literal["{ver}"]\n'
        f'VERSION = "{ver}"\n'
    )


def _init_source(module_names: List[str]) -> str:
    """A package ``__init__`` that re-exports every module, matching the
    hand-written style of the tree."""
    names = list(module_names)
    quoted = ", ".join(f'"{n}"' for n in names)
    lines = [f"__all__ = [{quoted}]", "", ""]
    lines.append("from . import (")
    for name in names:
        lines.append(f"    {name},")
    lines.append(")")
    for name in names:
        lines.append(f"from .{name} import *  # noqa: F403")
        lines.append(f"from .{name} import __all__ as __all_{name}")
    lines.append("")
    for name in names:
        lines.append(f"__all__ += __all_{name}")
    return "\n".join(lines) + "\n"


def _module_names(version_modules: Modules) -> List[str]:
    """Ordered module names for a version's package (``version`` last but one,
    matching the committed ordering: modules alphabetical, then version,
    wells)."""
    names = sorted(version_modules)
    ordered = [n for n in names if n not in ("version", "wells")]
    ordered.append("version")
    ordered.append("wells")
    return ordered


def _render_version(
    version: str, version_modules: Modules
) -> Dict[str, str]:
    """Return ``{filename: source}`` for one generated version package."""
    files: Dict[str, str] = {}
    for name, node in version_modules.items():
        files[name + ".py"] = _module_source(
            node, version, _GENERATED_HEADER
        )
    files["version.py"] = _GENERATED_HEADER + "\n" + _version_source(version)
    names = _module_names(version_modules)
    files["__init__.py"] = (
        _GENERATED_HEADER + "\n" + _init_source(names)
    )
    return files


# ==========================================================================
#
#                         STRUCTURAL SIGNATURE (--check)
#
# ==========================================================================


def _signature(source: str) -> Dict[str, Any]:
    """A formatting-, import- and docstring-independent fingerprint of a
    module: its ``__all__`` set, module-level type aliases, and every class's
    fields and ``register_subclass`` keys."""
    module = ast.parse(source)
    return {
        "all": _all_set(module),
        "aliases": _aliases(module),
        "classes": _classes(module, ()),
    }


def _all_set(module: ast.Module) -> frozenset:
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__"
            for t in node.targets
        ):
            if isinstance(node.value, (ast.List, ast.Tuple)):
                return frozenset(
                    e.value
                    for e in node.value.elts
                    if isinstance(e, ast.Constant)
                )
    return frozenset()


def _aliases(module: ast.Module) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and (
            isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name == "__all__":
                continue
            aliases[name] = ast.unparse(node.value)
    return aliases


def _classes(scope: Any, path: Tuple[str, ...]) -> Dict[tuple, dict]:
    out: Dict[tuple, dict] = {}
    for node in scope.body:
        if not isinstance(node, ast.ClassDef):
            continue
        qual = path + (node.name,)
        out[qual] = {
            "bases": tuple(ast.unparse(b) for b in node.bases),
            "register": _register_key(node),
            "fields": _fields(node),
        }
        out.update(_classes(node, qual))
    return out


def _fields(class_node: ast.ClassDef) -> tuple:
    fields = []
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            fields.append(
                (
                    node.target.id,
                    ast.unparse(node.annotation),
                    ast.unparse(node.value) if node.value else None,
                )
            )
    return tuple(fields)


def _register_key(class_node: ast.ClassDef) -> tuple:
    keys = []
    for deco in class_node.decorator_list:
        if (
            isinstance(deco, ast.Call)
            and isinstance(deco.func, ast.Name)
            and deco.func.id == "register_subclass"
        ):
            positional = tuple(ast.unparse(a) for a in deco.args)
            kw = tuple(
                sorted(
                    (k.arg, ast.unparse(k.value)) for k in deco.keywords
                )
            )
            keys.append((positional, kw))
    return tuple(keys)


# ==========================================================================
#
#                                DRIVERS
#
# ==========================================================================


def _committed_files(version: str) -> Dict[str, str]:
    directory = METADATA_DIR / version
    return {
        p.name: p.read_text()
        for p in directory.glob("*.py")
    }


def check() -> List[str]:
    """Regenerate every derived version and compare structurally to the
    committed tree.  Returns a list of human-readable drift messages (empty
    when the tree is in sync)."""
    built = _build()
    problems: List[str] = []
    for version in GENERATED:
        generated = _render_version(version, built[version])
        committed = _committed_files(version)

        gen_names = set(generated)
        com_names = set(committed)
        if gen_names != com_names:
            problems.append(
                f"{version}: file set differs: "
                f"only generated={sorted(gen_names - com_names)}, "
                f"only committed={sorted(com_names - gen_names)}"
            )

        for name in sorted(gen_names & com_names):
            if name == "__init__.py":
                _compare_init(version, generated[name], committed[name],
                              problems)
                continue
            gen_sig = _signature(generated[name])
            com_sig = _signature(committed[name])
            if gen_sig != com_sig:
                detail = _explain(gen_sig, com_sig)
                problems.append(
                    f"{version}/{name}: structural drift\n{detail}"
                )
    return problems


def _compare_init(
    version: str,
    generated: str,
    committed: str,
    problems: List[str],
) -> None:
    # __init__ is generated deterministically; only its re-export surface is
    # structural.  (The committed v0_4 __init__ carries a harmless duplicate
    # `__all__ += __all_plates` line the generator drops -- so compare the
    # submodule set and the `__all__` literal, not the statement list.)
    gen = ast.parse(generated)
    com = ast.parse(committed)
    if _all_set(gen) != _all_set(com):
        problems.append(
            f"{version}/__init__.py: __all__ module list differs"
        )
    if _imported_submodules(gen) != _imported_submodules(com):
        problems.append(
            f"{version}/__init__.py: re-exported submodule set differs"
        )


def _imported_submodules(module: ast.Module) -> frozenset:
    names = set()
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and (
            node.module is None
        ):
            names.update(a.name for a in node.names)
    return frozenset(names)


def _explain(gen: Dict[str, Any], com: Dict[str, Any]) -> str:
    lines: List[str] = []
    if gen["all"] != com["all"]:
        lines.append(
            "  __all__: +{} -{}".format(
                sorted(gen["all"] - com["all"]),
                sorted(com["all"] - gen["all"]),
            )
        )
    if gen["aliases"] != com["aliases"]:
        for key in sorted(set(gen["aliases"]) | set(com["aliases"])):
            if gen["aliases"].get(key) != com["aliases"].get(key):
                lines.append(
                    "  alias {}: {!r} != {!r}".format(
                        key, gen["aliases"].get(key), com["aliases"].get(key)
                    )
                )
    all_classes = set(gen["classes"]) | set(com["classes"])
    for qual in sorted(all_classes):
        g = gen["classes"].get(qual)
        c = com["classes"].get(qual)
        if g != c:
            lines.append("  class {}:".format(".".join(qual)))
            lines.append(f"    generated: {g}")
            lines.append(f"    committed: {c}")
    return "\n".join(lines)


def write() -> None:
    """Overwrite the committed derived packages with freshly generated files,
    then normalise them with ruff."""
    built = _build()
    written: List[Path] = []
    for version in GENERATED:
        directory = METADATA_DIR / version
        directory.mkdir(exist_ok=True)
        # remove any stale generated module (e.g. a module a delta no longer
        # produces) before writing the fresh set
        keep = set(_render_version(version, built[version]))
        for existing in directory.glob("*.py"):
            if existing.name not in keep:
                existing.unlink()
        for name, source in _render_version(version, built[version]).items():
            path = directory / name
            path.write_text(source)
            written.append(path)

    # mark the template package as the source of truth (header only)
    _ensure_template_header()

    _ruff(written)
    print(f"wrote {len(written)} files across {GENERATED}")


def _ensure_template_header() -> None:
    init = METADATA_DIR / TEMPLATE / "__init__.py"
    text = init.read_text()
    if "Hand-written source of truth" not in text:
        init.write_text(_TEMPLATE_HEADER + "\n" + text)


def _ruff(paths: List[Path]) -> None:
    files = [str(p) for p in paths]
    subprocess.run(
        ["ruff", "check", "--fix", "--quiet", *files], check=False
    )
    subprocess.run(["ruff", "format", "--quiet", *files], check=False)


def _tmp_write(built: Dict[str, Modules]) -> Path:
    """Render every generated version into a fresh temp dir (used only to
    smoke-test that outputs import; not part of --check)."""
    tmp = Path(tempfile.mkdtemp(prefix="ome_gen_"))
    for version in GENERATED:
        directory = tmp / version
        directory.mkdir()
        for name, source in _render_version(version, built[version]).items():
            (directory / name).write_text(source)
    return tmp


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="assert the committed tree matches what would be generated",
    )
    group.add_argument(
        "--write",
        action="store_true",
        help="overwrite the derived version packages",
    )
    args = parser.parse_args(argv)

    if sys.version_info < (3, 9):
        parser.error("this tool needs Python >= 3.9 (ast.unparse)")

    if args.check:
        problems = check()
        if problems:
            print("OME metadata tree is out of sync with the template:\n")
            print("\n\n".join(problems))
            return 1
        print(f"OME metadata tree is in sync ({len(GENERATED)} versions).")
        return 0

    write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

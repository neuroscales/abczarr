#!/usr/bin/env python3
"""Generate the OME-Zarr metadata version trees from hand-written templates.

The packages under ``src/abczarr/ome/metadata/`` restate almost the same
class surface once per NGFF version.  Adjacent versions are near-identical, so
keeping the copies in step by hand is error-prone.  This tool keeps one
hand-written source of truth per *chain* of versions and derives the rest from
it by applying a small, explicit *forward delta table*.

There are two chains:

* the **stable** chain -- ``v0_1`` (template) generating ``v0_2``..``v0_5``;
* the **0.6 pre-release** chain -- ``v0_6dev1`` (template) generating
  ``v0_6dev2``, ``v0_6dev3``, ``v0_6dev4`` and ``v0_6rc0``.

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
tool just writes the derived ones from the template.

Modules that are genuinely version-specific -- a rewritten
``transformations.py``, the ``ome.py`` that gains a scene carrier, the new
``scenes.py`` -- are not squeezed into field-level deltas.  They are supplied
as whole template source files under ``tools/ome_templates/`` and dropped in
with the ``AddModule`` op, which replaces (or introduces) a module wholesale.
Smaller tweaks between versions stay field-level deltas.

Usage
-----
``python tools/gen_ome_metadata.py --check``
    Regenerate every derived version into memory and assert it is structurally
    equivalent to the committed tree (same classes, fields, requirement
    levels, ``register_subclass`` keys, module-level type aliases and
    ``__all__``).  Exits non-zero on any drift.  Whitespace, formatting,
    imports and docstrings are normalised away for this comparison.

``python tools/gen_ome_metadata.py --write``
    Overwrite the committed derived packages with freshly generated files,
    then run ``ruff format`` and ``ruff check --fix`` over them.

This script uses ``ast`` + ``ast.unparse`` and therefore needs Python >= 3.9.
That is fine: it is a dev-only tool.  Its *outputs* are written in the same
Python-3.8-safe syntax as the templates.
"""

# stdlib
import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# A module set: module name -> its parsed AST.
Modules = Dict[str, ast.Module]

# --------------------------------------------------------------------------
#   Where the trees live
# --------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = _ROOT / "src" / "abczarr" / "ome" / "metadata"
OME_TEMPLATES = _ROOT / "tools" / "ome_templates"


def _read_template(relpath: str) -> str:
    """Read an injected-module template source (a real ``.py`` fragment)."""
    return (OME_TEMPLATES / relpath).read_text()


# ==========================================================================
#
#                         THE FORWARD DELTA OPS
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


class SetDoc:
    """Replace a class's docstring.

    A field's prose travels with the field: a description that mentions a
    field (or a cross-reference to a module) only holds once that field or
    module exists, so the docstring is set at the version that introduces
    it, not carried in the template. ``doc`` is written plain -- the first
    line flush, every later line without leading indentation -- and this op
    re-indents the continuation lines to sit under the opening quote of a
    top-level class, matching a hand-written docstring literal.
    """

    def __init__(
        self,
        module: str,
        cls: Tuple[str, ...],
        doc: str,
        indent: int = 4,
    ) -> None:
        self.module, self.cls, self.doc = module, cls, doc
        self.indent = indent

    def apply(self, modules: Modules) -> None:
        node = _class_node(modules[self.module], self.cls)
        pad = " " * self.indent
        lines = self.doc.strip("\n").split("\n")
        rendered = [lines[0]] + [
            (pad + line if line else "") for line in lines[1:]
        ]
        value = "\n".join(rendered) + "\n" + pad
        expr = ast.Expr(value=ast.Constant(value=value))
        body = node.body
        if body and _is_docstring(body[0]):
            body[0] = expr
        else:
            body.insert(0, expr)


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
    """Introduce or replace a module wholesale from template source.

    Used both for a brand-new module (``scenes`` at 0.6.dev3) and for a
    module whose whole shape changes at a version (``transformations`` at
    every 0.6 pre-release, ``ome`` when it gains a scene carrier).  The
    source is authored with the chain's cross-reference token so it is
    version-substituted like every other module.
    """

    def __init__(self, name: str, source: str) -> None:
        self.name, self.source = name, source

    def apply(self, modules: Modules) -> None:
        modules[self.name] = ast.parse(self.source)


# ==========================================================================
#
#                     STABLE CHAIN: v0_1 -> v0_2..v0_5
#
# ==========================================================================

# -- New modules introduced at v0.4 ----------------------------------------
# Real ``.py`` fragments under tools/ome_templates/stable/v0_4/, authored with
# the ``v0_1`` cross-reference token (the canonical placeholder every managed
# module uses); the emitter substitutes it per output version.

_AXES_SOURCE = _read_template("stable/v0_4/axes.py")
_TRANSFORMATIONS_SOURCE = _read_template("stable/v0_4/transformations.py")


# -- Docstrings that grow as fields are added ------------------------------
# Authored plain (continuation lines flush left); SetDoc re-indents them.
# Written with the ``v0_1`` cross-reference token, substituted per version,
# so the transformations links only appear from v0.4 where that module lives.

_MULTISCALE_DOC_V0_3 = """\
A multiscale image pyramid: its axes and resolution levels.

`axes` names and orders the pyramid's dimensions: `t`, `c`, `z`,
`y`, `x`, in whatever subset and order the image uses. `datasets`
lists its resolution levels from full resolution down, each a
[Dataset][abczarr.ome.metadata.v0_1.images.Dataset].
"""

_DATASET_DOC_V0_4 = """\
One resolution level of a multiscale pyramid.

`path` is the name of the Zarr array holding this level, relative
to the image group. `coordinateTransformations` places it in the
pyramid's physical space: a
[Scale][abczarr.ome.metadata.v0_1.transformations.Scale], optionally
followed by a
[Translation][abczarr.ome.metadata.v0_1.transformations.Translation],
one value per axis.
"""

_MULTISCALE_DOC_V0_4 = """\
A multiscale image pyramid: its axes and resolution levels.

`axes` names and orders the pyramid's dimensions: `t`, `c`, `z`,
`y`, `x`, in whatever subset and order the image uses. `datasets`
lists its resolution levels from full resolution down, each a
[Dataset][abczarr.ome.metadata.v0_1.images.Dataset].
`coordinateTransformations` here, if given, applies to every
level before its own.
"""


#: The stable forward delta table.  ``DELTAS[v]`` moves the running module set
#: from the previous version to ``v``.
DELTAS_STABLE = {
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
        SetDoc("images", ("Multiscale",), _MULTISCALE_DOC_V0_3),
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
        SetDoc("images", ("Dataset",), _DATASET_DOC_V0_4),
        AddField(
            "images", ("Multiscale",), "coordinateTransformations",
            "Optional[tx.List[CoordinateTransformation]]",
            after="datasets",
        ),
        SetDoc("images", ("Multiscale",), _MULTISCALE_DOC_V0_4),
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
#              0.6 PRE-RELEASE CHAIN: v0_6dev1 -> dev2..rc0
#
# ==========================================================================
#
# The 0.6 previews evolve by rewriting a few modules wholesale, not by nudging
# individual fields, so the table is almost all ``AddModule`` (whole-module
# replacement) from real template files.  The version-stable modules
# (``images``, ``labels``, ``omero``, ``plates``, ``systems``, ``wells``) never
# change across the 0.6 line; they come straight from the ``v0_6dev1``
# template.
#
#   dev1 -> dev2  nothing but the version string (identical class surface;
#                 the schema differences between them are pure JSON-schema
#                 constraints, invisible at the metadata layer).
#   dev2 -> dev3  the transformation model is rewritten (index-list mapAxis,
#                 wrapped byDimension, inverseOf dropped); ``ome`` gains an
#                 ``OMEScene`` carrier; a new ``scenes`` module appears.
#   dev3 -> dev4  the transform input/output overhaul: a coordinate-system
#                 name string becomes a ``Space`` object.
#   dev4 -> rc0   ``projectAxis`` is added and byDimension's inner axis keys
#                 are re-spelled (``input_axes`` -> ``inputAxes``).

DELTAS_DEV = {
    # dev1 -> dev2: version string only.
    "v0_6dev2": [],
    # dev2 -> dev3
    "v0_6dev3": [
        AddModule(
            "transformations",
            _read_template("dev/v0_6dev3/transformations.py"),
        ),
        AddModule("ome", _read_template("dev/v0_6dev3/ome.py")),
        AddModule("scenes", _read_template("dev/v0_6dev3/scenes.py")),
    ],
    # dev3 -> dev4
    "v0_6dev4": [
        AddModule(
            "transformations",
            _read_template("dev/v0_6dev4/transformations.py"),
        ),
    ],
    # dev4 -> rc0
    "v0_6rc0": [
        AddModule(
            "transformations",
            _read_template("dev/v0_6rc0/transformations.py"),
        ),
    ],
}


# ==========================================================================
#
#                                CHAINS
#
# ==========================================================================


class Chain:
    """A template version and the derived versions folded forward from it."""

    def __init__(
        self,
        template: str,
        generated: List[str],
        version_string: Dict[str, str],
        base_modules: List[str],
        deltas: Dict[str, List[Any]],
        init_private: Sequence[str] = (),
    ) -> None:
        self.template = template
        self.generated = generated
        self.version_string = version_string
        self.base_modules = base_modules
        self.deltas = deltas
        #: modules written as files but deliberately kept out of the package
        #: ``__init__`` re-export surface (matching the hand-written tree).
        self.init_private = frozenset(init_private)

    @property
    def versions(self) -> List[str]:
        return [self.template] + self.generated

    @property
    def generated_header(self) -> str:
        return (
            f"# Generated from {self.template} by tools/gen_ome_metadata.py"
            " -- do not edit\n"
        )

    @property
    def template_header(self) -> str:
        return (
            "# Hand-written source of truth. tools/gen_ome_metadata.py"
            " generates\n"
            "# the sibling versions from this package; edit here, then"
            " regenerate.\n"
        )


STABLE = Chain(
    template="v0_1",
    generated=["v0_2", "v0_3", "v0_4", "v0_5"],
    version_string={
        "v0_1": "0.1",
        "v0_2": "0.2",
        "v0_3": "0.3",
        "v0_4": "0.4",
        "v0_5": "0.5",
    },
    base_modules=["images", "labels", "ome", "omero", "plates", "wells"],
    deltas=DELTAS_STABLE,
)

DEV = Chain(
    template="v0_6dev1",
    generated=["v0_6dev2", "v0_6dev3", "v0_6dev4", "v0_6rc0"],
    version_string={
        "v0_6dev1": "0.6.dev1",
        "v0_6dev2": "0.6.dev2",
        "v0_6dev3": "0.6.dev3",
        "v0_6dev4": "0.6.dev4",
        "v0_6rc0": "0.6rc0",
    },
    base_modules=[
        "images", "labels", "ome", "omero", "plates", "systems",
        "transformations", "wells",
    ],
    deltas=DELTAS_DEV,
    # ``scenes`` is reachable through ``ome`` and rendered directly in the
    # API docs, but -- matching the hand-written 0.6 packages -- it is not
    # re-exported from the package ``__init__``.
    init_private=["scenes"],
)

CHAINS = [STABLE, DEV]


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


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


# ==========================================================================
#
#                              GENERATION
#
# ==========================================================================


def _template_modules(chain: Chain) -> Modules:
    """Parse the delta-editable template modules into ASTs."""
    modules: Modules = {}
    for name in chain.base_modules:
        source = (
            METADATA_DIR / chain.template / (name + ".py")
        ).read_text()
        modules[name] = ast.parse(source)
    return modules


def _build(chain: Chain) -> Dict[str, Modules]:
    """Return ``{version: {module_name: ast.Module}}`` for every version by
    folding the forward deltas over the template."""
    modules = _template_modules(chain)
    built = {chain.template: _copy_modules(modules)}
    for version in chain.generated:
        for op in chain.deltas[version]:
            op.apply(modules)
        built[version] = _copy_modules(modules)
    return built


def _copy_modules(modules: Modules) -> Modules:
    # Re-parse from the unparsed source: a cheap deep copy that also proves
    # each intermediate stays valid Python.
    return {
        name: ast.parse(ast.unparse(node)) for name, node in modules.items()
    }


def _substitute_version(chain: Chain, source: str, version: str) -> str:
    """Rewrite the template cross-reference token in docstrings to *version*.

    The token ``abczarr.ome.metadata.<template>`` appears only inside
    docstrings (imports are relative), so a plain string replacement is
    unambiguous.
    """
    return source.replace(
        "abczarr.ome.metadata." + chain.template,
        "abczarr.ome.metadata." + version,
    )


def _module_source(
    chain: Chain, node: ast.Module, version: str, header: str
) -> str:
    body = _substitute_version(chain, ast.unparse(node), version)
    return header + "\n" + body + "\n"


def _version_source(chain: Chain, version: str) -> str:
    ver = chain.version_string[version]
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


def _module_names(
    version_modules: Modules, init_private: frozenset
) -> List[str]:
    """Ordered module names for a version's package ``__init__`` (``version``
    last but one, matching the committed ordering: modules alphabetical, then
    version, wells).  ``init_private`` modules are written as files but left
    out of the re-export surface."""
    names = sorted(n for n in version_modules if n not in init_private)
    ordered = [n for n in names if n not in ("version", "wells")]
    ordered.append("version")
    ordered.append("wells")
    return ordered


def _render_version(
    chain: Chain, version: str, version_modules: Modules
) -> Dict[str, str]:
    """Return ``{filename: source}`` for one generated version package."""
    files: Dict[str, str] = {}
    for name, node in version_modules.items():
        files[name + ".py"] = _module_source(
            chain, node, version, chain.generated_header
        )
    files["version.py"] = (
        chain.generated_header + "\n" + _version_source(chain, version)
    )
    names = _module_names(version_modules, chain.init_private)
    files["__init__.py"] = (
        chain.generated_header + "\n" + _init_source(names)
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
    """Regenerate every derived version in every chain and compare
    structurally to the committed tree.  Returns a list of human-readable
    drift messages (empty when the tree is in sync)."""
    problems: List[str] = []
    for chain in CHAINS:
        problems.extend(_check_chain(chain))
    return problems


def _check_chain(chain: Chain) -> List[str]:
    built = _build(chain)
    problems: List[str] = []
    for version in chain.generated:
        generated = _render_version(chain, version, built[version])
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
    written: List[Path] = []
    for chain in CHAINS:
        written.extend(_write_chain(chain))
    _ruff(written)
    print(f"wrote {len(written)} files across {len(CHAINS)} chains")


def _write_chain(chain: Chain) -> List[Path]:
    built = _build(chain)
    written: List[Path] = []
    for version in chain.generated:
        directory = METADATA_DIR / version
        directory.mkdir(exist_ok=True)
        rendered = _render_version(chain, version, built[version])
        # remove any stale generated module (e.g. a module a delta no longer
        # produces) before writing the fresh set
        keep = set(rendered)
        for existing in directory.glob("*.py"):
            if existing.name not in keep:
                existing.unlink()
        for name, source in rendered.items():
            path = directory / name
            path.write_text(source)
            written.append(path)

    # mark the template package as the source of truth (header only)
    _ensure_template_header(chain)
    return written


def _ensure_template_header(chain: Chain) -> None:
    init = METADATA_DIR / chain.template / "__init__.py"
    text = init.read_text()
    if "Hand-written source of truth" not in text:
        init.write_text(chain.template_header + "\n" + text)


def _ruff(paths: List[Path]) -> None:
    files = [str(p) for p in paths]
    subprocess.run(
        ["ruff", "check", "--fix", "--quiet", *files], check=False
    )
    subprocess.run(["ruff", "format", "--quiet", *files], check=False)


def _tmp_write(chain: Chain, built: Dict[str, Modules]) -> Path:
    """Render every generated version into a fresh temp dir (used only to
    smoke-test that outputs import; not part of --check)."""
    tmp = Path(tempfile.mkdtemp(prefix="ome_gen_"))
    for version in chain.generated:
        directory = tmp / version
        directory.mkdir()
        for name, source in _render_version(
            chain, version, built[version]
        ).items():
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

    generated_count = sum(len(chain.generated) for chain in CHAINS)

    if args.check:
        problems = check()
        if problems:
            print("OME metadata tree is out of sync with the template:\n")
            print("\n\n".join(problems))
            return 1
        print(
            f"OME metadata tree is in sync ({generated_count} generated "
            f"versions across {len(CHAINS)} chains)."
        )
        return 0

    write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

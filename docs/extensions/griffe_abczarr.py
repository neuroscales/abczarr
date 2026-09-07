"""Synthesizes ``__init__`` signatures for the project's attrs classes.

Griffe ships built-in support for ``@dataclasses.dataclass``, which
recreates a class's ``__init__`` signature from its annotated fields
during static analysis. It has no equivalent support for attrs, so a
class built by an attrs-based decorator is left with no ``__init__`` in
the documentation model: its constructor renders with no parameters,
and every field's default renders as "required" regardless of the
field's real default.

abczarr's data classes are built by wrappers around ``attrs.define`` and
``attrs.frozen`` (``abczarr._core.auto.attrs.define``, ``.frozen``,
``.autodefine``, ``.autofrozen``), and most of their fields are given no
literal default in the class body. Their defaults are computed from the
field's type at class-creation time instead, through the same resolvers
``bagof-magic`` uses (``bagof-converters``, ``bagof-validators``,
``bagof-factories``). A default computed this way cannot be recovered
by reading the source text, so this extension imports the already-built
class and reads its resolved attrs fields directly, rather than
re-deriving the fields from the class body the way the built-in
dataclasses extension does.

For each class with no ``__init__`` already in the documentation model,
this extension imports the corresponding runtime object and, if it is
an attrs class, adds a synthesized ``__init__`` built from
``attrs.fields()``: one parameter per init field, in the field's real
position and keyword-only-ness, with its default rendered from the
resolved value. A field whose default cannot be resolved without
constructing a partial instance, or whose factory raises when called on
its own, is left without a default, which renders as required. A
field's annotation and description are read from the matching
attribute already present in the documentation model (searched up the
class's MRO), so a type still cross-references the way it does today.
"""

from __future__ import annotations

import importlib
from typing import Any

import attrs
import typing_extensions as tx
from griffe import (
    Attribute,
    Class,
    Extension,
    Function,
    Module,
    Parameter,
    ParameterKind,
    Parameters,
    logger,
)


def _runtime_object(class_: Class) -> tx.Optional[tx.Any]:
    """The live object a documentation-model class was built from.

    Returns `None` when the owning module cannot be imported, or when
    the class cannot be reached from it under the same dotted name
    (for example, a class assembled dynamically rather than with a
    `class` statement).
    """
    try:
        module_path = class_.module.canonical_path
    except ValueError:
        return None
    try:
        obj: tx.Any = importlib.import_module(module_path)
    except Exception:
        return None
    relative = class_.canonical_path[len(module_path) + 1:]
    for part in relative.split(".") if relative else ():
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    return obj


def _find_attribute(class_: Class, name: str) -> tx.Optional[Attribute]:
    """The documented attribute `name` on `class_` or an ancestor.

    Carries a field's annotation and docstring over from wherever it
    was declared, so an inherited field keeps the type and description
    written on the base class.
    """
    try:
        mro = (class_, *class_.mro())
    except ValueError:
        mro = (class_,)
    for klass in mro:
        member = klass.members.get(name)
        if isinstance(member, Attribute):
            return member
    return None


def _default_text(field: attrs.Attribute) -> tx.Optional[str]:
    """The source text an attrs field's default renders as, if any.

    A plain default is shown as its `repr`. A factory default is called
    to obtain the value it produces, and that value's `repr` is shown
    instead, since most of this project's factories build a value from
    the field's type rather than naming a reusable callable. A factory
    that depends on the instance under construction, or that raises
    when called with no arguments (a required field with no usable
    default), is left without a default, which renders as required.
    """
    value = field.default
    if value is attrs.NOTHING:
        return None
    factory = None
    while isinstance(value, attrs.Factory):
        if value.takes_self:
            return None
        factory = value.factory
        value = factory
    if factory is not None:
        try:
            value = factory()
        except Exception:
            return None
    try:
        return repr(value)
    except Exception:
        return None


def _build_init(class_: Class, cls: tx.Any) -> Function:
    """A synthesized ``__init__`` for the attrs class `cls`.

    One `Parameter` is added per init field of `cls`, in the order and
    keyword-only-ness attrs itself gives them, so an inherited field
    from a base built by the same decorators is included in its real
    position.
    """
    parameters = [
        Parameter(
            "self", annotation=None, kind=ParameterKind.positional_or_keyword
        )
    ]
    for field in attrs.fields(cls):
        if not field.init:
            continue
        attribute = _find_attribute(class_, field.name)
        parameters.append(
            Parameter(
                field.alias or field.name,
                annotation=attribute.annotation if attribute else None,
                kind=(
                    ParameterKind.keyword_only
                    if field.kw_only
                    else ParameterKind.positional_or_keyword
                ),
                default=_default_text(field),
                docstring=attribute.docstring if attribute else None,
            )
        )
    return Function(
        "__init__",
        lineno=0,
        endlineno=0,
        parent=class_,
        parameters=Parameters(*parameters),
        returns="None",
    )


def _set_init(class_: Class) -> None:
    if "__init__" in class_.members:
        return
    cls = _runtime_object(class_)
    if not hasattr(cls, "__attrs_attrs__"):
        return
    try:
        init = _build_init(class_, cls)
    except Exception:
        logger.debug("Could not synthesize __init__ for %s", class_.path)
        return
    class_.set_member("__init__", init)


def _apply_recursively(
    mod_cls: tx.Union[Module, Class], seen: tx.Set[str]
) -> None:
    if mod_cls.canonical_path in seen:
        return
    seen.add(mod_cls.canonical_path)
    if isinstance(mod_cls, Class):
        _set_init(mod_cls)
    for member in mod_cls.members.values():
        if not member.is_alias and (member.is_module or member.is_class):
            _apply_recursively(member, seen)  # type: ignore[arg-type]


class AttrsExtension(Extension):
    """Adds attrs support to the documentation model.

    Recreates the ``__init__`` signature of a class built by one of
    this project's attrs-based decorators (``define``, ``frozen``,
    ``autodefine``, ``autofrozen``), the way griffe's built-in
    dataclasses extension does for ``@dataclass``.
    """

    def on_package(self, *, pkg: Module, **kwargs: Any) -> None:
        """Adds a synthesized ``__init__`` to every attrs class in `pkg`.

        Parameters
        ----------
        pkg : Module
            The loaded package.
        """
        _apply_recursively(pkg, set())

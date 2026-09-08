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

Some modules build their classes at import time by calling a factory
rather than with a ``class`` statement. The Zarr v3 data type packages
do this for the core and extended-precision types, one class per type
name. Static analysis never sees a class built this way, so it is absent
from the documentation model even though the module exports it. For each
such class, this extension adds a synthesized class to the model, with
the base classes it has at runtime, its fixed ``name``, and a synthesized
``__init__``, and mirrors it into the packages that re-export it, so it
renders in the API reference like a class written by hand.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import typing as _t

import attrs
import typing_extensions as tx
from griffe import (
    Alias,
    Attribute,
    Class,
    Docstring,
    ExprConstant,
    ExprName,
    ExprSubscript,
    ExprTuple,
    Extension,
    Function,
    Module,
    Parameter,
    ParameterKind,
    Parameters,
    logger,
)

# The griffe parameter kind for each kind an inspected signature reports.
_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: ParameterKind.positional_only,
    inspect.Parameter.POSITIONAL_OR_KEYWORD: (
        ParameterKind.positional_or_keyword
    ),
    inspect.Parameter.VAR_POSITIONAL: ParameterKind.var_positional,
    inspect.Parameter.KEYWORD_ONLY: ParameterKind.keyword_only,
    inspect.Parameter.VAR_KEYWORD: ParameterKind.var_keyword,
}

# The annotation shown for an init field that has no documented attribute to
# borrow a type from. Only the extra-items catch-all field is in this
# position, and its stored type is a long union, so a plain mapping is shown.
_EXTRA_ITEMS_ANNOTATION = "Mapping[str, Any]"


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
    that raises when called with no arguments describes a required field
    with no usable default, and is left without a default, which renders
    as required. A factory that depends on the instance under
    construction has a real default whose value is not known statically,
    and is shown as `...`.
    """
    value = field.default
    if value is attrs.NOTHING:
        return None
    if isinstance(value, attrs.Factory):
        if value.takes_self:
            return "..."
        try:
            value = value.factory()
        except Exception:
            return None
    try:
        return repr(value)
    except Exception:
        return None


# The documentation module the named type aliases are defined in, used as
# the scope a rendered alias name resolves against, so it cross-references
# the type-aliases reference page. Set when the package is loaded.
_ALIAS_SCOPE: tx.Optional[Module] = None


_BUILTIN_GENERIC_NAMES = {
    tuple: "Tuple",
    list: "List",
    dict: "Dict",
    set: "Set",
    frozenset: "FrozenSet",
    type: "Type",
}


@functools.lru_cache(maxsize=1)
def _type_aliases() -> tx.Tuple[tx.Tuple[tx.Any, str], ...]:
    """The project's named type aliases, each resolved value with its name.

    Ordered most specific first, so a composite alias such as `JsonDict`
    is matched before the smaller aliases it is built from.
    """
    from abczarr._core import typing as tz

    names = (
        "JsonDict",
        "Json",
        "JsonScalar",
        "MutableJsonDict",
        "MutableJson",
        "FrozenJson",
        "Shape",
        "ShapeIsh",
        "ShapeLike",
    )
    pairs = []
    for name in names:
        value = getattr(tz, name, None)
        if value is not None:
            pairs.append((value, name))
    return tuple(pairs)


def _is_typing(obj: tx.Any) -> bool:
    """Whether a runtime type or generic origin belongs to ``typing``."""
    return obj in _BUILTIN_GENERIC_NAMES or getattr(
        obj, "__module__", None
    ) in ("typing", "typing_extensions")


def _name(name: str, parent: Class, obj: tx.Any = None) -> ExprName:
    """A name expression. One that stands for a ``typing`` object resolves
    to ``typing.<name>``, so it cross-references and modernizes like an
    annotation written in source; any other resolves in `parent`'s scope."""
    if obj is not None and _is_typing(obj):
        return ExprName(name, parent=ExprName("typing"))
    return ExprName(name, parent)


def _subscript(
    name: str, elements: list, parent: Class, obj: tx.Any = None
) -> ExprSubscript:
    """A ``name[...]`` griffe expression whose slice cross-references."""
    if len(elements) == 1:
        slice_: tx.Any = elements[0]
    else:
        slice_ = ExprTuple(elements, implicit=True)
    return ExprSubscript(_name(name, parent, obj), slice_)


def _like_annotation(annotation: tx.Any, parent: Class) -> tx.Any:
    """The type a constructor parameter accepts, as a griffe expression.

    A field accepts more than the type it stores. A converter resolved
    from the field's type coerces a mapping, a scalar, or a short-hand
    spelling into that type, and the constructor's signature carries the
    wider accepted type. This builds that accepted type as an expression
    whose names cross-reference: a named alias such as `Json` or `Shape`
    is shown by its name rather than expanded to its full definition, an
    `Annotated` type is shown without its metadata, and a union with
    `None` is shown as `Optional`.
    """
    try:
        return _like_expr(annotation, parent)
    except Exception:
        # A type this renderer does not handle falls back to its plain
        # text, so an unusual annotation never breaks the build.
        return str(annotation).replace("typing.", "")


def _like_expr(annotation: tx.Any, parent: Class) -> tx.Any:
    for value, name in _type_aliases():
        try:
            if annotation == value:
                return ExprName(name, _ALIAS_SCOPE or parent)
        except Exception:
            pass
    if annotation is type(None):
        return ExprName("None", parent)
    if annotation is Ellipsis:
        return ExprConstant("...")
    if hasattr(annotation, "__metadata__"):
        return _like_expr(annotation.__origin__, parent)
    origin = tx.get_origin(annotation)
    args = tx.get_args(annotation)
    if origin is None:
        if hasattr(annotation, "__forward_arg__"):
            return ExprName(annotation.__forward_arg__, parent)
        return _name(
            getattr(annotation, "__name__", None)
            or str(annotation).replace("typing.", ""),
            parent,
            annotation,
        )
    if origin is _t.Literal:
        return _subscript(
            "Literal", [ExprConstant(repr(a)) for a in args], parent, origin
        )
    if origin is _t.Union:
        present = [
            _like_expr(a, parent) for a in args if a is not type(None)
        ]
        if len(present) == len(args):
            return _subscript("Union", present, parent, origin)
        if len(present) == 1:
            return _subscript("Optional", present, parent, origin)
        return _subscript(
            "Optional",
            [_subscript("Union", present, parent, origin)],
            parent,
            origin,
        )
    name = (
        _BUILTIN_GENERIC_NAMES.get(origin)
        or getattr(origin, "__name__", None)
        or str(origin).replace("typing.", "")
    )
    return _subscript(
        name, [_like_expr(a, parent) for a in args], parent, origin
    )


def _build_init(class_: Class, cls: tx.Any) -> Function:
    """A synthesized ``__init__`` for the attrs class `cls`.

    The parameters, their order, and their positional or keyword-only
    kind are read from the real signature of the class's constructor, so
    a keyword-only field that attrs moves after the positional ones is
    placed correctly. Each parameter's default is resolved from the
    matching attrs field, its annotation is the type the constructor
    accepts, and its description is read from the documented attribute of
    the same name, including one inherited from a base built by the same
    decorators. The type a field stores, which is narrower than the type
    its constructor accepts, is shown in the attributes section instead.
    """
    fields = {
        field.alias or field.name: field
        for field in attrs.fields(cls)
        if field.init
    }
    parameters = [
        Parameter(
            "self", annotation=None, kind=ParameterKind.positional_or_keyword
        )
    ]
    for name, param in inspect.signature(cls.__init__).parameters.items():
        if name == "self":
            continue
        field = fields.get(name)
        attribute = _find_attribute(class_, field.name) if field else None
        # The signature shows the type the parameter accepts (the wider
        # pre-conversion type from the real constructor). The stored type
        # a field keeps is shown in the attributes section instead.
        if param.annotation is not inspect.Parameter.empty:
            annotation = _like_annotation(param.annotation, class_)
        elif attribute is not None:
            annotation = attribute.annotation
        elif field is not None:
            annotation = _EXTRA_ITEMS_ANNOTATION
        else:
            annotation = None
        parameters.append(
            Parameter(
                name,
                annotation=annotation,
                kind=_KINDS[param.kind],
                default=_default_text(field) if field else None,
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


def _set_init(class_: Class) -> bool:
    """Add a synthesized ``__init__`` to `class_`, if one is needed.

    Returns whether a signature was added. A class that already has an
    ``__init__``, one whose runtime object is not an attrs class, and one
    built with ``init=False`` (so attrs writes ``__attrs_init__`` and no
    ``__init__`` of its own) are all left unchanged.
    """
    if "__init__" in class_.members:
        return False
    cls = _runtime_object(class_)
    # ``__attrs_init__`` is tested on the class's own namespace, not with
    # hasattr, so a class that has its own field ``__init__`` is not skipped
    # because a base built with init=False put ``__attrs_init__`` in the MRO.
    if (
        cls is None
        or not attrs.has(cls)
        or "__attrs_init__" in vars(cls)
    ):
        return False
    try:
        init = _build_init(class_, cls)
    except Exception:
        logger.debug("Could not synthesize __init__ for %s", class_.path)
        return False
    class_.set_member("__init__", init)
    return True


def _name_field(cls: tx.Any) -> tx.Optional[attrs.Attribute]:
    """The ``name`` field of an attrs class, or `None` when it has none.

    A data type or codec identifies its kind through a ``name`` field
    fixed to a single value, and a class built dynamically carries the
    value there.
    """
    for field in attrs.fields(cls):
        if field.name == "name":
            return field
    return None


def _synthesize_class(module: Module, class_name: str, cls: tx.Any) -> None:
    """Add a documentation-model class for a dynamically built class.

    The synthesized class is given the base classes it has at runtime, a
    ``name`` attribute carrying its fixed data-type name, and a
    synthesized ``__init__``, so it renders in the API reference like a
    class written with a ``class`` statement.
    """
    field = _name_field(cls)
    klass = Class(class_name, parent=module)
    if field is not None and field.default is not attrs.NOTHING:
        klass.docstring = Docstring(
            f"The `{field.default}` data type.", parent=klass
        )
    # A base is stored as a name reference, not a plain string, so the
    # class's MRO resolves and an inherited field keeps its own type.
    klass.bases = [
        ExprName(base.__name__, klass)
        for base in cls.__bases__
        if base is not object
    ]
    module.set_member(class_name, klass)
    # The module builds this class after its literal ``__all__``, so its
    # name is added to the exported set that marks a member public.
    if module.exports is not None and class_name not in module.exports:
        module.exports.append(class_name)
    if field is not None and field.default is not attrs.NOTHING:
        attribute = Attribute(
            "name",
            annotation=f"Literal[{field.default!r}]",
            value=repr(field.default),
        )
        attribute.docstring = Docstring(
            f'Always `"{field.default}"`.', parent=attribute
        )
        klass.set_member("name", attribute)
    _set_init(klass)
    _reexport_upwards(module, class_name)


def _reexport_upwards(module: Module, class_name: str) -> None:
    """Mirror a synthesized class into the packages that re-export it.

    A package that does ``from .submodule import *`` carries each of the
    submodule's public names as an alias. Those aliases are resolved
    before this extension runs, so a class synthesized afterward is
    missing from every package above the module that defines it. This
    walks up from `module` and, through each ancestor package that
    re-exports it, adds the same alias and marks the name exported.
    """
    child = module
    parent = child.parent
    while isinstance(parent, Module):
        target = child.canonical_path + "." + class_name
        prefix = child.canonical_path + "."
        reexports = any(
            member.is_alias
            and str(getattr(member, "target_path", "")).startswith(prefix)
            for member in parent.members.values()
        )
        if not reexports or class_name in parent.members:
            return
        parent.set_member(class_name, Alias(class_name, target, parent=parent))
        if parent.exports is not None and class_name not in parent.exports:
            parent.exports.append(class_name)
        child = parent
        parent = child.parent


def _synthesize_generated_classes(module: Module) -> int:
    """Add a class to `module` for each one it builds dynamically.

    A few modules create their classes by calling a factory at import
    time rather than with a ``class`` statement, so static analysis never
    sees them. Each such class is exported through the module's
    ``__all__`` and defined on the module, yet is absent from the
    documentation model. This imports the module and adds one synthesized
    class for each, returning the number added.
    """
    try:
        runtime = importlib.import_module(module.canonical_path)
    except Exception:
        return 0
    count = 0
    for name in getattr(runtime, "__all__", ()):
        if name in module.members:
            continue
        obj = getattr(runtime, name, None)
        if (
            isinstance(obj, type)
            and attrs.has(obj)
            and getattr(obj, "__module__", None) == module.canonical_path
        ):
            _synthesize_class(module, name, obj)
            count += 1
    return count


def _apply_recursively(
    mod_cls: tx.Union[Module, Class], seen: tx.Set[str]
) -> int:
    """Add synthesized signatures throughout `mod_cls`, returning the
    number of classes augmented."""
    if mod_cls.canonical_path in seen:
        return 0
    seen.add(mod_cls.canonical_path)
    if isinstance(mod_cls, Module):
        count = _synthesize_generated_classes(mod_cls)
    else:
        count = _set_init(mod_cls)
    for member in list(mod_cls.members.values()):
        if not member.is_alias and (member.is_module or member.is_class):
            count += _apply_recursively(member, seen)  # type: ignore[arg-type]
    return count


class AttrsExtension(Extension):
    """Adds attrs support to the documentation model.

    Recreates the ``__init__`` signature of a class built by one of
    this project's attrs-based decorators (``define``, ``frozen``,
    ``autodefine``, ``autofrozen``), the way griffe's built-in
    dataclasses extension does for ``@dataclass``. Also adds a
    synthesized class for each data type a module builds dynamically at
    import time, which static analysis cannot see.
    """

    def on_package(self, *, pkg: Module, **kwargs: tx.Any) -> None:
        """Augments every attrs class in `pkg`, and adds dynamic ones.

        Parameters
        ----------
        pkg : Module
            The loaded package.
        """
        if pkg.name != "abczarr":
            return
        global _ALIAS_SCOPE
        try:
            _ALIAS_SCOPE = pkg["_core.typing"]
        except (KeyError, ValueError):
            _ALIAS_SCOPE = None
        if _apply_recursively(pkg, set()) == 0:
            logger.warning(
                "griffe_abczarr augmented no classes; the abczarr package "
                "could not be imported in the documentation environment, so "
                "its data classes will render without their fields. Install "
                "the package in the docs environment.",
            )

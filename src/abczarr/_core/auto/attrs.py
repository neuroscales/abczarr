"""`attrs`-based class building, with the converter, validator, and default
factory of each field resolved from its type hint.

`define` and `frozen` build a class the way `attrs.define` does, plus a
`json=` field alias and a field ordering fix. `autodefine` and
`autofrozen` additionally resolve every field's converter and default
factory from its type hint. `field`, `factory`, `autofield`,
`autofactory`, `autoconvert`, and `autovalidate` build one field, each
turning on a different combination of the per-field steps `field` itself
implements: an aliased JSON key, and a converter, validator, or default
resolved from the field's type when asked for.
"""

__all__ = [
    "define",
    "frozen",
    "autodefine",
    "autofrozen",
    "fields",
    "evolve",
    "field",
    "factory",
    "autofield",
    "autofactory",
    "autoconvert",
    "eq_safenan",
    "json_key",
]

# stdlib
from functools import wraps

# dependencies
import typing_extensions as tx
from attrs import NOTHING, evolve, make_class
from attrs import Factory as _Factory
from attrs import define as _define
from attrs import field as _field
from attrs import fields as _attrs_fields

# internals
from ..frozendict import FrozenDict
from ..rfc2119 import MISSING
from ._typing import ClassDecorator, FieldTransformer
from ._utils import eq_safenan, get_default
from .converters import get_converter as _get_converter
from .converters import wrap_converter
from .factories import get_factory
from .validators import get_validator as _get_validator

#: attrs field-metadata key under which a field's JSON key alias is stored.
#: A field declared ``field(json="bioformats2raw.layout")`` carries its
#: spec key here. A field with none serializes under its own Python name.
_JSON_KEY = "abczarr.json_key"


def get_converter(hint: tx.Any) -> tx.Optional[tx.Callable]:
    """Resolve `hint`'s converter from `bagof.converters`, or `None` when
    it has none, adapted so `attrs` reports the right `__init__` type.

    A bare `bagof.converters` converter's `__call__` is annotated with
    the generic `FROM` type variable, which would make `attrs` type the
    `__init__` parameter as that variable rather than the hints the
    field actually accepts. `wrap_converter` re-annotates it with the
    converter's own `like()` result and `hint` itself. The returned
    callable also passes the `MISSING` sentinel of an unset required
    field through unconverted.
    """
    converter = _get_converter(hint)
    if converter is None:
        return None
    wrapped = wrap_converter(converter, TO=hint)

    def convert(value: tx.Any) -> tx.Any:
        # a required-but-unset field carries the MISSING sentinel (from an
        # RFC-2119 requirement factory); pass it through unconverted.
        if value is MISSING:
            return value
        return wrapped(value)

    # keep wrap_converter's annotations so attrs still sees the right signature
    convert.__annotations__ = dict(getattr(wrapped, "__annotations__", {}))
    return convert


def get_validator(hint: tx.Any) -> tx.Optional[tx.Callable]:
    """Resolve `hint`'s validator from `bagof.validators`, or `None`
    when it has none, adapted to the three-argument signature `attrs`
    calls a validator with.

    A `bagof.validators` validator takes the value alone and raises on
    an invalid one. The returned callable accepts and ignores the
    `attrs`-supplied instance and attribute, and validates the value.
    """
    validator = _get_validator(hint)
    if validator is None:
        return None

    def _attrs_validator(
        instance: tx.Any, attribute: tx.Any, value: tx.Any
    ) -> None:
        validator(value)

    return _attrs_validator


def fields(cls_or_instance: tx.Any) -> tx.Any:
    """Like ``attrs.fields``, but accepts an instance as well as a class.

    ``attrs.fields`` accepts an instance only from attrs 26.1 onward,
    which does not install on Python 3.8. Code in this package calls it
    on instances, for example when serializing a metadata object, so an
    instance passed here is normalized to its class first.
    """
    if not isinstance(cls_or_instance, type):
        cls_or_instance = type(cls_or_instance)
    return _attrs_fields(cls_or_instance)


def _auto(kwargs: dict) -> dict:
    """Install a `transform_fields` field transformer on `kwargs`, built
    from its `factory`/`converter`/`validator` keywords.

    Those three keywords are consumed here and never reach `attrs.define`
    itself.
    """
    factory = kwargs.pop("factory", True)
    converter = kwargs.pop("converter", True)
    validator = kwargs.pop("validator", False)
    transformer = transform_fields(
        factory=factory, converter=converter, validator=validator)
    kwargs.setdefault("field_transformer", transformer)
    return kwargs


def _extra(kwargs: dict) -> dict:
    """Install an `extra_items` field transformer on `kwargs`, when its
    `extra_items` keyword is set.

    The extra-items field is typed as a `FrozenDict` for a frozen class,
    and a plain `dict` otherwise, so it matches the mutability of the
    class it is added to. The `extra_items` keyword is consumed here and
    never reaches `attrs.define` itself.
    """
    extra = kwargs.pop("extra_items", None)
    if extra is not None:
        transformer = kwargs.pop("field_transformer", None)
        dict_type = FrozenDict if kwargs.get("frozen", False) else tx.Dict
        extra_transformer = extra_items(extra, transformer, dict_type)
        kwargs["field_transformer"] = extra_transformer
    return kwargs


def _freeze(kwargs: dict) -> dict:
    """Default `kwargs` to a frozen class with attribute assignment
    disabled, unless already set.
    """
    kwargs.setdefault("frozen", True)
    kwargs.setdefault("on_setattr", None)
    return kwargs


def _fix_order(kwargs: dict) -> dict:
    """Wrap whatever field transformer `kwargs` already carries with
    `fix_order`.
    """
    transformer = kwargs.pop("field_transformer", None)
    kwargs["field_transformer"] = fix_order(transformer)
    return kwargs


@tx.overload
def define(maybe_cls: tx.Type) -> tx.Type:
    ...


@tx.overload
def define(**kwargs) -> ClassDecorator:
    ...


@wraps(_define)
def define(*args, **kwargs):
    """Build an `attrs` class, as `attrs.define` does, with an
    `extra_items` field appended when asked for and a field ordering fix
    applied.

    An `extra_items` keyword adds a field by that name to the class,
    holding whatever the class's own fields do not account for. See
    `extra_items` for the type it accepts. The field ordering fix
    resolves a base and its subclass declaring the same field name at
    different positions, as `fix_order` describes. Every other keyword
    is passed to `attrs.define` unchanged.
    """
    kwargs = _extra(kwargs)
    kwargs = _fix_order(kwargs)
    return _define(*args, **kwargs)


@tx.overload
def frozen(maybe_cls: tx.Type) -> tx.Type:
    ...


@tx.overload
def frozen(**kwargs) -> ClassDecorator:
    ...


@wraps(define)
def frozen(*args, **kwargs) -> tx.Callable[[tx.Type], tx.Type]:
    """Build a frozen `attrs` class, as `define` does, with attribute
    assignment disabled after construction.
    """
    kwargs = _freeze(kwargs)
    return define(*args, **kwargs)


@tx.overload
def autodefine(maybe_cls: tx.Type) -> tx.Type:
    ...


@tx.overload
def autodefine(**kwargs) -> ClassDecorator:
    ...


@wraps(define)
def autodefine(*args, **kwargs) -> tx.Callable[[tx.Type], tx.Type]:
    """Build an `attrs` class, as `define` does, with each field's
    default and converter resolved from its type hint.

    A field with no default of its own gets one derived from its type
    hint, and a field with no converter of its own gets one derived the
    same way, as `transform_fields` describes. A `validator=True`
    keyword resolves validators the same way, though this is off by
    default. Passing `factory=False` or `converter=False` turns the
    matching step off for every field in the class.
    """
    kwargs = _auto(kwargs)
    return define(*args, **kwargs)


@tx.overload
def autofrozen(maybe_cls: tx.Type) -> tx.Type:
    ...


@tx.overload
def autofrozen(**kwargs) -> ClassDecorator:
    ...


@wraps(define)
def autofrozen(*args, **kwargs):
    """Build a frozen `attrs` class, combining `autodefine`'s
    type-derived fields with `frozen`'s immutability.
    """
    kwargs = _auto(kwargs)
    return frozen(*args, **kwargs)


def json_key(f: tx.Any) -> str:
    """The JSON key a field serializes under: its ``json=`` alias, or its name.

    A field declared with ``field(json="image-label")`` reads and writes
    that spec key. A field with no alias uses its own Python name
    unchanged.
    """
    metadata = getattr(f, "metadata", None)
    if metadata:
        alias = metadata.get(_JSON_KEY)
        if alias is not None:
            return alias
    return f.name


@wraps(_field)
def field(**kwargs) -> tx.Any:
    """Declare one `attrs` field, as `attrs.field` does, with a `json=`
    alias and `True`/`False` converter, validator, and factory steps.

    A `json=` keyword records the field's serialized key, read back
    through `json_key`. A field with no `json=` serializes under its
    own Python name. When `type` is also given, `validator=True`,
    `converter=True`, and `factory=True` each resolve that step from
    `type`, through `get_validator`, `get_converter`, and
    `get_default`/`get_factory` respectively. `False` turns a step off.
    A `factory=True` is dropped in favor of an explicit `default`, when
    both are given. Every other keyword, including a plain callable
    passed as `converter`, `validator`, or `factory`, is passed to
    `attrs.field` unchanged.
    """
    # A ``json=`` alias records the field's serialized key (which may not
    # be a Python identifier, e.g. ``bioformats2raw.layout``) in the attrs
    # field metadata, where `json_key` reads it back. A field with no
    # `json=` keeps its own name, unaffected.
    json = kwargs.pop("json", None)
    if json is not None:
        metadata = dict(kwargs.get("metadata") or {})
        metadata[_JSON_KEY] = json
        kwargs["metadata"] = metadata

    if "type" in kwargs:

        # Validator
        if kwargs.get("validator") is True:
            kwargs["validator"] = get_validator(kwargs["type"])
        elif kwargs.get("validator") is False:
            kwargs.pop("validator")

        # Converter
        if kwargs.get("converter") is True:
            kwargs["converter"] = get_converter(kwargs["type"])
        elif kwargs.get("converter") is False:
            kwargs.pop("converter")

        # Default
        if kwargs.get("factory") is True and "default" in kwargs:
            kwargs.pop("factory")

        # Factory
        if kwargs.get("factory") is True:
            try:
                kwargs["default"] = get_default(kwargs["type"])
                kwargs.pop("factory")
            except TypeError:
                kwargs["factory"] = get_factory(kwargs["type"])
        elif kwargs.get("factory", None) is False:
            kwargs.pop("factory")

    else:
        if kwargs.get("validator") is False:
            kwargs.pop("validator")
        if kwargs.get("converter") is False:
            kwargs.pop("converter")
        if kwargs.get("factory") is False:
            kwargs.pop("factory")

    return _field(**kwargs)


@wraps(field)
def factory(factory: tx.Callable[[], tx.Any], **kwargs) -> tx.Any:
    """Declare a field whose default is built by calling `factory` with
    no arguments, as `field(factory=factory, **kwargs)` does.
    """
    kwargs.setdefault("factory", factory)
    return field(**kwargs)


@wraps(field)
def autofield(type: tx.Type, **kwargs) -> tx.Any:
    """Declare a field typed as `type`, with its converter and default
    resolved from `type`.

    Equivalent to `field(type=type, converter=True, factory=True,
    **kwargs)`. Either step can still be turned off, or replaced with an
    explicit callable, through the matching keyword in `kwargs`.
    """
    kwargs.setdefault("converter", True)
    kwargs.setdefault("factory", True)
    kwargs["type"] = type
    return field(**kwargs)


@wraps(field)
def autofactory(type: tx.Type, **kwargs) -> tx.Any:
    """Declare a field typed as `type`, with its default resolved from
    `type`.

    Equivalent to `field(type=type, factory=True, **kwargs)`.
    """
    kwargs.setdefault("factory", True)
    kwargs["type"] = type
    return field(**kwargs)


@wraps(field)
def autoconvert(type: tx.Type, **kwargs) -> tx.Any:
    """Declare a field typed as `type`, with its converter resolved from
    `type`.

    Equivalent to `field(type=type, converter=True, **kwargs)`.
    """
    kwargs.setdefault("converter", True)
    kwargs["type"] = type
    return field(**kwargs)


@wraps(field)
def autovalidate(type: tx.Type, **kwargs) -> tx.Any:
    """Declare a field typed as `type`, with its validator resolved from
    `type`.

    Equivalent to `field(type=type, validator=True, **kwargs)`.
    """
    kwargs.setdefault("validator", True)
    kwargs["type"] = type
    return field(**kwargs)


def transform_fields(
    factory: bool = True,
    converter: bool = True,
    validator: bool = False,
) -> FieldTransformer:
    """Build an `attrs` field transformer that resolves each field's
    default, converter, and validator from its type hint.

    A field with no type hint is left untouched. Otherwise, the
    resolution requested by `factory`, `converter`, and `validator` is
    skipped for a field that already has its own default, converter, or
    validator, so an explicit one set on the field is never overridden.
    A default is resolved through `get_default` first and, when that
    raises `TypeError`, through `get_factory` instead.

    Parameters
    ----------
    factory : bool
        Resolve a default for a field that has none.
    converter : bool
        Resolve a converter for a field that has none.
    validator : bool
        Resolve a validator for a field that has none.

    Returns
    -------
    FieldTransformer
        A field transformer suitable for `attrs.define`'s
        `field_transformer` argument.
    """

    def _transform_fields(
        cls: tx.Type,
        attrs_fields: tx.Sequence[tx.Any]
    ) -> tx.Sequence[tx.Any]:
        new_fields = []
        for f in attrs_fields:
            if f.type is not None:

                if factory and (f.default is NOTHING):
                    try:
                        f = f.evolve(default=get_default(f.type))
                    except TypeError:
                        f = f.evolve(default=_Factory(get_factory(f.type)))

                if converter and f.converter is None:
                    f = f.evolve(converter=get_converter(f.type))

                if validator and f.validator is None:
                    f = f.evolve(validator=get_validator(f.type))

            new_fields.append(f)
        return new_fields

    return _transform_fields


def extra_items(
    extra_items: tx.Any = tx.Any,
    transform_fields: tx.Optional[FieldTransformer] = None,
    dict_type: tx.Type = tx.Dict
) -> FieldTransformer:
    """Build an `attrs` field transformer that appends an `extra_items`
    field to the class.

    `extra_items` sets the value type the appended field accepts, so
    every unnamed field lands there typed accordingly. `True` accepts
    any value, matching a plain `typing.Any`. `False` instead appends a
    field of type `Literal[False]`, excluded from both `__init__` and
    `repr`, marking the class as accepting no extra items rather than
    omitting the field altogether. `None` returns `transform_fields`
    unchanged, appending no field at all.

    Parameters
    ----------
    extra_items : Any, bool, or None
        The value type for the appended field, or one of the special
        values above.
    transform_fields : FieldTransformer, optional
        A field transformer to run first, whose result is appended to.
    dict_type : type, optional
        The mapping type the appended field is typed as, subscripted
        with `[str, extra_items]`.

    Returns
    -------
    FieldTransformer
        A field transformer suitable for `attrs.define`'s
        `field_transformer` argument.
    """
    if extra_items is None:
        return transform_fields
    if extra_items is True:
        extra_items = tx.Any

    def field_transformer(
        cls: tx.Type,
        old_fields: tx.Sequence[tx.Any]
    ) -> tx.Sequence[tx.Any]:
        if transform_fields:
            old_fields = transform_fields(cls, old_fields)
        new_fields = list(old_fields)
        if extra_items is False:
            f = autofield(tx.Literal[False], repr=False, init=False)
        else:
            f = autofield(dict_type[str, extra_items])
        dummy = make_class("Dummy", {"extra_items": f})
        f = fields(dummy)[0]
        new_fields.append(f)
        return new_fields

    return field_transformer


def update(
    transform_fields: tx.Optional[FieldTransformer] = None,
    **kwargs
) -> FieldTransformer:
    """Build an `attrs` field transformer that replaces named fields with
    an evolved copy.

    A keyword argument names a field, with its value a dict of the
    `attrs.Attribute.evolve` keywords to apply to it. A field not named
    in `kwargs` is left unchanged.

    Parameters
    ----------
    transform_fields : FieldTransformer, optional
        A field transformer to run first, whose result is what gets
        evolved.
    **kwargs : dict
        Field name and `evolve` keyword mapping pairs.

    Returns
    -------
    FieldTransformer
        A field transformer suitable for `attrs.define`'s
        `field_transformer` argument.
    """

    def _transformer(
        cls: tx.Type, attrs_fields: tx.Sequence[tx.Any]
    ) -> tx.Sequence[tx.Any]:
        if transform_fields:
            attrs_fields = transform_fields(cls, attrs_fields)
        return [
            f.evolve(**kwargs[f.name]) if f.name in kwargs else f
            for f in attrs_fields
        ]

    return _transformer


def fix_order(
    transform_fields: tx.Optional[FieldTransformer] = None,
) -> FieldTransformer:
    """Build an `attrs` field transformer that restores a base class's
    field order for a field a subclass redeclares.

    `attrs` moves a field a subclass redeclares to the position of the
    redeclaration, ahead of fields the base class placed after it,
    changing the positional constructor order it inherited. This
    transformer instead keeps each redeclared field at the position its
    most specific base gave it, walked from the class's own bases toward
    `object`, so redeclaring a field to add a converter or default does
    not silently reorder positional arguments. A field the class
    declares that no base has keeps its own position, appended after
    every inherited field.

    Parameters
    ----------
    transform_fields : FieldTransformer, optional
        A field transformer to run first, whose result is what gets
        reordered.

    Returns
    -------
    FieldTransformer
        A field transformer suitable for `attrs.define`'s
        `field_transformer` argument.
    """

    def _transform_fields(
        cls: tx.Type, old_fields: tx.Sequence[tx.Any]
    ) -> tx.Sequence[tx.Any]:
        if transform_fields:
            old_fields = transform_fields(cls, old_fields)
        old_fields = {f.name: f for f in old_fields}

        new_fields = {}
        for base in reversed(cls.__mro__[1:]):
            if not hasattr(base, "__attrs_attrs__"):
                continue
            for f in fields(base):
                if f.name in old_fields:
                    new_fields[f.name] = old_fields[f.name]

        for f in old_fields.values():
            new_fields[f.name] = f

        return list(new_fields.values())

    return _transform_fields

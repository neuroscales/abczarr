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
]

# stdlib
from functools import wraps

# dependencies
import typing_extensions as tx
from attrs import NOTHING, evolve, fields, make_class
from attrs import Factory as _Factory
from attrs import define as _define
from attrs import field as _field

# internals
from ..frozendict import FrozenDict
from ._typing import ClassDecorator, FieldTransformer
from ._utils import eq_safenan, get_default
from .converters import get_converter
from .factories import get_factory
from .validators import get_validator


def _auto(kwargs: dict) -> dict:
    factory = kwargs.pop("factory", True)
    converter = kwargs.pop("converter", True)
    validator = kwargs.pop("validator", False)
    transformer = transform_fields(
        factory=factory, converter=converter, validator=validator)
    kwargs.setdefault("field_transformer", transformer)
    return kwargs


def _extra(kwargs: dict) -> dict:
    extra = kwargs.pop("extra_items", None)
    if extra is not None:
        transformer = kwargs.pop("field_transformer", None)
        dict_type = FrozenDict if kwargs.get("frozen", False) else tx.Dict
        extra_transformer = extra_items(extra, transformer, dict_type)
        kwargs["field_transformer"] = extra_transformer
    return kwargs


def _freeze(kwargs: dict) -> dict:
    kwargs.setdefault("frozen", True)
    kwargs.setdefault("on_setattr", None)
    return kwargs


def _fix_order(kwargs: dict) -> dict:
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
    kwargs = _auto(kwargs)
    return frozen(*args, **kwargs)


@wraps(_field)
def field(**kwargs) -> tx.Any:
    if "type" in kwargs:

        # Validator
        if kwargs.get("validator") is True:
            kwargs["validator"] = get_validator(kwargs["type"])
        elif kwargs.get("validator") is False:
            kwargs.pop("validator")

        # Converter
        if kwargs.get("converter") is True:
            kwargs["converter"] = get_converter(kwargs["type"], wrap=True)
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
    kwargs.setdefault("factory", factory)
    return field(**kwargs)


@wraps(field)
def autofield(type: tx.Type, **kwargs) -> tx.Any:
    kwargs.setdefault("converter", True)
    kwargs.setdefault("factory", True)
    kwargs["type"] = type
    return field(**kwargs)


@wraps(field)
def autofactory(type: tx.Type, **kwargs) -> tx.Any:
    kwargs.setdefault("factory", True)
    kwargs["type"] = type
    return field(**kwargs)


@wraps(field)
def autoconvert(type: tx.Type, **kwargs) -> tx.Any:
    kwargs.setdefault("converter", True)
    kwargs["type"] = type
    return field(**kwargs)


@wraps(field)
def autovalidate(type: tx.Type, **kwargs) -> tx.Any:
    kwargs.setdefault("validator", True)
    kwargs["type"] = type
    return field(**kwargs)


def transform_fields(
    factory: bool = True,
    converter: bool = True,
    validator: bool = False,
) -> FieldTransformer:

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
                    f = f.evolve(converter=get_converter(f.type, wrap=True))

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

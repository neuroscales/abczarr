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
from attrs import Factory as _Factory, define as _define, field as _field
from attrs import NOTHING, evolve, fields, make_class

# internals
from ..frozendict import FrozenDict
from .converters import get_converter
from .factories import get_factory
from ._utils import eq_safenan, get_default


@wraps(_define)
def define(*args, **kwargs):
    extra = kwargs.pop("extra_items", None)
    if extra is not None:
        transformer = kwargs.pop("field_transformer", None)
        dict_type = FrozenDict if kwargs.get("frozen", False) else tx.Dict
        extra_transformer = extra_items(extra, transformer, dict_type=dict_type)
        kwargs["field_transformer"] = extra_transformer

    kwargs["field_transformer"] = fix_order(kwargs.get("field_transformer"))
    return _define(*args, **kwargs)


@wraps(define)
def frozen(*args, **kwargs):
    kwargs.setdefault("frozen", True)
    kwargs.setdefault("on_setattr", None)
    return define(*args, **kwargs)


@wraps(define)
def autodefine(*args, **kwargs):
    factory = kwargs.pop("factory", True)
    converter = kwargs.pop("converter", True)
    transformer = transform_fields(factory=factory, converter=converter)
    kwargs.setdefault("field_transformer", transformer)
    return define(*args, **kwargs)


@wraps(define)
def autofrozen(*args, **kwargs):
    factory = kwargs.pop("factory", True)
    converter = kwargs.pop("converter", True)
    transformer = transform_fields(factory=factory, converter=converter)
    kwargs.setdefault("field_transformer", transformer)
    return frozen(*args, **kwargs)


@wraps(_field)
def field(**kwargs) -> tx.Any:
    if "type" in kwargs:
        if kwargs.get("converter", None) is True:
            kwargs["converter"] = get_converter(kwargs["type"], wrap=True)
        elif kwargs.get("converter", None) is False:
            kwargs.pop("converter")

        if kwargs.get("factory", None) is True and "default" in kwargs:
            kwargs.pop("factory")

        if kwargs.get("factory", None) is True:
            try:
                kwargs["default"] = get_default(kwargs["type"])
                kwargs.pop("factory")
            except TypeError:
                kwargs["factory"] = get_factory(kwargs["type"])
        elif kwargs.get("factory", None) is False:
            kwargs.pop("factory")

    return _field(**kwargs)


@wraps(field)
def factory(factory, **kwargs) -> tx.Any:
    kwargs.setdefault("factory", factory)
    return field(**kwargs)


@wraps(field)
def autofield(type, **kwargs) -> tx.Any:
    kwargs.setdefault("converter", True)
    kwargs.setdefault("factory", True)
    kwargs["type"] = type
    return field(**kwargs)


@wraps(field)
def autofactory(type, **kwargs) -> tx.Any:
    kwargs.setdefault("factory", True)
    kwargs["type"] = type
    return field(**kwargs)


@wraps(field)
def autoconvert(type, **kwargs) -> tx.Any:
    kwargs.setdefault("converter", True)
    kwargs["type"] = type
    return field(**kwargs)


def transform_fields(factory: bool = True, converter: bool = True):
    def _transform_fields(cls: tx.Type, attrs_fields: tx.Sequence[tx.Any]) -> tx.Sequence[tx.Any]:
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

            new_fields.append(f)
        return new_fields

    return _transform_fields


def extra_items(extra_items=tx.Any, transform_fields=None, dict_type=tx.Dict):
    if extra_items is None:
        return transform_fields
    if extra_items is True:
        extra_items = tx.Any

    def field_transformer(cls: tx.Type, old_fields: tx.Sequence[tx.Any]) -> tx.Sequence[tx.Any]:
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


def update(transform_fields=None, **kwargs):
    def _transformer(cls, attrs_fields):
        if transform_fields:
            attrs_fields = transform_fields(cls, attrs_fields)
        return [
            f.evolve(**kwargs[f.name]) if f.name in kwargs else f
            for f in attrs_fields
        ]

    return _transformer


def fix_order(transform_fields=None):
    def _transform_fields(cls, old_fields):
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

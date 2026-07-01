"""
This module implements wrappers around `attrs` utilities.

Functions
---------
define
    Wrapper around [`attrs.define`][attrs.define] that adds the option
    `extra_items` to automatically add an `extra_items` field to the class.
frozen
    Backward-compatible `attrs.frozen`.
    Sets `frozen=True` and `on_setattr=None` by default.
autodefine, autofrozen
    Wrappers around [`define`][.define] and [`frozen`][.frozen] that
    automatically register converters and factories for all fields.
    See [`autofield`][.autofield].
fields
    See [`attrs.fields`][attrs.fields].
evolve
    See [`attrs.evolve`][attrs.evolve].
field
    Wrapper around [`attrs.field`][attrs.field] that supports automatic
    converters and factories:

    * `converter=True` will automatically register a converter for the
      field's type hint.
    * `factory=True` will automatically register a factory for
      the field's type hint.
factory
    Wrapper around [`attrs.field`][attrs.field] that sets
    `factory` to the provided callable. E.g. `factory(list)` is
    equivalent to `field(factory=list)`.
autofield
    Wrapper around [`attrs.field`][attrs.field] that automatically
    registers a converter and factory for the field's type hint.
    E.g. `autofield(list)` is equivalent to
    `field(type=list, converter=True, factory=True)`.
autofactory
    Wrapper around [`attrs.field`][attrs.field] that automatically
    registers a factory for the field's type hint.
    E.g. `autofactory(list)` is equivalent to
    `field(type=list, factory=True)`.
autoconvert
    Wrapper around [`attrs.field`][attrs.field] that automatically
    registers a converter for the field's type hint.
    E.g. `autoconvert(list)` is equivalent to
    `field(type=list, converter=True)`.
register_factory
    Decorator that registers a factory for the provided type hints.
register_converter
    Decorator that registers a converter for the provided type hints.
get_factory
    Returns a factory for the provided type hint.
get_converter
    Returns a converter for the provided type hint.

Classes
-------
Factory
    Base class for factories. Subclass and implement `__call__` to create
    a factory for a specific type hint.
Converter
    Base class for converters. Subclass and implement `__call__` to create
    a converter for a specific type hint.

!!! warning "Not all type hints are supported"

    This module implement "magic" factories and converters for a subset
    of (common) hints and types (and their subclasses). E.g.

    * `NoneType`
    * `Union`
    * `Literal`
    * `abc.Sequence` (and its subtypes)
    * `abc.Mapping`  (and its subtypes)
    * `tuple`        (and its subtypes)
    * `Number`       (and its subtypes)

    Other non-type hints (e.g. `Protocol`, `TypedDict`) are not supported.

    For other types, the factory and converter fallback to the type
    itself. I.e., the factory is `hint()` and the converter is `hint(value)`.

    This may not work for all types!

"""
__all__ = [
    "define",
    "frozen",
    "fields",
    "evolve",
    "field",
    "factory",
    "autofield",
    "autofactory",
    "autoconvert",
    "register_factory",
    "register_converter",
    "get_factory",
    "get_converter",
    "Factory",
    "Converter",
]
# stdlib
import inspect
import math
import numbers
import re
from collections import abc
from types import UnionType, NoneType
from functools import wraps

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx
from attrs import define as _define, field as _field, Factory as _Factory
from attrs import fields, evolve, make_class, NOTHING

# locals
from .dtypes import asdtype

# typing
T = tx.TypeVar("T", bound=tx.Any)
ClassDecorator = tx.Callable[[tx.Type[T]], tx.Type[T]]
MagicRegistry = tx.Dict[tx.Any, tx.Type[T]]

ITERABLE = tx.TypeVar("ITERABLE", bound=abc.Iterable, default=abc.Iterable)
SEQUENCE = tx.TypeVar("SEQUENCE", bound=abc.Sequence, default=abc.Sequence)
MAPPING = tx.TypeVar("MAPPING", bound=abc.Mapping, default=abc.Mapping)
NUMBER = tx.TypeVar("NUMBER", bound=numbers.Number, default=numbers.Number)
TUPLE = tx.TypeVar("TUPLE", bound=tx.Tuple, default=tuple)
STR = tx.TypeVar("STR", bound=str, default=str)
NONETYPE = tx.TypeVar("NONETYPE", bound=NoneType, default=NoneType)
DTYPE = tx.TypeVar("DTYPE", bound=np.dtype, default=np.dtype)

KEY = tx.TypeVar("KEY", bound=tx.Any, default=tx.Any)
VAL = tx.TypeVar("VAL", bound=tx.Any, default=tx.Any)
_MAPPINGLIKE = tx.Union[
    tx.Mapping[KEY, VAL],
    tx.Iterable[tx.Tuple[KEY, VAL]],
]
_NUMBERLIKE = tx.Union[numbers.Number, np.number, np.bool_]

TO = tx.TypeVar("TO", bound=tx.Any, default=tx.Any)
FROM = tx.TypeVar("FROM", bound=tx.Any, default=tx.Any)
SEQLIKE = tx.TypeVar("SEQLIKE", bound=abc.Sequence, default=abc.Sequence)
NONETYPELIKE = tx.TypeVar("NONETYPELIKE", bound=NoneType, default=NoneType)
MAPPINGLIKE = tx.TypeVar("MAPPINGLIKE", bound=_MAPPINGLIKE, default=_MAPPINGLIKE)
TUPLELIKE = tx.TypeVar("TUPLELIKE", bound=tx.Tuple, default=tx.Tuple)
NUMBERLIKE = tx.TypeVar("NUMBERLIKE", bound=_NUMBERLIKE, default=_NUMBERLIKE)
ITERABLELIKE = tx.TypeVar("ITERABLELIKE", bound=abc.Iterable, default=abc.Iterable)
DTYPELIKE = tx.TypeVar("DTYPELIKE", bound=npt.DTypeLike, default=npt.DTypeLike)


# ======================================================================
#
#                       F I E L D   W R A P P E R S
#
# ======================================================================


@wraps(_define)
def define(*args, **kwargs):
    extra = kwargs.pop("extra_items", None)
    if extra is not None:
        transformer = kwargs.pop("field_transformer", None)
        kwargs["field_transformer"] = extra_items(extra, transformer)
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

        # Find best converter
        if kwargs.get("converter", None) is True:
            converter = get_converter(kwargs["type"], wrap=True)
            kwargs["converter"] = converter

        # Pop converter to make attrs happy
        elif kwargs.get("converter", None) is False:
            kwargs.pop("converter")

        # Do not set a factory if a default value is provided.
        if (
            kwargs.get("factory", None) is True
            and "default" in kwargs
        ):
            kwargs.pop("factory")

        # Find best factory (or default value)
        if kwargs.get("factory", None) is True:
            try:
                kwargs["default"] = get_default(kwargs["type"])
                kwargs.pop("factory")
            except TypeError:
                kwargs["factory"] = get_factory(kwargs["type"])

        # Pop factory to make attrs happy
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
    """
    Return a `transform_fields` callable that automatically registers
    factories and converters for all fields.

    Parameters
    ----------
    factory : bool
        If `True`, automatically register a factory for each field.
    converter : bool
        If `True`, automatically register a converter for each field.

    Returns
    -------
    callable
        A `transform_fields` callable that can be passed to `attrs.define`
        or `attrs.frozen`.
    """

    def _transform_fields(
        cls: tx.Type[T], fields: tx.Sequence[tx.Any]
    ) -> tx.Type[T]:
        new_fields = []
        for f in fields:
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


def extra_items(extra_items=tx.Any, transform_fields=None):
    """
    Return a `field_transformer` callable that adds an `extra_items`
    field to the class, with the provided type hint.

    Parameters
    ----------
    extra_items
        The type hint for the items in the `extra_items` dictionary.
        * If `None`, no `extra_items` field is added.
        * If `False`, `extra_items` is a `Literal[False]` field instead,
          which allows to deactivate any potentially inherited field.
        * If `True`, the type hint is `Any`.
        * Otherwise, the provided type hint is used.
    transform_fields
        A `field_transformer` callable that is applied to the fields
        before adding the `extra_items` field. This allows to modify
        the fields before adding the `extra_items` field.

    Returns
    -------
    callable
    """

    if extra_items is None:
        return transform_fields
    if extra_items is True:
        extra_items = tx.Any

    def field_transformer(
        cls: tx.Type[T], old_fields: tx.Sequence[tx.Any]
    ) -> tx.Sequence[tx.Any]:
        if transform_fields:
            old_fields = transform_fields(cls, old_fields)
        new_fields = list(old_fields)
        if extra_items is False:
            field = autofield(tx.Literal[False], repr=False, init=False)
        else:
            field = autofield(tx.Dict[str, extra_items])
        dummy = make_class("Dummy", {"extra_items": field})
        field = fields(dummy)[0]
        new_fields.append(field)
        return new_fields

    return field_transformer


def update(transform_fields=None, **kwargs):
    """
    Return a `field_transformer` callable that updates the some of the
    options of some of the fields.

    The `kwargs` should be a mapping from field names to a dictionary of
    options to update. E.g. `update(foo={"default": 42})` will update
    the `default` option of the `foo` field.
    """

    def _transformer(cls, fields):
        if transform_fields:
            fields = transform_fields(cls, fields)
        return [
            f.evolve(**kwargs[f.name])
            if f.name in kwargs else f
            for f in fields
        ]

    return _transformer


def fix_order(transform_fields=None):
    """
    Return a `field_transformer` callable that fixes the order of the
    fields, so that it matches the inheritence strategy from `dataclass`
    (i.e., fields from base classes come first, then fields from subclasses).
    """

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


def eq_safenan(x):
    if isinstance(x, (numbers.Real, np.floating)) and math.isnan(x):
        return "NaN"
    return x


# ======================================================================
#
#                    M A G I C   R E G I S T R I E S
#
# ======================================================================


_UNSET = object()


class HintMagic(tx.Generic[T]):
    """
    Base class for magic objects (factories, converters) that return a
    value of type `T` when called.
    """

    DEFAULT = tx.Any
    FALLBACK = _UNSET

    def __init__(self, hint: tx.Any = _UNSET) -> None:
        """
        Parameters
        ----------
        hint
            The type or type hint this magic object is associated with.
            E.g. `list`, `tuple`, `Union[int, str]`.
        """
        if hint is _UNSET:
            hint = self.DEFAULT
        self.hint = hint

    @property
    def origin(self) -> tx.Any:
        """Return the origin of the type hint (or the type itself)."""
        return _get_origin(self.hint, unwrap=tx.Annotated)

    @property
    def args(self) -> tx.Tuple[tx.Any, ...]:
        """Return the arguments of the type hint (or an empty tuple)."""
        return _get_args(self.hint, unwrap=tx.Annotated)

    @property
    def fallback(self) -> tx.Any:
        return get_type(self.hint, self.FALLBACK)

    def __call__(self, *args, **kwargs) -> T:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __call__"
        )


class Factory(HintMagic[T]):
    """Base class for magic factories."""

    def __call__(self) -> T:
        """Return an object of type `T`."""
        return self.fallback()


class Converter(HintMagic[TO], tx.Generic[TO, FROM]):
    """Base class for magic converters."""

    def __init__(
        self, hint: tx.Any = _UNSET, compose: bool = False
    ) -> None:
        """
        Parameters
        ----------
        hint
            The type or type hint this magic object is associated with.
            E.g. `list`, `tuple`, `Union[int, str]`.
        compose : bool
            Only used if the converter is used in a `Annotated` context.
            * If `False`, this converter replaces the base type converter.
            * If `True`, the base type converter is applied first, before
              this one is applied.
        """
        super().__init__(hint)
        self.compose = compose

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        """Compute a hint representing valid inputs for this converter."""
        return _typevar_fallback(FROM)

    def __call__(self, value: FROM) -> TO:
        if not _isinstance(value, self.origin):
            value = self.fallback(value)
        return tx.cast(TO, value)


def wrap_converter(
    converter: Converter,
    TO: tx.Any = _UNSET,
    FROM: tx.Any = _UNSET
) -> tx.Callable[[FROM], TO]:
    if TO is _UNSET:
        TO = converter.hint
    if FROM is _UNSET:
        FROM = converter.like(TO)

    def convert(value: FROM) -> TO:
        return converter(value)

    return convert


class TypeVarMixin:

    @property
    def fallback(self) -> tx.Any:
        return _typevar_fallback(self.hint)


def _typevar_fallback(hint: tx.Any) -> tx.Any:
    origin = _get_origin(hint, unwrap=tx.Annotated)
    if not _isinstance(origin, tx.TypeVar):
        return hint
    if getattr(origin, '__default__', tx.NoDefault) is not tx.NoDefault:
        return origin.__default__
    if getattr(origin, '__constraints__', ()):
        return tx.Union[origin.__constraints__]
    if getattr(origin, '__bound__', None) is not None:
        return origin.__bound__
    return tx.Any


_FACTORIES: MagicRegistry[Factory] = {}
_CONVERTERS: MagicRegistry[Converter] = {}


def register_factory(*hints: tx.Unpack[tx.Tuple[tx.Any]]) -> ClassDecorator:
    """Register a factory in the registry."""

    def decorator(cls: tx.Type[Factory]) -> tx.Type[Factory]:
        for hint in hints:
            _FACTORIES[hint] = cls
        return cls

    return decorator


def register_converter(*hints: tx.Unpack[tx.Tuple[tx.Any]]) -> ClassDecorator:
    """Register a converter in the registry."""

    def decorator(cls: tx.Type[Converter]) -> tx.Type[Converter]:
        for hint in hints:
            _CONVERTERS[hint] = cls
        return cls

    return decorator


def get_type(hint: tx.Any, fallback: type = _UNSET) -> tx.Type[tx.Any]:
    """Return a concrete type for the provided type hint."""
    origin = _get_origin(hint, unwrap=tx.Annotated)
    origin = _typevar_fallback(origin)
    if _isinstance(origin, type) and not inspect.isabstract(origin):
        return origin
    if _isinstance(fallback, type):
        return fallback
    raise TypeError(
        f"Cannot get concrete type for hint {hint} and fallback {fallback}"
    )


def get_default(hint: tx.Any) -> tx.Any:
    """
    Try to find a default value (rather than factory) for the provided
    type hint.

    * If the type hint is an optional hint (i.e., a union that contains
      `NoneType`), then the default value is `None`.

    * If the type hint is a literal
      - If the literal contains `None`, then the default value is `None`.
      - Otherwise, the default value is the first literal value.

    If none of these cases match the provided type hint, a `TypeError`
    is raised.
    """
    origin  = _get_origin(hint, unwrap=tx.Annotated)
    args = _get_args(hint, unwrap=tx.Annotated)
    if origin is tx.Literal:
        if None in args:
            return None
        return args[0]
    if origin in (tx.Union, UnionType):
        if NoneType in args:
            return None
        for arg in args:
            try:
                return get_default(arg)
            except TypeError:
                continue
    raise TypeError(f"Cannot get default for hint {hint}")


def get_factory(hint: tx.Any) -> tx.Callable[[], T]:
    """Return the most appropriate factory for the provided type hint."""

    if _get_origin(hint) is tx.Annotated:
        for arg in reversed(_get_args(hint)):  # use last factory
            if _isinstance(arg, Factory):
                return arg
            elif _issubclass(arg, Factory):
                return arg[hint](hint)

    factory_cls = _get(hint, _FACTORIES) or Factory
    return factory_cls[hint](hint)


def get_converter(hint: tx.Any, wrap: bool = False) -> tx.Callable[[tx.Any], T]:
    cls = obj = get_converter_class(hint)
    if not _isinstance(cls, Converter):
        inp = cls.like(hint)
        obj = cls(hint)
    else:
        hint = obj.hint
        inp = obj.like(hint)
    if wrap:
        obj = wrap_converter(obj, TO=hint, FROM=inp)
    return obj


def get_converter_class(hint: tx.Any) -> tx.Callable[[tx.Any], T]:
    """Return the most appropriate converter for the provided type hint."""

    if (
        _get_origin(hint) is tx.Annotated and
        any(
            _isinstance(arg, re.Pattern) or
            _isinstance(arg, Converter) or
            _issubclass(arg, Converter)
            for arg in _get_args(hint)
        )
    ):
        return AnnotatedConverter

    return _get(hint, _CONVERTERS) or Converter


def _get(hint: tx.Any, registry: dict) -> tx.Any:

    hint = _get_origin(hint, unwrap=tx.Annotated)

    best_match, best_dist = None, float('inf')
    for key in registry:
        dist = _type_dist(hint, key)
        if dist == 0:
            return registry[key]
        if dist < best_dist:
            best_match, best_dist = key, dist
        elif dist == best_dist:
            # If there is equality (e.g., with collections.abc base classes),
            # prefer the more specific type.
            if _issubclass(key, best_match):
                best_match = key
    if best_match is not None:
        return registry[best_match]

    return None


def _type_dist(typ: type, ref: type) -> int:
    """Compute the distance between two types in the inheritance hierarchy."""
    if _isinstance(typ, tx.TypeVar):
        # Special case
        typ = tx.TypeVar
    if typ is ref:
        return 0
    if not isinstance(typ, type) or not isinstance(ref, type):
        return float('inf')
    if not issubclass(typ, ref):
        return float('inf')
    distance = 0
    for base in typ.__mro__:
        if base is ref:
            return distance
        distance += 1
    # This can happen with collections.abc base classes:
    # issubclass(typ, ref) is True, but ref is not the type hierarchy.
    # -> Return a big number, but less than infinity.
    return 1000


def _issubclass(typ: tx.Any, ref: tx.Any) -> bool:
    """Safe version of issubclass that returns False for non-types."""
    if isinstance(typ, type) and isinstance(ref, type):
        return issubclass(typ, ref)
    return False


def _isinstance(obj: tx.Any, typ: tx.Any) -> bool:
    """Safe version of isinstance that returns False for non-types."""
    if isinstance(typ, type) and typ is not tx.Any:
        return isinstance(obj, typ)
    return False


def _unwrap(hint: tx.Any, unwrap: tx.Any = (tx.Annotated,)) -> tx.Any:
    """Unwrap a type hint to its origin, if it is in the unwrap list."""
    origin = _get_origin(hint)
    if unwrap is not None:
        if not isinstance(unwrap, abc.Sequence):
            unwrap = (unwrap,)
        if origin in unwrap:
            return _unwrap(tx.get_args(hint)[0], unwrap=unwrap)
    return hint


def _get_origin(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Any:
    """Get the origin of a type hint, or the type itself if not a hint."""
    if unwrap:
        hint = _unwrap(hint, unwrap=unwrap)
    origin = tx.get_origin(hint)
    if origin is None:
        return hint
    return origin


def _get_args(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Tuple[tx.Any, ...]:
    """Get the args of a type hint, or an empty tuple if not a hint."""
    hint = _unwrap(hint, unwrap=unwrap)
    origin = tx.get_origin(hint)
    if origin is None:
        return ()
    return tx.get_args(hint)


# ======================================================================
#
#                           F A C T O R I E S
#
# ======================================================================


@register_factory(NoneType)
class NoneFactory(Factory[NONETYPE]):

    DEFAULT = NoneType

    def __call__(self) -> NONETYPE:
        return None


@register_factory(tx.Union, UnionType)
class UnionFactory(Factory[T]):

    DEFAULT = tx.Union

    def __call__(self) -> T:
        # check for None first
        if NoneType in self.args:
            return None
        # check each type in the union
        for arg in self.args:
            try:
                factory = get_factory(arg)
                return factory()
            except TypeError:
                continue
        raise TypeError(
            f"Cannot create an instance of any of the union types: "
            f"{' | '.join(str(arg) for arg in self.args)}"
        )


@register_factory(tx.Literal)
class LiteralFactory(Factory[T]):

    DEFAULT = tx.Literal

    def __call__(self) -> T:
        if not self.args:
            raise TypeError("Cannot create an instance of an empty literal")
        if None in self.args:
            return None
        return self.args[0]


@register_factory(tx.TypeVar)
class TypeVarFactory(TypeVarMixin, Factory[T]):

    DEFAULT = tx.TypeVar("T")

    def __call__(self) -> T:
        return get_factory(self.fallback)()


@register_factory(abc.Sequence)
class SequenceFactory(Factory[SEQUENCE]):

    DEFAULT = abc.Sequence
    FALLBACK = list


@register_factory(abc.Mapping)
class MappingFactory(Factory[MAPPING]):

    DEFAULT = abc.Mapping
    FALLBACK = dict


# ======================================================================
#
#                           C O N V E R T E R S
#
# ======================================================================


@register_converter(tx.Any)
class AnyConverter(Converter[TO, FROM]):

    DEFAULT = tx.Any

    def __call__(self, value: FROM) -> TO:
        return value


@register_converter(NoneType)
class NoneConverter(Converter[NONETYPE, NONETYPELIKE]):

    DEFAULT = NoneType

    def __call__(self, value: NONETYPELIKE) -> NONETYPE:
        if value is not None:
            raise TypeError(f"Value {value} is not None")
        return value


@register_converter(tx.Union, UnionType)
class UnionConverter(Converter[TO, FROM]):

    DEFAULT = tx.Union

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        args = _get_args(hint, unwrap=tx.Annotated)
        args = tuple(get_converter_class(arg).like(arg) for arg in args)
        if tx.Any in args:
            return tx.Any
        filtered_args = []
        for arg in args:
            for filtered_arg in filtered_args:
                if _issubclass(arg, filtered_arg):
                    continue
                if _issubclass(filtered_arg, arg):
                    filtered_args.remove(filtered_arg)
                    break
            filtered_args.append(arg)
        args = tuple(filtered_args)
        return tx.Union[args]

    def __call__(self, value: FROM) -> TO:
        # check for None first
        if value is None and NoneType in self.args:
            return None
        # check each type in the union
        for arg in self.args:
            try:
                converter = get_converter(arg)
                return converter(value)
            except (TypeError, ValueError):
                continue
        raise TypeError(
            f"Value {value} of type {type(value)} is not compatible with "
            f"any of the union types: "
            f"{ ' | '.join(str(arg) for arg in self.args) }"
        )


@register_converter(tx.Literal)
class LiteralConverter(Converter[TO, FROM]):

    DEFAULT = tx.Literal

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        return hint

    def __call__(self, value: FROM) -> TO:
        if value not in self.args:
            raise TypeError(
                f"Value {value} is not compatible with any of the "
                f"literal types: { ' | '.join(str(arg) for arg in self.args) }"
            )
        return value


@register_converter(tx.TypeVar)
class TypeVarConverter(TypeVarMixin, Converter[TO, FROM]):

    DEFAULT = tx.TypeVar("T")

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        fallback = _typevar_fallback(hint)
        return get_converter_class(fallback).like(fallback)

    def __call__(self, value: FROM) -> TO:
        return get_converter(self.fallback)(value)


@register_converter(str)
class StringConverter(Converter[STR, FROM]):
    # Need to register this explicitely to not trigger iterable/sequence

    DEFAULT = tx.TypeVar("STR", bound=str, default=str)
    FALLABACK = str

    @classmethod
    def like(cls, hint: tx.Any = STR) -> tx.Any:
        return tx.Union[str, bytes]

    def __call__(self, value: FROM) -> STR:
        if not isinstance(value, (str, bytes)):
            # Only accept pure string-like values
            # (since `str(...)`` accepts everything)
            raise TypeError(
                f"Value {value} of type {type(value)} is not a string or bytes"
            )
        input_type = type(value)
        if _isinstance(value, bytes):
            value = value.decode()
        if _issubclass(input_type, self.origin):
            output_type = input_type
        else:
            output_type = self.fallback
        return output_type(value)


@register_converter(abc.Iterable)
class IterableConverter(Converter[ITERABLE, ITERABLELIKE]):

    DEFAULT = abc.Iterable
    FALLBACK = abc.Iterable

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        origin = _get_origin(hint, unwrap=tx.Annotated)
        args = _get_args(hint, unwrap=tx.Annotated)
        if ... in args:
            args = args[:1]
        if _issubclass(origin, abc.Mapping):
            args = (tx.Tuple[args],)
        args = tuple(get_converter_class(arg).like(arg) for arg in args)
        return tx.Iterable[args] if args else tx.Iterable

    def __call__(self, value: ITERABLELIKE) -> ITERABLE:
        input_type = type(value)
        if self.args:
            arg_converter = get_converter(self.args[0])
            value = map(arg_converter, value)
        if _issubclass(input_type, self.origin):
            output_type = input_type
        else:
            self.output = self.fallback
        if not inspect.isabstract(output_type):
            value = output_type(value)
        return value


@register_converter(abc.Sequence)
class SequenceConverter(Converter[SEQUENCE, SEQLIKE]):

    DEFAULT = abc.Sequence
    FALLBACK = list

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        args = _get_args(hint, unwrap=tx.Annotated)
        args = tuple(get_converter_class(arg).like(arg) for arg in args)
        return tx.Iterable[args] if args else tx.Iterable

    def __call__(self, value: SEQLIKE) -> SEQUENCE:
        input_type = type(value)
        if self.args:
            arg_converter = get_converter(self.args[0])
            value = map(arg_converter, value)
        if _issubclass(input_type, self.origin):
            output_type = input_type
        else:
            self.output = self.fallback
        return output_type(value)


@register_converter(abc.Mapping)
class MappingConverter(Converter[MAPPING, MAPPINGLIKE]):

    DEFAULT = abc.Mapping
    FALLBACK = dict

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        args = _get_args(hint, unwrap=tx.Annotated)
        if args:
            args = tuple(get_converter_class(arg).like(arg) for arg in args)
            return tx.Union[
                tx.Iterable[tx.Tuple[args]],
                tx.Mapping[args]
            ]
        else:
            return tx.Union[
                tx.Iterable[tx.Tuple[tx.Any, tx.Any]],
                tx.Mapping[tx.Any, tx.Any]
            ]

    def __call__(self, value: MAPPINGLIKE) -> MAPPING:
        input_type = type(value)
        if self.args:
            key_converter = get_converter(self.args[0])
            val_converter = get_converter(self.args[1])
            if isinstance(value, abc.Mapping):
                value = value.items()
            value = {
                key_converter(k): val_converter(v)
                for k, v in value
            }
        if _issubclass(input_type, self.origin):
            output_type = input_type
        else:
            output_type = self.fallback
        return output_type(value)


@register_converter(tuple)
class TupleConverter(Converter[TUPLE, TUPLELIKE]):

    DEFAULT = tuple

    @classmethod
    def like(cls, hint: tx.Any = TUPLE) -> tx.Any:
        origin = _get_origin(hint, unwrap=tx.Annotated)
        args = _get_args(hint, unwrap=tx.Annotated)
        if ... in args:
            return IterableConverter.like(origin[args[0]])
        args = tuple(get_converter_class(arg).like(arg) for arg in args)
        return tx.Tuple[args] if args else tx.Tuple

    def __call__(self, value: TUPLELIKE) -> TUPLE:
        input_type = type(value)
        if self.args:
            if len(self.args) == 2 and self.args[1] is Ellipsis:
                arg_converter = get_converter(self.args[0])
                value = map(arg_converter, value)
            else:
                value = tuple(value)
                if len(value) != len(self.args):
                    raise TypeError(
                        f"Value {value} does not match the expected "
                        f"tuple length {len(self.args)}"
                    )
                value = (
                    get_converter(arg)(v) for arg, v in zip(self.args, value)
                )

        if _issubclass(input_type, self.origin):
            output_type = input_type
        else:
            output_type = self.fallback
        return output_type(value)


@register_converter(numbers.Number)
class NumberConverter(Converter[NUMBER, NUMBERLIKE]):

    DEFAULT = numbers.Number
    FALLBACKS = {
        numbers.Number: (bool, int, float, complex),
        numbers.Real: (bool, int, float),
        numbers.Integral: (bool, int,)
    }

    @classmethod
    def like(cls, hint: tx.Any = NUMBER) -> tx.Any:
        hint = _unwrap(hint)
        fallback = _typevar_fallback(hint)
        if _issubclass(fallback, numbers.Integral):
            return tx.Union[numbers.Integral, np.integer, np.bool_]
        if _issubclass(fallback, numbers.Real):
            return tx.Union[numbers.Real, np.floating, np.bool_]
        if _issubclass(fallback, numbers.Number):
            return tx.Union[numbers.Number, np.number, np.bool_]
        return fallback

    def __call__(self, value: NUMBERLIKE) -> NUMBER:
        float_like = ("inf", "infinity", "-inf", "-infinity", "nan")
        if isinstance(value, str) and value.lower() in float_like:
            value = float(value)
        if _isinstance(value, self.hint):
            return value
        if _isinstance(self.hint, type):
            return self.hint(value)
        fallbacks = self.FALLBACKS.get(self.hint, None)
        fallbacks = fallbacks or self.FALLBACKS[numbers.Number]
        for fallback in fallbacks:
            if fallback(value) == value:
                return fallback(value)
        return fallbacks(value)


@register_converter(np.dtype, np.generic)
class DTypeConverter(Converter[DTYPE, DTYPELIKE]):

    DEFAULT = np.dtype
    FALLBACK = np.dtype

    @classmethod
    def like(cls, hint: tx.Any = DTYPE) -> tx.Any:
        return DTYPELIKE

    def __call__(self, value: DTYPELIKE) -> DTYPE:
        type = None
        if _issubclass(self.origin, np.generic):
            type = self.origin
        if self.args:
            type = self.args[0]
        return asdtype(value, type=type)


# ======================================================================
#
#                A N N O T A T E D   C O N V E R T E R S
#
# ======================================================================


class AnnotatedConverter(Converter[TO, FROM]):

    @property
    def converters(self) -> tx.Tuple[Converter, ...]:
        origin = _get_origin(self.hint, unwrap=tx.Annotated)
        converters = []
        for arg in _get_args(self.hint):
            if _isinstance(arg, re.Pattern):
                arg = RegexConverter(arg)
            if _issubclass(arg, Converter):
                arg = arg(origin)
            if _isinstance(arg, Converter):
                if getattr(arg, "compose", False):
                    converters.append(arg)
                else:
                    converters = [arg]

        if not converters or getattr(converters[0], "compose", False):
            converters.insert(0, get_converter(origin))

        return tuple(converters)

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        converters = cls(hint).converters
        for converter in converters:
            hint = converter.like(hint)
        return hint

    def __call__(self, value: FROM) -> TO:
        for converter in self.converters:
            value = converter(value)
        return value


class PositiveConverter(NumberConverter[NUMBER, NUMBERLIKE]):

    def __call__(self, value: NUMBERLIKE) -> NUMBER:
        value = super().__call__(value)
        if value <= 0:
            raise ValueError(f"Expected positive int, got {value}")
        return value


class NegativeConverter(NumberConverter[NUMBER, NUMBERLIKE]):

    def __call__(self, value: NUMBERLIKE) -> NUMBER:
        default_converter = get_converter(self.origin)
        value = default_converter(value)

        value = super().__call__(value)
        if value >= 0:
            raise ValueError(f"Expected negative int, got {value}")
        return value


class NonNegativeConverter(NumberConverter[NUMBER, NUMBERLIKE]):

    def __call__(self, value: NUMBERLIKE) -> NUMBER:
        value = super().__call__(value)
        if value < 0:
            raise ValueError(f"Expected non-negative int, got {value}")
        return value


class NonPositiveConverter(NumberConverter[NUMBER, NUMBERLIKE]):

    def __call__(self, value: NUMBERLIKE) -> NUMBER:
        value = super().__call__(value)
        if value > 0:
            raise ValueError(f"Expected non-positive int, got {value}")
        return value


class _ComparatorConverter(NumberConverter[NUMBER, NUMBERLIKE]):

    def __init__(self, threshold: NUMBER, hint: tx.Any = _UNSET) -> None:
        super().__init__(hint)
        self.threshold = threshold


class LessThan(_ComparatorConverter[NUMBER, NUMBERLIKE]):

    def __call__(self, value: NUMBERLIKE) -> NUMBER:
        value = super().__call__(value)
        if value >= self.threshold:
            raise ValueError(
                f"Expected int less than {self.threshold}, got {value}"
            )
        return value


class LessEqual(_ComparatorConverter[NUMBER, NUMBERLIKE]):

    def __call__(self, value: NUMBERLIKE) -> NUMBER:
        value = super().__call__(value)
        if value > self.threshold:
            raise ValueError(
                f"Expected int less than or equal to {self.threshold}, got {value}"
            )
        return value


class GreaterThan(_ComparatorConverter[NUMBER, NUMBERLIKE]):

    def __call__(self, value: NUMBERLIKE) -> NUMBER:
        value = super().__call__(value)
        if value <= self.threshold:
            raise ValueError(
                f"Expected int greater than {self.threshold}, got {value}"
            )
        return value


class GreaterEqual(_ComparatorConverter[NUMBER, NUMBERLIKE]):

    def __call__(self, value: NUMBERLIKE) -> NUMBER:
        value = super().__call__(value)
        if value < self.threshold:
            raise ValueError(
                f"Expected int greater than or equal to {self.threshold}, got {value}"
            )
        return value


class RangeConverter(NumberConverter[NUMBER, NUMBERLIKE]):

    def __init__(
        self,
        min_value: NUMBER,
        max_value: NUMBER,
        hint: tx.Any = _UNSET
    ) -> None:
        super().__init__(hint)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, value: NUMBERLIKE) -> NUMBER:
        value = super().__call__(value)
        if not (self.min_value <= value <= self.max_value):
            raise ValueError(
                f"Expected int in range [{self.min_value}, {self.max_value}], got {value}"
            )
        return value


class RegexConverter(StringConverter[STR, FROM]):

    def __init__(
        self, pattern: tx.Union[str, re.Pattern], hint: tx.Any = _UNSET
    ) -> None:
        super().__init__(hint)
        if not isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)
        self.pattern = pattern

    def __call__(self, value: tx.Any) -> STR:
        value = super().__call__(value)
        if not self.pattern.match(value):
            raise ValueError(
                f"Value {value} does not match pattern {self.pattern.pattern}"
            )
        return value

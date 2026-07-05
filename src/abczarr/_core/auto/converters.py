__all__ = [
    "get_converter",
    "get_converter_class",
    "register_converter",
    "Converter",
    "AnyConverter",
    "NoneConverter",
    "UnionConverter",
    "LiteralConverter",
    "TypeVarConverter",
    "StringConverter",
    "IterableConverter",
    "SequenceConverter",
    "MappingConverter",
    "TupleConverter",
    "NumberConverter",
    "DTypeConverter",
    "AnnotatedConverter",
    "PositiveConverter",
    "NegativeConverter",
    "NonNegativeConverter",
    "NonPositiveConverter",
    "LessThan",
    "LessEqual",
    "GreaterThan",
    "GreaterEqual",
    "RangeConverter",
    "RegexConverter",
]

# stdlib
import inspect
import numbers
import re
from collections import abc

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from ..dtypes import asdtype
from ._typing import (
    DICT_LIKE,
    DTYPE,
    DTYPE_LIKE,
    FROM,
    ITER_LIKE,
    ITERABLE,
    MAPPING,
    NONE_LIKE,
    NONETYPE,
    NUMBER,
    NUMBER_LIKE,
    SEQUENCE,
    SEQUENCE_LIKE,
    STR,
    TO,
    TUPLE,
    TUPLE_LIKE,
    ClassDecorator,
    MagicRegistry,
    T,
)
from ._utils import (
    _UNSET,
    HintMagic,
    NoneType,
    TypeVarMixin,
    UnionType,
    _get_args,
    _get_origin,
    _isinstance,
    _issubclass,
    _typevar_fallback,
    _unwrap,
    get_from_registry,
)

# ======================================================================
#       BASE
# ======================================================================


class Converter(HintMagic[TO], tx.Generic[TO, FROM]):
    """Base class for magic converters."""

    def __init__(self, hint: tx.Any = _UNSET, compose: bool = False) -> None:
        super().__init__(hint)
        self.compose = compose

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        return _typevar_fallback(FROM)

    def __call__(self, value: FROM) -> TO:
        if not _isinstance(value, self.origin):
            value = self.fallback(value)
        return tx.cast(TO, value)


_CONVERTERS: MagicRegistry[Converter] = {}


def wrap_converter(
    converter: Converter,
    TO: tx.Any = _UNSET,
    FROM: tx.Any = _UNSET,
) -> tx.Callable[[FROM], TO]:
    """
    Wrap a converter so that it has the correct input and output annotations.
    """
    if TO is _UNSET:
        TO = converter.hint
    if FROM is _UNSET:
        FROM = converter.like(TO)

    def convert(value: FROM) -> TO:
        return converter(value)

    return convert


def register_converter(*hints: tx.Unpack[tx.Tuple[tx.Any]]) -> ClassDecorator:
    """
    Decorator to register a converter class for one or more type hints.

    !!! example
        ```python
        @register_converter(int)
        class IntConverter(Converter[str, int]):
            def __call__(self, value: str) -> int:
                return int(value)
        ```
    """
    def decorator(cls: tx.Type[Converter]) -> tx.Type[Converter]:
        for hint in hints:
            _CONVERTERS[hint] = cls
        return cls

    return decorator


def get_converter(
    hint: tx.Any,
    wrap: bool = False,
    registry: MagicRegistry[Converter] = _CONVERTERS,
    fallback: tx.Optional[tx.Type[Converter]] = Converter
) -> tx.Optional[tx.Callable[[tx.Any], T]]:
    """
    Get the best-matching conversion function for a given type hint.
    """
    cls = get_converter_class(hint, registry, fallback)
    if cls is None:
        return None
    inp = cls.like(hint)
    obj = cls(hint)
    if wrap:
        obj = wrap_converter(obj, TO=hint, FROM=inp)
    return obj


def get_converter_class(
    hint: tx.Any,
    registry: MagicRegistry[Converter] = _CONVERTERS,
    fallback: tx.Optional[tx.Type[Converter]] = Converter
) -> tx.Type[Converter]:
    """
    Get the best-matching conversion class for a given type hint.
    """
    return get_from_registry(hint, registry) or fallback


# ======================================================================
#       IMPL
# ======================================================================


@register_converter(tx.Any)
class AnyConverter(Converter[TO, FROM]):

    DEFAULT = tx.Any

    def __call__(self, value: FROM) -> TO:
        return value


@register_converter(NoneType)
class NoneConverter(Converter[NONETYPE, NONE_LIKE]):

    DEFAULT = NoneType

    def __call__(self, value: NONE_LIKE) -> NONETYPE:
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
        filtered_args = tuple(filtered_args)
        return tx.Union[tuple(filtered_args)] if filtered_args else tx.Never

    def __call__(self, value: FROM) -> TO:
        if value is None and NoneType in self.args:
            return None
        for arg in self.args:
            try:
                converter = get_converter(arg)
                return converter(value)
            except (TypeError, ValueError):
                continue
        raise TypeError(
            f"Value {value} of type {type(value)} is not compatible with any "
            f"of the union types: {' | '.join(str(arg) for arg in self.args)}"
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
                f"Value {value} is not compatible with any of the literal "
                f"types: {' | '.join(str(arg) for arg in self.args)}"
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

    DEFAULT = tx.TypeVar("STR", bound=str, default=str)
    FALLBACK = str

    @classmethod
    def like(cls, hint: tx.Any = STR) -> tx.Any:
        return tx.Union[str, bytes]

    def __call__(self, value: FROM) -> STR:
        if not _isinstance(value, str) or _isinstance(value, bytes):
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
class IterableConverter(Converter[ITERABLE, ITER_LIKE]):

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

    def __call__(self, value: ITER_LIKE) -> ITERABLE:
        input_type = type(value)
        if self.args:
            arg_converter = get_converter(self.args[0])
            value = map(arg_converter, value)
        if _issubclass(input_type, self.origin):
            output_type = input_type
        else:
            output_type = self.fallback
        if not inspect.isabstract(output_type):
            value = output_type(value)
        return value


@register_converter(abc.Sequence)
class SequenceConverter(Converter[SEQUENCE, SEQUENCE_LIKE]):

    DEFAULT = abc.Sequence
    FALLBACK = list

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        args = _get_args(hint, unwrap=tx.Annotated)
        args = tuple(get_converter_class(arg).like(arg) for arg in args)
        return tx.Iterable[args] if args else tx.Iterable

    def __call__(self, value: SEQUENCE_LIKE) -> SEQUENCE:
        input_type = type(value)
        if self.args:
            arg_converter = get_converter(self.args[0])
            value = map(arg_converter, value)
        if _issubclass(input_type, self.origin):
            output_type = input_type
        else:
            output_type = self.fallback
        return output_type(value)


@register_converter(abc.Mapping)
class MappingConverter(Converter[MAPPING, DICT_LIKE]):

    DEFAULT = abc.Mapping
    FALLBACK = dict

    @classmethod
    def like(cls, hint: tx.Any = TO) -> tx.Any:
        args = _get_args(hint, unwrap=tx.Annotated)
        if args:
            args = tuple(get_converter_class(arg).like(arg) for arg in args)
            return tx.Union[tx.Iterable[tx.Tuple[args]], tx.Mapping[args]]
        return tx.Union[
            tx.Iterable[tx.Tuple[tx.Any, tx.Any]],
            tx.Mapping[tx.Any, tx.Any],
        ]

    def __call__(self, value: DICT_LIKE) -> MAPPING:
        input_type = type(value)
        if self.args:
            key_converter = get_converter(self.args[0])
            val_converter = get_converter(self.args[1])
            if _isinstance(value, abc.Mapping):
                value = value.items()
            value = {key_converter(k): val_converter(v) for k, v in value}
        if _issubclass(input_type, self.origin):
            output_type = input_type
        else:
            output_type = self.fallback
        return output_type(value)


@register_converter(tuple)
class TupleConverter(Converter[TUPLE, TUPLE_LIKE]):

    DEFAULT = tuple

    @classmethod
    def like(cls, hint: tx.Any = TUPLE) -> tx.Any:
        origin = _get_origin(hint, unwrap=tx.Annotated)
        args = _get_args(hint, unwrap=tx.Annotated)
        if ... in args:
            return IterableConverter.like(origin[args[0]])
        args = tuple(get_converter_class(arg).like(arg) for arg in args)
        return tx.Tuple[args] if args else tx.Tuple

    def __call__(self, value: TUPLE_LIKE) -> TUPLE:
        input_type = type(value)
        if self.args:
            if len(self.args) == 2 and self.args[1] is Ellipsis:
                arg_converter = get_converter(self.args[0])
                value = map(arg_converter, value)
            else:
                value = tuple(value)
                if len(value) != len(self.args):
                    raise TypeError(
                        f"Value {value} does not match the expected tuple "
                        f"length {len(self.args)}"
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
class NumberConverter(Converter[NUMBER, NUMBER_LIKE]):

    DEFAULT = numbers.Number
    FALLBACKS = {
        numbers.Number: (bool, int, float, complex),
        numbers.Real: (bool, int, float),
        numbers.Integral: (bool, int),
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

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        float_like = ("inf", "infinity", "-inf", "-infinity", "nan")
        if _isinstance(value, str) and value.lower() in float_like:
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
class DTypeConverter(Converter[DTYPE, DTYPE_LIKE]):

    DEFAULT = np.dtype
    FALLBACK = np.dtype

    @classmethod
    def like(cls, hint: tx.Any = DTYPE) -> tx.Any:
        return DTYPE_LIKE

    def __call__(self, value: DTYPE_LIKE) -> DTYPE:
        dtype = None
        if _issubclass(self.origin, np.generic):
            dtype = self.origin
        if self.args:
            dtype = self.args[0]
        return asdtype(value, type=dtype)


@register_converter(tx.Annotated)
class AnnotatedConverter(Converter[TO, FROM]):

    _REGISTRY: MagicRegistry[Converter] = {}

    @classmethod
    def register(cls, *hints: tx.Unpack[tx.Tuple[tx.Any]]) -> ClassDecorator:

        def decorator(converter_cls: tx.Type[Converter]) -> tx.Type[Converter]:
            for hint in hints:
                cls._REGISTRY[hint] = converter_cls
            return converter_cls

        return decorator

    @classmethod
    def _get_converter(cls, hint: tx.Any) -> tx.Optional[tx.Type[Converter]]:
        return get_converter(
            hint, wrap=False, registry=cls._REGISTRY, fallback=None
        )

    @property
    def converters(self) -> tx.Tuple[Converter, ...]:
        if getattr(self, "_converters", None) is None:
            self._converters = self._get_converters()
        return self._converters

    def _get_converters(self) -> tx.Tuple[Converter, ...]:
        origin = _get_origin(self.hint, unwrap=tx.Annotated)
        converters = []
        for arg in _get_args(self.hint):
            if _issubclass(arg, Converter):
                arg = arg(origin)
            if not _isinstance(arg, Converter):
                # Look into annotation registry
                arg = self._get_converter(arg)
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


class PositiveConverter(NumberConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value <= 0:
            raise ValueError(f"Expected positive int, got {value}")
        return value


class NegativeConverter(NumberConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        default_converter = get_converter(self.origin)
        value = default_converter(value)

        value = super().__call__(value)
        if value >= 0:
            raise ValueError(f"Expected negative int, got {value}")
        return value


class NonNegativeConverter(NumberConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value < 0:
            raise ValueError(f"Expected non-negative int, got {value}")
        return value


class NonPositiveConverter(NumberConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value > 0:
            raise ValueError(f"Expected non-positive int, got {value}")
        return value


class _ComparatorConverter(NumberConverter[NUMBER, NUMBER_LIKE]):

    def __init__(self, threshold: NUMBER, hint: tx.Any = _UNSET) -> None:
        super().__init__(hint)
        self.threshold = threshold


class LessThan(_ComparatorConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value >= self.threshold:
            raise ValueError(
                f"Expected int less than {self.threshold}, got {value}"
            )
        return value


class LessEqual(_ComparatorConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value > self.threshold:
            raise ValueError(
                "Expected int less than or equal to "
                f"{self.threshold}, got {value}"
            )
        return value


class GreaterThan(_ComparatorConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value <= self.threshold:
            raise ValueError(
                f"Expected int greater than {self.threshold}, got {value}"
            )
        return value


class GreaterEqual(_ComparatorConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value < self.threshold:
            raise ValueError(
                "Expected int greater than or equal to "
                f"{self.threshold}, got {value}"
            )
        return value


class RangeConverter(NumberConverter[NUMBER, NUMBER_LIKE]):

    def __init__(
        self,
        min_value: NUMBER,
        max_value: NUMBER,
        hint: tx.Any = _UNSET,
    ) -> None:
        super().__init__(hint)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if not (self.min_value <= value <= self.max_value):
            raise ValueError(
                f"Expected int in range [{self.min_value}, {self.max_value}], "
                f"got {value}"
            )
        return value


class LengthConverter(SequenceConverter[ITERABLE, ITER_LIKE]):

    def __init__(
        self,
        length: int,
        hint: tx.Any = _UNSET,
    ) -> None:
        super().__init__(hint)
        self.length = length

    def __call__(self, value: ITER_LIKE) -> ITERABLE:
        value = super().__call__(value)
        value = value[:self.length]
        if len(value) != self.length:
            raise ValueError(
                f"Expected iterable of length {self.length}, got {len(value)}"
            )
        return value


AnnotatedConverter.register(re.Pattern)
class RegexConverter(StringConverter[STR, FROM]):

    def __init__(
        self, pattern: tx.Union[str, re.Pattern], hint: tx.Any = _UNSET
    ) -> None:
        super().__init__(hint)
        if not _isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)
        self.pattern = pattern

    def __call__(self, value: tx.Any) -> STR:
        value = super().__call__(value)
        if not self.pattern.match(value):
            raise ValueError(
                f"Value {value} does not match pattern {self.pattern.pattern}"
            )
        return value

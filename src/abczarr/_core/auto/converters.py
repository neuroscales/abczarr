__all__ = [
    "get_converter",
    "get_converter_class",
    "register_converter",
    "Converter",
    "ToAny",
    "ToNone",
    "ToUnion",
    "ToLiteral",
    "ToTypeVar",
    "ToString",
    "ToIterable",
    "ToSequence",
    "ToMapping",
    "ToTuple",
    "ToNumber",
    "ToDType",
    "ToAnnotated",
    "ToPositive",
    "ToNegative",
    "ToNonNegative",
    "ToNonPositive",
    "ToLessThan",
    "ToLessEqual",
    "ToGreaterThan",
    "ToGreaterEqual",
    "ToInRange",
    "ToRegexMatch",
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
    NONE,
    NONE_LIKE,
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
    UNSET,
    HintMagic,
    MagicError,
    MultipleCauses,
    NoneType,
    UnionType,
    get_args_uw,
    get_from_registry,
    get_origin_uw,
    issubhint,
    safe_get_args,
    safe_isinstance,
    safe_issubclass,
    unwrap,
)

# ======================================================================
#       EXCEPTIONS
# ======================================================================


class ConversionError(MagicError):

    def __init__(self, *args, **kwargs) -> None:
        if "converter" in kwargs:
            kwargs["this"] = kwargs.pop("converter")
        super().__init__(*args, **kwargs)


class ValueConversionError(ValueError, ConversionError):
    ...


class TypeConversionError(TypeError, ConversionError):
    ...


# ======================================================================
#       BASE
# ======================================================================


class Converter(HintMagic[TO], tx.Generic[TO, FROM]):
    """Base class for magic converters."""

    def __init__(self, hint: tx.Any = UNSET, compose: bool = False) -> None:
        super().__init__(hint)
        self.compose = compose

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        return unwrap(FROM, (tx.Annotated, tx.TypeVar))

    def __call__(self, value: FROM) -> TO:
        if not safe_isinstance(value, self.origin):
            value = self._wrap_converter(self.origin)(value)
        return tx.cast(TO, value)

    def error(
        self, value: tx.Any, message: tx.Optional[str] = None, **kwargs
    ) -> ConversionError:
        """Return a ConversionError with the given value and message."""
        type = kwargs.pop("type", ConversionError)
        type = {
            "value": ValueConversionError,
            "type": TypeConversionError
        }.get(type, type)
        kwargs.setdefault("this", self)
        kwargs.setdefault("value", value)
        if message is None:
            message = "Invalid value."
        return type(message, **kwargs)

    def type_error(
        self, value: tx.Any, message: tx.Optional[str] = None
    ) -> TypeConversionError:
        """Return a TypeConversionError with the given value."""
        if message is None:
            message = f"Invalid value type: {type(value)}"
        return self.error(value, message, type=TypeConversionError)

    def value_error(
        self, value: tx.Any, message: tx.Optional[str] = None
    ) -> ValueConversionError:
        """Return a ValueConversionError with the given value."""
        if message is None:
            message = "Invalid value."
        return self.error(value, message, type=ValueConversionError)

    def _wrap_converter(self, converter: tx.Callable) -> tx.Callable:
        """
        A wrapper that wraps a converter to catch errors and raise a
        ConversionError instead. Defined here so that subclasses to not
        need to each implement this.
        """
        return _trywrap_converter(converter, self.value_error)


ConverterRegistry = MagicRegistry[tx.Type[Converter]]
_CONVERTERS: ConverterRegistry = {}


def wrap_converter(
    converter: Converter,
    TO: tx.Any = UNSET,
    FROM: tx.Any = UNSET,
) -> tx.Callable[[FROM], TO]:
    """
    Wrap a converter so that it has the correct input and output annotations.
    """
    if TO is UNSET:
        TO = converter.hint

    if FROM is UNSET:
        to_converter = converter
        if TO != converter.hint:
            to_converter = get_converter(TO)
        FROM = to_converter.like()

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
    registry: ConverterRegistry = _CONVERTERS,
    fallback: tx.Optional[tx.Type[Converter]] = Converter
) -> tx.Optional[tx.Callable[[tx.Any], T]]:
    """
    Get the best-matching conversion function for a given type hint.
    """
    if wrap:
        obj = get_converter(hint, registry=registry, fallback=fallback)
        obj = wrap_converter(obj, TO=hint)
        return obj

    cls = get_converter_class(hint, registry, fallback)
    if cls is None:
        return None
    return cls(hint)


def get_converter_class(
    hint: tx.Any,
    registry: ConverterRegistry = _CONVERTERS,
    fallback: tx.Optional[tx.Type[Converter]] = Converter
) -> tx.Type[Converter]:
    """
    Get the best-matching conversion class for a given type hint.
    """
    return get_from_registry(hint, registry) or fallback


def _process_reentrant(inp: tx.Any, reentrant: tuple = ()) -> tx.Any:
    """
    Process a reentrant type hint to avoid infinite recursion.

    If the input hint is already in the reentrant tuple, return an empty tuple.
    Otherwise, add the input hint to the reentrant tuple and return it.

    Note that an empty tuple can *only* be returned if the input hint is
    already in the reentrant tuple. We can therefore use this as a test
    in the calling context.
    """
    if inp in reentrant:
        return ()
    reentrant += (inp,)
    return reentrant


def _trywrap_converter(
    converter: tx.Callable[[FROM], TO], error: Exception
) -> tx.Callable[[FROM], TO]:
    """
    Wrap a converter to catch errors and raise a ConversionError instead.
    """
    def wrapped(value: FROM) -> TO:
        try:
            return converter(value)
        except (TypeError, ValueError) as e:
            _error = error
            if safe_issubclass(_error, Exception):
                _error = _error(value)
            raise _error from e
    return wrapped


# ======================================================================
#       IMPL
# ======================================================================

# --- Any --------------------------------------------------------------


@register_converter(tx.Any)
class ToAny(Converter[TO, FROM]):

    BOUND = DEFAULT = tx.Any

    def __call__(self, value: FROM) -> TO:
        return value


# --- None -------------------------------------------------------------


@register_converter(NoneType)
class ToNone(Converter[NONE, NONE_LIKE]):

    BOUND = DEFAULT = NoneType

    def __call__(self, value: NONE_LIKE) -> NONE:
        if value is not None:
            raise self.type_error(value, "Value is not None")
        return value



# --- Union ------------------------------------------------------------


@register_converter(tx.Union, UnionType)
class ToUnion(Converter[TO, FROM]):

    BOUND = DEFAULT = tx.Union

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.args:
            raise TypeError(
                f"Hint cannot be a empty or general union: {self.hint}"
            )

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        return _like_union(self.hint, __reentrant)

    def __call__(self, value: FROM) -> TO:
        return _to_union(value, self.hint, self._notinunion_error(value))

    def _notinunion_error(self, value: tx.Any) -> TypeConversionError:
        return self.type_error(
            value,
            "Value not compatible with any of the union types",
        )


def _like_union(
    hint: tx.Any, __reentrant: tuple = ()
) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        if hint in (tx.Union, UnionType):
            return UNSET
        return hint

    # Check hint is valid
    if not issubhint(hint, tx.Union):
        raise TypeError(f"Hint {hint} is not a Union type")

    # Get `like` hint for each argument in the union
    args = get_args_uw(hint)
    args = tuple(
        get_converter(arg).like(__reentrant)
        for arg in args
    )
    args = tuple(arg for arg in args if arg is not UNSET)

    # Only keep the more specific hints (remove super hints)
    filtered_args = []
    for arg in args:
        for filtered_arg in filtered_args:
            if issubhint(arg, filtered_arg):
                continue
            if issubhint(filtered_arg, arg):
                filtered_args.remove(filtered_arg)
                break
        filtered_args.append(arg)
    filtered_args = tuple(filtered_args)

    # If union is empty, return `Never`
    return tx.Union[tuple(filtered_args)] if filtered_args else tx.Never


def _to_union(
    value: FROM, hint: tx.Any, type_error: tx.Callable
) -> TO:
    args = get_args_uw(hint)

    # short-circuit for NoneType
    if value is None and NoneType in args:
        return None

    errors = []
    for arg in args:
        try:
            converter = get_converter(arg)
            return converter(value)
        except (TypeError, ValueError) as e:
            errors.append(e)
            continue

    raise type_error from MultipleCauses(errors)


# --- Literal ----------------------------------------------------------


@register_converter(tx.Literal)
class ToLiteral(Converter[TO, FROM]):

    BOUND = DEFAULT = tx.Literal

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        return self.hint

    def __call__(self, value: FROM) -> TO:
        if value not in self.args:
            raise self.value_error(
                "Value is not compatible with any of the literals."
            )
        return value


# --- TypeVar ----------------------------------------------------------


@register_converter(tx.TypeVar)
class ToTypeVar(Converter[TO, FROM]):

    BOUND = DEFAULT = tx.TypeVar("T")

    @property
    def unwrapped(self) -> tx.Any:
        return unwrap(self.hint, (tx.Annotated, tx.TypeVar))

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        __reentrant = _process_reentrant(self.hint, __reentrant)
        if not __reentrant:
            return self.hint
        return get_converter(self.unwrapped).like(__reentrant)

    def __call__(self, value: FROM) -> TO:
        return get_converter(self.unwrapped)(value)


# --- String -----------------------------------------------------------


@register_converter(str)
class ToString(Converter[STR, FROM]):

    BOUND = DEFAULT = tx.TypeVar("STR", bound=str, default=str)
    FALLBACK = str

    def like(self,  __reentrant: tuple = ()) -> tx.Any:
        return tx.Union[str, bytes]

    def __call__(self, value: FROM) -> STR:
        return _to_str(
            value,
            self.hint,
            self.fallback,
            self._wrap_converter,
            self._nostrlike_error(value),
        )

    def _nostrlike_error(self, value: tx.Any) -> TypeConversionError:
        return self.type_error(
            value,
            f"Value of type {type(value)} is not a string or bytes"
        )


def _to_str(
    value: FROM, hint: tx.Any, fallback: tx.Any,
    wrapper: tx.Callable, type_error: Exception,
) -> STR:
    input_type = type(value)
    origin = get_origin_uw(hint)

    # Fail for non-string-like
    if not safe_isinstance(value, str) or safe_isinstance(value, bytes):
        raise type_error

    # Decode bytes
    if safe_isinstance(value, bytes):
        value = value.decode()

    # Select best output type
    if safe_issubclass(input_type, origin):
        output_type = input_type
    else:
        output_type = fallback

    # Convert
    output_type = wrapper(output_type)
    return output_type(value)



# --- Iterable ---------------------------------------------------------


@register_converter(abc.Iterable)
class ToIterable(Converter[ITERABLE, ITER_LIKE]):

    DEFAULT = abc.Iterable
    FALLBACK = abc.Iterable

    def like(self,  __reentrant: tuple = ()) -> tx.Any:
        return _like_iterable(self.hint, __reentrant)

    def __call__(self, value: ITER_LIKE) -> ITERABLE:
        return _to_iterable(
            value,
            self.hint,
            self.fallback,
            self._wrap_converter
        )


def _like_iterable(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        return hint

    origin = get_origin_uw(hint)
    args = get_args_uw(hint)
    if ... in args:
        args = args[:1]
    if safe_issubclass(origin, abc.Mapping):
        args = (tx.Tuple[args],)
    args = tuple(
        get_converter(arg).like(__reentrant)
        for arg in args
    )
    args = tuple(arg for arg in args if arg is not UNSET)
    return tx.Iterable[args] if args else tx.Iterable


def _to_iterable(
    value: ITER_LIKE, hint: tx.Any, fallback: tx.Any,
    wrapper: tx.Callable
) -> ITERABLE:
    input_type = type(value)
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    # Make generator of converters
    if args:
        arg_converter = wrapper(get_converter(args[0]))
        value = map(arg_converter, value)

    # Find best output type
    if safe_issubclass(input_type, origin):
        output_type = input_type
    else:
        output_type = fallback

    # Convert
    if not inspect.isabstract(output_type):
        output_type = wrapper(output_type)
        return output_type(value)

    return value


# --- Sequence ---------------------------------------------------------


@register_converter(abc.Sequence)
class ToSequence(Converter[SEQUENCE, SEQUENCE_LIKE]):

    DEFAULT = abc.Sequence
    FALLBACK = list

    def like(self,  __reentrant: tuple = ()) -> tx.Any:
        return _like_sequence(self.hint, __reentrant)

    def __call__(self, value: SEQUENCE_LIKE) -> SEQUENCE:
        return _to_sequence(
            value,
            self.hint,
            self.fallback,
            self._wrap_converter
        )


def _like_sequence(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        return hint

    args = get_args_uw(hint)
    args = tuple(
        get_converter(arg).like(__reentrant)
        for arg in args
    )
    args = tuple(arg for arg in args if arg is not UNSET)
    return tx.Iterable[args] if args else tx.Iterable


def _to_sequence(
    value: SEQUENCE_LIKE, hint: tx.Any, fallback: tx.Any,
    wrapper: tx.Callable
) -> SEQUENCE:
    input_type = type(value)
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    if args:
        converter = wrapper(get_converter(args[0]))
        value = map(converter, value)

    if safe_issubclass(input_type, origin):
        output_type = input_type
    else:
        output_type = fallback
    output_type = wrapper(output_type)
    return output_type(value)


# --- Mapping ----------------------------------------------------------


@register_converter(abc.Mapping)
class ToMapping(Converter[MAPPING, DICT_LIKE]):

    DEFAULT = abc.Mapping
    FALLBACK = dict

    def like(self,  __reentrant: tuple = ()) -> tx.Any:
        return _like_mapping(self.hint, __reentrant)

    def __call__(self, value: DICT_LIKE) -> MAPPING:
        return _to_mapping(
            value,
            self.hint,
            self.fallback,
            self._wrap_converter,
        )


def _like_mapping(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        return hint

    args = get_args_uw(hint)
    if args:
        args = tuple(
            get_converter(arg).like(__reentrant)
            for arg in args
        )
        args = tuple(arg for arg in args if arg is not UNSET)
        return tx.Union[
            tx.Iterable[tx.Tuple[args]],
            tx.Mapping[args]
        ]

    return tx.Union[
        tx.Iterable[tx.Tuple[tx.Any, tx.Any]],
        tx.Mapping[tx.Any, tx.Any],
    ]

def _to_mapping(
    value: DICT_LIKE, hint: tx.Any, fallback: tx.Any,
    wrapper: tx.Callable,
) -> MAPPING:
    input_type = type(value)
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    if args:
        key_converter = wrapper(get_converter(args[0]))
        val_converter = wrapper(get_converter(args[1]))
        if safe_isinstance(value, abc.Mapping):
            value = value.items()
        value = {key_converter(k): val_converter(v) for k, v in value}

    if safe_issubclass(input_type, origin):
        output_type = input_type
    else:
        output_type = fallback
    output_type = wrapper(output_type)
    return output_type(value)


# --- Tuple ------------------------------------------------------------


@register_converter(tuple)
class ToTuple(Converter[TUPLE, TUPLE_LIKE]):

    DEFAULT = tuple

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        return _like_tuple(self.hint, __reentrant)

    def __call__(self, value: TUPLE_LIKE) -> TUPLE:
        return _to_tuple(
            value,
            self.hint,
            self.fallback,
            self._wrap_converter,
            self.length_error(value, len(get_args_uw(self.hint)))
        )

    def length_error(self, value: TUPLE, target: int) -> ValueConversionError:
        message = (f"Expected iterable of length {target}, "
                   f"got {len(value)}.")
        return self.value_error(value, message)


def _like_tuple(hint: tx.Any, __reentrant: tuple = ()) -> tx.Any:
    __reentrant = _process_reentrant(hint, __reentrant)
    if not __reentrant:
        return hint

    origin = get_origin_uw(hint)
    args = get_args_uw(hint)
    if ... in args:
        return ToIterable(origin[args[0]]).like(__reentrant)
    args = tuple(
        get_converter(arg).like(__reentrant)
        for arg in args
    )
    args = tuple(tx.Any if arg is UNSET else arg for arg in args)
    return tx.Tuple[args] if args else tx.Tuple


def _to_tuple(
    value: TUPLE_LIKE, hint: tx.Any, fallback: tx.Any,
    wrapper: tx.Callable, length_error: Exception
) -> TUPLE:
    """Convert a tuple-like value to a tuple."""
    input_type = type(value)
    origin = get_origin_uw(hint)
    args = get_args_uw(hint)

    if args:
        if len(args) == 2 and args[1] is Ellipsis:
            converter = wrapper(get_converter(args[0]))
            value = map(converter, value)
        else:
            value = tuple(value)
            if len(value) != len(args):
                raise length_error
            converters = map(wrapper, map(get_converter, args))
            value = (
                converter(val)
                for val, converter in zip(value, converters)
            )

    if safe_issubclass(input_type, origin):
        output_type = input_type
    else:
        output_type = fallback
    output_type = wrapper(output_type)
    return output_type(value)


# --- Number -----------------------------------------------------------


@register_converter(numbers.Number)
class ToNumber(Converter[NUMBER, NUMBER_LIKE]):

    DEFAULT = numbers.Number
    FALLBACKS = {
        numbers.Number: (bool, int, float, complex),
        numbers.Real: (bool, int, float),
        numbers.Integral: (bool, int),
    }
    FLOAT_LIKE = ("inf", "infinity", "-inf", "-infinity", "nan")

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        __reentrant = _process_reentrant(self.hint, __reentrant)
        if not __reentrant:
            return self.hint

        hint = unwrap(self.hint)
        fallback = unwrap(hint, tx.TypeVar)
        if safe_issubclass(fallback, numbers.Integral):
            return tx.Union[numbers.Integral, np.integer, np.bool_]
        if safe_issubclass(fallback, numbers.Real):
            return tx.Union[numbers.Real, np.floating, np.bool_]
        if safe_issubclass(fallback, numbers.Number):
            return tx.Union[numbers.Number, np.number, np.bool_]
        return fallback

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        if safe_isinstance(value, str) and value.lower() in self.FLOAT_LIKE:
            value = float(value)
        if safe_isinstance(value, self.hint):
            return value

        # If a concrete type, use it as a converter
        origin = self.origin
        if safe_isinstance(origin, type) and not inspect.isabstract(origin):
            return self._try_convert(value, origin)

        # Otherwise, try the fallbacks
        # -> Only accept output if equality is preserved
        fallbacks = self.FALLBACKS.get(origin, None)
        fallbacks = fallbacks or self.FALLBACKS[numbers.Number]
        for fallback in fallbacks:
            value = self._softtry_convert(value, fallback)
            if value is not None:
                return value

        # Try the most general fallback
        return self._try_convert(value, fallbacks[0])

    def _try_convert(self, value: NUMBER_LIKE, type: type) -> NUMBER:
        return _trywrap_converter(type, self.value_error)(value)

    def _softtry_convert(self, value: NUMBER_LIKE, type: type) -> NUMBER:
        try:
            new_value = type(value)
            if new_value == value:
                return new_value
        except (ValueError, TypeError):
            ...


# --- DType ------------------------------------------------------------


@register_converter(np.dtype, np.generic)
class ToDType(Converter[DTYPE, DTYPE_LIKE]):

    DEFAULT = np.dtype
    FALLBACK = np.dtype

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        return DTYPE_LIKE

    def __call__(self, value: DTYPE_LIKE) -> DTYPE:
        dtype = None
        if safe_issubclass(self.origin, np.generic):
            dtype = self.origin
        if self.args:
            dtype = self.args[0]
        try:
            return asdtype(value, type=dtype)
        except (ValueError, TypeError) as e:
            raise self.value_error(
                value, f"Cannot convert value to dtype {dtype}"
            ) from e


# --- Annotated --------------------------------------------------------


@register_converter(tx.Annotated)
class ToAnnotated(Converter[TO, FROM]):

    _REGISTRY: ConverterRegistry = {}

    @classmethod
    def register(cls, *hints: tx.Unpack[tx.Tuple[tx.Any]]) -> ClassDecorator:

        def decorator(converter_cls: tx.Type[Converter]) -> tx.Type[Converter]:
            for hint in hints:
                cls._REGISTRY[hint] = converter_cls
            return converter_cls

        return decorator

    @classmethod
    def _get_converter(cls, hint: tx.Any) -> tx.Optional[tx.Type[Converter]]:
        return get_converter(hint, registry=cls._REGISTRY, fallback=None)

    @property
    def converters(self) -> tx.Tuple[Converter, ...]:
        if getattr(self, "_converters", None) is None:
            self._converters = self._get_converters()
        return self._converters

    def _get_converters(self) -> tx.Tuple[Converter, ...]:
        unwrapped = unwrap(self.hint)
        converters = []
        for arg in safe_get_args(self.hint):
            if safe_issubclass(arg, Converter):
                arg = arg(unwrapped)
            if not safe_isinstance(arg, Converter):
                # Look into annotation registry
                arg = self._get_converter(arg)
            if safe_isinstance(arg, Converter):
                if getattr(arg, "compose", False):
                    converters.append(arg)
                else:
                    converters = [arg]

        if not converters or getattr(converters[0], "compose", False):
            converters.insert(0, get_converter(unwrapped))

        return tuple(converters)

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        __reentrant = _process_reentrant(self.hint, __reentrant)
        if not __reentrant:
            return self.hint
        return self.converters[0].like(__reentrant)

    def __call__(self, value: FROM) -> TO:
        for converter in self.converters:
            # NOTE: do not catch and rethrow here. Helps with legibility.
            value = converter(value)
        return value


# ======================================================================
#       CONVERTERS TO USE AS ANNOTATED METADATA
# ======================================================================


# --- Range ------------------------------------------------------------


class ToPositive(ToNumber[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value <= 0:
            raise self.range_error(value)
        return value

    def range_error(self, value: tx.Any) -> ValueConversionError:
        message = "Expected positive value"
        return self.value_error(value, message)


class ToNegative(ToNumber[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value >= 0:
            raise self.range_error(value)
        return value

    def range_error(self, value: tx.Any) -> ValueConversionError:
        message = "Expected negative value"
        return self.value_error(value, message)


class ToNonNegative(ToNumber[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value < 0:
            raise self.range_error(value)
        return value

    def range_error(self, value: tx.Any) -> ValueConversionError:
        message = "Expected non-negative value"
        return self.value_error(value, message)


class ToNonPositive(ToNumber[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value > 0:
            raise self.range_error(value)
        return value

    def range_error(self, value: tx.Any) -> ValueConversionError:
        message = "Expected non-positive value"
        return self.value_error(value, message)


class _ComparatorConverter(ToNumber[NUMBER, NUMBER_LIKE]):

    def __init__(self, threshold: NUMBER, hint: tx.Any = UNSET) -> None:
        super().__init__(hint)
        self.threshold = threshold


class ToLessThan(_ComparatorConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value >= self.threshold:
            raise self.range_error(value)
        return value

    def range_error(self, value: tx.Any) -> ValueConversionError:
        message = f"Expected int less than  {self.threshold}"
        return self.value_error(value, message)


class ToLessEqual(_ComparatorConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value > self.threshold:
            raise self.range_error(value)
        return value

    def range_error(self, value: tx.Any) -> ValueConversionError:
        message = f"Expected int less than or equal to {self.threshold}"
        return self.value_error(value, message)


class ToGreaterThan(_ComparatorConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value <= self.threshold:
            raise self.range_error(value)
        return value

    def range_error(self, value: tx.Any) -> ValueConversionError:
        message = f"Expected int greater than {self.threshold}"
        return self.value_error(value, message)


class ToGreaterEqual(_ComparatorConverter[NUMBER, NUMBER_LIKE]):

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if value < self.threshold:
            raise self.range_error(value)
        return value

    def range_error(self, value: tx.Any) -> ValueConversionError:
        message = f"Expected int greater than or equal to {self.threshold}"
        return self.value_error(value, message)


class ToInRange(ToNumber[NUMBER, NUMBER_LIKE]):

    def __init__(
        self,
        min_value: NUMBER,
        max_value: NUMBER,
        hint: tx.Any = UNSET,
    ) -> None:
        super().__init__(hint)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, value: NUMBER_LIKE) -> NUMBER:
        value = super().__call__(value)
        if not (self.min_value <= value <= self.max_value):
            raise self.range_error(value)
        return value

    def range_error(self, value: tx.Any) -> ValueConversionError:
        mn, mx = self.min_value, self.max_value
        message = f"Expected int in range [{mn}, {mx}]."
        return self.value_error(value, message)


# --- Length ------------------------------------------------------------


class ToLength(ToSequence[ITERABLE, ITER_LIKE]):

    def __init__(
        self,
        length: int,
        hint: tx.Any = UNSET,
    ) -> None:
        super().__init__(hint)
        self.length = length

    def __call__(self, value: ITER_LIKE) -> ITERABLE:
        value = super().__call__(value)
        value = value[:self.length]
        if len(value) != self.length:
            raise self.length_error(value)
        return value

    def length_error(self, value: tx.Any) -> ValueConversionError:
        message = (f"Expected iterable of length {self.length}, "
                   f"got {len(value)}")
        return self.value_error(value, message)


# --- Regex ------------------------------------------------------------


ToAnnotated.register(re.Pattern)
class ToRegexMatch(ToString[STR, FROM]):

    def __init__(
        self, pattern: tx.Union[str, re.Pattern], hint: tx.Any = UNSET
    ) -> None:
        super().__init__(hint)
        if not safe_isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)
        self.pattern = pattern

    def __call__(self, value: tx.Any) -> STR:
        value = super().__call__(value)
        if not self.pattern.match(value):
            raise self.pattern_error(value)
        return value

    def pattern_error(self, value: tx.Any) -> ValueConversionError:
        message = f"Value does not match pattern {self.pattern.pattern!r}"
        return self.value_error(value, message)

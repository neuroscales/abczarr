__all__ = [
    "get_validator",
    "get_validator_class",
    "register_validator",
    "Validator",
    "IsAny",
    "IsNone",
    "IsUnion",
    "IsLiteral",
    "IsTypeVar",
    "IsIterable",
    "IsSequence",
    "IsMapping",
    "IsTuple",
    "IsNumber",
    "IsDType",
    "IsAnnotated",
    "IsPositive",
    "IsNegative",
    "IsNonNegative",
    "IsNonPositive",
    "IsLessThan",
    "IsLessEqual",
    "IsGreaterThan",
    "IsGreaterEqual",
    "IsInRange",
    "MatchesRegex",
    "IsNotOneOfValidator",
]

# stdlib
import numbers
import re
from collections import abc

# dependencies
import numpy as np
import typing_extensions as tx

from ._typing import (
    DTYPE,
    ITERABLE,
    MAPPING,
    NONE,
    NUMBER,
    SEQUENCE,
    STR,
    TUPLE,
    ClassDecorator,
    MagicRegistry,
    NoneType,
    T,
    UnionType,
)

# internals
from ._utils import (
    UNSET,
    HintMagic,
    MagicError,
    get_from_registry,
    safe_get_args,
    safe_get_origin,
    safe_isinstance,
    safe_issubclass,
    unwrap,
)

# ======================================================================
#       EXCEPTIONS
# ======================================================================


class ValidationError(MagicError):

    def __init__(self, *args, **kwargs) -> None:
        if "validator" in kwargs:
            kwargs["this"] = kwargs.pop("validator")
        super().__init__(*args, **kwargs)


class ValueValidationError(ValueError, ValidationError):
    ...


class TypeValidationError(TypeError, ValidationError):
    ...


# ======================================================================
#       BASE
# ======================================================================


class Validator(HintMagic[T]):
    """Base class for magic validators."""

    def __init__(self, hint: tx.Any = UNSET, compose: bool = False) -> None:
        super().__init__(hint)
        self.compose = compose

    def __call__(self, value: T) -> None:
        if not safe_isinstance(value, self.origin):
            raise self.type_error(value,  "Not a valid instance.")

    def error(
        self, value: tx.Any, message: tx.Optional[str] = None, **kwargs
    ) -> ValidationError:
        """Return a ConversionError with the given value and message."""
        type = kwargs.pop("type", ValidationError)
        type = {
            "value": ValueValidationError,
            "type": TypeValidationError
        }.get(type, type)
        kwargs.setdefault("this", self)
        kwargs.setdefault("value", value)
        if message is None:
            message = "Invalid value."
        return type(message, **kwargs)

    def type_error(
        self, value: tx.Any, message: tx.Optional[str] = None
    ) -> TypeValidationError:
        """Return a TypeValidationError with the given value."""
        if message is None:
            message = f"Invalid value type: {type(value)}"
        return self.error(value, message, type=TypeValidationError)

    def value_error(
        self, value: tx.Any, message: tx.Optional[str] = None
    ) -> ValueValidationError:
        """Return a ValueValidationError with the given value."""
        if message is None:
            message = "Invalid value."
        return self.error(value, message, type=ValueValidationError)

    def _wrap_converter(self, converter: tx.Callable) -> tx.Callable:
        """
        A wrapper that wraps a converter to catch errors and raise a
        ConversionError instead. Defined here so that subclasses to not
        need to each implement this.
        """
        return _trywrap_validator(converter, self.value_error)


ValidatorRegistry = MagicRegistry[tx.Type[Validator]]
_VALIDATORS: ValidatorRegistry = {}


def register_validator(*hints: tx.Unpack[tx.Tuple[tx.Any]]) -> ClassDecorator:
    """
    Decorator to register a validator class for one or more type hints.

    !!! example
        ```python
        @register_validator(int)
        class IntValidator(Validator[str, int]):
            def __call__(self, value: str) -> int:
                return int(value)
        ```
    """
    def decorator(cls: tx.Type[Validator]) -> tx.Type[Validator]:
        for hint in hints:
            _VALIDATORS[hint] = cls
        return cls

    return decorator


def get_validator(
    hint: tx.Any,
    registry: ValidatorRegistry = _VALIDATORS,
    fallback: tx.Optional[tx.Type[Validator]] = Validator
) -> tx.Optional[tx.Callable[[tx.Any], T]]:
    """
    Get the best-matching conversion function for a given type hint.
    """
    cls = get_validator_class(hint, registry, fallback)
    if cls is None:
        return None
    return cls(hint)


def get_validator_class(
    hint: tx.Any,
    registry: ValidatorRegistry = _VALIDATORS,
    fallback: tx.Optional[tx.Type[Validator]] = Validator
) -> tx.Type[Validator]:
    """
    Get the best-matching conversion class for a given type hint.
    """
    return get_from_registry(hint, registry) or fallback


def _trywrap_validator(
    validator: tx.Callable[[T], None], error: Exception
) -> tx.Callable[[T], None]:
    """
    Wrap a validator to catch errors and raise a ValidationError instead.
    """
    def wrapped(value: T) -> None:
        try:
            return validator(value)
        except (TypeError, ValueError) as e:
            _error = error
            if safe_issubclass(_error, Exception):
                _error = _error(value)
            raise _error from e
    return wrapped


# ======================================================================
#       IMPL
# ======================================================================


@register_validator(tx.Any)
class IsAny(Validator[T]):

    DEFAULT = tx.Any

    def __call__(self, value: T) -> None:
        return


@register_validator(NoneType)
class IsNone(Validator[NONE]):

    DEFAULT = NoneType

    def __call__(self, value: NONE) -> None:
        if value is not None:
            raise self.type_error(value, "None, None")


@register_validator(tx.Union, UnionType)
class IsUnion(Validator[T]):

    DEFAULT = tx.Union

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        UNION_TYPES = (tx.Union, UnionType)
        if safe_get_origin(self.hint, unwrap=tx.Annotated) not in UNION_TYPES:
            raise TypeError(f"{self!r}: Hint is not a Union type")
        if len(self.args) == 0:
            raise TypeError(f"{self!r}: No arguments provided")

    def __call__(self, value: T) -> None:
        errors = []
        for arg in self.args:
            try:
                validator = get_validator(arg)
                return validator(value)
            except ValidationError as e:
                errors.append(e)
                continue

        raise self.type_error(
            "Not compatible with any of the union types.",
            validator=self, parents=errors, value=value
        )


@register_validator(tx.Literal)
class IsLiteral(Validator[T]):

    DEFAULT = tx.Literal

    def __call__(self, value: T) -> None:
        if value not in self.args:
            raise self.type_error(
                value, "Not compatible with any of the literals",
            )


@register_validator(tx.TypeVar)
class IsTypeVar(Validator[T]):

    DEFAULT = tx.TypeVar("T")

    @property
    def unwrapped(self) -> tx.Any:
        return unwrap(self.hint, (tx.Annotated, tx.TypeVar))


    def __call__(self, value: T) -> None:
        return get_validator(self.unwrapped)(value)


@register_validator(abc.Iterable)
class IsIterable(Validator[ITERABLE]):

    DEFAULT = abc.Iterable

    def __call__(self, value: ITERABLE) -> None:
        super().__call__(value)  # check type
        if self.args:

            if not safe_isinstance(value, abc.Sequence):
                raise self.type_error(
                    value, "Cannot validate generator arguments",
                )

            arg_validator = get_validator(self.args[0])
            for i, item in enumerate(value):
                try:
                    arg_validator(item)
                except ValidationError as e:
                    th = {1: "st", 2: "nd", 3: "rd"}.get(i, "th")
                    raise self.value_error(
                        value, f"Iterable's {i}{th} element is not valid.",
                    ) from e


@register_validator(abc.Sequence)
class IsSequence(IsIterable[SEQUENCE]):

    DEFAULT = abc.Sequence
    FALLBACK = list


@register_validator(abc.Mapping)
class IsMapping(Validator[MAPPING]):

    DEFAULT = abc.Mapping
    FALLBACK = dict

    def __call__(self, value: MAPPING) -> None:
        super().__call__(value)  # check type
        if self.args:
            key_hint, val_hint = self.args
            key_validator = get_validator(key_hint)
            val_validator = get_validator(val_hint)
            if safe_isinstance(value, abc.Mapping):
                value = value.items()

            for k, v in value:

                try:
                    key_validator(k)
                except ValidationError as e:
                    raise self.value_error(
                        value, f"Key {k!r} is not valid.",
                    ) from e

                try:
                    val_validator(v)
                except ValidationError as e:
                    raise self.value_error(
                        value, f"At key {k!r}, value {v!r} is invalid.",
                    ) from e


@register_validator(tuple)
class IsTuple(Validator[TUPLE]):

    DEFAULT = tuple

    def __call__(self, value: TUPLE) -> None:
        if self.args:
            # If args are provided, we accept either lists or tuples.
            # This is because the tuple annotation is often used to specify
            # per-item types, but the value may be a list
            # (e.g., for JSON serialization).

            IsSequence()(value)  # check type

            if len(self.args) == 2 and self.args[1] is Ellipsis:
                arg_validator = get_validator(self.args[0])
                validators = [arg_validator] * len(value)

            else:
                if len(value) != len(self.args):
                    raise self.value_error(
                        value,
                        f"Invalid tuple length "
                        f"{len(value)!r} != {len(self.args)!r}",
                    )
                validators = map(get_validator, self.args)

            for i, (validator, val) in enumerate(zip(validators, value)):
                try:
                    validator(val)
                except ValidationError as e:
                    th = {1: "st", 2: "nd", 3: "rd"}.get(i, "th")
                    raise self.value_error(
                        value, f"Tuple's {i}{th} element is not valid.",
                    ) from e

        else:
            # If no args are provided, we do make a "strong" type check.
            super().__call__(value)  # check type


@register_validator(dict)
class IsDict(IsMapping[MAPPING]):
    # Need to register a dict validator to avoid having the TypedDict
    # validator being used for dicts.
    DEFAULT = dict


@register_validator(tx.TypedDict)
class IsTypedDict(Validator[MAPPING]):

    DEFAULT = tx.TypedDict

    def __call__(self, value: MAPPING) -> None:
        # Check type - do not use super() -> instances are not `TypedDict`
        IsMapping(dict)(value)

        # Get typeddict options
        origin = self.origin
        total = getattr(origin, "__total__", True)
        extra_items = getattr(origin, "__extra_items__", tx.Never)
        closed = getattr(origin, "__closed__", extra_items is tx.Never)
        annots = tx.get_type_hints(origin, include_extras=True)

        # Check explicitly defined keys
        for key, arg in annots.items():
            if key not in value:
                arg_origin = safe_get_origin(arg)
                if (
                    (total and arg_origin is not tx.NotRequired) or
                    (not total and arg_origin is tx.Required)
                ):
                    raise self.value_error(
                        value, f"Missing required key {key!r}"
                    )
            else:
                arg = unwrap(arg, (tx.Required, tx.NotRequired))
                validator = get_validator(arg)
                try:
                    validator(value[key])
                except ValidationError as e:
                    raise self.value_error(
                        value, f"Value for key {key!r} is not valid."
                    ) from e

        # Check extra keys
        for key, arg in value.items():
            if key not in annots:
                if closed:
                    raise self.value_error(value, f"Unexpected key {key!r}")
                validator = get_validator(extra_items)
                try:
                    validator(arg)
                except ValidationError as e:
                    raise self.value_error(
                        value, f"Value for extra key {key!r} is not valid."
                    ) from e


@register_validator(numbers.Number)
class IsNumber(Validator[NUMBER]):

    DEFAULT = numbers.Number

    def __call__(self, value: NUMBER) -> None:
        # Deal with int / float / complex differently
        # (i.e., accept int for float, and float for complex)
        if self.origin is float and isinstance(value, int):
            return
        if self.origin is complex and isinstance(value, (int, float)):
            return
        super().__call__(value)  # check type


@register_validator(np.dtype, np.generic)
class IsDType(Validator[DTYPE]):

    DEFAULT = np.dtype


@register_validator(tx.Annotated)
class IsAnnotated(Validator[T]):

    _REGISTRY: ValidatorRegistry = {}

    @classmethod
    def register(cls, *hints: tx.Unpack[tx.Tuple[tx.Any]]) -> ClassDecorator:

        def decorator(validator_cls: tx.Type[Validator]) -> tx.Type[Validator]:
            for hint in hints:
                cls._REGISTRY[hint] = validator_cls
            return validator_cls

        return decorator

    @classmethod
    def _get_validator(cls, hint: tx.Any) -> tx.Optional[tx.Type[Validator]]:
        return get_validator(hint, registry=cls._REGISTRY, fallback=None)

    @property
    def validators(self) -> tx.Tuple[Validator, ...]:
        if getattr(self, "_validators", None) is None:
            self._validators = self._get_validators()
        return self._validators

    def _get_validators(self) -> tx.Tuple[Validator, ...]:
        wrapped_type = unwrap(self.hint, tx.Annotated)
        validators = []
        for arg in safe_get_args(self.hint):
            if safe_issubclass(arg, Validator):
                arg = arg(wrapped_type)
            if not safe_isinstance(arg, Validator):
                # Look into annotation registry
                arg = self._get_validator(arg)
            if safe_isinstance(arg, Validator):
                if getattr(arg, "compose", False):
                    validators.append(arg)
                else:
                    validators = [arg]

        if not validators or getattr(validators[0], "compose", False):
            validators.insert(0, get_validator(wrapped_type))

        return tuple(validators)

    def __call__(self, value: T) -> None:
        for validator in self.validators:
            validator(value)


class IsPositive(IsNumber[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value <= 0:
            raise self.value_error(value, "Not a positive value.")


class IsNegative(IsNumber[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value >= 0:
            raise self.value_error(value, "Not a negative value.")


class IsNonNegative(IsNumber[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value < 0:
            raise self.value_error(value, "Not a non-negative value.")


class IsNonPositive(IsNumber[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value > 0:
            raise self.value_error(value, "Not a non-positive value.")


class _ComparatorValidator(IsNumber[NUMBER]):

    def __init__(self, threshold: NUMBER, hint: tx.Any = UNSET) -> None:
        super().__init__(hint)
        self.threshold = threshold


class IsLessThan(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value >= self.threshold:
            raise self.value_error(value, f"Not less than {self.threshold!r}")


class IsLessEqual(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value > self.threshold:
            raise self.value_error(
                value, f"Not less than or equal to {self.threshold!r}"
            )


class IsGreaterThan(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value <= self.threshold:
            raise self.value_error(
                value, f"Not greater than {self.threshold!r}."
            )


class IsGreaterEqual(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value < self.threshold:
            raise self.value_error(
                value, f"Not greater than or equal to {self.threshold!r}."
            )


class IsInRange(IsNumber[NUMBER]):

    def __init__(
        self,
        min_value: NUMBER,
        max_value: NUMBER,
        hint: tx.Any = UNSET,
    ) -> None:
        super().__init__(hint)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        mn, mx = self.min_value, self.max_value
        if not (mn <= value <= mx):
            raise self.value_error(value, f"Not in range [{mn!r}, {mx!r}].")


class HasLength(IsSequence[ITERABLE]):

    def __init__(
        self,
        length: int,
        hint: tx.Any = UNSET,
    ) -> None:
        super().__init__(hint)
        self.length = length

    def __call__(self, value: ITERABLE) -> None:
        super().__call__(value)
        if len(value) != self.length:
            raise self.value_error(
                value, f"Does not match expected length "
                f"{len(value)} != {self.length!r}."
            )


IsAnnotated.register(re.Pattern)
class MatchesRegex(Validator[STR]):

    DEFAULT = str

    def __init__(
        self, pattern: tx.Union[str, re.Pattern], hint: tx.Any = UNSET
    ) -> None:
        super().__init__(hint)
        if not safe_isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)
        self.pattern = pattern

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern!r})"

    def __call__(self, value: STR) -> None:
        super().__call__(value)
        if not self.pattern.match(value):
            raise self.value_error(value, "Does not match pattern.")


class IsNotOneOfValidator(Validator[T]):

    def __init__(
        self, forbidden: tx.Iterable[T], hint: tx.Any = UNSET
    ) -> None:
        super().__init__(hint)
        self.forbidden = set(forbidden)

    def __call__(self, value: T) -> None:
        super().__call__(value)
        if value in self.forbidden:
            raise self.value_error(value, "Forbidden value.")

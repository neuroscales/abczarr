__all__ = [
    "get_validator",
    "get_validator_class",
    "register_validator",
    "Validator",
    "AnyValidator",
    "NoneValidator",
    "UnionValidator",
    "LiteralValidator",
    "TypeVarValidator",
    "IterableValidator",
    "SequenceValidator",
    "MappingValidator",
    "TupleValidator",
    "NumberValidator",
    "DTypeValidator",
    "AnnotatedValidator",
    "PositiveValidator",
    "NegativeValidator",
    "NonNegativeValidator",
    "NonPositiveValidator",
    "LessThan",
    "LessEqual",
    "GreaterThan",
    "GreaterEqual",
    "RangeValidator",
    "RegexValidator",
    "NotOneOfValidator",
]

# stdlib
import numbers
import re
from collections import abc
from types import NoneType, UnionType

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from ._utils import (
    HintMagic,
    TypeVarMixin,
    _UNSET,
    get_from_registry,
    _get_args,
    _get_origin,
    _isinstance,
    _issubclass,
    _unwrap,
)
from ._typing import (
    DTYPE,
    ITERABLE,
    MAPPING,
    NONETYPE,
    NUMBER,
    SEQUENCE,
    STR,
    T,
    TUPLE,
    MagicRegistry,
    ClassDecorator,
)


# ======================================================================
#       EXCEPTIONS
# ======================================================================


class ValidationError(Exception):
    ...


class ValueValidationError(ValueError, ValidationError):
    ...


class TypeValidationError(TypeError, ValidationError):
    ...


# ======================================================================
#       BASE
# ======================================================================


class Validator(HintMagic[T]):
    """Base class for magic validators."""

    def __init__(self, hint: tx.Any = _UNSET, compose: bool = False) -> None:
        super().__init__(hint)
        self.compose = compose

    def __call__(self, value: T) -> None:
        if not _isinstance(value, self.origin):
            raise TypeValidationError(
                f"{self!r}:\n"
                f"Value {value!r} of type {type(value)} is not compatible with "
                f"the expected type {self.origin!r}"
            )


_VALIDATORS: MagicRegistry[Validator] = {}


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
    registry: MagicRegistry[Validator] = _VALIDATORS,
    fallback: tx.Optional[tx.Type[Validator]] = Validator
) -> tx.Optional[tx.Callable[[tx.Any], T]]:
    """
    Get the best-matching conversion function for a given type hint.
    """
    cls = get_validator_class(hint, registry, fallback)
    if cls is None:
        return None
    obj = cls(hint)
    return obj


def get_validator_class(
    hint: tx.Any,
    registry: MagicRegistry[Validator] = _VALIDATORS,
    fallback: tx.Optional[tx.Type[Validator]] = Validator
) -> tx.Type[Validator]:
    """
    Get the best-matching conversion class for a given type hint.
    """
    return get_from_registry(hint, registry) or fallback


# ======================================================================
#       IMPL
# ======================================================================


@register_validator(tx.Any)
class AnyValidator(Validator[T]):

    DEFAULT = tx.Any

    def __call__(self, value: T) -> None:
        return


@register_validator(NoneType)
class NoneValidator(Validator[NONETYPE]):

    DEFAULT = NoneType

    def __call__(self, value: NONETYPE) -> None:
        if value is not None:
            raise TypeValidationError(f"{self!r}: Value {value!r} is not None")


@register_validator(tx.Union, UnionType)
class UnionValidator(Validator[T]):

    DEFAULT = tx.Union

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        if _get_origin(self.hint, unwrap=tx.Annotated) not in (tx.Union, UnionType):
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

        message = (
            f"{self!r}:\n"
            f"Value {value!r} of type {type(value)} is not compatible with "
            f"any of the union types."
        )
        for e in errors:
            message += f"\n?> {str(e)}"
        raise TypeValidationError(message)


@register_validator(tx.Literal)
class LiteralValidator(Validator[T]):

    DEFAULT = tx.Literal

    def __call__(self, value: T) -> None:
        if value not in self.args:
            raise TypeValidationError(
                f"{self!r}:\n"
                f"Value {value!r} is not compatible with any of the literal "
                f"types"
            )


@register_validator(tx.TypeVar)
class TypeVarValidator(TypeVarMixin, Validator[T]):

    DEFAULT = tx.TypeVar("T")

    def __call__(self, value: T) -> None:
        return get_validator(self.fallback)(value)


@register_validator(abc.Iterable)
class IterableValidator(Validator[ITERABLE]):

    DEFAULT = abc.Iterable

    def __call__(self, value: ITERABLE) -> None:
        super().__call__(value)  # check type
        if self.args:
            if not _isinstance(value, abc.Sequence):
                raise TypeValidationError(
                    f"{self!r}: Cannot validate generator arguments"
                )
            arg_validator = get_validator(self.args[0])
            list(map(arg_validator, value))


@register_validator(abc.Sequence)
class SequenceValidator(IterableValidator[SEQUENCE]):

    DEFAULT = abc.Sequence
    FALLBACK = list


@register_validator(abc.Mapping)
class MappingValidator(Validator[MAPPING]):

    DEFAULT = abc.Mapping
    FALLBACK = dict

    def __call__(self, value: MAPPING) -> None:
        super().__call__(value)  # check type
        if self.args:
            key_hint, val_hint = self.args
            key_validator = get_validator(key_hint)
            val_validator = get_validator(val_hint)
            if _isinstance(value, abc.Mapping):
                value = value.items()

            for k, v in value:

                try:
                    key_validator(k)
                except ValidationError as e:
                    raise type(e)(
                        f"{self!r}: Key {k!r} of type {type(k)} is not "
                        f"compatible with the expected key type {key_hint}.\n"
                        f"-> {str(e)}"
                    ) from e

                try:
                    val_validator(v)
                except ValidationError as e:
                    raise type(e)(
                        f"{self!r}: Value {v!r} of type {type(v)} is not "
                        f"compatible with the expected value type {val_hint}.\n"
                        f"-> {str(e)}"
                    ) from e



@register_validator(tuple)
class TupleValidator(Validator[TUPLE]):

    DEFAULT = tuple

    def __call__(self, value: TUPLE) -> None:
        super().__call__(value)  # check type
        if self.args:

            if len(self.args) == 2 and self.args[1] is Ellipsis:
                arg_validator = get_validator(self.args[0])
                validators = [arg_validator] * len(value)

            else:
                if len(value) != len(self.args):
                    raise ValueValidationError(
                        f"{self!r}: Value {value!r} does not match the "
                        f"expected tuple length {len(self.args)!r}"
                    )
                validators = map(get_validator, self.args)

            for i, (validator, val) in enumerate(zip(validators, value)):
                try:
                    validator(val)
                except ValidationError as e:
                    raise type(e)(
                        f"{self!r}:\n"
                        f"Tuple's {i}th element {value!r} is not compatible "
                        f"with the expected type.\n"
                        f"-> {str(e)}"
                    ) from e


@register_validator(tx.TypedDict)
class TypedDictValidator(Validator[MAPPING]):

    DEFAULT = tx.TypedDict

    def __call__(self, value: MAPPING) -> None:
        # Check type - do not use super() -> instances are not `TypedDict`
        MappingValidator(dict)(value)

        # Get typeddict options
        origin = self.origin
        total = getattr(origin, "__total__", True)
        extra_items = getattr(origin, "__extra_items__", tx.Never)
        closed = getattr(origin, "__closed__", extra_items is tx.Never)
        annots = tx.get_type_hints(origin, include_extras=True)

        # Check explicitely defined keys
        for key, arg in annots.items():
            if key not in value:
                arg_origin = _get_origin(arg)
                if (
                    (total and arg_origin is not tx.NotRequired) or
                    (not total and arg_origin is tx.Required)
                ):
                    raise ValueValidationError(
                        f"{self!r}:\n"
                        f"Missing required key {key!r} in value {value!r}"
                    )
            else:
                arg = _unwrap(arg, (tx.Required, tx.NotRequired))
                validator = get_validator(arg)
                try:
                    validator(value[key])
                except ValidationError as e:
                    raise type(e)(
                        f"{self!r}:\n"
                        f"Key {key!r} with value {value[key]!r} is not "
                        f"compatible with the expected type {arg!r}.\n"
                        f"-> {str(e)}"
                    ) from e

        # Check extra keys
        for key, arg in value.items():
            if key not in annots:
                if closed:
                    raise ValueValidationError(
                        f"{self!r}:\n"
                        f"Unexpected key {key!r} in value {value!r}"
                    )
                validator = get_validator(extra_items)
                try:
                    validator(arg)
                except ValidationError as e:
                    raise type(e)(
                        f"{self!r}:\n"
                        f"Key {key!r} with value {value[key]!r} is not "
                        f"compatible with the expected type {extra_items!r}.\n"
                        f"-> {str(e)}"
                    ) from e


@register_validator(numbers.Number)
class NumberValidator(Validator[NUMBER]):

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
class DTypeValidator(Validator[DTYPE]):

    DEFAULT = np.dtype


@register_validator(tx.Annotated)
class AnnotatedValidator(Validator[T]):

    _REGISTRY: MagicRegistry[Validator] = {}

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
        wrapped_type = _unwrap(self.hint, tx.Annotated)
        validators = []
        for arg in _get_args(self.hint):
            if _issubclass(arg, Validator):
                arg = arg(wrapped_type)
            if not _isinstance(arg, Validator):
                # Look into annotation registry
                arg = self._get_validator(arg)
            if _isinstance(arg, Validator):
                if getattr(arg, "compose", False):
                    validators.append(arg)
                else:
                    validators = [arg]

        if not validators or getattr(validators[0], "compose", False):
            validators.insert(0, get_validator(wrapped_type))

        assert len(validators) > 0, f"Cannot get validators for hint {self.hint}"
        return tuple(validators)

    def __call__(self, value: T) -> None:
        for validator in self.validators:
            try:
                validator(value)
            except ValidationError as e:
                raise type(e)(
                    f"{self!r}:\n"
                    f"Value {value!r} is not compatible with the expected type.\n"
                    f"-> {str(e)}"
                ) from e


class PositiveValidator(NumberValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value <= 0:
            raise ValueValidationError(
                f"{self!r}: Value {value!r} is not a positive int."
            )


class NegativeValidator(NumberValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value >= 0:
            raise ValueValidationError(
                f"{self!r}: Value {value!r} is not a negative int."
            )


class NonNegativeValidator(NumberValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value < 0:
            raise ValueValidationError(
                f"{self!r}: Value {value!r} is not a non-negative int."
            )


class NonPositiveValidator(NumberValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value > 0:
            raise ValueValidationError(
                f"{self!r}: Value {value!r} is not a non-positive int."
            )


class _ComparatorValidator(NumberValidator[NUMBER]):

    def __init__(self, threshold: NUMBER, hint: tx.Any = _UNSET) -> None:
        super().__init__(hint)
        self.threshold = threshold


class LessThan(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value >= self.threshold:
            raise ValueValidationError(
                f"{self!r}: "
                f"Value {value!r} is not less than {self.threshold!r}"
            )


class LessEqual(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value > self.threshold:
            raise ValueValidationError(
                f"{self!r}: Value {value!r} is not less than or equal "
                f"to {self.threshold!r}"
            )


class GreaterThan(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value <= self.threshold:
            raise ValueValidationError(
                f"{self!r}: "
                f"Value {value!r} is not greater than {self.threshold!r}."
            )


class GreaterEqual(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value < self.threshold:
            raise ValueValidationError(
                f"{self!r}: Value {value!r} is not greater than or equal to "
                f"{self.threshold!r}."
            )


class RangeValidator(NumberValidator[NUMBER]):

    def __init__(
        self,
        min_value: NUMBER,
        max_value: NUMBER,
        hint: tx.Any = _UNSET,
    ) -> None:
        super().__init__(hint)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if not (self.min_value <= value <= self.max_value):
            raise ValueValidationError(
                f"{self!r}: Value {value!r} is not in range."
            )


class LengthValidator(SequenceValidator[ITERABLE]):

    def __init__(
        self,
        length: int,
        hint: tx.Any = _UNSET,
    ) -> None:
        super().__init__(hint)
        self.length = length

    def __call__(self, value: ITERABLE) -> None:
        super().__call__(value)
        if len(value) != self.length:
            raise ValueValidationError(
                f"{self!r}: Sequence {value!r} does not match expected length."
            )


AnnotatedValidator.register(re.Pattern)
class RegexValidator(Validator[STR]):

    DEFAULT = str

    def __init__(
        self, pattern: tx.Union[str, re.Pattern], hint: tx.Any = _UNSET
    ) -> None:
        super().__init__(hint)
        if not _isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)
        self.pattern = pattern

    def __call__(self, value: STR) -> None:
        super().__call__(value)
        if not self.pattern.match(value):
            raise ValueValidationError(
                f"{self!r}: Value {value!r} does not match pattern."
            )


class NotOneOfValidator(Validator[T]):

    def __init__(
        self, forbidden: tx.Iterable[T], hint: tx.Any = _UNSET
    ) -> None:
        super().__init__(hint)
        self.forbidden = set(forbidden)

    def __call__(self, value: T) -> None:
        super().__call__(value)
        if value in self.forbidden:
            raise ValueValidationError(f"{self!r}: Value {value!r} is forbidden.")

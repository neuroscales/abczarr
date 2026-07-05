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

# dependencies
import numpy as np
import typing_extensions as tx

from ._typing import (
    DTYPE,
    ITERABLE,
    MAPPING,
    NONETYPE,
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
    _UNSET,
    HintMagic,
    TypeVarMixin,
    _get_args,
    _get_origin,
    _isinstance,
    _issubclass,
    _unwrap,
    get_from_registry,
)

# ======================================================================
#       EXCEPTIONS
# ======================================================================


class ValidationError(Exception):

    def __init__(self, *args, **kwargs) -> None:
        validator = kwargs.pop("validator", None)
        parents = kwargs.pop("parents", None)
        value = kwargs.pop("value", None)
        super().__init__(*args, **kwargs)
        if parents is None:
            parents = ()
        elif not isinstance(parents, (list, tuple)):
            parents = (parents,)
        self.parents = tuple(parents)
        self.validator = validator
        self.value = value

    @property
    def message(self) -> str:
        return super().__str__()

    @property
    def depth(self) -> int:
        return 1 + max(getattr(p, "depth", 0) for p in self.parents)

    @property
    def best_parent(self) -> tx.Optional[tx.Self]:
        return max(
            self.parents,
            key=lambda p: getattr(p, "depth", 0),
            default=None
        )

    def _make_str(
        self,
        validator: bool = True,
        value: bool = True,
        parents: bool = True
    ) -> str:
        message = self.message or ""
        if validator:
            if message:
                message = f"{self.validator!r}: {message}"
            else:
                message = f"{self.validator!r}"
        if value:
            message = f"{message}\n=> value: {self.value!r}"
        if parents and self.parents:
            arrow = "?> " if len(self.parents) > 1 else "->"
            value = len(self.parents) == 1
            for parent in self.parents:
                parent_message = parent._make_str(
                    validator=validator, value=value
                )
                message = f"{message}\n{arrow} {parent_message}"
        return message

    def __str__(self) -> str:
        return self._make_str()


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
                "Not a valid instance.",
                validator=self, value=value
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
            raise TypeValidationError("Not None", validator=self, value=value)


@register_validator(tx.Union, UnionType)
class UnionValidator(Validator[T]):

    DEFAULT = tx.Union

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        UNION_TYPES = (tx.Union, UnionType)
        if _get_origin(self.hint, unwrap=tx.Annotated) not in UNION_TYPES:
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

        raise TypeValidationError(
            "Not compatible with any of the union types.",
            validator=self, parents=errors, value=value
        )


@register_validator(tx.Literal)
class LiteralValidator(Validator[T]):

    DEFAULT = tx.Literal

    def __call__(self, value: T) -> None:
        if value not in self.args:
            raise TypeValidationError(
                "Not compatible with any of the literals",
                validator=self, value=value
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
                    "Cannot validate generator arguments",
                    validator=self, value=value
                )

            arg_validator = get_validator(self.args[0])
            for i, item in enumerate(value):
                try:
                    arg_validator(item)
                except ValidationError as e:
                    th = {1: "st", 2: "nd", 3: "rd"}.get(i, "th")
                    raise type(e)(
                        f"Iterable's {i}{th} element is not valid.",
                        validator=self, parents=e, value=value
                    ) from e


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
                        f"Key {k!r} has invalid type {type(k)!r}.",
                        validator=self, parents=e, value=value
                    ) from e

                try:
                    val_validator(v)
                except ValidationError as e:
                    raise type(e)(
                        f"At key {k!r}, value {v!r} is invalid.",
                        validator=self, parents=e, value=value
                    ) from e



@register_validator(tuple)
class TupleValidator(Validator[TUPLE]):

    DEFAULT = tuple

    def __call__(self, value: TUPLE) -> None:
        if self.args:
            # If args are provided, we accept either lists or tuples.
            # This is because the tuple annotation is often used to specify
            # per-item types, but the value may be a list
            # (e.g., for JSON serialization).

            SequenceValidator()(value)  # check type

            if len(self.args) == 2 and self.args[1] is Ellipsis:
                arg_validator = get_validator(self.args[0])
                validators = [arg_validator] * len(value)

            else:
                if len(value) != len(self.args):
                    raise ValueValidationError(
                        f"Invalid tuple length "
                        f"{len(value)!r} != {len(self.args)!r}",
                        validator=self, value=value
                    )
                validators = map(get_validator, self.args)

            for i, (validator, val) in enumerate(zip(validators, value)):
                try:
                    validator(val)
                except ValidationError as e:
                    th = {1: "st", 2: "nd", 3: "rd"}.get(i, "th")
                    raise type(e)(
                        f"Tuple's {i}{th} element is not valid.",
                        validator=self, parents=e, value=value
                    ) from e

        else:
            # If no args are provided, we do make a "strong" type check.
            super().__call__(value)  # check type


@register_validator(dict)
class DictValidator(MappingValidator[MAPPING]):
    # Need to register a dict validator to avoid having the TypedDict
    # validator being used for dicts.
    DEFAULT = dict


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
        if origin.__name__ == "Metadata":
            print(origin, total, extra_items, closed)

        # Check explicitly defined keys
        for key, arg in annots.items():
            if key not in value:
                arg_origin = _get_origin(arg)
                if (
                    (total and arg_origin is not tx.NotRequired) or
                    (not total and arg_origin is tx.Required)
                ):
                    raise ValueValidationError(
                        f"Missing required key {key!r}",
                        validator=self, value=value
                    )
            else:
                arg = _unwrap(arg, (tx.Required, tx.NotRequired))
                validator = get_validator(arg)
                try:
                    validator(value[key])
                except ValidationError as e:
                    raise type(e)(
                        f"Value for key {key!r} is not valid.",
                        validator=self, parents=e, value=value
                    ) from e

        # Check extra keys
        for key, arg in value.items():
            if key not in annots:
                if closed:
                    raise ValueValidationError(
                        f"Unexpected key {key!r}",
                        validator=self, value=value
                    )
                validator = get_validator(extra_items)
                try:
                    validator(arg)
                except ValidationError as e:
                    raise type(e)(
                        f"Value for extra key {key!r} is not valid.",
                        validator=self, parents=e, value=value
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

        return tuple(validators)

    def __call__(self, value: T) -> None:
        for validator in self.validators:
            validator(value)


class PositiveValidator(NumberValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value <= 0:
            raise ValueValidationError(
                "Not a positive int.",
                validator=self, value=value
            )


class NegativeValidator(NumberValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value >= 0:
            raise ValueValidationError(
                "Not a negative int.",
                validator=self, value=value
            )


class NonNegativeValidator(NumberValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value < 0:
            raise ValueValidationError(
                "Not a non-negative int.",
                validator=self, value=value
            )


class NonPositiveValidator(NumberValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value > 0:
            raise ValueValidationError(
                "Not a non-positive int.",
                validator=self, value=value
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
                f"Not less than {self.threshold!r}",
                validator=self, value=value
            )


class LessEqual(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value > self.threshold:
            raise ValueValidationError(
                f"Not less than or equal to {self.threshold!r}",
                validator=self, value=value
            )


class GreaterThan(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value <= self.threshold:
            raise ValueValidationError(
                f"Not greater than {self.threshold!r}.",
                validator=self, value=value
            )


class GreaterEqual(_ComparatorValidator[NUMBER]):

    def __call__(self, value: NUMBER) -> None:
        super().__call__(value)
        if value < self.threshold:
            raise ValueValidationError(
                f"Not greater than or equal to {self.threshold!r}.",
                validator=self, value=value
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
                f"Not in range [{self.min_value!r}, {self.max_value!r}].",
                validator=self, value=value
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
                f"Does not match expected length "
                f"{len(value)} != {self.length!r}.",
                validator=self, value=value
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern!r})"

    def __call__(self, value: STR) -> None:
        super().__call__(value)
        if not self.pattern.match(value):
            raise ValueValidationError(
                "Does not match pattern.",
                validator=self, value=value
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
            raise ValueValidationError(
                "Forbidden value.", validator=self, value=value
            )

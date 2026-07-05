# stdlib
import inspect
import math
import numbers
from collections import abc

# dependencies
import numpy as np
import typing_extensions as tx

# locals
from ._typing import NoneType, T, UnionType

# constants
_UNSET = object()


class HintMagic(tx.Generic[T]):
    """Base class for magic objects (factories, converters)."""

    DEFAULT = tx.Any
    FALLBACK = _UNSET

    def __init__(self, hint: tx.Any = _UNSET) -> None:
        if hint is _UNSET:
            hint = self.DEFAULT
        self.hint = hint

    @property
    def origin(self) -> tx.Any:
        if getattr(self, "_origin", None) is None:
            self._origin = self._get_origin()
        return self._origin

    def _get_origin(self) -> tx.Any:
        return _get_origin(self.hint, unwrap=tx.Annotated)

    @property
    def args(self) -> tx.Tuple[tx.Any, ...]:
        if getattr(self, "_args", None) is None:
            self._args = self._get_args()
        return self._args

    def _get_args(self) -> tx.Tuple[tx.Any, ...]:
        return _get_args(self.hint, unwrap=tx.Annotated)

    @property
    def fallback(self) -> tx.Any:
        if getattr(self, "_fallback", None) is None:
            self._fallback = self._get_fallback()
        return self._fallback

    def _get_fallback(self) -> tx.Any:
        return get_type(self.hint, self.FALLBACK)

    def __call__(self, *args, **kwargs) -> T:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __call__"
        )

    def __repr__(self) -> str:
        hint_arg = self.hint if self.hint != self.DEFAULT else ""
        return f"{type(self).__name__}({hint_arg})"

    def __str__(self) -> str:
        return repr(self)


class TypeVarMixin:

    def _get_fallback(self) -> tx.Any:
        return _typevar_fallback(self.hint)


def _typevar_fallback(hint: tx.Any) -> tx.Any:
    origin = _get_origin(hint, unwrap=tx.Annotated)
    if not _isinstance(origin, tx.TypeVar):
        return hint
    if getattr(origin, "__default__", tx.NoDefault) is not tx.NoDefault:
        return origin.__default__
    if getattr(origin, "__constraints__", ()):
        return tx.Union[origin.__constraints__]
    if getattr(origin, "__bound__", None) is not None:
        return origin.__bound__
    return tx.Any


def get_type(hint: tx.Any, fallback: type = _UNSET) -> tx.Type[tx.Any]:
    """Get a valid concrete type from a type hint."""
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
    Get a default value from a type hint.

    If the hint is a `Literal` (or a `Union` that contains a `Literal`),
    the first value in the literal is returned.

    If the hint is a `Union` that contains `NoneType`, `None` is returned.

    Otherwise, if the hint is a `Union`, we recurse through its sub-hints.

    If no default value can be found, a `TypeError` is raised.
    A factory should then be used.
    """
    origin = _get_origin(hint, unwrap=tx.Annotated)
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


def get_from_registry(hint: tx.Any, registry: dict) -> tx.Any:
    """
    Get the best matching value from a registry whose keys are types or
    type hints.
    """
    best_match, best_dist = _get_best_match(hint, registry)

    if best_dist != 0 and _get_origin(hint) is tx.Annotated:
        hint = _get_origin(hint, unwrap=tx.Annotated)
        better_match, better_dist = _get_best_match(hint, registry)
        if better_dist < best_dist:
            best_match, best_dist = better_match, better_dist

    if best_match is not None:
        return registry[best_match]

    return None


def _get_best_match(hint: tx.Any, registry: dict) -> tx.Tuple[tx.Any, float]:
    """
    Get the best matching value from a registry whose keys are types or
    type hints, and return the key and value as a tuple.
    """
    hint = _get_origin(hint)

    best_match, best_dist = None, float("inf")
    for key in registry:

        dist = _type_dist(hint, key)

        if dist == 0:
            # Perfect match -> stop here
            best_match, best_dist = key, dist
            break

        elif dist < best_dist:
            if _is_typeddict(best_match) and not _is_typeddict(key):
                # Prefer typeddict over other types if they are compatible
                continue
            else:
                # Update best match
                best_match, best_dist = key, dist

        elif dist < float('inf') and _is_typeddict(key):
            # Prefer typeddict over other types if they are compatible
            best_match, best_dist = key, dist

        elif dist == best_dist:
            if _issubclass(key, best_match):
                # Prefer more specific subclass
                best_match = key

    return best_match, best_dist


def _type_dist(subcls: type, cls: type) -> int:
    """Distance between two types, based on their inheritance hierarchy."""
    if _isinstance(subcls, tx.TypeVar):
        subcls = tx.TypeVar
    if subcls is cls:
        return 0
    if not _issubclassable(subcls) or not _issubclassable(cls):
        return float("inf")
    if not _issubclass(subcls, cls):
        return float("inf")
    if tx.is_typeddict(cls):
        bases = _all_orig_bases(subcls)
    else:
        bases = subcls.__mro__
    distance = 0
    for base in bases:
        if base is cls:
            return distance
        distance += 1
    return 1000


def _issubclassable(cls: tx.Any) -> bool:
    if cls is tx.TypedDict:
        return True
    return isinstance(cls, type)


def _is_typeddict(cls: tx.Any) -> bool:
    if cls is tx.TypedDict:
        return True
    return tx.is_typeddict(cls)


def _all_orig_bases(cls: type, _self: bool = True) -> tx.Tuple[type, ...]:
    """Get all original bases of a type, including the type itself."""
    if not _is_typeddict(cls):
        return ()
    bases = (cls,) if _self else ()
    bases += getattr(cls, '__orig_bases__', ())
    for base in getattr(cls, '__orig_bases__', ()):
        bases += _all_orig_bases(base, _self=False)
    return bases


def _issubclass(subcls: tx.Any, cls: tx.Any) -> bool:
    """Safe subclass (does not fail if arguments are not types)."""
    if _is_typeddict(cls):
        return cls in _all_orig_bases(subcls) or subcls is dict
    if isinstance(subcls, type) and isinstance(cls, type):
        return issubclass(subcls, cls)
    return False


def _isinstance(obj: tx.Any, cls: tx.Any) -> bool:
    """Safe isinstance (does not fail if second argument is not a type)."""
    if _is_typeddict(cls):
        return _issubclass(type(obj), cls)
    if isinstance(cls, type) and cls is not tx.Any:
        return isinstance(obj, cls)
    return False


def _unwrap(hint: tx.Any, unwrap: tx.Any = (tx.Annotated,)) -> tx.Any:
    """
    Unwrap a type hint from its origin, if it is in the unwrap list.

    !!! example
        ```python
        _unwrap(Annotated[int, "metadata"], unwrap=Annotated)   # returns int
        _unwrap(Optional[int], unwrap=Optional)                 # returns int
        ```
    """
    origin = _get_origin(hint)
    if unwrap is not None:
        if not isinstance(unwrap, abc.Sequence):
            unwrap = (unwrap,)
        if origin in unwrap:
            return _unwrap(tx.get_args(hint)[0], unwrap=unwrap)
    return hint


def _get_origin(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Any:
    """
    Safe `get_origin`.

    Returns the input type, instead of `None`, if the input is not a
    generic type. Can also unwrap some hints (e.g. `Annotated`) if asked.
    """
    if unwrap:
        hint = _unwrap(hint, unwrap=unwrap)
    origin = tx.get_origin(hint)
    if origin is None:
        return hint
    return origin


def _get_args(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Tuple[tx.Any, ...]:
    """
    Safe `get_args`.

    Returns an empty tuple if the input is not a generic type.
    Can also unwrap some hints (e.g. `Annotated`) if asked.
    """
    hint = _unwrap(hint, unwrap=unwrap)
    origin = tx.get_origin(hint)
    if origin is None:
        return ()
    return tx.get_args(hint)


def eq_safenan(x: tx.Any) -> bool:
    """
    Safe equality comparison that treats NaN as equal to NaN.
    """
    if isinstance(x, (numbers.Real, np.floating)) and math.isnan(x):
        return "NaN"
    return x

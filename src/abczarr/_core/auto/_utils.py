"""Deriving a default value from a type hint, and NaN-tolerant equality."""

__all__ = [
    "get_default",
    "eq_safenan",
]

# stdlib
import math
import numbers

# dependencies
import numpy as np
import typing_extensions as tx

# locals
from ..rfc2119 import MUST, Requirement
from ._typing import NoneType, UnionType


def _unwrap_annotated(hint: tx.Any) -> tx.Any:
    """Strip any `Annotated[...]` layers, returning the wrapped hint."""
    while tx.get_origin(hint) is tx.Annotated:
        hint = tx.get_args(hint)[0]
    return hint


def _permits_absence(hint: tx.Any) -> bool:
    """Whether *hint* carries a `Requirement` that lets the field be
    absent.

    A `Recommended`, `Optional`, or other non-`Required` requirement
    level means the field may be unset, so `get_default` derives no
    value from a `Literal` or `Optional` in the hint. The requirement
    factory yields `MISSING` for it instead.
    """
    while tx.get_origin(hint) is tx.Annotated:
        args = tx.get_args(hint)
        for meta in args[1:]:
            if isinstance(meta, Requirement):
                return meta is not MUST
        hint = args[0]
    return False


def get_default(hint: tx.Any) -> tx.Any:
    """Derive a default value from a type hint, or raise `TypeError`
    when none can be derived.

    A hint carrying a non-`Required` `Requirement` (`Recommended`,
    `Optional`, and so on) always raises, since an unset optional field
    is absent rather than defaulted to an invented value. A `Union`
    hint containing `NoneType` defaults to `None`. A `Literal` hint
    defaults to its first value. Any other `Union` hint is tried
    sub-hint by sub-hint, defaulting to the first one that itself
    yields a default. Any other hint raises `TypeError`, since it names
    no value a default could be derived from. A factory should be used
    for it instead.
    """
    if _permits_absence(hint):
        raise TypeError(
            f"optional requirement has no derived default: {hint}"
        )
    hint = _unwrap_annotated(hint)
    origin = tx.get_origin(hint)
    args = tx.get_args(hint)
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


def eq_safenan(x: tx.Any) -> tx.Any:
    """Map a NaN value to a value that compares equal to itself, for use
    as an `attrs` field's `eq` key.

    A real or floating-point NaN, which is never equal to itself under
    ordinary comparison, is mapped to the string `"NaN"`, so a field
    compared through this function treats two NaN values as equal. Any
    other value is returned unchanged.
    """
    if isinstance(x, (numbers.Real, np.floating)) and math.isnan(x):
        return "NaN"
    return x

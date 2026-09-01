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
from ._typing import NoneType, UnionType


def _unwrap_annotated(hint: tx.Any) -> tx.Any:
    """Strip any ``Annotated[...]`` layers, returning the wrapped hint."""
    while tx.get_origin(hint) is tx.Annotated:
        hint = tx.get_args(hint)[0]
    return hint


def get_default(hint: tx.Any) -> tx.Any:
    """
    Get a default value from a type hint.

    * If the hint is a `Union` that contains `NoneType`, `None` is returned.
    * If the hint is a `Literal`, the first value in the literal is returned.
    * Otherwise, if the hint is a `Union`, we recurse through its sub-hints.
    * If no default value can be found, a `TypeError` is raised.
      A factory should then be used.
    """
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
    """
    Safe equality comparison that treats NaN as equal to NaN.
    """
    if isinstance(x, (numbers.Real, np.floating)) and math.isnan(x):
        return "NaN"
    return x

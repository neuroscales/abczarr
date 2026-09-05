"""
Invariant TypeVars.

An invariant parameter *ignores* the subtype relation of its argument:
even though `bool` is a subtype of `int`, `Box[bool]` is neither usable
where `Box[int]` is expected nor the other way round. This is the only
sound variance when the parameter appears in both input and output
positions, as it does for anything mutable.

Invariance is the default for `#!python tx.TypeVar`, so these are
declared without a variance keyword.
"""

__all__ = [
    "T",
]

import typing_extensions as tx

T = tx.TypeVar("T")
"""An invariant TypeVar for the element of a one-or-many hint."""

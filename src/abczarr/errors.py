"""The errors abczarr raises, gathered in one place.

Every error is defined in ``abczarr._core.errors`` at the bottom of the
import graph and re-exported here so callers can reach them as
``abczarr.errors.UnsupportedZarrOperation`` (or straight off the
package, ``abczarr.UnsupportedZarrOperation``).
"""

from ._core.errors import (
    TransactionConflict,
    UnsupportedConversion,
    UnsupportedZarrOperation,
)

__all__ = [
    "UnsupportedZarrOperation",
    "UnsupportedConversion",
    "TransactionConflict",
]

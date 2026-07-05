__all__ = [
    "v0_1",
    "v0_2",
    "v0_3",
    "v0_4",
    "v0_5",
    "v0_6dev4",
    "v0_6rc0",
    "OMESchemaItem",
]


import typing_extensions as tx

from . import (
    v0_1,
    v0_2,
    v0_3,
    v0_4,
    v0_5,
    v0_6dev4,
    v0_6rc0,
)
from .base import OMESchemaItem

OMEAttributes = tx.Union[
    v0_1.OMEAttributes,
    v0_2.OMEAttributes,
    v0_3.OMEAttributes,
    v0_4.OMEAttributes,
    v0_5.OMEAttributes,
    v0_6dev4.OMEAttributes,
    v0_6rc0.OMEAttributes,
]

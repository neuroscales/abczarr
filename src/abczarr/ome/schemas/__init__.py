__all__ = [
    "v0_1",
    "v0_2",
    "v0_3",
    "v0_4",
    "v0_5",
    "v0_6dev1",
    "v0_6dev2",
    "v0_6dev3",
    "v0_6dev4",
    "v0_6rc0",
    "OMESchemaItem",
    # JSON-schema validation (offline, official NGFF schemas)
    "VERSIONS",
    "documents",
    "get_validator",
    "validate",
]


import typing_extensions as tx

from . import (
    v0_1,
    v0_2,
    v0_3,
    v0_4,
    v0_5,
    v0_6dev1,
    v0_6dev2,
    v0_6dev3,
    v0_6dev4,
    v0_6rc0,
)
from ._validation import VERSIONS, documents, get_validator, validate
from .base import OMESchemaItem

OMEAttributes = tx.Union[
    v0_1.OMEAttributes,
    v0_2.OMEAttributes,
    v0_3.OMEAttributes,
    v0_4.OMEAttributes,
    v0_5.OMEAttributes,
    v0_6dev1.OMEAttributes,
    v0_6dev2.OMEAttributes,
    v0_6dev3.OMEAttributes,
    v0_6dev4.OMEAttributes,
    v0_6rc0.OMEAttributes,
]

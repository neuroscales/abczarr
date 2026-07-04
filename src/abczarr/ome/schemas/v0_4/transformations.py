__all__ = [
    "CoordinateTransformation",
    "Translation",
    "Scale",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem

# typing
Required = RequirementForTypedDict.Required
List = tz.BuiltinSequence  # list | tuple


class CoordinateTransformation(OMESchemaItem):
    type: Required[str]


class Translation(CoordinateTransformation):
    type: Required[tx.Literal["translation"]]
    translation: Required[List[float]]


class Scale(CoordinateTransformation):
    type: Required[tx.Literal["scale"]]
    scale: Required[List[float]]

__all__ = ["Scene"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem
from .transformations import CoordinateTransformationBase


# typing
Required = RequirementForTypedDict.Required
List = tz.BuiltinSequence  # list | tuple


class Scene(OMESchemaItem):
    coordinateTransformations: Required[List[CoordinateTransformationBase]]

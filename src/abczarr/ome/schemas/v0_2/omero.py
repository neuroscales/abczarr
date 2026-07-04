__all__ = ["Omero", "Channel"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem
from .version import Version

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
List = tz.BuiltinSequence  # list | tuple


class Channel(OMESchemaItem):


    class Window(OMESchemaItem):
        min: Required[float]
        max: Required[float]
        start: Required[float]
        end: Required[float]


    color: Required[str]
    window: Required[Window]


class Omero(OMESchemaItem):
    channels: Required[List[Channel]]
    version: Recommended[Version]

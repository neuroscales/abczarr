__all__ = ["Omero", "Channel"]

# dependencies

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem, ome_schema_opt

# typing
Required = RequirementForTypedDict.Required
List = tz.BuiltinSequence  # list | tuple


class Channel(OMESchemaItem, **ome_schema_opt):

    class Window(OMESchemaItem, **ome_schema_opt):
        min: Required[float]
        max: Required[float]
        start: Required[float]
        end: Required[float]

    color: Required[str]
    window: Required[Window]


class Omero(OMESchemaItem, **ome_schema_opt):
    channels: Required[List[Channel]]

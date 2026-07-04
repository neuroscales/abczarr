__all__ = ["Well"]

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
Optional = RequirementForTypedDict.Optional
List = tz.BuiltinSequence  # list | tuple


class Well(OMESchemaItem):

    class Image(OMESchemaItem):
        path: Required[str]
        acquisition: Recommended[int]

    images: Required[List[Image]]
    version: Recommended[Version]

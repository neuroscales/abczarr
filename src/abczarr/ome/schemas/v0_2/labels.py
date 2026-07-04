__all__ = ["ImageLabel"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.validators import RangeValidator
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem
from .version import Version

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
UInt8 = tx.Annotated[int, RangeValidator(0, 255)]
List = tz.BuiltinSequence  # list | tuple


class ImageLabel(OMESchemaItem):

    class Color(OMESchemaItem):
        __annotations__ = {
            "label-value": Required[int],
            "rgba": Optional[tx.Tuple[UInt8, UInt8, UInt8, UInt8]],
        }

    class Property(OMESchemaItem):
        __annotations__ = {
            "label-value": Required[int],
        }

    class Source(OMESchemaItem):
        __annotations__ = {
            "image": Optional[str],
            "label-value": Required[int],
        }


    colors: Recommended[List[Color]]
    properties: Optional[Property]
    source: Optional[Source]
    version: Recommended[Version]

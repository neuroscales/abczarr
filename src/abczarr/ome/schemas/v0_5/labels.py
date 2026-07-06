__all__ = ["ImageLabel"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.validators import IsInRange
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
UInt8 = tx.Annotated[int, IsInRange(0, 255)]
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
        }


    colors: Recommended[List[Color]]
    properties: Optional[List[Property]]
    source: Optional[Source]

__all__ = ["ImageLabel"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine
from abczarr._core.auto.converters import RangeConverter
from abczarr._core.rfc2119 import Optional, Recommended, Required

# locals
from ..base import OMEMetadata
from .version import Version

# typing
UInt8 = tx.Annotated[int, RangeConverter(0, 255)]


@autodefine
class ImageLabel(OMEMetadata):


    @autodefine
    class Color(OMEMetadata):
        label_value: Required[int]
        rgba: Optional[tx.Tuple[UInt8, UInt8, UInt8, UInt8]]


    @autodefine
    class Property(OMEMetadata):
        label_value: Required[int]


    @autodefine
    class Source(OMEMetadata):
        image: Optional[str] = None
        label_value: Required[int]


    colors: Recommended[tx.List[Color]]
    properties: Optional[Property]
    source: Optional[Source]
    version: Required[Version]

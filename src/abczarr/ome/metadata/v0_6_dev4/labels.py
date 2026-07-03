__all__ = ["ImageLabel"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine, field, RangeConverter
from abczarr._core.rfc2119 import Required, Recommended, Optional

# locals
from ..base import OMEMetadata

# typing
UInt8 = tx.Annotated[int, RangeConverter(0, 255)]


@autodefine
class ImageLabel(OMEMetadata):


    @autodefine
    class Color(OMEMetadata):
        label_value: Required[int] = field(alias="label-value")
        rgba: Optional[tx.Tuple[UInt8, UInt8, UInt8, UInt8]]


    @autodefine
    class Property(OMEMetadata):
        label_value: Required[int] = field(alias="label-value")


    @autodefine
    class Source(OMEMetadata):
        image: Optional[str] = None
        label_value: Required[int] = field(alias="label-value")


    colors: Recommended[tx.List[Color]]
    properties: Optional[Property]
    source: Optional[Source]

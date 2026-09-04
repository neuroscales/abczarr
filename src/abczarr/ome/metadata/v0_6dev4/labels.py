# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

__all__ = ["ImageLabel"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine
from abczarr._core.auto.converters import ToInRange
from abczarr._core.rfc2119 import Optional, Recommended

from ..base import OMEMetadata

UInt8 = tx.Annotated[int, ToInRange(0, 255)]


@autodefine
class ImageLabel(OMEMetadata):
    @autodefine
    class Color(OMEMetadata):
        label_value: Optional[int]
        rgba: Optional[tx.Tuple[UInt8, UInt8, UInt8, UInt8]]

    @autodefine
    class Property(OMEMetadata):
        label_value: Optional[int]

    @autodefine
    class Source(OMEMetadata):
        image: Optional[str] = None
        label_value: Optional[int]

    colors: Recommended[tx.List[Color]]
    properties: Optional[tx.List[Property]]
    source: Optional[Source]

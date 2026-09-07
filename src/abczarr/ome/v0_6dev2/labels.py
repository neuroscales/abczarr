# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

__all__ = ["ImageLabel"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.auto.converters import ToInRange
from abczarr._core.rfc2119 import Optional, Recommended

from ..base import OMEMetadata

UInt8 = tx.Annotated[int, ToInRange(0, 255)]


@autodefine
class ImageLabel(OMEMetadata):
    """Describes a segmentation label image: an array whose integer values
    name segments.

    An `ImageLabel` is attached to a label image group alongside its own
    [Multiscale][abczarr.ome.v0_6dev2.images.Multiscale]. `colors` maps each
    integer label value to a display color. `properties` and `source` carry
    further attributes for a label value, and record where the label image
    was derived from.
    """

    @autodefine
    class Color(OMEMetadata):
        label_value: Optional[int] = field(json="label-value")
        rgba: Optional[tx.Tuple[UInt8, UInt8, UInt8, UInt8]]

    @autodefine
    class Property(OMEMetadata):
        label_value: Optional[int] = field(json="label-value")

    @autodefine
    class Source(OMEMetadata):
        image: Optional[str] = None
        label_value: Optional[int] = field(json="label-value")

    colors: Recommended[tx.List[Color]]
    properties: Optional[tx.List[Property]]
    source: Optional[Source]

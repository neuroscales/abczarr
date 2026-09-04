# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""A segmentation label image: display colors and per-label properties."""

__all__ = ["ImageLabel"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine
from abczarr._core.auto.converters import ToInRange
from abczarr._core.rfc2119 import Optional, Recommended, Required

from ..base import OMEMetadata
from .version import Version

UInt8 = tx.Annotated[int, ToInRange(0, 255)]


@autodefine
class ImageLabel(OMEMetadata):
    """Metadata for a label image: an array whose integer values name segments.

    Attach one of these to a label image group alongside its own
    [Multiscale][abczarr.ome.metadata.v0_2.images.Multiscale]. `colors`
    maps each integer label value to a display color. `properties`
    and `source` carry any further attributes for a label value, and
    where the label image was derived from.
    """

    @autodefine
    class Color(OMEMetadata):
        """The display color for one label value.

        `rgba` is red, green, blue, and alpha, each `0`-`255`.
        """

        label_value: Required[int]
        rgba: Optional[tx.Tuple[UInt8, UInt8, UInt8, UInt8]]

    @autodefine
    class Property(OMEMetadata):
        """Extra, application-defined attributes for one label value.

        Beyond `label_value`, any other key is carried through as
        extra data. See
        [OMEMetadata][abczarr.ome.metadata.base.OMEMetadata].
        """

        label_value: Required[int]

    @autodefine
    class Source(OMEMetadata):
        """Where a label value came from.

        `image` names the intensity image this label was derived
        from, relative to the label image group.
        """

        image: Optional[str] = None
        label_value: Required[int]

    colors: Recommended[tx.List[Color]]
    properties: Optional[Property]
    source: Optional[Source]
    version: Recommended[Version]

# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""A segmentation label image: display colors and per-label properties."""

__all__ = ["ImageLabel"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.auto.converters import ToInRange
from abczarr._core.rfc2119 import Optional, Recommended, Required

from ..base import OMEMetadata

UInt8 = tx.Annotated[int, ToInRange(0, 255)]


@autodefine
class ImageLabel(OMEMetadata):
    """Metadata for a label image: an array whose integer values name segments.

    Attach one of these to a label image group alongside its own
    [Multiscale][abczarr.ome.v0_5.images.Multiscale].
    """

    @autodefine
    class Color(OMEMetadata):
        """The display color for one label value."""

        label_value: Required[int] = field(json="label-value")
        """The integer value this color applies to."""
        rgba: Optional[tx.Tuple[UInt8, UInt8, UInt8, UInt8]]
        """The color as red, green, blue, and alpha components, each
        from `0` to `255`. Optional."""

    @autodefine
    class Property(OMEMetadata):
        """Extra, application-defined attributes for one label value.

        Any key beyond `label_value` is carried through as extra data.
        See [OMEMetadata][abczarr.ome.base.OMEMetadata].
        """

        label_value: Required[int] = field(json="label-value")
        """The integer value these properties apply to."""

    @autodefine
    class Source(OMEMetadata):
        """Where a label image was derived from."""

        image: Optional[str] = None
        """The path of the intensity image this label was derived from,
        relative to the label image group. Optional."""

    colors: Recommended[tx.List[Color]]
    """The display color for each labeled integer value. Recommended."""
    properties: Optional[tx.List[Property]]
    """Further, application-defined attributes for each labeled value.
    Optional."""
    source: Optional[Source]
    """Where the label image was derived from. Optional."""

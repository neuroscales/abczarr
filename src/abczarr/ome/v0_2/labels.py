# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""A segmentation label image: display colors and per-label properties."""

__all__ = ["ImageLabel"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.auto.converters import ToInRange
from abczarr._core.rfc2119 import Optional, Recommended, Required

from ..base import OMEMetadata
from .version import Version

UInt8 = tx.Annotated[int, ToInRange(0, 255)]


@autodefine
class ImageLabel(OMEMetadata):
    """Metadata for a label image: an array whose integer values name segments.

    Attach one of these to a label image group alongside its own
    [Multiscale][abczarr.ome.v0_2.images.Multiscale].

    Parameters
    ----------
    colors : list of Color
        The display color for each labeled integer value. Recommended.
    properties : list of Property
        Further, application-defined attributes for each labeled value.
        Optional.
    source : Source
        Where the label image was derived from. Optional.
    version : Version
        The OME-NGFF version the metadata is written against. Recommended
        in OME-NGFF 0.1 and 0.2, and required from 0.3 on.
    """

    @autodefine
    class Color(OMEMetadata):
        """The display color for one label value.

        Parameters
        ----------
        label_value : int
            The integer value this color applies to.
        rgba : tuple of int
            The color as red, green, blue, and alpha components, each
            from `0` to `255`. Optional.
        """

        label_value: Required[int] = field(json="label-value")
        rgba: Optional[tx.Tuple[UInt8, UInt8, UInt8, UInt8]]

    @autodefine
    class Property(OMEMetadata):
        """Extra, application-defined attributes for one label value.

        Any key beyond `label_value` is carried through as extra data.
        See [OMEMetadata][abczarr.ome.base.OMEMetadata].

        Parameters
        ----------
        label_value : int
            The integer value these properties apply to.
        """

        label_value: Required[int] = field(json="label-value")

    @autodefine
    class Source(OMEMetadata):
        """Where a label image was derived from.

        Parameters
        ----------
        image : str
            The path of the intensity image this label was derived from,
            relative to the label image group. Optional.
        """

        image: Optional[str] = None

    colors: Recommended[tx.List[Color]]
    properties: Optional[tx.List[Property]]
    source: Optional[Source]
    version: Recommended[Version]

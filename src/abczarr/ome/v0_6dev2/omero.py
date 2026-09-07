# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

__all__ = ["Omero", "Channel"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine
from abczarr._core.rfc2119 import Required

from ..base import OMEMetadata


@autodefine
class Channel(OMEMetadata):
    """A `Channel` object specifies how to render one channel of a
    multi-channel image.

    `color` is a hex RGB string, such as `"FF0000"` for red. `window` gives
    the intensity range that is mapped onto that color.
    """

    @autodefine
    class Window(OMEMetadata):
        min: Required[float]
        max: Required[float]
        start: Required[float]
        end: Required[float]

    color: Required[str]
    window: Required[Window]


@autodefine
class Omero(OMEMetadata):
    """An `Omero` object holds rendering settings for an image, one entry per
    channel.

    An `Omero` object is attached to an image group, alongside its
    [Multiscale][abczarr.ome.v0_6dev2.images.Multiscale], to suggest how a
    viewer should display it. `channels` lists a
    [Channel][abczarr.ome.v0_6dev2.omero.Channel] for each channel of the
    image, in the image's channel order.
    """

    channels: Required[tx.List[Channel]]

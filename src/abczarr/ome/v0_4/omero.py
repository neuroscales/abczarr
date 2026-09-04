# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""Rendering settings: how to display an image's channels."""

__all__ = ["Omero", "Channel"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine
from abczarr._core.rfc2119 import Required

from ..base import OMEMetadata
from .version import Version


@autodefine
class Channel(OMEMetadata):
    """How to render one channel of a multi-channel image.

    `color` is a hex RGB string (`"FF0000"` for red); `window` gives
    the intensity range to map onto it.
    """

    @autodefine
    class Window(OMEMetadata):
        """The intensity range a channel's color is mapped over.

        `min`/`max` bound the channel's data; `start`/`end` are the
        (possibly narrower) range a viewer should render at full
        contrast.
        """

        min: Required[float]
        max: Required[float]
        start: Required[float]
        end: Required[float]

    color: Required[str]
    window: Required[Window]


@autodefine
class Omero(OMEMetadata):
    """Rendering settings for an image: one entry per channel.

    Attach one of these to an image group, alongside its
    [Multiscale][abczarr.ome.v0_4.images.Multiscale], to
    suggest how a viewer should display it. `channels` lists a
    [Channel][abczarr.ome.v0_4.omero.Channel] for each
    channel of the image, in order.
    """

    channels: Required[tx.List[Channel]]
    version: Required[Version]

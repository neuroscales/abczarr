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
    """How to render one channel of a multi-channel image."""

    @autodefine
    class Window(OMEMetadata):
        """The intensity range a channel's color is mapped over."""

        min: Required[float]
        """The lowest value the channel's data can take."""
        max: Required[float]
        """The highest value the channel's data can take."""
        start: Required[float]
        """The value a viewer should render at zero intensity. This may
        narrow the range `min` and `max` bound."""
        end: Required[float]
        """The value a viewer should render at full intensity. This may
        narrow the range `min` and `max` bound."""

    color: Required[str]
    """The channel's display color, as a hex RGB string such as
    `"FF0000"` for red."""
    window: Required[Window]
    """The intensity range mapped onto that color."""


@autodefine
class Omero(OMEMetadata):
    """Rendering settings for an image: one entry per channel.

    Attach one of these to an image group, alongside its
    [Multiscale][abczarr.ome.v0_3.images.Multiscale], to suggest how a
    viewer should display it.
    """

    channels: Required[tx.List[Channel]]
    """A [Channel][abczarr.ome.v0_3.omero.Channel] for each channel of
    the image, in the image's channel order."""
    version: Required[Version]
    """The OME-NGFF version the metadata is written against. Recommended
    in OME-NGFF 0.1 and 0.2, and required from 0.3 on."""

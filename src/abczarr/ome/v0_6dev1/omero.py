"""Rendering settings: how to display an image's channels."""

__all__ = ["Omero", "Channel"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine
from abczarr._core.rfc2119 import Required

# locals
from ..base import OMEMetadata


@autodefine
class Channel(OMEMetadata):
    """Specifies how to render one channel of a multi-channel image.

    Parameters
    ----------
    color : str
        The channel's display color, as a hex RGB string such as
        `"FF0000"` for red.
    window : Window
        The intensity range mapped onto that color.
    """

    @autodefine
    class Window(OMEMetadata):
        """The intensity range a channel's color is mapped over.

        Parameters
        ----------
        min : float
            The lowest value the channel's data can take.
        max : float
            The highest value the channel's data can take.
        start : float
            The value a viewer should render at zero intensity. This may
            narrow the range `min` and `max` bound.
        end : float
            The value a viewer should render at full intensity. This may
            narrow the range `min` and `max` bound.
        """

        min: Required[float]
        max: Required[float]
        start: Required[float]
        end: Required[float]


    color: Required[str]
    window: Required[Window]


@autodefine
class Omero(OMEMetadata):
    """Holds rendering settings for an image, one entry per channel.

    Attach one of these to an image group, alongside its
    [Multiscale][abczarr.ome.v0_6dev1.images.Multiscale], to suggest how a
    viewer should display it.

    Parameters
    ----------
    channels : list of Channel
        A [Channel][abczarr.ome.v0_6dev1.omero.Channel] for each channel
        of the image, in the image's channel order.
    """

    channels: Required[tx.List[Channel]]

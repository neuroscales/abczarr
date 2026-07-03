__all__ = ["Omero", "Channel"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine
from abczarr._core.rfc2119 import Required

# locals
from ..base import OMEMetadata
from .version import Version


@autodefine
class Channel(OMEMetadata):


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
    channels: Required[tx.List[Channel]]
    version: Required[Version]

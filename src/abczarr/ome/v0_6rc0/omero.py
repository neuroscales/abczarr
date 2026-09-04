# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

__all__ = ["Omero", "Channel"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine
from abczarr._core.rfc2119 import Required

from ..base import OMEMetadata


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

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine
from abczarr._core.metadata import FlexibleMetadata

# locals
from ..rfc2119 import Required


@autodefine
class Window(FlexibleMetadata):
    min: Required[float]
    max: Required[float]
    start: Required[float]
    end: Required[float]


@autodefine
class Channel(FlexibleMetadata):
    color: Required[str]
    window: Required[Window]


@autodefine
class Omero(FlexibleMetadata):
    channels: Required[tx.List[Channel]]

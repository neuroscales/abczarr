__all__ = ["Well"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Required, Recommended

# locals
from ..base import OMEMetadata
from .version import Version


@autodefine
class Well(OMEMetadata):


    @autodefine
    class Image(OMEMetadata):
        path: Required[str] = field(factory=False)
        acquisition: Recommended[int]


    images: Required[tx.List[Image]]
    version: Required[Version]

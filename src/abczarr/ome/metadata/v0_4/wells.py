__all__ = ["Well"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine, field
from abczarr._core.metadata import FlexibleMetadata

# locals
from .version import Version
from ..rfc2119 import Required, Recommended


@autodefine
class Well(FlexibleMetadata):


    @autodefine
    class Image(FlexibleMetadata):
        path: Required[str] = field(factory=False)
        acquisition: Recommended[int]


    images: Required[tx.List[Image]]
    version: Version

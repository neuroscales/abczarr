# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine, field
from abczarr._core.metadata import FlexibleMetadata

# locals
from ..rfc2119 import Required, Recommended


@autodefine
class WellImage(FlexibleMetadata):
    path: Required[str] = field(factory=False)
    acquisition: Recommended[int]


@autodefine
class Well(FlexibleMetadata):
    images: Required[tx.List[WellImage]]

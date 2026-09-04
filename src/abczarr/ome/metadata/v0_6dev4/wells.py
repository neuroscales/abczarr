# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

__all__ = ["Well"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Recommended, Required

from ..base import OMEMetadata


@autodefine
class Well(OMEMetadata):
    @autodefine
    class Image(OMEMetadata):
        path: Required[str] = field(factory=False)
        acquisition: Recommended[int]

    images: Required[tx.List[Image]]

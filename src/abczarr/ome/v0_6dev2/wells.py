# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

__all__ = ["Well"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Recommended, Required

from ..base import OMEMetadata


@autodefine
class Well(OMEMetadata):
    """A `Well` object lists a well's images: one field of view per acquisition
    run.

    A well group holds one subgroup per field of view. `images` lists them,
    each naming its subgroup and, when the plate ran more than one
    acquisition, which acquisition it belongs to.
    """

    @autodefine
    class Image(OMEMetadata):
        path: Required[str] = field(factory=False)
        acquisition: Recommended[int]

    images: Required[tx.List[Image]]

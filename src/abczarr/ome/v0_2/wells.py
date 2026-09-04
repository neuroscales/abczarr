# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""A well: the images acquired at one position of a screening plate."""

__all__ = ["Well"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Recommended, Required

from ..base import OMEMetadata
from .version import Version


@autodefine
class Well(OMEMetadata):
    """A well's images: one field of view per acquisition run.

    A well group holds one subgroup per field of view; `images` lists
    them, each naming its subgroup and, when the plate carries more
    than one, which acquisition it belongs to.
    """

    @autodefine
    class Image(OMEMetadata):
        """One field of view within a well.

        `path` is the image's group, relative to the well group.
        `acquisition` is the id of the
        [Plate.Acquisition][abczarr.ome.v0_2.plates.Plate.Acquisition]
        it was captured in, when the plate ran more than one.
        """

        path: Required[str] = field(factory=False)
        acquisition: Recommended[int]

    images: Required[tx.List[Image]]
    version: Recommended[Version]

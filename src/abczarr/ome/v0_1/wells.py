"""A well: the images acquired at one position of a screening plate."""

__all__ = ["Well"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Recommended, Required

# locals
from ..base import OMEMetadata
from .version import Version


@autodefine
class Well(OMEMetadata):
    """A well's images: one field of view per acquisition run.

    A well group holds one subgroup per field of view.
    """

    @autodefine
    class Image(OMEMetadata):
        """One field of view within a well."""

        path: Required[str] = field(factory=False)
        """The image's group, relative to the well group."""
        acquisition: Recommended[int]
        """The id of the
        [Plate.Acquisition][abczarr.ome.v0_1.plates.Plate.Acquisition]
        this field of view was captured in, when the plate ran more
        than one. Recommended."""


    images: Required[tx.List[Image]]
    """The well's fields of view, each naming its subgroup and, when
    the plate ran more than one acquisition, which acquisition it
    belongs to."""
    version: Recommended[Version]
    """The OME-NGFF version the metadata is written against. Recommended
    in OME-NGFF 0.1 and 0.2, and required from 0.3 on."""

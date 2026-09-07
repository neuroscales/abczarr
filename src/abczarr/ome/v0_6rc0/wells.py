# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

"""A well: the images acquired at one position of a screening plate."""

__all__ = ["Well"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Recommended, Required

from ..base import OMEMetadata


@autodefine
class Well(OMEMetadata):
    """Lists a well's images: one field of view per acquisition run.

    A well group holds one subgroup per field of view.

    Parameters
    ----------
    images : list of Image
        The well's fields of view, each naming its subgroup and, when the
        plate ran more than one acquisition, which acquisition it belongs
        to.
    """

    @autodefine
    class Image(OMEMetadata):
        """One field of view within a well.

        Parameters
        ----------
        path : str
            The image's group, relative to the well group.
        acquisition : int
            The id of the
            [Plate.Acquisition][abczarr.ome.v0_6rc0.plates.Plate.Acquisition]
            this field of view was captured in, when the plate ran more
            than one acquisition. Recommended.
        """

        path: Required[str] = field(factory=False)
        acquisition: Recommended[int]

    images: Required[tx.List[Image]]

# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""The multiscale image pyramid.

[Multiscale][abczarr.ome.v0_5.images.Multiscale] describes a
pyramid of progressively downsampled resolution levels. Each level is
a [Dataset][abczarr.ome.v0_5.images.Dataset], naming a Zarr
array and how it is positioned relative to the others.
"""

__all__ = ["Dataset", "Multiscale"]
import typing_extensions as tx

from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Optional, Recommended, Required

from ..base import OMEMetadata
from .axes import Axis
from .transformations import CoordinateTransformation, Scale, Translation


@autodefine
class Dataset(OMEMetadata):
    """One resolution level of a multiscale pyramid.

    Parameters
    ----------
    path : str
        The name of the Zarr array holding this level, relative to the
        image group.
    coordinateTransformations : tuple of CoordinateTransformation
        Places this level in the pyramid's physical space: a
        [Scale][abczarr.ome.v0_5.transformations.Scale], optionally
        followed by a
        [Translation][abczarr.ome.v0_5.transformations.Translation],
        one value per axis.
    """

    path: Required[str] = field(factory=False)
    coordinateTransformations: Required[
        tx.Union[tx.Tuple[Scale], tx.Tuple[Scale, Translation]]
    ]


@autodefine
class Multiscale(OMEMetadata):
    """A multiscale image pyramid: its axes and resolution levels.

    Parameters
    ----------
    axes : list of Axis
        The pyramid's dimensions, in the order every array shape and
        every coordinate transformation the pyramid carries uses. Each
        entry is an [Axis][abczarr.ome.v0_5.axes.Axis].
    datasets : list of Dataset
        The pyramid's resolution levels, from full resolution down.
        Each entry is a
        [Dataset][abczarr.ome.v0_5.images.Dataset].
    coordinateTransformations : list of CoordinateTransformation
        Transformations applied to every level, before that level's own
        transformations run. Optional.
    name : str
        A name for the multiscale image. Recommended.
    type : str
        The method used to generate the lower resolutions, such as
        ``"gaussian"``. Recommended.
    metadata : Metadata
        Further, free-form detail about how the lower resolutions were
        generated. Recommended.
    """

    @autodefine
    class Metadata(OMEMetadata):
        """Free-form detail about how a pyramid's lower resolutions were
        generated.

        Parameters
        ----------
        method : str
            The name of the downsampling function. Optional.
        version : str
            The version of the software that ran `method`. Optional.
        args : JSON value
            The positional arguments `method` was called with. The
            upstream corpus writes this as a bare string as well as a
            list, so it is read as any JSON value rather than coerced
            into a list. Optional.
        kwargs : dict
            The keyword arguments `method` was called with. Optional.
        """

        method: Optional[str]
        version: Optional[str]
        args: Optional[tz.Json]
        kwargs: Optional[tx.Dict[str, tz.Json]]

    axes: Required[tx.List[Axis]]
    datasets: Required[tx.List[Dataset]]
    coordinateTransformations: Optional[tx.List[CoordinateTransformation]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]

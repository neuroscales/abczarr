# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

"""The multiscale image pyramid.

[Multiscale][abczarr.ome.v0_6dev3.images.Multiscale] describes a pyramid of
progressively downsampled resolution levels, placed within one or more
named coordinate systems. Each level is a
[Dataset][abczarr.ome.v0_6dev3.images.Dataset], naming a Zarr array and how
it maps into those coordinate systems.
"""

__all__ = ["Dataset", "Multiscale"]
import typing_extensions as tx

from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Optional, Recommended, Required

from ..base import OMEMetadata
from .systems import CoordinateSystem
from .transformations import CoordinateTransformation


@autodefine
class Dataset(OMEMetadata):
    """One resolution level of a multiscale pyramid.

    Parameters
    ----------
    path : str
        The name of the Zarr array holding this level, relative to the
        image group.
    coordinateTransformations : list of CoordinateTransformation
        One or more transformations that map this level's own coordinate
        system into another coordinate system of the pyramid.
    """

    path: Required[str] = field(factory=False)
    coordinateTransformations: Required[tx.List[CoordinateTransformation]]


@autodefine
class Multiscale(OMEMetadata):
    """An image pyramid: its coordinate systems and its resolution levels.

    Parameters
    ----------
    coordinateSystems : list of CoordinateSystem
        Every
        [CoordinateSystem][abczarr.ome.v0_6dev3.systems.CoordinateSystem]
        that a [Dataset][abczarr.ome.v0_6dev3.images.Dataset] or a
        transformation in `coordinateTransformations` can refer to.
    datasets : list of Dataset
        The pyramid's resolution levels, from full resolution down.
    coordinateTransformations : list of CoordinateTransformation
        Transformations applied to every level, before that level's own
        transformations run. Recommended.
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
            The positional arguments `method` was called with. Optional.
        kwargs : dict
            The keyword arguments `method` was called with. Optional.
        """

        method: Optional[str]
        version: Optional[str]
        args: Optional[tz.Json]
        kwargs: Optional[tx.Dict[str, tz.Json]]

    coordinateSystems: Required[tx.List[CoordinateSystem]]
    datasets: Required[tx.List[Dataset]]
    coordinateTransformations: Recommended[tx.List[CoordinateTransformation]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]

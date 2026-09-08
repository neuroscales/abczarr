# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

"""The multiscale image pyramid.

[Multiscale][abczarr.ome.v0_6rc0.images.Multiscale] describes a pyramid of
progressively downsampled resolution levels, placed within one or more
named coordinate systems. Each level is a
[Dataset][abczarr.ome.v0_6rc0.images.Dataset], naming a Zarr array and how
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
    """One resolution level of a multiscale pyramid."""

    path: Required[str] = field(factory=False)
    """The name of the Zarr array holding this level, relative to the
    image group."""
    coordinateTransformations: Required[tx.List[CoordinateTransformation]]
    """One or more transformations that map this level's own coordinate
    system into another coordinate system of the pyramid."""


@autodefine
class Multiscale(OMEMetadata):
    """An image pyramid: its coordinate systems and its resolution levels."""

    @autodefine
    class Metadata(OMEMetadata):
        """Free-form detail about how a pyramid's lower resolutions were
        generated.
        """

        method: Optional[str]
        """The name of the downsampling function. Optional."""
        version: Optional[str]
        """The version of the software that ran `method`. Optional."""
        args: Optional[tz.Json]
        """The positional arguments `method` was called with. Optional."""
        kwargs: Optional[tx.Dict[str, tz.Json]]
        """The keyword arguments `method` was called with. Optional."""

    coordinateSystems: Required[tx.List[CoordinateSystem]]
    """Every
    [CoordinateSystem][abczarr.ome.v0_6rc0.systems.CoordinateSystem]
    that a [Dataset][abczarr.ome.v0_6rc0.images.Dataset] or a
    transformation in `coordinateTransformations` can refer to."""
    datasets: Required[tx.List[Dataset]]
    """The pyramid's resolution levels, from full resolution down."""
    coordinateTransformations: Recommended[tx.List[CoordinateTransformation]]
    """Transformations applied to every level, before that level's own
    transformations run. Recommended."""
    name: Recommended[str]
    """A name for the multiscale image. Recommended."""
    type: Recommended[str]
    """The method used to generate the lower resolutions, such as
    ``"gaussian"``. Recommended."""
    metadata: Recommended[Metadata]
    """Further, free-form detail about how the lower resolutions were
    generated. Recommended."""

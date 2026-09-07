__all__ = ["Dataset", "Multiscale"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Optional, Recommended, Required

# locals
from ..base import OMEMetadata
from .systems import CoordinateSystem
from .transformations import CoordinateTransformation


@autodefine
class Dataset(OMEMetadata):
    """One resolution level of a multiscale pyramid.

    `path` names the Zarr array that holds this level, relative to the image
    group. `coordinateTransformations` lists one or more
    `CoordinateTransformation` objects that map this level's own coordinate
    system into another coordinate system of the pyramid.
    """

    path: Required[str] = field(factory=False)
    coordinateTransformations: Required[tx.List[CoordinateTransformation]]


@autodefine
class Multiscale(OMEMetadata):
    """Describes an image pyramid: its coordinate systems and its resolution
    levels.

    `coordinateSystems` names every
    [CoordinateSystem][abczarr.ome.v0_6dev1.systems.CoordinateSystem] that a
    [Dataset][abczarr.ome.v0_6dev1.images.Dataset] or a transformation in
    `coordinateTransformations` can refer to. `datasets` lists the pyramid's
    resolution levels, from full resolution down. When
    `coordinateTransformations` is given, it applies to every level before
    that level's own transformations run.
    """

    @autodefine
    class Metadata(OMEMetadata):
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

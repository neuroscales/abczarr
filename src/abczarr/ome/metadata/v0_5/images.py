"""The multiscale image pyramid.

[Multiscale][abczarr.ome.metadata.v0_5.images.Multiscale] describes a
pyramid of progressively downsampled resolution levels. Each level is
a [Dataset][abczarr.ome.metadata.v0_5.images.Dataset], naming a Zarr
array and how it is positioned relative to the others.
"""

__all__ = ["Dataset", "Multiscale"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Optional, Recommended, Required

# locals
from ..base import OMEMetadata
from .axes import Axis
from .transformations import CoordinateTransformation, Scale, Translation


@autodefine
class Dataset(OMEMetadata):
    """One resolution level of a multiscale pyramid.

    `path` is the name of the Zarr array holding this level, relative
    to the image group. `coordinateTransformations` places it in the
    pyramid's physical space: a
    [Scale][abczarr.ome.metadata.v0_5.transformations.Scale], optionally
    followed by a
    [Translation][abczarr.ome.metadata.v0_5.transformations.Translation],
    one value per axis.
    """

    path: Required[str] = field(factory=False)
    coordinateTransformations: Required[tx.Union[
        tx.Tuple[Scale],
        tx.Tuple[Scale, Translation],
    ]]


@autodefine
class Multiscale(OMEMetadata):
    """A multiscale image pyramid: its axes and resolution levels.

    `axes` names and orders the pyramid's dimensions: `t`, `c`, `z`,
    `y`, `x`, in whatever subset and order the image uses. `datasets`
    lists its resolution levels from full resolution down, each a
    [Dataset][abczarr.ome.metadata.v0_5.images.Dataset].
    `coordinateTransformations` here, if given, applies to every
    level before its own.
    """

    @autodefine
    class Metadata(OMEMetadata):
        """How the pyramid's lower resolutions were generated.

        Free-form: `method` names the downsampling function, `args`
        and `kwargs` are what it was called with.
        """

        method: Optional[str]
        version: Optional[str]
        args: Optional[tx.List[tz.JSON]]
        kwargs: Optional[tx.Dict[str, tz.JSON]]

    axes: Required[tx.List[Axis]]
    datasets: Required[tx.List[Dataset]]
    coordinateTransformations: Optional[tx.List[CoordinateTransformation]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]

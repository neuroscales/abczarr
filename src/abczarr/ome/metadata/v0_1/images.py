"""The multiscale image pyramid.

[Multiscale][abczarr.ome.metadata.v0_1.images.Multiscale] describes a
pyramid of progressively downsampled resolution levels. Each level is
a [Dataset][abczarr.ome.metadata.v0_1.images.Dataset], naming a Zarr
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
from .version import Version

# typing
SpaceAxis = tx.Literal["x", "y", "z"]
TimeAxis = tx.Literal["t"]
ChannelAxis = tx.Literal["c"]
Axis = tx.Union[SpaceAxis, TimeAxis, ChannelAxis]


@autodefine
class Dataset(OMEMetadata):
    """One resolution level of a multiscale pyramid.

    `path` is the name of the Zarr array holding this level, relative
    to the image group.
    """

    path: Required[str] = field(factory=False)


@autodefine
class Multiscale(OMEMetadata):
    """A multiscale image pyramid: its resolution levels.

    `datasets` lists the pyramid's resolution levels from full
    resolution down, each a
    [Dataset][abczarr.ome.metadata.v0_1.images.Dataset].
    """

    @autodefine
    class Metadata(OMEMetadata):
        """How the pyramid's lower resolutions were generated.

        Free-form: `method` names the downsampling function, `args`
        and `kwargs` are what it was called with.
        """

        method: Optional[str]
        version: Optional[str]
        args: Optional[tx.List[tz.Json]]
        kwargs: Optional[tx.Dict[str, tz.Json]]

    datasets: Required[tx.List[Dataset]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]
    version: Recommended[Version]

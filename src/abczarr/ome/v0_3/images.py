# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""The multiscale image pyramid.

[Multiscale][abczarr.ome.v0_3.images.Multiscale] describes a
pyramid of progressively downsampled resolution levels. Each level is
a [Dataset][abczarr.ome.v0_3.images.Dataset], naming a Zarr
array and how it is positioned relative to the others.
"""

__all__ = ["Dataset", "Multiscale"]
import typing_extensions as tx

from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Optional, Recommended, Required

from ..base import OMEMetadata
from .version import Version

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
    """A multiscale image pyramid: its axes and resolution levels.

    `axes` names and orders the pyramid's dimensions: `t`, `c`, `z`,
    `y`, `x`, in whatever subset and order the image uses. `datasets`
    lists its resolution levels from full resolution down, each a
    [Dataset][abczarr.ome.v0_3.images.Dataset].
    """

    @autodefine
    class Metadata(OMEMetadata):
        """How the pyramid's lower resolutions were generated.

        Free-form: `method` names the downsampling function, `args`
        and `kwargs` are what it was called with. `args` is any JSON
        value -- the upstream corpus writes it as a bare string as
        well as a list -- so it is not coerced into a list.
        """

        method: Optional[str]
        version: Optional[str]
        args: Optional[tz.Json]
        kwargs: Optional[tx.Dict[str, tz.Json]]

    axes: Required[tx.List[Axis]]
    datasets: Required[tx.List[Dataset]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]
    version: Required[Version]

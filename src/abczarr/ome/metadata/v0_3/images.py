__all__ = ["Dataset", "Multiscale"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Required, Recommended, Optional

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
    path: Required[str] = field(factory=False)


@autodefine
class Multiscale(OMEMetadata):

    @autodefine
    class Metadata(OMEMetadata):
        method: Optional[str]
        version: Optional[str]
        args: Optional[tx.List[tz.JSON]]
        kwargs: Optional[tx.Dict[str, tz.JSON]]

    axes: Required[tx.List[Axis]]
    datasets: Required[tx.List[Dataset]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]
    version: Required[Version]

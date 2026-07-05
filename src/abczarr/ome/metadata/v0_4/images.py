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
from .version import Version


@autodefine
class Dataset(OMEMetadata):
    path: Required[str] = field(factory=False)
    coordinateTransformations: Required[tx.Union[
        tx.Tuple[Scale],
        tx.Tuple[Scale, Translation],
    ]]


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
    coordinateTransformations: Optional[tx.List[CoordinateTransformation]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]
    version: Required[Version]

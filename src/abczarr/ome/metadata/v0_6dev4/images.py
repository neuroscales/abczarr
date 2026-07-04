__all__ = ["Dataset", "Multiscale"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Required, Recommended, Optional

# locals
from ..base import OMEMetadata
from .systems import CoordinateSystem
from .transformations import CoordinateTransformation


@autodefine
class Dataset(OMEMetadata):
    path: Required[str] = field(factory=False)
    coordinateTransformations: Required[tx.List[CoordinateTransformation]]


@autodefine
class Multiscale(OMEMetadata):

    @autodefine
    class Metadata(OMEMetadata):
        method: Optional[str]
        version: Optional[str]
        args: Optional[tx.List[tz.JSON]]
        kwargs: Optional[tx.Dict[str, tz.JSON]]

    coordinateSystems: Required[tx.List[CoordinateSystem]]
    datasets: Required[tx.List[Dataset]]
    coordinateTransformations: Recommended[tx.List[CoordinateTransformation]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]

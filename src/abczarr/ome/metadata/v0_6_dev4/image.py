# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine, field
from abczarr._core.metadata import FlexibleMetadata

# locals
from .systems import CoordinateSystem
from .transformations import CoordinateTransformation
from ..rfc2119 import Required, Recommended


@autodefine
class Dataset(FlexibleMetadata):
    path: Required[str] = field(factory=False)
    coordinateTransformations: Required[tx.List[CoordinateTransformation]]


@autodefine
class Multiscale(FlexibleMetadata):
    coordinateSystems: Required[tx.List[CoordinateSystem]]
    datasets: Required[tx.List[Dataset]]
    coordinateTransformations: Recommended[tx.List[CoordinateTransformation]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[FlexibleMetadata]

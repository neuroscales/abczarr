# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

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
    path: Required[str] = field(factory=False)
    coordinateTransformations: Required[tx.List[CoordinateTransformation]]


@autodefine
class Multiscale(OMEMetadata):
    @autodefine
    class Metadata(OMEMetadata):
        method: Optional[str]
        version: Optional[str]
        args: Optional[tx.List[tz.Json]]
        kwargs: Optional[tx.Dict[str, tz.Json]]

    coordinateSystems: Required[tx.List[CoordinateSystem]]
    datasets: Required[tx.List[Dataset]]
    coordinateTransformations: Recommended[tx.List[CoordinateTransformation]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]

__all__ = ["Dataset", "Multiscale"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem, ome_schema_opt
from .systems import CoordinateSystem
from .transformations import CoordinateTransformation

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
List = tz.BuiltinSequence  # list | tuple
JSON = tz.JSON


class Dataset(OMESchemaItem, **ome_schema_opt):
    path: Required[str]
    coordinateTransformations: Required[List[CoordinateTransformation]]


class Multiscale(OMESchemaItem, **ome_schema_opt):

    class Metadata(OMESchemaItem, **ome_schema_opt):
        method: Optional[str]
        version: Optional[str]
        args: Optional[JSON]
        kwargs: Optional[tx.Dict[str, JSON]]

    coordinateSystems: Required[List[CoordinateSystem]]
    datasets: Required[List[Dataset]]
    coordinateTransformations: Recommended[List[CoordinateTransformation]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]

__all__ = ["Dataset", "Multiscale"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem
from .systems import CoordinateSystem
from .transformations import CoordinateTransformationBase

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
List = tz.BuiltinSequence  # list | tuple
JSON = tz.JSON


class Dataset(OMESchemaItem):
    path: Required[str]
    coordinateTransformations: Required[List[CoordinateTransformationBase]]


class Multiscale(OMESchemaItem):

    class Metadata(OMESchemaItem):
        method: Optional[str]
        version: Optional[str]
        args: Optional[List[JSON]]
        kwargs: Optional[tx.Dict[str, JSON]]

    coordinateSystems: Required[List[CoordinateSystem]]
    datasets: Required[List[Dataset]]
    coordinateTransformations: Recommended[List[CoordinateTransformationBase]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]

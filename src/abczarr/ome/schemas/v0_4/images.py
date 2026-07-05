__all__ = ["Dataset", "Multiscale"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem
from .axes import Axis
from .transformations import CoordinateTransformation, Scale, Translation
from .version import Version

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
List = tz.BuiltinSequence  # list | tuple
JSON = tz.JSON


class Dataset(OMESchemaItem):
    path: Required[str]
    coordinateTransformations: Required[tx.Union[
        tx.Tuple[Scale],
        tx.Tuple[Scale, Translation],
    ]]


class Multiscale(OMESchemaItem):

    class Metadata(OMESchemaItem):
        method: Optional[str]
        version: Optional[str]
        args: Optional[JSON]
        kwargs: Optional[tx.Dict[str, JSON]]

    axes: Required[List[Axis]]
    datasets: Required[List[Dataset]]
    coordinateTransformations: Optional[List[CoordinateTransformation]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]
    version: Required[Version]

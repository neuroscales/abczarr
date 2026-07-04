__all__ = ["Dataset", "Multiscale"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem
from .version import Version

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
List = tz.BuiltinSequence  # list | tuple
JSON = tz.JSON

SpaceAxis = tx.Literal["x", "y", "z"]
TimeAxis = tx.Literal["t"]
ChannelAxis = tx.Literal["c"]
Axis = tx.Union[SpaceAxis, TimeAxis, ChannelAxis]


class Dataset(OMESchemaItem):
    path: Required[str]


class Multiscale(OMESchemaItem):

    class Metadata(OMESchemaItem):
        method: Optional[str]
        version: Optional[str]
        args: Optional[List[JSON]]
        kwargs: Optional[tx.Dict[str, JSON]]

    datasets: Required[List[Dataset]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]
    version: Recommended[Version]

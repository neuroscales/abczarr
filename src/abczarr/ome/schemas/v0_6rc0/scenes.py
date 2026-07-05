__all__ = ["Scene"]

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem, ome_schema_opt
from .transformations import CoordinateTransformation
from .systems import CoordinateSystem


# typing
Required = RequirementForTypedDict.Required
Optional = RequirementForTypedDict.Optional
List = tz.BuiltinSequence  # list | tuple


class Scene(OMESchemaItem, **ome_schema_opt):
    coordinateTransformations: Required[List[CoordinateTransformation]]
    coordinateSystems: Optional[List[CoordinateSystem]]

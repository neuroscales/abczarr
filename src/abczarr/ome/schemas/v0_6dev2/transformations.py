__all__ = [
    "CoordinateTransformation",
    "Identity",
    "MapAxis",
    "Translation",
    "Scale",
    "Affine",
    "Rotation",
    "InverseOf",
    "Bijection",
    "Sequence",
    "ByDimension",
    "Displacements",
    "Coordinates",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem, ome_schema_opt

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
List = tz.BuiltinSequence  # list | tuple
JSON = tz.Json
Interpolation = tx.Union[tx.Literal["nearest", "linear", "bspline-cubic"], str]
#: An input/output is a coordinate-system name (string), or -- inside a
#: ``byDimension`` transformation -- an array of axis names.
InputOutput = tx.Union[str, List[str]]


class CoordinateTransformationBase(OMESchemaItem, **ome_schema_opt):
    type: Required[str]
    input: Recommended[InputOutput]
    output: Recommended[InputOutput]
    name: Optional[str]


class Identity(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["identity"]]


class MapAxis(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["mapAxis"]]
    mapAxis: Required[tx.Dict[str, str]]


class Translation(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["translation"]]
    translation: Optional[List[float]]
    path: Optional[str]


class Scale(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["scale"]]
    scale: Optional[List[float]]
    path: Optional[str]


class Affine(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["affine"]]
    affine: Optional[tx.Union[List[List[float]], List[float]]]
    path: Optional[str]


class Rotation(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["rotation"]]
    rotation: Optional[tx.Union[List[List[float]], List[float]]]
    path: Optional[str]


class InverseOf(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["inverseOf"]]
    transformation: Required["CoordinateTransformation"]


class Bijection(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["bijection"]]
    forward: Required["CoordinateTransformation"]
    inverse: Required["CoordinateTransformation"]


class Sequence(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["sequence"]]
    transformations: Required[List["CoordinateTransformation"]]


class ByDimension(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["byDimension"]]
    transformations: Required[List["CoordinateTransformation"]]


class Displacements(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["displacements"]]
    path: Optional[str]
    interpolation: Optional[Interpolation]


class Coordinates(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["coordinates"]]
    path: Optional[str]
    interpolation: Optional[Interpolation]


CoordinateTransformation = tx.Union[
    Identity,
    MapAxis,
    Translation,
    Scale,
    Affine,
    Rotation,
    InverseOf,
    Bijection,
    Sequence,
    ByDimension,
    Displacements,
    Coordinates,
]

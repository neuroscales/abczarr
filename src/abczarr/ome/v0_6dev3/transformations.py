# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

"""Coordinate transformations: how one coordinate system maps to another."""

__all__ = [
    "CoordinateTransformation",
    "Identity",
    "MapAxis",
    "Translation",
    "Scale",
    "Affine",
    "Rotation",
    "Sequence",
    "Displacements",
    "Coordinates",
    "Bijection",
    "ByDimension",
]
import typing_extensions as tx

from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import Optional, Required

from ..base import OMEMetadata

Interpolation = tx.Union[tx.Literal["nearest", "linear", "bspline-cubic"], str]


@autodefine
class CoordinateTransformation(OMEMetadata):
    """Maps coordinates from one coordinate system to another.

    Constructing a `CoordinateTransformation` with a recognized `type`
    returns the matching subclass, such as
    [Scale][abczarr.ome.v0_6dev3.transformations.Scale].
    """

    type: Required[str] = field(factory=False)
    """Which kind of transformation this is."""
    input: Optional[tz.Json]
    """Identifies the
    [CoordinateSystem][abczarr.ome.v0_6dev3.systems.CoordinateSystem]
    the transformation maps from. Optional."""
    output: Optional[tz.Json]
    """Identifies the
    [CoordinateSystem][abczarr.ome.v0_6dev3.systems.CoordinateSystem]
    the transformation maps to. Optional."""
    name: Optional[str]
    """A label for the transformation itself. Optional."""


@register_subclass(type="identity")
@autodefine
class Identity(CoordinateTransformation):
    """Leaves coordinates unchanged.

    States that `input` and `output` are the same coordinate system, or
    that no numeric adjustment is needed between them.
    """

    type: Required[tx.Literal["identity"]]


@register_subclass(type="mapAxis")
@autodefine
class MapAxis(CoordinateTransformation):
    """Permutes axes without changing any coordinate values."""

    type: Required[tx.Literal["mapAxis"]]
    mapAxis: Required[tx.List[int]]
    """One entry per output axis, giving the index of the input axis
    whose values that output axis carries."""


@register_subclass(type="translation")
@autodefine
class Translation(CoordinateTransformation):
    """Adds a fixed offset to every coordinate, one value per axis."""

    type: Required[tx.Literal["translation"]]
    translation: Optional[tx.List[float]]
    """The offset, given inline, one number per axis. Optional."""
    path: Optional[str]
    """The path of an array to read the offset from instead, for a
    translation that varies from point to point rather than staying
    constant. Optional."""


@register_subclass(type="scale")
@autodefine
class Scale(CoordinateTransformation):
    """Multiplies every coordinate by a per-axis factor."""

    type: Required[tx.Literal["scale"]]
    scale: Optional[tx.List[float]]
    """The factor, given inline, one number per axis. Optional."""
    path: Optional[str]
    """The path of an array to read the factor from instead, for a
    scale that varies from point to point rather than staying
    constant. Optional."""


@register_subclass(type="affine")
@autodefine
class Affine(CoordinateTransformation):
    """Applies a linear map and a translation, given as a matrix."""

    type: Required[tx.Literal["affine"]]
    affine: Optional[tz.Json]
    """The matrix, given inline. Optional."""
    path: Optional[str]
    """The path of an array to read the matrix from instead. Optional."""


@register_subclass(type="rotation")
@autodefine
class Rotation(CoordinateTransformation):
    """Rotates coordinates, given as a matrix."""

    type: Required[tx.Literal["rotation"]]
    rotation: Optional[tz.Json]
    """The matrix, given inline. Optional."""
    path: Optional[str]
    """The path of an array to read the matrix from instead. Optional."""


@register_subclass(type="sequence")
@autodefine
class Sequence(CoordinateTransformation):
    """Composes several transformations into one, applied in order.

    Each transformation's output feeds into the next transformation as
    its input.
    """

    type: Required[tx.Literal["sequence"]]
    transformations: Required[tx.List[CoordinateTransformation]]
    """The transformations to compose, from `input` to `output`."""


@register_subclass(type="displacements")
@autodefine
class Displacements(CoordinateTransformation):
    """Defined by a displacement field.

    The vector read for each point is added to the input coordinate to
    produce the output coordinate.
    """

    type: Required[tx.Literal["displacements"]]
    path: Optional[str]
    """The path of an array that gives a displacement vector for each
    point. Optional."""
    interpolation: Optional[Interpolation]
    """How to sample the array between its own points. Optional."""


@register_subclass(type="coordinates")
@autodefine
class Coordinates(CoordinateTransformation):
    """Defined by an explicit coordinate lookup."""

    type: Required[tx.Literal["coordinates"]]
    path: Optional[str]
    """The path of an array that gives the output coordinate for each
    point directly. Optional."""
    interpolation: Optional[Interpolation]
    """How to sample the array between its own points. Optional."""


@register_subclass(type="bijection")
@autodefine
class Bijection(CoordinateTransformation):
    """An explicit forward and inverse pair of transformations.

    Used when a transformation's inverse cannot be derived automatically
    from its forward direction.
    """

    type: Required[tx.Literal["bijection"]]
    forward: Required[CoordinateTransformation]
    """The transformation from `input` to `output`."""
    inverse: Required[CoordinateTransformation]
    """The transformation from `output` back to `input`."""


@register_subclass(type="byDimension")
@autodefine
class ByDimension(CoordinateTransformation):
    """Combines several transformations, each acting on its own subset of
    axes.

    Together, the entries in `transformations` cover every axis between
    `input` and `output`.
    """

    @autodefine
    class Transformation(OMEMetadata):
        """One transformation of a
        [ByDimension][abczarr.ome.v0_6dev3.transformations.ByDimension],
        and the axes it applies to.
        """

        transformation: Optional[CoordinateTransformation]
        """The transformation to apply. Optional."""
        input_axes: Optional[tx.List[int]]
        """The indices, into the enclosing `ByDimension`'s `input`
        coordinate system, of the axes `transformation` reads from.
        Optional."""
        output_axes: Optional[tx.List[int]]
        """The indices, into the enclosing `ByDimension`'s `output`
        coordinate system, of the axes `transformation` writes to.
        Optional."""

    type: Required[tx.Literal["byDimension"]]
    transformations: Required[tx.List[Transformation]]
    """The transformations to combine."""

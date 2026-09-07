# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

__all__ = [
    "Axis",
    "SpaceAxis",
    "TimeAxis",
    "ChannelAxis",
    "ArrayAxis",
    "DisplacementAxis",
    "CoordinateAxis",
    "CoordinateSystem",
    "AxisType",
    "SpaceUnit",
    "TimeUnit",
    "Unit",
]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import (
    NotRecommended,
    Optional,
    Recommended,
    Required,
)

from ..base import OMEMetadata

AxisType = tx.Literal[
    "array", "space", "time", "channel", "coordinate", "displacement"
]
SpaceUnit = tx.Literal[
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
]
TimeUnit = tx.Literal[
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
]
Unit = tx.Union[SpaceUnit, TimeUnit]


@autodefine
class Axis(OMEMetadata):
    """An `Axis` object describes one dimension of a
    [CoordinateSystem][abczarr.ome.v0_6dev3.systems.CoordinateSystem].

    `name` is the axis's label, such as `"x"` or `"channel"`. `type`
    identifies what kind of axis it is: `"space"`, `"time"`, `"channel"`,
    `"array"`, `"displacement"`, or `"coordinate"`. `unit` gives its
    physical unit, when it has one. `discrete` marks an axis whose values
    are integer indices rather than continuous coordinates. `longName` is a
    human-readable label beyond `name`. Constructing an `Axis` with a
    recognized `type` returns the matching subclass, such as
    [SpaceAxis][abczarr.ome.v0_6dev3.systems.SpaceAxis].
    """

    name: Required[str] = field(factory=False)
    type: Recommended[tx.Union[AxisType, str]]
    discrete: Optional[bool]
    unit: Recommended[tx.Union[Unit, str]]
    longName: Optional[str]


@register_subclass(type="space")
class SpaceAxis(Axis):
    """A `SpaceAxis` is a spatial axis, such as `x`, `y`, or `z`, with a length
    unit.
    """

    type: Recommended[tx.Literal["space"]]
    unit: Recommended[SpaceUnit]


@register_subclass(type="time")
class TimeAxis(Axis):
    """A `TimeAxis` is a temporal axis, with a duration unit."""

    type: Recommended[tx.Literal["time"]]
    unit: Recommended[TimeUnit]


@register_subclass(type="channel")
class ChannelAxis(Axis):
    """A `ChannelAxis` indexes an image's channels. It carries no physical
    unit.
    """

    type: Recommended[tx.Literal["channel"]]
    unit: NotRecommended[Unit]


@register_subclass(type="array")
class ArrayAxis(Axis):
    """An `ArrayAxis` is a discrete axis of an array's own coordinate system.

    Its values are raw pixel or voxel indices, before any transformation
    into a physical or temporal coordinate system.
    """

    type: Required[tx.Literal["array"]]
    unit: NotRecommended[Unit]


@register_subclass(type="displacement")
class DisplacementAxis(Axis):
    """A `DisplacementAxis` is an axis whose values are the components of a
    displacement vector.

    It labels a dimension of the array that a
    [Displacements][abczarr.ome.v0_6dev3.transformations.Displacements]
    transformation reads its offsets from.
    """

    type: Required[tx.Literal["displacement"]]
    unit: NotRecommended[Unit]


@register_subclass(type="coordinate")
class CoordinateAxis(Axis):
    """A `CoordinateAxis` is an axis whose values are the components of a
    coordinate vector.

    It labels a dimension of the array that a
    [Coordinates][abczarr.ome.v0_6dev3.transformations.Coordinates]
    transformation reads its positions from.
    """

    type: Required[tx.Literal["coordinate"]]
    unit: NotRecommended[Unit]


@autodefine
class CoordinateSystem(OMEMetadata):
    """A `CoordinateSystem` is a named set of axes: a space that data and
    transformations can refer to by name.

    `axes` lists its [Axis][abczarr.ome.v0_6dev3.systems.Axis] objects in
    order. That order is the order every coordinate tuple in this coordinate
    system uses. A [Dataset][abczarr.ome.v0_6dev3.images.Dataset] or a
    `CoordinateTransformation` names a `CoordinateSystem` by its `name` as
    its input or output.
    """

    name: Required[str] = field(factory=False)
    axes: Required[tx.List[Axis]]

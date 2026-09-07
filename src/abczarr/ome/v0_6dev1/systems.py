"""Coordinate systems: the named axes a transformation maps between."""

__all__ = [
    "Axis", "SpaceAxis", "TimeAxis", "ChannelAxis", "ArrayAxis",
    "DisplacementAxis", "CoordinateAxis", "CoordinateSystem",
    "AxisType", "SpaceUnit", "TimeUnit", "Unit",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import (
    NotRecommended,
    Optional,
    Recommended,
    Required,
)

# locals
from ..base import OMEMetadata

# typing
AxisType = tx.Literal[
    "array", "space", "time", "channel", "coordinate", "displacement"
]

SpaceUnit = tx.Literal[
    'angstrom', 'attometer', 'centimeter', 'decimeter', 'exameter',
    'femtometer', 'foot', 'gigameter', 'hectometer', 'inch', 'kilometer',
    'megameter', 'meter', 'micrometer', 'mile', 'millimeter', 'nanometer',
    'parsec', 'petameter', 'picometer', 'terameter', 'yard', 'yoctometer',
    'yottameter', 'zeptometer', 'zettameter'
]

TimeUnit = tx.Literal[
    'attosecond', 'centisecond', 'day', 'decisecond', 'exasecond',
    'femtosecond', 'gigasecond', 'hectosecond', 'hour', 'kilosecond',
    'megasecond', 'microsecond', 'millisecond', 'minute', 'nanosecond',
    'petasecond', 'picosecond', 'second', 'terasecond', 'yoctosecond',
    'yottasecond', 'zeptosecond', 'zettasecond'
]

Unit = tx.Union[SpaceUnit, TimeUnit]


@autodefine
class Axis(OMEMetadata):
    """Describes one dimension of a
    [CoordinateSystem][abczarr.ome.v0_6dev1.systems.CoordinateSystem].

    Constructing an `Axis` with a recognized `type` returns the matching
    subclass, such as [SpaceAxis][abczarr.ome.v0_6dev1.systems.SpaceAxis].

    Parameters
    ----------
    name : str
        The axis's label, such as `"x"` or `"channel"`.
    type : str
        What kind of axis this is: `"space"`, `"time"`, `"channel"`,
        `"array"`, `"displacement"`, or `"coordinate"`. Recommended.
    discrete : bool
        Marks an axis whose values are integer indices rather than
        continuous coordinates. Optional.
    unit : str
        The axis's physical unit, when it has one. Recommended.
    longName : str
        A human-readable label for the axis, beyond `name`. Optional.
    """

    name: Required[str] = field(factory=False)
    type: Recommended[tx.Union[AxisType, str]]
    discrete: Optional[bool]
    unit: Recommended[tx.Union[Unit, str]]
    longName: Optional[str]


@register_subclass(type="space")
class SpaceAxis(Axis):
    """A spatial axis, such as `x`, `y`, or `z`, with a length unit.

    Parameters
    ----------
    unit : str
        The axis's physical length unit, such as `"micrometer"`.
        Recommended.
    """

    type: Recommended[tx.Literal["space"]]
    unit: Recommended[SpaceUnit]


@register_subclass(type="time")
class TimeAxis(Axis):
    """A temporal axis, with a duration unit.

    Parameters
    ----------
    unit : str
        The axis's physical duration unit, such as `"second"`.
        Recommended.
    """

    type: Recommended[tx.Literal["time"]]
    unit: Recommended[TimeUnit]


@register_subclass(type="channel")
class ChannelAxis(Axis):
    """Indexes an image's channels, carrying no physical unit."""

    type: Recommended[tx.Literal["channel"]]
    unit: NotRecommended[Unit]


@register_subclass(type="array")
class ArrayAxis(Axis):
    """A discrete axis of an array's own coordinate system.

    The axis's values are raw pixel or voxel indices, before any
    transformation into a physical or temporal coordinate system.
    """

    type: Required[tx.Literal["array"]]
    unit: NotRecommended[Unit]


@register_subclass(type="displacement")
class DisplacementAxis(Axis):
    """An axis whose values are the components of a displacement vector.

    The axis labels a dimension of the array that a
    [Displacements][abczarr.ome.v0_6dev1.transformations.Displacements]
    transformation reads its offsets from.
    """

    type: Required[tx.Literal["displacement"]]
    unit: NotRecommended[Unit]


@register_subclass(type="coordinate")
class CoordinateAxis(Axis):
    """An axis whose values are the components of a coordinate vector.

    The axis labels a dimension of the array that a
    [Coordinates][abczarr.ome.v0_6dev1.transformations.Coordinates]
    transformation reads its positions from.
    """

    type: Required[tx.Literal["coordinate"]]
    unit: NotRecommended[Unit]


@autodefine
class CoordinateSystem(OMEMetadata):
    """A named set of axes.

    A coordinate system is a space that data and transformations refer to
    by name. A [Dataset][abczarr.ome.v0_6dev1.images.Dataset] or a
    `CoordinateTransformation` names a `CoordinateSystem` by its `name` as
    its input or output.

    Parameters
    ----------
    name : str
        The coordinate system's name.
    axes : list of Axis
        The system's [Axis][abczarr.ome.v0_6dev1.systems.Axis] objects, in
        order. That order is the order every coordinate tuple in this
        coordinate system uses.
    """

    name: Required[str] = field(factory=False)
    axes: Required[tx.List[Axis]]

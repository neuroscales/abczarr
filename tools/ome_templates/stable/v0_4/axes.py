"""An axis of a multiscale pyramid: its name, type, and unit."""

__all__ = [
    "Axis", "SpaceAxis", "TimeAxis", "ChannelAxis",
    "AxisType", "SpaceUnit", "TimeUnit", "Unit",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import NotRecommended, Recommended, Required

# locals
from ..base import OMEMetadata

# typing
AxisType = tx.Literal["space", "time", "channel"]

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
    """One dimension of a
    [Multiscale][abczarr.ome.metadata.v0_1.images.Multiscale] pyramid.

    `name` is the axis's label, such as `"x"` or `"channel"`. Its
    position in a [Multiscale][abczarr.ome.metadata.v0_1.images.Multiscale]'s
    `axes` list is its position in every array shape and every
    coordinate transformation the pyramid carries. `type` says what
    kind of axis it is: `"space"`, `"time"`, or `"channel"`. `unit`
    is its physical unit. Constructing with `type="space"` gives back
    a [SpaceAxis][abczarr.ome.metadata.v0_1.axes.SpaceAxis], and
    likewise for `"time"` and `"channel"`, each restricting `unit` to
    the units that type allows.
    """

    name: Required[str] = field(factory=False)
    type: Recommended[tx.Union[AxisType, str]]
    unit: Recommended[tx.Union[Unit, str]]


@register_subclass(type="space")
class SpaceAxis(Axis):
    """A spatial axis (`x`, `y`, or `z`), with a length unit."""

    type: Recommended[tx.Literal["space"]]
    unit: Recommended[SpaceUnit]


@register_subclass(type="time")
class TimeAxis(Axis):
    """A time axis, with a duration unit."""

    type: Recommended[tx.Literal["time"]]
    unit: Recommended[TimeUnit]


@register_subclass(type="channel")
class ChannelAxis(Axis):
    """A channel axis. It carries no physical unit."""

    type: Recommended[tx.Literal["channel"]]
    unit: NotRecommended[Unit]

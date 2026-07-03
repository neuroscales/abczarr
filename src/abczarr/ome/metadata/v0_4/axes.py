__all__ = [
    "Axis", "SpaceAxis", "TimeAxis", "ChannelAxis",
    "AxisType", "SpaceUnit", "TimeUnit", "Unit",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine, field
from abczarr._core.metadata import FlexibleMetadata, register_subclass

# locals
from ..rfc2119 import Required, Recommended, NotRecommended

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
class Axis(FlexibleMetadata):
    name: Required[str] = field(factory=False)
    type: Recommended[tx.Union[AxisType, str]]
    unit: Recommended[tx.Union[Unit, str]]


@register_subclass(type="space")
class SpaceAxis(Axis):
    type: Recommended[tx.Literal["space"]]
    unit: Recommended[SpaceUnit]


@register_subclass(type="time")
class TimeAxis(Axis):
    type: Recommended[tx.Literal["time"]]
    unit: Recommended[TimeUnit]


@register_subclass(type="channel")
class ChannelAxis(Axis):
    type: Recommended[tx.Literal["channel"]]
    unit: NotRecommended[Unit]

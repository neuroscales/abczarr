__all__ = [
    "Axis", "SpaceAxis", "TimeAxis", "ChannelAxis", "ArrayAxis",
    "DisplacementAxis", "CoordinateAxis",
    "AxisType", "SpaceUnit", "TimeUnit", "Unit",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import Required, Recommended, Optional, NotRecommended

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
    name: Required[str] = field(factory=False)
    type: Recommended[tx.Union[AxisType, str]]
    discrete: Optional[bool]
    unit: Recommended[tx.Union[Unit, str]]
    longName: Optional[str]


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


@register_subclass(type="array")
class ArrayAxis(Axis):
    type: Required[tx.Literal["array"]]
    unit: NotRecommended[Unit]


@register_subclass(type="displacement")
class DisplacementAxis(Axis):
    type: Required[tx.Literal["displacement"]]
    unit: NotRecommended[Unit]


@register_subclass(type="coordinate")
class CoordinateAxis(Axis):
    type: Required[tx.Literal["coordinate"]]
    unit: NotRecommended[Unit]


@autodefine
class CoordinateSystem(OMEMetadata):
    name: Required[str] = field(factory=False)
    axes: Required[tx.List[Axis]]

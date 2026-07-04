__all__ = [
    "Axis", "SpaceAxis", "TimeAxis", "ChannelAxis",
    "AxisType", "SpaceUnit", "TimeUnit", "Unit",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
NotRecommended = RequirementForTypedDict.NotRecommended
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


class Axis(OMESchemaItem):
    name: Required[str]
    type: Recommended[tx.Union[AxisType, str]]
    unit: Recommended[tx.Union[Unit, str]]


class SpaceAxis(Axis):
    type: Recommended[tx.Literal["space"]]
    unit: Recommended[SpaceUnit]


class TimeAxis(Axis):
    type: Recommended[tx.Literal["time"]]
    unit: Recommended[TimeUnit]


class ChannelAxis(Axis):
    type: Recommended[tx.Literal["channel"]]
    unit: NotRecommended[Unit]

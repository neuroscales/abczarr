__all__ = [
    # types
    "AxisType", "SpaceUnit", "TimeUnit", "Unit",
    # axes
    "Axis", "SpaceAxis", "TimeAxis", "ChannelAxis", "ArrayAxis",
    "DisplacementAxis", "CoordinateAxis",
    # coordinate system
    "CoordinateSystem"
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.validators import NotOneOfValidator
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
NotRecommended = RequirementForTypedDict.NotRecommended
List = tz.BuiltinSequence  # list | tuple

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


class AxisBase(OMESchemaItem):
    name: Required[str]
    type: Recommended[tx.Union[AxisType, str]]
    discrete: Optional[bool]
    unit: Recommended[tx.Union[Unit, str]]
    longName: Optional[str]


class SpaceAxis(AxisBase):
    type: Recommended[tx.Literal["space"]]
    unit: Recommended[SpaceUnit]


class TimeAxis(AxisBase):
    type: Recommended[tx.Literal["time"]]
    unit: Recommended[TimeUnit]


class ChannelAxis(AxisBase):
    type: Recommended[tx.Literal["channel"]]
    unit: NotRecommended[Unit]


class ArrayAxis(AxisBase):
    type: Required[tx.Literal["array"]]
    unit: NotRecommended[Unit]


class DisplacementAxis(AxisBase):
    type: Required[tx.Literal["displacement"]]
    unit: NotRecommended[Unit]


class CoordinateAxis(AxisBase):
    type: Required[tx.Literal["coordinate"]]
    unit: NotRecommended[Unit]


class OtherAxis(AxisBase):
    type: Required[tx.Annotated[
        str,
        NotOneOfValidator({"array", "space", "time", "channel", "coordinate", "displacement"})
    ]]


Axis = tx.Union[
    SpaceAxis, TimeAxis, ChannelAxis,
    ArrayAxis, DisplacementAxis, CoordinateAxis, OtherAxis
]


class CoordinateSystem(OMESchemaItem):
    name: Required[str]
    axes: Required[List[Axis]]

# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""An axis of a multiscale pyramid: its name, type, and unit."""

__all__ = [
    "Axis",
    "SpaceAxis",
    "TimeAxis",
    "ChannelAxis",
    "AxisType",
    "SpaceUnit",
    "TimeUnit",
    "Unit",
]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import NotRecommended, Recommended, Required

from ..base import OMEMetadata

AxisType = tx.Literal["space", "time", "channel"]
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
    """One dimension of a
    [Multiscale][abczarr.ome.v0_5.images.Multiscale] pyramid.

    An axis's position in a
    [Multiscale][abczarr.ome.v0_5.images.Multiscale]'s `axes` list is its
    position in every array shape and every coordinate transformation the
    pyramid carries. Constructing an `Axis` with `type="space"` returns a
    [SpaceAxis][abczarr.ome.v0_5.axes.SpaceAxis], and likewise for
    `"time"` and `"channel"`, each restricting `unit` to the units that
    type allows.
    """

    name: Required[str] = field(factory=False)
    """The axis's label, such as `"x"` or `"channel"`."""
    type: Recommended[tx.Union[AxisType, str]]
    """What kind of axis this is: `"space"`, `"time"`, or `"channel"`.
    Recommended."""
    unit: Recommended[tx.Union[Unit, str]]
    """The axis's physical unit. Recommended."""


@register_subclass(type="space")
class SpaceAxis(Axis):
    """A spatial axis: `x`, `y`, or `z`, with a length unit."""

    type: Recommended[tx.Literal["space"]]
    unit: Recommended[SpaceUnit]
    """The axis's physical length unit, such as `"micrometer"`.
    Recommended."""


@register_subclass(type="time")
class TimeAxis(Axis):
    """A time axis, with a duration unit."""

    type: Recommended[tx.Literal["time"]]
    unit: Recommended[TimeUnit]
    """The axis's physical duration unit, such as `"second"`.
    Recommended."""


@register_subclass(type="channel")
class ChannelAxis(Axis):
    """Indexes an image's channels, carrying no physical unit."""

    type: Recommended[tx.Literal["channel"]]
    unit: NotRecommended[Unit]

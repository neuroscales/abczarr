# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""The multiscale image pyramid.

[Multiscale][abczarr.ome.v0_3.images.Multiscale] describes a
pyramid of progressively downsampled resolution levels. Each level is
a [Dataset][abczarr.ome.v0_3.images.Dataset], naming a Zarr
array and how it is positioned relative to the others.
"""

__all__ = ["Dataset", "Multiscale"]
import typing_extensions as tx

from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Optional, Recommended, Required

from ..base import OMEMetadata
from .version import Version

SpaceAxis = tx.Literal["x", "y", "z"]
TimeAxis = tx.Literal["t"]
ChannelAxis = tx.Literal["c"]
Axis = tx.Union[SpaceAxis, TimeAxis, ChannelAxis]


@autodefine
class Dataset(OMEMetadata):
    """One resolution level of a multiscale pyramid."""

    path: Required[str] = field(factory=False)
    """The name of the Zarr array holding this level, relative to the
    image group."""


@autodefine
class Multiscale(OMEMetadata):
    """A multiscale image pyramid: its axes and resolution levels."""

    @autodefine
    class Metadata(OMEMetadata):
        """Free-form detail about how a pyramid's lower resolutions were
        generated.
        """

        method: Optional[str]
        """The name of the downsampling function. Optional."""
        version: Optional[str]
        """The version of the software that ran `method`. Optional."""
        args: Optional[tz.Json]
        """The positional arguments `method` was called with. The
        upstream corpus writes this as a bare string as well as a list,
        so it is read as any JSON value rather than coerced into a
        list. Optional."""
        kwargs: Optional[tx.Dict[str, tz.Json]]
        """The keyword arguments `method` was called with. Optional."""

    axes: Required[tx.List[Axis]]
    """The pyramid's dimensions, named and ordered as `t`, `c`, `z`, `y`,
    `x`, in whatever subset and order the image uses."""
    datasets: Required[tx.List[Dataset]]
    """The pyramid's resolution levels, from full resolution down. Each
    entry is a [Dataset][abczarr.ome.v0_3.images.Dataset]."""
    name: Recommended[str]
    """A name for the multiscale image. Recommended."""
    type: Recommended[str]
    """The method used to generate the lower resolutions, such as
    ``"gaussian"``. Recommended."""
    metadata: Recommended[Metadata]
    """Further, free-form detail about how the lower resolutions were
    generated. Recommended."""
    version: Required[Version]
    """The OME-NGFF version the metadata is written against.
    Recommended."""

# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""The multiscale image pyramid.

[Multiscale][abczarr.ome.v0_4.images.Multiscale] describes a
pyramid of progressively downsampled resolution levels. Each level is
a [Dataset][abczarr.ome.v0_4.images.Dataset], naming a Zarr
array and how it is positioned relative to the others.
"""

__all__ = ["Dataset", "Multiscale"]
import typing_extensions as tx

from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.rfc2119 import Optional, Recommended, Required

from ..base import OMEMetadata
from .axes import Axis
from .transformations import CoordinateTransformation, Scale, Translation
from .version import Version


@autodefine
class Dataset(OMEMetadata):
    """One resolution level of a multiscale pyramid."""

    path: Required[str] = field(factory=False)
    """The name of the Zarr array holding this level, relative to the
    image group."""
    coordinateTransformations: Required[
        tx.Union[tx.Tuple[Scale], tx.Tuple[Scale, Translation]]
    ]
    """Places this level in the pyramid's physical space: a
    [Scale][abczarr.ome.v0_4.transformations.Scale], optionally followed
    by a [Translation][abczarr.ome.v0_4.transformations.Translation],
    one value per axis."""


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
    """The pyramid's dimensions, in the order every array shape and every
    coordinate transformation the pyramid carries uses. Each entry is
    an [Axis][abczarr.ome.v0_4.axes.Axis]."""
    datasets: Required[tx.List[Dataset]]
    """The pyramid's resolution levels, from full resolution down. Each
    entry is a [Dataset][abczarr.ome.v0_4.images.Dataset]."""
    coordinateTransformations: Optional[tx.List[CoordinateTransformation]]
    """Transformations applied to every level, before that level's own
    transformations run. Optional."""
    name: Recommended[str]
    """A name for the multiscale image. Recommended."""
    type: Recommended[str]
    """The method used to generate the lower resolutions, such as
    ``"gaussian"``. Recommended."""
    metadata: Recommended[Metadata]
    """Further, free-form detail about how the lower resolutions were
    generated. Recommended."""
    version: Required[Version]
    """The OME-NGFF version the metadata is written against."""

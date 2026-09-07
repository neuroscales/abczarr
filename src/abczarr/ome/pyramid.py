"""Build a multiscale image pyramid from a base-resolution array.

A multiscale image is a stack of the same image at falling resolutions.
[downsample_array][abczarr.ome.pyramid.downsample_array] writes one coarser
copy of an array back into its group.
[create_pyramid][abczarr.ome.pyramid.create_pyramid] applies that repeatedly
to build a whole pyramid and records it as OME-Zarr multiscales metadata.

Each coarser level is a windowed reduction of the level above it, in the way
`dask.array.coarsen` reduces every block of voxels to a single value. Each
axis halves by default. The `factor` argument sets how much each axis
shrinks, and an axis given a factor of 1 keeps its full resolution, which
suits a channel or time axis whose values should not be blended together.

The metadata is produced by an
[ImageConfig][abczarr.ome.config.ImageConfig], so the coordinate transform on
each level follows the same factors the data was downsampled by. The pyramid
lowers to any OME version. A version below 0.6 that cannot represent part of
the metadata drops that part under the chosen conversion policy.
"""

__all__ = [
    "downsample_array",
    "create_pyramid",
    "default_levels",
]

# stdlib
import math
from collections import abc as _abc

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz

# abc + ome
from abczarr.abc.sync import ZarrArray, ZarrGroup

from .base import ConversionPolicy
from .config import ImageConfig

#: How much each axis shrinks per level. One number applies to every axis. A
#: sequence gives one factor per axis. A mapping keys a factor by axis index
#: or dimension name, and a `None` key sets the default for the axes it does
#: not mention (2, a halving, when there is no `None` key).
FactorSpec = tx.Union[
    int,
    tx.Sequence[int],
    tx.Mapping[tx.Optional[tx.Union[int, str]], int],
]

#: The `dask.array` reduction each named mode uses. The reductions ignore
#: NaNs, so a padded or trimmed edge does not poison a coarse voxel.
_REDUCTIONS = {
    "mean": "nanmean",
    "median": "nanmedian",
    "min": "nanmin",
    "max": "nanmax",
    "sum": "nansum",
}


# ----------------------------------------------------------------------
#   factor resolution
# ----------------------------------------------------------------------


def _axis_index(
    key: tx.Union[int, str],
    ndim: int,
    names: tx.Optional[tx.Sequence[tx.Optional[str]]],
) -> int:
    """The axis that an index-or-name *key* refers to."""
    if isinstance(key, int) and not isinstance(key, bool):
        if not -ndim <= key < ndim:
            raise ValueError(f"axis {key} is out of range for {ndim} axes")
        return key % ndim
    if names and key in names:
        return list(names).index(key)
    known = (
        f"; the axes are {list(names)}"
        if names
        else "; the array has no dimension names"
    )
    raise ValueError(f"no axis named {key!r}{known}")


def _resolve_factors(
    factor: FactorSpec,
    ndim: int,
    names: tx.Optional[tx.Sequence[tx.Optional[str]]] = None,
) -> tz.Shape:
    """Turn a *factor* spec into one integer factor per axis.

    An `int` applies to every axis. A sequence gives one factor per axis, and
    its length must be *ndim*. A mapping keys a factor by axis index or
    dimension name. A `None` key in the mapping sets the default for the axes
    it does not mention, and that default is 2 when there is no `None` key.
    """
    if isinstance(factor, int) and not isinstance(factor, bool):
        return (factor,) * ndim
    if isinstance(factor, _abc.Mapping):
        default = factor.get(None, 2)
        resolved = [default] * ndim
        for key, value in factor.items():
            if key is None:
                continue
            resolved[_axis_index(key, ndim, names)] = int(value)
        return tuple(resolved)
    factors = tuple(int(f) for f in factor)
    if len(factors) != ndim:
        raise ValueError(f"factor has {len(factors)} entries for {ndim} axes")
    return factors


def default_levels(
    shape: tz.ShapeLike,
    chunks: tz.ShapeLike,
    factor: FactorSpec = 2,
) -> int:
    """The number of extra levels that shrinks an array down to one chunk.

    The count is how many times an axis can be divided by its factor before it
    reaches its chunk size, taken over the axis that needs the most divisions.
    The result is never negative. An axis with a factor of 1 is left out of the
    count.

    Parameters
    ----------
    shape : tuple of int
        The full-resolution array's shape.
    chunks : tuple of int
        The full-resolution array's chunk shape.
    factor : int, sequence or mapping, optional
        How much each axis shrinks per level, as in
        [downsample_array][abczarr.ome.pyramid.downsample_array]. A mapping
        keyed by dimension name needs the array, so pass the resolved per-axis
        factors here instead.
    """
    factors = _resolve_factors(factor, len(tuple(shape)))
    counts = [
        int(math.ceil(math.log(size / chunk, f)))
        for f, size, chunk in zip(factors, shape, chunks)
        if f > 1 and chunk and size > chunk
    ]
    return max(max(counts, default=0), 0)


# ----------------------------------------------------------------------
#   one level
# ----------------------------------------------------------------------


def downsample_array(
    group: ZarrGroup,
    source: str,
    target: str,
    *,
    factor: FactorSpec = 2,
    reduction: str = "mean",
) -> ZarrArray:
    """Write *target* as *source* coarsened by *factor*.

    The array named *source* is read from *group* and shrunk one axis at a
    time by that axis's factor, with a windowed *reduction*. An axis whose
    factor is 1, or that is already length one, keeps its resolution. An axis
    whose length is not a multiple of its factor is trimmed to the largest
    multiple before the reduction. The result is written as a new array named
    *target* in the same group and returned.

    Parameters
    ----------
    group : ZarrGroup
        The group that holds *source* and receives *target*.
    source : str
        The name of the array to downsample.
    target : str
        The name to give the downsampled array.
    factor : int, sequence or mapping, optional
        How much each axis shrinks. A single `int` (the default, 2) halves
        every axis. A sequence gives one factor per axis. A mapping keys a
        factor by axis index or dimension name and halves the rest.
    reduction : str, optional
        How to combine each window of voxels: ``"mean"`` (the default),
        ``"median"``, ``"min"``, ``"max"``, or ``"sum"``.

    Returns
    -------
    ZarrArray
        The new coarser array.
    """
    import dask.array as da

    if reduction not in _REDUCTIONS:
        raise ValueError(
            f"unknown reduction {reduction!r}; "
            f"choose from {sorted(_REDUCTIONS)}"
        )
    src = group[source]
    darr = src.to_dask()
    names = getattr(src.metadata, "dimension_names", None)
    factors = _resolve_factors(factor, darr.ndim, names)
    reducer = getattr(da, _REDUCTIONS[reduction])
    # An axis with a factor of 1, or already length one, is left uncoarsened.
    coarsen_by = {
        axis: (f if f > 1 and darr.shape[axis] > 1 else 1)
        for axis, f in enumerate(factors)
    }
    coarse = da.coarsen(reducer, darr, coarsen_by, trim_excess=True)
    coarse = coarse.astype(darr.dtype)
    made = group.create_array(
        target,
        shape=coarse.shape,
        dtype=src.dtype,
        chunks=_fit_chunks(src.chunks, coarse.shape),
        dimension_names=names,
    )
    made.store(coarse)
    return made


def _fit_chunks(chunks: tz.ShapeLike, shape: tz.ShapeLike) -> tz.Shape:
    """Clamp each chunk to the coarser axis it now sits on."""
    return tuple(min(chunk, size) for chunk, size in zip(chunks, shape))


def _level_scale(factors: tz.ShapeLike, level: int) -> tx.Any:
    """The cumulative shrink at *level*, for naming a level by its scale.

    A single number is returned when every shrinking axis uses the same
    factor. Otherwise one number per axis is returned.
    """
    shrinking = {f for f in factors if f > 1}
    if len(shrinking) == 1:
        return next(iter(shrinking)) ** level
    return tuple(f**level for f in factors)


# ----------------------------------------------------------------------
#   the whole pyramid
# ----------------------------------------------------------------------


def create_pyramid(
    group: ZarrGroup,
    source: str,
    *,
    levels: tx.Optional[int] = None,
    factor: FactorSpec = 2,
    reduction: str = "mean",
    strategy: tx.Union[str, int] = "window",
    scale: tx.Any = None,
    translation: tx.Any = None,
    axes: tx.Any = None,
    image_name: tx.Optional[str] = None,
    version: str = "stable",
    policy: ConversionPolicy = "warn",
    name: tx.Union[str, tx.Callable[[int], str]] = "{level}",
    write_metadata: bool = True,
) -> tx.List[ZarrArray]:
    """Build a pyramid of downsampled arrays from *source* and record it.

    Level 0 is the array already named *source*. Each further level is the one
    above it coarsened by *factor* through
    [downsample_array][abczarr.ome.pyramid.downsample_array]. Building stops
    after *levels* extra levels, or earlier once no axis can shrink further.
    Every level's array is returned, the base first and the coarsest last.

    The pyramid is recorded as OME-Zarr multiscales metadata on *group*
    through an [ImageConfig][abczarr.ome.config.ImageConfig]. The metadata
    names each level's array and carries the coordinate transform that places
    it in a shared coordinate system. Those transforms are built from the same
    factors the data was downsampled by, so the metadata matches the data.

    Parameters
    ----------
    group : ZarrGroup
        The group that holds *source*. The coarser levels are written into it,
        and its OME metadata is set to describe the pyramid.
    source : str
        The name of the full-resolution array (level 0).
    levels : int, optional
        How many coarser levels to add. The default fills the pyramid down to
        about one chunk, through
        [default_levels][abczarr.ome.pyramid.default_levels].
    factor : int, sequence or mapping, optional
        How much each axis shrinks per level, as in
        [downsample_array][abczarr.ome.pyramid.downsample_array].
    reduction : str, optional
        The windowed reduction, as in
        [downsample_array][abczarr.ome.pyramid.downsample_array].
    strategy : {"window", "edge", "center"}, optional
        How each level's coordinate transform is worked out from the
        downsampling. The default, ``"window"``, matches the windowed
        reduction exactly: a coarse voxel sits at the centre of the window it
        reduced. ``"edge"`` and ``"center"`` describe the level grids from
        their shapes instead.
    scale : number, sequence or mapping, optional
        The physical size of one base-resolution voxel, as
        [ImageConfig.scale][abczarr.ome.config.ImageConfig]. Defaults to 1.
    translation : number, sequence or mapping, optional
        The offset of the base-resolution voxel grid, as
        [ImageConfig.translation][abczarr.ome.config.ImageConfig]. Defaults
        to 0.
    axes : sequence or mapping, optional
        The axes, as [ImageConfig.axes][abczarr.ome.config.ImageConfig]. The
        default reads them from the base array's `dimension_names`.
    image_name : str, optional
        The multiscale image's name.
    version : str, optional
        The OME version to record, as
        [ImageConfig.ome_version][abczarr.ome.config.ImageConfig]. Defaults to
        the latest released version.
    policy : {"warn", "strict", "lossy"}, optional
        How to treat metadata a version below 0.6 cannot hold.
    name : str or callable, optional
        How to name each coarser level. A format string is given the level
        index as ``level`` and the cumulative shrink as ``scale``, so the
        default ``"{level}"`` names levels ``"1"``, ``"2"``, and so on, and
        ``"{scale}"`` names them by factor. A callable is given the level
        index and returns the name. Level 0 keeps the name *source*.
    write_metadata : bool, optional
        Whether to record the pyramid as OME metadata on *group*. Defaults to
        `True`.

    Returns
    -------
    list of ZarrArray
        Every level, the base first and the coarsest last.
    """
    base = group[source]
    names = getattr(base.metadata, "dimension_names", None)
    factors = _resolve_factors(factor, len(base.shape), names)
    # Resolve the axes before writing any array, so a base array that cannot
    # be turned into OME axes fails before the pyramid is half-built.
    resolved_axes = (
        _resolve_axes(axes, names, len(base.shape)) if write_metadata else None
    )
    if levels is None:
        levels = default_levels(base.shape, base.chunks, factors)

    pyramid = [base]
    paths = [source]
    previous = source
    for level in range(1, levels + 1):
        if callable(name):
            target = name(level)
        else:
            target = name.format(
                level=level, scale=_level_scale(factors, level)
            )
        made = downsample_array(
            group, previous, target, factor=factors, reduction=reduction
        )
        # Nothing shrank, so a further level would only copy this one.
        if made.shape == pyramid[-1].shape:
            del group[target]
            break
        pyramid.append(made)
        paths.append(target)
        previous = target

    if write_metadata:
        _write_multiscales(
            group,
            paths,
            factors,
            resolved_axes,
            strategy=strategy,
            scale=scale,
            translation=translation,
            image_name=image_name,
            version=version,
            policy=policy,
        )
    return pyramid


def _write_multiscales(
    group: ZarrGroup,
    paths: tx.Sequence[str],
    factors: tz.Shape,
    axes: tx.Any,
    *,
    strategy: tx.Union[str, int],
    scale: tx.Any,
    translation: tx.Any,
    image_name: tx.Optional[str],
    version: str,
    policy: ConversionPolicy,
) -> None:
    """Set *group*'s OME metadata to describe the pyramid at *paths*.

    An [ImageConfig][abczarr.ome.config.ImageConfig] is built from the axes,
    the base voxel geometry, and the downsampling factors, then lowered to the
    requested version and assigned to the group. The level shapes come from the
    written arrays, so the metadata reflects the arrays exactly.
    """
    config = ImageConfig(
        axes=axes,
        scale=scale,
        translation=translation,
        factor=tuple(float(f) for f in factors),
        strategy=strategy,
        name=image_name,
        ome_version=version,
    )
    level_shapes = [tuple(group[path].shape) for path in paths]
    config.apply(
        group,
        level_shapes=level_shapes,
        level_paths=list(paths),
        policy=policy,
    )


def _resolve_axes(
    axes: tx.Any,
    names: tx.Optional[tx.Sequence[tx.Optional[str]]],
    ndim: int,
) -> tx.Any:
    """The axes for the [ImageConfig][abczarr.ome.config.ImageConfig].

    An explicit *axes* argument is used as it is. Otherwise the axes are the
    base array's dimension names. A base array with no dimension names cannot
    be turned into OME axes, so the caller is asked for *axes* instead.
    """
    if axes is not None:
        return axes
    if not names or any(n is None for n in names):
        raise ValueError(
            "the base array has no dimension names, so the axes cannot be "
            "inferred; pass axes=... or set dimension_names on the array"
        )
    return list(names)

"""Build a multiscale image pyramid from a base-resolution array.

A multiscale image is a stack of the same image at falling resolutions. The
base level and its OME metadata are written first, through an
[ImageConfig][abczarr.ome.config.ImageConfig] or by any other means.
[create_pyramid][abczarr.ome.pyramid.create_pyramid] then adds the coarser
levels and extends that metadata to describe them.

Each coarser level is a windowed reduction of the level above it, in the way
`dask.array.coarsen` reduces every block of voxels to a single value. Each
axis halves by default. The `factor` argument sets how much each axis shrinks,
and an axis given a factor of 1 keeps its full resolution, which suits a
channel or time axis whose values should not be blended together.

The coordinate transform on each new level is read from the base level's
transform and scaled by the factor, so the metadata continues to place every
level in the coordinate system the base level was written into. The version of
the existing metadata is kept.
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

#: The `dask.array` reduction each downsampling method uses. The reductions
#: ignore NaNs, so a padded or trimmed edge does not poison a coarse voxel.
_METHODS = {
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
    method: str = "mean",
) -> ZarrArray:
    """Write *target* as *source* coarsened by *factor*.

    The array named *source* is read from *group* and shrunk one axis at a
    time by that axis's factor, with a windowed reduction chosen by *method*.
    An axis whose factor is 1, or that is already length one, keeps its
    resolution. An axis whose length is not a multiple of its factor is
    trimmed to the largest multiple before the reduction. The result is
    written as a new array named *target* in the same group and returned.

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
    method : str, optional
        How to combine each window of voxels: ``"mean"`` (the default),
        ``"median"``, ``"min"``, ``"max"``, or ``"sum"``.

    Returns
    -------
    ZarrArray
        The new coarser array.
    """
    import dask.array as da

    if method not in _METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(_METHODS)}"
        )
    src = group[source]
    darr = src.to_dask()
    names = getattr(src.metadata, "dimension_names", None)
    factors = _resolve_factors(factor, darr.ndim, names)
    reducer = getattr(da, _METHODS[method])
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
    method: str = "mean",
    name: tx.Union[str, tx.Callable[[int], str]] = "{level}",
) -> tx.List[ZarrArray]:
    """Add downsampled levels below *source* and record them in the metadata.

    Level 0 is the array already named *source*, and its OME metadata is
    already written on *group*. Each further level is the one above it
    coarsened by *factor* through
    [downsample_array][abczarr.ome.pyramid.downsample_array]. Building stops
    after *levels* extra levels, or earlier once no axis can shrink further.
    Every level's array is returned, the base first and the coarsest last.

    The group's OME metadata is read and extended to name the new levels. Each
    new level's coordinate transform is the base level's transform scaled by
    the factor, so every level stays in the coordinate system the base level
    was written into. The version of the existing metadata is kept.

    Parameters
    ----------
    group : ZarrGroup
        The group that holds *source* and its OME metadata. The coarser levels
        are written into the group, and the metadata is extended to describe
        them.
    source : str
        The name of the full-resolution array (level 0).
    levels : int, optional
        How many coarser levels to add. The default fills the pyramid down to
        about one chunk, through
        [default_levels][abczarr.ome.pyramid.default_levels].
    factor : int, sequence or mapping, optional
        How much each axis shrinks per level, as in
        [downsample_array][abczarr.ome.pyramid.downsample_array].
    method : str, optional
        The windowed reduction, as in
        [downsample_array][abczarr.ome.pyramid.downsample_array].
    name : str or callable, optional
        How to name each coarser level. A format string is given the level
        index as ``level`` and the cumulative shrink as ``scale``, so the
        default ``"{level}"`` names levels ``"1"``, ``"2"``, and so on, and
        ``"{scale}"`` names them by factor. A callable is given the level
        index and returns the name. Level 0 keeps the name *source*.

    Returns
    -------
    list of ZarrArray
        Every level, the base first and the coarsest last.

    Raises
    ------
    ValueError
        If *group* has no OME metadata to extend.
    """
    ome = group.ome
    if ome is None:
        raise ValueError(
            "the group has no OME metadata to extend; write the base level's "
            "metadata first, for example with ImageConfig(...).apply(group)"
        )
    multiscale = ome.multiscales[0]
    base = group[source]
    names = getattr(base.metadata, "dimension_names", None)
    factors = _resolve_factors(factor, len(base.shape), names)
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
            group, previous, target, factor=factors, method=method
        )
        # Nothing shrank, so a further level would only copy this one.
        if made.shape == pyramid[-1].shape:
            del group[target]
            break
        pyramid.append(made)
        paths.append(target)
        previous = target

    _extend_metadata(group, ome, multiscale, source, paths, factors)
    return pyramid


# ----------------------------------------------------------------------
#   extending the existing metadata
# ----------------------------------------------------------------------


def _extend_metadata(
    group: ZarrGroup,
    ome: tx.Any,
    multiscale: tx.Any,
    source: str,
    paths: tx.Sequence[str],
    factors: tz.Shape,
) -> None:
    """Rewrite *group*'s OME metadata to include the new levels.

    The axes, the base level's voxel geometry, the image name, and the version
    are read from the existing metadata. An
    [ImageConfig][abczarr.ome.config.ImageConfig] rebuilds the multiscale from
    them with one dataset per level, and the result is assigned back to the
    group. The base level's dataset is reproduced from its own transform, so
    only the new levels are added.
    """
    scale, translation = _base_geometry(multiscale, source)
    image_name = getattr(multiscale, "name", None)
    config = ImageConfig(
        axes=_axis_specs(multiscale),
        scale=scale,
        translation=translation,
        factor=tuple(float(f) for f in factors),
        strategy="window",
        name=image_name if isinstance(image_name, str) else None,
        ome_version=ome.version,
    )
    level_shapes = [tuple(group[path].shape) for path in paths]
    config.apply(group, level_shapes=level_shapes, level_paths=list(paths))


def _axis_specs(multiscale: tx.Any) -> tx.List[tx.Dict[str, tx.Any]]:
    """The axes of *multiscale* as ``{name, type, unit}`` dicts.

    The stable versions carry the axes on the multiscale. The 0.6 pre-releases
    carry them on the first coordinate system. Both are read here.
    """
    axes = getattr(multiscale, "axes", None)
    if not axes:
        systems = getattr(multiscale, "coordinateSystems", None)
        axes = systems[0].axes if systems else None
    if not axes:
        raise ValueError("the OME metadata has no axes")
    specs = []
    for axis in axes:
        spec = {"name": axis.name}  # type: tx.Dict[str, tx.Any]
        atype = getattr(axis, "type", None)
        unit = getattr(axis, "unit", None)
        if isinstance(atype, str):
            spec["type"] = atype
        if isinstance(unit, str):
            spec["unit"] = unit
        specs.append(spec)
    return specs


def _base_geometry(
    multiscale: tx.Any, source: str
) -> tx.Tuple[tx.List[float], tx.List[float]]:
    """The base level's scale and translation, read from its transform.

    The dataset named *source* holds the base level's transform. Its scale is
    read, and its translation when it has one. A missing translation reads as
    zero on every axis.
    """
    datasets = list(multiscale.datasets)
    dataset = next((d for d in datasets if d.path == source), datasets[0])
    scale = None
    translation = None
    for transform in _flatten_transforms(dataset.coordinateTransformations):
        if getattr(transform, "type", None) == "scale":
            scale = [float(v) for v in transform.scale]
        elif getattr(transform, "type", None) == "translation":
            translation = [float(v) for v in transform.translation]
    if scale is None:
        raise ValueError(
            f"the base dataset {source!r} has no scale transform to build on"
        )
    if translation is None:
        translation = [0.0] * len(scale)
    return scale, translation


def _flatten_transforms(
    transforms: tx.Sequence[tx.Any],
) -> tx.Iterator[tx.Any]:
    """Each transform, stepping into a sequence's own transformations.

    The stable versions list the scale and the translation side by side. The
    0.6 pre-releases wrap them in one sequence. Both are flattened to the
    individual transforms here.
    """
    for transform in transforms:
        if getattr(transform, "type", None) == "sequence":
            yield from transform.transformations
        else:
            yield transform

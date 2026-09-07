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
    # A v2 array has no dimension names of its own; the OME metadata names its
    # axes instead.
    names = getattr(src.metadata, "dimension_names", None) or _ome_axis_names(
        group.ome
    )
    factors = _resolve_factors(factor, darr.ndim, names)
    reducer = getattr(da, _METHODS[method])
    # An axis with a factor of 1, or already length one, is left uncoarsened.
    coarsen_by = {
        axis: (f if f > 1 and darr.shape[axis] > 1 else 1)
        for axis, f in enumerate(factors)
    }
    coarse = da.coarsen(reducer, darr, coarsen_by, trim_excess=True)
    coarse = coarse.astype(darr.dtype)
    made = _create_level(group, target, src, coarse.shape, names)
    made.store(coarse)
    return made


def _create_level(
    group: ZarrGroup,
    target: str,
    source: ZarrArray,
    shape: tz.ShapeLike,
    names: tx.Optional[tx.Sequence[tx.Optional[str]]],
) -> ZarrArray:
    """Create *target* in *group* as a coarser copy of *source*.

    The new level reuses the base array's metadata, so its data type, chunk
    grid, codecs, compressor, fill value, and every other stored option match
    the base level. Only the shape and the dimension names change. A backend's
    array creation expresses fewer options than its metadata can hold, so the
    level is built from the base metadata directly to carry the base encoding
    across in full. The chunk and shard shapes are kept as they are, so a
    coarser level holds fewer chunks of the same size rather than smaller ones.
    """
    from ..api.entrypoint import create
    from ..drivers._metadata import metadata_from_json

    document = dict(source.metadata.to_json())
    document["shape"] = list(shape)
    document["attributes"] = {}
    if names is not None and "dimension_names" in document:
        document["dimension_names"] = list(names)
    location = "{}/{}".format(str(group.store_path).rstrip("/"), target)
    return create(location, metadata_from_json(document))


def _level_scale(factors: tz.ShapeLike, level: int) -> tx.Any:
    """The cumulative shrink at *level*, for naming a level by its scale.

    A single number is returned when every axis uses the same factor, so
    ``"s{scale}"`` names a halving pyramid ``s2``, ``s4``, and so on. When the
    axes shrink by different factors, the per-axis shrink is joined with ``x``,
    so a ``(2, 2, 2, 1)`` factor names the first level ``s2x2x2x1``.
    """
    values = [f**level for f in factors]
    if len(set(factors)) == 1:
        return values[0]
    return "x".join(str(v) for v in values)


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
    base = group[source]
    names = getattr(base.metadata, "dimension_names", None) or _ome_axis_names(
        ome
    )
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

    _extend_metadata(group, ome, source, paths, factors)
    return pyramid


# ----------------------------------------------------------------------
#   extending the existing metadata
# ----------------------------------------------------------------------


def _extend_metadata(
    group: ZarrGroup,
    ome: tx.Any,
    source: str,
    paths: tx.Sequence[str],
    factors: tz.Shape,
) -> None:
    """Rewrite *group*'s OME metadata to include the new levels.

    The metadata is read as its JSON document and only the new levels are
    added, so everything else it carries is kept, whether or not it was written
    by abczarr's own tooling. Each new level's transform is the base level's
    scale and translation, scaled by the factor for that level. The version of
    the document is unchanged.
    """
    document = ome.to_json()
    multiscale = _first_multiscale(document)
    datasets = multiscale["datasets"]
    scale, offset, shape = _base_transform(_base_dataset(datasets, source))
    for level, path in enumerate(paths[1:], start=1):
        level_scale = [s * f**level for s, f in zip(scale, factors)]
        level_offset = [
            o + s * (f**level - 1) / 2
            for o, s, f in zip(offset, scale, factors)
        ]
        datasets.append(_level_dataset(path, level_scale, level_offset, shape))
    group.ome = document


def _first_multiscale(document: tx.Any) -> tx.Any:
    """The first multiscale in an OME document."""
    multiscales = document.get("multiscales")
    if not multiscales:
        raise ValueError("the OME metadata has no multiscale to extend")
    return multiscales[0]


def _base_dataset(datasets: tx.Sequence[tx.Any], source: str) -> tx.Any:
    """The dataset for the base level named *source*, or the first one."""
    for dataset in datasets:
        if dataset.get("path") == source:
            return dataset
    return datasets[0]


def _base_transform(
    dataset: tx.Any,
) -> tx.Tuple[tx.List[float], tx.List[float], tx.Tuple[bool, tx.Any]]:
    """The base level's scale and physical offset, and its transform shape.

    The scale and the translation are read from the dataset's transforms. A
    translation applied before the scale is against the spec but seen in the
    wild. Its stored values are then in input units, so they are scaled to the
    physical offset. A transform that carries an input or an output reference
    marks a 0.6 dataset, and the output reference is kept so that the new
    levels map into the same coordinate system.
    """
    flat, has_refs, output = _flatten_json(
        dataset["coordinateTransformations"]
    )
    scale = None
    translation = None
    order = []
    for transform in flat:
        kind = transform.get("type")
        if kind == "scale":
            scale = [float(v) for v in transform["scale"]]
            order.append("scale")
        elif kind == "translation":
            translation = [float(v) for v in transform["translation"]]
            order.append("translation")
    if scale is None:
        raise ValueError("the base dataset has no scale transform to build on")
    if translation is None:
        offset = [0.0] * len(scale)
    elif order[:2] == ["translation", "scale"]:
        offset = [s * t for s, t in zip(scale, translation)]
    else:
        offset = list(translation)
    return scale, offset, (has_refs, output)


def _flatten_json(
    transforms: tx.Sequence[tx.Any],
) -> tx.Tuple[tx.List[tx.Any], bool, tx.Any]:
    """The individual transforms, and whether they carry system references.

    A 0.6 dataset wraps its transforms in one sequence and names its input and
    output coordinate systems. The individual transforms are returned, along
    with whether such references are present and the output reference to reuse.
    """
    flat = []
    has_refs = False
    output = None
    for transform in transforms:
        if transform.get("input") is not None:
            has_refs = True
        if transform.get("output") is not None:
            has_refs = True
            output = transform["output"]
        if transform.get("type") == "sequence":
            flat.extend(transform.get("transformations", []))
        else:
            flat.append(transform)
    return flat, has_refs, output


def _level_dataset(
    path: str,
    scale: tx.Sequence[float],
    translation: tx.Sequence[float],
    shape: tx.Tuple[bool, tx.Any],
) -> tx.Dict[str, tx.Any]:
    """A new level's dataset, in the transform shape the base level used."""
    has_refs, output = shape
    transforms = [
        {"type": "scale", "scale": list(scale)}
    ]  # type: tx.List[tx.Dict[str, tx.Any]]
    if any(offset != 0 for offset in translation):
        transforms.append(
            {"type": "translation", "translation": list(translation)}
        )
    if has_refs:
        sequence = {
            "type": "sequence",
            "input": {"path": path},
            "transformations": transforms,
        }  # type: tx.Dict[str, tx.Any]
        if output is not None:
            sequence["output"] = output
        return {"path": path, "coordinateTransformations": [sequence]}
    return {"path": path, "coordinateTransformations": transforms}


def _ome_axis_names(ome: tx.Any) -> tx.Optional[tx.Tuple[str, ...]]:
    """The axis names in a group's OME metadata, or `None` when unavailable.

    A v2 array carries no dimension names of its own. Its axis names are then
    read from the OME metadata, where the stable versions name the axes on the
    multiscale and the 0.6 pre-releases name them on the first coordinate
    system.
    """
    if ome is None:
        return None
    try:
        multiscale = ome.multiscales[0]
        axes = getattr(multiscale, "axes", None)
        if not axes:
            systems = getattr(multiscale, "coordinateSystems", None)
            axes = systems[0].axes if systems else None
        return tuple(axis.name for axis in axes) if axes else None
    except (AttributeError, IndexError, TypeError):
        return None

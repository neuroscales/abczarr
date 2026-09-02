"""Build a multiscale image pyramid from a base-resolution array.

[downsample_array][abczarr.ome.pyramid.downsample_array] reads one array
from a group and writes a coarser copy of it back into the same group.
[create_pyramid][abczarr.ome.pyramid.create_pyramid] applies that
repeatedly to build a whole pyramid, with the base array as level 0.

!!! example
    ```python
    from abczarr.ome import create_pyramid

    levels = create_pyramid(group, "0")  # "0" is the full-resolution array
    # levels[0] is group["0"]; levels[1] is half the size, and so on
    ```

Each coarser level shrinks the array with a windowed reduction, the way
`dask.array.coarsen` does. By default every axis is halved; the `factor`
argument sets how much each axis shrinks, and an axis given a factor of 1
is left at full resolution, which is what you want for a channel or time
axis that should not be blurred together.
"""

from __future__ import annotations

__all__ = [
    "downsample_array",
    "create_pyramid",
    "default_levels",
]

# stdlib
import math

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz

if tx.TYPE_CHECKING:
    from abczarr.abc.array import ZarrArray
    from abczarr.abc.group import ZarrGroup

#: How much each axis shrinks per level. One number applies to every axis;
#: a sequence gives one per axis; a mapping keys a factor by axis index or
#: dimension name, halving any axis it does not mention.
FactorSpec = tx.Union[
    int, "tx.Sequence[int]", "tx.Mapping[tx.Union[int, str], int]"
]

#: The `dask.array` reduction each named downsampling mode uses. The window
#: reductions ignore NaNs, so a padded edge does not poison a coarse voxel.
_REDUCTIONS = {
    "mean": "nanmean",
    "median": "nanmedian",
    "min": "nanmin",
    "max": "nanmax",
    "sum": "nansum",
}


def _axis_index(
    key: tx.Union[int, str],
    ndim: int,
    names: tx.Optional[tx.Sequence[tx.Optional[str]]],
) -> int:
    """The axis an index-or-name *key* refers to."""
    if isinstance(key, int) and not isinstance(key, bool):
        if not -ndim <= key < ndim:
            raise ValueError(f"axis {key} is out of range for {ndim} axes")
        return key % ndim
    if names and key in names:
        return list(names).index(key)
    known = f"; the axes are {list(names)}" if names else (
        "; the array has no dimension names"
    )
    raise ValueError(f"no axis named {key!r}{known}")


def _resolve_factors(
    factor: FactorSpec,
    ndim: int,
    names: tx.Optional[tx.Sequence[tx.Optional[str]]] = None,
) -> tz.Shape:
    """Turn a *factor* spec into one factor per axis.

    - an `int` applies to every axis;
    - a sequence gives one factor per axis (its length must be *ndim*);
    - a mapping keys a factor by axis index or dimension name, and every
      axis it does not mention is halved.
    """
    if isinstance(factor, int) and not isinstance(factor, bool):
        return (factor,) * ndim
    if hasattr(factor, "keys"):  # a mapping of some axes to their factor
        resolved = [2] * ndim
        for key, value in factor.items():  # type: ignore[union-attr]
            resolved[_axis_index(key, ndim, names)] = int(value)
        return tuple(resolved)
    factors = tuple(int(f) for f in factor)  # type: ignore[union-attr]
    if len(factors) != ndim:
        raise ValueError(
            f"factor has {len(factors)} entries for {ndim} axes"
        )
    return factors


def default_levels(
    shape: tz.ShapeLike,
    chunks: tz.ShapeLike,
    factor: FactorSpec = 2,
) -> int:
    """The number of extra levels that shrinks the array down to a chunk.

    Counts how many times each axis can be divided by its factor before it
    reaches its chunk size, and returns the largest such count (never less
    than zero). An axis with a factor of 1 is left out of the count.

    Parameters
    ----------
    shape : tuple of int
        The full-resolution array's shape.
    chunks : tuple of int
        Its chunk shape.
    factor : int, sequence or mapping, optional
        How much each axis shrinks per level, as in
        [downsample_array][abczarr.ome.pyramid.downsample_array]. A mapping
        keyed by dimension name needs the array, so pass the resolved
        per-axis factors here instead.
    """
    factors = _resolve_factors(factor, len(tuple(shape)))
    counts = [
        int(math.ceil(math.log(size / chunk, f)))
        for f, size, chunk in zip(factors, shape, chunks)
        if f > 1 and chunk and size > chunk
    ]
    return max(max(counts, default=0), 0)


def downsample_array(
    group: ZarrGroup,
    source: str,
    target: str,
    *,
    factor: FactorSpec = 2,
    reduction: str = "mean",
) -> ZarrArray:
    """Write *target*: the array *source*, coarsened by *factor*.

    Reads the array named *source* from *group*, shrinks each axis by its
    factor (leaving any axis whose factor is 1, or that is already length
    one) with the windowed *reduction*, and writes the result as a new
    array named *target* in the same group. Returns the new array.

    Parameters
    ----------
    group : ZarrGroup
        The group that holds *source* and receives *target*.
    source : str
        The name of the array to downsample.
    target : str
        The name to give the downsampled array.
    factor : int, sequence or mapping, optional
        How much each axis shrinks. A single `int` (the default, 2, halves
        every axis) applies to all axes; a sequence gives one factor per
        axis; a mapping keys a factor by axis index or dimension name and
        halves the rest. An axis with a factor of 1 is left at full
        resolution.
    reduction : str, optional
        How to combine each window of voxels: `"mean"` (the default),
        `"median"`, `"min"`, `"max"`, or `"sum"`.
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
    # a factor of 1, or an axis already length one, is left uncoarsened
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
    )
    made.store(coarse)
    return made


def _fit_chunks(chunks: tz.ShapeLike, shape: tz.ShapeLike) -> tz.Shape:
    """Clamp each chunk to the (now smaller) coarser axis it sits on."""
    return tuple(min(chunk, size) for chunk, size in zip(chunks, shape))


def _level_scale(factors: tz.ShapeLike, level: int) -> tx.Any:
    """The cumulative shrink at *level*: a single int when every shrinking
    axis uses the same factor, otherwise one per axis."""
    shrinking = {f for f in factors if f > 1}
    if len(shrinking) == 1:
        return next(iter(shrinking)) ** level
    return tuple(f ** level for f in factors)


def create_pyramid(
    group: ZarrGroup,
    source: str,
    *,
    levels: tx.Optional[int] = None,
    factor: FactorSpec = 2,
    reduction: str = "mean",
    name: tx.Union[str, tx.Callable[[int], str]] = "{level}",
) -> tx.List[ZarrArray]:
    """Build a pyramid of downsampled arrays from *source*.

    Level 0 is the existing array named *source*. Each further level is the
    one before it, coarsened by *factor* through
    [downsample_array][abczarr.ome.pyramid.downsample_array]. Building stops
    after *levels* extra levels, or earlier if no axis can shrink further.
    Returns every level's array, the base first and the coarsest last.

    Parameters
    ----------
    group : ZarrGroup
        The group that holds *source*; the coarser levels are written into
        it too.
    source : str
        The name of the full-resolution array (level 0).
    levels : int, optional
        How many coarser levels to add. The default fills the pyramid down
        to about one chunk, using
        [default_levels][abczarr.ome.pyramid.default_levels].
    factor : int, sequence or mapping, optional
        How much each axis shrinks per level, as in
        [downsample_array][abczarr.ome.pyramid.downsample_array].
    reduction : str, optional
        The windowed reduction, as in
        [downsample_array][abczarr.ome.pyramid.downsample_array].
    name : str or callable, optional
        How to name each coarser level. A format string is given the level
        index as `level` and the cumulative shrink as `scale`, so the
        default `"{level}"` names levels `"1"`, `"2"`, ... and `"{scale}"`
        names them by factor, `"2"`, `"4"`, ... for a halving pyramid. A
        callable is passed the level index and returns the name. Level 0
        keeps the name *source*.
    """
    base = group[source]
    names = getattr(base.metadata, "dimension_names", None)
    factors = _resolve_factors(factor, len(base.shape), names)
    if levels is None:
        levels = default_levels(base.shape, base.chunks, factors)
    pyramid = [base]
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
        # nothing shrank, so a further level would just copy this one
        if made.shape == pyramid[-1].shape:
            del group[target]
            break
        pyramid.append(made)
        previous = target
    return pyramid

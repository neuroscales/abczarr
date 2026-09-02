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

Each coarser level halves every axis (by default) with a windowed
reduction, the way `dask.array.coarsen` does. An axis can be held at full
resolution with `no_pyramid_axis`, which is what you want for a channel or
time axis that should not be blurred together.
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

#: The `dask.array` reduction each named downsampling mode uses. The window
#: reductions ignore NaNs, so a padded edge does not poison a coarse voxel.
_REDUCTIONS = {
    "mean": "nanmean",
    "median": "nanmedian",
    "min": "nanmin",
    "max": "nanmax",
    "sum": "nansum",
}


def default_levels(
    shape: tz.ShapeLike,
    chunks: tz.ShapeLike,
    no_pyramid_axis: tx.Optional[int] = None,
) -> int:
    """The number of extra levels that halves the array down to a chunk.

    Counts how many times each axis can be halved before it reaches its
    chunk size, and returns the largest such count (never less than zero).
    The axis named by *no_pyramid_axis*, if any, is left out of the count.

    Parameters
    ----------
    shape : tuple of int
        The full-resolution array's shape.
    chunks : tuple of int
        Its chunk shape.
    no_pyramid_axis : int, optional
        An axis to leave at full resolution.
    """
    counts = [
        int(math.ceil(math.log2(size / chunk)))
        for axis, (size, chunk) in enumerate(zip(shape, chunks))
        if axis != no_pyramid_axis and chunk and size > chunk
    ]
    return max(max(counts, default=0), 0)


def downsample_array(
    group: ZarrGroup,
    source: str,
    target: str,
    *,
    factor: int = 2,
    reduction: str = "mean",
    no_pyramid_axis: tx.Optional[int] = None,
) -> ZarrArray:
    """Write *target*: the array *source*, coarsened by *factor*.

    Reads the array named *source* from *group*, coarsens every axis by
    *factor* (except *no_pyramid_axis*, and any axis already length one)
    with the windowed *reduction*, and writes the result as a new array
    named *target* in the same group. Returns the new array.

    Parameters
    ----------
    group : ZarrGroup
        The group that holds *source* and receives *target*.
    source : str
        The name of the array to downsample.
    target : str
        The name to give the downsampled array.
    factor : int, optional
        How much to shrink each downsampled axis. The default, 2, halves.
    reduction : str, optional
        How to combine each window of voxels: `"mean"` (the default),
        `"median"`, `"min"`, `"max"`, or `"sum"`.
    no_pyramid_axis : int, optional
        An axis to leave at full resolution, such as a channel or time
        axis.
    """
    import dask.array as da

    if reduction not in _REDUCTIONS:
        raise ValueError(
            f"unknown reduction {reduction!r}; "
            f"choose from {sorted(_REDUCTIONS)}"
        )
    src = group[source]
    darr = src.to_dask()
    reducer = getattr(da, _REDUCTIONS[reduction])
    factors = {
        axis: (
            1
            if axis == no_pyramid_axis or darr.shape[axis] == 1
            else factor
        )
        for axis in range(darr.ndim)
    }
    coarse = da.coarsen(reducer, darr, factors, trim_excess=True)
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


def create_pyramid(
    group: ZarrGroup,
    source: str,
    *,
    levels: tx.Optional[int] = None,
    factor: int = 2,
    reduction: str = "mean",
    no_pyramid_axis: tx.Optional[int] = None,
    name: str = "{level}",
) -> tx.List[ZarrArray]:
    """Build a pyramid of downsampled arrays from *source*.

    Level 0 is the existing array named *source*. Each further level is the
    one before it, coarsened by *factor* through
    [downsample_array][abczarr.ome.pyramid.downsample_array]. Building stops
    after *levels* extra levels, or earlier if an axis can no longer be
    halved. Returns every level's array, the base first and the coarsest
    last.

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
    factor : int, optional
        How much each level shrinks. The default, 2, halves.
    reduction : str, optional
        The windowed reduction, as in
        [downsample_array][abczarr.ome.pyramid.downsample_array].
    no_pyramid_axis : int, optional
        An axis to leave at full resolution across every level.
    name : str, optional
        The name pattern for each coarser level, given the level index as
        `level`. The default, `"{level}"`, names them `"1"`, `"2"`, and so
        on. Level 0 keeps the name *source*.
    """
    base = group[source]
    if levels is None:
        levels = default_levels(base.shape, base.chunks, no_pyramid_axis)
    pyramid = [base]
    previous = source
    for level in range(1, levels + 1):
        target = name.format(level=level)
        made = downsample_array(
            group,
            previous,
            target,
            factor=factor,
            reduction=reduction,
            no_pyramid_axis=no_pyramid_axis,
        )
        # nothing shrank, so a further level would just copy this one
        if made.shape == pyramid[-1].shape:
            del group[target]
            break
        pyramid.append(made)
        previous = target
    return pyramid

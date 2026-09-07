"""Shapes and array computation for a multiscale pyramid's downsampled
levels.
"""

# stdlib
import logging
import math

# dependencies
import dask.array as da
import typing_extensions as tx

# locals
from . import typing as tz

# logger
logger = logging.getLogger(__name__)


def default_levels(
    spatial_shape: tz.ShapeLike,
    spatial_chunk: tz.ShapeLike,
    no_pyramid_axis: tx.Optional[int],
) -> int:
    """Compute the default number of downsampling levels for a spatial
    pyramid.

    For each axis in *spatial_shape*, except the one named by
    *no_pyramid_axis*, this counts how many times that axis can be halved
    down to its own chunk size in *spatial_chunk*, rounded up to the next
    whole level. The result is the largest of those per-axis counts, so
    every axis reaches at most its own chunk size by the last level. The
    result is never negative.

    Parameters
    ----------
    spatial_shape : tuple of int
        The full size of each spatial dimension.
    spatial_chunk : tuple of int
        The chunk size along each spatial dimension.
    no_pyramid_axis : int or None
        An axis index to exclude from the computation. `None` includes
        every axis.

    Returns
    -------
    int
        The number of pyramid levels needed to reduce every included axis
        by repeated factors of two, down to its own chunk size.
    """
    default_levels = max(
        int(math.ceil(math.log2(s / spatial_chunk[i])))
        for i, s in enumerate(spatial_shape)
        if no_pyramid_axis is None or i != no_pyramid_axis
    )
    levels = max(default_levels, 0)
    return levels


def next_level_shape(
    prev_shape: tz.ShapeLike, no_pyramid_axis: tx.Optional[int]
) -> tz.Shape:
    """Compute the shape of the next coarser pyramid level.

    Each axis in *prev_shape* is halved by integer division, with a
    minimum of 1, except the axis named by *no_pyramid_axis*, which is
    carried over unchanged.

    Parameters
    ----------
    prev_shape : sequence of int
        The shape of the current level.
    no_pyramid_axis : int or None
        An axis index to leave unchanged. `None` halves every axis.

    Returns
    -------
    tuple of int
        The next level's shape, the same length as *prev_shape*.
    """
    new_shape = []
    for i, length in enumerate(prev_shape):
        if i == no_pyramid_axis:
            new_shape.append(length)
        else:
            new_shape.append(max(1, length // 2))
    return tuple(new_shape)


def compute_next_level(
    arr: da.Array,
    ndim: int,
    no_pyramid_axis: tx.Optional[int] = None,
    window_func: tx.Callable = da.nanmean,
) -> da.Array:
    """Downsample a dask array by one pyramid level.

    The last *ndim* dimensions of *arr* are the pyramid dimensions. Any
    leading dimensions are left untouched. Each pyramid dimension of
    length greater than 1 is downsampled by a factor of two through
    *window_func*, except the one named by *no_pyramid_axis*, which is
    left unchanged. The array's dtype is preserved.

    Parameters
    ----------
    arr : dask.array.Array
        The array to downsample.
    ndim : int
        The number of pyramid dimensions, counted from the end of
        `arr.ndim`.
    no_pyramid_axis : int or None
        A pyramid-dimension index, in `range(ndim)`, to leave
        undownsampled. `None` downsamples every pyramid dimension.
    window_func : callable
        The reduction applied within each downsampling window, such as
        `dask.array.mean` or `dask.array.median`.

    Returns
    -------
    dask.array.Array
        The downsampled array. Its leading dimensions are unchanged.
        Each pyramid dimension of length ``n`` becomes ``ceil(n / 2)``,
        except the one named by *no_pyramid_axis*, which is unchanged.
    """
    # The pyramid dimensions are the last `ndim` axes of `arr`. Anything
    # before them is a leading, non-pyramid dimension.
    start = arr.ndim - ndim
    pyramid_axes = list(range(start, arr.ndim))

    # Each pyramid axis is coarsened by a factor of two, except the one
    # named by `no_pyramid_axis` and one already at length 1, both of
    # which keep a factor of one.
    factors = {
        axis: (
            1
            if (
                no_pyramid_axis is not None
                and axis == pyramid_axes[no_pyramid_axis]
            )
            or arr.shape[axis] == 1
            else 2
        )
        for axis in pyramid_axes
    }
    dtype = arr.dtype

    return da.coarsen(window_func, arr, factors, trim_excess=True).astype(
        dtype
    )

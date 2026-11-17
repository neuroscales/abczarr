"""Normalisation of NumPy-style indices against an array shape."""

import operator

import numpy as np
import typing_extensions as tx

from abczarr._core import typing as tz


def _consumed(item: tx.Any) -> int:
    """The number of array axes that one index entry selects along."""
    if item is None or item is Ellipsis or isinstance(item, (bool, np.bool_)):
        return 0
    if isinstance(item, np.ndarray) and item.dtype == bool:
        return item.ndim
    return 1


def _as_entry(item: tx.Any) -> tx.Any:
    """An index entry with lists and array-likes turned into arrays."""
    if isinstance(item, (list, np.ndarray)):
        return np.asarray(item)
    return item


def _normalize_slice(item: slice, size: int) -> slice:
    """A slice with non-negative bounds that selects the same elements."""
    start, stop, step = item.indices(size)
    if len(range(start, stop, step)) == 0:
        return slice(0, 0, 1)
    if step < 0 and stop < 0:
        return slice(start, None, step)
    return slice(start, stop, step)


def _check_bounds(value: tx.Any, size: int, axis: int) -> tx.Any:
    """`value` with negatives wrapped, raising IndexError when out of range."""
    if np.any(value < -size) or np.any(value >= size):
        raise IndexError(
            f"index {value} is out of bounds for axis {axis} with size {size}"
        )
    return value % size if size else value


def normalize_index(index: tx.Any, shape: tz.Shape) -> tx.Tuple[tx.Any, ...]:
    """Rewrite a NumPy-style `index` so that it holds no negative positions.

    Integers and integer arrays are wrapped and bounds-checked, slices are
    resolved against the axis size, and every other entry (`Ellipsis`,
    `None`, boolean arrays) is returned unchanged. The result is a tuple.
    """
    items = index if isinstance(index, tuple) else (index,)
    items = tuple(_as_entry(item) for item in items)
    n_ellipsis = sum(item is Ellipsis for item in items)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    used = sum(_consumed(item) for item in items)
    if used > len(shape):
        raise IndexError(
            f"too many indices for array: array is {len(shape)}-dimensional, "
            f"but {used} were indexed"
        )
    out = []
    axis = 0
    for item in items:
        if item is Ellipsis:
            axis += len(shape) - used
            out.append(item)
            continue
        width = _consumed(item)
        is_mask = isinstance(item, np.ndarray) and item.dtype == bool
        if width == 1 and not is_mask:
            size = shape[axis]
            if isinstance(item, slice):
                item = _normalize_slice(item, size)
            elif isinstance(item, np.ndarray):
                if not np.issubdtype(item.dtype, np.integer):
                    raise IndexError(
                        "arrays used as indices must be of integer "
                        "or boolean type"
                    )
                item = _check_bounds(item, size, axis)
            else:
                item = int(_check_bounds(operator.index(item), size, axis))
        out.append(item)
        axis += width
    return tuple(out)


def bounding_region(
    index: tx.Any, shape: tz.Shape
) -> tx.Tuple[tx.Tuple[slice, ...], tx.Tuple[tx.Any, ...]]:
    """Split `index` into a step-1 region and an index into that region.

    The first element of the result holds one step-1 slice per axis and
    covers every element that `index` selects. Applying the second element
    to the region gives the same result as applying `index` to the array.
    """
    items = []
    for item in normalize_index(index, shape):
        if isinstance(item, np.ndarray) and item.dtype == bool:
            # a boolean mask selects exactly what its nonzero() positions do
            items.extend(np.nonzero(item))
        else:
            items.append(item)
    used = sum(_consumed(item) for item in items)
    region = []
    local = []
    for item in items:
        if item is Ellipsis:
            for _ in range(len(shape) - used):
                region.append(slice(0, shape[len(region)]))
        elif _consumed(item) == 0:
            pass
        elif isinstance(item, slice) and item.step < 0:
            lo = 0 if item.stop is None else item.stop + 1
            region.append(slice(lo, item.start + 1))
            item = slice(item.start - lo, None, item.step)
        elif isinstance(item, slice):
            region.append(slice(item.start, item.stop))
            item = slice(0, item.stop - item.start, item.step)
        elif isinstance(item, np.ndarray):
            lo = int(item.min()) if item.size else 0
            hi = int(item.max()) + 1 if item.size else 0
            region.append(slice(lo, hi))
            item = item - lo
        else:
            region.append(slice(item, item + 1))
            item = 0
        local.append(item)
    region.extend(slice(0, n) for n in shape[len(region):])
    return tuple(region), tuple(local)

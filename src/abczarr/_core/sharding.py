"""Automatic chunk and shard size selection, kept within a byte budget."""

# stdlib
import math
from collections import abc

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# locals
from . import typing as tz

# constants
SHARD_FILE_SIZE_LIMIT = (
    2  # compression ratio
    * 2  # GB
    * 2**30  # GB->Bytes
)

# typing
SizeOrAuto = tx.Union[int, tx.Literal["auto"]]
ChunkSpec = tx.Union[
    SizeOrAuto,
    tx.Iterable[SizeOrAuto],
    tx.Mapping[tx.Optional[str], SizeOrAuto],
]
ChunkSize = tz.Shape


class ShardsAndChunks(tx.NamedTuple):
    """A resolved shard and chunk shape, returned together because
    neither is chosen without the other.

    Parameters
    ----------
    shards : tuple of int
        The shard size along each dimension.
    chunks : tuple of int
        The chunk size along each dimension.
    """

    shards: tz.Shape
    chunks: tz.Shape


def broadcast_spec(
    shape: tz.ShapeLike,
    spec: ChunkSpec = "auto",
    names: tx.Iterable[tx.Optional[str]] = (),
) -> tx.Tuple[tx.Union[int, tx.Literal["auto"]], ...]:
    """Spread a chunk-size specification across every dimension of
    `shape`.

    A single integer or `"auto"` applies to every dimension. A sequence
    shorter than `shape` has its last entry repeated to fill the
    remaining dimensions. One longer than `shape` is truncated. A
    mapping is read by dimension name, matched against `names`. A
    dimension with no matching name falls back to the mapping's `None`
    key, then its `""` key, then its own full size in `shape`.

    A size of zero means no chunking along that dimension, and
    `"auto"` defers the choice to `auto_chunk` or `auto_shard`.

    Parameters
    ----------
    shape : sequence of int
        The shape of the array.
    spec : int, {"auto"}, sequence, or mapping
        The chunk-size specification, in any of the forms above.
    names : sequence of str
        The name of each dimension, used to resolve `spec` when it is a
        mapping.

    Returns
    -------
    tuple of (int or {"auto"})
        The chunk or shard size along each dimension.
    """

    if isinstance(spec, (int, str)):
        spec = (spec,)

    # Right-pad chunk size
    if not isinstance(spec, abc.Mapping):
        chunks = list(spec)
        if not chunks:
            # No sizes given: nothing to chunk on, so each dimension is left
            # at its full extent (its shape size), the same fallback the
            # mapping branch uses for a dimension it does not name.
            return list(shape)
        chunks += [chunks[-1]] * max(0, len(shape) - len(chunks))
        chunks = chunks[:len(shape)]
        return chunks

    # Ensure names is a list of the same length as shape.
    names = list(names)
    names += [None] * max(0, len(shape) - len(names))

    # Map name to chunk size
    chunks = []
    for size, name in zip(shape, names):
        # Fall through on a genuinely-absent key, not on a falsy value: a
        # ``0`` ("no chunking along that dimension") is a real answer and
        # must be kept, not read as missing.
        chunk_size = spec.get(name)
        if chunk_size is None:
            chunk_size = spec.get(None)
        if chunk_size is None:
            chunk_size = spec.get("")
        if chunk_size is None:
            chunk_size = size
        chunks.append(chunk_size)
    return chunks


def auto_chunk(
    shape: tz.ShapeLike,
    spec: ChunkSpec = "auto",
    itemsize: tx.Union[int, npt.DTypeLike] = 4,
    maxsize: int = 8 * 1024**2,
    compression_ratio: float = 1.8,
    names: tx.Iterable[tx.Optional[str]] = (),
) -> tz.ShapeLike:
    """Choose a chunk size for each dimension that keeps the estimated
    on-disk chunk size under `maxsize`.

    A dimension fixed by `spec` keeps that size. Every dimension left as
    `"auto"` starts at 1 and doubles in turn, one dimension per
    iteration, until either every dimension reaches the full extent of
    `shape` or no dimension can grow further without the projected
    chunk size, divided by `compression_ratio`, exceeding `maxsize`.

    Parameters
    ----------
    shape : sequence of int
        The shape to chunk.
    spec : ChunkSpec
        The chunk-size specification, as `broadcast_spec` accepts.
    itemsize : np.dtype or int
        The array's data type, or its itemsize in bytes.
    maxsize : int
        The maximum estimated chunk size, in bytes.
    compression_ratio : float
        The estimated compression factor applied to the raw chunk size
        before it is compared against `maxsize`.
    names : sequence of str
        The name of each dimension, used to resolve `spec` when it is a
        mapping.

    Returns
    -------
    tuple of int
        The chosen chunk size along each dimension.
    """
    if not isinstance(itemsize, int):
        itemsize = np.dtype(itemsize).itemsize

    spec = broadcast_spec(shape, spec, names)

    # A zero in the spec falls back to the dimension's full size.
    spec = [(c or d) for c, d in zip(spec, shape)]

    # The chunk byte budget, expressed as a maximum element count so the
    # loop below compares against `math.prod(chunks)` directly.
    max_numel = maxsize * compression_ratio / itemsize

    chunks = [1 if c == "auto" else c for c in spec]

    while True:

        # Every dimension has reached the full extent of the array: no
        # further growth is possible or needed.
        if all(x >= s for x, s in zip(chunks, shape)):
            break

        improved = False
        for d in range(len(chunks)):

            if spec[d] != "auto":
                continue

            old_chunk = chunks[d]
            new_chunk = min(2 * chunks[d], shape[d])
            chunks[d] = new_chunk

            if math.prod(chunks) > max_numel:
                # The doubled chunk would exceed the byte budget, so the
                # dimension keeps its previous size.
                chunks[d] = old_chunk
            elif new_chunk > old_chunk:
                # Only an actual growth counts as an improvement. A dimension
                # already at its full size does not, so the loop settles
                # instead of spinning when one axis is capped and another is
                # saturated.
                improved = True

        if not improved:
            # No dimension could grow this pass, so no further pass would
            # change anything.
            break

    return tuple(chunks)


def auto_shard(
    shape: tz.ShapeLike,
    shard_spec: ChunkSpec = "auto",
    chunk_spec: ChunkSpec = "auto",
    itemsize: tx.Union[int, npt.DTypeLike] = 4,
    maxsize: int = 2 * 1024**4,
    compression_ratio: float = 1.8,
    names: tx.Iterable[tx.Optional[str]] = (),
) -> ShardsAndChunks:
    """Choose a shard size for each dimension that keeps the estimated
    on-disk shard size under `maxsize`, then chunk each shard to the
    same byte budget `auto_chunk` applies.

    Growing the shard follows the same doubling strategy as `auto_chunk`.
    A dimension fixed by `shard_spec` keeps that size. One left as
    `"auto"` starts from `chunk_spec`'s size for that dimension when
    that is fixed, or from 1 otherwise, and doubles from there. Once the
    shard shape is settled, `auto_chunk` sizes the chunks within it
    against `itemsize` directly, so the chunk byte budget reflects the
    real data type rather than the estimate `broadcast_spec` alone would
    give. The chunk and shard shapes are then reconciled through
    `fix_shard_chunk`, since a shard has to be an exact multiple of its
    chunk.

    Parameters
    ----------
    shape : sequence of int
        The shape to shard and chunk.
    shard_spec : ChunkSpec
        The shard-size specification, as `broadcast_spec` accepts.
    chunk_spec : ChunkSpec
        The chunk-size specification, as `broadcast_spec` accepts.
    itemsize : np.dtype or int
        The array's data type, or its itemsize in bytes.
    maxsize : int
        The maximum estimated shard size, in bytes. The default of 2 TB
        stays under S3's 5 TB per-object limit even though the estimate
        is only as good as `compression_ratio`.
    compression_ratio : float
        The estimated compression factor applied to the raw shard size
        before it is compared against `maxsize`.
    names : sequence of str
        The name of each dimension, used to resolve `shard_spec` and
        `chunk_spec` when either is a mapping.

    Returns
    -------
    ShardsAndChunks
        The chosen shard and chunk size along each dimension.
    """
    if not isinstance(itemsize, int):
        itemsize = np.dtype(itemsize).itemsize

    shard_spec = broadcast_spec(shape, shard_spec, names)
    chunk_spec = broadcast_spec(shape, chunk_spec, names)

    # A zero in the shard spec falls back to the dimension's full size.
    shard_spec = [(s or d) for s, d in zip(shard_spec, shape)]

    # A zero in the chunk spec falls back to the (already resolved) shard
    # size for that dimension.
    chunk_spec = [(c or s) for c, s in zip(chunk_spec, shard_spec)]

    # The shard byte budget, expressed as a maximum element count so the
    # loop below compares against `math.prod(shards)` directly.
    max_numel = maxsize * compression_ratio / itemsize

    # A fixed shard size is used as given. Otherwise, a fixed chunk size
    # is the starting point to grow from. With neither fixed, growth
    # starts from 1.
    shards = [
        1 if s == "auto" and c == "auto" else
        c if s == "auto" else
        s
        for s, c in zip(shard_spec, chunk_spec)
    ]

    while True:

        # Every dimension has reached the full extent of the array: no
        # further growth is possible or needed.
        if all(x >= s for x, s in zip(shards, shape)):
            break

        improved = False
        for d in range(len(shards)):

            if shard_spec[d] != "auto":
                continue

            old_shard = shards[d]
            new_shard = min(2 * shards[d], shape[d])
            shards[d] = new_shard

            if math.prod(shards) > max_numel:
                # The doubled shard would exceed the byte budget, so the
                # dimension keeps its previous size.
                shards[d] = old_shard
            elif new_shard > old_shard:
                # Only an actual growth counts as an improvement. A dimension
                # already at its full size does not, so the loop settles
                # instead of spinning when one axis is capped and another is
                # saturated.
                improved = True

        if not improved:
            # No dimension could grow this pass, so no further pass would
            # change anything.
            break

    # Any "auto" chunk size is resolved against the real itemsize, so the
    # chunk byte budget reflects the actual dtype rather than the default.
    chunks = auto_chunk(
        shape,
        chunk_spec,
        itemsize=itemsize,
        compression_ratio=compression_ratio,
    )
    chunks = [min(c, s) for c, s in zip(chunks, shards)]

    shards, chunks = fix_shard_chunk(shards, chunks, shape)

    return ShardsAndChunks(shards=tuple(shards), chunks=tuple(chunks))


def fix_shard_chunk(
    shard: tz.ShapeLike,
    chunk: tz.ShapeLike,
    shape: tz.ShapeLike,
) -> ShardsAndChunks:
    """Adjust `chunk` and `shard` so that every shard is a whole number
    of chunks.

    On a dimension where the chunk already spans the entire array, the
    chunk is resized to match the shard, since a chunk cannot be larger
    than the shard it belongs to. On every dimension, a shard not evenly
    divisible by its chunk is rounded up to the next multiple of that
    chunk.

    Parameters
    ----------
    shard : iterable of int
        The shard size along each dimension.
    chunk : iterable of int
        The chunk size along each dimension.
    shape : iterable of int
        The full shape being sharded and chunked.

    Returns
    -------
    ShardsAndChunks
        The adjusted shard and chunk size along each dimension.
    """
    shard = list(shard)
    chunk = list(chunk)
    for i in range(len(chunk)):
        if chunk[i] == shape[i] and chunk[i] != shard[i]:
            chunk[i] = shard[i]
        if shard[i] % chunk[i]:
            shard[i] = chunk[i] * int(math.ceil(shard[i] / chunk[i]))
    return ShardsAndChunks(tuple(shard), tuple(chunk))

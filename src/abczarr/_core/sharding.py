"""Helper functions for Zarr I/O."""

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
    shards: tz.Shape
    chunks: tz.Shape


def broadcast_spec(
    shape: tz.ShapeLike,
    spec: ChunkSpec = "auto",
    names: tx.Iterable[tx.Optional[str]] = (),
) -> tx.Tuple[tx.Union[int, tx.Literal["auto"]], ...]:
    """
    Assign a chunk size to each dimension, based on a specification.

    Parameters
    ----------
    shape : sequence[int]
        Shape of the data array.
    spec : int | {"auto"} | sequence | mapping
        Chunk size along each dimension, or a mapping from dimension names to
        chunk sizes, or a single integer to use for all dimensions.
        * Zero means no chunking along that dimension.
        * "auto" means that the chunk size will be automatically determined
          later.
    names : sequence[str]
        Names of the dimensions, if `spec` is a mapping.

    Returns
    -------
    chunks : tuple[int | {"auto"}, ...]
        Chunk or shard size along each dimension.
    """

    if isinstance(spec, (int, str)):
        spec = (spec,)

    # Right-pad chunk size
    if not isinstance(spec, abc.Mapping):
        chunks = list(spec)
        chunks += [chunks[-1]] * max(0, len(shape) - len(chunks))
        chunks = chunks[:len(shape)]
        return chunks

    # Ensure names is a list of the same length as shape.
    names = list(names)
    names += [None] * max(0, len(shape) - len(names))

    # Map name to chunk size
    chunks = []
    for size, name in zip(shape, names):
        chunk_size = spec.get(name)
        chunk_size = chunk_size or spec.get(None)
        chunk_size = chunk_size or spec.get("")
        chunk_size = chunk_size or size
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
    """
    Compute chunk size that ensures blob size below cap.

    Parameters
    ----------
    shape : sequence[int]
        (Maximum) shape along each dimension.
    itemsize : np.dtype or int
        Data type, or data type size
    spec : ChunkSpec
        See `broadcast_spec` for details.
    maxsize : int
        Maximum size of each chunk, in bytes (default: 8 MB).
    compression_ratio : float
        Estimated compression factor.
    names : sequence[str]
        Names of the dimensions, if `spec` is a mapping.

    Returns
    -------
    chunks : tuple[int, ...]
        Estimated chunk size along each dimension.
    """
    if not isinstance(itemsize, int):
        itemsize = np.dtype(itemsize).itemsize

    # Broadcast specifications
    spec = broadcast_spec(shape, spec, names)

    # Replace 0 with the shape size in spec
    spec = [(c or d) for c, d in zip(spec, shape)]

    # Maximum number of elements in the chunk
    max_numel = maxsize * compression_ratio / itemsize

    # Initial chunk size
    chunks = [1 if c == "auto" else c for c in spec]

    # Optimization loop
    while True:

        # If chunk larger than volume, we can stop
        if all(x >= s for x, s in zip(chunks, shape)):
            break

        # Loop over dimensions
        improved = False
        for d in range(len(chunks)):

            if spec[d] != "auto":
                continue

            # Compute candidate shard size
            old_chunk = chunks[d]
            new_chunk = min(2 * chunks[d], shape[d])
            chunks[d] = new_chunk

            if math.prod(chunks) > max_numel:
                # If chunk is too large, stop and keep previous chunk
                chunks[d] = old_chunk
            else:
                # Otherwise, use larger chunk and continue
                improved = True

        if not improved:
            # We cannot improve any further, so we stop
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
    """
    Find maximal shard size that ensures file size below cap.

    Parameters
    ----------
    shape : sequence[int]
        (Maximum) shape along each dimension.
    itemsize : np.dtype or int
        Data type, or data type size
    shard_spec : ChunkSpec
        See `broadcast_spec` for details.
    chunk_spec : ChunkSpec
        See `broadcast_spec` for details.
    maxsize : int
        Maximum size of each shard, in bytes (default: 2 TB).
        S3 has a 5TB/file limit, but given that we use an estimated
        compression factor, we aim for 2TB to leave some leeway.
    compression_ratio : float
        Estimated compression factor.
    names : sequence[str]
        Names of the dimensions, if `spec` is a mapping.

    Returns
    -------
    shards : tuple[int, ...]
        Estimated shard size along each dimension.
    chunks : tuple[int, ...]
        Estimated chunk size along each dimension.
    """
    if not isinstance(itemsize, int):
        itemsize = np.dtype(itemsize).itemsize

    # Broadcast specifications
    shard_spec = broadcast_spec(shape, shard_spec, names)
    chunk_spec = broadcast_spec(shape, chunk_spec, names)

    # Replace 0 with the shape size in shard spec
    shard_spec = [(s or d) for s, d in zip(shard_spec, shape)]

    # Replace 0 with either the shard size in chunk spec
    chunk_spec = [(c or s) for c, s in zip(chunk_spec, shard_spec)]

    # Maximum number of elements in the shard
    max_numel = maxsize * compression_ratio / itemsize

    # Initial shard size
    # =>            if shard is fixed -> use shard
    # => otherwise, if chunk is fixed -> use chunk
    # => otherwise,                   -> use 1
    shards = [
        1 if s == "auto" and c == "auto" else
        c if s == "auto" else
        s
        for s, c in zip(shard_spec, chunk_spec)
    ]

    # Optimization loop
    while True:

        # If shard larger than volume, we can stop
        if all(x >= s for x, s in zip(shards, shape)):
            break


        # Loop over dimensions
        improved = False
        for d in range(len(shards)):

            if shard_spec[d] != "auto":
                continue

            # Compute candidate shard size
            old_shard = shards[d]
            new_shard = min(2 * shards[d], shape[d])
            shards[d] = new_shard

            if math.prod(shards) > max_numel:
                # If shard is too large, stop and keep previous shard
                shards[d] = old_shard
            else:
                # Otherwise, use larger shard and continue
                improved = True

        if not improved:
            # We cannot improve any further, so we stop
            break

    # replace "auto" chunk size
    chunks = auto_chunk(shape, chunk_spec, compression_ratio=compression_ratio)
    chunks = [min(c, s) for c, s in zip(chunks, shards)]

    # Fix incompatibilities between chunk and shard size
    shards, chunks = fix_shard_chunk(shards, chunks, shape)

    return ShardsAndChunks(shards=tuple(shards), chunks=tuple(chunks))


def fix_shard_chunk(
    shard: tz.ShapeLike,
    chunk: tz.ShapeLike,
    shape: tz.ShapeLike,
) -> ShardsAndChunks:
    """
    Fix incompatibilities between chunk and shard size.

    Parameters
    ----------
    shard : iterable[int]
    chunk : iterable[int]
    shape : iterable[int]

    Returns
    -------
    shard : tuple[int, ...]
    chunk : tuple[int, ...]
    """
    shard = list(shard)
    chunk = list(chunk)
    for i in range(len(chunk)):
        # if chunk spans the entire volume, match chunk and shard
        if chunk[i] == shape[i] and chunk[i] != shard[i]:
            chunk[i] = shard[i]
        # ensure that shard is a multiple of chunk
        if shard[i] % chunk[i]:
            shard[i] = chunk[i] * int(math.ceil(shard[i] / chunk[i]))
    return ShardsAndChunks(tuple(shard), tuple(chunk))

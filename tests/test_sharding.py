"""Automatic chunk and shard sizing.

The sizing loops grow each auto dimension until a byte budget is reached.
They must terminate once every auto dimension is either capped by the budget
or already at the array's full extent, and they must size against the real
item size so the byte budget holds for the actual dtype.
"""

import math

from abczarr._core.sharding import auto_chunk, auto_shard, broadcast_spec


def test_auto_shard_terminates_with_a_saturated_and_a_capped_axis() -> None:
    # dim 0 (size 2) reaches its extent at once; a tiny budget caps dim 1
    # before it reaches its extent, so neither dimension can grow further
    result = auto_shard(shape=(2, 1000000), itemsize=4, maxsize=1000)
    assert result.shards[0] == 2
    assert result.shards[1] < 1000000


def test_auto_chunk_terminates_on_a_fully_saturated_shape() -> None:
    assert auto_chunk((2, 2), itemsize=4) == (2, 2)


def test_auto_chunk_stays_within_the_byte_budget() -> None:
    chunk = auto_chunk((1024, 1024, 1024), itemsize=4, maxsize=8 * 1024**2)
    assert math.prod(chunk) * 4 <= 8 * 1024**2 * 1.8


def test_auto_shard_sizes_chunks_for_the_real_itemsize() -> None:
    # the same chunk byte budget holds fewer elements of a wider dtype
    narrow = auto_shard((1024, 1024, 1024), itemsize=4).chunks
    wide = auto_shard((1024, 1024, 1024), itemsize=8).chunks
    assert math.prod(wide) < math.prod(narrow)


def test_broadcast_spec_honors_a_zero_mapping_value() -> None:
    # 0 means "no chunking along that dimension"; a falsy 0 must be kept, not
    # read as a missing key that falls through to the dimension's full size.
    assert broadcast_spec((10, 20), {"x": 0, "y": 5}, ["x", "y"]) == [0, 5]


def test_broadcast_spec_handles_an_empty_sequence() -> None:
    # an empty spec names no sizes; each dimension is left at its full extent
    # instead of raising IndexError on the right-pad step.
    assert broadcast_spec((10, 20), ()) == [10, 20]

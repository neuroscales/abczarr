"""The Dask bridge on ZarrArray: to_dask block sizing and store lock='auto'.

The block-alignment helper is pure arithmetic and needs no backend; the
end-to-end to_dask / store tests need zarr-python and dask.
"""

import pathlib

import numpy as np
import pytest

from abczarr.abc.sync import _blocks_align_to

# --------------------------------------------------------------------------
# the alignment helper (no backend needed)
# --------------------------------------------------------------------------


def test_blocks_align_when_they_fall_on_whole_chunks() -> None:
    assert _blocks_align_to(((4, 4), (4, 4)), (4, 4)) is True
    # blocks twice the unit still fall on chunk boundaries
    assert _blocks_align_to(((8, 8), (8,)), (4, 4)) is True
    # a ragged final block is fine: it is the array end, one writer only
    assert _blocks_align_to(((4, 4, 2), (4, 4)), (4, 4)) is True


def test_blocks_do_not_align_when_a_boundary_splits_a_chunk() -> None:
    assert _blocks_align_to(((2, 2, 2, 2), (4, 4)), (4, 4)) is False
    assert _blocks_align_to(((4, 3, 1), (4, 4)), (4, 4)) is False
    # a different number of axes never aligns
    assert _blocks_align_to(((4, 4),), (4, 4)) is False


# --------------------------------------------------------------------------
# end to end, over a real zarr array
# --------------------------------------------------------------------------

zarr = pytest.importorskip("zarr")
da = pytest.importorskip("dask.array")

import abczarr  # noqa: E402
from abczarr.api.config import ArrayConfig  # noqa: E402


def _array(tmp_path: pathlib.Path) -> object:
    return abczarr.create(
        str(tmp_path / "a.zarr"),
        ArrayConfig(shape=(8, 8), dtype="float32", chunks=(4, 4)),
    )


def test_to_dask_chunk_alignment_option(tmp_path: pathlib.Path) -> None:
    arr = _array(tmp_path)
    assert arr.to_dask().chunksize == (4, 4)
    assert arr.to_dask("chunks").chunksize == (4, 4)
    assert arr.to_dask("shards").chunksize == (4, 4)
    # an explicit block size is passed straight through
    assert arr.to_dask((2, 2)).chunksize == (2, 2)


def test_store_auto_lock_writes_correctly_when_aligned(
    tmp_path: pathlib.Path,
) -> None:
    arr = _array(tmp_path)
    arr.store(da.zeros((8, 8), dtype="float32", chunks=(4, 4)) + 5.0)
    assert float(np.asarray(arr[0, 0])) == 5.0
    assert float(np.asarray(arr[7, 7])) == 5.0


def test_store_auto_lock_writes_correctly_when_misaligned(
    tmp_path: pathlib.Path,
) -> None:
    arr = _array(tmp_path)
    # 3x3 blocks straddle the 4x4 chunks: auto locks, result still correct
    arr.store(da.zeros((8, 8), dtype="float32", chunks=(3, 3)) + 9.0)
    assert float(np.asarray(arr[0, 0])) == 9.0
    assert float(np.asarray(arr[7, 7])) == 9.0

"""The abstract array's synthesized helpers.

``to_dask`` aligns its blocks to the write unit -- the shard where the array
is sharded, otherwise the chunk -- so a read never re-fetches the same shard
once per inner chunk. Dask is imported lazily, inside the method.
"""

import numpy as np
import numpy.typing as npt
import pytest
import typing_extensions as tx

from abczarr.abc.sync import ZarrArray


class _FakeArray(ZarrArray):
    """A minimal concrete array with a settable shard shape."""

    def __init__(self, shards: tx.Optional[tx.Tuple[int, ...]]) -> None:
        self._shards = shards
        super().__init__("/store")

    @property
    def metadata(self) -> None:
        return None

    @property
    def attrs(self) -> dict:
        return {}

    @property
    def zarr_version(self) -> int:
        return 3

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        return (8, 8)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype("uint8")

    @property
    def chunks(self) -> tx.Tuple[int, ...]:
        return (2, 2)

    @property
    def shards(self) -> tx.Optional[tx.Tuple[int, ...]]:
        return self._shards

    def __getitem__(self, index: tx.Any) -> npt.ArrayLike:
        return np.zeros(self.shape, "uint8")[index]

    def __setitem__(self, index: tx.Any, value: npt.ArrayLike) -> None:
        ...


def test_to_dask_blocks_align_to_shards_when_sharded() -> None:
    d = _FakeArray(shards=(4, 4)).to_dask()
    assert d.chunksize == (4, 4)


def test_to_dask_blocks_fall_back_to_chunks_when_unsharded() -> None:
    d = _FakeArray(shards=None).to_dask()
    assert d.chunksize == (2, 2)


def test_to_dask_reads_the_array_values() -> None:
    d = _FakeArray(shards=(4, 4)).to_dask()
    assert np.asarray(d).shape == (8, 8)


def test_array_module_does_not_import_dask_itself() -> None:
    # dask is imported inside to_dask, not at module load, so the abstract
    # array does not require dask merely to be defined
    import ast
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / (
        "src/abczarr/abc/sync.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_level.append(node.module)
    assert not any(m == "dask" or m.startswith("dask.") for m in module_level)


def test_array_accepts_the_numpy2_copy_keyword() -> None:
    # NumPy 2 calls __array__(dtype, copy); the method must accept copy
    # (regression: the signature lacked it, so this call raised TypeError).
    a = _FakeArray(shards=None)
    assert np.asarray(a.__array__(copy=True)).shape == (8, 8)
    assert np.asarray(a.__array__(copy=None)).shape == (8, 8)


def test_array_refuses_copy_false() -> None:
    # reading always materializes a fresh array, so a no-copy view is refused
    a = _FakeArray(shards=None)
    with pytest.raises(ValueError):
        a.__array__(copy=False)


def test_np_array_copy_true_reads_values() -> None:
    # the NumPy-2 integration path: np.array(..., copy=True) round-trips
    if int(np.__version__.split(".")[0]) < 2:
        pytest.skip("copy= is only passed to __array__ on NumPy 2+")
    a = _FakeArray(shards=None)
    assert np.array(a, copy=True).shape == (8, 8)

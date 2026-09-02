"""The TensorStore driver: feature declaration and reading/writing v3 arrays.

Needs both zarr-python (to write the fixtures) and tensorstore; runs on the
coverage CI leg where both are installed.
"""

import pathlib

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("tensorstore")

import abczarr  # noqa: E402
from abczarr.abc.capabilities import Support  # noqa: E402
from abczarr.abc.errors import UnsupportedZarrOperation  # noqa: E402
from abczarr.drivers.tensorstore import (  # noqa: E402
    TensorStoreArray,
    TensorStoreDriver,
)
from abczarr.metadata.base import ArrayMetadata  # noqa: E402
from abczarr.registry import available_drivers  # noqa: E402


def _array_path(tmp_path: pathlib.Path, **create: object) -> str:
    root = str(tmp_path / "d.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "img", shape=(8, 8), chunks=(4, 4), dtype="float32", **create
    )
    array[:] = np.arange(64).reshape(8, 8)
    return root + "/img"


# --------------------------------------------------------------------------
# registration and features
# --------------------------------------------------------------------------


def test_tensorstore_is_registered() -> None:
    names = [d.name for d in available_drivers()]
    assert "tensorstore" in names


def test_coarse_capabilities() -> None:
    d = TensorStoreDriver()
    assert d.available is True
    assert d.support("sharding") is Support.NATIVE
    assert d.support("async") is Support.NATIVE
    assert d.supports("partial_read") is True


def test_codec_support() -> None:
    d = TensorStoreDriver()
    assert d.support("v3:codec:zstd") is Support.NATIVE
    assert d.support("v3:codec:sharding_indexed") is Support.NATIVE
    assert d.support("v3:codec:transpose") is Support.NATIVE
    # a codec tensorstore does not implement
    assert d.support("v3:codec:packbits") is Support.NONE
    assert d.support("v3:chunk_grid:rectilinear") is Support.NONE


# --------------------------------------------------------------------------
# reading and writing an array
# --------------------------------------------------------------------------


def test_open_read_a_v3_array(tmp_path: pathlib.Path) -> None:
    path = _array_path(
        tmp_path,
        compressors=[{"name": "zstd", "configuration": {"level": 5}}],
    )
    arr = abczarr.open_array(path, mode="r", driver="tensorstore")
    assert isinstance(arr, TensorStoreArray)
    assert arr.shape == (8, 8)
    assert arr.ndim == 2
    assert arr.dtype == np.dtype("float32")
    assert arr.chunks == (4, 4)
    assert arr.zarr_version == 3
    assert np.asarray(arr[:2, :2]).tolist() == [[0.0, 1.0], [8.0, 9.0]]


def test_metadata_is_abczarr_metadata(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open_array(
        _array_path(tmp_path), mode="r", driver="tensorstore"
    )
    meta = arr.metadata
    assert isinstance(meta, ArrayMetadata)
    assert meta.shape == (8, 8)
    assert "v3:data_type:float32" in meta.required_features()


def test_write_through_the_surface(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open_array(
        _array_path(tmp_path), mode="a", driver="tensorstore"
    )
    arr[0, 0] = 99.0
    assert float(np.asarray(arr[0, 0])) == 99.0


def test_native_is_the_tensorstore(tmp_path: pathlib.Path) -> None:
    import tensorstore as ts

    arr = abczarr.open_array(
        _array_path(tmp_path), mode="r", driver="tensorstore"
    )
    assert isinstance(arr.native, ts.TensorStore)


def test_sharded_array_reports_shards(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "s.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "img", shape=(8, 8), chunks=(2, 2), shards=(4, 4), dtype="uint8"
    )
    array[:] = 0
    arr = abczarr.open_array(root + "/img", mode="r", driver="tensorstore")
    assert arr.chunks == (2, 2)
    assert arr.shards == (4, 4)


def test_can_open_a_supported_array(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open_array(
        _array_path(tmp_path), mode="r", driver="tensorstore"
    )
    assert bool(TensorStoreDriver().can_open(arr.metadata)) is True


# --------------------------------------------------------------------------
# groups are not a TensorStore concept
# --------------------------------------------------------------------------


def test_opening_a_group_with_tensorstore_raises(
    tmp_path: pathlib.Path,
) -> None:
    root = str(tmp_path / "g.zarr")
    zarr.open_group(root, mode="w").create_array(
        "a", shape=(2,), chunks=(2,), dtype="u1"
    )
    with pytest.raises(UnsupportedZarrOperation, match="group"):
        abczarr.open(root, mode="r", driver="tensorstore")

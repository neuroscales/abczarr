"""Creating arrays and groups through the surface: the array config
(chunking, sharding, compression, fill value) and create.

Runs where zarr-python 3.x is installed (the coverage CI leg).
"""

import pathlib

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

import abczarr  # noqa: E402
from abczarr.api.config import ArrayConfig, GroupConfig  # noqa: E402
from abczarr.drivers.zarr_python import ZarrPythonArray  # noqa: E402


def _group(tmp_path: pathlib.Path) -> abczarr.ZarrGroup:
    return abczarr.create(
        str(tmp_path / "d.zarr"), GroupConfig(zarr_version=3, overwrite=True)
    )


# --------------------------------------------------------------------------
# create
# --------------------------------------------------------------------------


def test_create_creates_a_group(tmp_path: pathlib.Path) -> None:
    group = _group(tmp_path)
    assert isinstance(group, abczarr.ZarrGroup)
    assert group.zarr_version == 3


def test_create_without_overwrite_refuses_existing(
    tmp_path: pathlib.Path,
) -> None:
    root = str(tmp_path / "d.zarr")
    abczarr.create(root, GroupConfig(zarr_version=3, overwrite=True))
    with pytest.raises(FileExistsError):
        abczarr.create(root, GroupConfig(zarr_version=3, overwrite=False))


# --------------------------------------------------------------------------
# create_array config
# --------------------------------------------------------------------------


def test_create_array_with_a_compressor(tmp_path: pathlib.Path) -> None:
    arr = _group(tmp_path).create_array(
        "a",
        shape=(8, 8),
        dtype="float32",
        chunks=(4, 4),
        compressor="zstd",
        compressor_options={"level": 7},
        fill_value=1.5,
    )
    assert isinstance(arr, ZarrPythonArray)
    assert arr.chunks == (4, 4)
    assert "v3:codec:zstd" in arr.metadata.required_features()
    assert arr.metadata.fill_value == 1.5


def test_create_array_no_compressor(tmp_path: pathlib.Path) -> None:
    arr = _group(tmp_path).create_array(
        "a", shape=(4,), dtype="int16", chunks=(4,), compressor=None
    )
    assert [c.name for c in arr.metadata.codecs] == ["bytes"]


def test_create_array_sharded(tmp_path: pathlib.Path) -> None:
    arr = _group(tmp_path).create_array(
        "a", shape=(8, 8), dtype="uint8", chunks=(2, 2), shards=(4, 4)
    )
    assert arr.chunks == (2, 2)
    assert arr.shards == (4, 4)
    assert "v3:codec:sharding_indexed" in arr.metadata.required_features()


def test_create_array_dimension_separator(tmp_path: pathlib.Path) -> None:
    arr = _group(tmp_path).create_array(
        "a", shape=(4,), dtype="uint8", chunks=(2,), dimension_separator="."
    )
    assert arr.metadata.chunk_key_encoding.configuration.separator == "."


def test_config_dict_and_kwargs_are_equivalent(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    by_kwargs = group.create_array(
        "k", shape=(4,), dtype="u1", chunks=(2,), compressor="gzip"
    )
    by_config = group.create_array(
        "c",
        shape=(4,),
        dtype="u1",
        config={"chunks": (2,), "compressor": "gzip"},
    )
    assert (
        by_kwargs.metadata.required_features()
        == by_config.metadata.required_features()
    )


def test_created_array_round_trips_data(tmp_path: pathlib.Path) -> None:
    arr = _group(tmp_path).create_array(
        "a", shape=(4, 4), dtype="float32", chunks=(2, 2), compressor="zstd"
    )
    arr[:] = np.arange(16).reshape(4, 4)
    assert np.asarray(arr[1, :2]).tolist() == [4.0, 5.0]


# --------------------------------------------------------------------------
# create() from a config
# --------------------------------------------------------------------------


def test_create_an_array_from_a_config(tmp_path: pathlib.Path) -> None:
    arr = abczarr.create(
        str(tmp_path / "a.zarr"),
        ArrayConfig(
            shape=(8, 8), dtype="float32", chunks=(4, 4), compressor="zstd"
        ),
    )
    assert isinstance(arr, ZarrPythonArray)
    assert arr.chunks == (4, 4)
    assert "v3:codec:zstd" in arr.metadata.required_features()
    arr[:] = np.arange(64).reshape(8, 8)
    assert float(np.asarray(arr[1, 0])) == 8.0


def test_create_a_group_from_a_config(tmp_path: pathlib.Path) -> None:
    group = abczarr.create(str(tmp_path / "g.zarr"), GroupConfig())
    assert isinstance(group, abczarr.ZarrGroup)
    assert group.zarr_version == 3


# --------------------------------------------------------------------------
# the write-then-open fallback: our written metadata must be valid
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        ArrayConfig(shape=(8, 8), dtype="float32", chunks=(4, 4)),
        ArrayConfig(
            shape=(8, 8), dtype="float32", chunks=(4, 4), compressor="zstd"
        ),
        ArrayConfig(shape=(8, 8), dtype="uint8", chunks=(2, 2), shards=(4, 4)),
        ArrayConfig(
            shape=(4,), dtype="int16", compressor=None,
            dimension_separator=".",
        ),
        ArrayConfig(
            shape=(8, 8), dtype="float32", chunks=(4, 4), order="F",
            zarr_version=2,
        ),
    ],
)
def test_fallback_metadata_opens_and_writes_in_zarr(
    tmp_path: pathlib.Path, config: ArrayConfig
) -> None:
    from bagof.paths import Path as BagofPath

    root = pathlib.Path(tmp_path) / "arr.zarr"
    root.mkdir()
    config.resolve().to_metadata().to_file(BagofPath(str(root)))
    array = zarr.open_array(str(root), mode="r+")
    array[...] = 0
    assert tuple(array.shape) == config.shape


# --------------------------------------------------------------------------
# create from raw metadata (the escape hatch beyond the config helpers)
# --------------------------------------------------------------------------


def _raw_v3_array(**over: object) -> dict:
    doc = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [8, 8],
        "data_type": "float32",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [4, 4]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 5}},
        ],
        "fill_value": 7,
        "attributes": {"note": "custom"},
    }
    doc.update(over)
    return doc


def test_create_from_a_metadata_object(tmp_path: pathlib.Path) -> None:
    from abczarr.metadata import v3

    meta = v3.ArrayMetadata.from_dict(_raw_v3_array())
    arr = abczarr.create(str(tmp_path / "raw.zarr"), meta)
    assert isinstance(arr, ZarrPythonArray)
    # the exact metadata is honoured, including a custom fill and codec level
    assert arr.metadata.to_dict()["fill_value"] == 7
    assert arr.attrs["note"] == "custom"
    assert "v3:codec:zstd" in arr.metadata.required_features()


def test_create_rejects_a_bare_dict(tmp_path: pathlib.Path) -> None:
    # a dict is ambiguous (config fields vs a metadata doc); it must be wrapped
    with pytest.raises(TypeError, match="wrap it"):
        abczarr.create(str(tmp_path / "d.zarr"), _raw_v3_array())


def test_create_from_metadata_rejects_stray_keywords(
    tmp_path: pathlib.Path,
) -> None:
    from abczarr.metadata import v3

    meta = v3.ArrayMetadata.from_dict(_raw_v3_array())
    with pytest.raises(TypeError, match="unexpected keyword"):
        abczarr.create(str(tmp_path / "x.zarr"), meta, chunks=(2, 2))

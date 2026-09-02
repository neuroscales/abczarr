"""The creation config: ArrayConfig, GroupConfig, and their lowering.

A config describes what to create. It resolves "auto" chunking and sharding,
lowers to the metadata a driver writes, and spreads as ``**config`` because it
behaves as a mapping of its fields.
"""

from abczarr._core.attrs import fields
from abczarr.api.config import ArrayConfig, ArrayOptions, GroupConfig


def test_a_config_behaves_as_a_mapping_of_its_fields() -> None:
    config = ArrayConfig(shape=(8, 8), dtype="float32", compressor="zstd")
    assert "compressor" in config.keys()
    assert config["compressor"] == "zstd"
    assert dict(config)["shape"] == (8, 8)


def test_double_star_unpack_round_trips_through_a_config() -> None:
    config = ArrayConfig(shape=(4,), dtype="int16", chunks=(2,))
    remade = ArrayConfig(**config)
    assert remade == config


def test_resolve_works_out_auto_chunks() -> None:
    resolved = ArrayConfig(shape=(1024, 1024), dtype="uint8").resolve()
    assert all(isinstance(c, int) for c in resolved.chunks)
    assert resolved.chunks[0] <= 1024


def test_minus_one_means_the_whole_axis() -> None:
    resolved = ArrayConfig(
        shape=(10, 20), dtype="uint8", chunks=(-1, 5)
    ).resolve()
    assert resolved.chunks == (10, 5)


def test_auto_fill_value_lowers_to_the_dtype_zero() -> None:
    metadata = ArrayConfig(shape=(4,), dtype="float32").to_metadata()
    assert metadata.to_dict()["fill_value"] == 0.0


def test_auto_compressor_lowers_to_zstd() -> None:
    metadata = ArrayConfig(shape=(4,), dtype="uint8").to_metadata()
    names = [codec.get("name") for codec in metadata.to_dict()["codecs"]]
    assert "zstd" in names


def test_no_compressor_leaves_only_the_bytes_codec() -> None:
    metadata = ArrayConfig(
        shape=(4,), dtype="uint8", compressor=None
    ).to_metadata()
    names = [codec.get("name") for codec in metadata.to_dict()["codecs"]]
    assert names == ["bytes"]


def test_lowering_needs_a_shape_and_a_dtype() -> None:
    import pytest

    with pytest.raises(ValueError, match="shape and a dtype"):
        ArrayConfig(compressor="zstd").to_metadata()


def test_group_config_carries_the_store_fields() -> None:
    config = GroupConfig(zarr_version=2, overwrite=True)
    assert config.zarr_version == 2
    assert config.overwrite is True


def test_array_options_mirror_the_array_config_fields() -> None:
    config_fields = {f.name for f in fields(ArrayConfig)}
    option_keys = set(ArrayOptions.__optional_keys__)
    assert option_keys == config_fields - {"shape", "dtype"}

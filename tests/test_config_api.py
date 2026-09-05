"""The creation config: ArrayConfig, GroupConfig, and their lowering.

A config describes what to create. It resolves "auto" chunking and sharding,
lowers to the metadata a driver writes, and spreads as ``**config`` because it
behaves as a mapping of its fields.
"""

from abczarr._core.attrs import fields
from abczarr.api._config import ArrayConfig, ArrayOptions, GroupConfig


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
    assert metadata.to_json()["fill_value"] == 0.0


def test_auto_compressor_lowers_to_zstd() -> None:
    metadata = ArrayConfig(shape=(4,), dtype="uint8").to_metadata()
    names = [codec.get("name") for codec in metadata.to_json()["codecs"]]
    assert "zstd" in names


def test_no_compressor_leaves_only_the_bytes_codec() -> None:
    metadata = ArrayConfig(
        shape=(4,), dtype="uint8", compressor=None
    ).to_metadata()
    names = [codec.get("name") for codec in metadata.to_json()["codecs"]]
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


def test_default_zstd_compressor_writes_a_full_configuration() -> None:
    # the v3 zstd schema requires `level` (checksum is optional), so an
    # empty configuration is invalid; abczarr writes the full one.
    meta = ArrayConfig(
        shape=(4, 4), dtype="float32", chunks=(2, 2)
    ).to_metadata()
    zstd = [c for c in meta.to_json()["codecs"] if c["name"] == "zstd"]
    assert zstd == [
        {"name": "zstd", "configuration": {"level": 0, "checksum": False}}
    ]


def test_zstd_survives_a_round_trip_through_v2() -> None:
    from abczarr.metadata import v3

    meta = ArrayConfig(
        shape=(4, 4), dtype="float32", chunks=(2, 2)
    ).to_metadata()
    m3 = v3.ArrayMetadata.from_json(meta.to_json())
    # v2's numcodecs zstd carries only the level; it comes back in full
    again = m3.to_version(2).to_version(3)
    zstd = [
        c.to_json() for c in again.codecs
        if getattr(c, "name", "") == "zstd"
    ]
    assert zstd == [
        {"name": "zstd", "configuration": {"level": 0, "checksum": False}}
    ]


def _codecs(**kw: object) -> list:
    dtype = kw.pop("dtype", "float32")
    meta = ArrayConfig(
        shape=(4, 4), dtype=dtype, chunks=(2, 2), **kw
    ).to_metadata()
    return meta.to_json()["codecs"]


def _named(codecs: list, name: str) -> dict:
    return next(c for c in codecs if c["name"] == name)


def test_blosc_writes_its_required_typesize_from_the_dtype() -> None:
    # v3 blosc requires typesize (a positive int) unless shuffle is noshuffle
    blosc = _named(_codecs(compressor="blosc"), "blosc")
    assert blosc["configuration"]["typesize"] == 4  # float32
    assert _named(
        _codecs(dtype="uint8", compressor="blosc"), "blosc"
    )["configuration"]["typesize"] == 1


def test_blosc_omits_typesize_when_shuffle_is_off() -> None:
    blosc = _named(
        _codecs(
            compressor="blosc",
            compressor_options={"shuffle": "noshuffle"},
        ),
        "blosc",
    )
    assert "typesize" not in blosc["configuration"]


def test_bytes_endian_follows_the_dtype_and_is_absent_for_one_byte() -> None:
    # multi-byte carries endianness; a single-byte dtype carries none
    assert _named(_codecs(compressor=None), "bytes") == {
        "name": "bytes", "configuration": {"endian": "little"}
    }
    assert _named(_codecs(dtype=">f8", compressor=None), "bytes") == {
        "name": "bytes", "configuration": {"endian": "big"}
    }
    assert _named(_codecs(dtype="uint8", compressor=None), "bytes") == {
        "name": "bytes"
    }


def test_crc32c_is_written_as_a_bare_name() -> None:
    # the shard index uses crc32c; it takes no configuration
    meta = ArrayConfig(
        shape=(8, 8), dtype="float32", chunks=(2, 2), shards=(4, 4)
    ).to_metadata()
    index = meta.to_json()["codecs"][0]["configuration"]["index_codecs"]
    assert {"name": "crc32c"} in index

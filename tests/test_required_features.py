"""``ArrayMetadata.required_features`` -- the namespaced feature keys an array
needs a driver to provide, per format version.
"""

from abczarr._core.features import feature_key
from abczarr.metadata import v1, v2, v3


def _v3(**over: object) -> dict:
    base = {
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
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0,
        "attributes": {},
    }
    base.update(over)
    return base


def test_v3_names_grid_encoding_dtype_and_codecs() -> None:
    meta = v3.ArrayMetadata.from_json(
        _v3(
            codecs=[
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "gzip", "configuration": {"level": 5}},
            ]
        )
    )
    assert meta.required_features() == frozenset(
        {
            "v3:chunk_grid:regular",
            "v3:chunk_key_encoding:default",
            "v3:data_type:float32",
            "v3:codec:bytes",
            "v3:codec:gzip",
        }
    )


def test_v3_recurses_into_sharding_inner_and_index_codecs() -> None:
    meta = v3.ArrayMetadata.from_json(
        _v3(
            codecs=[
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": [2, 2],
                        "codecs": [
                            {
                                "name": "bytes",
                                "configuration": {"endian": "little"},
                            },
                            {"name": "blosc", "configuration": {
                                "cname": "zstd", "clevel": 5,
                                "shuffle": "shuffle", "blocksize": 0,
                                "typesize": None,
                            }},
                        ],
                        "index_codecs": [
                            {
                                "name": "bytes",
                                "configuration": {"endian": "little"},
                            },
                            {"name": "crc32c"},
                        ],
                    },
                }
            ]
        )
    )
    feats = meta.required_features()
    # the shard codec and every nested codec is named
    assert "v3:codec:sharding_indexed" in feats
    assert "v3:codec:blosc" in feats
    assert "v3:codec:crc32c" in feats


def test_v3_unknown_codec_is_named_not_crashed() -> None:
    # a codec the metadata layer carries opaquely still contributes a key,
    # so driver selection can report it as unsupported rather than crash
    meta = v3.ArrayMetadata.from_json(
        _v3(
            codecs=[
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "numcodecs.zfpy", "configuration": {"mode": 4}},
            ]
        )
    )
    assert feature_key("v3", "codec", "numcodecs.zfpy") in (
        meta.required_features()
    )


def test_v2_names_compressor_and_filters() -> None:
    meta = v2.ArrayMetadata.from_json(
        {
            "zarr_format": 2,
            "shape": [10, 10],
            "chunks": [5, 5],
            "dtype": "<f8",
            "compressor": {"id": "gzip", "level": 5},
            "filters": [{"id": "delta", "dtype": "<f8"}],
            "fill_value": 0,
            "order": "C",
            "dimension_separator": ".",
            "attributes": {},
        }
    )
    assert meta.required_features() == frozenset(
        {"v2:codec:gzip", "v2:filter:delta"}
    )


def test_v2_without_a_compressor_is_empty() -> None:
    meta = v2.ArrayMetadata.from_json(
        {
            "zarr_format": 2,
            "shape": [10],
            "chunks": [5],
            "dtype": "<f8",
            "compressor": None,
            "filters": [],
            "fill_value": 0,
            "order": "C",
            "dimension_separator": ".",
            "attributes": {},
        }
    )
    assert meta.required_features() == frozenset()


def test_v1_names_the_compressor() -> None:
    meta = v1.ArrayMetadata.from_json(
        {
            "zarr_format": 1,
            "shape": [10],
            "chunks": [2],
            "dtype": "<f8",
            "compression": "zlib",
            "compression_opts": {"level": 5},
            "fill_value": 0,
            "order": "C",
            "attributes": {},
        }
    )
    assert meta.required_features() == frozenset({"v1:codec:zlib"})


def test_v1_without_compression_is_empty() -> None:
    meta = v1.ArrayMetadata.from_json(
        {
            "zarr_format": 1,
            "shape": [10],
            "chunks": [2],
            "dtype": "<f8",
            "compression": None,
            "compression_opts": None,
            "fill_value": 0,
            "order": "C",
            "attributes": {},
        }
    )
    assert meta.required_features() == frozenset()

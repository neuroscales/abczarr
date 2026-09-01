"""Cross-version conversion of array metadata.

Covers the structural ``v2`` <-> ``v3`` conversion -- data type, chunk grid,
chunk-key encoding, endianness, and same-version identity -- and the
lossy / warn / strict policy for a field the target version cannot hold.

A ``v2`` array survives a round trip through ``v3`` unchanged (``v3`` is the
richer model). The other direction, ``v3`` -> ``v2`` -> ``v3``, is marked
expected-fail: ``v2`` cannot hold a ``v3`` array-to-bytes codec or the
default-vs-``v2`` chunk-key encoding, so a down-conversion is genuinely lossy.
"""

from __future__ import annotations

import warnings

import pytest

from abczarr.metadata import v2, v3
from abczarr.metadata.base import UnsupportedConversion


def _v2(**over: object) -> dict:
    base = {
        "zarr_format": 2,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": "<f8",
        "compressor": None,
        "filters": [],
        "fill_value": 0,
        "order": "C",
        "dimension_separator": ".",
        "attributes": {},
    }
    base.update(over)
    return base


def _v3(**over: object) -> dict:
    base = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 100],
        "data_type": "float64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [10, 10]},
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


# --------------------------------------------------------------------------
# fixed: v2 -> v3 no longer crashes on the encoding and the dtype
# --------------------------------------------------------------------------


def test_v2_dtype_converts_to_v3_without_crashing() -> None:
    m2 = v2.ArrayMetadata.from_dict(_v2())
    assert m2.dtype.to_version(3).name == "float64"


def test_v2_to_v3_uses_v2_chunk_key_encoding_with_the_separator() -> None:
    m3 = v2.ArrayMetadata.from_dict(_v2(dimension_separator="/")).to_version(3)
    assert m3.chunk_key_encoding.name == "v2"
    assert m3.chunk_key_encoding.configuration.separator == "/"


def test_same_version_conversion_is_identity() -> None:
    m3 = v3.ArrayMetadata.from_dict(_v3())
    assert m3.to_version(3) is m3
    m2 = v2.ArrayMetadata.from_dict(_v2())
    assert m2.to_version(2) is m2


# --------------------------------------------------------------------------
# up-and-back: a v2 array survives a trip through the richer v3 model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["<f8", ">f8", ">i4", "<i2", "|u1"])
def test_v2_roundtrips_through_v3_losslessly(dtype: str) -> None:
    m2 = v2.ArrayMetadata.from_dict(_v2(dtype=dtype))
    # endianness is carried by the v3 array-to-bytes codec and folded back
    assert m2.to_version(3).to_version(2) == m2


def test_v2_to_v3_carries_endianness_in_a_bytes_codec() -> None:
    m3 = v2.ArrayMetadata.from_dict(_v2(dtype=">f8")).to_version(3)
    endians = [
        c.configuration.endian for c in m3.codecs if c.name == "bytes"
    ]
    assert endians == ["big"]


@pytest.mark.parametrize(
    "compressor",
    [
        {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1},
        {"id": "gzip", "level": 5},
    ],
)
def test_builtin_compressor_roundtrips_through_v3(compressor: dict) -> None:
    m2 = v2.ArrayMetadata.from_dict(_v2(compressor=compressor))
    assert m2.to_version(3).to_version(2) == m2


# --------------------------------------------------------------------------
# the policy for a field the target can't hold (a shard grid, here)
# --------------------------------------------------------------------------


def _v3_sharded() -> dict:
    return _v3(
        codecs=[
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [5, 5],
                    "codecs": [
                        {
                            "name": "bytes",
                            "configuration": {"endian": "little"},
                        }
                    ],
                },
            }
        ]
    )


def test_lossy_policy_drops_silently() -> None:
    m3 = v3.ArrayMetadata.from_dict(_v3_sharded())
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        m3.to_version(2, policy="lossy")


def test_warn_policy_warns_naming_the_field() -> None:
    m3 = v3.ArrayMetadata.from_dict(_v3_sharded())
    with pytest.warns(UserWarning, match="sharding"):
        m3.to_version(2, policy="warn")


def test_strict_policy_raises_naming_the_field_and_version() -> None:
    m3 = v3.ArrayMetadata.from_dict(_v3_sharded())
    with pytest.raises(UnsupportedConversion) as info:
        m3.to_version(2, policy="strict")
    assert info.value.field == "sharding"
    assert info.value.version == 2


# --------------------------------------------------------------------------
# the reverse direction is genuinely lossy (no annotation mode by design)
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="v2 cannot hold a v3 array-to-bytes codec or the default chunk-key "
    "encoding, so a down-conversion is lossy by design",
    strict=False,
)
def test_v3_roundtrips_losslessly_through_v2() -> None:
    m3 = v3.ArrayMetadata.from_dict(_v3())
    assert m3.to_version(2).to_version(3) == m3

"""The cross-version conversion matrix.

A systematic sweep of ``to_version`` across every ordered Zarr version pair
(1->2, 1->3, 2->1, 2->3, 3->1, 3->2) plus identity, for every object kind
that carries ``to_version``: array metadata, group metadata, and the
sub-objects (data type, codec, filter). Each cell asserts one of three
outcomes:

* it round-trips losslessly (``src -> dst -> src`` equals the original);
* it is lossy by policy (the ``warn`` / ``strict`` behaviour is asserted); or
* it is a documented unsupported case (``xfail`` / ``raises`` with a reason).

This module is the forcing function: if any cell regresses -- a crash, a
silent corruption, or a lost field -- exactly one parametrization fails and
names what broke. The previously-crashing v2 filters and the ``transpose``
permutation corruption are covered explicitly.
"""

import warnings

import pytest
import typing_extensions as tx

from abczarr.abc.errors import UnsupportedConversion
from abczarr.metadata import v1, v2, v3
from abczarr.metadata.base import GroupMetadataV2, GroupMetadataV3

VERSIONS = (1, 2, 3)
ORDERED_PAIRS = [(s, d) for s in VERSIONS for d in VERSIONS]


# ==========================================================================
#   Array metadata
# ==========================================================================
#
# The fixtures below are chosen to be representable in every version, so a
# round trip through any other version is lossless: a regular chunk grid, a
# v2-style chunk-key encoding, C order, a single numcodecs compressor and no
# filters. The lossy directions (a shard grid, a v3 default chunk-key
# encoding, an F-order array, a v1 target with filters) are exercised
# separately, under the policy that governs them.


def _v1_array() -> v1.ArrayMetadata:
    return v1.ArrayMetadata.from_dict(
        {
            "zarr_format": 1,
            "shape": [10, 10],
            "chunks": [5, 5],
            "dtype": "<f8",
            "compression": "zlib",
            "compression_opts": {"level": 4},
            "fill_value": 0,
            "order": "C",
            "attributes": {},
        }
    )


def _v2_array() -> v2.ArrayMetadata:
    return v2.ArrayMetadata.from_dict(
        {
            "zarr_format": 2,
            "shape": [10, 10],
            "chunks": [5, 5],
            "dtype": "<f8",
            "compressor": {"id": "zstd", "level": 3},
            "filters": [],
            "fill_value": 0,
            "order": "C",
            "dimension_separator": ".",
            "attributes": {},
        }
    )


def _v3_array() -> v3.ArrayMetadata:
    # A v2-style chunk-key encoding and a plain compressor keep this v3 array
    # fully representable in v2 and v1, so it round-trips both ways.
    return v3.ArrayMetadata.from_dict(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [10, 10],
            "data_type": "float64",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [5, 5]},
            },
            "chunk_key_encoding": {
                "name": "v2",
                "configuration": {"separator": "."},
            },
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "zstd", "configuration": {"level": 3}},
            ],
            "fill_value": 0,
            "attributes": {},
        }
    )


_ARRAY_BUILDERS = {1: _v1_array, 2: _v2_array, 3: _v3_array}


@pytest.mark.parametrize("src", VERSIONS)
def test_array_identity_returns_self(src: int) -> None:
    m = _ARRAY_BUILDERS[src]()
    assert m.to_version(src) is m


@pytest.mark.parametrize(("src", "dst"), ORDERED_PAIRS)
def test_array_conversion_stamps_the_target_version(
    src: int, dst: int
) -> None:
    m = _ARRAY_BUILDERS[src]()
    assert m.to_version(dst).zarr_format == dst


@pytest.mark.parametrize(("src", "dst"), ORDERED_PAIRS)
def test_array_roundtrips_losslessly(src: int, dst: int) -> None:
    m = _ARRAY_BUILDERS[src]()
    with warnings.catch_warnings():
        # a lossless fixture must not trip the loss policy
        warnings.simplefilter("error")
        back = m.to_version(dst).to_version(src)
    assert back == m


# ==========================================================================
#   Array metadata -- the lossy directions, under policy
# ==========================================================================


def _v3_sharded() -> v3.ArrayMetadata:
    return v3.ArrayMetadata.from_dict(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [10, 10],
            "data_type": "float64",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [10, 10]},
            },
            "chunk_key_encoding": {
                "name": "v2",
                "configuration": {"separator": "."},
            },
            "codecs": [
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
            ],
            "fill_value": 0,
            "attributes": {},
        }
    )


@pytest.mark.parametrize("target", [2, 1])
def test_sharding_loss_follows_policy(target: int) -> None:
    m = _v3_sharded()
    # lossy: silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.to_version(target, policy="lossy")
    # warn: one warning naming the field
    with pytest.warns(UserWarning, match="sharding"):
        m.to_version(target, policy="warn")
    # strict: raises, naming the field and version
    with pytest.raises(UnsupportedConversion) as info:
        m.to_version(target, policy="strict")
    assert info.value.field == "sharding"


@pytest.mark.parametrize("target", [1, 2])
def test_f_order_loss_to_v3_follows_policy(target: int) -> None:
    # v3 has no C/F memory-order field, so an F-order v2 array loses it
    m = v2.ArrayMetadata.from_dict(
        {
            "zarr_format": 2,
            "shape": [10, 10],
            "chunks": [5, 5],
            "dtype": "<f8",
            "compressor": None,
            "filters": [],
            "fill_value": 0,
            "order": "F",
            "dimension_separator": ".",
            "attributes": {},
        }
    )
    with pytest.warns(UserWarning, match="order"):
        m.to_version(3, policy="warn")
    with pytest.raises(UnsupportedConversion) as info:
        m.to_version(3, policy="strict")
    assert info.value.field == "order"


def test_filters_lost_to_v1_follows_policy() -> None:
    # v1 predates filters
    m = v2.ArrayMetadata.from_dict(
        {
            "zarr_format": 2,
            "shape": [10],
            "chunks": [5],
            "dtype": "<f8",
            "compressor": {"id": "zstd", "level": 3},
            "filters": [{"id": "delta", "dtype": "<f8"}],
            "fill_value": 0,
            "order": "C",
            "dimension_separator": ".",
            "attributes": {},
        }
    )
    with pytest.warns(UserWarning, match="filters"):
        m1 = m.to_version(1, policy="warn")
    assert m1.compression == "zstd"
    with pytest.raises(UnsupportedConversion) as info:
        m.to_version(1, policy="strict")
    assert info.value.field == "filters"


def test_non_regular_chunk_grid_has_no_v2_form() -> None:
    # a rectilinear grid leaves no chunk shape to build a v2 array from
    m = v3.ArrayMetadata.from_dict(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [10],
            "data_type": "float64",
            "chunk_grid": {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": [4, 6]},
            },
            "chunk_key_encoding": {
                "name": "v2",
                "configuration": {"separator": "."},
            },
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}}
            ],
            "fill_value": 0,
            "attributes": {},
        }
    )
    # unrepresentable in any policy -- a named error, not a bare ValueError
    for policy in ("lossy", "warn", "strict"):
        with pytest.raises(UnsupportedConversion) as info:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.to_version(2, policy=policy)
        assert info.value.field == "chunk_grid"


@pytest.mark.xfail(
    reason="v2 cannot hold a v3 array-to-bytes codec or the default "
    "chunk-key encoding, so this down-conversion is lossy by design",
    strict=True,
)
def test_v3_default_key_roundtrip_through_v2_is_lossy() -> None:
    # the intended-lossy case: a v3 array with the *default* chunk-key
    # encoding cannot survive a trip through v2
    m = v3.ArrayMetadata.from_dict(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [10, 10],
            "data_type": "float64",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [5, 5]},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}}
            ],
            "fill_value": 0,
            "attributes": {},
        }
    )
    assert m.to_version(2).to_version(3) == m


# ==========================================================================
#   Group metadata
# ==========================================================================
#
# A group carries only attributes and a format version. v2 <-> v3 is
# lossless; v1 has no group concept, so a group has no v1 form.


def _v2_group() -> GroupMetadataV2:
    return GroupMetadataV2.from_dict(
        {"zarr_format": 2, "attributes": {"a": 1, "list": [1, 2, 3]}}
    )


def _v3_group() -> GroupMetadataV3:
    return GroupMetadataV3.from_dict(
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"a": 1, "nested": {"b": [4, 5]}},
        }
    )


_GROUP_BUILDERS = {2: _v2_group, 3: _v3_group}


@pytest.mark.parametrize("src", [2, 3])
def test_group_identity_returns_self(src: int) -> None:
    g = _GROUP_BUILDERS[src]()
    assert g.to_version(src) is g


@pytest.mark.parametrize(("src", "dst"), [(2, 3), (3, 2)])
def test_group_roundtrips_losslessly(src: int, dst: int) -> None:
    g = _GROUP_BUILDERS[src]()
    converted = g.to_version(dst)
    assert converted.zarr_format == dst
    assert converted.to_version(src) == g
    # attributes, including nested containers, survive intact
    assert dict(converted.attributes) == dict(g.attributes)


@pytest.mark.parametrize("src", [2, 3])
def test_group_to_v1_is_unsupported(src: int) -> None:
    g = _GROUP_BUILDERS[src]()
    # v1 has no groups: raises under every policy, naming the node kind
    for policy in ("lossy", "warn", "strict"):
        with pytest.raises(UnsupportedConversion) as info:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g.to_version(1, policy=policy)
        assert info.value.field == "group"
        assert info.value.version == 1


# ==========================================================================
#   Data type
# ==========================================================================
#
# A data type is representable in every version (v1 shares v2's model), so
# every pair round-trips on the underlying numpy dtype.

_DTYPES = ["|b1", "|u1", "<i2", ">i4", "<i8", "<f4", ">f8", "<c8"]


def _dtype(version: int, spec: str) -> tx.Any:
    d2 = v2.DType(spec)
    return {1: d2, 2: d2, 3: d2.to_version(3)}[version]


@pytest.mark.parametrize("spec", _DTYPES)
@pytest.mark.parametrize(("src", "dst"), ORDERED_PAIRS)
def test_dtype_roundtrips_over_every_pair(
    spec: str, src: int, dst: int
) -> None:
    d = _dtype(src, spec)
    back = d.to_version(dst).to_version(src)
    # A v3 data type carries no byte order -- endianness lives in the array's
    # bytes codec, not the type -- so a trip through v3 normalizes it to
    # native. The type's kind and width always survive; full equality
    # (byte order included) holds only when v3 is not on the path.
    assert back.numpy.kind == d.numpy.kind
    assert back.numpy.itemsize == d.numpy.itemsize
    if dst != 3:
        assert back.numpy == d.numpy


def test_v3_data_type_drops_byte_order() -> None:
    # documents why a big-endian dtype does not round-trip through v3 alone
    big = v2.DType(">i4")
    assert big.numpy.byteorder == ">"
    assert big.to_version(3).numpy.byteorder in ("=", "<")


# ==========================================================================
#   Codec
# ==========================================================================
#
# v1, v2 and v3 all describe a numcodecs compressor, so a codec round-trips
# across every pair. Sources are built in each version and swept to every
# target and back.

_COMPRESSORS = [
    {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1,
     "blocksize": 0, "typesize": None},
    {"id": "gzip", "level": 5},
    {"id": "zstd", "level": 3},
    {"id": "zlib", "level": 4},
]


@pytest.mark.parametrize("spec", _COMPRESSORS)
@pytest.mark.parametrize("dst", VERSIONS)
def test_v2_codec_roundtrips_to_every_version(
    spec: dict, dst: int
) -> None:
    c = v2.Codec.from_dict(dict(spec))
    assert c.to_version(dst).to_version(2) == c


@pytest.mark.parametrize("spec", _COMPRESSORS)
@pytest.mark.parametrize("dst", VERSIONS)
def test_v3_codec_roundtrips_to_every_version(
    spec: dict, dst: int
) -> None:
    c3 = v2.Codec.from_dict(dict(spec)).to_version(3)
    back = c3.to_version(dst).to_version(3)
    # compare through v2, the common numcodecs form
    assert back.to_version(2) == c3.to_version(2)


@pytest.mark.parametrize("spec", [
    {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1,
     "blocksize": 0, "typesize": None},
    {"id": "gzip", "level": 5},
])
@pytest.mark.parametrize("dst", VERSIONS)
def test_v1_codec_options_roundtrip_to_every_version(
    spec: dict, dst: int
) -> None:
    # v1 codec options built from a v2 codec (the direct construction path)
    c1 = v2.Codec.from_dict(dict(spec)).to_version(1)
    assert c1.to_version(dst).to_version(1).to_dict() == c1.to_dict()


# ==========================================================================
#   Filter (v2) -> v3 codec
# ==========================================================================
#
# A v2 filter converts to a v3 codec. Five filters used to crash reading a
# non-existent ``self.name`` attribute; three others emitted a bare id that
# names no v3 codec.


# filter spec -> the v3 codec name it must produce
_NATIVE_FILTERS = [
    ({"id": "bitround", "keepbits": 3}, "bitround"),
    ({"id": "packbits"}, "packbits"),
    (
        {"id": "fixedscaleoffset", "offset": 1000.0, "scale": 10.0,
         "dtype": "<f8", "astype": "<i2"},
        "scale_offset",
    ),
    (
        {"id": "astype", "encode_dtype": "<i2", "decode_dtype": "<f8"},
        "cast_value",
    ),
    (
        {"id": "categorize", "labels": ["a", "b"], "dtype": "<i1",
         "astype": "<i1"},
        "cast_value",
    ),
]


@pytest.mark.parametrize(("spec", "v3_name"), _NATIVE_FILTERS)
def test_native_filter_maps_to_its_v3_codec(
    spec: dict, v3_name: str
) -> None:
    f = v2.Filter.from_dict(dict(spec))
    codec = f.to_version(3)  # used to raise AttributeError
    assert codec.name == v3_name
    # to_dict must be valid (no numpy objects, real JSON)
    codec.to_dict()
    # converting to the same version is identity
    assert f.to_version(2) is f


# a numcodecs filter with no dedicated v3 codec maps to numcodecs.<id>
_NUMCODECS_FILTERS = [
    {"id": "delta", "dtype": "<f8", "astype": "<f8"},
    {"id": "quantize", "digits": 3, "dtype": "<f8", "astype": "<f8"},
    {"id": "shuffle", "elementsize": 4},
]


@pytest.mark.parametrize("spec", _NUMCODECS_FILTERS)
def test_numcodecs_filter_maps_to_namespaced_v3_codec(spec: dict) -> None:
    f = v2.Filter.from_dict(dict(spec))
    codec = f.to_version(3)
    # a valid v3 codec name, not a bare id that names no v3 codec
    assert codec.name == "numcodecs." + spec["id"]
    # and it round-trips back to the original v2 filter's payload (the v3
    # codec resolves to the generic v2 Codec, so compare the serialized form
    # rather than the concrete filter type)
    assert codec.to_version(2).to_dict() == f.to_dict()


def test_filter_to_v1_is_unsupported() -> None:
    # v1 has no filter concept
    f = v2.Filter.from_dict({"id": "delta", "dtype": "<f8"})
    with pytest.raises(ValueError):
        f.to_version(1)


# ==========================================================================
#   transpose -- the v3 -> v2 permutation must survive (regression)
# ==========================================================================


def test_transpose_permutation_survives_v3_to_v2() -> None:
    # a v3 transpose codec carries an ``order`` list; converting the array
    # to v2 used to coerce that list to the boolean ``True``
    m3 = v3.ArrayMetadata.from_dict(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [4, 6],
            "data_type": "int32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [2, 3]},
            },
            "chunk_key_encoding": {
                "name": "v2",
                "configuration": {"separator": "."},
            },
            "codecs": [
                {"name": "transpose", "configuration": {"order": [1, 0]}},
                {"name": "bytes", "configuration": {"endian": "little"}},
            ],
            "fill_value": 0,
            "attributes": {},
        }
    )
    m2 = m3.to_version(2)
    (transpose,) = [f for f in m2.filters if f.to_dict()["id"] == "transpose"]
    assert transpose.to_dict()["order"] == [1, 0]


def test_transpose_codec_to_v2_preserves_order_directly() -> None:
    tc = v3.TransposeCodec.from_dict(
        {"name": "transpose", "configuration": {"order": [2, 0, 1]}}
    )
    assert tc.to_version(2).to_dict()["order"] == [2, 0, 1]


# ==========================================================================
#   Filters inside a full array conversion (array-level path)
# ==========================================================================


@pytest.mark.parametrize(("spec", "v3_name"), _NATIVE_FILTERS)
def test_array_with_filter_converts_to_v3(
    spec: dict, v3_name: str
) -> None:
    # the array-level path routes each filter through its to_version(3)
    m2 = v2.ArrayMetadata.from_dict(
        {
            "zarr_format": 2,
            "shape": [8],
            "chunks": [4],
            "dtype": "<f8",
            "compressor": None,
            "filters": [dict(spec)],
            "fill_value": 0,
            "order": "C",
            "dimension_separator": ".",
            "attributes": {},
        }
    )
    m3 = m2.to_version(3)
    names = [c.name for c in m3.codecs]
    assert v3_name in names

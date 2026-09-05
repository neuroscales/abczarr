"""Tests for variable-length string/bytes v3 data types (#127).

Zarr v3 spells variable-length text ``string`` and variable-length bytes
``bytes``. numpy has no fixed-width scalar for either, so the conventional
Zarr representation -- the one zarr-python uses -- is numpy ``object`` (``|O``)
carrying a vlen codec: ``vlen-utf8`` for ``string`` and ``vlen-bytes`` for
``bytes``. ``asdtype`` maps both names to numpy ``object`` (rather than
raising ``UnsupportedConversion``), and the array-metadata conversion carries
the matching vlen codec across the v3 <-> v2 boundary so the "is-a-string /
is-a-bytes" tag numpy ``object`` drops is restored.
"""

# dependencies
import numpy as np
import pytest

# package
from abczarr._core.dtypes import asdtype, to_zarr2
from abczarr.metadata.v2 import ArrayMetadata as ArrayMetadataV2
from abczarr.metadata.v3 import ArrayMetadata as ArrayMetadataV3
from abczarr.metadata.v3.dtypes import DType as DTypeV3


@pytest.mark.parametrize("name", ["string", "bytes"])
def test_asdtype_vlen_maps_to_object(name: str) -> None:
    # the v3 extension mapping ({"name": "string"}) and the bare name both
    # resolve to numpy object rather than raising UnsupportedConversion
    assert asdtype({"name": name}) == np.dtype("object")
    assert asdtype(name) == np.dtype("object")


@pytest.mark.parametrize("name", ["string", "bytes"])
def test_asdtype_vlen_dtype_object(name: str) -> None:
    # a v3 DType serializes to the bare name; asdtype resolves it to object
    assert asdtype(DTypeV3.from_json(name)) == np.dtype("object")


@pytest.mark.parametrize("name", ["string", "bytes"])
def test_to_zarr2_vlen_is_object(name: str) -> None:
    # the v2 dtype for a variable-length string/bytes array is "|O"
    assert to_zarr2({"name": name}) == "|O"


def _v3_vlen(data_type: str, vlen: str) -> ArrayMetadataV3:
    """A v3 array whose pipeline serializes with *vlen* (like zarr-python)."""
    return ArrayMetadataV3.from_json(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [3],
            "data_type": data_type,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [3]},
            },
            "chunk_key_encoding": {
                "name": "v2",
                "configuration": {"separator": "."},
            },
            "fill_value": None,
            "codecs": [
                {"name": vlen, "configuration": {}},
                {"name": "zstd", "configuration": {"level": 0,
                                                   "checksum": False}},
            ],
            "attributes": {},
        }
    )


def _v2_vlen(vlen: str) -> ArrayMetadataV2:
    """A v2 |O array carrying the *vlen* filter (like zarr-python)."""
    return ArrayMetadataV2.from_json(
        {
            "zarr_format": 2,
            "node_type": "array",
            "shape": [3],
            "chunks": [3],
            "dtype": "|O",
            "fill_value": None,
            "order": "C",
            "filters": [{"id": vlen}],
            "compressor": None,
            "dimension_separator": ".",
            "attributes": {},
        }
    )


@pytest.mark.parametrize(
    "data_type, vlen",
    [("string", "vlen-utf8"), ("bytes", "vlen-bytes")],
)
def test_v3_to_v2_wires_vlen_filter(data_type: str, vlen: str) -> None:
    # v3 string/bytes -> v2 |O dtype with the matching vlen filter, matching
    # zarr-python (dtype "|O", filters=[{"id": "vlen-utf8"/"vlen-bytes"}])
    v2 = _v3_vlen(data_type, vlen).to_version(2)
    assert str(v2.dtype) == "|O"
    assert [f.id for f in v2.filters] == [vlen]
    # the trailing bytes->bytes codec still becomes the v2 compressor
    assert v2.compressor is not None and v2.compressor.id == "zstd"


@pytest.mark.parametrize(
    "data_type, vlen",
    [("string", "vlen-utf8"), ("bytes", "vlen-bytes")],
)
def test_v3_to_v2_emits_vlen_even_without_pipeline_codec(
    data_type: str, vlen: str
) -> None:
    # even if the pipeline omits the vlen codec, the string/bytes data type
    # still yields the vlen filter -- the tag is never silently dropped
    meta = ArrayMetadataV3.from_json(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [3],
            "data_type": data_type,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [3]},
            },
            "chunk_key_encoding": {
                "name": "v2",
                "configuration": {"separator": "."},
            },
            "fill_value": None,
            "codecs": [{"name": "bytes"}],
            "attributes": {},
        }
    )
    v2 = meta.to_version(2)
    assert str(v2.dtype) == "|O"
    assert [f.id for f in v2.filters] == [vlen]


@pytest.mark.parametrize(
    "vlen, data_type",
    [("vlen-utf8", "string"), ("vlen-bytes", "bytes")],
)
def test_v2_to_v3_restores_data_type(vlen: str, data_type: str) -> None:
    # v2 |O + vlen filter -> v3 string/bytes data type whose vlen codec is the
    # serializer; no separate bytes serializer is added (matches zarr-python)
    v3 = _v2_vlen(vlen).to_version(3)
    assert v3.data_type.to_json() == data_type
    assert [c.name for c in v3.codecs] == [vlen]


@pytest.mark.parametrize(
    "data_type, vlen",
    [("string", "vlen-utf8"), ("bytes", "vlen-bytes")],
)
def test_v3_v2_v3_round_trip(data_type: str, vlen: str) -> None:
    round_trip = _v3_vlen(data_type, vlen).to_version(2).to_version(3)
    assert round_trip.data_type.to_json() == data_type
    assert [c.name for c in round_trip.codecs] == [vlen, "zstd"]


@pytest.mark.parametrize(
    "vlen, data_type",
    [("vlen-utf8", "string"), ("vlen-bytes", "bytes")],
)
def test_v2_v3_v2_round_trip(vlen: str, data_type: str) -> None:
    round_trip = _v2_vlen(vlen).to_version(3).to_version(2)
    assert str(round_trip.dtype) == "|O"
    assert [f.id for f in round_trip.filters] == [vlen]

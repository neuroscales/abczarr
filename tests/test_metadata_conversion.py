"""Cross-version array-metadata conversion.

The conversion paths were previously untested. This suite locks in the two
crashes fixed here and tracks the remaining gap in an executable form.

Fixed here -- every ``v2 -> v3`` conversion used to raise:
  * the chunk-key encoding was built with the separator in the ``name`` slot
    (a ``Literal["v2"]``), so construction rejected it;
  * the dtype conversion handed a numpy object to a registry that matches
    type names by regex on strings.

Still open (see the roadmap, "all-versions metadata & lossless conversion"),
and a larger, version-sensitive effort of its own -- a v3 round trip through
v2 drops the array-to-bytes codec and the default-vs-v2 chunk-key encoding;
a v2 array carrying a compressor still errors in the codec layer; conversion
to/from Zarr v1 is not implemented; and the round trip touches attrs
internals that differ across interpreters. These want the strict / annotate
/ lossy policy, not a silent best effort. Only version-independent facts are
asserted below.
"""

from __future__ import annotations

import pytest

from abczarr.metadata import v2, v3


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
# tracked gap (executable follow-up): lossless round trips
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="v3->v2 drops the bytes codec and default/v2 chunk-key encoding; "
    "needs the strict/annotate/lossy conversion policy",
    strict=False,
)
def test_v3_roundtrips_losslessly_through_v2() -> None:
    m3 = v3.ArrayMetadata.from_dict(_v3())
    assert m3.to_version(2).to_version(3) == m3

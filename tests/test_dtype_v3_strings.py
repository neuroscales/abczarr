"""Regression tests for Unicode/bytes numpy dtypes and categorize (#104).

Two related defects produced non-Zarr-v3 data:

* ``to_zarr3`` on a fixed-length Unicode (``<U5``) or byte (``S3``) numpy
  dtype fell through to ``dtype.name`` and returned numpy's internal
  spelling (``"str160"``, ``"bytes24"``), neither of which is a valid Zarr
  v3 data type. It now maps to the ``fixed_length_utf32`` /
  ``null_terminated_bytes`` extension data types, carrying the storage size
  in ``length_bytes``, and ``asdtype`` reverses the mapping so the numpy
  dtype round-trips.
* ``CategorizeFilter.to_version(3)`` built its ``scalar_map`` ``encode``
  entries from ``reversed`` iterators, which are single-use: reading the
  mapping a second time yielded nothing. The entries are now materialized.
"""

# dependencies
import numpy as np
import pytest
import typing_extensions as tx

# package
from abczarr._core.dtypes import asdtype, to_zarr3
from abczarr.metadata.v2.filters.extensions import CategorizeFilter


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("<U5", {"name": "fixed_length_utf32",
                 "configuration": {"length_bytes": 20}}),
        ("<U1", {"name": "fixed_length_utf32",
                 "configuration": {"length_bytes": 4}}),
        ("S3", {"name": "null_terminated_bytes",
                "configuration": {"length_bytes": 3}}),
        ("S1", {"name": "null_terminated_bytes",
                "configuration": {"length_bytes": 1}}),
    ],
)
def test_to_zarr3_string_bytes_maps_to_extension(
    spelling: str, expected: dict
) -> None:
    # never numpy's internal name ("str160"/"bytes24"); the v3 extension type
    assert to_zarr3(np.dtype(spelling)) == expected


@pytest.mark.parametrize("spelling", ["<U1", "<U5", "S1", "S3"])
def test_to_zarr3_string_bytes_round_trips(spelling: str) -> None:
    # the v3 extension dict reverses back to the original numpy dtype
    dtype = np.dtype(spelling)
    assert asdtype(to_zarr3(dtype)) == dtype


def test_to_zarr3_numeric_still_converts() -> None:
    # the refusal is scoped to Unicode/bytes; numeric dtypes are unaffected
    assert to_zarr3(np.dtype("<i4")) == "int32"
    assert to_zarr3(np.dtype("<f8")) == "float64"


def test_categorize_to_v3_encode_mapping_is_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abczarr.metadata.v3 as v3

    captured = {}
    original = v3.CastValueCodec.from_json.__func__

    def spy(cls: type, data: dict) -> tx.Any:
        captured["configuration"] = data["configuration"]
        return original(cls, data)

    monkeypatch.setattr(
        v3.CastValueCodec, "from_json", classmethod(spy)
    )

    filter_ = CategorizeFilter(
        id="categorize",
        labels=("a", "b", "c"),
        dtype=np.dtype("uint8"),
        astype=np.dtype("uint8"),
    )
    filter_.to_version(3)

    encode = captured["configuration"]["scalar_map"]["encode"]
    first = [list(pair) for pair in encode]
    second = [list(pair) for pair in encode]

    # non-empty, and every pair survives a second read (not a spent iterator)
    assert first == [["a", 0], ["b", 1], ["c", 2]]
    assert first == second

"""Attribute values survive parsing unchanged.

A zarr node's attributes are arbitrary Json. Reading metadata must preserve
those values exactly -- strings, numbers, booleans, lists and nested dicts --
and must not coerce a value to some other type it happens to look like.
"""

import math

from abczarr.metadata import v2, v3


def _v2(attributes: dict) -> dict:
    return {
        "zarr_format": 2,
        "shape": [2],
        "chunks": [1],
        "dtype": "<f8",
        "compressor": None,
        "filters": [],
        "fill_value": 0,
        "order": "C",
        "dimension_separator": ".",
        "attributes": attributes,
    }


def test_attribute_values_are_preserved_exactly() -> None:
    attrs = {
        "s": "x",
        "i": 42,
        "f": 3.5,
        "b": True,
        "l": [1, 2, "three"],
        "d": {"a": 1, "nested": {"b": [True, "y"]}},
    }
    parsed = dict(v2.ArrayMetadata.from_dict(_v2(attrs)).attributes)
    assert parsed == attrs


def test_string_attribute_is_not_coerced_to_bool() -> None:
    parsed = v2.ArrayMetadata.from_dict(_v2({"note": "true-ish"})).attributes
    assert parsed["note"] == "true-ish"


def test_fill_value_string_still_coerces_to_number() -> None:
    # a value that does NOT already fit the field is still coerced
    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2],
        "data_type": "float64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [1]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": "NaN",
        "attributes": {},
    }
    assert math.isnan(v3.ArrayMetadata.from_dict(meta).fill_value)

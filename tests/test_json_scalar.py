"""Tests for values annotated with `JsonScalar` and `FrozenJson`."""

import math

import numpy as np
import pytest

from abczarr._core import typing as tz
from abczarr._core.attrs import ConversionError, get_converter
from abczarr._core.frozendict import FrozenDict
from abczarr.metadata.v3.codecs.extensions import ScalarMap

SCALAR_CASES = [
    (None, None, type(None)),
    ("abc", "abc", str),
    (np.str_("abc"), "abc", str),
    ("3", "3", str),
    ("NaN", "NaN", str),
    ("-Infinity", "-Infinity", str),
    (True, True, bool),
    (False, False, bool),
    (np.True_, True, bool),
    (np.False_, False, bool),
    (3, 3, int),
    (np.int64(4), 4, int),
    (np.uint8(7), 7, int),
    (1.5, 1.5, float),
    (np.float32(0.25), 0.25, float),
    (np.float64(2.0), 2.0, float),
]

INVALID_SCALARS = [1 + 2j, b"abc", object()]


@pytest.mark.parametrize("value, expected, kind", SCALAR_CASES)
@pytest.mark.parametrize("hint", [tz.JsonScalar, tz.FrozenJson])
def test_scalar_converter(
    hint: object, value: object, expected: object, kind: type
) -> None:
    """Each scalar becomes the builtin type JSON uses for it."""
    result = get_converter(hint)(value)
    assert result == expected
    assert type(result) is kind


@pytest.mark.parametrize("hint", [tz.JsonScalar, tz.FrozenJson])
def test_scalar_converter_keeps_nan(hint: object) -> None:
    """A NaN stays a builtin float NaN."""
    result = get_converter(hint)(np.float32("nan"))
    assert type(result) is float
    assert math.isnan(result)


@pytest.mark.parametrize("value", INVALID_SCALARS)
@pytest.mark.parametrize("hint", [tz.JsonScalar, tz.FrozenJson])
def test_scalar_converter_rejects_non_json(
    hint: object, value: object
) -> None:
    """A value that is not a JSON scalar cannot be converted."""
    with pytest.raises(ConversionError):
        get_converter(hint)(value)


def test_json_scalar_rejects_containers() -> None:
    """A container is not a JSON scalar."""
    with pytest.raises(ConversionError):
        get_converter(tz.JsonScalar)([1, 2])


def test_frozen_json_converts_nested_leaves() -> None:
    """Every leaf of a nested value is converted."""
    value = {
        "a": [np.float32(0.25), np.int64(2), {"b": np.True_}],
        "c": (np.str_("x"), None, "1"),
    }
    result = get_converter(tz.FrozenJson)(value)
    assert result == FrozenDict(
        a=(0.25, 2, FrozenDict(b=True)), c=("x", None, "1")
    )
    inner = result["a"]
    assert [type(item) for item in inner[:2]] == [float, int]
    assert type(inner[2]["b"]) is bool
    assert type(result["c"][0]) is str
    hash(result)


def test_frozen_json_rejects_nested_non_json() -> None:
    """A leaf that is not JSON fails the whole conversion."""
    with pytest.raises(ConversionError):
        get_converter(tz.FrozenJson)({"a": [1, b"x"]})


def test_mutable_json_keeps_scalar_kind() -> None:
    """A numpy scalar at the top of `MutableJson` keeps its kind."""
    convert = get_converter(tz.MutableJson)
    assert type(convert(np.float32(0.25))) is float
    assert type(convert(np.True_)) is bool
    assert convert([1, 2]) == [1, 2]


def test_scalar_map_keeps_values() -> None:
    """The scalar map keeps special strings and converts numbers."""
    scalar_map = ScalarMap(
        encode=[["NaN", 0], [np.float32(1.5), "Infinity"]],
        decode=[[np.int64(3), np.True_], ["-Infinity", -1.0]],
    )
    assert scalar_map.encode == (("NaN", 0), (1.5, "Infinity"))
    assert scalar_map.decode == ((3, True), ("-Infinity", -1.0))
    assert type(scalar_map.encode[1][0]) is float
    assert type(scalar_map.decode[0][0]) is int
    assert type(scalar_map.decode[0][1]) is bool

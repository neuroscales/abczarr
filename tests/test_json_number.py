"""Tests for fields annotated with `JsonNumber`."""

import numpy as np
import pytest

from abczarr._core import typing as tz
from abczarr._core.attrs import ConversionError, get_converter
from abczarr.metadata.v3.codecs.extensions import ScaleOffsetConfig
from abczarr.ome.v0_5.omero import Channel
from abczarr.ome.v0_5.transformations import Scale
from abczarr.ome.v0_6.transformations import Affine


@pytest.mark.parametrize(
    "value, expected, kind",
    [
        (3, 3, int),
        (1.5, 1.5, float),
        (np.float32(0.25), 0.25, float),
    ],
)
def test_scale_offset_keeps_number_kind(
    value: object, expected: float, kind: type
) -> None:
    """An int stays an int, and a numpy float is not truncated."""
    offset = ScaleOffsetConfig(offset=value, scale=1).offset
    assert offset == expected
    assert type(offset) is kind


def test_omero_window_keeps_number_kind() -> None:
    """The omero window keeps ints and converts numpy floats."""
    window = Channel.Window(min=0, max=np.float32(0.5), start=1.5, end=255)
    assert [window.min, window.max, window.start, window.end] == [
        0,
        0.5,
        1.5,
        255,
    ]
    assert [type(window.min), type(window.max)] == [int, float]


@pytest.mark.parametrize(
    "value, expected, kind",
    [
        (4, 4, int),
        (np.int64(4), 4, int),
        (np.uint8(7), 7, int),
        (1.5, 1.5, float),
        (np.float32(0.25), 0.25, float),
        (np.float64(2.0), 2.0, float),
        ("3", 3, int),
        ("-12", -12, int),
        ("2.5", 2.5, float),
        ("1e3", 1000.0, float),
        (True, True, bool),
        (False, False, bool),
    ],
)
@pytest.mark.parametrize("hint", [tz.JsonNumber, tz.JsonNumberLike])
def test_json_number_converter(
    hint: object, value: object, expected: object, kind: type
) -> None:
    """Integral values become ints, other reals become floats."""
    result = get_converter(hint)(value)
    assert result == expected
    assert type(result) is kind


@pytest.mark.parametrize("value", ["abc", 1 + 2j, None])
def test_json_number_rejects_non_numbers(value: object) -> None:
    """A value that is not a real number cannot be converted."""
    with pytest.raises(ConversionError):
        get_converter(tz.JsonNumber)(value)


def test_ome_scale_keeps_number_kinds() -> None:
    """Each item of an OME list of numbers is converted on its own."""
    scale = Scale(scale=[np.int64(2), np.float32(0.25), 3]).scale
    assert scale == [2, 0.25, 3]
    assert [type(item) for item in scale] == [int, float, int]


def test_ome_affine_keeps_number_kinds() -> None:
    """The items of a nested OME list of numbers are converted."""
    affine = Affine(affine=[[np.int64(1), np.float32(0.5)], [0, 1]]).affine
    assert affine == [[1, 0.5], [0, 1]]
    assert [type(item) for item in affine[0]] == [int, float]

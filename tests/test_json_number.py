"""Tests for fields annotated with `JsonNumber`."""

import numpy as np
import pytest

from abczarr.metadata.v3.codecs.extensions import ScaleOffsetConfig
from abczarr.ome.v0_5.omero import Channel


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

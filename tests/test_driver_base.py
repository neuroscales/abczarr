"""The Driver abstraction: a driver declares its feature support, and
selection is a set difference against an array's required features.

Uses fake drivers with a static feature map, so it needs no backend.
"""

import pytest

from abczarr._errors import UnsupportedZarrOperation
from abczarr.abc.capabilities import Support
from abczarr.api._registry import select_driver
from abczarr.drivers.base import Driver, Verdict
from abczarr.metadata import v3


def _array(codecs: list) -> "v3.ArrayMetadata":
    return v3.ArrayMetadata.from_json(
        {
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
            "codecs": codecs,
            "fill_value": 0,
            "attributes": {},
        }
    )


_BYTES = {"name": "bytes", "configuration": {"endian": "little"}}
_ZSTD = {"name": "zstd", "configuration": {"level": 5}}


def _caps(*keys: str) -> dict:
    return {key: Support.NATIVE for key in keys}


_CORE = (
    "v3:chunk_grid:regular",
    "v3:chunk_key_encoding:default",
    "v3:data_type:float32",
    "v3:codec:bytes",
)


class _Full(Driver):
    name = "full"
    _CAPABILITIES = _caps(*_CORE, "v3:codec:zstd")


class _NoZstd(Driver):
    name = "no-zstd"
    _CAPABILITIES = _caps(*_CORE)


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def test_verdict_true_when_nothing_missing() -> None:
    v = Verdict("d", [])
    assert bool(v) is True
    assert v.missing == ()
    assert "can open" in v.reason


def test_verdict_false_lists_missing_sorted() -> None:
    v = Verdict("d", ["v3:codec:zstd", "v3:codec:blosc"])
    assert bool(v) is False
    assert v.missing == ("v3:codec:blosc", "v3:codec:zstd")
    assert "v3:codec:zstd" in v.reason


# --------------------------------------------------------------------------
# can_open
# --------------------------------------------------------------------------


def test_can_open_true_when_every_feature_is_provided() -> None:
    assert bool(_Full().can_open(_array([_BYTES, _ZSTD]))) is True


def test_can_open_names_the_missing_codec() -> None:
    verdict = _NoZstd().can_open(_array([_BYTES, _ZSTD]))
    assert bool(verdict) is False
    assert verdict.missing == ("v3:codec:zstd",)


def test_can_open_a_plainer_array_that_needs_no_extra_codec() -> None:
    # no-zstd can still open an array that only uses bytes
    assert bool(_NoZstd().can_open(_array([_BYTES]))) is True


# --------------------------------------------------------------------------
# select_driver
# --------------------------------------------------------------------------


def test_select_returns_first_capable_driver() -> None:
    meta = _array([_BYTES, _ZSTD])
    # the capable driver is chosen even though it is listed second
    assert select_driver(meta, [_NoZstd(), _Full()]).name == "full"


def test_select_prefers_the_earlier_capable_driver() -> None:
    meta = _array([_BYTES])
    assert select_driver(meta, [_NoZstd(), _Full()]).name == "no-zstd"


def test_select_raises_naming_each_driver_gap() -> None:
    meta = _array([_BYTES, _ZSTD])
    with pytest.raises(UnsupportedZarrOperation) as info:
        select_driver(meta, [_NoZstd()])
    assert "v3:codec:zstd" in str(info.value)


def test_select_with_no_drivers_raises() -> None:
    with pytest.raises(UnsupportedZarrOperation):
        select_driver(_array([_BYTES]), [])

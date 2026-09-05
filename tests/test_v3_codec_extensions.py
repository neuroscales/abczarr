"""The v3 extension codecs added last: reshape and zfp.

These round-trip through the metadata model (no backend needed).
"""

import pytest

from abczarr.metadata.v3.codecs import (
    ConditionalCodec,
    N5DefaultCodec,
    ReshapeCodec,
    ZfpCodec,
)
from abczarr.metadata.v3.codecs.base import Codec
from abczarr.metadata.v3.codecs.builtin import BytesCodec, TransposeCodec


@pytest.mark.parametrize(
    "shape",
    [[2, -1], [[2, 3], 4], [10]],
)
def test_reshape_round_trips(shape: list) -> None:
    spec = {"name": "reshape", "configuration": {"shape": shape}}
    assert ReshapeCodec.from_json(spec).to_json() == spec


def test_reshape_dispatches_by_name() -> None:
    codec = Codec.from_json(
        {"name": "reshape", "configuration": {"shape": [1]}}
    )
    assert isinstance(codec, ReshapeCodec)


@pytest.mark.parametrize(
    "configuration",
    [
        {"mode": "reversible"},
        {"mode": "expert", "minbits": 1, "maxbits": 4, "maxprec": 8,
         "minexp": -1},
        {"mode": "fixed_accuracy", "tolerance": 0.001},
        {"mode": "fixed_rate", "rate": 8.5},
        {"mode": "fixed_precision", "precision": 12},
    ],
)
def test_zfp_each_mode_round_trips_with_only_its_own_parameters(
    configuration: dict,
) -> None:
    spec = {"name": "zfp", "configuration": configuration}
    # a mode writes only its own parameters; the others are omitted
    assert ZfpCodec.from_json(spec).to_json() == spec


def test_zfp_dispatches_by_name() -> None:
    codec = Codec.from_json(
        {"name": "zfp", "configuration": {"mode": "reversible"}}
    )
    assert isinstance(codec, ZfpCodec)


@pytest.mark.parametrize(
    "codecs",
    [
        [{"name": "vlen-bytes"}],
        [{"name": "vlen-bytes"}, {"name": "vlen-utf8"}],
    ],
)
def test_conditional_round_trips_with_any_number_of_codecs(
    codecs: list,
) -> None:
    # conditional's codecs list is variable length: one codec or several
    # both round-trip (a two-codec list once failed the length-1 tuple type).
    spec = {"name": "conditional", "configuration": {"codecs": codecs}}
    assert ConditionalCodec.from_json(spec).to_json() == spec


def test_conditional_dispatches_by_name() -> None:
    codec = Codec.from_json(
        {
            "name": "conditional",
            "configuration": {
                "codecs": [{"name": "vlen-bytes"}, {"name": "vlen-utf8"}]
            },
        }
    )
    assert isinstance(codec, ConditionalCodec)


_N5_DEFAULT_SPEC = {
    "name": "n5_default",
    "configuration": {
        "codecs": [
            {"name": "transpose", "configuration": {"order": [0, 1]}},
            {"name": "bytes", "configuration": {"endian": "big"}},
            {"name": "gzip", "configuration": {"level": 5}},
        ]
    },
}


def test_n5_default_codecs_are_converted_objects_not_raw_dicts() -> None:
    # The stored codec chain must be converted codec objects, not the raw
    # dicts they were read from -- a transpose codec, a bytes codec, then
    # the trailing codecs.
    codecs = N5DefaultCodec.from_json(_N5_DEFAULT_SPEC).configuration.codecs
    assert not any(isinstance(codec, dict) for codec in codecs)
    assert all(isinstance(codec, Codec) for codec in codecs)
    assert isinstance(codecs[0], TransposeCodec)
    assert isinstance(codecs[1], BytesCodec)


def test_n5_default_round_trips() -> None:
    assert N5DefaultCodec.from_json(_N5_DEFAULT_SPEC).to_json() == (
        _N5_DEFAULT_SPEC
    )


def test_n5_default_dispatches_by_name() -> None:
    codec = Codec.from_json(_N5_DEFAULT_SPEC)
    assert isinstance(codec, N5DefaultCodec)

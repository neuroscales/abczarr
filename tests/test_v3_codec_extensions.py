"""The v3 extension codecs added last: reshape and zfp.

These round-trip through the metadata model (no backend needed).
"""

import pytest

from abczarr.metadata.v3.codecs import ReshapeCodec, ZfpCodec
from abczarr.metadata.v3.codecs.base import Codec


@pytest.mark.parametrize(
    "shape",
    [[2, -1], [[2, 3], 4], [10]],
)
def test_reshape_round_trips(shape: list) -> None:
    spec = {"name": "reshape", "configuration": {"shape": shape}}
    assert ReshapeCodec.from_dict(spec).to_dict() == spec


def test_reshape_dispatches_by_name() -> None:
    codec = Codec.from_dict(
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
    assert ZfpCodec.from_dict(spec).to_dict() == spec


def test_zfp_dispatches_by_name() -> None:
    codec = Codec.from_dict(
        {"name": "zfp", "configuration": {"mode": "reversible"}}
    )
    assert isinstance(codec, ZfpCodec)

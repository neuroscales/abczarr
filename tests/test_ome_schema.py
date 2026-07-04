import json
from pathlib import Path

import pytest
import typing_extensions as tx

from abczarr.ome import schemas
from abczarr._core.auto.validators import get_validator


TESTDIR = Path(__file__).parent


class SystemsAndTransforms(schemas.OMESchemaItem):
    coordinateSystems: tx.List[schemas.v0_6rc0.CoordinateSystem]
    coordinateTransformations: tx.List[schemas.v0_6rc0.CoordinateTransformation]


@pytest.mark.parametrize("example", [
    "affine2d2d_with_channel",
    "affine2d2d",
    "affine2d3d",
    "bijection",
    "byDimension1",
    "byDimension2",
    "byDimensionXarray",
    "identity",
    "mapAxis1",
    "rotation",
    "scale_with_discrete",
    "scale",
    "sequence",
    "translation",
    "xarrayLike"
])
def test_06rc0_xforms(example):
    path = TESTDIR / "data" / "ome" / "v0_6rc0" / f"{example}.json"
    with path.open("r") as f:
        data = json.load(f)
    validator = get_validator(SystemsAndTransforms)
    validator(data)  # should not raise

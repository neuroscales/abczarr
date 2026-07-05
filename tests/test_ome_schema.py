import json
from pathlib import Path

import pytest
import typing_extensions as tx

from abczarr.ome import schemas
from abczarr._core.auto.validators import get_validator, ValidationError


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
    "projectAxis",
    "projectAxis2",
    "rotation",
    "scale",
    "scale_with_discrete",
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


# @pytest.mark.parametrize("example", [
#     "byDimensionInvalid1",
#     "byDimensionInvalid2",
# ])
# def test_06rc0_xforms_invalid(example):
#     path = TESTDIR / "data" / "ome" / "v0_6rc0" / f"{example}.json"
#     with path.open("r") as f:
#         data = json.load(f)
#     validator = get_validator(SystemsAndTransforms)
#     with pytest.raises(ValidationError):
#         validator(data)


@pytest.mark.parametrize("example", [
    "colors_properties",
    "multiscales_example",
    "multiscales_example_relative",
    "multiscales_reference_to_label",
    "multiscales_transformations",
    "plate",
    "plate_2wells",
    "plate_6wells",
    "scene_registration",
    "scene_stitching",
    "series-2",
    "well_2fields",
    "well_4fields",
])
def test_06rc0_ome(example):
    path = TESTDIR / "data" / "ome" / "v0_6rc0" / f"{example}.json"
    with path.open("r") as f:
        data = json.load(f)
    if "attributes" in data:
        # Some JSON files are `zarr.json` (which contain a top-level
        # `attributes` key) and some are just the attributes themselves.
        data = data["attributes"]
    validator = get_validator(schemas.v0_6rc0.OMEAttributes)
    validator(data)  # should not raise

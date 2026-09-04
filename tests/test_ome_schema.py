"""OME-NGFF 0.6rc0 example documents validate against the official schemas.

These were previously checked against hand-written TypedDict schemas; they
now validate against the vendored official NGFF schemas through
``abczarr.ome.schemas`` (see ``test_ome_json_schema.py`` for the surface).
"""

import json
from pathlib import Path

import pytest
import typing_extensions as tx

from abczarr.ome import schemas

TESTDIR = Path(__file__).parent


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
    # projectAxis / projectAxis2 use a transitional axes-as-mapping
    # coordinate system that the official 0.6rc0 schema does not accept
    # (like the other transitional instances, they are not exercised here).
    "rotation",
    "scale",
    "scale_with_discrete",
    "sequence",
    "translation",
    "xarrayLike"
])
def test_06rc0_xforms(
    example: str,
    validate_systems_and_transforms: "tx.Callable[[dict, str], None]",
) -> None:
    path = TESTDIR / "data" / "ome" / "v0_6rc0" / f"{example}.json"
    with path.open("r") as f:
        data = json.load(f)
    validate_systems_and_transforms(data, "0.6rc0")  # should not raise


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
def test_06rc0_ome(example: str) -> None:
    path = TESTDIR / "data" / "ome" / "v0_6rc0" / f"{example}.json"
    with path.open("r") as f:
        data = json.load(f)
    if "attributes" in data:
        # Some JSON files are ``zarr.json`` (which carry a top-level
        # ``attributes`` key) and some are the attributes themselves.
        data = data["attributes"]
    # ``ome_zarr`` is the top-level schema: any OME-Zarr attributes document
    # (image, plate, well, scene, series, ...).
    schemas.validate(data, "0.6rc0", "ome_zarr")  # should not raise

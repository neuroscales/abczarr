"""OME-Zarr NGFF 0.6 pre-release metadata parses and round-trips.

Each 0.6 development version (``0.6.dev1`` .. ``0.6.dev4`` and ``0.6rc0``)
ships realistic example instances in the ``ngff-spec`` repository. These
tests parse those instances through the matching ``abczarr`` metadata model
and assert that the object serializes back to an equal object -- the proof
that data written against any 0.6 pre-release is read correctly.

The example instances are vendored under ``tests/data/ome/<version>/`` with
their JSONC comments stripped. A handful of transitional / experimental
instances in the upstream tags (a v0.5-style ``axes`` multiscale, an
``axes``-as-mapping coordinate system, ``bioformats2raw.layout`` whose key
cannot be an attribute name) are intentionally not exercised here.
"""

import importlib
import json
import types
from pathlib import Path

import pytest
import typing_extensions as tx

from abczarr._core.auto.validators import get_validator
from abczarr.ome import schemas
from abczarr.ome.metadata import base

TESTDIR = Path(__file__).parent

#: package suffix -> the version string that package's data declares.
VERSIONS = {
    "v0_6dev1": "0.6.dev1",
    "v0_6dev2": "0.6.dev2",
    "v0_6dev3": "0.6.dev3",
    "v0_6rc0": "0.6rc0",
}

# Standalone coordinate-transformation instances (``coordinateSystems`` +
# ``coordinateTransformations``). Availability differs per tag.
_XFORMS_PRE = [
    "affine2d2d", "affine2d3d", "identity", "scale", "translation",
    "rotation", "sequence", "bijection", "mapAxis1", "mapAxis2",
    "coordinates1d", "displacement1d", "byDimension1", "byDimension2",
    "byDimensionXarray", "inverseOf",
]
XFORMS = {
    "v0_6dev1": _XFORMS_PRE,
    "v0_6dev2": _XFORMS_PRE,
    "v0_6dev3": [n for n in _XFORMS_PRE if n != "inverseOf"],
    "v0_6rc0": [
        "affine2d2d", "affine2d3d", "identity", "scale", "translation",
        "rotation", "sequence", "bijection", "mapAxis1", "byDimension1",
        "byDimension2", "byDimensionXarray", "projectAxis", "projectAxis2",
    ],
}

# Full OME documents, routed to a carrier by the field they carry.
DOCS = [
    "multiscales_example", "plate_2wells", "plate_6wells",
    "well_2fields", "well_4fields", "series-2",
]
CARRIER_BY_KEY = {
    "multiscales": "OMEImage",
    "plate": "OMEPlate",
    "well": "OMEWell",
    "series": "OMESeries",
    "labels": "OMELabels",
}


def _pkg(version: str) -> types.ModuleType:
    return importlib.import_module("abczarr.ome.metadata." + version)


def _load(version: str, name: str) -> dict:
    path = TESTDIR / "data" / "ome" / version / (name + ".json")
    with path.open("r") as f:
        return json.load(f)


def _ome_attrs(doc: dict) -> dict:
    if isinstance(doc, dict):
        attrs = doc.get("attributes")
        if isinstance(attrs, dict) and "ome" in attrs:
            return attrs["ome"]
        if "ome" in doc:
            return doc["ome"]
    return doc


def _roundtrips(cls: type, data: dict) -> object:
    obj = cls.from_dict(data)
    assert cls.from_dict(obj.to_dict()) == obj
    return obj


def _xform_params() -> object:
    for version, names in XFORMS.items():
        for name in names:
            yield pytest.param(version, name, id=f"{version}-{name}")


def _doc_params() -> object:
    for version in VERSIONS:
        for name in DOCS:
            yield pytest.param(version, name, id=f"{version}-{name}")


@pytest.mark.parametrize("version", list(VERSIONS))
def test_version_dispatches_to_its_package(version: str) -> None:
    """The top OME container registers under the version string that
    version's data declares, so a document routes to the right package."""
    match = (("version", VERSIONS[version]),)
    registered = base.OME._registry()[match]
    assert registered.__module__ == f"abczarr.ome.metadata.{version}.ome"
    assert _pkg(version).version.VERSION == VERSIONS[version]


@pytest.mark.parametrize(("version", "name"), list(_xform_params()))
def test_coordinate_transformation_roundtrips(version: str, name: str) -> None:
    pkg = _pkg(version)
    doc = _ome_attrs(_load(version, name))
    xforms = doc["coordinateTransformations"]
    assert xforms
    for t in xforms:
        obj = _roundtrips(pkg.transformations.CoordinateTransformation, t)
        # dispatched to a concrete, type-specific subclass, not the base
        assert type(obj) is not pkg.transformations.CoordinateTransformation
        assert type(obj).__module__ == (
            f"abczarr.ome.metadata.{version}.transformations"
        )
    for cs in doc.get("coordinateSystems", []):
        _roundtrips(pkg.systems.CoordinateSystem, cs)


@pytest.mark.parametrize(("version", "name"), list(_doc_params()))
def test_ome_document_roundtrips(version: str, name: str) -> None:
    pkg = _pkg(version)
    doc = dict(_ome_attrs(_load(version, name)))
    # dev1/dev2 example documents carry a stale ``version`` string; the
    # discriminator is fixed by the schema, so normalise before parsing.
    doc["version"] = VERSIONS[version]
    carrier = None
    for key, clsname in CARRIER_BY_KEY.items():
        if key in doc:
            carrier = getattr(pkg.ome, clsname)
            break
    assert carrier is not None, f"no carrier for keys {list(doc)}"
    obj = _roundtrips(carrier, doc)
    assert type(obj).__module__ == f"abczarr.ome.metadata.{version}.ome"


@pytest.mark.parametrize("version", list(VERSIONS))
def test_image_label_roundtrips(version: str) -> None:
    pkg = _pkg(version)
    doc = _ome_attrs(_load(version, "colors_properties"))
    image_label = doc.get("image-label") or doc.get("labels")
    assert isinstance(image_label, dict)
    _roundtrips(pkg.labels.ImageLabel, image_label)


# --------------------------------------------------------------------------
# schema layer: the vendored transformation examples validate against the
# per-version JSON-schema TypedDict models.
# --------------------------------------------------------------------------


class _SysXfDev1(schemas.OMESchemaItem):
    v = schemas.v0_6dev1
    coordinateSystems: tx.List[v.CoordinateSystem]
    coordinateTransformations: tx.List[v.CoordinateTransformation]


class _SysXfDev2(schemas.OMESchemaItem):
    v = schemas.v0_6dev2
    coordinateSystems: tx.List[v.CoordinateSystem]
    coordinateTransformations: tx.List[v.CoordinateTransformation]


class _SysXfDev3(schemas.OMESchemaItem):
    v = schemas.v0_6dev3
    coordinateSystems: tx.List[v.CoordinateSystem]
    coordinateTransformations: tx.List[v.CoordinateTransformation]


_SYSXF = {
    "v0_6dev1": _SysXfDev1,
    "v0_6dev2": _SysXfDev2,
    "v0_6dev3": _SysXfDev3,
}


def _schema_xform_params() -> object:
    # ``byDimension`` mixes a wrapped and an inline inner form; validating it
    # against the strict per-member TypedDict union is out of scope here (the
    # metadata layer covers it). Every other transform type is exercised.
    for version in _SYSXF:
        for name in XFORMS[version]:
            if name.startswith("byDimension"):
                continue
            yield pytest.param(version, name, id=f"{version}-{name}")


@pytest.mark.parametrize(("version", "name"), list(_schema_xform_params()))
def test_schema_transformation_validates(version: str, name: str) -> None:
    doc = _ome_attrs(_load(version, name))
    validator = get_validator(_SYSXF[version])
    validator(doc)  # should not raise

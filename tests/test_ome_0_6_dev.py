"""OME-Zarr NGFF 0.6 pre-release metadata parses and round-trips.

Each 0.6 development version (``0.6.dev1`` .. ``0.6.dev4`` and ``0.6rc0``)
ships realistic example instances in the ``ngff-spec`` repository. These
tests parse those instances through the matching ``abczarr`` metadata model
and assert that the object serializes back to an equal object -- the proof
that data written against any 0.6 pre-release is read correctly.

The example instances are vendored under ``tests/data/ome/<version>/`` with
their JSONC comments stripped. A ``bioformats2raw.layout`` document -- whose
key is not a Python identifier -- is now read through the field's JSON-key
alias and is exercised alongside the rest. A couple of transitional /
experimental instances in the upstream tags (a v0.5-style ``axes`` multiscale,
an ``axes``-as-mapping coordinate system) are intentionally not exercised here.
"""

import importlib
import json
import types
from pathlib import Path

import pytest
import typing_extensions as tx

from abczarr.ome import base

TESTDIR = Path(__file__).parent

#: package suffix -> the version string that package's data declares.
VERSIONS = {
    "v0_6dev1": "0.6.dev1",
    "v0_6dev2": "0.6.dev2",
    "v0_6dev3": "0.6.dev3",
    "v0_6dev4": "0.6.dev4",
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
    # dev4 completes the input/output string->object overhaul: a transform's
    # `input`/`output` is a coordinate-system object, not a name string. Its
    # `mapAxis` examples still carry the transitional string form, so -- like
    # the other transitional instances -- they are not exercised here.
    "v0_6dev4": [
        "affine2d2d", "affine2d3d", "identity", "scale", "translation",
        "rotation", "sequence", "bijection", "byDimension1", "byDimension2",
        "byDimensionXarray", "coordinates1d", "displacement1d",
    ],
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
    return importlib.import_module("abczarr.ome." + version)


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
    obj = cls.from_json(data)
    assert cls.from_json(obj.to_json()) == obj
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
    assert registered.__module__ == f"abczarr.ome.{version}.ome"
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
            f"abczarr.ome.{version}.transformations"
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
    assert type(obj).__module__ == f"abczarr.ome.{version}.ome"


@pytest.mark.parametrize("version", list(VERSIONS))
def test_image_label_roundtrips(version: str) -> None:
    pkg = _pkg(version)
    doc = _ome_attrs(_load(version, "colors_properties"))
    image_label = doc.get("image-label") or doc.get("labels")
    assert isinstance(image_label, dict)
    obj = _roundtrips(pkg.labels.ImageLabel, image_label)
    # the ``label-value`` spec key reaches the typed field and serializes back
    # under the same key -- no ``label_value`` underscore twin.
    out = obj.to_json()
    for color in out.get("colors") or []:
        assert "label_value" not in color
        assert "label-value" in color


# A whole OME document, its carrier discriminator -> the carrier class name it
# must dispatch to when handed to the version-independent top-level entry
# point. ``scene`` arrives in 0.6.dev3.
_TOP_LEVEL_DOCS = [
    ("multiscales_example", "OMEImage"),
    ("plate_2wells", "OMEPlate"),
    ("well_2fields", "OMEWell"),
    ("series-2", "OMESeries"),
]
_SCENE_VERSIONS = {"v0_6dev3", "v0_6dev4", "v0_6rc0"}


def _top_level_params() -> object:
    for version in VERSIONS:
        for name, carrier in _TOP_LEVEL_DOCS:
            yield pytest.param(
                version, name, carrier, id=f"{version}-{carrier}"
            )
        if version in _SCENE_VERSIONS:
            yield pytest.param(
                version, "scene_registration", "OMEScene",
                id=f"{version}-OMEScene",
            )


@pytest.mark.parametrize(
    ("version", "name", "carrier"), list(_top_level_params())
)
def test_top_level_ome_dispatches_to_carrier(
    version: str, name: str, carrier: str
) -> None:
    """A whole OME document handed to the version-independent
    ``base.OME.from_json`` routes to the carrier its discriminator key names,
    and the result round-trips through ``to_json``."""
    pkg = _pkg(version)
    doc = dict(_ome_attrs(_load(version, name)))
    doc["version"] = VERSIONS[version]
    obj = base.OME.from_json(doc)
    assert type(obj) is getattr(pkg.ome, carrier)
    assert base.OME.from_json(obj.to_json()) == obj


@pytest.mark.parametrize("version", list(VERSIONS))
def test_bioformats2raw_layout_document_roundtrips(version: str) -> None:
    """A ``bioformats2raw.layout`` document (spec key with a dot) dispatches to
    the bf2raw carrier, populates the typed field, and serializes back under
    the spec key alone -- no ``bioformats2raw_layout`` twin in ``extra_items``
    or in the output."""
    pkg = _pkg(version)
    doc = dict(_ome_attrs(_load(version, "plate")))
    doc["version"] = VERSIONS[version]
    obj = base.OME.from_json(doc)
    assert type(obj) is pkg.ome.OMEBioformats2Raw
    assert obj.bioformats2raw_layout == 3
    assert not obj.extra_items
    out = obj.to_json()
    assert out["bioformats2raw.layout"] == 3
    assert "bioformats2raw_layout" not in out
    assert base.OME.from_json(out) == obj


# --------------------------------------------------------------------------
# schema layer: the vendored transformation examples validate against the
# official NGFF JSON schemas (via abczarr.ome.schemas).
#
# The official schemas are stricter than the round-trip metadata model above:
# a few upstream dev-tag example instances use transitional forms their own
# published schema rejects (e.g. dev1 `rotation`, dev2 `mapAxis`, and every
# dev3 instance -- whose coordinate-system axes are written `{"name": ...}`
# with no `type`, which the 0.6.dev3 axes schema does not accept). Those are
# covered by the round-trip tests above; only instances that conform to their
# own version's schema are asserted here.
# --------------------------------------------------------------------------

_SCHEMA_XFORMS = {
    "v0_6dev1": [
        "affine2d2d", "affine2d3d", "identity", "scale", "translation",
        "sequence", "bijection", "mapAxis1", "mapAxis2", "coordinates1d",
        "displacement1d", "byDimension1", "inverseOf",
    ],
    "v0_6dev2": [
        "affine2d2d", "affine2d3d", "identity", "scale", "translation",
        "rotation", "sequence", "bijection", "coordinates1d",
        "displacement1d", "byDimension1", "inverseOf",
    ],
}


def _schema_xform_params() -> object:
    for version, names in _SCHEMA_XFORMS.items():
        for name in names:
            yield pytest.param(version, name, id=f"{version}-{name}")


@pytest.mark.parametrize(("version", "name"), list(_schema_xform_params()))
def test_schema_transformation_validates(
    version: str,
    name: str,
    validate_systems_and_transforms: "tx.Callable[[dict, str], None]",
) -> None:
    doc = _ome_attrs(_load(version, name))
    # version's official string (e.g. "0.6.dev1") drives schema selection.
    validate_systems_and_transforms(doc, VERSIONS[version])


# --------------------------------------------------------------------------
# An unset optional ``omero`` is OMITTED, not emitted as ``"omero": null``.
# The NGFF ``image`` schema types ``omero`` as an object, so a serialized
# ``"omero": null`` fails validation ("omero must be object"); an absent
# ``omero`` conforms. Regression for issue #116: the field was declared with
# the typing ``tx.Optional`` (a bare ``Union[Omero, None]``) instead of the
# RFC-2119 ``Recommended`` marker, so it defaulted to ``None`` and
# round-tripped invalid metadata. ``Recommended`` makes an unset ``omero``
# the MISSING sentinel, which ``to_json`` drops.
# --------------------------------------------------------------------------

# A minimal ``omero`` conforming to the 0.6 ``image`` schema (``channels``
# required, each channel's ``window`` carrying start/min/end/max).
_OMERO = {
    "channels": [
        {
            "color": "FF0000",
            "window": {"start": 0.0, "min": 0.0, "end": 255.0, "max": 255.0},
        }
    ]
}


@pytest.mark.parametrize("version", list(VERSIONS))
def test_unset_omero_is_omitted_and_validates(version: str) -> None:
    from abczarr.ome import schemas

    pkg = _pkg(version)
    doc = dict(_ome_attrs(_load(version, "multiscales_example")))
    doc["version"] = VERSIONS[version]

    obj = base.OME.from_json(doc)
    assert type(obj) is pkg.ome.OMEImage
    out = obj.to_json()
    # no ``omero`` key at all -- not ``"omero": null``
    assert "omero" not in out
    # and the serialized image conforms to the official schema
    schemas.validate({"ome": out}, VERSIONS[version], "image")


@pytest.mark.parametrize("version", list(VERSIONS))
def test_present_omero_round_trips(version: str) -> None:
    from abczarr.ome import schemas

    pkg = _pkg(version)
    doc = dict(_ome_attrs(_load(version, "multiscales_example")))
    doc["version"] = VERSIONS[version]
    doc["omero"] = _OMERO

    obj = base.OME.from_json(doc)
    assert type(obj) is pkg.ome.OMEImage
    out = obj.to_json()
    # a present ``omero`` is preserved on the way out and still round-trips
    assert out.get("omero") == _OMERO
    assert base.OME.from_json(out) == obj
    schemas.validate({"ome": out}, VERSIONS[version], "image")

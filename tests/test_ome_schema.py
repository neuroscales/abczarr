"""OME-NGFF example documents validate against the official schemas.

These were previously checked against hand-written TypedDict schemas; they
now validate against the vendored official NGFF schemas through
``abczarr.ome.schemas`` (see ``test_ome_json_schema.py`` for the surface).

The ``0.6rc0`` corpus is validated below (``test_06rc0_*``). The conforming
whole-document instances of the ``0.6.dev1`` .. ``0.6.dev4`` pre-release
corpus are validated by ``test_ome_document_validates``; the transitional
instances that do not conform to their own tag's schema are listed -- with a
reason apiece -- in ``_NONCONFORMING`` and asserted to be genuinely rejected
by ``test_known_nonconforming_document_is_rejected``. All of these files are
upstream-verbatim vendored examples (see ``tests/data/ome/README.md``); none
is rewritten to make a test pass.
"""

import json
from pathlib import Path

import pytest
import typing_extensions as tx

from abczarr._errors import SchemaValidationError
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


# --------------------------------------------------------------------------
# The pre-release corpus (0.6.dev1 .. 0.6.dev4): whole OME documents held to
# their own tag's official ``ome_zarr`` schema, mirroring ``test_06rc0_ome``.
#
# Only the CONFORMING instances are asserted to validate; the transitional
# instances that predate a later cleanup are excluded by ``_NONCONFORMING``,
# each with the reason it fails. Every file here is an upstream-verbatim
# vendored example -- see ``tests/data/ome/README.md`` for provenance and the
# two non-conformance groups. The abczarr models round-trip all of them
# (``test_ome_0_6_dev.py``); this is the stricter official-schema check.
# --------------------------------------------------------------------------

#: ``abczarr`` version dir -> the official NGFF string its schema carries.
_SCHEMA_VERSIONS = {
    "v0_6dev1": "0.6.dev1",
    "v0_6dev2": "0.6.dev2",
    "v0_6dev3": "0.6.dev3",
    "v0_6dev4": "0.6.dev4",
}

#: The whole-document instance names that make up the OME-document corpus
#: (the standalone coordinate systems + transformations instances are checked
#: by ``test_ome_0_6_dev.py`` instead). Not every name exists in every dir;
#: missing files are skipped at collection.
_OME_DOCUMENTS = (
    "colors_properties",
    "image",
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
)

#: ``(version dir, document)`` -> why that vendored instance does NOT conform
#: to its own tag's official ``ome_zarr`` schema. Every entry is an
#: upstream-verbatim transitional example, kept byte-for-byte; see
#: ``tests/data/ome/README.md``. Two groups:
#:
#: * dev1 (all documents) and dev2 (all but ``multiscales_transformations``)
#:   carry a stale ``version`` string from an earlier draft ("0.5",
#:   "0.5-dev", "0.6dev2"); the schema pins ``version``, so they fail on that
#:   field alone. The models normalise it before parsing.
#: * ``multiscales_example_relative`` (dev1-dev4), ``scene_stitching``
#:   (dev3/dev4) and ``multiscales_reference_to_label`` (dev4) use a
#:   coordinate-systems shape a later tag replaced (axes as a mapping rather
#:   than a list; a transform ``output`` naming a system by string rather
#:   than an object). 0.6rc0 uses the final forms and conforms.
_NONCONFORMING = {
    # -- Group 1: stale ``version`` string ---------------------------------
    ("v0_6dev1", "colors_properties"): "stale version string (0.5)",
    ("v0_6dev1", "image"): "stale version string (0.5)",
    ("v0_6dev1", "multiscales_example"): "stale version string (0.6dev2)",
    ("v0_6dev1", "multiscales_transformations"): "stale version (0.5)",
    ("v0_6dev1", "plate"): "stale version string (0.5)",
    ("v0_6dev1", "plate_2wells"): "stale version string (0.5)",
    ("v0_6dev1", "plate_6wells"): "stale version string (0.5)",
    ("v0_6dev1", "series-2"): "stale version string (0.5)",
    ("v0_6dev1", "well_2fields"): "stale version string (0.5)",
    ("v0_6dev1", "well_4fields"): "stale version string (0.5)",
    ("v0_6dev2", "colors_properties"): "stale version string (0.5)",
    ("v0_6dev2", "image"): "stale version string (0.5)",
    ("v0_6dev2", "multiscales_example"): "stale version string (0.5)",
    ("v0_6dev2", "plate"): "stale version string (0.5)",
    ("v0_6dev2", "plate_2wells"): "stale version string (0.5)",
    ("v0_6dev2", "plate_6wells"): "stale version string (0.5)",
    ("v0_6dev2", "series-2"): "stale version string (0.5)",
    ("v0_6dev2", "well_2fields"): "stale version string (0.5)",
    ("v0_6dev2", "well_4fields"): "stale version string (0.5)",
    # -- Group 2: transitional structural forms ----------------------------
    ("v0_6dev1", "multiscales_example_relative"):
        "stale version + transitional axes-as-mapping form",
    ("v0_6dev2", "multiscales_example_relative"):
        "transitional axes-as-mapping coordinate system",
    ("v0_6dev3", "multiscales_example_relative"):
        "no top-level ome wrapper + axes-as-mapping coordinate system",
    ("v0_6dev4", "multiscales_example_relative"):
        "no top-level ome wrapper + axes-as-mapping coordinate system",
    ("v0_6dev3", "scene_stitching"):
        "transform output names a coordinate system by string, not object",
    ("v0_6dev4", "scene_stitching"):
        "transform output names a coordinate system by string, not object",
    ("v0_6dev4", "multiscales_reference_to_label"):
        "transitional axes-as-mapping coordinate system",
}


def _load_attributes(version_dir: str, name: str) -> dict:
    """Load a vendored instance and return the attributes to validate.

    Some vendored files are a whole ``zarr.json`` (a top-level
    ``attributes`` key); some are the attributes document itself. This
    mirrors the extraction in ``test_06rc0_ome``.
    """
    path = TESTDIR / "data" / "ome" / version_dir / (name + ".json")
    with path.open("r") as f:
        data = json.load(f)
    if "attributes" in data:
        data = data["attributes"]
    return data


def _document_params(conforming: bool) -> object:
    """Yield ``(version dir, official string, name)`` params.

    ``conforming=True`` yields the instances expected to validate;
    ``conforming=False`` yields the ``_NONCONFORMING`` instances. Only names
    whose file exists in the version dir are yielded.
    """
    for version_dir, official in _SCHEMA_VERSIONS.items():
        for name in _OME_DOCUMENTS:
            path = (
                TESTDIR / "data" / "ome" / version_dir / (name + ".json")
            )
            if not path.exists():
                continue
            excluded = (version_dir, name) in _NONCONFORMING
            if excluded == (not conforming):
                yield pytest.param(
                    version_dir, official, name,
                    id=f"{version_dir}-{name}",
                )


@pytest.mark.parametrize(
    ("version_dir", "official", "name"),
    list(_document_params(conforming=True)),
)
def test_ome_document_validates(
    version_dir: str, official: str, name: str
) -> None:
    """A conforming vendored OME document validates against its own tag's
    official ``ome_zarr`` schema."""
    data = _load_attributes(version_dir, name)
    schemas.validate(data, official, "ome_zarr")  # should not raise


@pytest.mark.parametrize(
    ("version_dir", "official", "name"),
    list(_document_params(conforming=False)),
)
def test_known_nonconforming_document_is_rejected(
    version_dir: str, official: str, name: str
) -> None:
    """A vendored instance listed in ``_NONCONFORMING`` is genuinely
    rejected by its own tag's official schema.

    This keeps the exclusion list honest: if an upstream instance is ever
    corrected (or re-vendored) so that it conforms, this test fails and the
    entry must move to the conforming set -- the non-conformance is never
    silently masked.
    """
    data = _load_attributes(version_dir, name)
    with pytest.raises(SchemaValidationError):
        schemas.validate(data, official, "ome_zarr")


def test_nonconforming_entries_all_exist() -> None:
    """Every ``_NONCONFORMING`` key names a file that is actually vendored,
    so the exclusion list cannot drift away from the corpus."""
    for version_dir, name in _NONCONFORMING:
        path = TESTDIR / "data" / "ome" / version_dir / (name + ".json")
        assert path.exists(), f"{version_dir}/{name}.json is missing"

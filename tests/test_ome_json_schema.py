"""The vendored NGFF JSON schemas compile and validate offline.

These guard the schema-based OME validation surface
(`abczarr.ome.schemas.get_validator` / `validate`), which compiles the
official schemas vendored under `ome/schemas/_ngff/` with fastjsonschema
and resolves every cross-file ``$ref`` locally -- no network.
"""

import json
from pathlib import Path

import pytest
import typing_extensions as tx

from abczarr.errors import SchemaValidationError
from abczarr.ome import schemas

TESTDIR = Path(__file__).parent
NGFF = Path(schemas.__file__).parent / "_ngff"

# draft-2020-12-only keywords fastjsonschema (2.x) does not implement. The
# vendored schemas must use none of them, or a validator would silently skip
# the constraint. This is asserted directly against the vendored files.
_UNSUPPORTED_2020 = {
    "prefixItems",
    "unevaluatedProperties",
    "unevaluatedItems",
    "$dynamicRef",
    "$dynamicAnchor",
}


def _iter_keys(node: object) -> tx.Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _iter_keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_keys(value)


def test_versions_match_vendored_dirs() -> None:
    on_disk = {p.name for p in NGFF.iterdir() if p.is_dir()}
    assert set(schemas.VERSIONS) == on_disk


@pytest.mark.parametrize("version", schemas.VERSIONS)
def test_every_document_compiles(version: str) -> None:
    docs = schemas.documents(version)
    assert docs  # every version ships at least image/label/...
    for document in docs:
        # compiling exercises the offline $ref handler across every file.
        assert callable(schemas.get_validator(version, document))


@pytest.mark.parametrize("version", schemas.VERSIONS)
def test_no_unsupported_2020_keywords(version: str) -> None:
    suffix = version
    offenders = set()
    for path in (NGFF / suffix).glob("*.schema"):
        doc = json.loads(path.read_text("utf-8"))
        offenders |= {k for k in _iter_keys(doc)} & _UNSUPPORTED_2020
    assert not offenders, (
        f"{version} uses draft-2020-12-only keywords fastjsonschema "
        f"ignores: {sorted(offenders)}"
    )


def test_version_spellings_share_one_validator() -> None:
    # the abczarr suffix and the official string resolve to the same compiled
    # validator (not two independent compiles).
    assert schemas.get_validator("v0_6rc0", "image") is schemas.get_validator(
        "0.6rc0", "image"
    )
    assert schemas.get_validator("v0_6dev1", "image") is schemas.get_validator(
        "0.6.dev1", "image"
    )


def test_unknown_version_and_document_raise() -> None:
    with pytest.raises(ValueError, match="unknown OME-NGFF version"):
        schemas.get_validator("0.9", "image")
    with pytest.raises(ValueError, match="no 'nope' schema"):
        schemas.get_validator("0.4", "nope")


def test_rc0_image_accepts_and_rejects() -> None:
    attrs = json.loads(
        (TESTDIR / "data" / "ome" / "v0_6rc0" / "multiscales_example.json")
        .read_text("utf-8")
    )["attributes"]
    # a real example validates against the official rc0 image schema...
    assert schemas.validate(attrs, "0.6rc0", "image") is attrs
    # ...and a malformed one is rejected with our error type.
    with pytest.raises(SchemaValidationError):
        schemas.validate({"ome": {"multiscales": "not-an-array"}},
                         "0.6rc0", "image")


def test_dev1_transformations_compile_despite_upstream_typo() -> None:
    # the dev1/dev2 coordinate-transformation schemas misplace a `required`
    # inside `properties`; the loader lifts it, so compilation succeeds.
    for version in ("0.6.dev1", "0.6.dev2"):
        assert callable(
            schemas.get_validator(version, "coordinate_transformation")
            if version == "0.6.dev1"
            else schemas.get_validator(version, "coordinate_transformations")
        )

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

# Keywords fastjsonschema (2.x) does not implement. A schema that uses one
# gets no enforcement of it -- the validator silently skips the constraint.
# The draft-2020-12 applicator/annotation keywords, plus the `contains`
# count bounds (`contains` itself is supported; its bounds are not).
_UNSUPPORTED = {
    "prefixItems",
    "unevaluatedProperties",
    "unevaluatedItems",
    "$dynamicRef",
    "$dynamicAnchor",
    "minContains",
    "maxContains",
}

# Of those, the ones the vendored OME schemas genuinely use -- so the guard
# tolerates them rather than pretend they are absent. Each is a documented,
# unenforced gap: `minContains`/`maxContains` express the "2-3 space axes"
# rule (image/axes/ome schemas, 0.4 on) and "at most one scale transform",
# neither of which fastjsonschema enforces. See
# test_axis_count_rule_is_a_known_gap and issue #95.
_KNOWN_UNENFORCED = {
    "minContains",
    "maxContains",
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


def _unsupported_in(version: str) -> tx.Set[str]:
    used = set()
    for path in (NGFF / version).glob("*.schema"):
        doc = json.loads(path.read_text("utf-8"))
        used |= set(_iter_keys(doc)) & _UNSUPPORTED
    return used


@pytest.mark.parametrize("version", schemas.VERSIONS)
def test_no_new_unsupported_keywords(version: str) -> None:
    # No vendored schema may use a keyword fastjsonschema ignores, except the
    # documented, already-tracked gaps in _KNOWN_UNENFORCED. A new one
    # creeping in would silently drop a constraint with no sign.
    new = _unsupported_in(version) - _KNOWN_UNENFORCED
    assert not new, (
        f"{version} uses keywords fastjsonschema ignores (constraint "
        f"silently dropped): {sorted(new)}"
    )


def test_known_unenforced_keywords_are_still_present() -> None:
    # Keep _KNOWN_UNENFORCED honest: if upstream ever drops these, tighten
    # the tolerance list (and revisit the gap) rather than keep tolerating a
    # keyword no vendored schema uses any more.
    used = set()
    for version in schemas.VERSIONS:
        used |= _unsupported_in(version)
    assert _KNOWN_UNENFORCED <= used, (
        "tolerated keywords no longer used by any vendored schema: "
        f"{sorted(_KNOWN_UNENFORCED - used)}"
    )


def _four_space_axis_image() -> dict:
    space = [{"name": n, "type": "space"} for n in ("x", "y", "z", "w")]
    return {"multiscales": [{
        "axes": space,
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]}]}],
    }]}


@pytest.mark.xfail(
    strict=True,
    reason="fastjsonschema 2.x ignores minContains/maxContains, so the OME "
           "0.4 axis-count rule (2-3 space axes) is unenforced (#95)",
)
def test_axis_count_rule_is_a_known_gap() -> None:
    # The 0.4 image schema bounds the space axes with minContains 2 /
    # maxContains 3; fastjsonschema drops both, so four space axes wrongly
    # validate. This documents the residual gap -- an xpass here means the
    # constraint became enforced, so tighten the guard and drop the tolerance.
    with pytest.raises(SchemaValidationError):
        schemas.validate(_four_space_axis_image(), "0.4", "image")


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

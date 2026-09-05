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

# Of those, the ones the vendored OME schemas genuinely use. fastjsonschema
# ignores `minContains`/`maxContains` (the "2-3 space axes" rule, 0.4 on, and
# "at most one scale transform"), but abczarr's own `_contains` pass enforces
# them after the compiled validator runs -- so they are tolerated here because
# they are enforced elsewhere, not because they are a silent gap. See
# test_axis_count_rule_is_enforced and issue #95.
_ENFORCED_SEPARATELY = {
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
    # ones in _ENFORCED_SEPARATELY (which abczarr's _contains pass enforces).
    # A new one creeping in would silently drop a constraint with no sign.
    new = _unsupported_in(version) - _ENFORCED_SEPARATELY
    assert not new, (
        f"{version} uses keywords fastjsonschema ignores (constraint "
        f"silently dropped): {sorted(new)}"
    )


def test_separately_enforced_keywords_are_still_present() -> None:
    # Keep _ENFORCED_SEPARATELY honest: if upstream ever drops these, drop the
    # _contains pass and this tolerance rather than keep carrying a keyword no
    # vendored schema uses any more.
    used = set()
    for version in schemas.VERSIONS:
        used |= _unsupported_in(version)
    assert _ENFORCED_SEPARATELY <= used, (
        "separately-enforced keywords no longer used by any vendored schema: "
        f"{sorted(_ENFORCED_SEPARATELY - used)}"
    )


def _axes(*types: str) -> tx.List[dict]:
    # distinct names -> the schemas' uniqueItems on axes is satisfied, so a
    # count case fails on the count bound and nothing else.
    return [{"name": f"{t}{i}", "type": t} for i, t in enumerate(types)]


def _image_04(axes: tx.List[dict], n_transforms: int = 1) -> dict:
    scale = {"type": "scale", "scale": [1.0] * len(axes)}
    return {"multiscales": [{
        "version": "0.4",
        "axes": axes,
        "datasets": [{"path": "0",
                      "coordinateTransformations": [scale] * n_transforms}],
    }]}


def test_axis_count_rule_is_enforced() -> None:
    # The 0.4 image schema bounds the space axes with minContains 2 /
    # maxContains 3 (reached through `$ref #/$defs/axes`). fastjsonschema
    # ignores both; the _contains pass enforces them.
    assert schemas.validate(_image_04(_axes("space", "space")), "0.4", "image")
    assert schemas.validate(
        _image_04(_axes("space", "space", "space")), "0.4", "image"
    )
    for bad in (
        _axes("space"),                                    # too few (< 2)
        _axes("space", "space", "space", "space"),         # too many (> 3)
    ):
        with pytest.raises(SchemaValidationError):
            schemas.validate(_image_04(bad), "0.4", "image")


def test_scale_transform_count_is_enforced() -> None:
    # The 0.4 image schema allows at most one scale transform per dataset
    # (maxContains 1 on `$defs/coordinateTransformations`).
    axes = _axes("space", "space")
    assert schemas.validate(_image_04(axes, n_transforms=1), "0.4", "image")
    with pytest.raises(SchemaValidationError):
        schemas.validate(_image_04(axes, n_transforms=2), "0.4", "image")


def test_oneof_axis_count_rule_is_enforced() -> None:
    # The 0.6 axes schema expresses the count rule as a `oneOf` of two bare
    # count-constraint branches (2-3 space axes XOR >=2 array axes), which the
    # branch a document satisfies is decided *by*. The _contains pass resolves
    # the branch selection the way a spec-complete validator does.
    assert schemas.validate(_axes("space", "space"), "0.6rc0", "axes")
    assert schemas.validate(_axes("space", "space", "space"), "0.6rc0", "axes")
    assert schemas.validate(_axes("array", "array"), "0.6rc0", "axes")
    assert schemas.validate(
        _axes("space", "space", "time"), "0.6rc0", "axes"
    )
    for bad in (
        _axes("space"),                             # < 2 space, no array
        _axes("space", "space", "space", "space"),  # > 3 space
        _axes("space", "space", "array", "array"),  # matches both branches
    ):
        with pytest.raises(SchemaValidationError):
            schemas.validate(bad, "0.6rc0", "axes")


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

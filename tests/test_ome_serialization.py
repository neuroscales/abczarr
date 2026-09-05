"""OME metadata serializes to real JSON, with unset fields omitted.

Regression tests for two defects where the round-trip assertion
``from_json(obj.to_json()) == obj`` held while ``obj.to_json()`` was in
fact not JSON-serializable and/or carried invented values:

* an unset ``Recommended``/``Optional`` field was emitted as the ``MISSING``
  sentinel (``json.dumps`` then raised);
* a ``Literal``-typed ``Recommended``/``Optional`` field silently defaulted
  to the first literal (data was invented).
"""

import json
from pathlib import Path

import pytest

from abczarr._core.rfc2119 import MISSING
from abczarr.ome import base

TESTDIR = Path(__file__).parent

# 0.6-line versions ship a vendored example corpus; each of these documents
# carries a carrier discriminator key, so base.OME dispatches it.
VERSIONS = {
    "v0_6dev1": "0.6.dev1",
    "v0_6dev2": "0.6.dev2",
    "v0_6dev3": "0.6.dev3",
    "v0_6dev4": "0.6.dev4",
    "v0_6rc0": "0.6rc0",
}
DOCS = [
    "multiscales_example", "plate_2wells", "plate_6wells",
    "well_2fields", "well_4fields", "series-2",
]


def _ome_attrs(doc: dict) -> dict:
    if isinstance(doc, dict):
        attrs = doc.get("attributes")
        if isinstance(attrs, dict) and "ome" in attrs:
            return attrs["ome"]
        if "ome" in doc:
            return doc["ome"]
    return doc


def _load(version: str, name: str) -> dict:
    return json.loads(
        (TESTDIR / "data" / "ome" / version / (name + ".json")).read_text()
    )


def _has_missing(obj: object) -> bool:
    if obj is MISSING:
        return True
    if isinstance(obj, dict):
        return any(_has_missing(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_missing(v) for v in obj)
    return False


def _doc_params() -> object:
    for version in VERSIONS:
        for name in DOCS:
            path = TESTDIR / "data" / "ome" / version / (name + ".json")
            if path.exists():
                yield pytest.param(version, name, id=f"{version}-{name}")


@pytest.mark.parametrize(("version", "name"), list(_doc_params()))
def test_ome_document_to_json_is_serializable(version: str, name: str) -> None:
    doc = dict(_ome_attrs(_load(version, name)))
    doc["version"] = VERSIONS[version]
    out = base.OME.from_json(doc).to_json()
    # the whole point: this must not raise (it did while MISSING leaked out)
    json.dumps({"ome": out})
    assert not _has_missing(out), "MISSING sentinel leaked into to_json output"


def test_unset_recommended_literal_is_omitted_not_invented() -> None:
    """A Literal-typed Recommended/Optional field is absent when unset -- it
    must not be filled with the first literal value."""
    from abczarr.ome import v0_4

    # `type` and `unit` are both omitted rather than invented as
    # "space"/"angstrom"; a time axis in particular must not gain a unit.
    assert v0_4.Axis.from_json({"name": "t", "type": "time"}).to_json() == {
        "name": "t", "type": "time",
    }
    assert v0_4.Axis.from_json({"name": "q"}).to_json() == {"name": "q"}


def test_unset_recommended_object_is_omitted() -> None:
    """A whole optional sub-object left unset is absent from the output, not
    emitted as MISSING (nor exploded)."""
    from abczarr.ome import v0_4

    m = v0_4.Multiscale.from_json({
        "axes": [{"name": "y", "type": "space"},
                 {"name": "x", "type": "space"}],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1, 1]}]}],
    })
    out = m.to_json()
    assert set(out) == {"axes", "datasets", "version"}
    assert not _has_missing(out)
    json.dumps(out)

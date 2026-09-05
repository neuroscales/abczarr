"""abczarr's Zarr metadata schemas compile and validate offline.

These guard `abczarr.schemas.get_validator` / `validate`: the authored
v1/v2/v3-core array/group schemas compile with fastjsonschema, the v3
array composes the vendored official extension schemas (codecs, data
types) with no network, and a re-vendored registry cannot silently drop
an extension from the valid set.
"""

import json
from pathlib import Path

import pytest
import typing_extensions as tx

from abczarr import schemas
from abczarr.errors import SchemaValidationError

ZARR = Path(schemas.__file__).parent / "_zarr"
EXT = ZARR / "v3" / "extensions"
RAW = (
    "https://raw.githubusercontent.com/zarr-developers/"
    "zarr-extensions/refs/heads/main/"
)


def _refs(node: object) -> tx.Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                yield value
            else:
                yield from _refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _refs(value)


def test_versions_and_documents() -> None:
    assert schemas.VERSIONS == ("v1", "v2", "v3")
    for version in schemas.VERSIONS:
        assert schemas.documents(version) == ("array", "group")


@pytest.mark.parametrize("version", schemas.VERSIONS)
@pytest.mark.parametrize("document", ["array", "group"])
def test_every_schema_compiles(version: str, document: str) -> None:
    # compiling the v3 array exercises the offline handler over every
    # vendored codec/data-type schema it composes.
    assert callable(schemas.get_validator(version, document))


def test_version_spellings_share_one_validator() -> None:
    assert schemas.get_validator("v3", "array") is schemas.get_validator(
        "3", "array"
    )


def test_unknown_version_and_document_raise() -> None:
    with pytest.raises(ValueError, match="unknown Zarr version"):
        schemas.get_validator("v9", "array")
    with pytest.raises(ValueError, match="no 'nope' schema"):
        schemas.get_validator("v3", "nope")


def test_v3_array_composes_core_and_vendored_codecs() -> None:
    doc = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10, 10],
        "data_type": "float32",
        "chunk_grid": {"name": "regular",
                       "configuration": {"chunk_shape": [5, 5]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0.0,
        # bytes/zstd are vendored extension schemas; blosc is authored core.
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "blosc", "configuration": {"cname": "zstd"}},
            {"name": "zstd", "configuration": {"level": 3}},
        ],
    }
    assert schemas.validate(doc, "v3", "array") is doc
    # a bogus codec name is not in the composed valid set.
    doc["codecs"] = [{"name": "not_a_codec"}]
    with pytest.raises(SchemaValidationError):
        schemas.validate(doc, "v3", "array")


def test_v2_array_accepts_and_rejects() -> None:
    good = {"zarr_format": 2, "shape": [10], "chunks": [5], "dtype": "<f8",
            "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5},
            "fill_value": 0, "order": "C", "filters": None}
    assert schemas.validate(good, "v2", "array") is good
    with pytest.raises(SchemaValidationError):
        schemas.validate({"zarr_format": 2}, "v2", "array")  # missing fields


def test_v2_array_accepts_object_and_vlen_codecs() -> None:
    # zarr-python string/object arrays: dtype "|O" with a vlen-* filter and a
    # numcodecs compressor the schema does not describe id-by-id (snappy).
    good = {"zarr_format": 2, "shape": [3], "chunks": [3], "dtype": "|O",
            "compressor": {"id": "snappy"},
            "fill_value": None, "order": "C",
            "filters": [{"id": "vlen-utf8"}]}
    assert schemas.validate(good, "v2", "array") is good
    # a filter/compressor entry is still an object with an "id".
    bad = dict(good, filters=[{"no_id": "vlen-utf8"}])
    with pytest.raises(SchemaValidationError):
        schemas.validate(bad, "v2", "array")
    # and "|O" does not open the dtype to arbitrary strings.
    with pytest.raises(SchemaValidationError):
        schemas.validate(dict(good, dtype="not-a-dtype"), "v2", "array")


def test_v3_array_rectilinear_chunk_grid_is_typed() -> None:
    # the v3 array composes the vendored rectilinear chunk-grid schema, whose
    # chunk_shapes is an array of arrays of integers -- so a malformed
    # chunk_shapes is rejected, not waved through by a loose {"type": "array"}
    # stub. This also exercises the prefixItems normalization in
    # _validation.py (the vendored schema's inner [start, length] pairs).
    base = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10, 10],
        "data_type": "float32",
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0.0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }

    def with_chunk_shapes(chunk_shapes: object) -> dict:
        doc = dict(base)
        doc["chunk_grid"] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": chunk_shapes},
        }
        return doc

    # well-formed: an array of arrays whose entries are integers or
    # [start, length] integer pairs.
    good = with_chunk_shapes([[[2, 3], [4, 1]], [5, 5]])
    assert schemas.validate(good, "v3", "array") is good

    # malformed: a bare string where a chunk-edge-length array belongs. Under
    # the old loose stub this validated; it must now be rejected.
    with pytest.raises(SchemaValidationError):
        schemas.validate(with_chunk_shapes([[[2, 3], [4, 1]], "bad"]), "v3",
                         "array")

    # a 3-element inner tuple violates the prefixItems (max two) constraint,
    # proving the normalization is enforced rather than silently ignored.
    with pytest.raises(SchemaValidationError):
        schemas.validate(with_chunk_shapes([[[2, 3, 9]]]), "v3", "array")


def test_normalized_extensions_drop_unknown_uint_format() -> None:
    # older fastjsonschema (the pinned floor) rejects the vendored rectilinear
    # schema's custom `"format": "uint"` at compile time. The loader drops it
    # in-memory, so no normalized extension keeps a `format: "uint"` that would
    # break compilation. Version-independent: it guards the normalization, not
    # a particular fastjsonschema's tolerance.
    from abczarr.schemas import _validation

    def _formats(node: object) -> tx.Iterator[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "format" and isinstance(value, str):
                    yield value
                else:
                    yield from _formats(value)
        elif isinstance(node, list):
            for value in node:
                yield from _formats(value)

    registry = _validation._extension_registry()
    assert "uint" not in {f for s in registry.values() for f in _formats(s)}


def test_v3_array_references_every_vendored_codec_and_dtype() -> None:
    # maintainability net: if the registry is re-vendored with a new codec or
    # data type, the composed v3 array schema must reference it (regenerate
    # via tools once the generator lands), or the valid set silently lags.
    array = json.loads((ZARR / "v3" / "core" / "array.schema").read_text())
    referenced = {r for r in _refs(array) if r.startswith(RAW)}
    for category in ("codecs", "data-types"):
        vendored = {
            RAW + p.relative_to(EXT).as_posix()
            for p in (EXT / category).glob("*/schema.json")
        }
        missing = vendored - referenced
        assert not missing, f"{category} not composed into v3 array: {missing}"

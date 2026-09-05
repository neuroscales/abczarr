"""Generate abczarr's authored Zarr v1/v2/v3-core JSON schemas.

Run once to (re)emit the .schema files under schemas/_zarr/{v1,v2,v3/core}/.
The v3 array schema composes the vendored extension codec/data-type schemas
by globbing, so re-vendoring the registry keeps the valid set in sync.
"""
import json
import pathlib

_REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = _REPO / "src" / "abczarr" / "schemas" / "_zarr"
EXT = ROOT / "v3" / "extensions"
RAW = ("https://raw.githubusercontent.com/zarr-developers/"
       "zarr-extensions/refs/heads/main/")
DRAFT = "https://json-schema.org/draft/2020-12/schema"

# ---- shared value shapes (faithful to abczarr._core TypedDicts) -------------

DTYPE_V3_BUILTIN = [
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "complex64", "complex128",
]
# _core.dtypes.RegexDataTypeV2, plus "|O" for object arrays (zarr-python
# writes dtype "|O" for string/object arrays alongside a vlen-* filter).
DTYPE_V2_PATTERN = (
    r"^(?:\|b1|\|O|[<>|][iu][1248]|[<>][f][248]|[<>][c][816]"
    r"|[<>|][mM][1248]"
    r"(?:\[(?:h|m|s|ms|us|μs|ns|ps|fs|as|Y|M|W|D"
    r"|nat|naT|nAt|nAT|Nat|NaT|NAt|NAT)\])?"
    r"|[<>|][SUV]\d+)$"
)
INT = {"type": "integer"}
NONNEG_INT = {"type": "integer", "minimum": 0}
# any numcodecs codec/filter identified by its "id" (open beyond the ids we
# describe field-by-field), so an unknown-but-valid numcodecs id still passes.
_OPEN_CODEC = {"type": "object", "required": ["id"]}
# a Zarr fill value: number / special-float string / bool / null / composite
FILL_VALUE = {"anyOf": [
    {"type": "number"},
    {"type": "string"},          # "NaN", "Infinity", "-Infinity", raw hex...
    {"type": "boolean"},
    {"type": "null"},
    {"type": "array"},           # complex [re, im] / structured
]}


def _dtype_v2() -> dict:
    # ScalarDataTypeV2 string, or a structured/nested descr (array form).
    return {"anyOf": [
        {"type": "string", "pattern": DTYPE_V2_PATTERN},
        {"type": "array"},
    ]}


def _levels(n: int) -> dict:
    return {"type": "integer", "minimum": 0, "maximum": n}


def write(path: pathlib.Path, schema: dict) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2) + "\n")
    return path


# ============================ Zarr v3 core ==================================

def v3_core_codecs() -> dict:
    """Inline $defs for the four core built-in codecs (blosc/gzip/crc32c/
    sharding_indexed). bytes/transpose and every other codec come from the
    vendored extension registry."""
    blosc = {
        "type": "object",
        "properties": {
            "name": {"const": "blosc"},
            "configuration": {"type": "object", "properties": {
                "cname": {"enum": ["blosclz", "lz4", "lz4hc",
                                   "snappy", "zlib", "zstd"]},
                "clevel": _levels(9),
                "shuffle": {"enum": ["noshuffle", "shuffle", "bitshuffle"]},
                "typesize": INT,
                "blocksize": INT,
            }},
        },
        "required": ["name"],
    }
    gzip = {
        "type": "object",
        "properties": {
            "name": {"const": "gzip"},
            "configuration": {"type": "object",
                              "properties": {"level": _levels(9)}},
        },
        "required": ["name"],
    }
    crc32c = {
        "type": "object",
        "properties": {"name": {"const": "crc32c"},
                       "configuration": {"type": "object"}},
        "required": ["name"],
    }
    sharding = {
        "type": "object",
        "properties": {
            "name": {"const": "sharding_indexed"},
            "configuration": {
                "type": "object",
                "properties": {
                    "chunk_shape": {"type": "array", "items": NONNEG_INT},
                    "codecs": {"type": "array",
                               "items": {"$ref": "#/$defs/codec"}},
                    "index_codecs": {"type": "array",
                                     "items": {"$ref": "#/$defs/codec"}},
                    "index_location": {"enum": ["start", "end"]},
                },
                "required": ["chunk_shape", "codecs", "index_codecs"],
            },
        },
        "required": ["name", "configuration"],
    }
    return {"blosc": blosc, "gzip": gzip, "crc32c": crc32c,
            "sharding_indexed": sharding}


def vendored_refs(category: str) -> list:
    # raw URL path mirrors the registry root: <category>/<name>/schema.json
    return [f"{RAW}{p.relative_to(EXT).as_posix()}"
            for p in sorted((EXT / category).glob("*/schema.json"))]


def v3_array() -> dict:
    core = v3_core_codecs()
    codec_alts = [{"$ref": f"#/$defs/{n}"} for n in core]
    codec_alts += [{"$ref": u} for u in vendored_refs("codecs")]
    dtype_alts = [
        {"enum": DTYPE_V3_BUILTIN},
        {"type": "string", "pattern": r"^r\d+$"},
    ] + [{"$ref": u} for u in vendored_refs("data-types")]

    defs = dict(core)
    defs["codec"] = {"anyOf": codec_alts}

    regular = {
        "type": "object",
        "properties": {
            "name": {"const": "regular"},
            "configuration": {"type": "object",
                              "properties": {"chunk_shape": {
                                  "type": "array", "items": NONNEG_INT}},
                              "required": ["chunk_shape"]},
        },
        "required": ["name", "configuration"],
    }
    rectilinear = {
        "type": "object",
        "properties": {
            "name": {"const": "rectilinear"},
            "configuration": {"type": "object", "properties": {
                "kind": {"const": "inline"},
                "chunk_shapes": {"type": "array"}}},
        },
        "required": ["name", "configuration"],
    }
    cke = {
        "type": "object",
        "properties": {
            "name": {"enum": ["default", "v2"]},
            "configuration": {"type": "object", "properties": {
                "separator": {"enum": [".", "/"]}}},
        },
        "required": ["name"],
    }
    return {
        "$schema": DRAFT,
        "title": "Zarr v3 array metadata",
        "description": ("abczarr-authored schema for a Zarr v3 array "
                        "zarr.json. Codecs and data types compose the "
                        "official vendored extension schemas."),
        "type": "object",
        "properties": {
            "zarr_format": {"const": 3},
            "node_type": {"const": "array"},
            "shape": {"type": "array", "items": NONNEG_INT},
            "data_type": {"anyOf": dtype_alts},
            "chunk_grid": {"oneOf": [
                {"$ref": "#/$defs/regular_chunk_grid"},
                {"$ref": "#/$defs/rectilinear_chunk_grid"}]},
            "chunk_key_encoding": {"$ref": "#/$defs/chunk_key_encoding"},
            "fill_value": FILL_VALUE,
            "codecs": {"type": "array", "items": {"$ref": "#/$defs/codec"}},
            "attributes": {"type": "object"},
            "storage_transformers": {"type": "array"},
            "dimension_names": {"type": "array",
                                "items": {"type": ["string", "null"]}},
        },
        "required": ["zarr_format", "node_type", "shape", "data_type",
                     "chunk_grid", "chunk_key_encoding", "fill_value",
                     "codecs"],
        "$defs": {**defs,
                  "regular_chunk_grid": regular,
                  "rectilinear_chunk_grid": rectilinear,
                  "chunk_key_encoding": cke},
    }


def v3_group() -> dict:
    return {
        "$schema": DRAFT,
        "title": "Zarr v3 group metadata",
        "type": "object",
        "properties": {
            "zarr_format": {"const": 3},
            "node_type": {"const": "group"},
            "attributes": {"type": "object"},
        },
        "required": ["zarr_format", "node_type"],
    }


# ============================ Zarr v2 =======================================

def _v2_numcodec(name: str, config_props: dict) -> dict:
    props = {"id": {"const": name}}
    props.update(config_props)
    return {"type": "object", "properties": props, "required": ["id"]}


def v2_codecs_defs() -> dict:
    return {
        "blosc": _v2_numcodec("blosc", {
            "cname": {"enum": ["blosclz", "lz4", "lz4hc",
                               "snappy", "zlib", "zstd"]},
            "clevel": _levels(9),
            "shuffle": {"enum": [-1, 0, 1, 2]},
            "blocksize": INT, "typesize": INT}),
        "bz2": _v2_numcodec("bz2", {"level": _levels(9)}),
        "gzip": _v2_numcodec("gzip", {"level": _levels(9)}),
        "lzma": _v2_numcodec("lzma", {
            "format": {"enum": [0, 1, 2, 3]},
            "check": {"enum": [0, 1, 4, 10, 15, 16]},
            "preset": _levels(9),
            "filters": {"type": "array", "items": {"type": "object"}}}),
        "lz4": _v2_numcodec("lz4", {"acceleration": INT}),
        "pcodec": _v2_numcodec("pcodec", {
            "level": _levels(12),
            "mode_spec": {"enum": ["auto", "classic"]},
            "delta_spec": {"enum": ["auto", "none",
                                    "try_consecutive", "try_lookback"]},
            "paging_spec": {"const": "equal_pages_up_to"},
            "delta_encoding_order": {"anyOf": [_levels(7), {"type": "null"}]},
            "equal_pages_up_to": INT}),
        "zfpy": _v2_numcodec("zfpy", {
            "mode": {"enum": [0, 1, 2, 3, 4]},
            "tolerance": {"type": "number"}, "rate": INT,
            "precision": INT,
            "compression_kwargs": {"type": "object"}}),
        "zlib": _v2_numcodec("zlib", {"level": _levels(9)}),
        "zstd": _v2_numcodec("zstd", {"level": INT}),
    }


def v2_filters_defs() -> dict:
    dt = _dtype_v2()
    return {
        "delta": {"type": "object", "properties": {
            "id": {"const": "delta"}, "dtype": dt, "astype": dt},
            "required": ["id", "dtype"]},
        "fixedscaleoffset": {"type": "object", "properties": {
            "id": {"const": "fixedscaleoffset"},
            "offset": {"type": "number"}, "scale": {"type": "number"},
            "dtype": dt, "astype": dt},
            "required": ["id", "offset", "scale", "dtype"]},
        "quantize": {"type": "object", "properties": {
            "id": {"const": "quantize"}, "digits": INT,
            "dtype": dt, "astype": dt},
            "required": ["id", "digits", "dtype"]},
        "bitround": {"type": "object", "properties": {
            "id": {"const": "bitround"}, "keepbits": INT},
            "required": ["id", "keepbits"]},
        "packbits": {"type": "object",
                     "properties": {"id": {"const": "packbits"}},
                     "required": ["id"]},
        "categorize": {"type": "object", "properties": {
            "id": {"const": "categorize"},
            "labels": {"type": "array", "items": {"type": "string"}},
            "dtype": dt, "astype": dt},
            "required": ["id", "labels", "dtype"]},
        "astype": {"type": "object", "properties": {
            "id": {"const": "astype"}, "encode_dtype": dt, "decode_dtype": dt},
            "required": ["id", "encode_dtype"]},
        "shuffle": {"type": "object", "properties": {
            "id": {"const": "shuffle"}, "elementsize": INT},
            "required": ["id"]},
    }


def v2_array() -> dict:
    codecs = v2_codecs_defs()
    filters = v2_filters_defs()
    return {
        "$schema": DRAFT,
        "title": "Zarr v2 array metadata (.zarray)",
        "type": "object",
        "properties": {
            "zarr_format": {"const": 2},
            "shape": {"type": "array", "items": NONNEG_INT},
            "chunks": {"type": "array", "items": NONNEG_INT},
            "dtype": _dtype_v2(),
            # the known-id $defs stay for the ids we describe in detail; the
            # open alternative lets any other numcodecs codec (vlen-utf8,
            # crc32, snappy, ...) validate on its "id" alone.
            "compressor": {"anyOf": [{"type": "null"}]
                           + [{"$ref": f"#/$defs/codec_{n}"} for n in codecs]
                           + [_OPEN_CODEC]},
            "fill_value": FILL_VALUE,
            "order": {"enum": ["C", "F"]},
            "filters": {"type": ["array", "null"], "items": {
                "anyOf": [{"$ref": f"#/$defs/filter_{n}"} for n in filters]
                         + [_OPEN_CODEC]}},
            "dimension_separator": {"enum": [".", "/"]},
        },
        "required": ["zarr_format", "shape", "chunks", "dtype",
                     "compressor", "fill_value", "order", "filters"],
        "$defs": {**{f"codec_{n}": s for n, s in codecs.items()},
                  **{f"filter_{n}": s for n, s in filters.items()}},
    }


def v2_group() -> dict:
    return {
        "$schema": DRAFT,
        "title": "Zarr v2 group metadata (.zgroup)",
        "type": "object",
        "properties": {"zarr_format": {"const": 2}},
        "required": ["zarr_format"],
    }


# ============================ Zarr v1 =======================================

def v1_array() -> dict:
    # v1 numcodecs (schemas.v1.codecs): compression is a name, options a union
    names = ["blosc", "bz2", "gzip", "lz4", "lzma",
             "pcodec", "zfpy", "zlib", "zstd"]
    return {
        "$schema": DRAFT,
        "title": "Zarr v1 array metadata",
        "type": "object",
        "properties": {
            "zarr_format": {"const": 1},
            "shape": {"type": "array", "items": NONNEG_INT},
            "chunks": {"type": "array", "items": NONNEG_INT},
            "dtype": _dtype_v2(),
            "compression": {"enum": names},
            "compression_opts": {"anyOf": [
                {"type": "integer"}, {"type": "string"},
                {"type": "object"}, {"type": "null"}]},
            "fill_value": FILL_VALUE,
            "order": {"enum": ["C", "F"]},
        },
        "required": ["zarr_format", "shape", "chunks", "dtype",
                     "compression", "compression_opts", "fill_value",
                     "order"],
    }


def v1_group() -> dict:
    return {
        "$schema": DRAFT,
        "title": "Zarr v1 group metadata",
        "type": "object",
        "properties": {"zarr_format": {"const": 1}},
        "required": ["zarr_format"],
    }


def main() -> None:
    outputs = {
        ROOT / "v3" / "core" / "array.schema": v3_array(),
        ROOT / "v3" / "core" / "group.schema": v3_group(),
        ROOT / "v2" / "array.schema": v2_array(),
        ROOT / "v2" / "group.schema": v2_group(),
        ROOT / "v1" / "array.schema": v1_array(),
        ROOT / "v1" / "group.schema": v1_group(),
    }
    for path, schema in outputs.items():
        write(path, schema)
        print("wrote", path)


if __name__ == "__main__":
    main()

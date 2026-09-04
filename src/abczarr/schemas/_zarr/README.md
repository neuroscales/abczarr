# Zarr JSON schemas

JSON schemas that `abczarr` validates Zarr metadata against, offline
(no network, no upstream package). Two provenances:

- **`v3/extensions/`** — the **official** Zarr v3 extension schemas, vendored
  verbatim from the Zarr extensions registry (see below).
- **`v1/`, `v2/`, `v3/core/`** — **authored by abczarr**, because the Zarr
  core specification (array/group metadata, the core built-in codecs) and the
  legacy v1/v2 formats publish no formal JSON Schema. These follow the
  normative spec text.

## Vendored: `v3/extensions/`

Copied verbatim from <https://github.com/zarr-developers/zarr-extensions> at
commit `4da7b37a84f76e660902f6d3de3eaef0e0febae6`, preserving the upstream `<category>/<name>/schema.json`
layout so that each schema's cross-references (absolute
`raw.githubusercontent.com/.../main/<category>/<name>/schema.json` URLs)
resolve to the vendored files. Covers every registered extension codec, data
type, chunk grid, chunk-key encoding and storage transformer.

### Known upstream defects (normalized at load time)

Two extension schemas use constructs an eager compiler cannot take verbatim.
The vendored bytes are untouched; the loader (`schemas/_validation.py`)
normalizes the in-memory copy:

- **`chunk-grids/rectilinear`** uses draft-2020-12 `prefixItems`, which
  fastjsonschema does not implement and would silently ignore. The loader
  rewrites `prefixItems` to the equivalent draft-07 tuple form (`items` as a
  list, plus `additionalItems` for any `items` "rest" schema), so the
  constraint is actually enforced.
- **`codecs/n5_default`** additionally writes `"type": "#/$defs/codec"` —
  a JSON-pointer where a type name belongs (a plain typo for `"$ref"`; it is
  meaningless as a `type` and every validator rejects it). The loader reads
  any such pointer-valued `type` as the `$ref` it was meant to be.

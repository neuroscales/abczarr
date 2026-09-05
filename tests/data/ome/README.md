# Vendored OME-NGFF example instances

These JSON files are the **official OME-NGFF example instances**, vendored
under one directory per version (`v0_6dev1` .. `v0_6rc0`) so the metadata
models and the official-schema validators can be exercised offline, with no
network access.

## Provenance

Copied from the OME `ngff` specification repository
(<https://github.com/ome/ngff>) at the matching version tag, with their
JSONC comments stripped so `json.load` accepts them. The directory names
mirror the `abczarr.ome.<version>` metadata packages and the vendored
schemas under `src/abczarr/ome/schemas/_ngff/<version>/` (see that
directory's `README.md` for the schema refs). The files are kept
**byte-for-byte** (comment-stripping aside); a fixture is never rewritten to
make it pass a test.

## These are pre-release, transitional instances

0.6 went through four development tags (`0.6.dev1` .. `0.6.dev4`) and a
release candidate (`0.6rc0`) before the format settled. Some vendored
instances therefore do **not** conform to their own tag's published schema:
they were written against an earlier draft of that same tag, and the format
was tightened later. This is expected, and is not an abczarr bug. The two
groups below are documented rather than corrected.

The `abczarr` metadata models read every one of these instances and
round-trip it (`tests/test_ome_0_6_dev.py`). The stricter official-schema
check is applied in `tests/test_ome_schema.py`, which excludes the
non-conforming instances below by an explicit, commented list and asserts
each of them is genuinely rejected (so a future upstream fix is noticed
rather than silently masked).

### Group 1 — stale `version` string (`v0_6dev1`, `v0_6dev2`)

Every whole-document instance under `v0_6dev1`, and all but
`multiscales_transformations.json` under `v0_6dev2`, carries a `version`
field from an earlier draft — `"0.5"`, `"0.5-dev"`, or `"0.6dev2"` — rather
than the tag's own string. The official `ome_zarr` schema pins `version`, so
these fail on that field alone; normalising the discriminator makes the rest
of each document conform. The models normalise `version` before parsing (see
`test_ome_0_6_dev.py::test_ome_document_roundtrips`).

### Group 2 — transitional structural forms (`v0_6dev1` .. `v0_6dev4`)

A few instances use a coordinate-systems shape that a later tag replaced:

- `multiscales_example_relative.json` (dev1–dev4) writes a coordinate
  system's `axes` as a **mapping** (`{"x": {...}}`) instead of a list, and
  the dev3/dev4 copies omit the top-level `ome` wrapper. `0.6rc0` uses the
  list form inside an `ome` wrapper.
- `scene_stitching.json` (dev3, dev4) gives a transform's `output` a
  coordinate-system **name string** rather than an object; the
  string→object overhaul completed in `0.6.dev4`/`0.6rc0` (the same
  transitional form that excludes some `mapAxis`/`byDimension` transform
  instances in `test_ome_0_6_dev.py`).
- `multiscales_reference_to_label.json` (dev4) uses the axes-as-mapping
  coordinate system; the `0.6rc0` copy uses the list form.

The corresponding `0.6rc0` instances all conform and are validated in
`test_ome_schema.py::test_06rc0_ome`.

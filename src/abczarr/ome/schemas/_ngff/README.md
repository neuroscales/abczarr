# Vendored NGFF JSON schemas

These are the **official OME-NGFF JSON schemas**, vendored verbatim so that
`abczarr` can validate OME-Zarr metadata offline, with no network access and
no dependency on an upstream package. Each `*.schema` file is byte-for-byte
the upstream source except where noted below.

## Provenance

Copied from the OME `ngff` specification repository
(<https://github.com/ome/ngff>) at these refs:

| version dir | upstream ref | upstream path |
|-------------|--------------|---------------|
| `v0_1`      | branch `0.1` | `schemas/` |
| `v0_2`      | *reconstructed* — see below | — |
| `v0_3`      | branch `0.3` | `schemas/` |
| `v0_4`      | branch `0.4` | `schemas/` |
| `v0_5`      | branch `0.5` | `schemas/` |
| `v0_6dev1`  | tag `0.6.dev1` | `ngff_spec/schemas/` |
| `v0_6dev2`  | tag `0.6.dev2` | `ngff_spec/schemas/` |
| `v0_6dev3`  | tag `0.6.dev3` | `schemas/` |
| `v0_6dev4`  | tag `0.6.dev4` | `schemas/` |
| `v0_6rc0`   | tag `0.6rc0`   | `schemas/` |

The directory names mirror the `abczarr.ome.<version>` metadata packages; each
file's `$id` carries the official version segment (`.../0.6.dev1/schemas/...`).

## `v0_2` is reconstructed, not official

NGFF 0.2 never published a distinct JSON schema — its version pointer resolves
to a `0.6.dev` release, and the `0.2` branch of the upstream repository holds
`0.6.dev2` content. The `v0_2` files here are therefore an **abczarr
reconstruction**, derived from the official `0.1` schema by bumping the version
identifier to `0.2`. This is faithful to the spec: the only normative 0.1→0.2
change was the requirement that arrays be 5-dimensional in `tczyx` order; the
multiscales JSON structure was unchanged (the `axes` list arrived in 0.3). Each
reconstructed file records this in a top-level `$comment`.

## Known upstream defects (normalized at load time)

The `0.6.dev1` and `0.6.dev2` `coordinate_transformation(s).schema` files
misplace a `required` array *inside* a `properties` object (in `mapAxis`, and
in dev1 also `affine`/`rotation`) — a shape that is never valid JSON Schema (a
property's value cannot be a bare array). Lenient validators tolerate it
lazily; an eager compiler rejects it. The vendored files keep the upstream
bytes; the loader lifts any such misplaced `required` to its correct sibling
position when building the validators. See `abczarr/ome/schemas/_validation.py`.

The 0.6 schemas put the "2-3 space axes" count bound in the shared
`axes.schema`, which `coordinate_systems.schema` `$ref`s — so the bound is
applied to *every* coordinate system's axes, not just an image's. RFC-5
(Coordinate Systems and Transformations) scopes that rule to axes "inside
multiscales metadata" only and leaves a general coordinate system's
dimensionality unbounded (an array coordinate system's length equals its
Zarr array's dimensionality; axis `type` is only *SHOULD*). The shared
schema is therefore over-broad: reference `jsonschema` (Draft 2020-12)
rejects upstream's own canonical examples against it — a 4-space-axis system
(`v0_6rc0/byDimension2.json`) and 1-D transformation systems
(`v0_6dev2/coordinates1d.json`, `displacement1d.json`). The vendored files
keep the upstream bytes; the `contains` count-bound enforcement in
`_contains.py` is suppressed once a `$ref` crosses into `axes.schema`, so
abczarr holds only image axes to the bound. Tracked upstream (issue #125).

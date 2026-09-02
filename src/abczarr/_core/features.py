"""Fine-grained feature keys -- the shared vocabulary for "what a specific
codec / chunk grid / dtype needs, and what a driver provides".

A feature key is a namespaced string, ``"<version>:<kind>:<name>"``, e.g.
``"v3:codec:zstd"`` or ``"v2:filter:delta"``. The metadata layer builds the
keys an array *requires* (:meth:`ArrayMetadata.required_features`); a driver
declares the keys it *provides*; validation is set difference. Keys are
open-ended -- an unknown one simply never matches, so a new codec never
crashes selection.

This lives in ``_core`` so the metadata layer and the ``abc`` capability
layer can share one definition without either importing the other.
"""

__all__ = [
    "feature_key",
    "FEATURE_VERSIONS",
    "FEATURE_KINDS",
]

#: The namespace a feature key starts with -- the Zarr format version.
FEATURE_VERSIONS = ("v1", "v2", "v3")

#: The kinds of extension a feature key names.
FEATURE_KINDS = (
    "codec",
    "filter",
    "compressor",
    "chunk_grid",
    "chunk_key_encoding",
    "data_type",
    "storage_transformer",
    "extension",
)


def feature_key(version: str, kind: str, name: str) -> str:
    """Build a fine-grained feature key, e.g.
    ``feature_key("v3", "codec", "zstd") -> "v3:codec:zstd"``.

    *version* is one of :data:`FEATURE_VERSIONS`, *kind* one of
    :data:`FEATURE_KINDS`; *name* is the codec/grid/dtype name as it appears
    in the metadata. The parts are validated so a typo becomes an error here
    rather than a key that silently never matches.
    """
    if version not in FEATURE_VERSIONS:
        raise ValueError(
            "unknown feature version {!r}; expected one of {}".format(
                version, ", ".join(FEATURE_VERSIONS)
            )
        )
    if kind not in FEATURE_KINDS:
        raise ValueError(
            "unknown feature kind {!r}; expected one of {}".format(
                kind, ", ".join(FEATURE_KINDS)
            )
        )
    return f"{version}:{kind}:{name}"

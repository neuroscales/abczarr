"""Fine-grained feature keys, the shared vocabulary describing what a
codec, chunk grid, or data type needs, and what a driver provides.

A feature key is a namespaced string, ``"<version>:<kind>:<name>"``, for
example ``"v3:codec:zstd"`` or ``"v2:filter:delta"``. The metadata layer
builds the keys an array requires
(:meth:`ArrayMetadata.required_features`), and a driver declares the
keys it provides. Driver selection compares the two sets. Keys are
open-ended. An unknown one never matches anything, so a new codec added
to the metadata layer never breaks selection for a driver that does not
yet know it.

Both the metadata layer and the ``abc`` capability layer import this
module, so a feature key has one definition shared by both.
"""

__all__ = [
    "feature_key",
    "FEATURE_VERSIONS",
    "FEATURE_KINDS",
]

#: The Zarr format version namespace a feature key starts with.
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
    """Build a fine-grained feature key from its three parts.

    ``feature_key("v3", "codec", "zstd")`` returns ``"v3:codec:zstd"``.
    *version* must be one of :data:`FEATURE_VERSIONS` and *kind* one of
    :data:`FEATURE_KINDS`. *name* is the codec, chunk grid, or data type
    name as it appears in the metadata. Validating the two fixed parts
    here turns a typo into an immediate error, instead of a key that
    would otherwise never match anything.
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

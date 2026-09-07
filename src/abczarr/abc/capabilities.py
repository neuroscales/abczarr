"""The capability vocabulary shared by nodes and stores.

Almost every operation on a store or a node works one way or
another, because abczarr falls back to building an operation from
simpler ones when a backend has no direct support for it. The useful
question is therefore rarely whether an operation works at all, but
whether it is fast on the current backend or built up from something
simpler. [Support][abczarr.abc.capabilities.Support] answers that
question in three states.
[supports][abczarr.abc.capabilities.SupportsCapabilities.supports]
collapses those three states to a plain `bool` for a caller that only
needs a yes or a no.

Two granularities of name live here. A coarse capability is a broad
feature a caller checks before committing to an operation. The named
coarse capabilities are collected in `KNOWN_CAPABILITIES`, and
include `"sharding"`, `"async"`, `"listing"`, `"partial_read"` and
`"transactions"`.

A feature key is a fine-grained, namespaced name for a single codec,
chunk grid, chunk-key encoding, or data type, such as
`"v3:codec:zstd"`, `"v2:filter:delta"`, or
`"v3:chunk_grid:rectilinear"`. Feature keys are open-ended. A driver
declares the ones it has, and a query about an unknown one simply
answers `Support.NONE`. Build a feature key with
[feature_key][abczarr.abc.capabilities.feature_key].
"""

__all__ = [
    "Support",
    "KNOWN_CAPABILITIES",
    "feature_key",
]

# stdlib
import enum

# dependencies
import typing_extensions as tx

# core: the feature-key vocabulary is shared with the metadata layer, so it
# is defined in _core and re-exported here. feature_key stays in __all__.
from abczarr._core.features import (  # noqa: F401
    FEATURE_KINDS,
    FEATURE_VERSIONS,
    feature_key,
)


class Support(enum.Enum):
    """How well a driver provides a capability.

    `NATIVE` means the backend performs the operation directly, the
    fast path. `SYNTHESIZED` means abczarr builds the operation from
    simpler ones. The result is correct, but possibly slower than a
    backend that performs the operation directly. `NONE` means the
    operation is not available at all, and raises
    [UnsupportedZarrOperation][abczarr.errors.UnsupportedZarrOperation].

    `bool(support)` is `True` unless the value is `NONE`. A plain
    truth test therefore answers whether the operation can happen at
    all, regardless of how.

    !!! example
        ```pycon
        >>> bool(Support.NATIVE), bool(Support.SYNTHESIZED)
        (True, True)
        >>> bool(Support.NONE)
        False
        ```
    """

    NATIVE = "native"
    SYNTHESIZED = "synthesized"
    NONE = "none"

    def __bool__(self) -> bool:
        return self is not Support.NONE


#: The coarse capability names ``supports`` understands. A driver
#: advertises the subset it provides. A query about any other name
#: simply answers ``Support.NONE``, so ``supports`` returns
#: ``False``. A caller written against a newer vocabulary therefore
#: never crashes an older driver.
KNOWN_CAPABILITIES = frozenset(
    {
        # -- node --
        "sharding",              # zarr v3 sharded chunk grids
        "async",                 # a native coroutine I/O surface
        "consolidated_metadata",
        "codecs_v2",
        "codecs_v3",
        # -- store I/O --
        "listing",               # enumerate keys under a prefix
        "writes",                # write a key
        "deletes",               # remove a key
        "partial_read",          # read a byte range of a key
        "partial_write",         # write a byte range of a key
        "transactions",          # batch operations into one commit
        "atomic_transactions",   # ... and commit them all-or-nothing
    }
)

class SupportsCapabilities:
    """Mixin that gives a node or store the capability query.

    A store or driver may report different capabilities from one
    instance to the next. A store over `memory://` lists differently
    from one over `s3://`, and whether a given codec is available
    can depend on what happens to be installed. `capability` can
    therefore be overridden by a subclass to answer from live state
    rather than from a fixed table.
    """

    #: What this class provides. A driver overrides this. The base
    #: class declares nothing.
    _CAPABILITIES: tx.ClassVar[tx.Mapping[str, Support]] = {}

    def capability(self, name: str) -> Support:
        """How this object provides the capability `name`.

        Parameters
        ----------
        name : str
            A capability name, such as `"listing"` or `"transactions"`.

        Returns
        -------
        Support
            `Support.NONE` for a name this object does not know.
        """
        return self._CAPABILITIES.get(name, Support.NONE)

    def supports(self, name: str, *, native: bool = False) -> bool:
        """Whether this object provides the capability `name`.

        Parameters
        ----------
        name : str
            A capability name, such as `"listing"` or `"transactions"`.
        native : bool, optional
            When `True`, only count it as supported if the backend
            does it directly. Otherwise `True` whenever it can happen
            at all, native or synthesized.

        Returns
        -------
        bool
            `False` for a name this object does not know.
        """
        state = self.capability(name)
        return state is Support.NATIVE if native else bool(state)

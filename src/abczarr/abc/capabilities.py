"""The capability vocabulary shared by nodes and stores.

Almost every operation on a store or a node works one way or another,
because abczarr falls back to building an operation from simpler ones
when a backend has no direct support for it. So the useful question
usually is not "does this work?" but "is it fast here, or built up from
something simpler?"
[Support][abczarr.abc.capabilities.Support] answers that in three
states, and
[supports][abczarr.abc.capabilities.SupportsCapabilities.supports]
collapses it to a plain `bool` when a caller only needs yes or no.

Two granularities of name live here:

* **coarse capabilities** -- broad features a caller checks before
  committing to an operation
  ([KNOWN_CAPABILITIES][abczarr.abc.capabilities.KNOWN_CAPABILITIES]):
  `"sharding"`, `"async"`, `"listing"`, `"partial_read"`,
  `"transactions"`, and so on.
* **feature keys** -- fine-grained, namespaced names for a single
  codec, chunk grid, chunk-key encoding, or data type, e.g.
  `"v3:codec:zstd"`, `"v2:filter:delta"`,
  `"v3:chunk_grid:rectilinear"`. These are open-ended: a driver
  declares the ones it has, and asking about an unknown one simply
  answers `Support.NONE`. Build one with
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

# core -- the feature-key vocabulary is shared with the metadata layer, so it
# is defined in _core and re-exported here (feature_key stays in __all__).
from abczarr._core.features import (  # noqa: F401
    FEATURE_KINDS,
    FEATURE_VERSIONS,
    feature_key,
)


class Support(enum.Enum):
    """How well a driver provides a capability.

    `NATIVE` -- the backend does it directly, the fast path.
    `SYNTHESIZED` -- abczarr builds it from simpler operations;
    correct, but possibly slower than a backend that does it
    directly.
    `NONE` -- not available; the operation raises
    [UnsupportedZarrOperation][abczarr.abc.errors.UnsupportedZarrOperation].

    `bool(support)` is `True` unless it is `NONE`, so a plain truth
    test answers "can this happen at all?".

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
#: advertises the subset it provides; asking about any other name
#: simply answers ``Support.NONE`` (``supports`` returns ``False``),
#: so a caller written against a newer vocabulary never crashes an
#: older driver.
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
    instance to the next -- a store over `memory://` lists
    differently from one over `s3://`, and whether a given codec is
    available can depend on what happens to be installed. So `support`
    can be overridden to answer from live state rather than a fixed
    table.
    """

    #: What this class provides. Overridden per driver; empty here.
    _CAPABILITIES: tx.ClassVar[tx.Mapping[str, Support]] = {}

    def support(self, capability: str) -> Support:
        """How this object provides *capability*.

        Parameters
        ----------
        capability : str
            A capability name, e.g. `"listing"` or `"transactions"`.

        Returns
        -------
        Support
            `Support.NONE` for a name this object does not know.
        """
        return self._CAPABILITIES.get(capability, Support.NONE)

    def supports(self, capability: str, *, native: bool = False) -> bool:
        """Whether this object provides *capability*.

        Parameters
        ----------
        capability : str
            A capability name, e.g. `"listing"` or `"transactions"`.
        native : bool, optional
            When `True`, only count it as supported if the backend
            does it directly. Otherwise `True` whenever it can happen
            at all, native or synthesized.

        Returns
        -------
        bool
            `False` for a name this object does not know.
        """
        state = self.support(capability)
        return state is Support.NATIVE if native else bool(state)

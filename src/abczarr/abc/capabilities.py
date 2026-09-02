"""The capability vocabulary shared by nodes and stores.

A member of the uniform surface is almost always *present* -- under the
delegate -> synthesize -> raise rule a store can nearly always answer, one
way or another. So the useful question is not "does this work?" but "is it
**native** here, or built from primitives?". :class:`Support` answers that in
three states, and :meth:`supports` collapses it to a bool when a caller only
needs yes/no.

Two granularities of name live here:

* **coarse capabilities** -- broad features a caller branches on before
  committing to an operation (:data:`KNOWN_CAPABILITIES`): ``"sharding"``,
  ``"async"``, ``"listing"``, ``"partial_read"``, ``"transactions"``, ...
* **feature keys** -- fine-grained, namespaced names for a single codec,
  chunk grid, chunk-key encoding or data type the extensible Zarr spec can
  carry: ``"v3:codec:zstd"``, ``"v2:filter:delta"``,
  ``"v3:chunk_grid:rectilinear"``. These are open-ended -- a driver declares
  the ones it has, an unknown one simply answers :attr:`Support.NONE`, and
  nothing enumerates the universe. Build one with :func:`feature_key`.
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

    ``NATIVE`` -- delegated to the backend (the fast path).
    ``SYNTHESIZED`` -- built from more primitive operations; correct, but
    possibly slower than a backend that does it directly.
    ``NONE`` -- neither possible; the operation raises
    :class:`~abczarr.abc.errors.UnsupportedZarrOperation`.

    ``bool(support)`` is ``True`` unless it is ``NONE``, so a plain truth test
    answers "can this happen at all?".
    """

    NATIVE = "native"
    SYNTHESIZED = "synthesized"
    NONE = "none"

    def __bool__(self) -> bool:
        return self is not Support.NONE


#: The coarse capability names :meth:`supports` understands. A driver
#: advertises the subset it provides; asking about any other name simply
#: answers :attr:`Support.NONE` (``supports`` returns ``False``), so a caller
#: written against a newer vocabulary never crashes an older driver.
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
    """Mixin giving a node or store the capability query.

    A subclass declares what it provides in :attr:`_CAPABILITIES`, a mapping
    from a capability name to a :class:`Support`. The query is an *instance*
    method -- real capability is per instance and per build (a store over
    ``memory://`` lists differently from one over ``s3://``; whether a given
    backend build has a codec is a property of that install) -- so a subclass
    may override :meth:`support` to answer from live state.
    """

    #: What this class provides. Overridden per driver; empty here.
    _CAPABILITIES: tx.ClassVar[tx.Mapping[str, Support]] = {}

    def support(self, capability: str) -> Support:
        """How this object provides *capability* (:class:`Support`)."""
        return self._CAPABILITIES.get(capability, Support.NONE)

    def supports(self, capability: str, *, native: bool = False) -> bool:
        """Whether this object provides *capability*.

        With ``native=True``, ``True`` only when the backend does it directly
        (:attr:`Support.NATIVE`); otherwise ``True`` whenever it can happen at
        all, native or synthesized. An unknown name is always ``False``.
        """
        state = self.support(capability)
        return state is Support.NATIVE if native else bool(state)

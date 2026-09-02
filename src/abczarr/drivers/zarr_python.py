"""The zarr-python backend driver.

Declares what a given install of zarr-python can read and write -- coarse
capabilities and the individual codecs, chunk grids and chunk-key encodings
it has -- by asking the installed library, so selection reflects the real
build rather than a guess. Reading and writing array data through it lands
with the node adapters in a later change; this module is the driver's
feature declaration and its part in driver selection.
"""

__all__ = [
    "ZarrPythonDriver",
]

# stdlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

# dependencies
import typing_extensions as tx

# core
from abczarr._core.features import FEATURE_KINDS, FEATURE_VERSIONS
from abczarr.abc.capabilities import Support
from abczarr.drivers.base import Driver

# optionals -- the module imports without zarr; a driver with no zarr simply
# reports that it can open nothing.
try:
    import zarr
    import zarr.registry as _registry
except ImportError:  # pragma: no cover - exercised only without zarr
    zarr = None
    _registry = None


#: Coarse capabilities a zarr-python 3.x install provides.
_V3_CAPABILITIES = {
    "async": Support.NATIVE,
    "sharding": Support.NATIVE,
    "consolidated_metadata": Support.NATIVE,
    "codecs_v2": Support.NATIVE,
    "codecs_v3": Support.NATIVE,
    "listing": Support.NATIVE,
    "writes": Support.NATIVE,
    "deletes": Support.NATIVE,
    "partial_read": Support.NATIVE,
}


def _installed_major() -> int:
    """The major version of the installed zarr, or 0 when it is absent."""
    if zarr is None:
        return 0
    try:
        return int(_dist_version("zarr").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        return 0


def _parse_feature(key: str) -> tx.Optional[tx.Tuple[str, str, str]]:
    """Split a feature key into (version, kind, name), or None when it is
    not a well-formed one."""
    parts = key.split(":", 2)
    if len(parts) != 3:
        return None
    version, kind, name = parts
    if version not in FEATURE_VERSIONS or kind not in FEATURE_KINDS:
        return None
    return version, kind, name


def _resolves(lookup: tx.Callable[[str], object], name: str) -> bool:
    """Whether *lookup* returns something for *name* rather than raising."""
    try:
        lookup(name)
        return True
    except Exception:
        return False


class ZarrPythonDriver(Driver):
    """The zarr-python backend, as a driver.

    Reports what the installed zarr-python can do -- its coarse capabilities
    and, codec by codec, what its registry holds -- so an array is only
    routed to it when it actually has everything the array needs.
    """

    name = "zarr-python"

    def __init__(self) -> None:
        self._major = _installed_major()

    def support(self, capability: str) -> Support:
        if self._major < 3:
            # zarr 2.x uses a different library API; its support lands with
            # the version adapter.
            return Support.NONE
        if capability in _V3_CAPABILITIES:
            return _V3_CAPABILITIES[capability]
        parsed = _parse_feature(capability)
        if parsed is None:
            return Support.NONE
        return self._feature_support(*parsed)

    def _feature_support(
        self, version: str, kind: str, name: str
    ) -> Support:
        """Whether the installed zarr provides one v3 codec / grid / etc."""
        if version != "v3":
            return Support.NONE
        if kind == "codec":
            found = _resolves(_registry.get_codec_class, name)
        elif kind == "chunk_key_encoding":
            found = _resolves(_registry.get_chunk_key_encoding_class, name)
        elif kind == "chunk_grid":
            # only the regular grid has a zarr-python representation
            found = name == "regular"
        elif kind == "data_type":
            # zarr-python handles the standard numeric data types
            found = True
        else:
            found = False
        return Support.NATIVE if found else Support.NONE

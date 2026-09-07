"""Read and write a group's OME-Zarr metadata as a typed object.

An OME-Zarr group carries its metadata in its user attributes. The
[ome][abczarr.abc.sync.ZarrNode.ome] property on a node reads those
attributes into a typed [OME][abczarr.ome.base.OME] object and writes
one back; this module holds the functions it delegates to, which handle
the two envelopes the spec uses:

* From OME-NGFF 0.5 on, the metadata lives under a single ``"ome"``
  attribute: ``attrs["ome"] == {"version": ..., "multiscales": [...]}``.
* Up to and including 0.4, the metadata's own keys (``multiscales``,
  ``plate``, ``well``, and so on) sit directly in the attributes, with
  no wrapper, and the version is recorded inside the multiscale rather
  than at the top.

These functions take any node exposing a write-through ``attrs`` mapping
and an ``update_attributes`` method. This is the same surface
[attrs][abczarr.abc.sync.ZarrNode.attrs] is built on.
"""

__all__ = [
    "read_ome",
    "write_ome",
    "update_ome",
    "delete_ome",
    "ome_version",
]

# stdlib
from collections import abc

# dependencies
import typing_extensions as tx

# locals
from .base import _MODULES, _VERSIONS, LATEST_STABLE, OME

if tx.TYPE_CHECKING:
    # For annotations only; a runtime import of the node contract would
    # cycle, since the node reaches back here for its ``ome`` property.
    from ..abc.sync import ZarrNode

#: The attribute key the 0.5+ envelope wraps the metadata in.
_OME_KEY = "ome"

#: The top-level keys a bare (<= 0.4) OME payload is recognized by -- each
#: is the discriminator of an OME carrier subclass (an image's
#: ``multiscales``, a plate, a well, ...), written under its JSON key.
_CARRIERS = (
    "multiscales",
    "labels",
    "plate",
    "well",
    "image-label",
    "bioformats2raw.layout",
    "series",
)

#: Every attribute key OME metadata can occupy, either envelope.
_OME_KEYS = (_OME_KEY,) + _CARRIERS


def read_ome(node: "ZarrNode") -> tx.Optional[OME]:
    """Read a group's OME-Zarr metadata as a typed object.

    Detects which envelope the group uses, the ``"ome"`` attribute of 0.5
    and later or the bare attributes of 0.4 and earlier, and parses it
    into the matching version's [OME][abczarr.ome.base.OME] subclass. For
    a bare payload, the version is read from the multiscale, plate, or
    well that carries it. When no such field names a version, the
    earliest version that accepts the payload is used instead.

    Parameters
    ----------
    node : ZarrNode
        The group whose attributes hold the metadata.

    Returns
    -------
    OME or None
        The typed metadata, or `None` if the group carries none.
    """
    attrs = node.attrs
    if _OME_KEY in attrs:
        inner = attrs[_OME_KEY]
        if isinstance(inner, abc.Mapping):
            return OME.from_json(inner)
        return None
    if _has_carrier(attrs):
        payload = dict(attrs)
        payload["version"] = _infer_version(payload)
        return OME.from_json(payload)
    return None


def write_ome(
    node: "ZarrNode", ome: tx.Union[OME, tx.Mapping[str, tx.Any]]
) -> None:
    """Write OME metadata to a group's attributes.

    Stores *ome* in the envelope its version calls for: under the
    ``"ome"`` attribute for 0.5 and later, or directly as the top-level
    attribute keys for 0.4 and earlier (where the version stays on the
    multiscale, not at the top). Other, unrelated attributes are left
    untouched.

    Parameters
    ----------
    node : ZarrNode
        The group to write to.
    ome : OME or mapping
        The metadata to store, either a typed
        [OME][abczarr.ome.base.OME] object or a plain JSON-style mapping
        of the inner metadata (which must carry a ``version``). Its
        version selects the envelope.
    """
    payload, stale = ome_write_plan(node.attrs, ome)
    # Replace the node's OME metadata wholesale: drop any stale OME keys the
    # new payload does not itself write (the other envelope, a carrier that
    # is no longer present), while leaving unrelated attributes untouched.
    attrs = node.attrs
    for key in stale:
        del attrs[key]
    node.update_attributes(payload)


def ome_write_plan(
    current: tx.Mapping[str, tx.Any],
    ome: tx.Union[OME, tx.Mapping[str, tx.Any]],
) -> tx.Tuple[tx.Dict[str, tx.Any], tx.List[str]]:
    """Plan the attribute write that stores *ome* over *current*.

    Works out the envelope from *ome*'s version and returns ``(payload,
    stale)``. `payload` holds the attribute keys to set. `stale` lists
    the OME keys already present in *current* that the new payload does
    not write, such as the other envelope's key or a carrier no longer
    used, and that must therefore be dropped. An attribute unrelated to
    OME metadata is named in neither and is left as it is.

    Parameters
    ----------
    current : mapping
        The node's current attributes.
    ome : OME or mapping
        The metadata to store; a mapping is parsed to a typed container so
        its version selects the envelope.

    Returns
    -------
    (dict, list of str)
        The attributes to set, and the OME keys to remove.
    """
    if not isinstance(ome, OME):
        # A plain mapping is parsed to the typed container first, so its
        # version -- and therefore the right envelope -- is known.
        ome = OME.from_json(ome)
    inner = ome.to_json()
    if _is_wrapped(ome.version):
        payload = {_OME_KEY: inner}  # type: tx.Dict[str, tx.Any]
    else:
        # The <= 0.4 envelope has no top-level ``version`` -- it lives inside
        # the multiscale (and the plate / well) -- so drop the one the typed
        # container emits before storing the payload's keys directly.
        inner.pop("version", None)
        payload = inner
    stale = [
        key for key in _OME_KEYS if key in current and key not in payload
    ]
    return payload, stale


def update_ome(
    node: "ZarrNode", ome: tx.Union[OME, tx.Mapping[str, tx.Any]]
) -> None:
    """Shallow-merge OME metadata into a group's, and persist it.

    This function does for OME metadata what
    [update_attributes][abczarr.abc.sync.ZarrNode.update_attributes] does
    for a node's plain attributes. A top-level key of *ome*, such as
    ``version``, ``multiscales``, or ``omero``, replaces the value
    already on the node's OME metadata with the same key. A top-level key
    the node's current metadata already has that *ome* does not name is
    kept unchanged. When the node has no OME metadata yet and the merged
    result still names no version, the latest released OME version is
    used.

    The merge is shallow: it replaces whole top-level keys rather than
    descending into a multiscale or a plate. For a structured edit, read
    the typed object, change it with ``evolve``, and assign it back.

    Parameters
    ----------
    node : ZarrNode
        The group to update.
    ome : OME or mapping
        The metadata whose top-level keys are merged in.
    """
    write_ome(node, merge_ome(read_ome(node), ome))


def merge_ome(
    current: tx.Optional[OME],
    incoming: tx.Union[OME, tx.Mapping[str, tx.Any]],
) -> tx.Dict[str, tx.Any]:
    """The shallow merge of *incoming* onto *current*, as an inner OME dict.

    A top-level key of *incoming* replaces the value of the same key in
    *current*. When neither side supplies a version, the result defaults
    to [LATEST_STABLE][abczarr.ome.base.LATEST_STABLE].
    """
    merged = current.to_json() if current is not None else {}
    if isinstance(incoming, OME):
        incoming = incoming.to_json()
    merged.update(incoming)
    if not merged.get("version"):
        merged["version"] = LATEST_STABLE
    return merged


def delete_ome(node: "ZarrNode") -> None:
    """Remove a group's OME-Zarr metadata from its attributes.

    Drops the ``"ome"`` attribute for a 0.5+ envelope, or every bare
    carrier key for a <= 0.4 one. Other attributes are left untouched. A
    group that carries no OME metadata is left unchanged.

    Parameters
    ----------
    node : ZarrNode
        The group to clear.
    """
    _, stale = ome_delete_plan(node.attrs)
    attrs = node.attrs
    for key in stale:
        del attrs[key]


def ome_delete_plan(
    current: tx.Mapping[str, tx.Any],
) -> tx.Tuple[tx.Dict[str, tx.Any], tx.List[str]]:
    """Plan the attribute write that clears OME metadata from *current*.

    Returns ``(payload, stale)`` in the same shape as
    [ome_write_plan][abczarr.ome.node.ome_write_plan]. `payload` is
    always empty, since nothing is written. `stale` lists every OME key
    present in *current*, so all of it is dropped.
    """
    return {}, [key for key in _OME_KEYS if key in current]


def ome_version(node: "ZarrNode") -> tx.Optional[str]:
    """The OME-NGFF version a group declares, or `None`.

    Reads the version from the ``"ome"`` envelope when present, and
    otherwise infers it from a bare payload the same way
    [read_ome][abczarr.ome.node.read_ome] does. Returns `None` when the
    group carries no OME metadata.

    Parameters
    ----------
    node : ZarrNode
        The group to inspect.

    Returns
    -------
    str or None
        The version string (such as ``"0.4"`` or ``"0.6rc0"``), or
        `None`.
    """
    attrs = node.attrs
    if _OME_KEY in attrs:
        inner = attrs[_OME_KEY]
        if isinstance(inner, abc.Mapping):
            version = inner.get("version")
            return version if isinstance(version, str) else None
        return None
    if _has_carrier(attrs):
        return _infer_version(attrs)
    return None


# ----------------------------------------------------------------------
#   helpers
# ----------------------------------------------------------------------


def _has_carrier(attrs: tx.Mapping[str, tx.Any]) -> bool:
    """Whether *attrs* is a bare (<= 0.4) OME payload."""
    return any(key in attrs for key in _CARRIERS)


def _is_wrapped(version: str) -> bool:
    """Whether *version* wraps its metadata under the ``"ome"`` key.

    This is true from 0.5 on. The position of ``"0.5"`` in the ordered
    version chain marks the boundary between the bare and wrapped
    envelopes.
    """
    return (
        version in _MODULES
        and _VERSIONS.index(version) >= _VERSIONS.index("0.5")
    )


def _infer_version(payload: tx.Mapping[str, tx.Any]) -> str:
    """The OME version of a bare (<= 0.4) *payload*.

    Read from the ``version`` inside the multiscale that carries it (or the
    plate / well / image-label), which is where <= 0.4 records it. When no
    such field is present, fall back to the earliest version whose
    ``OME.from_json`` accepts the payload.
    """
    multiscales = payload.get("multiscales")
    if (
        isinstance(multiscales, abc.Sequence)
        and not isinstance(multiscales, str)
        and multiscales
        and isinstance(multiscales[0], abc.Mapping)
    ):
        version = multiscales[0].get("version")
        if version in _MODULES:
            return version
    for key in ("plate", "well", "image-label"):
        carrier = payload.get(key)
        if isinstance(carrier, abc.Mapping):
            version = carrier.get("version")
            if version in _MODULES:
                return version
    return _earliest_fit(payload)


def _earliest_fit(payload: tx.Mapping[str, tx.Any]) -> str:
    """The earliest bare (<= 0.4) version whose ``OME`` accepts *payload*."""
    for version in ("0.1", "0.2", "0.3", "0.4"):
        candidate = dict(payload)
        candidate["version"] = version
        try:
            OME.from_json(candidate)
        except Exception:
            continue
        return version
    return "0.4"

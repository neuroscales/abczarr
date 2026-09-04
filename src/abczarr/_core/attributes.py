"""The node attributes mapping and the store-routed persistence of them.

A node's user attributes live in one place: the node's cached metadata
([NodeMetadata.attributes][abczarr.metadata.base.NodeMetadata]). Reads are
served from there, so the mapping and `node.metadata.attributes` never
disagree. Writes go through the node's persistence path -- for a backend that
wraps a real Zarr object, its own `update_attributes`; for everything else, a
rewrite of the metadata document through the [Store][abczarr.abc.store.Store]
-- so no attribute write ever bypasses the store.

This file contains code from the Zarr project
https://github.com/zarr-developers/zarr-python
"""

__all__ = [
    "NodeAttributes",
    "attribute_writes",
]

# stdlib
import json

# dependencies
import typing_extensions as tx

# locals
from . import constants
from . import typing as tz

if tx.TYPE_CHECKING:
    # imported for the type only; a runtime import would cycle, since the node
    # contract (abc.sync) reaches back here for its `attrs` property
    from ..abc import ZarrNode

AttributesBase = tx.MutableMapping[str, tx.Any]


class NodeAttributes(AttributesBase):
    """A live, write-through view of a node's user attributes.

    Reads come from the node's cached metadata, so this mapping and
    ``node.metadata.attributes`` are always the same values. A write persists
    through the node -- `node.attrs["k"] = v` adds or replaces ``k``, and
    `del node.attrs["k"]` removes it -- routed through the node's own
    persistence path rather than a separate file.

    Works for both arrays and groups, and for either Zarr format version:
    the node it wraps supplies the metadata and does the writing.
    """

    def __init__(self, node: "ZarrNode") -> None:
        """
        Parameters
        ----------
        node : ZarrNode
            The Zarr array or group whose attributes this mapping views.
        """
        self._node = node

    def _current(self) -> tx.Mapping[str, tx.Any]:
        """The node's current attributes, from its cached metadata."""
        return self._node.metadata.attributes

    # ---------- MutableMapping interface ----------

    def __getitem__(self, key: str) -> tx.Any:  # noqa: ANN401
        """Get an attribute by key."""
        return self._current()[key]

    def __setitem__(self, key: str, value: tx.Any) -> None:  # noqa: ANN401
        """Set or update a single attribute, and persist it."""
        self._node.update_attributes({key: value})

    def __delitem__(self, key: str) -> None:
        """Delete a single attribute, and persist the removal."""
        remaining = dict(self._current())
        del remaining[key]
        self._node._replace_attributes(remaining)

    def __iter__(self) -> tx.Iterator[str]:
        """Iterate over a snapshot of keys."""
        return iter(dict(self._current()))

    def __len__(self) -> int:
        """Return the number of attributes."""
        return len(self._current())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self._current())!r})"

    def asdict(self) -> tx.Dict[str, tx.Any]:
        """Return a plain-dict snapshot of the attributes."""
        return dict(self._current())


def attribute_writes(
    version: tz.ZarrVersion,
    attributes: tx.Mapping[str, tx.Any],
    existing_document: tx.Optional[tx.Dict[str, tx.Any]] = None,
) -> tx.List[tx.Tuple[str, bytes]]:
    """The store writes that persist *attributes* for a node of *version*.

    Returns a list of ``(key, value)`` pairs to write through a
    [Store][abczarr.abc.store.Store] (or its async twin). A Zarr v3 node
    keeps its attributes inside the single ``zarr.json`` document, so the
    other fields of *existing_document* are preserved and only its
    ``attributes`` are replaced. A v2 or v1 node keeps them in a separate
    attributes file, which is rewritten whole.

    Parameters
    ----------
    version : int
        The node's Zarr format version (1, 2 or 3).
    attributes : mapping
        The attributes to persist.
    existing_document : dict, optional
        The current ``zarr.json`` document, for a v3 node -- its
        non-attribute fields are carried over. Ignored for v1 and v2.

    Returns
    -------
    list of (str, bytes)
        The store key and the bytes to write at it.
    """
    if version >= 3:
        document = dict(existing_document or {})
        document["attributes"] = dict(attributes)
        return [(constants.Z3_JSON, _dumps(document))]
    if version == 2:
        return [(constants.Z2ATTRS_JSON, _dumps(dict(attributes)))]
    if version == 1:
        return [(constants.Z1ATTRS_JSON, _dumps(dict(attributes)))]
    raise ValueError(f"Unsupported zarr_version: {version}")


def _dumps(data: tx.Mapping[str, tx.Any]) -> bytes:
    """Serialize *data* to compact UTF-8 JSON bytes."""
    return json.dumps(
        data, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")

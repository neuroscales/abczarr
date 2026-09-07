"""An immutable mapping, and the recursive rebuild that undoes it."""

from collections import abc

import typing_extensions as tx

KT = tx.TypeVar("KT")
VT = tx.TypeVar("VT")


class FrozenDict(tx.Mapping[KT, VT]):
    """A hashable, immutable mapping.

    Behaves like a read-only `dict`: `len`, iteration, and item lookup
    all work, but there is no `__setitem__` or `__delitem__`. Two
    `FrozenDict` instances with the same items hash equal, so a
    `FrozenDict` can itself serve as a dictionary key or a member of a
    frozen attrs class.
    """

    __slots__ = ("_data", "_hash")

    def __init__(self, *args, **kwargs) -> None:
        self._data = dict(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key: KT) -> VT:
        return self._data[key]

    def __iter__(self) -> tx.Iterator[KT]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data})"

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash


def unfreeze(value: tx.Any) -> tx.Any:
    """Rebuild *value* from plain built-in types, recursively.

    A [FrozenDict][abczarr._core.frozendict.FrozenDict], or any other
    mapping, becomes a plain ``dict``, and each of its values is rebuilt in
    the same way. A list or a tuple becomes a list of rebuilt values. Any
    other value is returned unchanged.

    The result contains only built-in types and is therefore JSON
    serializable. A structure that still holds a `FrozenDict` is not, because
    ``json.dumps`` does not recognize the mapping. This function is applied to
    a node's attributes before they are handed to a backend's attribute write.
    """
    if isinstance(value, abc.Mapping):
        return {key: unfreeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [unfreeze(item) for item in value]
    return value

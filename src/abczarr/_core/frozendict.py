from collections.abc import Mapping

import typing_extensions as tx


KT = tx.TypeVar("KT")
VT = tx.TypeVar("VT")


if hasattr(Mapping, "__class_getitem__"):

    class FrozenDict(Mapping[KT, VT]):
        """An immutable dictionary."""

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

else:

    class FrozenDict(Mapping, tx.Generic[KT, VT]):
        """An immutable dictionary."""

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

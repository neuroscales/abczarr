__all__ = [
    "Store",
    "S3Store",
    "GCSStore",
    "AzureStore",
    "LocalStore",
    "MemoryStore",
    "HTTPStore"
]

# stdlib
from types import TracebackType

# dependencies
import typing_extensions as tx

# locals
from . import path as spath  # StorePath

# typing
Args = tx.Tuple[tx.Any, ...]
Kwargs = tx.Dict[str, tx.Any]


class Store:

    def __init__(self, path: spath.AsyncStorePath) -> None:
        if isinstance(path, Store):
            path = path._store_path
        self._store_path = path
        self._is_open = False

    @classmethod
    async def open(
        cls, *args: tx.Unpack[Args], **kwargs: tx.Unpack[Kwargs]
    ) -> tx.Self:
        store = cls(*args, **kwargs)
        await store._open()
        return store

    def __enter__(self) -> tx.Self:
        return self

    def __exit__(
        self,
        exc_type: tx.Optional[tx.Type[BaseException]],
        exc_value: tx.Optional[BaseException],
        traceback: tx.Optional[TracebackType],
    ) -> None:
        """Close the store."""
        self.close()

    async def _open(self) -> None:
        """
        Open the store.

        Raises
        ------
        ValueError
            If the store is already open.
        """
        if self._is_open:
            raise ValueError("store is already open")
        self._is_open = True

    @property
    def path(self) -> str:
        """Path within the store."""
        return self._store_path.path

    @property
    def url(self) -> str:
        """URL of the store."""
        return self._store_path.as_uri()

    @property
    def store_path(self) -> spath.StorePath:
        """Path to the store."""
        return self._store_path

    @property
    def spec(self) -> spath.StorePath:
        """Store specification."""
        return self.store_path

    async def list(self) -> tx.List[str]:
        """List all keys in the store."""
        keys = []
        async for item in self.iter_keys():
            keys.append(item.name)
        return keys


class S3Store(Store):

    @property
    def bucket(self) -> str:
        """Get the S3 bucket name."""
        return self.store_path.bucket


class GCSStore(Store):

    @property
    def bucket(self) -> str:
        """Get the GCS bucket name."""
        return self.store_path.bucket



class AzureStore(Store):

    @property
    def bucket(self) -> str:
        """Get the Azure bucket name."""
        return self.store_path.bucket


class LocalStore(Store):
    ...


class MemoryStore(Store):
    ...


class HTTPStore(Store):
    ...

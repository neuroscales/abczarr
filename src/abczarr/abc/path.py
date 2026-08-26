__all__ = [
    "StorePath",
    "S3StorePath",
    "GCSStorePath",
    "AzureStorePath",
    "LocalStorePath",
    "MemoryStorePath",
    "HTTPStorePath"
]
# core
from abczarr._core import path as wpath


class StorePath(wpath.WrappedPath):
    """Base class for zarr stores."""

    def __init__(self, *args, **kwargs) -> None:
        read_only = kwargs.pop("read_only", False)
        super().__init__(*args, **kwargs)
        self.read_only = read_only


@StorePath.register_subclass
class S3StorePath(wpath.S3Path, StorePath):
    ...


@StorePath.register_subclass
class GCSStorePath(wpath.GCSPath, StorePath):
    ...


@StorePath.register_subclass
class AzureStorePath(wpath.AzurePath, StorePath):
    ...


@StorePath.register_subclass
class LocalStorePath(wpath.LocalPath, StorePath):
    ...


@StorePath.register_subclass
class MemoryStorePath(wpath.WrappedPath, StorePath):
    ...


@StorePath.register_subclass
class HTTPStorePath(wpath.HTTPPath, StorePath):
    ...



class AsyncStorePath(wpath.AsyncWrappedPath):
    """Base class for zarr stores."""

    def __init__(self, *args, **kwargs) -> None:
        read_only = kwargs.pop("read_only", False)
        super().__init__(*args, **kwargs)
        self.read_only = read_only


@AsyncStorePath.register_subclass
class S3AsyncStorePath(wpath.AsyncS3Path, AsyncStorePath):
    ...


@AsyncStorePath.register_subclass
class GCSAsyncStorePath(wpath.AsyncGCSPath, AsyncStorePath):
    ...


@AsyncStorePath.register_subclass
class AzureAsyncStorePath(wpath.AsyncAzurePath, AsyncStorePath):
    ...


@AsyncStorePath.register_subclass
class LocalAsyncStorePath(wpath.AsyncLocalPath, AsyncStorePath):
    ...


@AsyncStorePath.register_subclass
class MemoryAsyncStorePath(wpath.AsyncWrappedPath, AsyncStorePath):
    ...


@AsyncStorePath.register_subclass
class HTTPAsyncStorePath(wpath.AsyncHTTPPath, AsyncStorePath):
    ...

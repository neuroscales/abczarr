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
class S3StorePath(wpath.WrappedS3Path, StorePath):
    ...


@StorePath.register_subclass
class GCSStorePath(wpath.WrappedGCSPath, StorePath):
    ...


@StorePath.register_subclass
class AzureStorePath(wpath.WrappedAzurePath, StorePath):
    ...


@StorePath.register_subclass
class LocalStorePath(wpath.WrappedLocalPath, StorePath):
    ...


@StorePath.register_subclass
class MemoryStorePath(wpath.WrappedMemoryPath, StorePath):
    ...


@StorePath.register_subclass
class HTTPStorePath(wpath.WrappedHTTPPath, StorePath):
    ...



class AsyncStorePath(wpath.AsyncWrappedPath):
    """Base class for zarr stores."""

    def __init__(self, *args, **kwargs) -> None:
        read_only = kwargs.pop("read_only", False)
        super().__init__(*args, **kwargs)
        self.read_only = read_only


@AsyncStorePath.register_subclass
class S3AsyncStorePath(wpath.AsyncWrappedS3Path, AsyncStorePath):
    ...


@AsyncStorePath.register_subclass
class GCSAsyncStorePath(wpath.AsyncWrappedGCSPath, AsyncStorePath):
    ...


@AsyncStorePath.register_subclass
class AzureAsyncStorePath(wpath.AsyncWrappedAzurePath, AsyncStorePath):
    ...


@AsyncStorePath.register_subclass
class LocalAsyncStorePath(wpath.AsyncWrappedLocalPath, AsyncStorePath):
    ...


@AsyncStorePath.register_subclass
class MemoryAsyncStorePath(wpath.AsyncWrappedMemoryPath, AsyncStorePath):
    ...


@AsyncStorePath.register_subclass
class HTTPAsyncStorePath(wpath.AsyncWrappedHTTPPath, AsyncStorePath):
    ...

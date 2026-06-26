__all__ = ["PathLike", "Path", "LocalPath", "UPath", "AnyPath"]

# stdlib
from os import PathLike
from pathlib import Path as LocalPath

# dependencies
import typing_extensions as _tx

# optionals
try:
    from upath import UPath
except ImportError:
    UPath = None
try:
    from cloudpathlib import AnyPath
except ImportError:
    AnyPath = None

Path = UPath or AnyPath or LocalPath

# typing
FilenameLike = _tx.Union[PathLike, str]
BinaryFileLike = _tx.Union[FilenameLike, _tx.BinaryIO]
TextFileLike = _tx.Union[FilenameLike, _tx.TextIO]
FileLike = _tx.Union[BinaryFileLike, TextFileLike]
BinaryContentLike = _tx.Union[bytes, bytearray, _tx.Iterable[bytes]]
TextContentLike = _tx.Union[str, _tx.Iterable[str]]
ContentLike = _tx.Union[BinaryContentLike, TextContentLike]
FileOrContentLike = _tx.Union[FileLike, ContentLike]
TextFileOrContentLike = _tx.Union[TextFileLike, TextContentLike]
BinaryFileOrContentLike = _tx.Union[BinaryFileLike, BinaryContentLike]

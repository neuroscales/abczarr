# TODO
# [ ] More fallback implementations for old pathlib
# [ ] Fix fallback implementation for match & fullmatch
# [ ] Ensure that the most appropriate PathLike class is used for the
#     given protocol (e.g. UPath for s3://, AnyPath for gs://, etc.)
# [ ] When a method takes a path argument (e.g., `copy`):
#     * If the target is a PathLike, ensure that the returned wrapped
#       type is the target's type.
#     * Else, if the argument is compatible with the current wrapped type,
#       return the current wrapped type.
#     * Else, ensure that a compatible PathLike type is used for the target.
# [ ] Methods documentation
# [ ] Check that examples make sense in docstrings
# [ ] Comparison operators

__all__ = ["PathLike", "Path", "LocalPath", "UPath", "AnyPath"]

# stdlib
import re
import shutil
from fnmatch import fnmatch, fnmatchcase
from os import PathLike
from pathlib import Path as LocalPath
from pathlib import PurePosixPath
from urllib.parse import urljoin

# dependencies
import typing_extensions as tx

# core
from .asyncutils import ensure_coroutine

# optionals
if tx.TYPE_CHECKING:
    # stdlib
    from pathlib.types import PathInfo

    # dependencies
    from anyio import Path as AsyncLocalPath
    from cloudpathlib import AnyPath
    from upath import UPath
    Path = UPath
    DefaultDriver = UPath
    DefaultAsyncDriver = AsyncLocalPath
else:
    try:
        from pathlib.types import PathInfo
    except ImportError:

        class PathInfo(tx.Protocol):
            """Protocol for pathlib.PathInfo."""

            def exists(self, *, follow_symlinks: bool = True) -> bool:
                ...

            def is_dir(self, *, follow_symlinks: bool = True) -> bool:
                ...

            def is_file(self, *, follow_symlinks: bool = True) -> bool:
                ...

            def is_symlink(self) -> bool:
                ...

    try:
        from upath import UPath
    except ImportError:
        UPath = None

    try:
        from cloudpathlib import AnyPath
    except ImportError:
        AnyPath = None

    try:
        from anyio import Path as AsyncLocalPath
    except ImportError:
        AsyncLocalPath = None

    DefaultDriver = Path = UPath or AnyPath or LocalPath
    DefaultAsyncDriver = AsyncLocalPath or None


# typing
PathOrStr = tx.Union[PathLike, str]
FilenameLike = PathOrStr
BinaryFileLike = tx.Union[FilenameLike, tx.BinaryIO]
TextFileLike = tx.Union[FilenameLike, tx.TextIO]
FileLike = tx.Union[BinaryFileLike, TextFileLike]
BinaryContentLike = tx.Union[bytes, bytearray, tx.Iterable[bytes]]
TextContentLike = tx.Union[str, tx.Iterable[str]]
ContentLike = tx.Union[BinaryContentLike, TextContentLike]
FileOrContentLike = tx.Union[FileLike, ContentLike]
TextFileOrContentLike = tx.Union[TextFileLike, TextContentLike]
BinaryFileOrContentLike = tx.Union[BinaryFileLike, BinaryContentLike]
AccessMode = tx.Literal[
    "r", "rb", "rt", "r+", "r+b", "r+t",
    "w", "wb", "wt", "w+", "w+b", "w+t",
    "x", "xb", "xt", "a", "ab", "at",
]
PATH_LIKE = tx.TypeVar("PATH_LIKE", bound=PathLike, default=PathLike)

# utils
RE_PROTOCOL = re.compile(r"^(\w+)://(.*)$")


def to_path(
    *path: tx.Unpack[PathOrStr],
    cls: tx.Type[PATH_LIKE] = Path,
    protocol: tx.Optional[str] = None,
    **kwargs
) -> PATH_LIKE:
    """Convert a string to a PathLike object.

    Parameters
    ----------
    *path : PathOrStr
        Path components to join together.
    cls : Type[PathLike], optional
        Class to use for the resulting PathLike object. Defaults to `Path`.
    protocol : str, optional
        Protocol to use for the resulting PathLike object.
    **kwargs
        Additional arguments to pass to the Path constructor.
    """

    # Parse protocol
    if protocol is None and path and isinstance(path[0], str):
        protocol_match = RE_PROTOCOL.match(str(path[0]))
        if protocol_match:
            protocol, path0 = protocol_match.groups()
            path = (protocol + "://", path0, *path[1:])
    elif protocol is not None:
        path = (f"{protocol}://", *path)

    if cls is LocalPath and protocol not in (None, "file", "local"):
        cls = FallbackPath

    # Do not pass protocol to LocalPath
    if cls is LocalPath:
        if protocol is not None:
            # Drop protocol prefix for local paths
            path = path[1:]

    return cls(*path, **kwargs)


# ======================================================================
#
#                   B A S E   P A T H   W R A P P E R
#
# ======================================================================


_DECORATOR = tx.Callable[[tx.Type[PathLike]], tx.Type[PathLike]]


class BaseWrappedPath(PathLike, tx.Generic[PATH_LIKE]):
    """
    A PathLike object that wraps another PathLike object.

    This includes some backward compatibility fixes for `pathib.Path`.
    """

    # Class attributes
    __slots__ = ("wrapped",)
    _DRIVER_REGISTRY: tx.ClassVar[tx.Dict[str, tx.Type[PathLike]]] = {}
    _SUBCLASS_REGISTRY: tx.ClassVar[tx.Dict[str, tx.Type["WrappedPath"]]] = {}

    # Attributes
    wrapped: PATH_LIKE

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # Polymorphic behavior
        obj = super().__new__(cls)
        obj.__init__(*args, **kwargs)
        subcls = cls.get_subclass(obj.protocol)
        return super().__new__(subcls)

    def __init__(
        self,
        path: PathOrStr,
        driver: tx.Union[str, tx.Type[PathLike], None] = None
    ) -> None:
        """
        Parameters
        ----------
        path : PathLike | str
            Path to wrap.
        driver : Type[PathLike] | str | None, optional
            The Path class to wrap.
            * If `None` and `path` is already a `PathLike`, keep it as is.
            * If `None` and `path` is a string, use the default `Path` class.
            * Else, use the provided `driver` class to wrap the path.
        """
        driver_cls = None
        if not isinstance(path, PathLike):
            driver_cls = self.get_driver(driver)
        elif driver is not None:
            driver_cls = self.get_driver(driver)
        if driver_cls:
            path = to_path(path, cls=driver_cls)
        self.wrapped: PATH_LIKE = path
        self.__post_init__()

    def __post_init__(self) -> None:
        """Post-initialization hook for subclasses."""
        if hasattr(self, "VALID_PROTOCOLS"):
            protocol = self.protocol
            if protocol not in self.VALID_PROTOCOLS:
                cls_name = type(self).__name__
                raise ValueError(
                    f"Protocol '{protocol}' is not supported for {cls_name}. "
                    f"Valid protocols are: {self.VALID_PROTOCOLS}"
                )

    @property
    def wrapped_type(self) -> tx.Type[PATH_LIKE]:
        """The type of the wrapped PathLike object."""
        return type(self.wrapped)

    def to(
        self, driver: tx.Union[str, PathLike, tx.Type[PathLike], None] = None
    ) -> tx.Self:
        """
        Convert wrapped path to a different PathLike type.
        """
        if driver is None:
            return self
        if isinstance(driver, WrappedPath):
            return self.to(driver.wrapped)
        if isinstance(driver, PathLike):
            return self.to(type(driver))
        return type(self)(self.wrapped, driver=driver)

    # --- Subclasses ---------------------------------------------------

    @tx.overload
    @classmethod
    def register_subclass(
        cls, path_cls: tx.Type[PathLike], *protocols: tx.Unpack[str]
    ) -> tx.Type[PathLike]:
        ...

    @tx.overload
    @classmethod
    def register_subclass(
        cls, *protocols: tx.Unpack[str]
    ) -> _DECORATOR:
        ...

    @classmethod
    def register_subclass(cls, *protocols):
        if not protocols or not isinstance(protocols[0], type):
            # Decorator factory usage
            def decorator(subcls: tx.Type[PathLike]) -> tx.Type[PathLike]:
                return cls.register_subclass(subcls, *protocols)
            return decorator
        else:
            # Decorator usage
            subcls, *protocols = protocols
            if not protocols:
                protocols = getattr(subcls, "VALID_PROTOCOLS", ())
            for protocol in protocols:
                protocol = protocol.lower()
                cls._SUBCLASS_REGISTRY[protocol] = subcls
            return subcls

    @classmethod
    def get_subclass(
        cls,
        protocol: tx.Optional[tx.Union[str, tx.Type[PathLike]]] = None,
        default: tx.Optional[tx.Type[PathLike]] = None
    ) -> tx.Type[PathLike]:
        """Get the subclass for the given protocol or class."""
        if protocol is None:
            protocol = default
        if protocol is None:
            protocol = WrappedPath

        if isinstance(protocol, str):
            protocol = protocol.lower()
            return cls._SUBCLASS_REGISTRY.get(protocol, default or WrappedPath)

        if issubclass(protocol, cls):
            return protocol

        raise TypeError(
            f"Protocol must be a string or a subclass of {cls}, "
            f"got {protocol}"
        )

    # --- Drivers ------------------------------------------------------

    @tx.overload
    @classmethod
    def register_driver(
        cls, path_cls: tx.Type[PathLike], *names: tx.Unpack[str]
    ) -> tx.Type[PathLike]:
        ...

    @tx.overload
    @classmethod
    def register_driver(cls, *names: tx.Unpack[str]) -> _DECORATOR:
        ...

    @classmethod
    def register_driver(cls, *names):
        if not names or not isinstance(names[0], type):
            # Decorator factory usage
            def decorator(subcls: tx.Type[PathLike]) -> tx.Type[PathLike]:
                return cls.register_subclass(subcls, *names)
            return decorator
        else:
            # Decorator usage
            driver, *names = names
            if not names:
                names = (driver.__name__,)
            for name in names:
                name = name.lower()
                cls._DRIVER_REGISTRY[name] = driver
            return driver

    @classmethod
    def get_driver(
        cls,
        driver: tx.Optional[tx.Union[str, tx.Type[PathLike]]] = None,
        default: tx.Optional[tx.Type[PathLike]] = None
    ) -> tx.Type[PathLike]:
        """Get the driver class for the given driver name or class."""
        if driver is None:
            driver = default
        if driver is None:
            driver = DefaultDriver

        if isinstance(driver, str):
            driver = driver.lower()
            if driver not in cls._DRIVER_REGISTRY:
                if default is not None:
                    return default
                raise ValueError(f"Unknown driver: {driver}")
            return cls._DRIVER_REGISTRY[driver]

        if issubclass(driver, PathLike):
            return driver

        raise TypeError(
            f"Driver must be a string or a subclass of PathLike, "
            f"got {driver}"
        )

    # --- Repr ---------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self.wrapped)!r})"

    def __str__(self) -> str:
        return str(self.wrapped)

    def __bytes__(self) -> bytes:
        if hasattr(self.wrapped, "__bytes__"):
            return self.wrapped.__bytes__()
        return str(self).encode()

    def __hash__(self) -> int:
        if hasattr(self.wrapped, ".__hash__"):
            return self.wrapped.__hash__()
        return hash(str(self))

    # --- PathLike -----------------------------------------------------

    def __fspath__(self) -> str:
        return getattr(self.wrapped, "__fspath__", "__str__")()

    # --- Operators ----------------------------------------------------

    def __truediv__(self, other: PathOrStr) -> str:
        return type(self)(self.wrapped / other)

    def __rtruediv__(self, other: PathOrStr) -> str:
        return type(self)(other / self.wrapped)

    # --- Properties ---------------------------------------------------

    @property
    def parts(self) -> tuple:
        """
        A tuple giving access to the path's various components:

        !!! example
            === "Windows"
                ```pycon
                >>> PureWindowsPath('C:/Here/There/Everywhere/').parts
                ('C:\\', 'Here', 'There', 'Everywhere')
                ```
            === "Posix"
                ```pycon
                >>> PurePosixPath('/here/there/everywhere').parts
                ('/', 'here', 'there', 'everywhere')
                ```
            === "Cloud"
                ```pycon
                >>> Path('s3://bucket-name/here/there/everywhere').parts
                ('bucket-name/', 'here', 'there', 'everywhere')
                >>> Path('gs://bucket-name/here/there/everywhere').parts
                ('bucket-name/', 'here', 'there', 'everywhere')
                >>> Path('az://bucket-name/here/there/everywhere').parts
                ('bucket-name/', 'here', 'there', 'everywhere')
                ```
            === "Memory"
                ```pycon
                >>> Path('memory://here/there/everywhere').parts
                ('/', 'here', 'there', 'everywhere')
                ```
            === "Others"
                ```pycon
                >>> Path('ftp://host/here/there/everywhere').parts
                ('/', 'here', 'there', 'everywhere')
                >>> Path('sftp://host/here/there/everywhere').parts
                ('/', 'here', 'there', 'everywhere')
                >>> Path('ssh://user@host:/here/there/everywhere').parts
                ('/', 'here', 'there', 'everywhere')
                >>> Path('github://org:repo@sha/here/there.yml').parts
                ('here', 'there.yml')
                ```
        """
        return self.wrapped.parts

    @property
    def drive(self) -> str:
        """
        A string representing the drive letter or name, if any:

        !!! example
            === "Windows"
                ```pycon
                >>> PureWindowsPath('C:/Here/There/Everywhere/').drive
                'C:'
                >>> PureWindowsPath('/Here/There/Everywhere/').drive
                ''
                >>> # UNC shares are also considered drives:
                >>> PureWindowsPath('//host/share/here_there_everywhere').drive
                '\\\\host\\share'
                ```
            === "Posix"
                ```pycon
                >>> PurePosixPath('/here/there/everywhere').drive
                ''
                ```
            === "Cloud"
                ```pycon
                >>> Path('s3://bucket-name/here/there/everywhere').drive
                'bucket-name'
                >>> Path('gs://bucket-name/here/there/everywhere').drive
                'bucket-name'
                >>> Path('az://bucket-name/here/there/everywhere').drive
                'bucket-name'
                ```
            === "Memory"
                ```pycon
                >>> Path('memory://here/there/everywhere').drive
                ''
                ```
            === "Others"
                ```pycon
                >>> Path('ftp://host/here/there/everywhere').drive
                ''
                >>> Path('sftp://host/here/there/everywhere').drive
                ''
                >>> Path('ssh://user@host:/here/there/everywhere').drive
                ''
                >>> Path('github://org:repo@sha/here/there.yml').drive
                ''
                ```
        """
        return getattr(self.wrapped, "drive", None) or ""

    @property
    def root(self) -> str:
        """
        A string representing the (local or global) root of the path, if any:

        !!! example
            === "Windows"
                ```pycon
                >>> PureWindowsPath('C:/Here/There/Everywhere/').root
                '\\'
                >>> PureWindowsPath('C:Here/There/Everywhere/').root
                ''
                >>> # UNC shares always have a root:
                >>> PureWindowsPath('//host/share/here_there_everywhere').root
                '\\'
                ```
            === "Posix"
                ```pycon
                >>> PurePosixPath('here/there/everywhere').root
                ''
                >>> PurePosixPath('/here/there/everywhere').root
                '/'
                >>> # If the path starts with more than two successive
                >>> # slashes, they are collapsed:
                >>> PurePosixPath('//etc').root
                '//'
                >>> PurePosixPath('///etc').root
                '/'
                >>> PurePosixPath('////etc').root
                '/'
                ```
            === "Cloud"
                ```pycon
                >>> Path('s3://bucket-name/here/there/everywhere').root
                '/'
                >>> Path('gs://bucket-name/here/there/everywhere').root
                '/'
                >>> Path('az://bucket-name/here/there/everywhere').root
                '/'
                ```
            === "Memory"
                ```pycon
                >>> Path('memory://here/there/everywhere').root
                '/'
                ```
            === "Others"
                ```pycon
                >>> Path('ftp://host/here/there/everywhere').root
                '/'
                >>> Path('sftp://host/here/there/everywhere').root
                '/'
                >>> Path('ssh://user@host:/here/there/everywhere').root
                '/'
                >>> Path('github://org:repo@sha/here/there.yml').root
                ''
                ```
        """
        return getattr(self.wrapped, "root", None) or ""

    @property
    def anchor(self) -> str:
        """
        The concatenation of the drive and root:

        !!! example
            === "Windows"
                ```pycon
                >>> PureWindowsPath('C:/Here/There/Everywhere/').anchor
                'C:\\'
                >>> PureWindowsPath('C:Here/There/Everywhere/').anchor
                'C:'
                >>> PureWindowsPath('//host/share/here_there').anchor
                '\\\\host\\share\\'
                ```
            === "Posix"
                ```pycon
                >>> PurePosixPath('here/there/everywhere').anchor
                ''
                >>> PurePosixPath('/here/there/everywhere').anchor
                '/'
                >>> # If the path starts with more than two successive
                >>> # slashes, they are collapsed:
                >>> PurePosixPath('//etc').anchor
                '//'
                >>> PurePosixPath('///etc').anchor
                '/'
                >>> PurePosixPath('////etc').anchor
                '/'
                ```
            === "Cloud"
                ```pycon
                >>> Path('s3://bucket-name/here/there/everywhere').anchor
                'bucket-name//'
                >>> Path('gs://bucket-name/here/there/everywhere').anchor
                'bucket-name//'
                >>> Path('az://bucket-name/here/there/everywhere').anchor
                'bucket-name//'
                ```
            === "Memory"
                ```pycon
                >>> Path('memory://here/there/everywhere').anchor
                '/'
                ```
            === "Others"
                ```pycon
                >>> Path('ftp://host/here/there/everywhere').anchor
                '/'
                >>> Path('sftp://host/here/there/everywhere').anchor
                '/'
                >>> Path('ssh://user@host:/here/there/everywhere').anchor
                '/'
                >>> Path('github://org:repo@sha/here/there.yml').anchor
                ''
                ```
        """
        return getattr(self, "anchor", self.drive + self.root)

    @property
    def parents(self) -> tx.Self:
        """
        An immutable sequence providing access to the logical ancestors
        of the path:

        !!! example
            === "Windows"
                ```pycon
                >>> p = PureWindowsPath('C:/Here/There/Everywhere/')
                >>> p.parents[0]
                PureWindowsPath('C:\\Here\\There')
                >>> p.parents[1]
                PureWindowsPath('C:\\Here')
                >>> p.parents[2]
                PureWindowsPath('C:\\')
                ```
            === "Posix"
                ```pycon
                >>> p = PurePosixPath('/here/there/everywhere')
                >>> p.parents[0]
                PurePosixPath('/here/there')
                >>> p.parents[1]
                PurePosixPath('/here')
                >>> p.parents[2]
                PurePosixPath('/')
                ```
            === "Cloud"
                ```pycon
                >>> p = Path('s3://bucket-name/here/there/everywhere')
                >>> p.parents[0]
                Path('s3://bucket-name/here/there')
                >>> p.parents[1]
                Path('s3://bucket-name/here')
                >>> p.parents[2]
                Path('s3://bucket-name/')
                >>> # Same behavior for Azure (az://) and Google (gs://)
                ```
            === "Memory"
                ```pycon
                >>> p = Path('memory://here/there/everywhere')
                >>> p.parents[0]
                Path('memory://here/there')
                >>> p.parents[1]
                Path('memory://here')
                >>> p.parents[2]
                Path('memory:///')
                ```
        """
        return self.wrapped.parents

    @property
    def parent(self) -> tx.Self:
        """
        The logical parent of the path:

        !!! example
            === "Windows"
                ```pycon
                >>> PureWindowsPath('C:/Here/There/Everywhere/').parent
                PureWindowsPath('C:\\Here\\There')
                ```
            === "Posix"
                ```pycon
                >>> PurePosixPath('/here/there/everywhere').parent
                PurePosixPath('/here/there')
                ```
            === "Cloud"
                ```pycon
                >>> Path('s3://bucket-name/here/there/everywhere').parent
                Path('s3://bucket-name/here/there')
                ```
            === "Memory"
                ```pycon
                >>> Path('memory://here/there/everywhere').parent
                Path('memory://here/there')
                ```

        !!! warning "You cannot go past an anchor, or empty path"
            ```pycon
            >>> Path('/').parent
            Path('/')
            >>> Path('.').parent
            Path('.')
            ```

        !!! note "This is a purely lexical operation"
            Hence the following behaviour:

            ```pycon
            >>> Path('foo/..').parent
            Path('foo')
            ```

            If you want to walk an arbitrary filesystem path upwards,
            it is recommended to first call `Path.resolve()` so as to
            resolve symlinks and eliminate ".." components.

        """
        return self.wrapped.parent

    @property
    def name(self) -> str:
        """
        The final path component, if any:

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere').name
            'everywhere'
            ```
        """
        return self.wrapped.name

    @property
    def suffix(self) -> str:
        """
        The last dot-separated portion of the final component, if any:

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere.txt').suffix
            '.txt'
            >>> Path('/here/there/everywhere.tar.gz').suffix
            '.gz'
            >>> Path('/here/there/everywhere').suffix
            ''
            ```
        """
        return self.wrapped.suffix

    @property
    def suffixes(self) -> tx.List[str]:
        """
        A list of the path's suffixes, often called file extensions:

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere.tar.gz').suffixes
            ['.tar', '.gz']
            >>> Path('/here/there/everywhere.txt').suffixes
            ['.txt']
            >>> Path('/here/there/everywhere').suffixes
            []
            ```
        """
        return self.wrapped.suffixes

    @property
    def stem(self) -> str:
        """
        The final path component, without its suffix:

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere.tar.gz').stem
            'everywhere.tar'
            >>> Path('/here/there/everywhere.txt').stem
            'everywhere'
            >>> Path('/here/there/everywhere').stem
            'everywhere'
            ```
        """
        return self.wrapped.stem

    # --- Methods ------------------------------------------------------

    def as_posix(self) -> str:
        """
        Return a string representation of the path with forward slashes (`/`):

        !!! example
            ```pycon
            >>> PureWindowsPath('C:\\Here\\There\\Everywhere').as_posix()
            'C:/Here/There/Everywhere'
            >>> PurePosixPath('/here/there/everywhere').as_posix()
            '/here/there/everywhere'
            >>> PurePosixPath('s3://bucket-name/here/there/everywhere').as_posix()
            's3://bucket-name/here/there/everywhere'
            ```
        """
        return self.wrapped.as_posix()

    # --- Methods ------------------------------------------------------

    def is_absolute(self) -> bool:
        """
        Return whether the path is absolute or not.

        A path is considered absolute if it has both a root and (if the
        flavour allows) a drive:

        !!! example
            === "Windows"
                ```pycon
                >>> PureWindowsPath('C:/Here/There/Everywhere').is_absolute()
                True
                >>> PureWindowsPath('C:Here/There/Everywhere').is_absolute()
                False
                >>> PureWindowsPath('//host/share/here_there).is_absolute()
                True
                ```
            === "Posix"
                ```pycon
                >>> PurePosixPath('/here/there/everywhere').is_absolute()
                True
                >>> PurePosixPath('here/there/everywhere').is_absolute()
                False
                ```
            === "Cloud"
                ```pycon
                >>> Path('s3://bucket-name/here/there').is_absolute()
                True
                >>> Path('gs://bucket-name/here/there').is_absolute()
                True
                >>> Path('az://bucket-name/here/there').is_absolute()
                True
                ```
            === "Memory"
                ```pycon
                >>> Path('memory://here/there/everywhere').is_absolute()
                True
                ```
            === "Others"
                ```pycon
                >>> Path('ftp://host/here/there/everywhere').is_absolute()
                True
                >>> Path('sftp://host/here/there/everywhere').is_absolute()
                True
                >>> Path('ssh://user@host:/here/there').is_absolute()
                True
        """
        return self.wrapped.is_absolute()

    def is_relative_to(self, other: PathOrStr) -> bool:
        """
        Return whether the path is relative to another path.

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere').is_relative_to('/here')
            True
            >>> Path('/here/there/everywhere').is_relative_to('/there')
            False
            ```

        !!! warning
            This method is string-based; it neither accesses the
            filesystem nor treats ".." segments specially.
            The following code is equivalent:

            ```pycon
            >>> u = Path('/usr')
            >>> u == p or u in p.parents
            False
            ```
        """
        return self.wrapped.is_relative_to(other)

    def joinpath(self, *pathsegments: tx.Unpack[PathOrStr]) -> tx.Self:
        """
        Calling this method is equivalent to combining the path with each
        of the given `pathsegments` in turn:

        !!! example
            ```pycon
            >>> Path('/here/there').joinpath('everywhere')
            Path('/here/there/everywhere')
            >>> Path('/here/there').joinpath('everywhere', 'and', 'beyond')
            Path('/here/there/everywhere/and/beyond')
            ```
        """
        return type(self)(self.wrapped.joinpath(*pathsegments))

    def full_match(
        self, pattern: str,
        *, case_sensitive: tx.Optional[bool] = None
    ) -> bool:
        """
        Match this path against the provided glob-style pattern.

        Return True if matching is successful, False otherwise.

        !!! example
            ```pycon
            >>> Path('a/b.py').full_match('a/*.py')
            True
            >>> Path('a/b.py').full_match('*.py')
            False
            >>> Path('/a/b/c.py').full_match('/a/**')
            True
            >>> Path('/a/b/c.py').full_match('**/*.py')
            True
            ```
        """
        pattern = str(pattern)
        if hasattr(self.wrapped, "fullmatch"):
            kwargs = {}
            if case_sensitive is not None:
                kwargs["case_sensitive"] = case_sensitive
            return self.wrapped.fullmatch(pattern, **kwargs)
        match = fnmatchcase if case_sensitive else fnmatch
        return match(str(self.wrapped), pattern)  # FIXME

    def match(
        self, pattern: str,
        *, case_sensitive: tx.Optional[bool] = None
    ) -> bool:
        """
        Match this path against the provided non-recursive glob-style pattern.
        Return True if matching is successful, False otherwise.

        This method is similar to full_match(), but:

        * empty patterns are not allowed (`ValueError` is raised);
        * the recursive wildcard `"**"` is not supported (it acts like
          non-recursive `"*"`);
        * if a relative pattern is provided, then matching is done from
          the right.

        !!! example
            ```pycon
            >>> Path('a/b.py').match('*.py')
            True
            >>> Path('/a/b/c.py').match('b/*.py')
            True
            >>> Path('/a/b/c.py').match('a/*.py')
            False
            ```
        """
        pattern = str(pattern)
        if hasattr(self.wrapped, "match"):
            kwargs = {}
            if case_sensitive is not None:
                kwargs["case_sensitive"] = case_sensitive
            return self.wrapped.match(pattern, **kwargs)
        match = fnmatchcase if case_sensitive else fnmatch
        return match(str(self.wrapped), pattern)  # FIXME

    def relative_to(self, other: PathOrStr, walk_up: bool = False) -> tx.Self:
        """
        Compute a version of this path relative to the path represented
        by other. If it is impossible, `ValueError` is raised.

        !!! example
            ```pycon
            >>> p = Path('/etc/passwd')
            >>> p.relative_to('/')
            Path('etc/passwd')
            >>> p.relative_to('/etc')
            Path('passwd')
            >>> p.relative_to('/usr')
            ValueError: '/etc/passwd' is not in the subpath of '/usr' OR
            one path is relative and the other is absolute.
            ```

        !!! note
            When `walk_up` is false (the default), the path must start with
            other. When the argument is true, `..` entries may be added to
            form the relative path. In all other cases, such as the paths
            referencing different drives, `ValueError` is raised.

            ```pycon
            >>> p.relative_to('/usr', walk_up=True)
            Path('../etc/passwd')
            >>> p.relative_to('foo', walk_up=True)
            ValueError: '/etc/passwd' is not on the same drive as 'foo' OR
            one path is relative and the other is absolute.
            ```

        !!! warning
            This function works with strings. It does not check or
            access the underlying file structure. This can impact the
            `walk_up` option as it assumes that no symlinks are present
            in the path; call `resolve()` first if necessary to resolve
            symlinks.
        """
        kwargs = {"walk_up": True} if walk_up else {}
        return type(self)(self.wrapped.relative_to(other, **kwargs))

    def with_name(self, name: str) -> tx.Self:
        """
        Return a new path with the `name` changed.

        If the original path does not have a name, `ValueError` is raised

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere').with_name('elsewhere')
            Path('/here/there/elsewhere')
            ```
        """
        return type(self)(self.wrapped.with_name(name))

    def with_stem(self, stem: str) -> tx.Self:
        """
        Return a new path with the `stem` changed.

        If the original path does not have a stem, `ValueError` is raised

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere.txt').with_stem('elsewhere')
            Path('/here/there/elsewhere.txt')
            ```
        """
        return type(self)(self.wrapped.with_stem(stem))

    def with_suffix(self, suffix: str) -> tx.Self:
        """
        Return a new path with the `suffix` changed.

        If the original path does not have a suffix, `ValueError` is raised

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere.txt').with_suffix('.md')
            Path('/here/there/everywhere.md')
            ```
        """
        return type(self)(self.wrapped.with_suffix(suffix))

    def with_segments(self, *segments: tx.Unpack[PathOrStr]) -> tx.Self:
        """
        Create a new path object of the same type by combining the given
        `segments`.

        !!! note
            This method is called whenever a derivative path is created,
            such as from `parent` and `relative_to()`. Subclasses may
            override this method to pass information to derivative paths.

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere').with_segments('elsewhere')
            Path('/here/there/elsewhere')
            >>> Path('/here/there/everywhere').with_segments(
            >>>     'elsewhere', 'and', 'beyond')
            Path('/here/elsewhere/and/beyond')
            ```
        """
        return type(self)(self.wrapped.with_segments(*segments))

    # --- Concrete paths -----------------------------------------------

    # >> Parsing URIs

    def as_uri(self) -> str:
        """
        Return a string representation of the path as a URI.

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere').as_uri()
            'file:///here/there/everywhere'
            >>> Path('s3://bucket-name/here/there/everywhere').as_uri()
            's3://bucket-name/here/there/everywhere'
            >>> Path('gs://bucket-name/here/there/everywhere').as_uri()
            'gs://bucket-name/here/there/everywhere'
            >>> Path('az://bucket-name/here/there/everywhere').as_uri()
            'az://bucket-name/here/there/everywhere'
            >>> Path('memory://here/there/everywhere').as_uri()
            'memory://here/there/everywhere'
            ```
        """
        return self.wrapped.as_uri()

    # >> Parsing URIs

    @classmethod
    def from_uri(
        cls, uri: PathOrStr,
        *, driver: tx.Union[tx.Type[PathLike], str, None] = None
    ) -> tx.Self:
        """
        Create a new path object of the same type from a URI.

        !!! example
            ```pycon
            >>> Path.from_uri('file://here/there/everywhere')
            Path('file://here/there/everywhere')
            >>> Path.from_uri('s3://bucket-name/here/there/everywhere')
            Path('s3://bucket-name/here/there/everywhere')
            >>> Path.from_uri('gs://bucket-name/here/there/everywhere')
            Path('gs://bucket-name/here/there/everywhere')
            >>> Path.from_uri('az://bucket-name/here/there/everywhere')
            Path('az://bucket-name/here/there/everywhere')
            >>> Path.from_uri('memory://here/there/everywhere')
            Path('memory://here/there/everywhere')
            ```
        """
        driver_cls = cls.get_driver(driver)
        return cls(driver_cls.from_uri(uri))

    # >> Reading directories

    def glob(
        self,
        pattern: PathOrStr,
        *,
        case_sensitive: tx.Optional[bool] = None,
        recurse_symlinks: bool = False
    ) -> tx.Iterator[tx.Self]:
        kwargs = {}
        if case_sensitive is not None:
            kwargs['case_sensitive'] = case_sensitive
        if recurse_symlinks:
            kwargs['recurse_symlinks'] = recurse_symlinks
        pattern = str(pattern)
        for p in self.wrapped.glob(pattern, **kwargs):
            yield type(self)(p)

    def rglob(
        self,
        pattern: PathOrStr,
        *,
        case_sensitive: tx.Optional[bool] = None,
        recurse_symlinks: bool = False
    ) -> tx.Iterator[tx.Self]:
        kwargs = {}
        if case_sensitive is not None:
            kwargs['case_sensitive'] = case_sensitive
        if recurse_symlinks:
            kwargs['recurse_symlinks'] = recurse_symlinks
        pattern = str(pattern)
        for p in self.wrapped.rglob(pattern, **kwargs):
            yield type(self)(p)

    # >> Querying File status

    @property
    def info(self) -> PathInfo:
        return self.wrapped.info

    # --- UPath --------------------------------------------------------

    @property
    def protocol(self) -> str:
        """
        The protocol for the path.

        !!! example
            ```pycon
            >>> WrappedPath("s3://my-bucket/path/to/file.txt").protocol
            's3'
            ```
        """
        return getattr(self.wrapped, "protocol", None) or ""

    @property
    def path(self) -> str:
        """
        The path, within its file system (stripped of protocol).

        This property returns the path suitable for use with a fsspec
        filesystem.
        """
        return getattr(self.wrapped, "path", str(self.wrapped))

    def joinuri(self, path: PathOrStr) -> str:
        """
        Join with urljoin behavior.

        !!! example
            ```pycon
            >>> p = WrappedPath("https://example.com/dir/subdir/")
            >>> p.joinuri("file.txt")
            WrappedPath('https://example.com/dir/subdir/file.txt')
            >>> p.joinuri("/anotherdir/otherfile.txt")
            WrappedPath('https://example.com/anotherdir/otherfile.txt')
            >>> p.joinuri("memory:///foo/bar.txt"
            WrappedPath('memory:///foo/bar.txt')
            ```
        """
        if hasattr(self.wrapped, "joinuri"):
            return self.wrapped.joinuri(path)
        cls = type(self)
        return cls(urljoin(self.as_uri(), str(path)))

    # --- CloudPath ----------------------------------------------------

    @property
    def cloud_prefix(self) -> str:
        """
        The cloud prefix for the path.
        The cloud prefix is the protocol followed by `"://"`.

        !!! example
            ```pycon
            >>> WrappedPath("s3://my-bucket/path/to/file.txt").cloud_prefix
            's3://'
            ```
        """
        return getattr(
            self.wrapped,
            "cloud_prefix",
            (self.protocol or "file") + "://"
        )

    @property
    def fspath(self) -> str:
        return str(self)

    def as_url(
        self, presign: bool = False, expire_seconds: int = 60 * 60
    ) -> str:
        """
        Return a URL representation of the path.

        If the path is a cloud path, this method will return a presigned
        URL if `presign` is True. The presigned URL will expire after
        `expire_seconds` seconds.

        !!! example
            ```pycon
            >>> WrappedPath("s3://my-bucket/path/to/file.txt").as_url()
            'https://my-bucket.s3.amazonaws.com/path/to/file.txt'
            >>> WrappedPath("s3://my-bucket/path/to/file.txt").as_url(presign=True)
            'https://my-bucket.s3.amazonaws.com/path/to/file.txt?AWSAccessKeyId=...&Expires=...&Signature=...'
            ```
        """
        if hasattr(self.wrapped, "as_url"):
            return self.wrapped.as_url(
                presign=presign, expire_seconds=expire_seconds
            )
        return self.as_uri()


# ======================================================================
#
#                       P A T H   W R A P P E R
#
# ======================================================================


class WrappedPath(BaseWrappedPath[PATH_LIKE]):

    # --- Concrete paths -----------------------------------------------

    # >> Expanding Paths

    @classmethod
    def home(
        cls,
        *, driver: tx.Union[tx.Type[PathLike], str, None] = None
    ) -> tx.Self:
        """
        Return a new path object representing the user's home directory
        (as returned by `os.path.expanduser()` with ~ construct).

        If the home directory cannot be resolved, `RuntimeError` is raised.

        !!! example
            ```pycon
            >>> Path.home()
            Path('/home/user')
            ```
        """
        driver_cls = cls.get_driver(driver)
        return cls(driver_cls.home())

    def expanduser(self) -> tx.Self:
        """
        Return a new path with expanded `~` and `~user` constructs, as
        returned by `os.path.expanduser()`.

        If a home directory cannot be resolved, `RuntimeError` is raised.

        !!! example
            ```pycon
            >>> Path('~/here/there/everywhere').expanduser()
            Path('/home/user/here/there/everywhere')
            ```
        """
        return type(self)(self.wrapped.expanduser())

    @classmethod
    def cwd(
        cls,
        *, driver: tx.Union[tx.Type[PathLike], str, None] = None
    ) -> tx.Self:
        """
        Return a new path object representing the current directory
        (as returned by `os.getcwd()`):

        !!! example
            ```pycon
            >>> Path.cwd()
            Path('/home/user/workspace')
            ```
        """
        driver_cls = cls.get_driver(driver)
        return cls(driver_cls.cwd())

    def absolute(self) -> tx.Self:
        """
        Make the path absolute, without normalization or resolving symlinks.
        Returns a new path object.

        !!! example
            ```pycon
            >>> Path('here/there/everywhere').absolute()
            Path('/home/user/workspace/here/there/everywhere')
            ```
        """
        return type(self)(self.wrapped.absolute())

    def resolve(self, strict: bool = False) -> tx.Self:
        """
        Make the path absolute, resolving any symlinks.
        A new path object is returned.

        !!! example
            ```pycon
            >>> Path('here/there/everywhere').resolve()
            Path('/home/user/workspace/here/there/everywhere')
            ```

        !!! note
            `..` components are also eliminated
            (this is the only method to do so)
            ```pycon
            >>> Path('there/../elsewhere.').resolve()
            Path('/home/user/workspace/here/elsewhere')
            ```

        !!! warning
            If a path does not exist or a symlink loop is encountered, and
            `strict=True`, `OSError` is raised. If `strict=False`, the path
            is resolved as far as possible and any remainder is appended
            without checking whether it exists.
        """
        kwargs = {"strict": True} if strict else {}
        return type(self)(self.wrapped.resolve(**kwargs))

    def readlink(self) -> tx.Self:
        """
        Return the path to which the symbolic link points
        (as returned by `os.readlink()`).

        !!! example
            ```pycon
            >>> Path('link_to_file').readlink()
            Path('target_file')
            ```
        """
        return type(self)(self.wrapped.readlink())

    # >> Querying File status

    def stat(self, *, follow_symlinks: bool = True) -> tx.Any:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        return self.wrapped.stat(**kwargs)

    def lstat(self) -> tx.Any:
        return self.wrapped.lstat()

    def exists(self, *, follow_symlinks: bool = True) -> bool:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        return self.wrapped.exists(**kwargs)

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        return self.wrapped.is_file(**kwargs)

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        return self.wrapped.is_dir(**kwargs)

    def is_symlink(self) -> bool:
        return self.wrapped.is_symlink()

    def is_mount(self) -> bool:
        return self.wrapped.is_mount()

    def is_socket(self) -> bool:
        return self.wrapped.is_socket()

    def is_fifo(self) -> bool:
        return self.wrapped.is_fifo()

    def is_block_device(self) -> bool:
        return self.wrapped.is_block_device()

    def is_char_device(self) -> bool:
        return self.wrapped.is_char_device()

    def samefile(self, other: PathOrStr) -> bool:
        return self.wrapped.samefile(other)

    # >> Reading & Writing Files

    def open(self, mode: AccessMode = "r", **kwargs) -> tx.IO:
        return self.wrapped.open(mode, **kwargs)

    def read_text(self, *a, **k) -> str:
        if hasattr(self.wrapped, "read_text"):
            return self.wrapped.read_text(*a, **k)
        return self.read_bytes().decode(**k)

    def read_bytes(self, *a, **k) -> bytes:
        if hasattr(self.wrapped, "read_bytes"):
            return self.wrapped.read_bytes(*a, **k)
        with self.open("rb", **k) as f:
            return f.read()

    def write_text(self, data: str, *a, **k) -> int:
        if hasattr(self.wrapped, "write_text"):
            return self.wrapped.write_text(data, *a, **k)
        return self.write_bytes(data.encode(**k))

    def write_bytes(self, data: bytes) -> int:
        if hasattr(self.wrapped, "write_bytes"):
            return self.wrapped.write_bytes(data)
        with self.open("wb") as f:
            return f.write(data)

    # >> Reading Directories

    def iterdir(self) -> tx.Iterator[tx.Self]:
        for p in self.wrapped.iterdir():
            yield type(self)(p)

    def walk(
        self,
        top_down: bool = True,
        on_error: tx.Optional[tx.Callable[[OSError], tx.Any]] = None,
        follow_symlinks: bool = False
    ) -> tx.Iterator[tx.Self]:
        kwargs = {"follow_symlinks": True} if follow_symlinks else {}
        for p in self.wrapped.walk(
            top_down=top_down, on_error=on_error, **kwargs
        ):
            yield type(self)(p)

    # >> Creating Files & Dirs

    def touch(self, mode: int = 0o666, exist_ok: bool = True) -> tx.Self:
        self.wrapped.touch(mode=mode, exist_ok=exist_ok)
        return self

    def mkdir(
        self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> tx.Self:
        kwargs = {"exist_ok": True} if exist_ok else {}
        self.wrapped.mkdir(mode=mode, parents=parents, **kwargs)
        return self

    def symlink_to(
            self, target: PathOrStr, target_is_directory: bool = False
    ) -> tx.Self:
        kwargs = {"target_is_directory": True} if target_is_directory else {}
        self.wrapped.symlink_to(target, **kwargs)
        return self

    def hardlink_to(self, target: PathOrStr) -> tx.Self:
        self.wrapped.hardlink_to(target)
        return self

    # >> Copying & Moving

    def copy(
        self, target: PathOrStr,
        *, follow_symlinks: bool = True, preserve_metadata: bool = False
    ) -> tx.Self:
        cls = type(self)
        if hasattr(self.wrapped, "copy"):
            kwargs = {}
            if not follow_symlinks:
                kwargs = {"follow_symlinks": False}
            if preserve_metadata:
                kwargs = {"preserve_metadata": True}
            return cls(self.wrapped.copy(target, **kwargs))
        return self._copy_fallback(
            target,
            follow_symlinks=follow_symlinks,
            preserve_metadata=preserve_metadata
        )

    def _copy_fallback(
        self, target: PathOrStr,
        *, follow_symlinks: bool = True, preserve_metadata: bool = False
    ) -> tx.Self:
        cls = type(self)
        copy = shutil.copy2 if preserve_metadata else shutil.copy
        if self.is_dir():
            kwargs = {"symlinks": True} if not follow_symlinks else {}
            kwargs["copy_function"] = copy
            shutil.copytree(self.path, cls(target).path, **kwargs)
            return cls(target)
        if self.is_file():
            kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
            copy(self.path, cls(target).path, **kwargs)
            return cls(target)
        raise FileNotFoundError(
            f"Cannot copy {self.path}: not a file or directory"
        )

    def copy_into(
        self, target_dir: PathOrStr,
        *, follow_symlinks: bool = True, preserve_metadata: bool = False
    ) -> tx.Self:
        cls = type(self)
        if hasattr(self.wrapped, "copy_into"):
            kwargs = {}
            if not follow_symlinks:
                kwargs = {"follow_symlinks": False}
            if preserve_metadata:
                kwargs = {"preserve_metadata": True}
            return cls(self.wrapped.copy_into(target_dir, **kwargs))
        return self.copy(
            cls(target_dir) / self.name,
            follow_symlinks=follow_symlinks,
            preserve_metadata=preserve_metadata
        )

    def rename(self, target: PathOrStr) -> tx.Self:
        self.wrapped.rename(target)
        return type(self)(target)

    def replace(self, target: PathOrStr) -> tx.Self:
        self.wrapped.replace(target)
        return type(self)(target)

    def move(self, target: PathOrStr) -> tx.Self:
        if hasattr(self.wrapped, "move"):
            return type(self)(self.wrapped.move(target))
        return self._move_fallback(target)

    def _move_fallback(self, target: PathOrStr) -> tx.Self:
        try:
            return self.replace(target)
        except OSError:
            obj = self.copy(target, preserve_metadata=True)
            self.unlink()
            return obj

    def move_into(self, target_dir: PathOrStr) -> tx.Self:
        cls = type(self)
        if hasattr(self.wrapped, "move_into"):
            return cls(self.wrapped.move_into(target_dir))
        return self.move(cls(target_dir) / self.name)

    def unlink(self, *, missing_ok: bool = False) -> None:
        kwargs = {"missing_ok": True} if missing_ok else {}
        self.wrapped.unlink(**kwargs)

    def rmdir(self, recursive: bool = True) -> None:
        if isinstance(self.wrapped, LocalPath):
            if recursive:
                shutil.rmtree(self.path)
            else:
                self.wrapped.rmdir()
        else:
            self.wrapped.rmdir(recursive)

    # >> Permission & Owner

    def owner(self, *, follow_symlinks: bool = True) -> str:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        return self.wrapped.owner(**kwargs)

    def group(self, *, follow_symlinks: bool = True) -> str:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        return self.wrapped.group(**kwargs)

    def chmod(self, *, follow_symlinks: bool = True) -> None:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        return self.wrapped.chmod(**kwargs)

    def lchmod(self) -> None:
        return self.wrapped.lchmod()


if UPath:
    WrappedPath.register_driver(UPath)
if AnyPath:
    names = ("anypath", "cloudpath", "cloudpathlib")
    WrappedPath.register_driver(AnyPath, *names)
if LocalPath:
    names = ("local", "pathlib")
    WrappedPath.register_driver(LocalPath, *names)



class BucketMixin:

    @property
    def bucket(self) -> str:
        """Cloud bucket name."""
        return self.drive


@WrappedPath.register_subclass
class WrappedS3Path(BucketMixin, WrappedPath):

    VALID_PROTOCOLS = {"s3", "s3a"}


@WrappedPath.register_subclass
class WrappedGCSPath(BucketMixin, WrappedPath):

    VALID_PROTOCOLS = {"gs", "gcs"}


@WrappedPath.register_subclass
class WrappedAzurePath(BucketMixin, WrappedPath):

    VALID_PROTOCOLS = {"az", "adl", "abfs", "abfss"}


@WrappedPath.register_subclass
class WrappedLocalPath(WrappedPath):

    VALID_PROTOCOLS = {"file", "local", ""}


@WrappedPath.register_subclass
class WrappedMemoryPath(WrappedPath):

    VALID_PROTOCOLS = {"memory"}


@WrappedPath.register_subclass
class WrappedHTTPPath(WrappedPath):

    VALID_PROTOCOLS = {"http", "https"}


# ======================================================================
#
#                  A S Y N C   P A T H   W R A P P E R
#
# ======================================================================


class AsyncPathLike(PathLike):
    ...


class AsyncWrappedPath(AsyncPathLike, BaseWrappedPath):

    # --- Concrete paths -----------------------------------------------

    # >> Expanding Paths

    @classmethod
    async def home(
        cls,
        *, driver: tx.Union[tx.Type[PathLike], str, None] = None
    ) -> tx.Self:
        """
        Return a new path object representing the user's home directory
        (as returned by `os.path.expanduser()` with ~ construct).

        If the home directory cannot be resolved, `RuntimeError` is raised.

        !!! example
            ```pycon
            >>> Path.home()
            Path('/home/user')
            ```
        """
        driver_cls = cls.get_driver(driver)
        home = ensure_coroutine(driver_cls.home)
        return cls(await home())

    async def expanduser(self) -> tx.Self:
        """
        Return a new path with expanded `~` and `~user` constructs, as
        returned by `os.path.expanduser()`.

        If a home directory cannot be resolved, `RuntimeError` is raised.

        !!! example
            ```pycon
            >>> Path('~/here/there/everywhere').expanduser()
            Path('/home/user/here/there/everywhere')
            ```
        """
        expanduser = ensure_coroutine(self.wrapped.expanduser)
        return type(self)(await expanduser())

    @classmethod
    async def cwd(
        cls,
        *, driver: tx.Union[tx.Type[PathLike], str, None] = None
    ) -> tx.Self:
        """
        Return a new path object representing the current directory
        (as returned by `os.getcwd()`):

        !!! example
            ```pycon
            >>> Path.cwd()
            Path('/home/user/workspace')
            ```
        """
        driver_cls = cls.get_driver(driver)
        cwd = ensure_coroutine(driver_cls.cwd)
        return cls(await cwd())

    async def absolute(self) -> tx.Self:
        """
        Make the path absolute, without normalization or resolving symlinks.
        Returns a new path object.

        !!! example
            ```pycon
            >>> Path('here/there/everywhere').absolute()
            Path('/home/user/workspace/here/there/everywhere')
            ```
        """
        absolute = ensure_coroutine(self.wrapped.absolute)
        return type(self)(await absolute())

    async def resolve(self, strict: bool = False) -> tx.Self:
        """
        Make the path absolute, resolving any symlinks.
        A new path object is returned.

        !!! example
            ```pycon
            >>> Path('here/there/everywhere').resolve()
            Path('/home/user/workspace/here/there/everywhere')
            ```

        !!! note
            `..` components are also eliminated
            (this is the only method to do so)
            ```pycon
            >>> Path('there/../elsewhere.').resolve()
            Path('/home/user/workspace/here/elsewhere')
            ```

        !!! warning
            If a path does not exist or a symlink loop is encountered, and
            `strict=True`, `OSError` is raised. If `strict=False`, the path
            is resolved as far as possible and any remainder is appended
            without checking whether it exists.
        """
        kwargs = {"strict": True} if strict else {}
        resolve = ensure_coroutine(self.wrapped.resolve)
        return type(self)(await resolve(**kwargs))

    async def readlink(self) -> tx.Self:
        """
        Return the path to which the symbolic link points
        (as returned by `os.readlink()`).

        !!! example
            ```pycon
            >>> Path('link_to_file').readlink()
            Path('target_file')
            ```
        """
        readlink = ensure_coroutine(self.wrapped.readlink)
        return type(self)(await readlink())

    # >> Querying File status

    async def stat(self, *, follow_symlinks: bool = True) -> tx.Any:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        stat = ensure_coroutine(self.wrapped.stat)
        return await stat(**kwargs)

    async def lstat(self) -> tx.Any:
        lstat = ensure_coroutine(self.wrapped.lstat)
        return await lstat()

    async def exists(self, *, follow_symlinks: bool = True) -> bool:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        exists = ensure_coroutine(self.wrapped.exists)
        return await exists(**kwargs)

    async def is_file(self, *, follow_symlinks: bool = True) -> bool:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        is_file = ensure_coroutine(self.wrapped.is_file)
        return await is_file(**kwargs)

    async def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        is_dir = ensure_coroutine(self.wrapped.is_dir)
        return await is_dir(**kwargs)

    async def is_symlink(self) -> bool:
        is_symlink = ensure_coroutine(self.wrapped.is_symlink)
        return await is_symlink()

    async def is_mount(self) -> bool:
        is_mount = ensure_coroutine(self.wrapped.is_mount)
        return await is_mount()

    async def is_socket(self) -> bool:
        is_socket = ensure_coroutine(self.wrapped.is_socket)
        return await is_socket()

    async def is_fifo(self) -> bool:
        is_fifo = ensure_coroutine(self.wrapped.is_fifo)
        return await is_fifo()

    async def is_block_device(self) -> bool:
        is_block_device = ensure_coroutine(self.wrapped.is_block_device)
        return await is_block_device()

    async def is_char_device(self) -> bool:
        is_char_device = ensure_coroutine(self.wrapped.is_char_device)
        return await is_char_device()

    async def samefile(self, other: PathOrStr) -> bool:
        samefile = ensure_coroutine(self.wrapped.samefile)
        return await samefile(other)

    # >> Reading & Writing Files

    async def open(self, mode: AccessMode = "r", **kwargs) -> tx.IO:
        open = ensure_coroutine(self.wrapped.open)
        return await open(mode, **kwargs)

    async def read_text(self, *a, **k) -> str:
        if hasattr(self.wrapped, "read_text"):
            read_text = ensure_coroutine(self.wrapped.read_text)
            return await read_text(*a, **k)
        return (await self.read_bytes()).decode(**k)

    async def read_bytes(self, *a, **k) -> bytes:
        if hasattr(self.wrapped, "read_bytes"):
            read_bytes = ensure_coroutine(self.wrapped.read_bytes)
            return await read_bytes(*a, **k)
        with await self.open("rb", **k) as f:
            read = ensure_coroutine(f.read)
            return await read()

    async def write_text(self, data: str, *a, **k) -> int:
        if hasattr(self.wrapped, "write_text"):
            write_text = ensure_coroutine(self.wrapped.write_text)
            return await write_text(data, *a, **k)
        return await self.write_bytes(data.encode(**k))

    async def write_bytes(self, data: bytes) -> int:
        if hasattr(self.wrapped, "write_bytes"):
            write_bytes = ensure_coroutine(self.wrapped.write_bytes)
            return await write_bytes(data)
        with await self.open("wb") as f:
            write = ensure_coroutine(f.write)
            return await write(data)

    # >> Reading Directories

    async def iterdir(self) -> tx.AsyncIterator[tx.Self]:
        async for p in self.wrapped.iterdir():
            yield type(self)(p)

    async def walk(
        self,
        top_down: bool = True,
        on_error: tx.Optional[tx.Callable[[OSError], tx.Any]] = None,
        follow_symlinks: bool = False
    ) -> tx.AsyncIterator[tx.Self]:
        kwargs = {"follow_symlinks": True} if follow_symlinks else {}
        async for p in self.wrapped.walk(
            top_down=top_down, on_error=on_error, **kwargs
        ):
            yield type(self)(p)

    # >> Creating Files & Dirs

    async def touch(self, mode: int = 0o666, exist_ok: bool = True) -> tx.Self:
        touch = ensure_coroutine(self.wrapped.touch)
        await touch(mode=mode, exist_ok=exist_ok)
        return self

    async def mkdir(
        self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> tx.Self:
        kwargs = {"exist_ok": True} if exist_ok else {}
        mkdir = ensure_coroutine(self.wrapped.mkdir)
        await mkdir(mode=mode, parents=parents, **kwargs)
        return self

    async def symlink_to(
            self, target: PathOrStr, target_is_directory: bool = False
    ) -> tx.Self:
        kwargs = {"target_is_directory": True} if target_is_directory else {}
        symlink_to = ensure_coroutine(self.wrapped.symlink_to)
        await symlink_to(target, **kwargs)
        return self

    async def hardlink_to(self, target: PathOrStr) -> tx.Self:
        hardlink_to = ensure_coroutine(self.wrapped.hardlink_to)
        await hardlink_to(target)
        return self

    # >> Copying & Moving

    async def copy(
        self, target: PathOrStr,
        *, follow_symlinks: bool = True, preserve_metadata: bool = False
    ) -> tx.Self:
        cls = type(self)
        if hasattr(self.wrapped, "copy"):
            kwargs = {}
            if not follow_symlinks:
                kwargs = {"follow_symlinks": False}
            if preserve_metadata:
                kwargs = {"preserve_metadata": True}
            copy = ensure_coroutine(self.wrapped.copy)
            return cls(await copy(target, **kwargs))
        return await self._copy_fallback(
            target,
            follow_symlinks=follow_symlinks,
            preserve_metadata=preserve_metadata
        )

    async def _copy_fallback(
        self, target: PathOrStr,
        *, follow_symlinks: bool = True, preserve_metadata: bool = False
    ) -> tx.Self:
        cls = type(self)
        copy = shutil.copy2 if preserve_metadata else shutil.copy
        if self.is_dir():
            kwargs = {"symlinks": True} if not follow_symlinks else {}
            kwargs["copy_function"] = copy
            copytree = ensure_coroutine(shutil.copytree)
            await copytree(self.path, cls(target).path, **kwargs)
            return cls(target)
        if self.is_file():
            kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
            copy = ensure_coroutine(copy)
            await copy(self.path, cls(target).path, **kwargs)
            return cls(target)
        raise FileNotFoundError(
            f"Cannot copy {self.path}: not a file or directory"
        )

    async def copy_into(
        self, target_dir: PathOrStr,
        *, follow_symlinks: bool = True, preserve_metadata: bool = False
    ) -> tx.Self:
        cls = type(self)
        if hasattr(self.wrapped, "copy_into"):
            kwargs = {}
            if not follow_symlinks:
                kwargs = {"follow_symlinks": False}
            if preserve_metadata:
                kwargs = {"preserve_metadata": True}
            copy_into = ensure_coroutine(self.wrapped.copy_into)
            return cls(await copy_into(target_dir, **kwargs))
        return await self.copy(
            cls(target_dir) / self.name,
            follow_symlinks=follow_symlinks,
            preserve_metadata=preserve_metadata
        )

    async def rename(self, target: PathOrStr) -> tx.Self:
        rename = ensure_coroutine(self.wrapped.rename)
        await rename(target)
        return type(self)(target)

    async def replace(self, target: PathOrStr) -> tx.Self:
        replace = ensure_coroutine(self.wrapped.replace)
        await replace(target)
        return type(self)(target)

    async def move(self, target: PathOrStr) -> tx.Self:
        if hasattr(self.wrapped, "move"):
            move = ensure_coroutine(self.wrapped.move)
            return type(self)(await move(target))
        return await self._move_fallback(target)

    async def _move_fallback(self, target: PathOrStr) -> tx.Self:
        try:
            return await self.replace(target)
        except OSError:
            obj = await self.copy(target, preserve_metadata=True)
            await self.unlink()
            return obj

    async def move_into(self, target_dir: PathOrStr) -> tx.Self:
        cls = type(self)
        if hasattr(self.wrapped, "move_into"):
            move_into = ensure_coroutine(self.wrapped.move_into)
            return cls(await move_into(target_dir))
        return await self.move(cls(target_dir) / self.name)

    async def unlink(self, *, missing_ok: bool = False) -> None:
        kwargs = {"missing_ok": True} if missing_ok else {}
        unlink = ensure_coroutine(self.wrapped.unlink)
        await unlink(**kwargs)

    async def rmdir(self, recursive: bool = True) -> None:
        if isinstance(self.wrapped, LocalPath):
            if recursive:
                rmtree = ensure_coroutine(shutil.rmtree)
                await rmtree(self.path)
            else:
                rmdir = ensure_coroutine(self.wrapped.rmdir)
                await rmdir()
        else:
            rmdir = ensure_coroutine(self.wrapped.rmdir)
            await rmdir(recursive)

    # >> Permission & Owner

    async def owner(self, *, follow_symlinks: bool = True) -> str:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        owner = ensure_coroutine(self.wrapped.owner)
        return await owner(**kwargs)

    async def group(self, *, follow_symlinks: bool = True) -> str:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        group = ensure_coroutine(self.wrapped.group)
        return await group(**kwargs)

    async def chmod(self, *, follow_symlinks: bool = True) -> None:
        kwargs = {"follow_symlinks": False} if not follow_symlinks else {}
        chmod = ensure_coroutine(self.wrapped.chmod)
        await chmod(**kwargs)

    async def lchmod(self) -> None:
        lchmod = ensure_coroutine(self.wrapped.lchmod)
        await lchmod()


if UPath:
    AsyncWrappedPath.register_driver(UPath)
if AnyPath:
    names = ("anypath", "cloudpath", "cloudpathlib")
    AsyncWrappedPath.register_driver(AnyPath, *names)
if LocalPath:
    names = ("local", "pathlib")
    AsyncWrappedPath.register_driver(LocalPath, *names)
if AsyncLocalPath:
    names = ("anyio", "async")
    AsyncWrappedPath.register_driver(AsyncLocalPath, *names)


@AsyncWrappedPath.register_subclass
class AsyncWrappedS3Path(BucketMixin, AsyncWrappedPath):

    VALID_PROTOCOLS = {"s3", "s3a"}


@AsyncWrappedPath.register_subclass
class AsyncWrappedGCSPath(BucketMixin, AsyncWrappedPath):

    VALID_PROTOCOLS = {"gs", "gcs"}


@AsyncWrappedPath.register_subclass
class AsyncWrappedAzurePath(BucketMixin, AsyncWrappedPath):

    VALID_PROTOCOLS = {"az", "adl", "abfs", "abfss"}


@AsyncWrappedPath.register_subclass
class AsyncWrappedLocalPath(AsyncWrappedPath):

    VALID_PROTOCOLS = {"file", "local", ""}


@AsyncWrappedPath.register_subclass
class AsyncWrappedMemoryPath(AsyncWrappedPath):

    VALID_PROTOCOLS = {"memory"}


@AsyncWrappedPath.register_subclass
class AsyncWrappedHTTPPath(AsyncWrappedPath):

    VALID_PROTOCOLS = {"http", "https"}


# ======================================================================
#
#                  F A L L B A C K   D R I V E R S
#
# ======================================================================


@WrappedPath.register_driver("fallback")
@AsyncWrappedPath.register_driver("fallback")
class FallbackPath(PathLike):
    """
    A fallback path driver that understands protocols but only implement
    the most basic operations (properties and composition).

    It is used when no other protocol-aware driver (UPath, AnyPath) is
    available.
    """

    @classmethod
    def _parse_protocol(
        self, *path: PathOrStr, protocol: tx.Optional[str] = None
    ) -> tx.Tuple[str, ...]:
        path = tuple(map(str, path))

        if not path:
            path = ("",)

        protocol_match = RE_PROTOCOL.match(path[0])
        if protocol_match:
            parsed_protocol, path0 = protocol_match.groups()
            if protocol is not None and protocol != parsed_protocol:
                raise ValueError(
                    f"Protocol mismatch: {protocol} != {parsed_protocol}"
                )
            protocol = parsed_protocol
            path = (protocol + "://", path0, *path[1:])

        else:
            if protocol is None:
                protocol = "file"
            path = (protocol + "://", *path)

        return path

    def __init__(
        self, *path: PathOrStr, protocol: tx.Optional[str] = None
    ) -> None:
        protocol, *path = self._parse_protocol(*path, protocol=protocol)
        protocol = protocol[::-3]  # Remove "://"
        if protocol in ("local", "file"):
            self._path = LocalPath(*path)
        else:
            self._path = PurePosixPath(*path)
        self._protocol = protocol

    # --- Repr ---------------------------------------------------------

    def __fspath__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        path = str(self._path)
        if self.protocol not in ("local", "file"):
            # Remove leading dots and slashes for non-local paths, as
            # there is not such thing as absolute vs relative.
            path = path.lstrip("./")
        return f"{self.protocol}://{path}"

    def __repr__(self) -> str:
        return f"FallbackPath({repr(str(self._path))})"

    def __bytes__(self) -> bytes:
        return str(self).encode()

    def __hash__(self) -> int:
        return hash(str(self))

    # --- Operators ----------------------------------------------------

    def __truediv__(self, other: PathOrStr) -> str:
        return self.joinpath(other)

    def __rtruediv__(self, other: PathOrStr) -> str:
        return type(self)(other) / self

    # --- Properties ---------------------------------------------------

    @property
    def parts(self) -> tuple:
        parts = self._path.parts
        if self.protocol not in ("local", "file") and parts[0] == "/":
            parts = parts[1:]
        return parts

    @property
    def drive(self) -> str:
        return self._path.drive

    @property
    def root(self) -> str:
        root = self._path.root
        if self.protocol not in ("local", "file") and root == "":
            root = "/"
        return root

    @property
    def anchor(self) -> str:
        return self.drive + self.root

    @property
    def parents(self) -> tx.Sequence[tx.Self]:
        parents = self._path.parents
        parents = (type(self)(p, protocol=self.protocol) for p in parents)
        return tuple(parents)

    @property
    def parent(self) -> tx.Self:
        return type(self)(self._path.parent, protocol=self.protocol)

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def suffix(self) -> str:
        return self._path.suffix

    @property
    def suffixes(self) -> tx.List[str]:
        return self._path.suffixes

    @property
    def stem(self) -> str:
        return self._path.stem

    # --- Methods ------------------------------------------------------

    def as_posix(self) -> str:
        path = self._path.as_posix()
        if self.protocol not in ("local", "file"):
            path = path.lstrip("./")
        return f"{self.protocol}://{path}"

    # --- Methods ------------------------------------------------------

    def is_absolute(self) -> bool:
        if self.protocol not in ("local", "file"):
            return True
        return self._path.is_absolute()

    def is_relative_to(self, other: PathOrStr) -> bool:
        if self.protocol not in ("local", "file"):
            return False
        other = type(self)(other)
        if other.protocol not in ("local", "file"):
            return False
        return self._path.is_relative_to(other._path)

    def joinpath(self, *pathsegments: tx.Unpack[PathOrStr]) -> tx.Self:
        """
        Calling this method is equivalent to combining the path with each
        of the given `pathsegments` in turn:

        !!! example
            ```pycon
            >>> Path('/here/there').joinpath('everywhere')
            Path('/here/there/everywhere')
            >>> Path('/here/there').joinpath('everywhere', 'and', 'beyond')
            Path('/here/there/everywhere/and/beyond')
            ```
        """
        return type(self)(self._path, *pathsegments, protocol=self.protocol)

    def full_match(
        self, pattern: str,
        *, case_sensitive: tx.Optional[bool] = None
    ) -> bool:
        if self.protocol in ("local", "file"):
            kwargs = {}
            if case_sensitive is not None:
                kwargs["case_sensitive"] = case_sensitive
            return self._path.full_match(pattern, **kwargs)
        raise NotImplementedError(
            f"full_match() is not implemented for protocol {self.protocol}"
        )

    def match(
        self, pattern: str,
        *, case_sensitive: tx.Optional[bool] = None
    ) -> bool:
        if self.protocol in ("local", "file"):
            kwargs = {}
            if case_sensitive is not None:
                kwargs["case_sensitive"] = case_sensitive
            return self._path.match(pattern, **kwargs)
        raise NotImplementedError(
            f"match() is not implemented for protocol {self.protocol}"
        )

    def relative_to(self, other: PathOrStr, walk_up: bool = False) -> tx.Self:
        other = type(self)(other, protocol=self.protocol)
        if self.protocol in ("local", "file"):
            kwargs = {"walk_up": True} if walk_up else {}
            return type(self)(self._path.relative_to(other._path, **kwargs))
        raise NotImplementedError(
            f"relative_to() is not implemented for protocol {self.protocol}"
        )

    def with_name(self, name: str) -> tx.Self:
        """
        Return a new path with the `name` changed.

        If the original path does not have a name, `ValueError` is raised

        !!! example
            ```pycon
            >>> Path('/here/there/everywhere').with_name('elsewhere')
            Path('/here/there/elsewhere')
            ```
        """
        path = self._path.with_name(name)
        return type(self)(path, protocol=self.protocol)

    def with_stem(self, stem: str) -> tx.Self:
        path = self._path.with_stem(stem)
        return type(self)(path, protocol=self.protocol)

    def with_suffix(self, suffix: str) -> tx.Self:
        path = self._path.with_suffix(suffix)
        return type(self)(path, protocol=self.protocol)

    def with_segments(self, *segments: tx.Unpack[PathOrStr]) -> tx.Self:
        path = self._path.with_segments(*segments)
        return type(self)(path, protocol=self.protocol)

    def with_protocol(self, protocol: str) -> tx.Self:
        return type(self)(self._path, protocol=protocol)

    # --- Concrete paths -----------------------------------------------

    # >> Parsing URIs

    def as_uri(self) -> str:
        if self.protocol in ("local", "file"):
            if hasattr(self._path, "as_uri"):
                return self._path.as_uri()
            return f"file://{self._path.as_posix()}"
        return str(self)

    def from_uri(cls, uri: PathOrStr) -> tx.Self:
        return cls(uri)

    # >> Reading directories

    def glob(
        self,
        pattern: PathOrStr,
        *,
        case_sensitive: tx.Optional[bool] = None,
        recurse_symlinks: bool = False
    ) -> tx.Iterator[tx.Self]:
        kwargs = {}
        if case_sensitive is not None:
            kwargs['case_sensitive'] = case_sensitive
        if recurse_symlinks:
            kwargs['recurse_symlinks'] = recurse_symlinks
        pattern = str(pattern)
        for p in self._path.glob(pattern, **kwargs):
            yield type(self)(p, protocol=self.protocol)

    def rglob(
        self,
        pattern: PathOrStr,
        *,
        case_sensitive: tx.Optional[bool] = None,
        recurse_symlinks: bool = False
    ) -> tx.Iterator[tx.Self]:
        kwargs = {}
        if case_sensitive is not None:
            kwargs['case_sensitive'] = case_sensitive
        if recurse_symlinks:
            kwargs['recurse_symlinks'] = recurse_symlinks
        pattern = str(pattern)
        for p in self._path.rglob(pattern, **kwargs):
            yield type(self)(p, protocol=self.protocol)

    # --- UPath --------------------------------------------------------

    @property
    def protocol(self) -> str:
        return self._protocol

    @property
    def path(self) -> str:
        return str(self._path)

    def joinuri(self, path: PathOrStr) -> str:
        return self.joinpath(path)

    # --- CloudPath ----------------------------------------------------

    @property
    def cloud_prefix(self) -> str:
        return self.protocol + "://"

    @property
    def fspath(self) -> str:
        return self.__fspath__()

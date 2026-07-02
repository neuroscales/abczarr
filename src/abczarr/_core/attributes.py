"""
In-memory, write-through (configurable) attributes mapping for Zarr.

This file contains code from the Zarr project
https://github.com/zarr-developers/zarr-python
"""

# stdlib
import json
import os
import tempfile
import threading
from collections.abc import Iterator, MutableMapping
from typing import Any

# dependencies
import typing_extensions as tx

# locals
from ..abc import ZarrNode
from . import typing as tz
from .path import Path

if hasattr(MutableMapping, "__class_getitem__"):
    AttributesBase = MutableMapping[str, Any]
else:
    AttributesBase = MutableMapping


class Attributes(AttributesBase):
    """
    In-memory, write-through (configurable) attributes mapping for Zarr v2/v3.

    Reads are served from an in-memory cache. On mutation, the cache is updated
    and, if `write_through=True`, the file is flushed atomically.
    Otherwise, call `flush()` explicitly to persist.

    Works for both arrays and groups, as long as the parent exposes:
      - .store_path (path-like, directory)
      - .zarr_version (2 or 3)
    """

    def __init__(self, obj: ZarrNode, *, write_through: bool = True) -> None:
        """
        Parameters
        ----------
        obj : ZarrNode
            The Zarr array or group object to which these attributes belong.
        write_through : bool, default True
            If True, writes to the attributes mapping are immediately
            flushed to disk.
            If False, writes are cached in memory and must be flushed
            explicitly.
        """
        self._obj = obj
        self._write_through = write_through
        self._lock = threading.RLock()
        self._loaded = False
        self._attrs: dict[str, Any] = {}

        # cache paths
        self._file_path = self._get_file_path()
        self._key_path = self._get_key_path()

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        """Return the Zarr version (2 or 3) for this attributes mapping."""
        return self._obj.zarr_version

    # ---------- public helpers ----------

    def asdict(self) -> dict[str, Any]:
        """Return a snapshot of attributes as a dict."""
        with self._lock:
            self._ensure_loaded()
            return dict(self._attrs)

    def put(self, d: dict[str, Any]) -> None:
        """Overwrite all attributes with d (in-memory), then flush."""
        with self._lock:
            self._ensure_loaded()
            self._attrs = dict(d)
            if self._write_through:
                self._flush_locked()

    def flush(self) -> None:
        """Persist current in-memory attributes to disk atomically."""
        with self._lock:
            self._ensure_loaded()
            self._flush_locked()

    def refresh(self) -> None:
        """Discard in-memory cache and re-read from disk."""
        with self._lock:
            self._loaded = False
            self._attrs.clear()
            self._ensure_loaded()

    # ---------- MutableMapping interface ----------

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get an attribute by key."""
        with self._lock:
            self._ensure_loaded()
            return self._attrs[key]

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Set or update an attribute."""
        with self._lock:
            self._ensure_loaded()
            self._attrs[key] = value
            if self._write_through:
                self._flush_locked()

    def __delitem__(self, key: str) -> None:
        """Delete an attribute."""
        with self._lock:
            self._ensure_loaded()
            del self._attrs[key]
            if self._write_through:
                self._flush_locked()

    def __iter__(self) -> Iterator[str]:
        """Iterate over a snapshot of keys."""
        with self._lock:
            self._ensure_loaded()
            # iterate over a snapshot
            # TODO: deepcopy?
            return iter(dict(self._attrs))

    def __len__(self) -> int:
        """Return number of attributes."""
        with self._lock:
            self._ensure_loaded()
            return len(self._attrs)

    # ---------- internals ----------

    def _get_file_path(self) -> os.PathLike:
        """Return the path to the attributes file on disk."""
        if self.zarr_version == 1:
            return Path(self._obj.store_path) / "attrs"
        if self.zarr_version == 2:
            return Path(self._obj.store_path) / ".zattrs"
        if self.zarr_version == 3:
            return Path(self._obj.store_path) / "zarr.json"
        raise ValueError(f"Unsupported zarr_version: {self.zarr_version}")

    def _get_key_path(self) -> tx.Tuple[str]:
        """
        Return the key path to the attributes object in the attributes file.
        """
        if self.zarr_version >= 3:
            return ("attributes",)
        return ()

    def _load_all(self) -> dict:
        """Load the entire file (including non-attributes keys)"""

        if not self._file_path.exists():
            return {}

        with self._file_path.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid attributes file: {self._file_path} "
                f"(expected JSON object, got {type(data).__name__})"
            )

        return data


    def _ensure_loaded(self) -> None:
        """Load if not already loaded."""
        if self._loaded:
            return

        # Load entire file
        data = self._load_all()

        # Extract target key
        for key in  self._key_path:
            data = data.get(key, {})

        # Save
        self._attrs = data
        self._loaded = True

    def _flush_locked(self) -> None:
        if self._key_path:
            # Load entire file
            data = self._load_all()
            # Update target key
            d = data
            for key in self._key_path[:-1]:
                d = d.setdefault(key, {})
            d[self._key_path[-1]] = self._attrs

        else:
            data = self._attrs

        _atomic_json_write(self._file_path, data)


def _atomic_json_write(path: os.PathLike, data: tx.Mapping[str, Any]) -> None:
    """
    Atomically write JSON to 'path' via a temp file + rename.

    Works with local and fsspec-backed paths that expose .open / .parent.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Use a temp file in the same directory to keep rename atomic on POSIX FS.
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".attrs_tmp_", dir=str(parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        # UPath may wrap non-local FS; try best-effort replace
        # pathlib.Path has replace(); UPath typically forwards.
        Path(tmp_name).replace(path)
    finally:
        # If replace failed, clean up temp file best-effort
        try:
            if Path(tmp_name).exists():
                Path(tmp_name).unlink()
        except Exception:
            pass

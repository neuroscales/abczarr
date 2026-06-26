"""
Metadata handling for Zarr.

This file contains code from the Zarr project
https://github.com/zarr-developers/zarr-python
"""

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace

# dependencies
import typing_extensions as tx

# locals
from . import typing as tz


@dataclass(frozen=True)
class Metadata:
    """Frozen, recursive, JSON-serializable metadata class."""

    def to_dict(self) -> tz.Attributes:
        """Convert this metadata to a JSON-serializable dict."""
        out: tz.Attributes = {}
        for f in fields(self):
            k = f.name
            v = getattr(self, k)
            if _is_metadata(v):
                out[k] = v.to_dict()
            elif _is_sequence(v):
                out[k] = tuple(
                    x.to_dict() if _is_metadata(x) else x for x in v
                )
            else:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, data: tz.FrozenAttributes) -> tx.Self:
        """Create an instance from a JSON-serializable dict."""
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class GroupMetadata(Metadata):
    """Metadata for a Zarr group, including attributes and format version."""

    attributes: tz.Attributes = field(default_factory=dict)
    zarr_format: tz.ZarrVersion = 3
    node_type: tx.Literal["group"] = field(default="group", init=False)

    # Convenience updaters (immutably return new metadata)
    def update_attributes(self, attributes: tz.FrozenAttributes) -> tx.Self:
        """Return a new GroupMetadata with updated attributes."""
        return replace(self, attributes=dict(attributes))

    # ---- I/O helpers for disk persistence ----
    @staticmethod
    def _atomic_write(path: os.PathLike, data: tz.Attributes) -> None:
        """Write data to path atomically."""
        PathType = type(path)
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".meta_tmp_", dir=str(parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            PathType(tmp).replace(path)
        finally:
            try:
                if PathType(tmp).exists():
                    PathType(tmp).unlink()
            except Exception:
                pass

    @classmethod
    def from_files(cls, root: os.PathLike) -> "GroupMetadata":
        """Load metadata from the specified root directory."""

        # Prefer zarr.json if present; otherwise v2 split files
        zarr_json = root / "zarr.json"
        if zarr_json.exists():
            with zarr_json.open("r", encoding="utf-8") as f:
                d = json.load(f)
            attrs = d.get("attributes", {}) or {}
            return cls(attributes=attrs, zarr_format=3)

        # v2: .zgroup + .zattrs (attributes may be missing)
        zgroup = root / ".zgroup"
        zattrs = root / ".zattrs"
        zf = 2
        if zgroup.exists():
            with zgroup.open("r", encoding="utf-8") as f:
                g = json.load(f)
                zf = g.get("zarr_format", 2)
        attrs = {}
        if zattrs.exists():
            with zattrs.open("r", encoding="utf-8") as f:
                attrs = json.load(f)
        return cls(attributes=attrs, zarr_format=zf)

    def to_files(self, root: os.PathLike) -> None:
        """Write this metadata to the specified root directory."""
        if self.zarr_format == 3:
            path = root / "zarr.json"
            data = {}
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            data["zarr_format"] = 3
            data["node_type"] = "group"
            data["attributes"] = self.attributes
            self._atomic_write(path, data)
        else:
            # v2 writes two files
            gpath, apath = root / ".zgroup", root / ".zattrs"
            self._atomic_write(gpath, {"zarr_format": 2})
            self._atomic_write(apath, dict(self.attributes))


def _is_sequence(obj: tx.Any) -> bool:
    """Check if an object is a sequence (e.g., list or tuple)."""
    str_like = (str, bytes, bytearray)
    return isinstance(obj, Sequence) and not isinstance(obj, str_like)


def _is_metadata(obj: tx.Any) -> bool:
    """Check if an object is an instance of Metadata."""
    return isinstance(obj, Metadata)

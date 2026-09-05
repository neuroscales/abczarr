"""How `_atomic_write` writes metadata on local vs remote stores.

On a local filesystem it writes a temporary file next to the target and
renames it over the target -- an atomic replace. A remote/object store has
no such rename, so the metadata is written directly through the store's own
API; a single object PUT is atomic at the object level, so a reader still
never sees a half-written file. This keeps the path-based group/array
fallback (`PathGroup.create_group` / `_create_array`) working on remote
backends rather than failing on them.
"""

import json
import pathlib

import pytest

from abczarr.metadata.base import GroupMetadataV3, _atomic_write


class _FakeRemotePath:
    """An in-memory stand-in for a remote/object store path.

    Carries just the surface `_atomic_write`'s remote branch touches -- a
    non-local ``protocol``, a ``parent`` that can ``mkdir``, and
    ``write_bytes`` -- so the direct-write path is exercised without a real
    universal-pathlib / fsspec backend (absent on the core-only leg).
    """

    def __init__(
        self,
        store: dict,
        key: str = "s3://bucket/grp/zarr.json",
        protocol: str = "s3",
    ) -> None:
        self._store = store
        self._key = key
        self.protocol = protocol

    @property
    def parent(self) -> "_FakeRemotePath":
        # an object store has no directories; mkdir is a no-op
        return self

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        pass

    def write_bytes(self, data: bytes) -> None:
        self._store[self._key] = data

    def __fspath__(self) -> str:
        return self._key


def test_plain_pathlib_path_is_written(tmp_path: pathlib.Path) -> None:
    # a stdlib path has no ``protocol`` and counts as local
    target = tmp_path / "zarr.json"
    _atomic_write(target, {"zarr_format": 3})
    assert json.loads(target.read_text()) == {"zarr_format": 3}
    # the temp file is cleaned up, not left beside the target
    assert [p.name for p in tmp_path.iterdir()] == ["zarr.json"]


def test_remote_path_is_written_directly() -> None:
    # a non-local protocol takes the direct-write branch instead of raising
    store: dict = {}
    target = _FakeRemotePath(store)
    _atomic_write(target, {"zarr_format": 3, "node_type": "group"})
    assert json.loads(store["s3://bucket/grp/zarr.json"]) == {
        "zarr_format": 3,
        "node_type": "group",
    }


def test_remote_group_metadata_round_trips_via_memory_store() -> None:
    # end-to-end: a group's metadata written to and read back from a real
    # (in-process) remote store, proving the path-based fallback works there.
    pytest.importorskip("upath")  # memory:// needs an fsspec-backed driver
    from bagof.paths import Path

    root = Path("memory://bucket/grp-atomic-test")
    GroupMetadataV3(attributes={"hello": "world"}).to_file(root)
    reloaded = GroupMetadataV3.from_file(root)
    assert reloaded.attributes == {"hello": "world"}

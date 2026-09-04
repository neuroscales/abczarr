"""Attributes read from the node's metadata, and writes go through the store.

The node's cached metadata is the single source of truth for attributes:
`node.attrs` reads from it, and a write -- `node.attrs[k] = v`,
`node.update_attributes(...)`, or the async coroutine twin -- persists by
producing new metadata and writing it through the node's persistence path
(zarr-python delegates to zarr; every other backend rewrites the metadata
document through the store). No attribute write touches a separate file
behind the store's back.

The path-based checks run with no backend installed; the tensorstore and
zarr-python checks are skipped when those libraries are absent.
"""

import asyncio
import json
import pathlib

import pytest

import abczarr
from abczarr.abc.asynchronous import AsyncPathGroup
from abczarr.abc.store import PathBasedStore
from abczarr.abc.sync import PathGroup
from abczarr.metadata.base import NodeMetadata

# --------------------------------------------------------------------------
# a backend-independent path group, so the store-routed path runs with no
# backend installed (mirrors tests/test_path_group.py)
# --------------------------------------------------------------------------


class _StubArray:
    def __init__(self, store_path: object) -> None:
        self.store_path = store_path


class _Group(PathGroup):
    def _open_array(self, store_path: object) -> _StubArray:
        return _StubArray(store_path)


_GROUP_DOC = {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {"kind": "demo"},
}


def _write_group(store: PathBasedStore, doc: dict) -> None:
    store.set(
        "zarr.json",
        json.dumps(doc).encode("utf-8"),
    )


def _local_group(tmp_path: pathlib.Path) -> str:
    root = str(pathlib.Path(tmp_path) / "grp.zarr")
    _write_group(PathBasedStore(root), _GROUP_DOC)
    return root


# --------------------------------------------------------------------------
# reads come from the metadata (the single source of truth)
# --------------------------------------------------------------------------


def test_attrs_read_from_metadata(tmp_path: pathlib.Path) -> None:
    group = _Group(_local_group(tmp_path))
    # the mapping and the metadata never disagree -- same values, one source
    assert dict(group.attrs) == {"kind": "demo"}
    assert dict(group.attrs) == dict(group.metadata.attributes)


def test_attrs_do_not_reread_a_separate_file(tmp_path: pathlib.Path) -> None:
    # once the metadata is cached, deleting the on-disk document does not
    # change what attrs reports: the read is served from memory, not the file
    root = _local_group(tmp_path)
    group = _Group(root)
    assert dict(group.attrs) == {"kind": "demo"}
    (pathlib.Path(root) / "zarr.json").unlink()
    assert dict(group.attrs) == {"kind": "demo"}


# --------------------------------------------------------------------------
# writes persist through the store, and update the in-memory cache
# --------------------------------------------------------------------------


def test_update_attributes_persists_and_caches(
    tmp_path: pathlib.Path,
) -> None:
    root = _local_group(tmp_path)
    group = _Group(root)
    group.update_attributes({"unit": "micrometer"})
    # merged with the existing attributes, and visible on the same node
    assert dict(group.attrs) == {"kind": "demo", "unit": "micrometer"}
    assert group.metadata.attributes["unit"] == "micrometer"
    # a freshly opened node reads it back through the store
    assert dict(_Group(root).attrs) == {
        "kind": "demo",
        "unit": "micrometer",
    }


def test_write_goes_through_the_store_document(
    tmp_path: pathlib.Path,
) -> None:
    # the write lands in the zarr.json document (not a side file), and the
    # document's other fields are preserved
    root = _local_group(tmp_path)
    _Group(root).update_attributes({"unit": "micrometer"})
    raw = PathBasedStore(root).get("zarr.json")
    doc = json.loads(raw)
    assert doc["attributes"] == {"kind": "demo", "unit": "micrometer"}
    assert doc["node_type"] == "group"
    assert doc["zarr_format"] == 3


def test_setitem_and_delitem_persist(tmp_path: pathlib.Path) -> None:
    root = _local_group(tmp_path)
    group = _Group(root)
    group.attrs["scale"] = 0.5
    assert _Group(root).attrs["scale"] == 0.5
    del group.attrs["scale"]
    assert "scale" not in _Group(root).attrs
    # the pre-existing attribute is untouched by the per-key edits
    assert _Group(root).attrs["kind"] == "demo"


def test_async_update_attributes_persists(tmp_path: pathlib.Path) -> None:
    root = _local_group(tmp_path)
    twin = _Group(root).as_async()
    assert isinstance(twin, AsyncPathGroup)

    async def go() -> None:
        await twin.update_attributes({"unit": "micrometer"})

    asyncio.run(go())
    # visible on the twin (through its sync node's cache) and on re-open
    assert twin.attrs["unit"] == "micrometer"
    assert dict(_Group(root).attrs) == {
        "kind": "demo",
        "unit": "micrometer",
    }


# --------------------------------------------------------------------------
# the store-routed write is not local-FS-only: a memory store round-trips
# --------------------------------------------------------------------------


def test_memory_store_round_trip() -> None:
    pytest.importorskip("upath")
    url = "memory://abczarr-attrs-test/grp"
    _write_group(PathBasedStore(url), dict(_GROUP_DOC))
    _Group(url).update_attributes({"unit": "micrometer"})
    # a fresh node over the same memory URL reads the store-routed write back
    assert dict(_Group(url).attrs) == {
        "kind": "demo",
        "unit": "micrometer",
    }
    doc = json.loads(PathBasedStore(url).get("zarr.json"))
    assert doc["attributes"] == {"kind": "demo", "unit": "micrometer"}


# --------------------------------------------------------------------------
# tensorstore: the store-rewrite path, against a real backend
# --------------------------------------------------------------------------


def _ts_array(tmp_path: pathlib.Path) -> str:
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("tensorstore")
    root = str(pathlib.Path(tmp_path) / "ts.zarr")
    zarr.open_group(root, mode="w").create_array(
        "img", shape=(4, 4), chunks=(2, 2), dtype="float32"
    )
    return root + "/img"


def test_tensorstore_update_attributes_round_trips(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("tensorstore")
    path = _ts_array(tmp_path)
    node = abczarr.open(path, mode="a", driver="tensorstore")
    node.update_attributes({"unit": "micrometer"})
    # visible on the same node (its metadata cache was updated in place)
    assert node.attrs["unit"] == "micrometer"
    # ...and on re-open, so the write reached the store
    reopened = abczarr.open(path, mode="r", driver="tensorstore")
    assert reopened.attrs["unit"] == "micrometer"


def test_tensorstore_async_update_attributes_round_trips(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("tensorstore")
    path = _ts_array(tmp_path)

    async def go() -> None:
        node = await abczarr.open(
            path, mode="a", asynchronous=True, driver="tensorstore"
        )
        await node.update_attributes({"unit": "micrometer"})

    asyncio.run(go())
    reopened = abczarr.open(path, mode="r", driver="tensorstore")
    assert reopened.attrs["unit"] == "micrometer"


# --------------------------------------------------------------------------
# zarr-python: the write delegates to zarr, so zarr's own object sees it
# --------------------------------------------------------------------------


def _zp_array(tmp_path: pathlib.Path) -> str:
    zarr = pytest.importorskip("zarr")
    root = str(pathlib.Path(tmp_path) / "zp.zarr")
    zarr.open_group(root, mode="w").create_array(
        "img", shape=(4, 4), chunks=(2, 2), dtype="float32"
    )
    return root + "/img"


def test_zarr_python_update_attributes_delegates_to_zarr(
    tmp_path: pathlib.Path,
) -> None:
    zarr = pytest.importorskip("zarr")
    path = _zp_array(tmp_path)
    node = abczarr.open(path, mode="a")
    node.update_attributes({"unit": "micrometer"})
    # the delegation keeps abczarr and zarr-python in step: the underlying
    # zarr object reports the change, and so does a plain zarr.open
    assert node.native.attrs["unit"] == "micrometer"
    assert dict(node.attrs) == {"unit": "micrometer"}
    assert zarr.open(path, mode="r").attrs["unit"] == "micrometer"


def test_zarr_python_reads_attrs_from_metadata(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("zarr")
    path = _zp_array(tmp_path)
    node = abczarr.open(path, mode="a")
    node.update_attributes({"a": 1, "b": 2})
    # attrs and the node's metadata are the same source of truth
    assert dict(node.attrs) == dict(node.metadata.attributes)


def test_zarr_python_async_update_attributes_delegates(
    tmp_path: pathlib.Path,
) -> None:
    zarr = pytest.importorskip("zarr")
    path = _zp_array(tmp_path)

    async def go() -> None:
        node = await abczarr.open(path, mode="a", asynchronous=True)
        await node.update_attributes({"unit": "micrometer"})

    asyncio.run(go())
    assert zarr.open(path, mode="r").attrs["unit"] == "micrometer"


# --------------------------------------------------------------------------
# NodeMetadata stays the source of truth
# --------------------------------------------------------------------------


def test_metadata_update_attributes_is_immutable() -> None:
    meta = NodeMetadata(attributes={"a": 1})
    evolved = meta.update_attributes({"a": 1, "b": 2})
    # the evolved copy carries the new attributes; the original is untouched
    assert evolved.attributes == {"a": 1, "b": 2}
    assert meta.attributes == {"a": 1}

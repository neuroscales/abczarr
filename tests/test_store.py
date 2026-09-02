"""The store surface: the five primitives, the synthesized members, and the
default bagof-paths store in both its sync and async forms.

These run with no zarr / tensorstore backend installed -- the default store
is a directory under ``tmp_path``.
"""

import asyncio
import pathlib

import pytest

from abczarr.abc.capabilities import Support
from abczarr.abc.path import StorePath
from abczarr.abc.store import AsyncPathStore, AsyncStore, PathStore, Store

# --------------------------------------------------------------------------
# the base classes are abstract -- the primitives must be supplied
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [Store, AsyncStore])
def test_base_store_is_abstract(cls: type) -> None:
    with pytest.raises(TypeError):
        cls("/store")


def test_primitives_are_the_abstract_set() -> None:
    assert Store.__abstractmethods__ == frozenset(
        {"get", "set", "delete", "exists", "list_keys"}
    )
    assert AsyncStore.__abstractmethods__ == frozenset(
        {"get", "set", "delete", "exists", "list_keys"}
    )


# --------------------------------------------------------------------------
# PathStore: the five primitives over a directory
# --------------------------------------------------------------------------


def _store(tmp_path: pathlib.Path) -> PathStore:
    s = PathStore(str(tmp_path))
    s.set("zarr.json", b"{}")
    s.set("c/0/0", b"chunk-0-0")
    s.set("c/0/1", b"chunk-0-1")
    s.set("c/1/0", b"chunk-1-0")
    return s


def test_get_reads_back_what_was_set(tmp_path: pathlib.Path) -> None:
    s = _store(tmp_path)
    assert s.get("zarr.json") == b"{}"
    assert s.get("c/0/0") == b"chunk-0-0"


def test_get_missing_key_is_none(tmp_path: pathlib.Path) -> None:
    assert PathStore(str(tmp_path)).get("absent") is None


def test_set_creates_nested_parents(tmp_path: pathlib.Path) -> None:
    s = PathStore(str(tmp_path))
    s.set("a/b/c/d", b"deep")
    assert s.get("a/b/c/d") == b"deep"


def test_exists_and_contains(tmp_path: pathlib.Path) -> None:
    s = _store(tmp_path)
    assert s.exists("c/0/0")
    assert not s.exists("c/9/9")
    assert "zarr.json" in s
    assert "absent" not in s


def test_delete_removes_and_is_forgiving(tmp_path: pathlib.Path) -> None:
    s = _store(tmp_path)
    s.delete("c/0/0")
    assert not s.exists("c/0/0")
    s.delete("never-existed")  # not an error


# --------------------------------------------------------------------------
# listing: the whole subtree, one directory level, and __iter__
# --------------------------------------------------------------------------


def test_list_keys_walks_the_whole_subtree(tmp_path: pathlib.Path) -> None:
    s = _store(tmp_path)
    assert sorted(s.list_keys()) == [
        "c/0/0", "c/0/1", "c/1/0", "zarr.json"
    ]
    assert sorted(s.list_keys("c/0")) == ["c/0/0", "c/0/1"]


def test_list_keys_missing_prefix_is_empty(
    tmp_path: pathlib.Path,
) -> None:
    assert list(PathStore(str(tmp_path)).list_keys("nowhere")) == []


def test_list_dir_names_one_level(tmp_path: pathlib.Path) -> None:
    s = _store(tmp_path)
    assert sorted(s.list_dir()) == ["c", "zarr.json"]
    assert sorted(s.list_dir("c")) == ["0", "1"]
    assert sorted(s.list_dir("c/0")) == ["0", "1"]


def test_iter_is_list_keys(tmp_path: pathlib.Path) -> None:
    s = _store(tmp_path)
    assert sorted(iter(s)) == sorted(s.list_keys())


# --------------------------------------------------------------------------
# synthesized: getsize, clear
# --------------------------------------------------------------------------


def test_getsize_reports_bytes_or_none(tmp_path: pathlib.Path) -> None:
    s = _store(tmp_path)
    assert s.getsize("c/0/0") == len(b"chunk-0-0")
    assert s.getsize("absent") is None


def test_clear_empties_the_store(tmp_path: pathlib.Path) -> None:
    s = _store(tmp_path)
    s.clear()
    assert list(s.list_keys()) == []


# --------------------------------------------------------------------------
# capability query and location
# --------------------------------------------------------------------------


def test_supports_reads_declared_and_unknown_is_false(
    tmp_path: pathlib.Path,
) -> None:
    s = PathStore(str(tmp_path))
    assert s.supports("listing") is True
    assert s.supports("writes") is True
    assert s.supports("deletes") is True
    assert s.supports("teleportation") is False


def test_support_reports_native_for_the_path_store(
    tmp_path: pathlib.Path,
) -> None:
    s = PathStore(str(tmp_path))
    assert s.support("listing") is Support.NATIVE
    assert s.supports("listing", native=True) is True
    assert s.support("teleportation") is Support.NONE


def test_native_is_the_backing_path(tmp_path: pathlib.Path) -> None:
    s = PathStore(str(tmp_path))
    assert s.native is s.store_path


def test_read_only_flag_comes_from_the_path(tmp_path: pathlib.Path) -> None:
    ro = PathStore(StorePath(str(tmp_path), read_only=True))
    assert ro.read_only is True
    assert PathStore(str(tmp_path)).read_only is False


def test_context_manager_closes(tmp_path: pathlib.Path) -> None:
    with PathStore(str(tmp_path)) as s:
        s.set("k", b"v")
    assert s.get("k") == b"v"


# --------------------------------------------------------------------------
# AsyncPathStore: the same behaviour, awaited
# --------------------------------------------------------------------------


def test_async_store_roundtrip(tmp_path: pathlib.Path) -> None:
    async def scenario() -> None:
        s = AsyncPathStore(str(tmp_path))
        await s.set("zarr.json", b"{}")
        await s.set("c/0/0", b"chunk")
        assert await s.get("zarr.json") == b"{}"
        assert await s.get("absent") is None
        assert await s.exists("c/0/0")
        assert not await s.exists("absent")
        keys = sorted([k async for k in s.list_keys()])
        assert keys == ["c/0/0", "zarr.json"]
        names = sorted([n async for n in s.list_dir("c")])
        assert names == ["0"]
        assert await s.getsize("c/0/0") == len(b"chunk")
        await s.delete("c/0/0")
        assert not await s.exists("c/0/0")

    asyncio.run(scenario())


def test_async_store_context_manager_and_clear(tmp_path: pathlib.Path) -> None:
    async def scenario() -> None:
        async with AsyncPathStore(str(tmp_path)) as s:
            await s.set("a", b"1")
            await s.set("b", b"2")
        await s.clear()
        assert [k async for k in s.list_keys()] == []

    asyncio.run(scenario())


def test_async_store_advertises_async(tmp_path: pathlib.Path) -> None:
    s = AsyncPathStore(str(tmp_path))
    assert s.supports("async") is True
    assert s.supports("listing") is True

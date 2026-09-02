"""StorePath / AsyncStorePath: the store layer over bagof.paths.

The store layer is now a thin subclass of ``bagof.paths.Path`` /
``AsyncPath`` that adds one bit of state, ``read_only``. These tests pin the
two things that make that swap correct: the flag rides onto derived paths,
and real I/O works -- on a local path (always) and on a cloud-style scheme
(``memory://`` via universal-pathlib, no network) when a backend is present.
"""

import asyncio
import pathlib

import pytest

from abczarr.abc.path import AsyncStorePath, StorePath

# --------------------------------------------------------------------------
# read_only flag + derivation (local, no backend needed)
# --------------------------------------------------------------------------


def test_read_only_defaults_false() -> None:
    assert StorePath("/store").read_only is False


def test_read_only_rides_onto_derived_paths() -> None:
    p = StorePath("/store/root", read_only=True)
    assert p.read_only is True
    assert (p / "sub").read_only is True
    assert p.joinpath("a", "b").read_only is True
    assert p.parent.read_only is True
    assert p.with_name("other").read_only is True


def test_reassignment_on_child_is_independent() -> None:
    p = StorePath("/store", read_only=True)
    child = p / "a"
    child.read_only = False
    assert child.read_only is False
    assert p.read_only is True  # a scalar flag is not shared


def test_identity_is_location_keyed_and_ignores_read_only() -> None:
    assert StorePath("/store/x") == StorePath("/store/x", read_only=True)
    assert hash(StorePath("/store/x")) == hash(
        StorePath("/store/x", read_only=True)
    )
    assert StorePath("/store/x") != StorePath("/store/y")


# --------------------------------------------------------------------------
# real local I/O through the store path
# --------------------------------------------------------------------------


def test_local_io_roundtrip(tmp_path: pathlib.Path) -> None:
    root = StorePath(str(tmp_path))
    f = root / "zarr.json"
    f.write_bytes(b'{"zarr_format": 3}')
    assert f.exists()
    assert f.read_bytes() == b'{"zarr_format": 3}'
    assert [p.name for p in root.iterdir()] == ["zarr.json"]


# --------------------------------------------------------------------------
# async surface
# --------------------------------------------------------------------------


def test_async_store_path_pairs_with_sync() -> None:
    a = AsyncStorePath("/store", read_only=True)
    assert a.read_only is True
    assert (a / "sub").read_only is True
    assert a._sync_type is StorePath


def test_async_local_io_roundtrip(tmp_path: pathlib.Path) -> None:
    root = AsyncStorePath(str(tmp_path))

    async def go() -> bytes:
        f = root / "zarr.json"
        await f.write_bytes(b'{"node_type": "group"}')
        assert await f.exists()
        return await f.read_bytes()

    assert asyncio.run(go()) == b'{"node_type": "group"}'


# --------------------------------------------------------------------------
# a cloud-style scheme, end to end, with no network (memory:// via upath)
# --------------------------------------------------------------------------


def test_memory_scheme_roundtrip() -> None:
    pytest.importorskip("upath")
    root = StorePath("memory://abczarr-store-test", read_only=False)
    assert root.protocol == "memory"
    assert (root / "a").read_only is False

    f = root / "zarr.json"
    f.write_bytes(b'{"zarr_format": 3}')
    assert f.exists()
    assert f.read_bytes() == b'{"zarr_format": 3}'
    assert [p.name for p in root.iterdir()] == ["zarr.json"]
    # the raw driver object stays reachable
    assert root.wrapped is not None

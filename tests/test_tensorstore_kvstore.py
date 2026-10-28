"""The TensorStore driver maps a location to the kvstore that holds its data.

A local path becomes TensorStore's file kvstore, a URL TensorStore reaches
(``gs://``, ``s3://``, ``http(s)://``) is passed on, and anything else --
fsspec's ``memory://`` above all -- is refused. Before, a path object at
``memory://`` was handed to the file kvstore (writing a local ``memory:``
directory), and a ``memory://`` string reached TensorStore's own memory
store, which is separate from the fsspec store the rest of abczarr reads.

Needs only tensorstore, so it also runs where zarr-python is absent.
"""

import asyncio
import os
import pathlib
import uuid

import numpy as np
import pytest

pytest.importorskip("tensorstore")

from bagof.paths import Path  # noqa: E402

import abczarr  # noqa: E402
from abczarr.drivers.tensorstore import (  # noqa: E402
    TensorStoreArray,
    _kvstore_spec,
)
from abczarr.errors import UnsupportedZarrOperation  # noqa: E402


def _memory_url() -> str:
    return "memory://abczarr-ts-" + uuid.uuid4().hex + "/a.zarr"


def _create(location: object, **kwargs: object) -> object:
    return abczarr.create(
        location, data=np.arange(6, dtype="float32").reshape(2, 3),
        driver="tensorstore", **kwargs,
    )


# --------------------------------------------------------------------------
# the kvstore spec built for each kind of location
# --------------------------------------------------------------------------


@pytest.mark.parametrize("location, expected", [
    ("gs://bucket/a.zarr", "gs://bucket/a.zarr"),
    ("gcs://bucket/a.zarr", "gs://bucket/a.zarr"),
    ("s3://bucket/a.zarr", "s3://bucket/a.zarr"),
    ("s3a://bucket/a.zarr", "s3://bucket/a.zarr"),
    ("https://host/a.zarr", "https://host/a.zarr"),
    ("file:///data/a.zarr", "file:///data/a.zarr"),
    ("local:///data/a.zarr", "file:///data/a.zarr"),
])
def test_a_reachable_url_is_passed_on(location: str, expected: str) -> None:
    assert _kvstore_spec(location) == expected


def test_a_local_path_becomes_the_file_kvstore() -> None:
    expected = {"driver": "file", "path": os.path.join("data", "a.zarr")}
    assert _kvstore_spec(os.path.join("data", "a.zarr")) == expected
    assert _kvstore_spec(pathlib.Path("data", "a.zarr")) == expected
    assert _kvstore_spec(Path(os.path.join("data", "a.zarr"))) == expected


def test_a_kvstore_dict_is_used_as_it_is() -> None:
    spec = {"driver": "memory", "path": "a.zarr/"}
    assert _kvstore_spec(spec) is spec


@pytest.mark.parametrize("location", [
    "memory://bucket/a.zarr",
    Path("memory://bucket/a.zarr"),
    "az://container/a.zarr",
    "simplecache::s3://bucket/a.zarr",
])
def test_an_unreachable_location_is_refused(location: object) -> None:
    with pytest.raises(UnsupportedZarrOperation, match="tensorstore"):
        _kvstore_spec(location)


# --------------------------------------------------------------------------
# a path object at memory:// is refused, not written to a local directory
# --------------------------------------------------------------------------


def test_create_at_a_memory_path_object_is_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(UnsupportedZarrOperation, match="memory://"):
        _create(Path(_memory_url()))
    assert list(tmp_path.iterdir()) == []


def test_create_at_a_memory_path_object_is_refused_async(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    async def go() -> object:
        return await _create(Path(_memory_url()), asynchronous=True)

    with pytest.raises(UnsupportedZarrOperation, match="memory://"):
        asyncio.run(go())
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# a memory:// string is refused, not split between two memory stores
# --------------------------------------------------------------------------


def test_create_at_a_memory_url_is_refused() -> None:
    url = _memory_url()
    with pytest.raises(UnsupportedZarrOperation, match="memory://"):
        _create(url, zarr_version=2, attributes={"a": 1})
    # nothing reached fsspec's memory store either, where the v2
    # attributes used to land apart from the array
    fsspec = pytest.importorskip("fsspec")
    name = url[len("memory://"):].split("/")[0]
    assert not [k for k in fsspec.filesystem("memory").store if name in k]


def test_create_at_a_memory_url_is_refused_async() -> None:
    async def go() -> object:
        return await _create(_memory_url(), asynchronous=True)

    with pytest.raises(UnsupportedZarrOperation, match="memory://"):
        asyncio.run(go())


def test_open_of_an_array_at_a_memory_url_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    # an array held in fsspec's memory store is refused, not looked for in
    # TensorStore's own memory store, where it does not exist
    fsspec = pytest.importorskip("fsspec")
    pytest.importorskip("upath")  # memory:// needs an fsspec-backed driver
    _create(str(tmp_path / "a.zarr"))
    url = _memory_url()
    fsspec.filesystem("memory").put(
        str(tmp_path / "a.zarr"), url, recursive=True
    )
    with pytest.raises(UnsupportedZarrOperation, match="memory://"):
        abczarr.open(url, driver="tensorstore")


# --------------------------------------------------------------------------
# a local path object still writes to the local file system
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrap", [pathlib.Path, Path], ids=["pathlib", "bagof"]
)
def test_create_at_a_local_path_object(
    tmp_path: pathlib.Path, wrap: type,
) -> None:
    root = tmp_path / "a.zarr"
    node = _create(wrap(str(root)))
    assert isinstance(node, TensorStoreArray)
    assert (root / "zarr.json").is_file()
    reopened = abczarr.open(wrap(str(root)), driver="tensorstore")
    np.testing.assert_array_equal(
        reopened[...], np.arange(6, dtype="float32").reshape(2, 3)
    )

"""The parsed OME metadata is memoized without ever going stale.

Parsing a node's OME-Zarr metadata into typed objects is the dominant cost
of reading the [ome][abczarr.abc.sync.ZarrNode.ome] property. The parse is
memoized against the attribute payload it was parsed from, so a repeated
read reuses it. The key is recomputed from the live attributes on every
read, so a change to the metadata, through abczarr or straight through the
backend, is always reflected.
"""

import pathlib

import pytest

zarr = pytest.importorskip("zarr")

import abczarr  # noqa: E402
from abczarr.ome import v0_5  # noqa: E402


def _image(name: str) -> "v0_5.OME":
    return v0_5.OME.from_json(
        {
            "version": "0.5",
            "multiscales": [
                {
                    "name": name,
                    "axes": [
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0]}
                            ],
                        }
                    ],
                }
            ],
        }
    )


def test_repeated_reads_reuse_the_parse(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "g.zarr")
    abczarr.open(root, mode="w").ome = _image("first")
    node = abczarr.open(root, mode="r")
    first = node.ome
    # The second read returns the identical parsed object rather than
    # parsing the attributes again.
    assert node.ome is first
    assert first.multiscales[0].name == "first"


def test_absent_ome_is_cached_as_none(tmp_path: pathlib.Path) -> None:
    node = abczarr.open(str(tmp_path / "g.zarr"), mode="w")
    assert node.ome is None
    assert node.ome is None


def test_cache_reflects_a_write_through_abczarr(
    tmp_path: pathlib.Path,
) -> None:
    node = abczarr.open(str(tmp_path / "g.zarr"), mode="a")
    node.ome = _image("first")
    assert node.ome.multiscales[0].name == "first"
    node.ome = _image("second")
    # The memoized parse is not returned; the new value is read instead.
    assert node.ome.multiscales[0].name == "second"


def test_cache_reflects_a_write_through_the_backend(
    tmp_path: pathlib.Path,
) -> None:
    node = abczarr.open(str(tmp_path / "g.zarr"), mode="a")
    node.ome = _image("first")
    assert node.ome.multiscales[0].name == "first"
    # Change the attribute directly on the backend object, bypassing abczarr
    # entirely. The next read still recomputes its key from the live
    # attributes, so it re-parses and reflects the change.
    payload = dict(node.native.attrs["ome"])
    payload["multiscales"] = [{**payload["multiscales"][0], "name": "third"}]
    node.native.attrs["ome"] = payload
    assert node.ome.multiscales[0].name == "third"


def test_cache_reflects_deletion(tmp_path: pathlib.Path) -> None:
    node = abczarr.open(str(tmp_path / "g.zarr"), mode="a")
    node.ome = _image("first")
    assert node.ome is not None
    del node.ome
    assert node.ome is None

"""The zarr-python driver reading Zarr v2-format data (through zarr 3.x).

Runs where zarr-python 3.x is installed (the coverage CI leg).
"""

import pathlib

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

import abczarr  # noqa: E402
from abczarr.drivers.zarr_python import ZarrPythonDriver  # noqa: E402
from abczarr.metadata.v2 import ArrayMetadata as V2ArrayMetadata  # noqa: E402


def _v2_store(tmp_path: pathlib.Path) -> str:
    root = str(tmp_path / "v2.zarr")
    group = zarr.open_group(root, mode="w", zarr_format=2)
    array = group.create_array(
        "x", shape=(6, 6), chunks=(3, 3), dtype="<f8"
    )
    array[:] = np.arange(36).reshape(6, 6)
    return root


# --------------------------------------------------------------------------
# reading a v2 array
# --------------------------------------------------------------------------


def test_open_a_v2_array(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open(_v2_store(tmp_path), mode="r")["x"]
    assert arr.shape == (6, 6)
    assert arr.zarr_version == 2
    assert np.asarray(arr[0, :3]).tolist() == [0.0, 1.0, 2.0]


def test_v2_metadata_is_v2_array_metadata(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open(_v2_store(tmp_path), mode="r")["x"]
    meta = arr.metadata
    assert isinstance(meta, V2ArrayMetadata)
    assert meta.shape == (6, 6)
    # its compressor is named as a v2 feature key
    assert any(f.startswith("v2:codec:") for f in meta.required_features())


# --------------------------------------------------------------------------
# v1/v2 codec support, probed from numcodecs
# --------------------------------------------------------------------------


def test_driver_probes_numcodecs_for_v2_features() -> None:
    driver = ZarrPythonDriver()
    assert driver.supports("v2:codec:blosc") is True
    assert driver.supports("v2:codec:gzip") is True
    # a filter that needs extra config is still recognised as provided
    assert driver.supports("v2:filter:delta") is True
    assert driver.supports("v2:codec:not_a_real_codec") is False


def test_driver_supports_v1_codecs() -> None:
    assert ZarrPythonDriver().supports("v1:codec:zlib") is True


def test_can_open_a_v2_array(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open(_v2_store(tmp_path), mode="r")["x"]
    verdict = ZarrPythonDriver().can_open(arr.metadata)
    assert bool(verdict) is True

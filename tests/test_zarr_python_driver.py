"""The zarr-python driver's feature declaration and selection.

Runs only where zarr-python 3.x is installed (the coverage CI leg); the
3.8/3.14 test legs install no backend and skip this module.
"""

import pytest

zarr = pytest.importorskip("zarr")

from abczarr.abc.capabilities import Support  # noqa: E402
from abczarr.api.registry import select_driver  # noqa: E402
from abczarr.drivers.zarr_python import ZarrPythonDriver  # noqa: E402
from abczarr.errors import UnsupportedZarrOperation  # noqa: E402
from abczarr.metadata import v3  # noqa: E402


def _array(codecs: list, **over: object) -> "v3.ArrayMetadata":
    data = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [8, 8],
        "data_type": "float32",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [4, 4]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "codecs": codecs,
        "fill_value": 0,
        "attributes": {},
    }
    data.update(over)
    return v3.ArrayMetadata.from_json(data)


_BYTES = {"name": "bytes", "configuration": {"endian": "little"}}
_ZSTD = {"name": "zstd", "configuration": {"level": 5}}


def test_driver_is_named_and_present() -> None:
    d = ZarrPythonDriver()
    assert d.name == "zarr-python"
    assert d._major >= 3


def test_coarse_capabilities_are_native() -> None:
    d = ZarrPythonDriver()
    assert d.capability("sharding") is Support.NATIVE
    assert d.capability("async") is Support.NATIVE
    assert d.supports("codecs_v3") is True


def test_probes_real_codecs_from_the_registry() -> None:
    d = ZarrPythonDriver()
    # codecs zarr-python 3.x registers
    assert d.capability("v3:codec:zstd") is Support.NATIVE
    assert d.capability("v3:codec:bytes") is Support.NATIVE
    # a codec no registry holds
    assert d.capability("v3:codec:not_a_real_codec") is Support.NONE


def test_chunk_grid_and_key_encoding_support() -> None:
    d = ZarrPythonDriver()
    assert d.capability("v3:chunk_grid:regular") is Support.NATIVE
    assert d.capability("v3:chunk_grid:rectilinear") is Support.NONE
    assert d.capability("v3:chunk_key_encoding:default") is Support.NATIVE


def test_a_malformed_feature_name_is_unsupported() -> None:
    d = ZarrPythonDriver()
    assert d.capability("not-a-feature-key") is Support.NONE
    assert d.capability("v9:codec:zstd") is Support.NONE


def test_can_open_and_select_a_supported_array() -> None:
    d = ZarrPythonDriver()
    meta = _array([_BYTES, _ZSTD])
    assert bool(d.can_open(meta)) is True
    assert select_driver(meta, [d]).name == "zarr-python"


def test_cannot_open_an_array_needing_a_missing_codec() -> None:
    d = ZarrPythonDriver()
    meta = _array([{"name": "not_a_real_codec"}])
    verdict = d.can_open(meta)
    assert bool(verdict) is False
    assert "v3:codec:not_a_real_codec" in verdict.missing


def test_select_names_the_missing_codec_when_none_can_open() -> None:
    d = ZarrPythonDriver()
    meta = _array([{"name": "not_a_real_codec"}])
    with pytest.raises(UnsupportedZarrOperation) as info:
        select_driver(meta, [d])
    assert "not_a_real_codec" in str(info.value)

"""The creation config accepts the compressors each Zarr version supports."""

import pytest
from bagof.converters.exceptions import ValueConversionError

from abczarr.config import ZarrArrayConfig


def test_v3_config_accepts_zstd() -> None:
    config = ZarrArrayConfig(shape=(4,), compressor="zstd")
    assert config.compressor == "zstd"


def test_v3_config_accepts_blosc_and_gzip() -> None:
    blosc = ZarrArrayConfig(shape=(4,), compressor="blosc")
    gzip = ZarrArrayConfig(shape=(4,), compressor="gzip")
    assert blosc.compressor == "blosc"
    assert gzip.compressor == "gzip"


def test_v3_config_rejects_an_unknown_compressor() -> None:
    with pytest.raises(ValueConversionError):
        ZarrArrayConfig(shape=(4,), compressor="not-a-real-codec")

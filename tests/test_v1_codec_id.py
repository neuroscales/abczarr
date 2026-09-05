"""Every v1 codec options class carries its numcodecs ``id`` as a ClassVar.

``LZ4CodecOptions`` was missing the ``id`` class attribute its siblings
declare, so ``getattr(instance, "id")`` fell back to ``None`` and only the
registry match recovered the codec name. This checks the whole family
exposes ``id`` uniformly.
"""

import pytest

from abczarr.metadata.v1.codecs.builtin import (
    BloscCodecOptions,
    GzipCodecOptions,
)
from abczarr.metadata.v1.codecs.extensions import (
    Bz2CodecOptions,
    LZ4CodecOptions,
    LZMACodecOptions,
    PCodecOptions,
    ZFPYCodecOptions,
    ZlibCodecOptions,
    ZstdCodecOptions,
)


@pytest.mark.parametrize(
    ("cls", "codec_id"),
    [
        (BloscCodecOptions, "blosc"),
        (GzipCodecOptions, "gzip"),
        (Bz2CodecOptions, "bz2"),
        (LZMACodecOptions, "lzma"),
        (LZ4CodecOptions, "lz4"),
        (PCodecOptions, "pcodec"),
        (ZFPYCodecOptions, "zfpy"),
        (ZlibCodecOptions, "zlib"),
        (ZstdCodecOptions, "zstd"),
    ],
)
def test_codec_options_expose_id_classvar(cls: type, codec_id: str) -> None:
    """The class attribute ``id`` names the numcodecs codec."""
    assert cls.id == codec_id


def test_lz4_codec_options_id() -> None:
    """``LZ4CodecOptions`` carries ``id == "lz4"`` like its siblings."""
    assert LZ4CodecOptions.id == "lz4"

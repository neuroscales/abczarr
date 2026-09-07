__all__ = [
    "Codec",
    "ArrayToArrayCodec",
    "ArrayToBytesCodec",
    "BytesToBytesCodec",
    "CompressorCodec",
]

# core
from abczarr._core import typing as tz
from abczarr._core.auto import autofrozen

# metadata
from abczarr.metadata.v3.extensions import Extension, TypedConfig


@autofrozen
class CodecConfig(TypedConfig):
    """This class is the base for a v3 codec's own configuration
    parameters."""


@autofrozen(extra_items=False)
class CodecConfigImpl(CodecConfig):
    """This class is the base for a codec configuration whose parameters
    are all declared."""

    def to_json(self) -> tz.JsonDict:
        # A codec configuration names only the fields the codec has, and the
        # codec schemas allow no other keys and no nulls. An unset optional
        # field (None) is therefore omitted rather than written as null.
        return {
            key: value
            for key, value in super().to_json().items()
            if value is not None
        }


def _v2_id(name: str) -> str:
    """The numcodecs id a v3 codec *name* maps back to.

    A v2 filter with no dedicated v3 codec is carried in v3 under the
    numcodecs extension namespace (``"numcodecs.delta"``); stripping that
    prefix recovers the original numcodecs id (``"delta"``) so a
    v2 -> v3 -> v2 round trip is lossless.
    """
    prefix = "numcodecs."
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


@autofrozen
class Codec(Extension):
    """A `Codec` names a codec through ``name`` and carries that codec's
    own ``configuration``.

    Each `Codec` represents one stage of an array's codec pipeline. See
    [`ArrayToArrayCodec`][abczarr.metadata.v3.codecs.base.ArrayToArrayCodec],
    [`ArrayToBytesCodec`][abczarr.metadata.v3.codecs.base.ArrayToBytesCodec]
    and
    [`BytesToBytesCodec`][abczarr.metadata.v3.codecs.base.BytesToBytesCodec]
    for what each stage does.

    See
    [`abczarr.metadata.v2.codecs.base.Codec`][abczarr.metadata.v2.codecs.base.Codec]
    for a worked example comparing the same codec across the v1, v2 and
    v3 metadata models.
    """

    configuration: CodecConfig

    def to_json(self) -> tz.JsonDict:
        # A codec with no configuration parameters (crc32c, or a bytes codec
        # for a single-byte dtype) is written as a bare name, not with an
        # empty configuration object.
        obj = super().to_json()
        if obj.get("configuration") == {}:
            obj.pop("configuration")
        return obj

    def to_version(self, version: tz.ZarrVersion) -> "Codec":
        if version == 3:
            return self
        if version == 1:
            # route through v2 -- v1 and v2 share the numcodecs model
            return self.to_version(2).to_version(1)
        if version == 2:
            from abczarr.metadata.v2 import Codec as CodecV2
            as_dict = self.to_json()
            if isinstance(as_dict, str):
                as_dict = {"id": _v2_id(as_dict)}
            else:
                config = as_dict.get("configuration") or {}
                as_dict = {"id": _v2_id(as_dict["name"]), **config}
            return CodecV2.from_json(as_dict)
        else:
            raise ValueError(f"Unsupported version: {version}")


@autofrozen
class ArrayToArrayCodec(Codec):
    """Transforms an array into another array.

    Runs before the pipeline's single array-to-bytes codec, for example
    by transposing axes, reshaping, or rounding values.
    """


@autofrozen
class ArrayToBytesCodec(Codec):
    """Serializes an array to bytes, or parses bytes back into an array.

    Every codec pipeline has exactly one array-to-bytes codec, forming
    the boundary between the array-to-array codecs before it and the
    bytes-to-bytes codecs after.
    """


@autofrozen
class BytesToBytesCodec(Codec):
    """Transforms a byte sequence into another byte sequence.

    Runs after the pipeline's array-to-bytes codec. A typical example is
    a compressor, or a checksum appended for integrity.
    """


@autofrozen
class CompressorCodec(BytesToBytesCodec):
    """This class is the base for a bytes-to-bytes codec that compresses
    its input."""

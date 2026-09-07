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
        """Serialize this configuration to its JSON representation.

        An unset optional field is omitted from the result rather
        than written as `null`, since the codec schemas allow no
        keys beyond the ones the codec declares.

        Returns
        -------
        dict
            The JSON-compatible representation of this configuration.
        """
        return {
            key: value
            for key, value in super().to_json().items()
            if value is not None
        }


def _v2_id(name: str) -> str:
    """Recover the numcodecs id a v3 codec *name* maps back to.

    A v2 filter with no dedicated v3 codec is carried in v3 under the
    numcodecs extension namespace, for example ``"numcodecs.delta"``.
    Stripping that prefix recovers the original numcodecs id
    (``"delta"``), so a v2-to-v3-to-v2 round trip is lossless.
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

    Attributes
    ----------
    name : str
        The name of the codec, such as ``"gzip"`` or ``"bytes"``.
    configuration : CodecConfig
        The codec's own parameters.
    """

    configuration: CodecConfig

    def to_json(self) -> tz.JsonDict:
        """Serialize this codec to its JSON representation.

        A codec with no configuration parameters, such as `crc32c`
        or a bytes codec for a single-byte dtype, is written as a
        bare name, not with an empty configuration object.

        Returns
        -------
        dict
            The JSON-compatible representation of this codec.
        """
        obj = super().to_json()
        if obj.get("configuration") == {}:
            obj.pop("configuration")
        return obj

    def to_version(self, version: tz.ZarrVersion) -> "Codec":
        """Convert this codec to another Zarr version.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.

        Returns
        -------
        Codec
            The equivalent codec for *version*: this object unchanged
            for version 3, or the corresponding v1 or v2 codec object
            otherwise.

        Raises
        ------
        ValueError
            If *version* is not 1, 2 or 3.
        """
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

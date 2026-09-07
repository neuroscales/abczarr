"""Extension codecs for Zarr v3, beyond the required built-in set.

Each class holds one codec's own parameters and, where a Zarr v1 or
v2 numcodecs equivalent exists, converts to it.
"""

__all__ = [
    "BitroundCodec",
    "CastValueCodec",
    "ConditionalCodec",
    "N5DefaultCodec",
    "PackBitsCodec",
    "ScaleOffsetCodec",
    "VLenBytesCodec",
    "VLenUTF8Codec",
    "ReshapeCodec",
    "ZfpCodec",
    "ZstdCodec",
]

# stdlib
import re

# dependencies
import numpy as np
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto import autofrozen

# locals
from abczarr.metadata.base import Metadata, register_subclass

from .base import (
    ArrayToArrayCodec,
    ArrayToBytesCodec,
    Codec,
    CodecConfigImpl,
    CompressorCodec,
)
from .builtin import BytesCodec, TransposeCodec


@autofrozen
class BitroundConfig(CodecConfigImpl):
    """Holds the bitround codec's parameters: how many mantissa bits to
    keep.

    Attributes
    ----------
    keepbits : int
        The number of mantissa bits kept. The rest are zeroed.
    """

    keepbits: int = 1


@register_subclass(name=re.compile(r"(?:bitround|numcodecs\.bitround)"))
@autofrozen
class BitroundCodec(ArrayToArrayCodec):
    """Rounds a float's mantissa to `configuration.keepbits` bits,
    zeroing the rest.

    The rounding is lossy and improves the compressibility of
    floating-point data by discarding low-order precision the data does
    not need.

    Attributes
    ----------
    name : str
        Either ``"bitround"`` or ``"numcodecs.bitround"``.
    configuration : BitroundConfig
        The codec's parameters.
    """

    name: tx.Literal["bitround", "numcodecs.bitround"]
    configuration: BitroundConfig


_ScalarMapItem = tx.Union[tz.JsonScalar, str]
_ScalarMap = tx.Tuple[tx.Tuple[_ScalarMapItem, _ScalarMapItem], ...]


@autofrozen
class ScalarMap(Metadata):
    """This class represents an explicit value-to-value mapping used by
    a cast-value codec.

    `encode` and `decode` each list ``(from, to)`` pairs that map
    individual values falling outside the ordinary numeric cast. An
    example is a sentinel value or a small set of categorical codes.

    Attributes
    ----------
    encode : tuple of tuple
        The ``(from, to)`` pairs applied on encode.
    decode : tuple of tuple
        The ``(from, to)`` pairs applied on decode.
    """

    encode: _ScalarMap
    decode: _ScalarMap


@autofrozen
class CastValueConfig(CodecConfigImpl):
    """Holds the cast-value codec's parameters: target type, rounding,
    and value mapping.

    Attributes
    ----------
    data_type : numpy.dtype
        The dtype values are cast to on encode.
    rounding : str
        How a value that does not fit `data_type` exactly is rounded:
        one of ``"nearest-even"``, ``"towards-zero"``,
        ``"towards-positive"``, ``"towards-negative"`` or
        ``"nearest-away"``.
    out_of_range : str or None
        How a value outside the range of `data_type` is handled:
        ``"clamp"``, ``"wrap"``, or `None` for unspecified behavior.
    scalar_map : ScalarMap
        An explicit mapping for values that need one instead of the
        ordinary numeric cast.
    """

    data_type: np.dtype
    rounding: tx.Literal[
        "nearest-even",
        "towards-zero",
        "towards-positive",
        "towards-negative",
        "nearest-away",
    ] = "nearest-even"
    out_of_range: tx.Optional[tx.Literal["clamp", "wrap"]] = None
    scalar_map: ScalarMap


@register_subclass(name="cast_value")
@autofrozen
class CastValueCodec(ArrayToArrayCodec):
    """Casts an array's values to `configuration.data_type`, and back
    on decode.

    `configuration.rounding` and `configuration.out_of_range` control
    how a value that does not fit the target type exactly is rounded
    and clamped or wrapped. `configuration.scalar_map` maps any
    values that need an explicit, non-numeric mapping instead.

    Attributes
    ----------
    name : str
        Always ``"cast_value"``.
    configuration : CastValueConfig
        The codec's parameters.
    """

    name: tx.Literal["cast_value"]
    configuration: CastValueConfig


@autofrozen
class ConditionalConfig(CodecConfigImpl):
    """Holds the conditional codec's parameters: the candidate codecs to
    choose among.

    Attributes
    ----------
    codecs : tuple of Codec
        The candidate codecs the conditional codec chooses among.
    """

    codecs: tx.Tuple[Codec, ...]


@register_subclass(name="conditional")
@autofrozen
class ConditionalCodec(Codec):
    """Selects one of several candidate codecs, based on the array to
    which it is applied.

    Lets a single pipeline entry vary by a property of the array, such
    as its data type, instead of naming one codec unconditionally.

    Attributes
    ----------
    name : str
        Always ``"conditional"``.
    configuration : ConditionalConfig
        The codec's parameters.
    """

    name: tx.Literal["conditional"]
    configuration: ConditionalConfig


class N5DefaultCodecList(list):
    """The fixed codec chain of an `n5_default` codec.

    A transpose codec, then a big-endian bytes codec, then any
    number of trailing codecs. Each element is converted to its
    concrete codec type on construction. The list holds codec
    objects, not the raw dicts it was built from.

    Parameters
    ----------
    codecs : iterable
        The codecs to store, as codec objects or as JSON-like mappings.
        The first is read as a transpose codec, the second as a big-endian
        bytes codec, and the rest as arbitrary codecs.

    Raises
    ------
    ValueError
        If fewer than two codecs are given, or the second codec is
        little-endian.
    """

    def __init__(self, codecs: tx.Iterable[Codec]) -> None:
        codecs = list(codecs)
        if len(codecs) < 2:
            raise ValueError(
                f"N5DefaultCodecList must have at least 2 codecs, "
                f"got {len(codecs)}"
            )
        first, second, *rest = codecs
        first = TransposeCodec(**first)
        second = BytesCodec(**second)
        if second.configuration.endian == "little":
            raise ValueError(
                f"N5DefaultCodecList second codec must be big-endian, "
                f"got {second.configuration.endian}"
            )
        rest = [Codec(**c) for c in rest]
        super().__init__([first, second, *rest])


@autofrozen
class N5DefaultConfig(CodecConfigImpl):
    """Holds the n5_default codec's parameters: the fixed chain of
    codecs it applies.

    Attributes
    ----------
    codecs : N5DefaultCodecList
        The codec chain: a transpose codec, a big-endian bytes codec,
        then any number of trailing codecs.
    """

    codecs: N5DefaultCodecList


@register_subclass(name="n5_default")
@autofrozen
class N5DefaultCodec(Codec):
    """Applies the codec chain an N5 array uses by default, for interop
    with N5.

    The chain is fixed to a transpose codec followed by a big-endian
    bytes codec, with any number of trailing codecs, matching how the
    N5 format lays out a chunk.

    Attributes
    ----------
    name : str
        Always ``"n5_default"``.
    configuration : N5DefaultConfig
        The codec's parameters.
    """

    name: tx.Literal["n5_default"]
    configuration: N5DefaultConfig


@autofrozen
class PackBitsConfig(CodecConfigImpl):
    """Holds the packbits codec's parameters: padding placement and the
    used bit range.

    Attributes
    ----------
    padding_encoding : str
        Where the padding bits sit within the packed bytes:
        ``"first_byte"``, ``"last_byte"`` or ``"none"``.
    first_bit : int or None
        The index of the first meaningful bit, when padding is
        present.
    last_bit : int or None
        The index of the last meaningful bit, when padding is
        present.
    """

    padding_encoding: tx.Literal["first_byte", "last_byte", "none"] = "none"
    first_bit: tx.Optional[int]
    last_bit: tx.Optional[int]


@register_subclass(name="packbits")
@autofrozen
class PackBitsCodec(ArrayToBytesCodec):
    """Packs a boolean array down to one bit per element for storage.

    Attributes
    ----------
    name : str
        Always ``"packbits"``.
    configuration : PackBitsConfig
        The codec's parameters.
    """

    name: tx.Literal["packbits"]
    configuration: PackBitsConfig


@autofrozen
class ScaleOffsetConfig(CodecConfigImpl):
    """Holds the scale-offset codec's parameters: the scale and offset
    to apply.

    Attributes
    ----------
    offset : int or float
        The value subtracted before scaling.
    scale : int or float
        The factor the offset value is multiplied by.
    """

    offset: tz.JsonNumber
    scale: tz.JsonNumber


@register_subclass(name="scale_offset")
@autofrozen
class ScaleOffsetCodec(ArrayToArrayCodec):
    """Quantizes values as ``(value - offset) * scale``, and reverses
    the transform on decode.

    The transform is lossy and useful for storing a bounded
    floating-point range in fewer bits.

    Attributes
    ----------
    name : str
        Always ``"scale_offset"``.
    configuration : ScaleOffsetConfig
        The codec's parameters.
    """

    name: tx.Literal["scale_offset"]
    configuration: ScaleOffsetConfig


@register_subclass(name="vlen-bytes")
@autofrozen
class VLenBytesCodec(ArrayToBytesCodec):
    """Serializes an array of variable-length byte strings to bytes.

    Attributes
    ----------
    name : str
        Always ``"vlen-bytes"``.
    """

    name: tx.Literal["vlen-bytes"]


@register_subclass(name="vlen-utf8")
@autofrozen
class VLenUTF8Codec(ArrayToBytesCodec):
    """Serializes an array of variable-length UTF-8 strings to bytes.

    Attributes
    ----------
    name : str
        Always ``"vlen-utf8"``.
    """

    name: tx.Literal["vlen-utf8"]


#: One axis of a reshape target: a size (with -1 for "the rest"), or a group
#: of sizes to split that axis into.
_ReshapeAxis = tx.Union[int, tx.Tuple[int, ...]]


@autofrozen
class ReshapeConfig(CodecConfigImpl):
    """Holds the reshape codec's parameters: the target shape.

    Attributes
    ----------
    shape : tuple of (int or tuple of int)
        The target shape. An axis size of ``-1`` stands for whatever
        size makes the reshape fit, and a group of sizes splits that
        axis into several.
    """

    shape: tx.Tuple[_ReshapeAxis, ...]


@register_subclass(name="reshape")
@autofrozen
class ReshapeCodec(ArrayToArrayCodec):
    """Reshapes an array to `configuration.shape` before the rest of
    the pipeline.

    The reshape is reversed on decode, so the array's logical shape
    is unchanged.

    Attributes
    ----------
    name : str
        Always ``"reshape"``.
    configuration : ReshapeConfig
        The codec's parameters.
    """

    name: tx.Literal["reshape"]
    configuration: ReshapeConfig


#: zfp's five modes; each carries only its own parameters.
_ZfpMode = tx.Literal[
    "reversible", "expert", "fixed_accuracy", "fixed_rate", "fixed_precision"
]


@autofrozen
class ZfpConfig(CodecConfigImpl):
    """Holds the ZFP codec's parameters: the mode, and that mode's own
    parameters.

    Only the parameters belonging to `mode` apply.
    ``"expert"`` takes `minbits`, `maxbits`, `maxprec` and `minexp`.
    ``"fixed_accuracy"`` takes `tolerance`. ``"fixed_rate"`` takes
    `rate`. ``"fixed_precision"`` takes `precision`.
    ``"reversible"`` takes none of them.

    Attributes
    ----------
    mode : str
        The accuracy mode ZFP compresses in: ``"reversible"``,
        ``"expert"``, ``"fixed_accuracy"``, ``"fixed_rate"`` or
        ``"fixed_precision"``.
    minbits : int or None
        The minimum number of bits per block, in expert mode.
    maxbits : int or None
        The maximum number of bits per block, in expert mode.
    maxprec : int or None
        The maximum precision retained, in expert mode.
    minexp : int or None
        The minimum bit-plane coding exponent, in expert mode.
    tolerance : float or None
        The maximum absolute error allowed, in fixed-accuracy mode.
    rate : float or None
        The number of bits per value, in fixed-rate mode.
    precision : int or None
        The number of bits of precision retained, in
        fixed-precision mode.
    """

    # zfp picks a mode, and each mode carries only its own parameters: expert
    # takes minbits/maxbits/maxprec/minexp; fixed_accuracy takes tolerance;
    # fixed_rate takes rate; fixed_precision takes precision; reversible takes
    # none. The parameters of the other modes stay None and are omitted on
    # serialization, so each mode round-trips exactly as the spec writes it.
    mode: _ZfpMode
    minbits: tx.Optional[int] = None
    maxbits: tx.Optional[int] = None
    maxprec: tx.Optional[int] = None
    minexp: tx.Optional[int] = None
    tolerance: tx.Optional[float] = None
    rate: tx.Optional[float] = None
    precision: tx.Optional[int] = None


@register_subclass(name="zfp")
@autofrozen
class ZfpCodec(ArrayToBytesCodec):
    """Compresses floating-point arrays with ZFP.

    `configuration.mode` selects the accuracy/size trade-off.
    Compression is lossy, except in ``"reversible"`` mode.

    Attributes
    ----------
    name : str
        Always ``"zfp"``.
    configuration : ZfpConfig
        The codec's parameters.
    """

    name: tx.Literal["zfp"]
    configuration: ZfpConfig


@autofrozen
class ZstdConfig(CodecConfigImpl):
    """Holds the Zstd codec's parameters: compression level and an
    optional checksum.

    Attributes
    ----------
    level : int
        The compression level. Zstd's levels run from -131072 to 22,
        wider than the 0-9 range the core compressors use, and
        negative for the fast modes.
    checksum : bool
        Whether a checksum of each frame is appended for integrity.
    """

    # The v3 zstd codec schema requires `level` (checksum is optional) and
    # declares no defaults, so an implementation picks its own. abczarr
    # defaults level to 0, matching zarr-python, and writes both fields.
    level: int = 0
    checksum: bool = False

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        """Convert this configuration to another Zarr version's Zstd
        codec.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.

        Returns
        -------
        Metadata
            The equivalent Zstd codec for `version`. Only the
            compression level carries over to v1 and v2, since their
            numcodecs zstd codec has no checksum option.

        Raises
        ------
        ValueError
            If `version` is not 1, 2 or 3.
        """
        if version == 3:
            return self
        if version == 2:
            from abczarr.metadata import v2
            # v2's numcodecs zstd carries only the level
            return v2.ZstdCodec(id="zstd", level=self.level)
        if version == 1:
            from abczarr.metadata import v1
            # v1's numcodecs zstd carries only the level
            return v1.ZstdCodecOptions(level=self.level)
        raise ValueError(f"Unsupported version: {version}")


@register_subclass(name="zstd")
@autofrozen
class ZstdCodec(CompressorCodec):
    """Compresses each chunk with Zstandard at a configurable
    compression level.

    `configuration.checksum`, when set, appends a checksum of each
    frame for integrity.

    Attributes
    ----------
    name : str
        Always ``"zstd"``.
    configuration : ZstdConfig
        The codec's parameters.
    """

    name: tx.Literal["zstd"]
    configuration: ZstdConfig

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        """Convert this codec to another Zarr version.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.

        Returns
        -------
        Metadata
            The equivalent Zstd codec for `version`.
        """
        if version == 3:
            return self
        return self.configuration.to_version(version)

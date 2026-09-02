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
    ...


@autofrozen(extra_items=False)
class CodecConfigImpl(CodecConfig):
    def to_dict(self) -> tz.JSONDict:
        # A codec configuration names only the fields the codec has, and the
        # codec schemas allow no other keys and no nulls. An unset optional
        # field (None) is therefore omitted rather than written as null.
        return {
            key: value
            for key, value in super().to_dict().items()
            if value is not None
        }


@autofrozen
class Codec(Extension):
    configuration: CodecConfig

    def to_dict(self) -> tz.JSONDict:
        # A codec with no configuration parameters (crc32c, or a bytes codec
        # for a single-byte dtype) is written as a bare name, not with an
        # empty configuration object.
        obj = super().to_dict()
        if obj.get("configuration") == {}:
            obj.pop("configuration")
        return obj

    def to_version(self, version: tz.ZarrVersion) -> "Codec":
        if version == 3:
            return self
        if version == 2:
            from abczarr.metadata.v2 import Codec as CodecV2
            as_dict = self.to_dict()
            if isinstance(as_dict, str):
                as_dict = {"id": as_dict}
            else:
                config = as_dict.get("configuration") or {}
                as_dict = {"id": as_dict["name"], **config}
            return CodecV2.from_dict(as_dict)
        else:
            raise ValueError(f"Unsupported version: {version}")


@autofrozen
class ArrayToArrayCodec(Codec):
    ...


@autofrozen
class ArrayToBytesCodec(Codec):
    ...


@autofrozen
class BytesToBytesCodec(Codec):
    ...


@autofrozen
class CompressorCodec(BytesToBytesCodec):
    ...

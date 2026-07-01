__all__ = [
    "Codec",
    "ArrayToArrayCodec",
    "ArrayToBytesCodec",
    "BytesToBytesCodec",
]

# stdlib
from collections import abc

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.attrs import autofrozen

# locals
from ..extensions import Extension, TypedConfig


@autofrozen
class CodecConfig(TypedConfig):
    ...


@autofrozen
class Codec(Extension):
    configuration: tx.Union[CodecConfig, tz.JSONDict]

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

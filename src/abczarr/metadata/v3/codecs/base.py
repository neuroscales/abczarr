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

    @classmethod
    def from_version(
        cls, obj: tx.Any, version: tx.Optional[tz.ZarrVersion] = None
    ) -> "Codec":
        # short-circuit if already the right type
        if isinstance(obj, cls):
            return obj

        # short-circuit if it is a codec name
        if isinstance(obj, str):
            return cls(id=obj)

        # guess version
        if version is None:

            # guess from keys
            if isinstance(obj, abc.Mapping):
                if "id" in obj:
                    version = 2
                elif "name" in obj:
                    version = 3

            # guess from type
            else:
                from abczarr.metadata.v3.codecs.base import Codec as CodecV3
                if isinstance(obj, CodecV3):
                    version = 3

        if version is None:
            raise ValueError("Cannot determine version from object")

        if version == 2:
            return cls.from_dict(obj)

        if version == 3:
            from abczarr.metadata.v3.codecs.base import Codec as CodecV3
            return CodecV3.from_version(obj, version).to_version(2)

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

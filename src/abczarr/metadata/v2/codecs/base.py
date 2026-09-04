__all__ = [
    "Codec",
    "CodecImpl",
]

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen
from abczarr._core.metadata import Metadata


@autofrozen(extra_items=tz.FrozenJson)
class Codec(Metadata):
    id: str

    def to_version(self, version: tz.ZarrVersion) -> "Codec":
        if version == 2:
            return self
        if version == 1:
            # v1 and v2 share the numcodecs model: a v2 codec is a valid v1
            # codec, carried as v1 codec options ({id, **options}).
            from abczarr.metadata.v1 import CodecOptions
            as_dict = self.to_json()
            if isinstance(as_dict, str):
                as_dict = {"id": as_dict}
            return CodecOptions.from_json(as_dict)
        if version == 3:
            from abczarr.metadata.v3 import Codec as CodecV3
            as_dict = self.to_json()
            if isinstance(as_dict, str):
                as_dict = {"id": as_dict}
            else:
                config = as_dict
                as_dict = {"name": config.pop("id"), "configuration": config}
            return CodecV3.from_json(as_dict)
        else:
            raise ValueError(f"Unsupported version: {version}")


@autofrozen(extra_items=False)
class CodecImpl(Codec):
    ...

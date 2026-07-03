__all__ = [
    "Filter",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen
from abczarr._core.metadata import Metadata


@autofrozen(extra_items=tz.FrozenJSON)
class Filter(Metadata):
    id: str

    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            from abczarr.metadata.v3 import Codec as CodecV3
            as_dict = self.to_dict()
            if isinstance(as_dict, str):
                as_dict = {"id": as_dict}
            else:
                config = as_dict
                as_dict = {"name": config.pop("id"), "configuration": config}
            return CodecV3.from_dict(as_dict)
        else:
            raise ValueError(f"Unsupported version: {version}")


@autofrozen(extra_items=False)
class FilterImpl(Filter):
    ...

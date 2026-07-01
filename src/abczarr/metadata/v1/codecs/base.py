__all__ = [
    "CodecOptions",
]
# core
from abczarr._core import typing as tz
from abczarr._core.attrs import autofrozen
from abczarr.metadata.base import Metadata


@autofrozen(extra_items=tz.JSON)
class CodecOptions(Metadata):

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        if version == 1:
            return self

        # guess codec ID from object type
        id = None
        registry = self._registry()
        for match, cls in registry.items():
            if type(self) is cls:
                id = match.get("id")
                break

        if id is None:
            raise ValueError(
                f"Cannot convert {type(self).__name__} to version {version}."
                "Unknown codec ID."
            )

        if version == 2:
            from abczarr.metadata.v2 import Codec

            return Codec(
                id=id,
                cname=self.cname,
                clevel=self.clevel,
                shuffle=self.shuffle,
                blocksize=self.blocksize,
                typesize=self.typesize,
            )

        elif version == 3:
            from abczarr.metadata.v3 import Codec

            return Codec(
                name=id,
                configuration=self.to_dict()
            )

        else:
            raise ValueError(f"Cannot convert BloscCodecOption to version {version}")


@autofrozen(extra_items=False)
class CodecOptionsImpl(CodecOptions):
    ...

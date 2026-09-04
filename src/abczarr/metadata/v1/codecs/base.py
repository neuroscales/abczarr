__all__ = [
    "CodecOptions",
]
# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen
from abczarr.metadata.base import Metadata


@autofrozen(extra_items=tz.FrozenJSON)
class CodecOptions(Metadata):

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        if version == 1:
            return self

        # A v1 codec is named by its numcodecs id. Depending on how the
        # instance was built, that id is either a field carried in `to_dict`,
        # a class attribute (subclasses declare it as a ClassVar), or only
        # recoverable from the registry match that selected the subclass --
        # whose keys are tuples of ``(field, value)`` pairs, so each is
        # wrapped in ``dict`` before reading ``"id"``.
        options = dict(self.to_dict())
        id = options.pop("id", None) or getattr(self, "id", None)
        if id is None:
            for match, cls in self._registry().items():
                if type(self) is cls:
                    id = dict(match).get("id")
                    break

        if id is None:
            raise ValueError(
                f"Cannot convert {type(self).__name__} to version {version}: "
                "unknown codec id."
            )

        # v1 and v2 share the numcodecs model: rebuild the codec as a v2
        # numcodecs codec ({id, **options}), then let v2 map it onward to v3.
        from abczarr.metadata.v2 import Codec as CodecV2

        codec_v2 = CodecV2.from_dict({"id": id, **options})
        if version == 2:
            return codec_v2
        if version == 3:
            return codec_v2.to_version(3)
        raise ValueError(f"Unsupported version: {version}")


@autofrozen(extra_items=False)
class CodecOptionsImpl(CodecOptions):
    ...

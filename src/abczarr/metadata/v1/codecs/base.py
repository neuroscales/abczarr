__all__ = [
    "CodecOptions",
]
# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen
from abczarr.metadata.base import Metadata


@autofrozen(extra_items=tz.FrozenJson)
class CodecOptions(Metadata):
    """The options of a Zarr v1 compressor, keyed by numcodecs id.

    A Zarr v1 array names its compressor through the separate
    `compression` field and carries the compressor's own options as
    a `CodecOptions` object in `compression_opts`. A built-in
    compressor, such as blosc or gzip, is represented by one of its
    declared subclasses. An unrecognized compressor's options are
    still readable through this class, whose fields are whatever
    keys the document carries.
    """

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        """Convert these codec options to another Zarr version.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.

        Returns
        -------
        Metadata
            The equivalent codec for `version`: this object unchanged
            for version 1, or a v2 or v3 codec object otherwise.

        Raises
        ------
        ValueError
            If `version` is not 1, 2 or 3, or if the compressor's
            numcodecs id cannot be determined.
        """
        if version == 1:
            return self

        # A v1 codec is named by its numcodecs id. Depending on how the
        # instance was built, that id is either a field carried in `to_json`,
        # a class attribute (subclasses declare it as a ClassVar), or only
        # recoverable from the registry match that selected the subclass --
        # whose keys are tuples of ``(field, value)`` pairs, so each is
        # wrapped in ``dict`` before reading ``"id"``.
        options = dict(self.to_json())
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
        # Local import: the version packages reference each other for
        # cross-version conversion (v1 -> v2 -> v3 and back), so a
        # module-level import between them would be a cycle.
        from abczarr.metadata.v2 import Codec as CodecV2

        codec_v2 = CodecV2.from_json({"id": id, **options})
        if version == 2:
            return codec_v2
        if version == 3:
            return codec_v2.to_version(3)
        raise ValueError(f"Unsupported version: {version}")


@autofrozen(extra_items=False)
class CodecOptionsImpl(CodecOptions):
    """This class is the base for a v1 codec's options whose fields are
    all declared, not open-ended."""

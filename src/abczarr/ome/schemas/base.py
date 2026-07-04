__all__ = ["OMESchemaItem", "OME"]

import typing_extensions as tx

from abczarr._core import typing as tz


class OMESchemaItem(tx.TypedDict, total=False, extra_items=tz.JSON):
    ...


class OME(OMESchemaItem):
    ...


class OMEAttributes(OMESchemaItem):
    ...

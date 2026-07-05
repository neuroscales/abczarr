# stdlib
import numbers
from collections import abc
from types import NoneType

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

T = tx.TypeVar("T", bound=tx.Any)
ClassDecorator = tx.Callable[[tx.Type[T]], tx.Type[T]]
MagicRegistry = tx.Dict[tx.Any, tx.Type[T]]

ITERABLE = tx.TypeVar("ITERABLE", bound=abc.Iterable, default=abc.Iterable)
SEQUENCE = tx.TypeVar("SEQUENCE", bound=abc.Sequence, default=abc.Sequence)
MAPPING = tx.TypeVar("MAPPING", bound=abc.Mapping, default=abc.Mapping)
NUMBER = tx.TypeVar("NUMBER", bound=numbers.Number, default=numbers.Number)
TUPLE = tx.TypeVar("TUPLE", bound=tx.Tuple, default=tuple)
STR = tx.TypeVar("STR", bound=str, default=str)
NONETYPE = tx.TypeVar("NONETYPE", bound=NoneType, default=NoneType)
DTYPE = tx.TypeVar("DTYPE", bound=np.dtype, default=np.dtype)

KEY = tx.TypeVar("KEY", bound=tx.Any, default=tx.Any)
VAL = tx.TypeVar("VAL", bound=tx.Any, default=tx.Any)
_MAPPINGLIKE = tx.Union[
    tx.Mapping[KEY, VAL],
    tx.Iterable[tx.Tuple[KEY, VAL]],
]
_NUMBERLIKE = tx.Union[numbers.Number, np.number, np.bool_]

TO = tx.TypeVar("TO", bound=tx.Any, default=tx.Any)
FROM = tx.TypeVar("FROM", bound=tx.Any, default=tx.Any)
SEQLIKE = tx.TypeVar("SEQLIKE", bound=abc.Sequence, default=abc.Sequence)
NONELIKE = tx.TypeVar("NONELIKE", bound=NoneType, default=NoneType)
DICTLIKE = tx.TypeVar("DICTLIKE", bound=_MAPPINGLIKE, default=_MAPPINGLIKE)
TUPLELIKE = tx.TypeVar("TUPLELIKE", bound=tx.Tuple, default=tx.Tuple)
NUMBERLIKE = tx.TypeVar("NUMBERLIKE", bound=_NUMBERLIKE, default=_NUMBERLIKE)
ITERLIKE = tx.TypeVar("ITERLIKE", bound=abc.Iterable, default=abc.Iterable)
DTYPELIKE = tx.TypeVar("DTYPELIKE", bound=npt.DTypeLike, default=npt.DTypeLike)

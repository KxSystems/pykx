"""
Wrappers for q data structures, with conversion functions to Python/Numpy/Pandas/Arrow.

Under PyKX, q has its own memory space in which it stores q data structures in the same way it is
stored within a regular q process. PyKX provides Pythonic wrappers around these objects in q
memory.

In general, these wrappers consist of a pointer to a location in q memory, and a collection
of methods to operate on/with that data.

Memory in q is managed by reference counting, and so wrapper objects (i.e. [`pykx.K`][pykx.K]
objects) hold a reference to the underlying object in q memory, and release it when the Python
wrapper is deallocated.

A benefit of this approach to interacting with q data is that all conversions are deferred until
explicitly performed via one of the conversion methods:

- `.py` for Python
- `.np` for Numpy
- `.pd` for Pandas
- `.pa` for PyArrow

Example:

```python
>>> import pykx as kx
>>> t = kx.q('([] x: 1 2 3; y: `a`b`c)') # Create a table in q memory
>>> x = t['x'] # Index into it in q
>>> x
pykx.LongVector(pykx.q('1 2 3'))
>>> x.np() # Convert the vector to Numpy
array([1, 2, 3])
>>> t.pd() # Convert the table to Pandas
   x  y
0  1  a
1  2  b
2  3  c
>>> t.pa() # Convert the table to PyArrow
pyarrow.Table
x: int64
y: string
----
x: [[1,2,3]]
y: [["a","b","c"]]
>>> t.py() # Convert the table to Python
{'x': [1, 2, 3], 'y': ['a', 'b', 'c']}
```

Conversions from q to Python avoid copying data where possible, but that is not always possible.
Furthermore, conversions can result in loss of data about type information - not a loss of the
data itself, but of the type information that informs how it will be interpreted.

For example, if we had a q second atom `pykx.SecondAtom('04:08:16')`, and we converted it to Python
using its `.py` method, we would get `datetime.timedelta(seconds=14896)`. If we convert that back
to q using the `pykx.K` constructor, we get `pykx.TimespanAtom(pykx.q('0D04:08:16.000000000'))`.

We started with a `pykx.SecondAtom`, but after converting to Python and then back to q, we get a
`pykx.TimestampAtom`. This is because that is what `datetime.timedelta` converts to by default, but
the net effect is that we lost type information about the q data.

This round-trip-lossiness can be prevented in 2 ways:

1. If possible, avoid converting the `pykx.K` object to a Python/Numpy/Pandas/PyArrow type in the
   first place. No information can be lost during a conversion if no conversion occurs. It is
   frequently the case that no conversion actually needs to occur. As an example, instead of
   converting an entire `pykx.Table` into a Pandas dataframe, only to use a few columns from it,
   one can index into the `pykx.Table` directly to get the desired columns, thereby avoiding the
   conversion of the entire table.
2. Conversions from Python to q can be controlled by specifying the desired type. Using `pykx.K` as
   a constructor forces it to chose what q type the data should be converted to (using the same
   mechanism as [`pykx.toq`][pykx.toq]), but by using the class of the desired q type directly,
   e.g. [`pykx.SecondAtom`][pykx.SecondAtom], one can override the defaults.

So to avoid the loss of type information from the previous example, we could run
`pykx.SecondAtom(datetime.timedelta(seconds=14896))` instead of
`pykx.K(datetime.timedelta(seconds=14896))`.

## Wrapper type hierarchy

The classes in the diagram are all attributes of `pykx`. They should be accessed as `pykx.K`,
`pykx.LongVector`, `pykx.Table`, etc.

```mermaid
graph LR
  K --> Atom
  K --> Collection
  Atom --> GUIDAtom
  Atom --> NumericAtom
  NumericAtom --> IntegralNumericAtom
  IntegralNumericAtom --> BooleanAtom
  IntegralNumericAtom --> ByteAtom
  IntegralNumericAtom --> ShortAtom
  IntegralNumericAtom --> IntAtom
  IntegralNumericAtom --> LongAtom
  NumericAtom --> NonIntegralNumericAtom
  NonIntegralNumericAtom --> RealAtom
  NonIntegralNumericAtom --> FloatAtom
  Atom --> CharAtom
  Atom --> SymbolAtom
  Atom --> TemporalAtom
  TemporalAtom --> TemporalFixedAtom
  TemporalFixedAtom --> TimestampAtom
  TemporalFixedAtom --> MonthAtom
  TemporalFixedAtom --> DateAtom
  TemporalFixedAtom --> DatetimeAtom
  TemporalAtom --> TemporalSpanAtom
  TemporalSpanAtom --> TimespanAtom
  TemporalSpanAtom --> MinuteAtom
  TemporalSpanAtom --> SecondAtom
  TemporalSpanAtom --> TimeAtom
  Atom --> EnumAtom
  Atom --> Function
  Function --> Lambda
  Function --> UnaryPrimitive
  UnaryPrimitive --> Identity
  Identity --> ProjectionNull
  Function --> Operator
  Function --> Iterator
  Function --> Projection
  Function --> Composition
  Function --> AppliedIterator
  AppliedIterator --> Each
  AppliedIterator --> Over
  AppliedIterator --> Scan
  AppliedIterator --> EachPrior
  AppliedIterator --> EachRight
  AppliedIterator --> EachLeft
  Function --> Foreign
  Collection --> Vector
  Vector --> List
  Vector --> GUIDVector
  Vector --> NumericVector
  NumericVector --> IntegralNumericVector
  IntegralNumericVector --> BooleanVector
  IntegralNumericVector --> ByteVector
  IntegralNumericVector --> ShortVector
  IntegralNumericVector --> IntVector
  IntegralNumericVector --> LongVector
  NumericVector --> NonIntegralNumericVector
  NonIntegralNumericVector --> RealVector
  NonIntegralNumericVector --> FloatVector
  Vector --> CharVector
  Vector --> SymbolVector
  Vector --> TemporalVector
  TemporalVector --> TemporalFixedVector
  TemporalFixedVector --> TimestampVector
  TemporalFixedVector --> MonthVector
  TemporalFixedVector --> DateVector
  TemporalFixedVector --> DatetimeVector
  TemporalVector --> TemporalSpanVector
  TemporalSpanVector --> TimespanVector
  TemporalSpanVector --> MinuteVector
  TemporalSpanVector --> SecondVector
  TemporalSpanVector --> TimeVector
  Vector --> EnumVector
  Collection --> Mapping
  Mapping --> Dictionary
  Dictionary --> KeyedTable
  Mapping --> Table
  Table --> SplayedTable
  SplayedTable --> PartitionedTable
```
"""

from abc import ABCMeta
from collections import abc
import copy
from datetime import datetime, timedelta
import importlib
from inspect import signature
import inspect
import math
from numbers import Integral, Number, Real
import operator
from uuid import UUID
from typing import Any, Optional, Tuple, Union
from warnings import warn
from io import StringIO

import numpy as np
import pandas as pd
import pytz

from . import _wrappers, beta_features, help
from ._pyarrow import pyarrow as pa
from .config import _check_beta, k_gc, licensed, pandas_2, suppress_warnings
from .core import keval as _keval
from .constants import INF_INT16, INF_INT32, INF_INT64, INF_NEG_INT16, INF_NEG_INT32, INF_NEG_INT64
from .constants import NULL_INT16, NULL_INT32, NULL_INT64
from .exceptions import LicenseException, PyArrowUnavailable, PyKXException, QError
from .util import cached_property, class_or_instancemethod, classproperty, detect_bad_columns, df_from_arrays # noqa E501

import importlib.util
_torch_unavailable = importlib.util.find_spec('torch') is None
if not _torch_unavailable:
    beta_features.append('PyTorch Conversions')

q_initialized = False


def _init(_q):
    global q
    global q_initialized
    q = _q
    q_initialized = True


# nanoseconds between 1970-01-01 and 2000-01-01
TIMESTAMP_OFFSET = 946684800000000000
#      months between 1970-01-01 and 2000-01-01
MONTH_OFFSET = 360
#        days between 1970-01-01 and 2000-01-01
DATE_OFFSET = 10957


def _idx_to_k(key, n):
    if isinstance(key, K):
        return key
    if isinstance(key, Integral):
        # replace negative index with equivalent positive index
        key = _key_preprocess(key, n)
    elif isinstance(key, slice):
        key = range(n)[key]
    return K(key)


def _get_type_char(val):
    return ' bg xhijefcspmdznuvts'[abs(val.t)]


def _key_preprocess(key, n, slice=False, ignore_error=False):
    if key is not None:
        if key < 0:
            key = n + key
        if (key >= n or key < 0) and not slice and not ignore_error:
            raise IndexError('index out of range')
        elif slice:
            if key < 0:
                key = 0
            if key > n:
                key = n
    return(key)


def _rich_convert(x: 'K', stdlib: bool = True, raw=False):
    if stdlib:
        return x.py(stdlib=stdlib, raw=raw)
    if isinstance(x, Mapping):
        return x.pd(raw=raw)
    return x.np(raw=raw)


# HACK: This gets overwritten by the toq module to avoid a circular import error.
def toq(*args, **kwargs): # nocov
    raise NotImplementedError


class K:
    """Base type for all q objects.

    Parameters:
        x (Any): An object that will be converted into a `pykx.K` object via [`pykx.toq`][pykx.toq].
    """
    def __new__(cls, x: Any, *args, cast: bool = None, **kwargs):
        return toq(x, ktype=None if cls is K else cls, cast=cast) # TODO: 'strict' and 'cast' flags

    # Signature must match `__new__`
    def __init__(self, x: Any, *args, cast: bool = None, **kwargs):
        pass

    def __del__(self):
        if hasattr(self, '_numpy_allocated'):
            _wrappers.decref_numpy_allocated_data(self._numpy_allocated)
        if q_initialized and k_gc:
            should_dealloc = self._k.r == 0
            del self._k
            if should_dealloc:
                _keval(bytes('.Q.gc[]', 'utf-8'))

    @classmethod
    def _from_addr(cls, addr: int, incref: bool = True):
        return _wrappers.k_from_addr(cls, addr, incref)

    def __repr__(self):
        preamble = f'pykx.{type(self).__name__}'
        if not licensed:
            return f'{preamble}._from_addr({hex(self._addr)})'
        data_str = K.__str__(self)
        if '\n' in data_str:
            data_str = f'\n{data_str}\n'
        return f"{preamble}(pykx.q('{data_str}'))"

    def __str__(self):
        return _wrappers.k_str(self)

    def __reduce__(self):
        return (_wrappers.k_unpickle, (_wrappers.k_pickle(self),))

    def _compare(self, other, op_str, *, failure=False):
        try:
            r = q(op_str, self, other)
        except Exception as ex:
            ex_str = str(ex)
            if ex_str.startswith('length') or ex_str.startswith('type'):
                return q('{x}', failure)
            raise
        else:
            if hasattr(r, '__len__') and len(r) == 0:
                # Handle comparisons of empty objects
                if op_str == '=':
                    return q('~', self, other)
                elif op_str == '{not x=y}':
                    return q('{not x~y}', self, other)
            return r

    def __lt__(self, other):
        return self._compare(other, '<')

    def __le__(self, other):
        return self._compare(other, '<=')

    def __eq__(self, other):
        try:
            return self._compare(other, '=')
        except TypeError:
            return q('0b')

    def __ne__(self, other):
        try:
            return self._compare(other, '{not x=y}', failure=True)
        except TypeError:
            return q('1b')

    def __gt__(self, other):
        return self._compare(other, '>')

    def __ge__(self, other):
        return self._compare(other, '>=')

    def __add__(self, other):
        return q('+', self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return q('-', self, other)

    def __rsub__(self, other):
        return q('-', other, self)

    def __mul__(self, other):
        return q('*', self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return q('%', self, other)

    # __rtruediv__ only has priority over __truediv__ if the right-hand operand
    # is an instance of a subclass of the left-hand operand's class
    def __rtruediv__(self, other):
        return q('%', other, self)

    def __floordiv__(self, other):
        return q('div', self, other)

    def __rfloordiv__(self, other):
        return q('div', other, self)

    def __mod__(self, other):
        return q('mod', self, other)

    def __rmod__(self, other):
        return q('mod', other, self)

    def __divmod__(self, other):
        return tuple(q('{(x div y;x mod y)}', self, other))

    def __rdivmod__(self, other):
        return tuple(q('{(x div y;x mod y)}', other, self))

    # TODO: Do we want to keep .pykx.modpow? Q lacks an integer pow, and
    # while one would be easy to implement in pure q it wouldn't be efficient,
    # nor would it be able to handle a modulo easily. Our options are:
    # A) Keep modpow, despite the complexity, and potential for errors
    # B) Raise an error when __pow__ is used on integers
    # C) Use xexp, and always get floats back regardless of the input type, and
    #    be unable to take advantage of the modulo
    # D) Convert the object to a numpy array, and call pow on that; note that
    #    this approach breaks conformity with how all the other similar methods
    #    are implemented
    # E) Use xexp, but convert back to whatever type the original object was;
    #    this approach also cannot take advantage of modulo, and would require
    #    special handling for each type (i.e. more code than ideal, but less
    #    than option A)
    def __pow__(self, other, modulo=None):
        return q('.pykx.modpow', self, other, modulo)

    def __rpow__(self, other, modulo=None):
        return q('.pykx.modpow', other, self, modulo)

    def __neg__(self):
        return q('neg', self)

    def __pos__(self):
        return self

    def __abs__(self):
        return q('abs', self)

    def __bool__(self):
        if self.is_null:
            return False
        if self.is_inf:
            return True
        if isinstance(self, TemporalAtom):
            # Temporal atoms are false if their underlying value is 0, or if they are null.
            return bool(self.py(raw=True))
        if isinstance(self, Foreign):
            return True
        x = self.py()
        if type(self) is type(x):
            raise TypeError(f"Unable to determine truth value of '{self!r}'")
        return bool(x)

    any = __bool__
    all = __bool__

    def copy(self):
        return copy.copy(self)

    @property
    def is_atom(self):
        return True

    @property
    def t(self):
        return _wrappers.k_t(self)

    def cast(self, ktype):
        return ktype(self)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        # """Converts the `pykx.K` object into a Python representation.

        # For direct instances of `pykx.K`, this method returns the `pykx.K` object it is called on.
        # Note that a `pykx.K` object is already a Python object that supports most of the methods
        # one would expect, e.g. `__len__` for collections.

        # This `py` method in particular exists as a fallback, for when a concrete `pykx.K` subclass
        # does not provide a `py` method. Refer to the K type hierarchy diagram to see which classes
        # will be checked for a method for any concrete type.

        # Parameters:
        #     raw: Ignored.
        #     has_nulls: Ignored.
        #     stdlib: Ignored.

        # Returns:
        #     The `pykx.K` instance
        # """
        return self

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return self.py(raw=raw)

    def pd(
        self,
        *,
        raw: bool = False,
        has_nulls: Optional[bool] = None,
        as_arrow: Optional[bool] = False,
    ):
        return self.np(raw=raw)

    def pa(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        if pa is None:
            raise PyArrowUnavailable # nocov
        return self.pd(raw=raw)

    def __typed_array__(self):
        if isinstance(self, List):
            # this verifies shape of the numpy array and verifies that the types all match
            shape = q("""{
                checkDimsType: {$[0=t:type x;
                    count[x],'distinct raze .z.s each x;
                    10=t;
                    t;
                    enlist(count x; neg t)]
                };
                checkData: {[x; checkDimsType]
                    if[1<count dt: checkDimsType x;
                    '"invalid ",$[1<count distinct last each dt;"type";"shape"]];
                    if[10h~dt:first dt;
                    dt:count[x],-10h];
                    -1_dt
                };
                checkData[x; checkDimsType]
                }""", self).np()
            if shape[-1] == 1 and len(shape) > 1:
                shape = shape[0:-1]
            flat = q(f'{{{"raze " * (len(shape) + 1)} x}}', self) # flatten all nested arrays
            return flat.np().reshape(shape)
        return self.np()

    def _to_vector(self):
        return self


class Atom(K):
    """Base type for all q atoms, including singular basic values, and functions.

    See Also:
        [`pykx.Collection`][pykx.Collection]
    """
    @property
    def is_null(self) -> bool:
        return q('null', self).py()

    @property
    def is_inf(self) -> bool:
        if self.t in {-1, -2, -4, -10, -11}:
            return False
        try:
            type_char = _get_type_char(self)
        except IndexError:
            return False
        return q(f'{{any -0W 0W{type_char}~\\:x}}')(self).py()

    @property
    def is_pos_inf(self) -> bool:
        if self.t in {-1, -2, -4, -10, -11}:
            return False
        try:
            type_char = _get_type_char(self)
        except IndexError:
            return False
        return q(f'{{0W{type_char}~x}}')(self).py()

    @property
    def is_neg_inf(self) -> bool:
        if self.t in {-1, -2, -4, -10, -11}:
            return False
        try:
            type_char = _get_type_char(self)
        except IndexError:
            return False
        return q(f'{{-0W{type_char}~x}}')(self).py()

    def __hash__(self):
        return _wrappers.k_hash(self)

    def __invert__(self):
        return ~self.py()

    def __lshift__(self, other):
        return self.py() << other

    def __rlshift__(self, other):
        return other << self.py()

    def __rshift__(self, other):
        return self.py() >> other

    def __rrshift__(self, other):
        return other >> self.py()

    def __and__(self, other):
        return self.py() & other

    def __rand__(self, other):
        return other & self.py()

    def __or__(self, other):
        return self.py() | other

    def __ror__(self, other):
        return other | self.py()

    def __xor__(self, other):
        return self.py() ^ other

    def __rxor__(self, other):
        return other ^ self.py()

    def _to_vector(self):
        return _wrappers.to_vec(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        res = self._to_vector().__array_ufunc__(
            ufunc,
            method,
            *[x._to_vector() if isinstance(x, K) else x for x in inputs],
            **kwargs
        )

        if res.t >= 0:
            res = res._unlicensed_getitem(0)
        return res


class EnumAtom(Atom):
    """Wrapper for q enum atoms.

    Parameters:
        variable: The name of a list in q memory.
        index: An index used in [Enumeration](https://code.kx.com/q/ref/enumeration/).
        value: An item that is used in [Enumerate](https://code.kx.com/q/ref/enumerate/) and [Enum Extend](https://code.kx.com/q/ref/enum-extend/).
        extend:  A boolean set to True to use [Enum Extend](https://code.kx.com/q/ref/enum-extend/) and False to use [Enumerate](https://code.kx.com/q/ref/enumerate/).
    """ # noqa: E501
    t = -20

    def __new__(cls, variable, index=None, value=None, extend=False):
        if not isinstance(variable, (str, SymbolAtom)):
            raise TypeError("Variable name must be of type String or Symbol.")
        if not (index is None) ^ (value is None):
            raise AttributeError("Can only set one of 'value' and 'index' at one time.")
        if index is not None:
            return q('!', variable, index)
        if value is not None:
            if extend:
                return q('?', variable, value)
            else:
                return q('$', variable, value)

    def value(self):
        """Returns the value of the enumeration"""
        return q.value(self)

    def domain(self):
        """Returns the name of the domain of the enum"""
        return q.key(self)

    def index(self):
        """Returns the index of the enum in the q list"""
        return q('`long$', self)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return _wrappers.k_j(self)
        return q('value', self).py()


class TemporalAtom(Atom):
    """Base type for all q temporal atoms."""
    pass


class TemporalSpanAtom(TemporalAtom):
    """Base type for all q temporal atoms which represent a span of time."""
    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return self.np(raw=True)
        as_np_timedelta = self.np()
        if pd.isnull(as_np_timedelta):
            return pd.NaT
        return timedelta(
            microseconds=int(as_np_timedelta.astype('timedelta64[ns]').astype(np.int64))//1000
        )

    def np(self,
           *,
           raw: bool = False,
           has_nulls: Optional[bool] = None,
    ) -> Union[np.timedelta64, int]:
        if raw:
            if self.t == -16:
                return _wrappers.k_j(self)
            return _wrappers.k_i(self)
        if self.is_null:
            return np.timedelta64('NaT')
        if self.t == -16:
            return np.timedelta64(_wrappers.k_j(self), self._np_type)
        return np.timedelta64(_wrappers.k_i(self), self._np_type)

    def pd(self,
           *,
           raw: bool = False,
           has_nulls: Optional[bool] = None,
           as_arrow: Optional[bool] = False,
    ) -> Union[pd.Timedelta, int]:
        if raw:
            return self.np(raw=True)
        return pd.Timedelta(self.np())


class TemporalFixedAtom(TemporalAtom):
    """Base type for all q temporal atoms which represent a fixed date/time."""
    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return self.np(raw=True)
        if self.is_null:
            return pd.NaT
        return self.np().astype(datetime)

    def np(self,
           *,
           raw: bool = False,
           has_nulls: Optional[bool] = None,
    ) -> Union[np.datetime64, int]:
        if raw:
            if self.t == -12:
                return _wrappers.k_j(self)
            return _wrappers.k_i(self)
        if self.is_null:
            return np.datetime64('NaT')
        if self.t == -12:
            epoch_offset = 0 if self.is_pos_inf else self._epoch_offset
            return np.datetime64(_wrappers.k_j(self) + epoch_offset, self._np_type)
        return np.datetime64(_wrappers.k_i(self) + self._epoch_offset, self._np_type)

    def pd(
        self,
        *,
        raw: bool = False,
        has_nulls: Optional[bool] = None,
        as_arrow: Optional[bool] = False,
    ):
        if raw:
            return self.np(raw=True)
        return pd.Timestamp(self.np())


class TimeAtom(TemporalSpanAtom):
    """Wrapper for q time atoms."""
    t = -19
    _null = '0Nt'
    _inf = '0Wt'
    _np_type = 'ms'
    _np_dtype = 'timedelta64[ms]'

    def __new__(cls, x: Any, *, cast: bool = None, **kwargs):
        if licensed and isinstance(x, str) and x == 'now': # noqa: E721
            return q('.z.T')
        return toq(x, ktype=None if cls is K else cls, cast=cast) # TODO: 'strict' and 'cast' flags

    def _prototype(self=None):
        return TimeAtom(np.timedelta64(59789214, 'ms'))

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls):  # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32

    @property
    def is_pos_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_INT32

    @property
    def is_neg_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_NEG_INT32


class SecondAtom(TemporalSpanAtom):
    """Wrapper for q second atoms."""
    t = -18
    _null = '0Nv'
    _inf = '0Wv'
    _np_type = 's'
    _np_dtype = 'timedelta64[s]'

    def _prototype(self=None):
        return SecondAtom(np.timedelta64(13019, 's'))

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32

    @property
    def is_pos_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_INT32

    @property
    def is_neg_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_NEG_INT32


class MinuteAtom(TemporalSpanAtom):
    """Wrapper for q minute atoms."""
    t = -17
    _null = '0Nu'
    _inf ='0Wu'
    _np_type = 'm'
    _np_dtype = 'timedelta64[m]'

    def _prototype(self=None):
        return MinuteAtom(np.timedelta64(216, 'm'))

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32

    @property
    def is_pos_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_INT32

    @property
    def is_neg_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_NEG_INT32


class TimespanAtom(TemporalSpanAtom):
    """Wrapper for q timespan atoms."""
    t = -16
    _null = '0Nn'
    _inf = '0Wn'
    _np_type = 'ns'
    _np_dtype = 'timedelta64[ns]'

    def __new__(cls, x: Any, *args, cast: bool = None, **kwargs):
        if licensed and isinstance(x, str) and x == 'now': # noqa: E721
            return q('.z.N')
        if isinstance(x, int):
            if not licensed:
                raise LicenseException('Cannot create object from numerical values, convert from "datetime.timedelta"') # noqa: E501
            if not all(isinstance(i, int) for i in args):
                raise TypeError("All values must be of type int when creating a TimespanAtom using numeric values") # noqa: E501
            if len(args) != 4:
                if len(args) > 4:
                    raise TypeError("Too many values. Numeric TimespanAtom creation requires 4 values only") # noqa: E501
                else:
                    raise TypeError("Too few values. Numeric TimespanAtom creation requires 4 values only") # noqa: E501
            elif all(isinstance(i, int) for i in args):
                return q('{[D;h;m;s;n]sum (1D;0D01;0D00:01;0D00:00:01;0D00:00:00.000000001) * (D;h;m;s;n)}', x, args[0], args[1], args[2], args[3]) # noqa: E501
        return toq(x, ktype=None if cls is K else cls, cast=cast) # TODO: 'strict' and 'cast' flags

    def _prototype(self=None):
        return TimespanAtom(np.timedelta64(3796312051664551936, 'ns'))

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return _wrappers.k_j(self) == NULL_INT64

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_j(self)) == INF_INT64

    @property
    def is_pos_inf(self) -> bool:
        return _wrappers.k_j(self) == INF_INT64

    @property
    def is_neg_inf(self) -> bool:
        return _wrappers.k_j(self) == INF_NEG_INT64


class DatetimeAtom(TemporalFixedAtom):
    """Wrapper for q datetime atoms.

    Warning: The q datetime type is deprecated.
        PyKX does not provide a rich interface for the q datetime type, as it is deprecated. Avoid
        using it whenever possible.
    """
    t = -15
    _null = '0Nz'
    _inf = '0Wz'
    _np_dtype = None

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @property
    def is_null(self) -> bool:
        return math.isnan(self.py(raw=True))

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @property
    def is_inf(self) -> bool:
        aspy = self.py(raw=True)
        return (math.inf == aspy) or (-math.inf == aspy)

    @property
    def is_pos_inf(self) -> bool:
        return math.inf == self.py(raw=True)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_neg_inf(self) -> bool:
        return -math.inf == self.py(raw=True)

    def __init__(self, *args, **kwargs):
        warn('The q datetime type is deprecated', DeprecationWarning)
        super().__init__(*args, **kwargs)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return _wrappers.k_f(self)
        raise TypeError('The q datetime type is deprecated, and can only be accessed with '
                        'the keyword argument `raw=True` in Python or `.pykx.toRaw` in q')

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return self.py(raw=raw)


class DateAtom(TemporalFixedAtom):
    """Wrapper for q date atoms."""
    t = -14
    _np_type = 'D'
    _null = '0Nd'
    _inf = '0Wd'
    _epoch_offset = DATE_OFFSET
    _np_dtype = 'datetime64[D]'

    def __new__(cls, x: Any, *args, cast: bool = None, **kwargs):
        if licensed and isinstance(x, str) and (x == 'today'):
            return q('.z.D')
        if isinstance(x, int):
            if not licensed:
                raise LicenseException('Cannot create object from numerical values, convert from "datetime.date"') # noqa: E501
            if not all(isinstance(i, int) for i in args):
                raise TypeError("All values must be of type int when creating a DateAtom using numeric values") # noqa: E501
            if len(args) != 2:
                if len(args) > 2:
                    raise TypeError("Too many values. Numeric DateAtom creation requires 3 values only") # noqa: E501
                else:
                    raise TypeError("Too few values. Numeric DateAtom creation requires 3 values only") # noqa: E501
            elif all(isinstance(i, int) for i in args) and (len(args) == 2):
                return q('{[y;m;d]"D"$"." sv string (y;m;d)}', x, args[0], args[1])
        return toq(x, ktype=None if cls is K else cls, cast=cast) # TODO: 'strict' and 'cast' flags

    def _prototype(self=None):
        return DateAtom(np.datetime64('1972-05-31', 'D'))

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32

    @property
    def is_pos_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_INT32

    @property
    def is_neg_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_NEG_INT32


class MonthAtom(TemporalFixedAtom):
    """Wrapper for q month atoms."""
    t = -13
    _null = '0Nm'
    _inf = '0Wm'
    _np_type = 'M'
    _epoch_offset = MONTH_OFFSET
    _np_dtype = 'datetime64[M]'

    def _prototype(self=None):
        return MonthAtom(np.datetime64('1972-05', 'M'))

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32

    @property
    def is_pos_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_INT32

    @property
    def is_neg_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_NEG_INT32


class TimestampAtom(TemporalFixedAtom):
    """Wrapper for q timestamp atoms."""
    t = -12
    _null = '0Np'
    _inf = '0Wp'
    _np_type = 'ns'
    _epoch_offset = TIMESTAMP_OFFSET
    _np_dtype = 'datetime64[ns]'

    def __new__(cls, x: Any, *args, cast: bool = None, **kwargs):
        if licensed and isinstance(x, str) and x == 'now': # noqa: E721
            return q('.z.P')
        if isinstance(x, int):
            if not licensed:
                raise LicenseException('Cannot create object from numerical values, convert from "datetime.datetime"') # noqa: E501
            if not all(isinstance(i, int) for i in args):
                raise TypeError("All values must be of type int when creating a TimestampAtom using numeric values") # noqa: E501
            if len(args) != 6:
                if len(args) > 6:
                    raise TypeError("Too many values. Numeric TimestampAtom creation requires 7 values only") # noqa: E501
                else:
                    raise TypeError("Too few values. Numeric TimestampAtom creation requires 7 values only") # noqa: E501
            elif all(isinstance(i, int) for i in args):
                return q('''{[Y;M;D;h;m;s;n]
                ("D"$"." sv string (Y;M;D))+sum(0D01;0D00:01;0D00:00:01;0D00:00:00.000000001)*(h;m;s;n)}''', # noqa: E501
                         x, args[0], args[1], args[2], args[3], args[4], args[5])
        return toq(x, ktype=None if cls is K else cls, cast=cast) # TODO: 'strict' and 'cast' flags

    def _prototype(self=None):
        return TimestampAtom(datetime(2150, 10, 22, 20, 31, 15, 70713))

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return _wrappers.k_j(self) == NULL_INT64

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_j(self)) == INF_INT64

    @property
    def is_pos_inf(self) -> bool:
        return _wrappers.k_j(self) == INF_INT64

    @property
    def is_neg_inf(self) -> bool:
        return _wrappers.k_j(self) == INF_NEG_INT64

    @property
    def date(self):
        return q('{`date$x}', self)

    @property
    def time(self):
        return q('{`time$x}', self)

    @property
    def year(self):
        return IntAtom(q('{`year$x}', self))

    @property
    def month(self):
        return IntAtom(q('{`mm$x}', self))

    @property
    def day(self):
        return IntAtom(q('{`dd$x}', self))

    @property
    def hour(self):
        return IntAtom(q('{`hh$x}', self))

    @property
    def minute(self):
        return IntAtom(q('{`uu$x}', self))

    @property
    def second(self):
        return IntAtom(q('{`ss$x}', self))

    def py(self,
           *,
           raw: bool = False,
           has_nulls: Optional[bool] = None,
           stdlib: bool = True,
           tzinfo: Optional[pytz.BaseTzInfo] = None,
           tzshift: bool = True
    ):
        # XXX: Since Python datetime objects don't support nanosecond
        #      precision (https://bugs.python.org/issue15443), we have to
        #      convert to datetime64[us] before converting to datetime objects
        if raw:
            return _wrappers.k_j(self)
        if self.is_null:
            return pd.NaT
        if tzinfo is not None:
            if tzshift:
                return self\
                    .np()\
                    .astype('datetime64[us]')\
                    .astype(datetime)\
                    .replace(tzinfo=pytz.utc)\
                    .astimezone(tzinfo)
            else:
                return self\
                    .np()\
                    .astype('datetime64[us]')\
                    .astype(datetime)\
                    .replace(tzinfo=tzinfo)
        return self.np().astype('datetime64[us]').astype(datetime)


class SymbolAtom(Atom):
    """Wrapper for q symbol atoms.

    Danger: Unique symbols are never deallocated!
        Reserve symbol data for values that are recurring. Avoid using symbols for data being
        generated over time (e.g. random symbols) as memory usage will continually increase.

    See Also:
        [`pykx.CharVector`][pykx.CharVector]
    """
    t = -11
    _null = '`'
    _inf = None
    _np_dtype = None

    def _prototype(self=None):# noqa
        return SymbolAtom('')

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return str(self) == ''

    @property
    def is_inf(self) -> bool:
        return False

    @property
    def is_pos_inf(self) -> bool:
        return False

    @property
    def is_neg_inf(self) -> bool:
        return False

    def __bytes__(self):
        return _wrappers.k_s(self)

    def __str__(self):
        return _wrappers.k_s(self).decode()

    def __add__(self, other):
        return type(self)(str(self) + other)

    def __radd__(self, other):
        return type(self)(other + str(self))

    def __int__(self):
        return int(str(self))

    def __float__(self):
        return float(str(self))

    def __complex__(self):
        return complex(str(self))

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return bytes(self)
        return str(self)


class CharAtom(Atom):
    """Wrapper for q char (i.e. 8 bit ASCII value) atoms."""
    t = -10
    _null = '" "'
    _inf = None
    _np_dtype = None
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)

    def _prototype(self=None):
        return CharAtom(b' ')

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return 32 == _wrappers.k_g(self)

    @property
    def is_inf(self) -> bool:
        return False

    @property
    def is_pos_inf(self) -> bool:
        return False

    @property
    def is_neg_inf(self) -> bool:
        return False

    def __bytes__(self):
        return self.py()

    def __len__(self):
        return 1

    def __getitem__(self, key):
        if not licensed:
            raise LicenseException('index into K object')
        if key != 0:
            raise IndexError('index out of range')
        return self

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return _wrappers.k_g(self)
        return bytes(chr(_wrappers.k_g(self)), 'utf-8')


class NumericAtom(Atom):
    """Base type for all q numeric atoms."""

    def __new__(cls, x: Any, *args, cast: bool = None, **kwargs):
        try:
            if math.isinf(x):
                return cls.inf if x>0 else -cls.inf
        except BaseException:
            pass
        return toq(x, ktype=None if cls is K else cls, cast=cast)

    def __int__(self):
        return int(self.py())

    def __float__(self):
        if self.is_null:
            return float('NaN')
        if self.is_inf:
            return float('inf' if self > 0 else '-inf')
        return float(self.py())

    def __complex__(self):
        return complex(float(self))


class NonIntegralNumericAtom(NumericAtom, Real):
    """Base type for all q non-integral numeric atoms."""
    @property
    def is_null(self) -> bool:
        return math.isnan(self.py())

    @property
    def is_inf(self) -> bool:
        return math.isinf(self.py())

    @property
    def is_pos_inf(self) -> bool:
        return np.isposinf(self.py())

    @property
    def is_neg_inf(self) -> bool:
        return np.isneginf(self.py())

    def __round__(self, ndigits=None):
        return round(self.py(), ndigits)

    def __trunc__(self):
        return math.trunc(self.py())

    def __floor__(self):
        return math.floor(self.py())

    def __ceil__(self):
        return math.ceil(self.py())

    def __pow__(self, other, mod=None):
        if mod is None:
            return q('xexp', self, other)
        raise TypeError(f"pow() 3rd argument not allowed when using '{type(self)}' as the base")

    def __rpow__(self, other, mod=None):
        if mod is None:
            return q('xexp', other, self)
        raise TypeError(f"pow() 3rd argument not allowed when using '{type(self)}' as the exponent")


class FloatAtom(NonIntegralNumericAtom):
    """Wrapper for q float (i.e. 64 bit float) atoms."""
    t = -9
    _null = '0n'
    _inf = '0w'
    _np_dtype = np.float64

    def _prototype(self=None):
        return FloatAtom(0.0)

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return _wrappers.k_f(self)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return np.float64(_wrappers.k_f(self))


class RealAtom(NonIntegralNumericAtom):
    """Wrapper for q real (i.e. 32 bit float) atoms."""
    t = -8
    _null = '0Ne'
    _inf = '0We'
    _np_dtype = np.float32

    def _prototype(self=None):
        return RealAtom(0.0)

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return _wrappers.k_e(self)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return np.float32(_wrappers.k_e(self))


class IntegralNumericAtom(NumericAtom, Integral):
    """Base type for all q integral numeric atoms."""
    def __int__(self):
        if self.is_null:
            raise PyKXException('Cannot convert null integral atom to Python int')
        if self.is_inf:
            raise PyKXException('Cannot convert infinite integral atom to Python int')
        return super().__int__()

    def __index__(self):
        return operator.index(self.py())

    def __round__(self, ndigits=None):
        return round(int(self.py()), ndigits)

    def __trunc__(self):
        return self.py()

    def __floor__(self):
        return self.py()

    def __ceil__(self):
        return self.py()

    def _py_null_or_inf(self, default, raw: bool):
        if not raw:
            if self.is_null:
                return pd.NA
            elif self.is_pos_inf:
                return math.inf
            elif self.is_neg_inf:
                return -math.inf
        return default

    def pd(self, *, raw: bool = False, has_nulls: Optional[bool] = None,
           as_arrow: Optional[bool] = False):
        if not raw and self.is_null:
            return pd.NA
        else:
            return self.np(raw=raw)


class LongAtom(IntegralNumericAtom):
    """Wrapper for q long (i.e. 64 bit signed integer) atoms."""
    t = -7
    _null = '0Nj'
    _inf = '0Wj'
    _np_dtype = np.int64

    def _prototype(self=None):
        return LongAtom(0)

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return _wrappers.k_j(self) == NULL_INT64

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_j(self)) == INF_INT64

    @property
    def is_pos_inf(self) -> bool:
        return _wrappers.k_j(self) == INF_INT64

    @property
    def is_neg_inf(self) -> bool:
        return _wrappers.k_j(self) == INF_NEG_INT64

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return self._py_null_or_inf(_wrappers.k_j(self), raw)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return np.int64(_wrappers.k_j(self))


class IntAtom(IntegralNumericAtom):
    """Wrapper for q int (i.e. 32 bit signed integer) atoms."""
    t = -6
    _null = '0Ni'
    _inf = '0Wi'
    _np_dtype = np.int32

    def _prototype(self=None):
        return IntAtom(0)

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32

    @property
    def is_pos_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_INT32

    @property
    def is_neg_inf(self) -> bool:
        return _wrappers.k_i(self) == INF_NEG_INT32

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return self._py_null_or_inf(_wrappers.k_i(self), raw)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return np.int32(_wrappers.k_i(self))


class ShortAtom(IntegralNumericAtom):
    """Wrapper for q short (i.e. 16 bit signed integer) atoms."""
    t = -5
    _null = '0Nh'
    _inf = '0Wh'
    _np_dtype = np.int16

    def _prototype(self=None):
        return ShortAtom(0)

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return _wrappers.k_h(self) == NULL_INT16

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_h(self)) == INF_INT16

    @property
    def is_pos_inf(self) -> bool:
        return _wrappers.k_h(self) == INF_INT16

    @property
    def is_neg_inf(self) -> bool:
        return _wrappers.k_h(self) == INF_NEG_INT16

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return self._py_null_or_inf(_wrappers.k_h(self), raw)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return np.int16(_wrappers.k_h(self))


class ByteAtom(IntegralNumericAtom):
    """Wrapper for q byte (i.e. 8 bit unsigned integer) atoms."""
    t = -4
    _null = None
    _inf = None
    _np_dtype = np.uint8

    def _prototype(self=None):
        return ByteAtom(0)

    @classproperty
    def null(cls): # noqa: B902
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() retrieval of null values not supported for this type") # noqa: E501

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return False

    @property
    def is_inf(self) -> bool:
        return False

    @property
    def is_pos_inf(self) -> bool:
        return False

    @property
    def is_neg_inf(self) -> bool:
        return False

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return _wrappers.k_g(self)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return np.uint8(_wrappers.k_g(self))


class GUIDAtom(Atom):
    """Wrapper for q GUID atoms."""
    t = -2
    _null = '0Ng'
    _inf = None
    _np_dtype = None

    def _prototype(self=None):
        return GUIDAtom(UUID(int=0))

    @classproperty
    def null(cls): # noqa: B902
        return toq.from_none(None, cls)

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return self.py(raw=True) == 0j

    @property
    def is_inf(self) -> bool:
        return False

    @property
    def is_pos_inf(self) -> bool:
        return False

    @property
    def is_neg_inf(self) -> bool:
        return False

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return _wrappers.guid_atom_py(self, raw, has_nulls, stdlib)


class BooleanAtom(IntegralNumericAtom):
    """Wrapper for q boolean atoms."""
    t = -1
    _null = None
    _inf = None
    _np_dtype = None

    def _prototype(self=None):
        return BooleanAtom(True)

    @classproperty
    def null(cls): # noqa: B902
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() retrieval of null values not supported for this type") # noqa: E501

    @classproperty
    def inf(cls): # noqa: B902
        return toq.create_inf(cls)

    @classproperty
    def inf_neg(cls):  # noqa: B902
        return toq.create_neg_inf(cls)

    @property
    def is_null(self) -> bool:
        return False

    @property
    def is_inf(self) -> bool:
        return False

    @property
    def is_pos_inf(self) -> bool:
        return False

    @property
    def is_neg_inf(self) -> bool:
        return False

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return _wrappers.k_g(self)
        return bool(_wrappers.k_g(self))


class Collection(K):
    """Base type for all q collections (i.e. non-atoms), including vectors, and mappings.

    See Also:
        [`pykx.Collection`][pykx.Collection]
    """
    @property
    def is_atom(self):
        return False

    def any(self) -> bool:
        return any(x.any() for x in self)

    def all(self) -> bool:
        return all(x.all() for x in self)

    def __bool__(self):
        raise TypeError("The truth value of a 'pykx.Collection' is ambiguous - "
                        "use the '.any()' or '.all()' method, or check the length")

    def __contains__(self, other):
        try:
            first = next(iter(self))
        except StopIteration:
            first = None
        # This special case prevents collections from being equal to atoms in any case. Without
        # this something like `(1,) in q('til 3')` would be `True` when it should be `False`.
        # Other being a str is a special special case, as it is converted to a symbol atom.
        if not isinstance(other, str) and first is not None and \
           not isinstance(first, abc.Collection) and isinstance(other, abc.Collection):
            return False
        return any((x == other) or (x is other) for x in self)

    def __getitem__(self, key):
        if not licensed:
            raise LicenseException('index into K object')
        return q('@', self, _idx_to_k(key, _wrappers.k_n(self)))


if pandas_2 and pa is not None:
    _as_arrow_map = {
        'List': 'object',
        'BooleanVector': 'bool[pyarrow]',
        'GUIDVector': 'object',
        'ByteVector': 'uint8[pyarrow]',
        'ShortVector': 'int16[pyarrow]',
        'IntVector': 'int32[pyarrow]',
        'LongVector': 'int64[pyarrow]',
        'RealVector': 'float[pyarrow]',
        'FloatVector': 'double[pyarrow]',
        'CharVector': pd.ArrowDtype(pa.binary(1)),
        'SymbolVector': 'string[pyarrow]',
        'TimestampVector': 'timestamp[ns][pyarrow]',
        'MonthVector': 'timestamp[s][pyarrow]',
        'DateVector': 'timestamp[s][pyarrow]',
        'TimespanVector': 'duration[ns][pyarrow]',
        'MinuteVector': 'duration[s][pyarrow]',
        'SecondVector': 'duration[s][pyarrow]',
        'TimeVector': 'duration[ms][pyarrow]'
    }

    _as_arrow_raw_map = {
        'List': 'object',
        'BooleanVector': 'bool[pyarrow]',
        'GUIDVector': 'object',
        'ByteVector': 'uint8[pyarrow]',
        'ShortVector': 'int16[pyarrow]',
        'IntVector': 'int32[pyarrow]',
        'LongVector': 'int64[pyarrow]',
        'RealVector': 'float[pyarrow]',
        'FloatVector': 'double[pyarrow]',
        'CharVector': pd.ArrowDtype(pa.binary(1)),
        'SymbolVector': pd.ArrowDtype(pa.binary()),
        'TimestampVector': 'int64[pyarrow]',
        'DatetimeVector': 'double[pyarrow]',
        'MonthVector': 'int32[pyarrow]',
        'DateVector': 'int32[pyarrow]',
        'TimespanVector': 'int64[pyarrow]',
        'MinuteVector': 'int32[pyarrow]',
        'SecondVector': 'int32[pyarrow]',
        'TimeVector': 'int32[pyarrow]',
    }


class Vector(Collection, abc.Sequence):
    """Base type for all q vectors, which are ordered collections of a particular type."""
    @property
    def has_nulls(self) -> bool:
        return q('{any null x}', self).py()

    @property
    def has_infs(self) -> bool:
        if self.t in {1, 2, 4, 10, 11}:
            return False
        try:
            type_char = _get_type_char(self)
        except IndexError:
            return False
        return q(f'{{any -0W 0W{type_char}=\\:x}}')(self).py()

    def __len__(self):
        return _wrappers.k_n(self)

    def __iter__(self):
        for i in range(_wrappers.k_n(self)):
            yield self._unlicensed_getitem(i)

    def __reversed__(self):
        for i in reversed(range(_wrappers.k_n(self))):
            yield self._unlicensed_getitem(i)

    def __setitem__(self, key, val):
        try:
            update_assign = q('{@[x;y;:;z]}', self, _idx_to_k(key, _wrappers.k_n(self)), val)
        except QError as err:
            if 'type' == str(err):
                raise QError(f'Failed to assign value of type: {type(val)} '
                             f'to list of type: {type(self)}')
            else:
                raise QError(str(err))
        self.__dict__.update(update_assign.__dict__)

    def __array__(self):
        # The `__array__` method must return a `np.ndarray`, not a `np.ma.masked_array`. As a
        # result, the null check we currently perform (by default) is a waste of time and memory,
        # since what will be returned in the end will ultimately expose the underlying values of
        # the nulls anyway. We can’t stop `__array__` from returning the raw null values, but we
        # can save that time and memory by using `has_nulls=False`.
        return self.np(has_nulls=False)

    def __arrow_array__(self, type=None):
        if pa is None:
            raise PyArrowUnavailable # nocov
        return self.pa()

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return self.np(raw=raw, has_nulls=has_nulls).tolist()

    def pd(
        self,
        *,
        raw: bool = False,
        has_nulls: Optional[bool] = None,
        as_arrow: Optional[bool] = False,
    ):
        res = pd.Series(self.np(raw=raw, has_nulls=has_nulls), copy=False)
        if not raw:
            null_inds = []
            for i in range(len(self)):
                if isinstance(self._unlicensed_getitem(i), IntegralNumericAtom)\
                        and self._unlicensed_getitem(i).is_null:
                    null_inds.append(i)
            if 0 != len(null_inds):
                res[null_inds] = pd.NA
        if as_arrow:
            if not pandas_2:
                raise RuntimeError('Pandas Version must be at least 2.0 to use as_arrow=True')
            if pa is None:
                raise PyArrowUnavailable # nocov
            if raw:
                if type(self).__name__ != 'GUIDVector':
                    res = res.astype(_as_arrow_raw_map[type(self).__name__])
            else:
                res = res.astype(_as_arrow_map[type(self).__name__])
        return res

    def pa(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        if pa is None:
            raise PyArrowUnavailable # nocov
        try:
            np_array = self.np(raw=raw, has_nulls=has_nulls)
            if not raw and isinstance(self, List):
                null_inds = []
                for i in range(len(self)):
                    if isinstance(self._unlicensed_getitem(i), IntegralNumericAtom)\
                            and self._unlicensed_getitem(i).is_null:
                        null_inds.append(i)
                if 0 != len(null_inds):
                    np_array[null_inds] = None
            return pa.array(np_array)
        except (pa.lib.ArrowNotImplementedError, pa.lib.ArrowInvalid) as err:
            if isinstance(self, List):
                raise QError('Unable to convert pykx.List with non conforming types '
                             f'to PyArrow,\n        failed with error: {err}')
            raise err

    def apply(self, func, *args, **kwargs):
        if not callable(func):
            raise RuntimeError("Provided value 'func' is not callable")

        return q(
            '{[f; data; args; kwargs] '
            '  func: $[.pykx.util.isw f;'
            '    f[; pyarglist args; pykwargs kwargs];'
            '    ['
            '      if[0<count kwargs;'
            '        \'"ERROR: Passing key word arguments to q is not supported"'
            '      ];'
            '      {[data; f; args]'
            '        r: f[data];'
            '        $[104h~type r; r . args; r]'
            '      }[; f; args]'
            '    ]'
            '  ];'
            '  func data'
            '}',
            func,
            self,
            args,
            kwargs
        )

    def sum(self):
        return q.sum(self)

    def prod(self):
        return q.prod(self)

    def min(self):
        return q.min(self)

    def max(self):
        return q.max(self)

    def mean(self):
        return q.avg(self)

    def median(self):
        return q.med(self)

    def mode(self):
        return q('{where max[c]=c:count each d:group x}', self)

    def append(self, data):
        """Append object to the end of a vector.

        Parameters:
            self: PyKX Vector/List object
            data: Data to be used when appending to a list/vector, when
                appending to a typed list this must be an object with a
                type which converts to an equivalent vector type.
                When appending to a List any type object can be appended.

        Raises:
            PyKXException: When dealing with typed vectors appending to this vector
                with data of a different type is unsupported and will raise an
                error

        Examples:

        Append to a vector object with an atom

        ```python
        >>> import pykx as kx
        >>> qvec = kx.q.til(3)
        >>> qvec
        pykx.LongVector(pykx.q('0 1 2'))
        >>> qvec.append(3)
        >>> qvec
        pykx.LongVector(pykx.q('0 1 2 3'))
        ```

        Attempt to append a vector object with an incorrect type:

        ```python
        >>> import pykx as kx
        >>> qvec = kx.q.til(3)
        >>> qvec.append([1, 2, 3])
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/wrappers.py", line 1262, ...
            raise QError(f'Appending data of type: {type(data)} '
        pykx.exceptions.QError: Appending data of type: <class 'pykx.wrappers.LongVector'> ...
        ```

        Append to a list object with an atom and list:

        ```python
        >>> import pykx as kx
        >>> qlist = kx.toq([1, 2.0])
        >>> qlist
        pykx.List(pykx.q('
        1
        2f
        '))
        >>> qlist.append('a')
        >>> qlist
        pykx.List(pykx.q('
        1
        2f
        `a
        '))
        >>> qlist.append([1, 2])
        >>> qlist
        pykx.List(pykx.q('
        1
        2f
        `a
        1 2
        '))
        ```
        """
        if (not isinstance(self, List)) and (not q('{(0>type[y])& type[x]=abs type y}', self, data)): # noqa: E501
            raise QError(f'Appending data of type: {type(K(data))} '
                         f'to vector of type: {type(self)} not supported')
        append_vec = q('{[orig;app]orig,$[0<=type app;enlist;]app}', self, data)
        self.__dict__.update(append_vec.__dict__)

    def extend(self, data):
        """Extend a vector by appending supplied values to the vector.

        Parameters:
            self: PyKX Vector/List object
            data: Data to be used when extending the a list/vector, this can
                be data of any type which can be converted to a PyKX object.

        Raises:
            PyKXException: When dealing with typed vectors extending this vector
                with data of a different type is unsupported and will raise an
                error

        Examples:

        Extend a vector object with an atom and list

        ```python
        >>> import pykx as kx
        >>> qvec = kx.q.til(3)
        >>> qvec
        pykx.LongVector(pykx.q('0 1 2'))
        >>> qvec.extend(3)
        >>> qvec
        pykx.LongVector(pykx.q('0 1 2 3'))
        >>> qvec.extend([4, 5, 6])
        >>> qvec
        pykx.LongVector(pykx.q('0 1 2 3 4 5 6'))
        ```

        Attempt to extend a vector object with an incorrect type:

        ```python
        >>> import pykx as kx
        >>> qvec = kx.q.til(3)
        >>> qvec.extend('a')
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/wrappers.py", line 1271, ...
            raise QError(f'Extending data of type: {type(data)} '
          pykx.exceptions.QError: Extending data of type: <class 'pykx.wrappers.SymbolAtom'> ...
        ```

        Extend a list object with an atom and list:

        ```python
        >>> import pykx as kx
        >>> qlist = kx.toq([1, 2.0])
        >>> qlist
        pykx.List(pykx.q('
        1
        2f
        '))
        >>> qlist.extend('a')
        >>> qlist
        pykx.List(pykx.q('
        1
        2f
        `a
        '))
        >>> qlist.extend([1, 2])
        >>> qlist
        pykx.List(pykx.q('
        1
        2f
        `a
        1
        2
        '))
        ```
        """
        if not isinstance(self, List):
            if q('{type[x]<>abs type y}', self, data):
                raise QError(f'Extending data of type: {type(K(data))} '
                             f'to vector of type: {type(self)} not supported')
        extend_vec = q('''
            {[orig;ext]
              ret:t!t:orig,$[type[ext]in 99 98h;enlist;]ext;
              value $[9h~type first t;count[t]#0x0;]ret}
            ''', self, data)
        self.__dict__.update(extend_vec.__dict__)

    def index(self, x, start=None, end=None):
        start = _key_preprocess(start, len(self), ignore_error=True)
        end = _key_preprocess(end, len(self), ignore_error=True)
        if start is None and end is None:
            i = q('{[v;x] i:v?x;$[i<count[v];i;(::)]}', self, x)
        elif start is not None and end is None:
            i = q('{[v;x;s] l:s _ v;i:l?x;$[i<count[l];s+i;(::)]}', self, x, start)
        elif start is None and end is not None:
            i = q('{[v;x;e] l:sublist[e;v];i:l?x;$[i<count[l];i;(::)]}', self, x, end)
        else:
            if end<=start:
                raise ValueError(f'{x!r} is not in {self!r}')
            i = q('{[v;x;s;e] l:v s+til e-s;i:l?x;$[i<count[l];s+i;(::)]}',
                  self, x, start, end)
        if i != None: # noqa: E711
            return i
        else:
            raise ValueError(f'{x!r} is not in {self!r}')

    def count(self, x):
        return sum((x == v) or (x is v) for v in self)

    def __ufunc_args__(self, *inputs, **kwargs):
        scalars = []
        for i in inputs:
            if isinstance(i, Number) or isinstance(i, list) or isinstance(i, np.ndarray):
                scalars.append(i)
            elif isinstance(i, K):
                scalars.append(i.__typed_array__())
            else:
                return NotImplemented
        vectors = {}
        for k, v in kwargs.items():
            if isinstance(v, K):
                vectors[k] = v.np()
            else:
                vectors[k] = v
        out = None
        if 'out' in vectors.keys():
            out = vectors['out']
            del vectors['out']
        dtype = None
        if 'dtype' in vectors.keys():
            dtype = vectors['dtype']
            del vectors['dtype']
        return (scalars, vectors, out, dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        scalars, vectors, out, dtype = self.__ufunc_args__(*inputs, **kwargs)
        res = None
        if method == '__call__':
            res = ufunc(*scalars, **vectors)
        elif method == 'accumulate':
            res = ufunc.accumulate(*scalars, **vectors)
        elif method == 'at':
            a = scalars[0]
            indices = scalars[1]
            if len(scalars) > 2:
                b = scalars[2]
                ufunc.at(a, indices, b, **vectors)
            else:
                # Numpy documents b's default as None, however passing none in segfaults,
                # so we have this weird if statement instead.
                ufunc.at(a, indices, **vectors)
            # The `at` ufunc method modifies arrays in place so there is no return value
            return
        elif method == 'outer':
            a = scalars[0]
            b = scalars[1]
            res = ufunc.outer(a, b, **vectors)
        elif method == 'reduce':
            res = ufunc.reduce(*scalars, **vectors)
        elif method == 'reduceat':
            a = scalars[0]
            indices = scalars[1]
            res = ufunc.reduceat(a, indices, **vectors)
        else:
            return NotImplemented # nocov
        return self.__ufunc_output__(res, out, dtype)

    def __ufunc_out_kwarg__(self, res, out, dtype):
        if len(out) == 1:
            if isinstance(out[0], List):
                raise ValueError('K List type objects cannot be used with the out keyword '
                                 'argument.')
            if isinstance(out[0], K):
                out[0].np()[:] = res.astype(out[0]._np_dtype)
            else:
                out[0][:] = res.astype(out[0].dtype)
        else:
            # TODO: We should add a test because this should be coverable, however I can't seem to
            # find any documentation about how to call a numpy ufunc with 2 `out` key-word
            # arguments, it requires a ufunc that has multiple outputs. But I can't seem to find
            # any.
            for o in out:
                self.__ufunc_out_kwarg__(res, (o), dtype)

    def __ufunc_output__(self, res, out, dtype):
        if out is not None:
            self.__ufunc_out_kwarg__(res, out, dtype)
        else:
            if dtype is None:
                if len(res.shape) > 1:
                    shape = str(res.shape).replace(',', '')
                    a = toq(res.flatten())
                    return q(f'{{{shape}#x}}', a)
                else:
                    return toq(res)
            elif issubclass(dtype, K):
                res = res.astype(dtype._np_dtype)
                if len(res.shape) > 1:
                    shape = str(res.shape).replace(',', '')
                    a = toq(res.flatten(), ktype=dtype)
                    return q(f'{{{shape}#x}}', a)
                else:
                    return toq(res, ktype=dtype)
            else:
                return res.astype(dtype)

    def __array_function__(self, func, types, args, kwargs):
        if not suppress_warnings:
            warn('Warning: Attempting to call numpy __array_function__ on a '
                 f'PyKX Vector type. __array_function__: {func}. Support for this method '
                 'is on a best effort basis. To suppress this warning please set the '
                 'configuration/environment variable PYKX_SUPPRESS_WARNINGS=True')
        a = []
        for i in range(len(args)):
            if isinstance(args[i], K):
                a.append(args[i].__typed_array__())
            else:
                a.append(args[i])
        return func(*a, **kwargs)

    def _unlicensed_getitem(self, index):
        if issubclass(type(index), K):
            index = index.np()
        if index < 0:
            index = len(self) + index
        return _wrappers.vector_unlicensed_getitem(self, index)

    def sorted(self):
        try:
            res = q('`s#', self)
            self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 's-type' in err_str:
                raise QError('Items are not in ascending order')
            elif 'type' in err_str:
                raise QError('Object does not support the sorted attribute')
            else:
                raise e

    def unique(self):
        try:
            res = q('`u#', self)
            self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 'u-type' in err_str:
                raise QError('Items are not unique')
            elif 'type' in err_str:
                raise QError('Object does not support the unique attribute')
            else:
                raise e

    def parted(self):
        try:
            res = q('`p#', self)
            self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 'u-type' in err_str:
                raise QError('Items are not parted')
            elif 'type' in err_str:
                raise QError('Object does not support the parted attribute')
            else:
                raise e

    def grouped(self):
        try:
            res = q('`g#', self)
            self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 'type' in err_str:
                raise QError('Object does not support the grouped attribute')
            else:
                raise e

    def replace(self, to_replace, replace_with):
        res = q('''
                {[l;s;r]
                lT:type l;
                rT:type r;
                sOp:$[(rT>=0) or lT=0;~/:;=];
                rI:where sOp[s;l];
                if[0=count rI;:l];
                atF:$[(0=lT) or neg[lT]=rT;@[;;:;];{1_ @[(::),x;1+y;:;z]}];
                r:count[rI]#enlist r;
                atF[l;rI;r]
            }''', self, to_replace, replace_with)
        return res


class List(Vector):
    """Wrapper for q lists, which are vectors of K objects of any type.

    Note: The memory layout of a q list is special.
        All other vector types (see: subclasses of [`pykx.Vector`][pykx.Vector]) are structured
        in-memory as a K object which contains metadata, followed immediately by the data in the
        vector. By contrast, q lists are a a vector of pointers to K objects, so they are structured
        in-memory as a K object containing metadata, followed immediately by pointers. As a result,
        the base data "contained" by the list is located elsewhere in memory. This has performance
        and ownership implications in q, which carry over to PyKX.
    """
    t = 0
    _np_dtype = object

    def _unlicensed_getitem(self, index):
        return _wrappers.list_unlicensed_getitem(self, index)

    @property
    def has_nulls(self) -> bool:
        return any(x.is_null if x.is_atom else False for x in self)

    @property
    def has_infs(self) -> bool:
        return any(x.is_inf if x.is_atom else False for x in self)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return [_rich_convert(x, stdlib, raw) for x in self]

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None, reshape: Union[bool, list] = False):  # noqa: E501
        """Provides a Numpy representation of the list."""
        if reshape == False: # noqa: E712
            return _wrappers.list_np(self, False, has_nulls, raw)
        if isinstance(reshape, bool):
            dims = q("{$[0=t:type x;count[x],'distinct raze .z.s each x;enlist(count x;neg t)]}", self)  # noqa: E501
            if len(dims) != 1:
                raise TypeError('Data must be a singular type "rectangular" matrix')
            dims = dims[0][:-1]
        else:
            dims = reshape
        razed = q('(raze/)', self)
        return razed.np().reshape(dims)

    def pt(self, *, reshape: Union[bool, list] = True):
        _check_beta('PyTorch Conversion')
        if _torch_unavailable:
            raise QError('PyTorch not available, please install PyTorch')
        import torch
        return torch.from_numpy(self.np(reshape=reshape))


class NumericVector(Vector):
    """Base type for all q numeric vectors."""
    pass


class IntegralNumericVector(NumericVector):
    """Base type for all q integral numeric vectors."""
    @property
    def has_nulls(self) -> bool:
        return (self._base_null_value == self.np(raw=True)).any()

    @property
    def has_infs(self) -> bool:
        a = self.np(raw=True)
        return (a == self._base_inf_value).any() or (a == -self._base_inf_value).any()

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return self.np(raw=True, has_nulls=has_nulls).tolist()
        return [pd.NA if x.is_null else x.py() for x in self]

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        if raw:
            has_nulls = False
        if has_nulls is None or has_nulls:
            nulls = self.np(raw=True) == self._base_null_value
            if has_nulls is None:
                has_nulls = nulls.any()
        arr = _wrappers.k_vec_to_array(self, self._np_type)
        if has_nulls:
            return np.ma.MaskedArray(
                arr,
                mask=nulls,
                copy=False,
                # Set the fill value to the value of the underlying null in q
                fill_value=-2 ** (arr.itemsize * 8 - 1)
            )
        return arr

    def pd(
        self,
        *,
        raw: bool = False,
        has_nulls: Optional[bool] = None,
        as_arrow: Optional[bool] = False,
    ):
        if as_arrow:
            if not pandas_2:
                raise RuntimeError('Pandas Version must be at least 2.0 to use as_arrow=True')
            if pa is None:
                raise PyArrowUnavailable # nocov
        arr = self.np(raw=raw, has_nulls=has_nulls)
        if as_arrow:
            arr = pa.array(arr)
            if raw:
                res = pd.Series(arr, copy=False, dtype=_as_arrow_raw_map[type(self).__name__])
            else:
                res = pd.Series(arr, copy=False, dtype=_as_arrow_map[type(self).__name__])
        else:
            if isinstance(arr, np.ma.MaskedArray):
                arr = pd.arrays.IntegerArray(arr, mask=arr.mask, copy=False)
            res = pd.Series(arr, copy=False)
        return res

    def pt(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        _check_beta('PyTorch Conversion')
        if _torch_unavailable:
            raise QError('PyTorch not available, please install PyTorch')
        import torch
        return torch.from_numpy(self.np(raw=raw, has_nulls=has_nulls))


class BooleanVector(IntegralNumericVector):
    """Wrapper for q boolean vectors."""
    t = 1
    _np_type = _wrappers.NPY_BOOL
    _np_dtype = None

    def _prototype(self=None):
        return BooleanVector([BooleanAtom._prototype(), BooleanAtom._prototype()])

    @property
    def has_nulls(self) -> bool:
        return False

    @property
    def has_infs(self) -> bool:
        return False

    # Custom np method for this IntegralNumericVector subclass because it never has nulls
    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return super().np(raw=raw, has_nulls=False)

    def __and__(self, rhs):
        return q('{x & y}', self, rhs)

    def __or__(self, rhs):
        return q('{x | y}', self, rhs)


def _raw_guids_to_arrow(guids):
    if pa is None:
        raise PyArrowUnavailable # nocov
    return pa.ExtensionArray.from_storage(
        arrow_uuid_type,
        pa.array(guids.view('S16'), pa.binary(16)),
    )


@pd.api.extensions.register_extension_dtype
class PandasUUIDType(pd.api.extensions.ExtensionDtype):
    na_value = 0+0j
    name = 'pykx.uuid'
    kind = 'c'
    type = np.complex128
    itemsize = 16

    def __from_arrow__(self, array):
        return _wrappers.pandas_uuid_type_from_arrow(self, array)


class PandasUUIDArray(pd.api.extensions.ExtensionArray):
    dtype = PandasUUIDType()

    def __init__(self, array):
        self.array = array

    def __arrow_array__(self, type=None):
        if pa is None:
            raise PyArrowUnavailable # nocov
        return _raw_guids_to_arrow(np.array(self))

    @classmethod
    def _from_sequence(cls, seq):
        return cls(np.asarray(seq))

    @classmethod
    def _from_factorized(cls, fac):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented. This function doesn't support PyKX UUID objects") # noqa: E501

    def __getitem__(self, key):
        return self.array[key]

    def __len__(self):
        return len(self.array)

    def __eq__(self, other):
        return self.array == other

    def __array__(self):
        return self.array

    def reshape(self, *args, **kwargs):
        return self

    @property
    def nbytes(self):
        return 16 * len(self)

    def isna(self):
        return pd.isna(self.array)

    def take(self, indices, allow_fill=False, fill_value=None):
        return self.array.take(indices)

    def copy(self):
        return PandasUUIDArray(np.array(self).copy())

    def _concat_same_type(self):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented. This function doesn't support PyKX UUID objects") # noqa: E501


if pa is not None:
    class ArrowUUIDType(pa.ExtensionType):
        def __init__(self):
            if pa is None:
                raise PyArrowUnavailable # nocov
            pa.ExtensionType.__init__(self, pa.binary(16), 'pykx.uuid')

        def __arrow_ext_serialize__(self):
            return b''

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return ArrowUUIDType()

        def to_pandas_dtype(self):
            return PandasUUIDType()

    arrow_uuid_type = ArrowUUIDType()
    pa.register_extension_type(arrow_uuid_type)
else: # nocov
    pass


class GUIDVector(Vector):
    """Wrapper for q GUID vectors."""
    t = 2
    _np_dtype = None

    def _prototype(self=None):
        return GUIDVector([UUID(int=0), UUID(int=0)])

    def _unlicensed_getitem(self, index):
        return _wrappers.guid_vector_unlicensed_getitem(self, index)

    @property
    def has_nulls(self) -> bool:
        return (0j == self.np(raw=True)).any()

    @property
    def has_infs(self) -> bool:
        return False

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return [x.py(raw=raw) for x in self]

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return _wrappers.guid_vector_np(self, raw, has_nulls)

    def pa(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        if pa is None:
            raise PyArrowUnavailable # nocov
        return _raw_guids_to_arrow(self.np(raw=True))

    def pd(
        self,
        *,
        raw: bool = False,
        has_nulls: Optional[bool] = None,
        as_arrow: Optional[bool] = False,
    ):
        if raw:
            return PandasUUIDArray(self.np(raw=raw))
        else:
            res = super().pd()
            return res


class ByteVector(IntegralNumericVector):
    """Wrapper for q byte (i.e. 8 bit unsigned integer) vectors."""
    t = 4
    _np_type = _wrappers.NPY_UINT8
    _np_dtype = np.uint8

    def _prototype(self=None):
        return ByteVector([ByteAtom._prototype(), ByteAtom._prototype()])

    @property
    def has_nulls(self) -> bool:
        return False

    @property
    def has_infs(self) -> bool:
        return False

    # Custom np method for this IntegralNumericVector subclass because it never has nulls
    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return super().np(raw=raw, has_nulls=False)


class ShortVector(IntegralNumericVector):
    """Wrapper for q short (i.e. 16 bit signed integer) vectors."""
    t = 5
    _np_type = _wrappers.NPY_INT16
    _np_dtype = np.int16
    _base_null_value = NULL_INT16
    _base_inf_value = INF_INT16

    def _prototype(self=None):
        return ShortVector([ShortAtom._prototype(), ShortAtom._prototype()])


class IntVector(IntegralNumericVector):
    """Wrapper for q int (i.e. 32 bit signed integer) vectors."""
    t = 6
    _np_type = _wrappers.NPY_INT32
    _np_dtype = np.int32
    _base_null_value = NULL_INT32
    _base_inf_value = INF_INT32

    def _prototype(self=None):
        return IntVector([IntAtom._prototype(), IntAtom._prototype()])


class LongVector(IntegralNumericVector):
    """Wrapper for q long (i.e. 64 bit signed integer) vectors."""
    t = 7
    _np_type = _wrappers.NPY_INT64
    _np_dtype = np.int64
    _base_null_value = NULL_INT64
    _base_inf_value = INF_INT64

    def _prototype(self=None):
        return LongVector([LongAtom._prototype(), LongAtom._prototype()])


class NonIntegralNumericVector(NumericVector):
    """Base type for all q non-integral numeric vectors."""
    @property
    def has_nulls(self) -> bool:
        a = self.np()
        return np.isnan(np.dot(a, a)) # `np.dot` can be used as a high-performance NaN check

    @property
    def has_infs(self) -> bool:
        return (self.np() == np.inf).any()

    def __pow__(self, other, mod=None):
        if mod is None:
            return q('xexp', self, other)
        raise TypeError(f"pow() 3rd argument not allowed when using '{type(self)}' as the base")

    def __rpow__(self, other, mod=None):
        if mod is None:
            return q('xexp', other, self)
        raise TypeError(f"pow() 3rd argument not allowed when using '{type(self)}' as the exponent")

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return _wrappers.k_vec_to_array(self, self._np_type)

    def pt(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        _check_beta('PyTorch Conversion')
        if _torch_unavailable:
            raise QError('PyTorch not available, please install PyTorch')
        import torch
        return torch.from_numpy(self.np(raw=raw, has_nulls=has_nulls))


class RealVector(NonIntegralNumericVector):
    """Wrapper for q real (i.e. 32 bit float) vectors."""
    t = 8
    _np_type = _wrappers.NPY_FLOAT32
    _np_dtype = np.float32

    def _prototype(self=None):
        return RealVector([RealAtom._prototype(), RealAtom._prototype()])


class FloatVector(NonIntegralNumericVector):
    """Wrapper for q float (i.e. 64 bit float) vectors."""
    t = 9
    _np_type = _wrappers.NPY_FLOAT64
    _np_dtype = np.float64

    def _prototype(self=None):
        return FloatVector([FloatAtom._prototype(), FloatAtom._prototype()])


class CharVector(Vector):
    """Wrapper for q char (i.e. 8 bit ASCII value) vectors.

    See Also:
        [`pykx.SymbolAtom`][pykx.SymbolAtom]
    """
    t = 10
    _np_dtype = '|S1'

    def _prototype(self=None):
        return CharVector("  ")

    def _unlicensed_getitem(self, index: int):
        return _wrappers.char_vector_unlicensed_getitem(self, index)

    def __bytes__(self):
        if _wrappers.k_n(self):
            return self.np().tobytes()
        return b''

    def __str__(self):
        if _wrappers.k_n(self):
            return self.np().tobytes().decode()
        return ''

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return _wrappers.get_char_list(self)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return _wrappers.k_vec_to_array(self, _wrappers.NPY_UINT8).view('|S1')


class SymbolVector(Vector):
    """Wrapper for q symbol vectors.

    Danger: Unique symbols are never deallocated!
        Reserve symbol data for values that are recurring. Avoid using symbols for data being
        generated over time (e.g. random symbols) as memory usage will continually increase.

    """
    t = 11
    _np_dtype = object

    def _prototype(self=None):
        return SymbolVector([SymbolAtom._prototype(), SymbolAtom._prototype()])

    def _unlicensed_getitem(self, index: int):
        return _wrappers.symbol_vector_unlicensed_getitem(self, index)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return _wrappers.get_symbol_list(self, raw)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return _wrappers.symbol_vector_np(self, raw, has_nulls)


class TemporalVector(Vector):
    """Base type for all q temporal vectors."""
    @property
    def has_nulls(self) -> bool:
        return (self._base_null_value == self.np(raw=True)).any()

    @property
    def has_infs(self) -> bool:
        a = self.np(raw=True)
        return (a == self._base_inf_value).any() or (a == -self._base_inf_value).any()

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return self.np(raw=True, has_nulls=has_nulls).tolist()
        return [pd.NaT if x.is_null else x.py() for x in self]

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        base_array = _wrappers.k_vec_to_array(self, self._np_base_type)
        if raw:
            return base_array

        if isinstance(self, TemporalFixedVector):
            # BUG: This will cause 0Wp to overflow and become 1707-09-22. We could check for 0Wp,
            # and not add the offset to it, but is the cost of doing that on every
            # `TimestampVector.np` call worth avoiding this edge case?
            array = base_array + self._epoch_offset
        else:
            array = base_array

        if _wrappers.NPY_TYPE_ITEMSIZE[self._np_base_type] == self._np_type.itemsize:
            array = array.view(self._np_type)
        else:
            array = array.astype(self._np_type, copy=False)
        if has_nulls is None or has_nulls:
            nulls = base_array == self._base_null_value
            if has_nulls is None:
                has_nulls = nulls.any()
        if has_nulls:
            is_fixed = isinstance(self, TemporalFixedVector)
            array[nulls] = np.datetime64('NaT') if is_fixed else np.timedelta64('NaT')
        return array


class TemporalFixedVector(TemporalVector):
    """Base type for all q temporal vectors which represent a fixed date/time."""
    pass


class TemporalSpanVector(TemporalVector):
    """Base type for all q temporal vectors which represent a span of time."""
    pass


class TimestampVector(TemporalFixedVector):
    """Wrapper for q timestamp vectors."""
    t = 12
    _np_base_type = _wrappers.NPY_INT64
    _np_dtype = 'datetime64[ns]'
    _np_type = _wrappers.NPY_DATETIME64_NS
    _epoch_offset = TIMESTAMP_OFFSET
    _base_null_value = NULL_INT64
    _base_inf_value = INF_INT64

    def _prototype(self=None):
        return TimestampVector([
            datetime(2150, 10, 22, 20, 31, 15, 70713),
            datetime(2150, 10, 22, 20, 31, 15, 70713)]
        )

    @property
    def date(self):
        return q('{`date$x}', self)

    @property
    def time(self):
        return q('{`time$x}', self)

    @property
    def year(self):
        return IntVector(q('{`year$x}', self))

    @property
    def month(self):
        return IntVector(q('{`mm$x}', self))

    @property
    def day(self):
        return IntVector(q('{`dd$x}', self))

    @property
    def hour(self):
        return IntVector(q('{`hh$x}', self))

    @property
    def minute(self):
        return IntVector(q('{`uu$x}', self))

    @property
    def second(self):
        return IntVector(q('{`ss$x}', self))

    def py(self,
           *,
           raw: bool = False,
           has_nulls: Optional[bool] = None,
           stdlib: bool = True,
           tzinfo: Optional[pytz.BaseTzInfo] = None,
           tzshift: bool = True
    ):
        # XXX: Since Python datetime objects don't support nanosecond
        #      precision (https://bugs.python.org/issue15443), we have to
        #      convert to datetime64[us] before converting to datetime objects
        if raw:
            return self.np(raw=True, has_nulls=has_nulls).tolist()
        if tzinfo is not None:
            if tzshift:
                return [x.replace(tzinfo=pytz.utc).astimezone(tzinfo)
                        for x in self.np().astype('datetime64[us]').astype(datetime).tolist()]
            else:
                return [x.replace(tzinfo=tzinfo)
                        for x in self.np().astype('datetime64[us]').astype(datetime).tolist()]
        converted_vector=self.np().astype('datetime64[us]').astype(datetime).tolist()
        null_pos=[]
        for x in converted_vector:
            if x is None:
                null_pos.append(converted_vector.index(x))
        for i in null_pos:
            converted_vector[i]=pd.NaT
        return converted_vector


class MonthVector(TemporalFixedVector):
    """Wrapper for q month vectors."""
    t = 13
    _np_base_type = _wrappers.NPY_INT32
    _np_type = _wrappers.NPY_DATETIME64_M
    _np_dtype = 'datetime64[M]'
    _epoch_offset = MONTH_OFFSET
    _base_null_value = NULL_INT32
    _base_inf_value = INF_INT32

    def _prototype(self=None):
        return MonthVector([np.datetime64('1972-05', 'M'), np.datetime64('1972-05', 'M')])


class DateVector(TemporalFixedVector):
    """Wrapper for q date vectors."""
    t = 14
    _np_base_type = _wrappers.NPY_INT32
    _np_type = _wrappers.NPY_DATETIME64_D
    _np_dtype = 'datetime64[D]'
    _epoch_offset = DATE_OFFSET
    _base_null_value = NULL_INT32
    _base_inf_value = INF_INT32

    def _prototype(self=None):
        return DateVector([np.datetime64('1972-05-31', 'D'), np.datetime64('1972-05-31', 'D')])


class DatetimeVector(TemporalFixedVector):
    """Wrapper for q datetime vectors.

    Warning: The q datetime type is deprecated.
        PyKX does not provide a rich interface for the q datetime type, as it is depreceated. Avoid
        using it whenever possible.
    """
    t = 15
    _np_dtype = np.float64

    def __init__(self, *args, **kwargs):
        warn('The q datetime type is deprecated', DeprecationWarning)
        super().__init__(*args, **kwargs)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        if raw:
            return _wrappers.k_vec_to_array(self, _wrappers.NPY_FLOAT64)
        raise TypeError('The q datetime type is deprecated, and can only be accessed with the '
                        'keyword argument `raw=True`')

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return self.np(raw=True, has_nulls=has_nulls).tolist()
        raise TypeError('The q datetime type is deprecated, and can only be accessed with the '
                        'keyword argument `raw=True`')


class TimespanVector(TemporalSpanVector):
    """Wrapper for q timespan vectors."""
    t = 16
    _np_base_type = _wrappers.NPY_INT64
    _np_dtype = 'timedelta64[ns]'
    _np_type = _wrappers.NPY_TIMEDELTA64_NS
    _base_null_value = NULL_INT64
    _base_inf_value = INF_INT64

    def _prototype(self=None):
        return TimespanVector([
            np.timedelta64(3796312051664551936, 'ns'),
            np.timedelta64(3796312051664551936, 'ns')]
        )


class MinuteVector(TemporalSpanVector):
    """Wrapper for q minute vectors."""
    t = 17
    _np_base_type = _wrappers.NPY_INT32
    _np_dtype = 'timedelta64[m]'
    _np_type = _wrappers.NPY_TIMEDELTA64_M
    _base_null_value = NULL_INT32
    _base_inf_value = INF_INT32

    def _prototype(self=None):
        return MinuteVector([np.timedelta64(216, 'm'), np.timedelta64(216, 'm')])


class SecondVector(TemporalSpanVector):
    """Wrapper for q second vectors."""
    t = 18
    _np_base_type = _wrappers.NPY_INT32
    _np_dtype = 'timedelta64[s]'
    _np_type = _wrappers.NPY_TIMEDELTA64_S
    _base_null_value = NULL_INT32
    _base_inf_value = INF_INT32

    def _prototype(self=None):
        return SecondVector([np.timedelta64(13019, 's'), np.timedelta64(13019, 's')])


class TimeVector(TemporalSpanVector):
    """Wrapper for q time vectors."""
    t = 19
    _np_base_type = _wrappers.NPY_INT32
    _np_dtype = 'timedelta64[ms]'
    _np_type = _wrappers.NPY_TIMEDELTA64_MS
    _base_null_value = NULL_INT32
    _base_inf_value = INF_INT32

    def _prototype(self=None):
        return TimeVector([np.timedelta64(59789214, 'ms'), np.timedelta64(59789214, 'ms')])


class EnumVector(Vector):
    """Wrapper for q enum vectors.

    Parameters:
        variable: The handle of a list in q memory.
        indices: An list used in [Enumeration](https://code.kx.com/q/ref/enumeration/).
        values: A list of items that is used in [Enumerate](https://code.kx.com/q/ref/enumerate/) and [Enum Extend](https://code.kx.com/q/ref/enum-extend/).
        extend:  A boolean set to True to use [Enum Extend](https://code.kx.com/q/ref/enum-extend/) and False to use [Enumerate](https://code.kx.com/q/ref/enumerate/).
    """ # noqa: E501
    t = 20

    def __new__(cls, variable, indices=None, values=None, extend=False):
        if not isinstance(variable, (str, SymbolAtom)):
            raise TypeError("Variable name must be of type String or Symbol.")
        if not (indices is None) ^ (values is None):
            raise AttributeError("Can only set one of 'values' and 'indices' at one time.")
        if indices is not None:
            return q('!', variable, indices)
        if values is not None:
            if extend:
                return q('?', variable, values)
            else:
                return q('$', variable, values)

    def values(self):
        """Returns the resolved value of the enumeration"""
        return q.value(self)

    def domain(self):
        """Returns the name of the domain of the enum"""
        return q.key(self)

    def indices(self):
        """Returns the indices of the enum in the q list"""
        return q('`long$', self)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        if raw:
            return _wrappers.k_vec_to_array(self, _wrappers.NPY_INT64)
        return q('value', self).np()

    def pd(
        self,
        *,
        raw: bool = False,
        has_nulls: Optional[bool] = None,
        as_arrow: Optional[bool] = False,
    ):
        if raw:
            res = super().pd(raw=raw, has_nulls=has_nulls)
            if as_arrow:
                if not pandas_2:
                    raise RuntimeError('Pandas Version must be at least 2.0 to use as_arrow=True')
                if pa is None:
                    raise PyArrowUnavailable # nocov
                res = res.astype('int64[pyarrow]')
            return res
        res = pd.Series(self.np(raw=raw, has_nulls=has_nulls), dtype='category')
        return res


class Anymap(List):
    """Wrapper for q mapped lists, also known as "anymaps"."""
    t = 77

    def _as_list(self):
        return q('{x til count x}', self)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return self._as_list().py(raw=raw, has_nulls=has_nulls, stdlib=stdlib)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return self._as_list().np(raw=raw, has_nulls=has_nulls)

    def pd(
        self,
        *,
        raw: bool = False,
        has_nulls: Optional[bool] = None,
        as_arrow: Optional[bool] = False,
    ):
        res = self._as_list().pd(raw=raw, has_nulls=has_nulls, as_arrow=as_arrow)
        return res

    def pa(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return self._as_list().pa(raw=raw, has_nulls=has_nulls)


# TODO:
# class NestedSymbolEnum(EnumVector):
#     pass


def _check_k_mapping_key(key: Any, k_key: K, valid_keys):
    if isinstance(k_key, abc.Iterable):
        if not all(any((k == x).all() for x in valid_keys) for k in k_key):
            raise KeyError(key)
    elif not any((k_key == x).all() for x in valid_keys):
        raise KeyError(key)


class Mapping(Collection, abc.Mapping):
    """Base type for all q mappings, including tables, and dictionaries."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys = None
        self._values = None

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return self._keys.__iter__()

    @property
    def has_nulls(self) -> bool:
        return any(x.is_null if x.is_atom else x.has_nulls for x in self._values)

    @property
    def has_infs(self) -> bool:
        return any(x.is_inf if x.is_atom else x.has_infs for x in self._values)

    def any(self) -> bool:
        return any(x.any() for x in self._values)

    def all(self) -> bool:
        return all(x.all() for x in self._values)

    def keys(self):
        return self._keys

    def values(self):
        return self._values

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        yield from ((k, self[k]) for k in self)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return dict(zip(
            self._keys.py(raw=raw, has_nulls=has_nulls, stdlib=stdlib),
            self._values.py(raw=raw, has_nulls=has_nulls, stdlib=stdlib),
        ))


from .pandas_api import GTable_init, PandasAPI


def _col_name_generator():
    name = 'x'
    yield name
    i = 0
    while True:
        i += 1
        yield name + str(i)


class Table(PandasAPI, Mapping):
    """Wrapper for q tables, including in-memory tables, splayed tables, and partitioned tables.

    Note: Despite the name, [keyed tables][pykx.KeyedTable] are actually dictionaries.

    See Also:
        - [`pykx.SplayedTable`][pykx.SplayedTable]
        - [`pykx.PartitionedTable`][pykx.PartitionedTable]
    """
    t = 98

    def __new__(cls, *args, cast: bool = None, **kwargs):
        if 'data' in kwargs.keys():
            return toq(q.flip(Dictionary(kwargs['data'])), ktype=Table, cast=cast)
        if len(args) > 0 and (
            (isinstance(args[0], list) or issubclass(type(args[0]), Vector)
             or isinstance(args[0], np.ndarray))
        ):
            cols = []
            if 'columns' in kwargs.keys():
                cols = kwargs['columns']
            n_cols = len(args[0][0])
            n_gen = _col_name_generator()
            while len(cols) < n_cols:
                name = next(n_gen)
                if name not in cols:
                    cols.append(name)
            return q('{x !/: y}', SymbolVector(cols), args[0])
        # TODO: 'strict' and 'cast' flags
        return toq(*args, ktype=None if cls is K else cls, cast=cast)

    def __init__(self, *args, **kwargs):
        if ('data' not in kwargs.keys()
            and not (len(args) > 0 and (isinstance(args[0], list)
                     or issubclass(type(args[0]), Vector) or isinstance(args[0], np.ndarray)))
        ):
            Mapping.__init__(self, *args, **kwargs)
            _wrappers.table_init(self)


    def prototype(self={}): # noqa
        _map = {}
        for k, v in self.items():
            _map[k] = v._prototype() if (isinstance(v, type) or isinstance(v, ABCMeta)) else v
        return Table(Dictionary(_map))

    def __len__(self):
        if licensed:
            return int(q('#:', self))
        return int(len(self._values._unlicensed_getitem(0)))

    def ungroup(self):
        return q.ungroup(self)

    def __getitem__(self, key):
        n = len(self)
        if isinstance(key, Integral):
            key = _key_preprocess(key, n)
        res = self.loc[key]
        if isinstance(res, List) and len(res) == 1:
            res = q('{raze x}', res)
        return res

    def __setitem__(self, key, val):
        self.loc[key] = val

    @property
    def flip(self):
        return _wrappers.k_from_addr(Dictionary, _wrappers.k_k(self), True)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return dict(zip(
            self._keys.py(raw=raw, has_nulls=has_nulls, stdlib=stdlib),
            self._values.py(raw=raw, has_nulls=has_nulls, stdlib=stdlib),
        ))

    def pd(
        self,
        *,
        raw: bool = False,
        has_nulls: Optional[bool] = None,
        raw_guids=False,
        as_arrow: Optional[bool] = False,
    ):
        if raw_guids and not raw:
            v = [x.np(raw=isinstance(x, GUIDVector), has_nulls=has_nulls) for x in self._values]
            v = [PandasUUIDArray(x) if x.dtype == complex else x for x in v]
        else:
            v = [x.np(raw=raw, has_nulls=has_nulls) for x in self._values]
        if pandas_2:
            # The current behavior is a bug and will raise an error in the future, this change
            # proactively fixes that for us
            for i in range(len(v)):
                if v[i].dtype == np.dtype('datetime64[D]'):
                    v[i] = v[i].astype(np.dtype('datetime64[s]'))
                elif v[i].dtype == np.dtype('datetime64[M]'):
                    v[i] = v[i].astype(np.dtype('datetime64[s]'))
        df = df_from_arrays(pd.Index(self._keys), v, pd.RangeIndex(len(self)))
        _pykx_base_types = {}
        for i, v in enumerate(self._values):
            if not raw and isinstance(v, EnumVector):
                df = df.astype({self._keys.py()[i]: 'category'})
            _pykx_base_types[self._keys.py()[i]] = str(type(v).__name__)
        df.attrs['_PyKX_base_types'] = _pykx_base_types
        if as_arrow:
            if not pandas_2:
                raise RuntimeError('Pandas Version must be at least 2.0 to use as_arrow=True')
            if pa is None:
                raise PyArrowUnavailable # nocov
            if raw:
                t_dict = dict(filter(lambda i: i[1] != 'GUIDVector', _pykx_base_types.items()))
                df = df.astype(dict([(k, _as_arrow_raw_map[v])
                                    for k, v in t_dict.items()]))
            else:
                df = df.astype(dict([(k, _as_arrow_map[v]) for k, v in _pykx_base_types.items()]))
        return df

    def pa(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        if pa is None:
            raise PyArrowUnavailable # nocov
        try:
            return pa.Table.from_pandas(self.pd(raw=raw, has_nulls=has_nulls, raw_guids=True))
        except (pa.lib.ArrowNotImplementedError, pa.lib.ArrowInvalid, pa.ArrowInvalid) as err:
            raise QError('Unable to convert pykx.List column with non conforming types '
                         f'to PyArrow,\n        failed with error: {err}')

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return self.pd(raw=raw, has_nulls=has_nulls).to_records(index=False)

    def insert(
        self,
        row: Union[list, List],
        match_schema: bool = False,
        test_insert: bool = False,
        inplace: bool = True
    ):
        """Helper function around `q`'s `insert` function which inserts a row or multiple rows into
        a q Table object.

        Parameters:
            row: A list of objects to be inserted as a row.
            match_schema: Whether the row/rows to be inserted must match the tables current schema.
            test_insert: Causes the function to modify a small local copy of the table and return
                the modified example, this can only be used with embedded q and will not modify the
                source tables contents.
            inplace: Causes the underlying Table python object to update itself with the
                resulting Table after the insert.

        Returns:
            The resulting table after the given row has been inserted.

        Raises:
            PyKXException: If the `match_schema` parameter is used this function may raise an error
                if the row to be inserted does not match the tables schema. The error message will
                contain information about which columns did not match.

        Examples:

        Insert a single row onto a Table, ensuring the new row matches the current tables schema.

        ```Python
        >>> tab.insert([1, 2.0, datetime.datetime(2020, 2, 24)], match_schema=True)
        ```
        """
        q['.pykx.i.itab'] = self
        q.insert('.pykx.i.itab', row, match_schema, test_insert)
        res = q('.pykx.i.itab')
        if inplace:
            self.__dict__.update(res.__dict__)
        q('delete itab from `.pykx.i')
        return res

    def upsert(
        self,
        row: Union[list, List],
        match_schema: bool = False,
        test_insert: bool = False,
        inplace: bool = True
    ):
        """Helper function around `q`'s `upsert` function which inserts a row or multiple rows into
        a q Table object.

        Parameters:
            row: A list of objects to be inserted as a row.
            match_schema: Whether the row/rows to be inserted must match the tables current schema.
            test_insert: Causes the function to modify a small local copy of the table and return
                the modified example, this can only be used with embedded q and will not modify the
                source tables contents.
            inplace: Causes the underlying Table python object to update itself with the
                resulting Table after the upsert.

        Returns:
            The resulting table after the given row has been upserted.

        Raises:
            PyKXException: If the `match_schema` parameter is used this function may raise an error
                if the row to be inserted does not match the tables schema. The error message will
                contain information about which columns did not match.

        Examples:

        Upsert a single row onto a Table, ensuring the new row matches the current tables schema.

        ```Python
        >>> tab.upsert([1, 2.0, datetime.datetime(2020, 2, 24)], match_schema=True)
        ```
        """
        res = q.upsert(self, row, match_schema, test_insert)
        if inplace:
            self.__dict__.update(res.__dict__)
        return res

    def sorted(self, cols: Union[List, str] = ''):
        try:
            if len(cols) == 0:
                cols = q.cols(self)[0]
            if not (isinstance(cols, List) or isinstance(cols, list)):
                cols = [cols]
            for col in cols:
                res = q.qsql.update(self, {col: f'`s#{col}'})
                self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 's-type' in err_str:
                raise QError('Items are not in ascending order')
            elif 'type' in err_str:
                raise QError('Object does not support the sorted attribute')
            else:
                raise e

    def sql(self, query, *args):
        """Execute an SQL query against the supplied PyKX tabular object.

        This function expects the table object to be supplied as a parameter
        to the query, additional parameters can be supplied as positional
        arguments.

        Parameters:
            query: A str object indicating the query to be executed, this
                must contain a required argument $1 associated with the
                table being queried.
            *args: Any additional positional arguments required for query
                execution.

        Returns:
            The queried table associated with the SQL statement

        Examples:

        Query a simple table supplying no additional arguments

        ```python
        >>> tab = kx.Table(data = {'x': [1, 2, 3], 'x1': ['a', 'b', 'a']})
        >>> tab.sql("select * from $1 where x1='a'")
        pykx.Table(pykx.q('
        x x1
        ----
        1 a
        3 a
        '))
        ```

        Query a simple table supplying multiple arguments

        ```python
        >>> tab = kx.Table(data = {'x': [1, 2, 3], 'x1': ['a', 'b', 'a']})
        >>> tab.sql("select * from $1 where x1=$2 and x=$3", 'a', 1)
        pykx.Table(pykx.q('
        x x1
        ----
        1 a
        '))
        ```
        """
        if not isinstance(query, str):
            raise TypeError('Supplied query is not of type "str"')
        if '$1' not in query:
            raise QError('Supplied query does not contain argument $1')
        return q.sql(query, self, *args)

    def select(self, columns=None, where=None, by=None, inplace=False):
        """Apply a q style select statement on the supplied table defined within the process.

        This implementation follows the q functional select syntax with limitations on
        structures supported for the various clauses a result of this.

        Parameters:
            columns: A dictionary mapping the name to be given to a column and the logic to be
                applied in aggregation to that column both as strings.
            where: Conditional filtering used to select subsets of the data on which by-clauses and
                appropriate aggregations are to be applied.
            by: A dictionary mapping the names to be assigned to the produced columns and the
                columns whose results are used to construct the groups of the by clause.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects.
                See [here](https://code.kx.com/q/basics/qsql/#result-and-side-effects).

        Examples:

        Define a q table in python, and give it a name in q memory

        ```python
        >>> import pykx as kx
        >>> qtab = kx.Table(data = {
        ...     'col1': kx.random.random(100, ['a', 'b', 'c']),
        ...     'col2': kx.random.random(100, 1.0),
        ...     'col3': kx.random.random(100, False),
        ...     'col4': kx.random.random(100, 10.0)})
        ```

        Select all items in the table

        ```python
        >>> qtab.select()
        ```

        Filter table based on various where conditions

        ```python
        >>> qtab.select(where='col2<0.5')
        ```

        Retrieve statistics by grouping data on symbol columns

        ```python
        >>> qtab.select(columns={'maxCol2': 'max col2'}, by={'col1': 'col1'})
        >>> qtab.select(columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'})
        ```

        Retrieve grouped statistics with restrictive where condition

        ```python
        >>> qtab.select(columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'}, where='col3=0b')
        ```
        """ # noqa: E501
        return q.qsql.select(self, columns, where, by, inplace)

    def exec(self, columns=None, where=None, by=None):
        """
        Apply a q style exec statement on the supplied PyKX Table.

        This implementation follows the q functional exec syntax with limitations on structures
        supported for the various clauses a result of this.

        Parameters:
            columns: A dictionary mapping the name to be given to a column and the logic to be
                applied in aggregation to that column both as strings. A string defining a single
                column to be retrieved from the table as a list.
            where: Conditional filtering used to select subsets of the data on which by clauses and
                appropriate aggregations are to be applied.
            by: A dictionary mapping the names to be assigned to the produced columns and the
                the columns whose results are used to construct the groups of the by clause.

        Examples:

        Define a PyKX Table

        ```python
        >>> qtab = kx.Table(data = {
        ...     'col1': kx.random.random(100, ['a', 'b', 'c']),
        ...     'col2': kx.random.random(100, 1.0),
        ...     'col3': kx.random.random(100, False),
        ...     'col4': kx.random.random(100, 10.0)}
        ...     )
        ```

        Select last item of the table

        ```python
        qtab.exec()
        ```

        Retrieve a column from the table as a list

        ```python
        qtab.exec('col3')
        ```

        Retrieve a set of columns from a table as a dictionary

        ```python
        qtab.exec({'symcol': 'col1'})
        qtab.exec({'symcol': 'col1', 'boolcol': 'col3'})
        ```

        Filter columns from a table based on various where conditions

        ```python
        qtab.exec('col3', where='col1=`a')
        qtab.exec({'symcol': 'col1', 'maxcol4': 'max col4'}, where=['col1=`a', 'col2<0.3'])
        ```

        Retrieve data grouping by data on symbol columns

        ```python
        qtab.exec('col2', by={'col1': 'col1'})
        qtab.exec(columns={'maxCol2': 'max col2'}, by={'col1': 'col1'})
        qtab.exec(columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'})
        ```

        Retrieve grouped statistics with restrictive where condition

        ```python
        qtab.exec(columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'}, where='col3=0b')
        ```
        """ # noqa: E501
        return q.qsql.exec(self, columns, where, by)

    def update(self, columns=None, where=None, by=None, inplace=False):
        """
        Apply a q style update statement on tables defined within the process.

        This implementation follows the q functional update syntax with limitations on
        structures supported for the various clauses a result of this.

        Parameters:
            columns: A dictionary mapping the name of a column present in the table or one to be
                added to the contents which are to be added to the column, this content can be a
                string denoting q data or the equivalent Python data.
            where: Conditional filtering used to select subsets of the data on which by-clauses and
                appropriate aggregations are to be applied.
            by: A dictionary mapping the names to be assigned to the produced columns and the
                columns whose results are used to construct the groups of the by clause.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects.
                See [here](https://code.kx.com/q/basics/qsql/#result-and-side-effects).

        Examples:

        Define a q table in python and named in q memory

        ```python
        >>> qtab = kx.Table(data={
        ...         'name': ['tom', 'dick', 'harry'],
        ...         'age': [28, 29, 35],
        ...         'hair': ['fair', 'dark', 'fair'],
        ...         'eye': ['green', 'brown', 'gray']}
        ...     )
        ```

        Update all the contents of a column

        ```python
        qtab.update({'eye': '`blue`brown`green'})
        ```

        Update the content of a column restricting scope using a where clause

        ```python
        qtab.update({'eye': ['blue']}, where='hair=`fair')
        ```

        Define a q table suitable for by clause example

        ```python
        >>> bytab = kx.Table(data={
        ...         'name': kx.random.random(100, ['nut', 'bolt', 'screw']),
        ...         'color': kx.random.random(100, ['red', 'green', 'blue']),
        ...         'weight': 0.5 * kx.random.random(100, 20),
        ...         'city': kx.random.random(100, ['london', 'paris', 'rome'])})
        ```

        Apply an update grouping based on a by phrase

        ```python
        bytab.update({'weight': 'avg weight'}, by={'city': 'city'})
        ```

        Apply an update grouping based on a by phrase and persist the result using the inplace keyword

        ```python
        bytab.update(columns={'weight': 'avg weight'}, by={'city': 'city'}, inplace=True)
        ```
        """ # noqa: E501
        return q.qsql.update(self, columns, where, by, inplace)

    def delete(self, columns=None, where=None, inplace=False):
        """
        Apply a q style delete statement a PyKX table defined.

        This implementation follows the q functional delete syntax with limitations on
        structures supported for the various clauses a result of this.

        Parameters:
            columns: Denotes the columns to be deleted from a table.
            where: Conditional filtering used to select subsets of the data which are to be
                deleted from the table.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects.
                See [here](https://code.kx.com/q/basics/qsql/#result-and-side-effects).

        Examples:

        Define a PyKX Table against which to run the examples

        ```python
        >>> qtab = kx.Table(data = {
        ...             'name': ['tom', 'dick', 'harry'],
        ...             'age': [28, 29, 35],
        ...             'hair': ['fair', 'dark', 'fair'],
        ...             'eye': ['green', 'brown', 'gray']}
        ...             )
        ```

        Delete all the contents of the table

        ```python
        >>> qtab.delete()
        ```

        Delete single and multiple columns from the table

        ```python
        >>> qtab.delete('age')
        >>> qtab.delete(['age', 'eye'])
        ```

        Delete rows of the dataset based on where condition

        ```python
        >>> qtab.delete(where='hair=`fair')
        >>> qtab.delete(where=['hair=`fair', 'age=28'])
        ```

        Delete a column from the dataset named in q memory and persist the result using the
        inplace keyword

        ```python
        >>> qtab.delete('age', inplace=True)
        ```
        """ # noqa: E501
        return q.qsql.delete(self, columns, where, inplace)

    def reorder_columns(self, cols: Union[List, str], inplace: bool = False):
        """
        Reorder the columns of a supplied table, using a supplied list of columns.
            This list order the columns in the supplied order, if less than the total number
            of columns in the original table are supplied then the supplied columns will be first
            columns in the new table

        Parameters:
            cols: The column(s) which will be used to reorder the columns of the table
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects.
                See [here](https://code.kx.com/q/basics/qsql/#result-and-side-effects).

        Returns:
            The resulting table after the columns have been rearranged.

        Examples:

        Order a single column to be the first column in a table

        ```python
        >>> tab = kx.Table(data={
        ...     'a': [1, 2, 3],
        ...     'b': ['a', 'b', 'c'],
        ...     'c': [1.0, 2.0, 3.0]
        ... })
        >>> tab.reorder_columns('c')
        pykx.Table(pykx.q('
        c a b
        -----
        1 1 a
        2 2 b
        3 3 c
        '))
        ```

        Reorder all columns within a table

        ```python
        >>> tab = kx.Table(data={
        ...     'a': [1, 2, 3],
        ...     'b': ['a', 'b', 'c'],
        ...     'c': [1.0, 2.0, 3.0]
        ... })
        >>> tab.reorder_columns(['b', 'c', 'a'])
        pykx.Table(pykx.q('
        b c a
        -----
        a 1 1
        b 2 2
        c 3 3
        '))
        ```
        """
        tab_cols = self.columns.py()
        if isinstance(cols, list):
            for i in cols:
                if not isinstance(i, str):
                    raise QError(f'Supplied column "{i}" is not a string')
                if i not in tab_cols:
                    raise QError(f'Supplied column "{i}" not in table columns')
        elif isinstance(cols, str):
            if cols not in tab_cols:
                raise QError(f'Supplied column "{cols}" not in table columns')
        else:
            raise QError('Supplied column is not a string or list')
        res = q.xcols(cols, self)
        if inplace:
            self.__dict__.update(res.__dict__)
        return res

    def unique(self, cols: Union[List, str] = ''):
        try:
            if len(cols) == 0:
                cols = q.cols(self)[0]
            if not (isinstance(cols, List) or isinstance(cols, list)):
                cols = [cols]
            for col in cols:
                res = q.qsql.update(self, {col: f'`u#{col}'})
                self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 'u-type' in err_str:
                raise QError('Items are not unique')
            elif 'type' in err_str:
                raise QError('Object does not support the unique attribute')
            else:
                raise e

    def parted(self, cols: Union[List, str] = ''):
        try:
            if len(cols) == 0:
                cols = q.cols(self)[0]
            if not (isinstance(cols, List) or isinstance(cols, list)):
                cols = [cols]
            for col in cols:
                res = q.qsql.update(self, {col: f'`p#{col}'})
                self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 'u-type' in err_str:
                raise QError('Items are not parted')
            elif 'type' in err_str:
                raise QError('Object does not support the parted attribute')
            else:
                raise e

    def grouped(self, cols: Union[List, str] = ''):
        try:
            if len(cols) == 0:
                cols = q.cols(self)[0]
            if not (isinstance(cols, List) or isinstance(cols, list)):
                cols = [cols]
            for col in cols:
                res = q.qsql.update(self, {col: f'`g#{col}'})
                self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 'type' in err_str:
                raise QError('Object does not support the grouped attribute')
            else:
                raise e

    def xbar(self, values):
        """
        Apply `xbar` round down operations on the column(s) of a table to a specified
            value

        Parameters:
            values: Provide a dictionary mapping the column to apply rounding to with
                the rounding value as follows `{column: value}`.

        Returns:
            A table with rounding applied to the specified columns.

        Example:

        ```python
        >>> import pykx as kx
        >>> N = 5
        >>> kx.random.seed(42)
        >>> tab = kx.Table(data = {
        ...     'x': kx.random.random(N, 100.0),
        ...     'y': kx.random.random(N, 10.0)})
        >>> tab
        pykx.Table(pykx.q('
        x        y
        -----------------
        77.42128 8.200469
        70.49724 9.857311
        52.12126 4.629496
        99.96985 8.518719
        1.196618 9.572477
        '))
        >>> tab.xbar({'x': 10})
        pykx.Table(pykx.q('
        x  y
        -----------
        70 8.200469
        70 9.857311
        50 4.629496
        90 8.518719
        0  9.572477
        '))
        >>> tab.xbar({'x': 10, 'y': 2})
        pykx.Table(pykx.q('
        x  y
        ----
        70 8
        70 8
        50 4
        90 8
        0  8
        '))
        ```
        """
        return q("{if[11h<>type key y;"
                 " '\"Column(s) supplied must convert to type pykx.SymbolAtom\"];"
                 " ![x;();0b;key[y]!{(xbar;x;y)}'[value y;key y]]}", self, values)

    def window_join(self, table, windows, cols, aggs):
        """
        Window joins provide the ability to analyse the behaviour of data
        in one table in the neighborhood of another.

        Parameters:
            table: A `pykx.Table` or Python table equivalent containing a `['sym' and 'time']`
                column (or equivalent) with a `parted` attribute on `'sym'`.
            windows: A pair of lists containing times/timestamps denoting the beginning and
                end of the windows
            cols: The names of the common columns `['sym' and 'time']` within each table
            aggs: A dictionary mapping the name of a new derived column to a list
                specifying the function to be applied as the first element and the columns
                which should be passed from the `table` to this function. These are mapped
                {'new_col0': [f0, 'c0'], 'new_col1': [f1, 'c0', 'c1']}.

        Returns:
            For each record of the original table, a record with additional columns
                denoted by the `new_col0` entries in the `aggs` argument are added which is
                the result of applying the function `f0` with the content of column `c0` over
                the matching intervals in the `table`.

        Example:

        ```python
        >>> trades = kx.Table(data={
        ...     'sym': ['ibm', 'ibm', 'ibm'],
        ...     'time': kx.q('10:01:01 10:01:04 10:01:08'),
        ...     'price': [100, 101, 105]})
        >>> quotes = kx.Table(data={
        ...     'sym': 'ibm',
        ...     'time': kx.q('10:01:01+til 9'),
        ...     'ask': [101, 103, 103, 104, 104, 107, 108, 107, 108],
        ...     'bid': [98, 99, 102, 103, 103, 104, 106, 106, 107]})
        >>> windows = kx.q('{-2 1+\\:x}', trades['time'])
        >>> trades.window_join(quotes,
        ...                    windows,
        ...                    ['sym', 'time'],
        ...                    {'ask_max': [lambda x: max(x), 'ask'],
        ...                     'ask_minus_bid': [lambda x, y: x - y, 'ask', 'bid']})
        pykx.Table(pykx.q('
        sym time     price ask_minus_bid ask_max
        ----------------------------------------
        ibm 10:01:01 100   3 4           103
        ibm 10:01:04 101   4 1 1 1       104
        ibm 10:01:08 105   3 2 1 1       108
        '))
        ```
        """
        return q("{[t;q;w;c;a]"
                 "(cols[t], key a) xcol wj[w; c; t;enlist[q],value a]}",
                 self, table, windows, cols, aggs)

    def _repr_html_(self):
        if not licensed:
            return self.__repr__()
        console = q.system.console_size.py()
        if detect_bad_columns(self):
            return self.__repr__()
        qtab = q('.pykx.util.html.memsplay', console, self)
        df = pd.read_json(StringIO(qtab.py().decode("utf-8")), orient='records',
                          convert_dates=False, dtype=False)
        if len(df) == 0:
            columns = self.columns.py()
            if len(columns)>console[1]:
                columns = (lambda x: x[:console[1]-1] + ['...'] + [x[-1]])(columns)
            df = pd.DataFrame(columns=columns)
        else:
            df.set_index(['pykxTableIndex'], inplace=True)
            df.index.names = ['']
        ht = CharVector(df.to_html())
        ht = q('.pykx.util.html.rowcols', console, self, ht).py().decode("utf-8")
        return ht


class SplayedTable(Table):
    """Wrapper for q splayed tables."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._splay_dir = _wrappers.k_from_addr(SymbolAtom, self._values._addr, True)
        self._values = None

    def __getitem__(self, key):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def __reduce__(self):
        raise TypeError('Unable to serialize pykx.SplayedTable objects')

    def any(self):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def all(self):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def items(self):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def values(self):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    @property
    def flip(self):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def pd(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def _repr_html_(self):
        if not licensed:
            return self.__repr__()
        console = q.system.console_size.py()
        if detect_bad_columns(self):
            return self.__repr__()
        qtab = q('.pykx.util.html.memsplay', console, self)
        df = pd.read_json(StringIO(qtab.py().decode("utf-8")), orient='records',
                          convert_dates=False, dtype=False)
        if len(df) == 0:
            columns = self.columns.py()
            if len(columns)>console[1]:
                columns = (lambda x: x[:console[1]-1] + ['...'] + [x[-1]])(columns)
            df = pd.DataFrame(columns=columns)
        else:
            df.set_index(['pykxTableIndex'], inplace=True)
            df.index.names = ['']
        ht = CharVector(df.to_html())
        ht = q('.pykx.util.html.rowcols', console, self, ht).py().decode("utf-8")
        return ht

    def add_prefix(self, prefix, axis=0):
        raise AttributeError("Operation 'add_prefix' not supported for SplayedTable type")

    def add_suffix(self, suffix, axis=0):
        raise AttributeError("Operation 'add_suffix' not supported for SplayedTable type")

    def agg(self, func, axis=0, *args, **kwargs):
        raise AttributeError("Operation 'agg' not supported for SplayedTable type")

    def apply(self, func, *args, axis: int = 0, raw=None, result_type=None, **kwargs):
        raise AttributeError("Operation 'apply' not supported for SplayedTable type")

    def cast(self, ktype):
        raise AttributeError("Operation 'cast' not supported for SplayedTable type")

    def count(self, axis=0, numeric_only=False):
        raise AttributeError("Operation 'count' not supported for SplayedTable type")

    def drop_duplicates(self, subset=None, keep='first', inplace=False, ignore_index=False):
        raise AttributeError("Operation 'drop_duplicates' not supported for SplayedTable type")

    def exec(self, columns=None, where=None, by=None):
        raise AttributeError("Operation 'exec' not supported for SplayedTable type")

    def groupby(self,
                by=None,
                axis=0,
                level=None,
                as_index=True,
                sort=True,
                group_keys=True,
                observed=False,
                dropna=True):
        raise AttributeError("Operation 'groupby' not supported for SplayedTable type")

    def grouped(self, cols: Union[List, str] = ''):
        raise AttributeError("Operation 'grouped' not supported for SplayedTable type")

    def has_infs(self):
        raise AttributeError("Operation 'has_infs' not supported for SplayedTable type")

    def has_nulls(self):
        raise AttributeError("Operation 'has_nulls' not supported for SplayedTable type")

    def merge(self,
              right,
              how='inner',
              on=None,
              left_on=None,
              right_on=None,
              left_index=False,
              right_index=False,
              sort=False,
              suffixes=('_x', '_y'),
              copy=True,
              validate=None,
              q_join=False):
        raise AttributeError("Operation 'merge' not supported for SplayedTable type")

    def merge_asof(self,
                   right,
                   on=None,
                   left_on=None,
                   right_on=None,
                   left_index=False,
                   right_index=False,
                   by=None,
                   left_by=None,
                   right_by=None,
                   suffixes=('_x', '_y'),
                   tolerance=None,
                   allow_exact_matches=True,
                   direction='backward'):
        raise AttributeError("Operation 'merge_asof' not supported for SplayedTable type")

    def prototype(self):
        raise AttributeError("Operation 'prototype' not supported for SplayedTable type")

    def ungroup(self):
        raise AttributeError("Operation 'ungroup' not supported for SplayedTable type")

    def upsert(self,
               row: Union[list, List],
               match_schema: bool = False,
               test_insert: bool = False,
               inplace: bool = True
              ):
        raise AttributeError("Operation 'upsert' not supported for SplayedTable type")

    def window_join(self, table, windows, cols, aggs):
        raise AttributeError("Operation 'window_join' not supported for SplayedTable type")


class PartitionedTable(SplayedTable):
    """Wrapper for q partitioned tables."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def __reduce__(self):
        raise TypeError('Unable to serialize pykx.PartitionedTable objects')

    def items(self):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def values(self):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    @property
    def flip(self):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def pd(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def _repr_html_(self):
        if not licensed:
            return self.__repr__()
        console = q.system.console_size.py()
        if detect_bad_columns(self):
            return self.__repr__()
        qtab = q('''{[c;t]
                ps:sums .Q.cn t;
                n:last ps;
                cls:{x!x}$[c[1]<count cls:cols t;((c[1]-1)sublist cls),last cls;cls];
                if[c[0]>=n;
                  :.j.j
                    .pykx.util.html.extendcols[c 1;count cols t;]
                    .pykx.util.html.stringify
                    .pykx.util.html.addindex[-1+n;]
                    ?[t;();0b;cls]
                  ];
                r:.Q.ind[t;distinct til[c 0],-1+n];
                .j.j
                  .pykx.util.html.extendcols[c 1;count cols t]
                  .pykx.util.html.extendrows[0;1]
                  .pykx.util.html.stringify
                  .pykx.util.html.addindex[-1+n;r]
                }''', console, self)
        df = pd.read_json(StringIO(qtab.py().decode("utf-8")), orient='records',
                          convert_dates=False, dtype=False)
        if len(df) == 0:
            columns = self.columns.py()
            if len(columns)>console[1]:
                columns = (lambda x: x[:console[1]-1] + ['...'] + [x[-1]])(columns)
            df = pd.DataFrame(columns=columns)
        else:
            df.set_index(['pykxTableIndex'], inplace=True)
            df.index.names = ['']
        ht = CharVector(df.to_html())
        ht = q('.pykx.util.html.rowcols', console, self, ht).py().decode("utf-8")
        return ht

    def astype(self, dtype, copy=True, errors='raise'):
        raise AttributeError("Operation 'astype' not supported for PartitionedTable type")

    def delete(self, columns=None, where=None, inplace=False):
        raise AttributeError("Operation 'delete' not supported for PartitionedTable type")

    def drop(self, labels=None, axis=0, index=None, columns=None,  # noqa: C901
             level=None, inplace=False, errors='raise'):
        raise AttributeError("Operation 'drop' not supported for PartitionedTable type")

    def get(self, key, default=None):
        raise AttributeError("Operation 'get' not supported for PartitionedTable type")

    def head(self, n: int = 5):
        raise AttributeError("Operation 'head' not supported for PartitionedTable type")

    def iloc(self):
        raise AttributeError("Operation 'iloc' not supported for PartitionedTable type")

    def loc(self):
        raise AttributeError("Operation 'loc' not supported for PartitionedTable type")

    def mode(self, axis: int = 0, numeric_only: bool = False, dropna: bool = True):
        raise AttributeError("Operation 'mode' not supported for PartitionedTable type")

    def nlargest(self, n, columns=None, keep='first'):
        raise AttributeError("Operation 'nlargest' not supported for PartitionedTable type")

    def nsmallest(self, n, columns=None, keep='first'):
        raise AttributeError("Operation 'nsmallest' not supported for PartitionedTable type")

    def sort_values(self, by=None, ascending=True):
        raise AttributeError("Operation 'sort_values' not supported for PartitionedTable type")

    def prod(self, axis=0, skipna=True, numeric_only=False, min_count=0):
        raise AttributeError("Operation 'prod' not supported for PartitionedTable type")

    def sample(self, n=None, frac=None, replace=False, weights=None,
               random_state=None, axis=None, ignore_index=False):
        raise AttributeError("Operation 'sample' not supported for PartitionedTable type")

    def select_dtypes(self, include=None, exclude=None):
        raise AttributeError("Operation 'select_dtypes' not supported for PartitionedTable type")

    def sorted(self, cols: Union[List, str] = ''):
        raise AttributeError("Operation 'sorted' not supported for PartitionedTable type")

    def sum(self, axis=0, skipna=True, numeric_only=False, min_count=0):
        raise AttributeError("Operation 'sum' not supported for PartitionedTable type")

    def std(self, axis: int = 0, ddof: int = 1, numeric_only: bool = False):
        raise AttributeError("Operation 'std' not supported for PartitionedTable type")

    def tail(self, n: int = 5):
        raise AttributeError("Operation 'tail' not supported for PartitionedTable type")

    def unique(self, cols: Union[List, str] = ''):
        raise AttributeError("Operation 'unique' not supported for PartitionedTable type")

    def xbar(self, values):
        raise AttributeError("Operation 'xbar' not supported for PartitionedTable type")


class Dictionary(Mapping):
    """Wrapper for q dictionaries, including regular dictionaries, and keyed tables."""
    t = 99

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        keys = self._keys.py(raw=raw, has_nulls=has_nulls, stdlib=stdlib)
        vals = self._values.py(raw=raw, has_nulls=has_nulls, stdlib=stdlib)
        if type(vals) is dict:
            d = {}
            for i, k in enumerate(keys):
                d[k] = {}
                for v in vals.keys():
                    d[k][v] = vals[v][i]
            return d
        return dict(zip(keys, vals))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _wrappers.dictionary_init(self)

    def __getitem__(self, key):
        if not isinstance(key, K):
            original_key = key
            key = K(key)
            _check_k_mapping_key(original_key, key, self._keys)
        return super().__getitem__(key)

    def __setitem__(self, key, val):
        if isinstance(key, tuple):
            raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented, .pykx.Dictionary objects do not support tuple key assignment") # noqa: E501
        self.__dict__.update(q('{x,((),y)!((),z)}', self, key, val).__dict__)

    def _repr_html_(self):
        if not licensed:
            return self.__repr__()
        if 0 == len(self.keys()):
            return '<p>Empty pykx.Dictionary: ' + q('.Q.s1', self).py().decode("utf-8") + '</p>'
        console = q.system.console_size.py()
        qtab = q('''{[c;d]
                 $[98h=type value d;
                 t:([] pykxDictionaryKeys:key d),'value d;
                 t:([] pykxDictionaryKeys:key d;pykxDictionaryValues:value d)];
                 cls:$[c[1]<count cls:cols t;((c[1]-1)sublist cls),last cls;cls];
                 n:count t;
                 h:.j.j $[c[1]<count cols t;
                   {[t]c:count cls:cols t;#[(c-1) sublist cls;t],'
                   (flip(enlist `$"...")!enlist count[t]#enlist "..."),'#[-1 sublist cls;t]};(::)]
                   $[c[0]<n;{(-1 _ x),(enlist {"..."} each flip 0#x)};::]
                   .pykx.util.html.stringify ?[t;enlist(<;`i;c[0]);0b;{x!x}cls];
                 h
                 }''', console, self)
        df = pd.read_json(StringIO(qtab.py().decode("utf-8")), orient='records',
                          convert_dates=False, dtype=False)
        df.set_index('pykxDictionaryKeys', inplace=True)
        if df.columns.values[0] == 'pykxDictionaryValues':
            df.columns.values[0] = ''
        df.index.names = ['']
        ht = CharVector(df.to_html())
        ht = q('''{[c;t;h]
               $[c[0]<n:count t;
                 h,"\n<p>",{reverse "," sv 3 cut reverse string x}[n]," keys</p>";h]
               }''', console, self, ht).py().decode("utf-8")
        return ht


def _idx_col_name_generator():
    name = 'idx'
    yield name
    yield 'index'
    i = 0
    while True:
        i += 1
        yield name + str(i)


class KeyedTable(Dictionary, PandasAPI):
    """Wrapper for q keyed tables, which are a kind of table-like dictionary.

    [Refer to chapter 8.4 of Q for
    Mortals](https://code.kx.com/q4m3/8_Tables/#84-primary-keys-and-keyed-tables) for more
    information about q keyed tables.
    """
    def __new__(cls, *args, cast: bool = None, **kwargs):
        tab = None
        if 'data' in kwargs.keys():
            idx_col = list(range(len(kwargs['data'][list(kwargs['data'].keys())[0]])))
            if 'index' in kwargs.keys():
                idx_col = kwargs['index']
            idx_name_gen = _idx_col_name_generator()
            idx_col_name = next(idx_name_gen)
            while idx_col_name in kwargs['data'].keys():
                idx_col_name = next(idx_name_gen)
            idx_tab = toq(
                q.flip(
                    Dictionary(
                        {
                            idx_col_name: idx_col
                        }
                    )
                ),
                ktype=Table,
                cast=cast
            )
            tab = toq(q.flip(Dictionary(kwargs['data'])), ktype=Table, cast=cast)
            return q("{1!(x,'y)}", idx_tab, tab)
        if (tab is None
            and len(args) > 0
            and (isinstance(args[0], list)
                 or issubclass(type(args[0]), Vector)
                 or isinstance(args[0], np.ndarray))
        ):
            cols = []
            if 'columns' in kwargs.keys():
                cols.extend(kwargs['columns'])
            n_cols = len(args[0][0])
            n_gen = _col_name_generator()
            tab_rows = args[0]
            while len(cols) < n_cols:
                name = next(n_gen)
                if name not in cols:
                    cols.append(name)
            tab = q('{x !/: y}', SymbolVector(cols), tab_rows)
            idx_col = list(range(len(tab_rows)))
            if 'index' in kwargs.keys():
                idx_col = kwargs['index']
            idx_name_gen = _idx_col_name_generator()
            idx_col_name = next(idx_name_gen)
            while idx_col_name in cols:
                idx_col_name = next(idx_name_gen)
            idx_tab = toq(
                q.flip(Dictionary({idx_col_name: idx_col})),
                ktype=Table,
                cast=cast
            )
            return q("{1!(x,'y)}", idx_tab, tab)
        # TODO: 'strict' and 'cast' flags
        return toq(*args, ktype=None if cls is K else cls, cast=cast)

    def __init__(self, *args, **kwargs):
        if ('data' not in kwargs.keys()
            and not (len(args) > 0
                     and (isinstance(args[0], list)
                          or issubclass(type(args[0]), Vector)
                          or isinstance(args[0], np.ndarray)))
        ):
            super().__init__(*args, **kwargs)

    def _compare(self, other, op_str, skip=False):
        vec = self
        if not skip:
            vec = q('{x 0}', q('value', q('flip', q('value', self))))
        try:
            r = q(op_str, vec, other)
        except Exception as ex:
            ex_str = str(ex)
            if ex_str.startswith('length') or ex_str.startswith('type'):
                return q('0b')
            elif ex_str.startswith('nyi'):
                return self._compare(other, op_str, skip=True)
            raise
        else:
            if hasattr(r, '__len__') and len(r) == 0:
                # Handle comparisons of empty objects
                if op_str == '=':
                    return q('~', vec, other)
                elif op_str == '{not x=y}':
                    return q('{not x~y}', vec, other)
            return r

    @property
    def has_nulls(self) -> bool:
        return any(x.is_null if x.is_atom else x.has_nulls for x in self._values._values)

    def ungroup(self):
        return q.ungroup(self)

    def any(self) -> bool:
        return any(x.any() for x in self._values._values)

    def all(self) -> bool:
        return all(x.all() for x in self._values._values)

    def get(self, key, default=None):
        return q('{0!x}', self).get(key, default=default)

    def __getitem__(self, key):
        res = self.loc[key]
        if isinstance(res, List) and len(res) == 1:
            res = q('{raze x}', res)
        return res

    def __setitem__(self, key, val):
        self.loc[key] = val

    def __iter__(self):
        for x in zip(*self._values._values):
            if len(x)==1:
                yield list(x).pop(0)
            else:
                yield list(x)

    def keys(self):
        return self._keys

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        if licensed:
            return q('0!', self).np(raw=raw, has_nulls=has_nulls)
        raise LicenseException('convert a keyed table to a Numpy representation')

    def pd(
        self,
        *,
        raw: bool = False,
        has_nulls: Optional[bool] = None,
        as_arrow: Optional[bool] = False,
    ):
        kk = self._keys._keys
        vk = self._values._keys
        kvg = self._keys._values._unlicensed_getitem
        vvg = self._values._values._unlicensed_getitem
        if len(self) == 0:
            df = pd.DataFrame(columns=kk.py() + vk.py())
            df = df.set_index(kk.py())
            if as_arrow:
                if not pandas_2:
                    raise RuntimeError('Pandas Version must be at least 2.0 to use as_arrow=True')
                if pa is None:
                    raise PyArrowUnavailable # nocov
                df = df.convert_dtypes(dtype_backend='pyarrow')
            return df
        idx = [kvg(i).np(raw=raw, has_nulls=has_nulls).reshape(-1)
               for i in range(len(kk))]
        cols = [vvg(i).np(raw=raw, has_nulls=has_nulls)
                for i in range(len(vk))]
        column_names = pd.Index(kk.py() + vk.py())
        columns = idx + cols
        index = pd.Index(np.arange(len(self)))
        df = df_from_arrays(column_names, columns, index)

        _pykx_base_types = {}
        for i, col in enumerate(kk.py()):
            if not raw and isinstance(kvg(i), EnumVector):
                df[col] = df[col].astype('category')
            _pykx_base_types[col] = str(type(kvg(i)).__name__)
        for i, col in enumerate(vk.py()):
            if not raw and isinstance(vvg(i), EnumVector):
                df[col] = df[col].astype('category')
            _pykx_base_types[col] = str(type(vvg(i)).__name__)
        if as_arrow:
            if not pandas_2:
                raise RuntimeError('Pandas Version must be at least 2.0 to use as_arrow=True')
            if pa is None:
                raise PyArrowUnavailable # nocov
            if raw:
                t_dict = dict(filter(lambda i: i[1] != 'GUIDVector', _pykx_base_types.items()))
                df = df.astype(dict([(k, _as_arrow_raw_map[v])
                                    for k, v in t_dict.items()]))
            else:
                df = df.astype(dict([(k, _as_arrow_map[v]) for k, v in _pykx_base_types.items()]))
        df.set_index(kk.py(), inplace=True)
        df.attrs['_PyKX_base_types'] = _pykx_base_types
        return df

    def pa(self):
        if pa is None:
            raise PyArrowUnavailable # nocov
        raise NotImplementedError(f"{__class__.__name__}.{inspect.stack()[0][3]}() is not implemented.") # noqa: E501

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        vkp = self._values._keys.py()
        vvp = self._values._values.py
        return dict(zip(
            zip(*self._keys._values.py(raw=raw, has_nulls=has_nulls, stdlib=stdlib)),
            (dict(zip(vkp, x)) for x in zip(*vvp(raw=raw, has_nulls=has_nulls, stdlib=stdlib))),
        ))

    def insert(
        self,
        row: Union[list, List],
        match_schema: bool = False,
        test_insert: bool = False,
        inplace: bool = True
    ):
        """Helper function around `q`'s `insert` function which inserts a row or multiple rows into
        a q Table object.

        Parameters:
            row: A list of objects to be inserted as a row.
            match_schema: Whether the row/rows to be inserted must match the tables current schema.
            test_insert: Causes the function to modify a small local copy of the table and return
                the modified example, this can only be used with embedded q and will not modify the
                source tables contents.
            inplace: Causes the underlying Table python object to update itself with the
                resulting Table after the insert.

        Returns:
            The resulting table after the given row has been inserted.

        Raises:
            PyKXException: If the `match_schema` parameter is used this function may raise an error
                if the row to be inserted does not match the tables schema. The error message will
                contain information about which columns did not match.

        Examples:

        Insert a single row onto a Table, ensuring the new row matches the current tables schema.

        ```Python
        >>> tab.insert([1, 2.0, datetime.datetime(2020, 2, 24)], match_schema=True)
        ```
        """
        q['.pykx.i.itab'] = self
        q.insert('.pykx.i.itab', row, match_schema, test_insert)
        res = q('.pykx.i.itab')
        if inplace:
            self.__dict__.update(res.__dict__)
        q('delete itab from `.pykx.i')
        return res

    def upsert(
        self,
        row: Union[list, List],
        match_schema: bool = False,
        test_insert: bool = False,
        inplace: bool = True
    ):
        """Helper function around `q`'s `upsert` function which inserts a row or multiple rows into
        a q Table object.

        Parameters:
            row: A list of objects to be inserted as a row, if the table is within embedded q you
                may also pass in a table object to be upserted.
            match_schema: Whether the row/rows to be inserted must match the tables current schema.
            test_insert: Causes the function to modify a small local copy of the table and return
                the modified example, this can only be used with embedded q and will not modify the
                source tables contents.
            inplace: Causes the underlying Table python object to update itself with the
                resulting Table after the insert.

        Returns:
            The resulting table after the given row has been upserted.

        Raises:
            PyKXException: If the `match_schema` parameter is used this function may raise an error
                if the row to be inserted does not match the tables schema. The error message will
                contain information about which columns did not match.

        Examples:

        Upsert a single row onto a Table, ensuring the new row matches the current tables schema.

        ```Python
        >>> tab.upsert([1, 2.0, datetime.datetime(2020, 2, 24)], match_schema=True)
        ```
        """
        res = q.upsert(self, row, match_schema, test_insert)
        if inplace:
            self.__dict__.update(res.__dict__)
        return res

    def sorted(self, cols: Union[List, str] = ''):
        try:
            if len(cols) == 0:
                cols = q.cols(self)[0]
            if not (isinstance(cols, List) or isinstance(cols, list)):
                cols = [cols]
            for col in cols:
                res = q.qsql.update(self, {col: f'`s#{col}'})
                self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 's-type' in err_str:
                raise QError('Items are not in ascending order')
            elif 'type' in err_str:
                raise QError('Object does not support the sorted attribute')
            else:
                raise e

    def unique(self, cols: Union[List, str] = ''):
        try:
            if len(cols) == 0:
                cols = q.cols(self)[0]
            if not (isinstance(cols, List) or isinstance(cols, list)):
                cols = [cols]
            for col in cols:
                res = q.qsql.update(self, {col: f'`u#{col}'})
                self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 'u-type' in err_str:
                raise QError('Items are not unique')
            elif 'type' in err_str:
                raise QError('Object does not support the unique attribute')
            else:
                raise e

    def parted(self, cols: Union[List, str] = ''):
        try:
            if len(cols) == 0:
                cols = q.cols(self)[0]
            if not (isinstance(cols, List) or isinstance(cols, list)):
                cols = [cols]
            for col in cols:
                res = q.qsql.update(self, {col: f'`p#{col}'})
                self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 'u-type' in err_str:
                raise QError('Items are not parted')
            elif 'type' in err_str:
                raise QError('Object does not support the parted attribute')
            else:
                raise e

    def grouped(self, cols: Union[List, str] = ''):
        try:
            if len(cols) == 0:
                cols = q.cols(self)[0]
            if not (isinstance(cols, List) or isinstance(cols, list)):
                cols = [cols]
            for col in cols:
                res = q.qsql.update(self, {col: f'`g#{col}'})
                self.__dict__.update(res.__dict__)
            return self
        except BaseException as e:
            err_str = str(e)
            if 'type' in err_str:
                raise QError('Object does not support the grouped attribute')
            else:
                raise e

    def sql(self, query, *args):
        """Execute an SQL query against the supplied PyKX KeyedTable object.

        This function expects the keyed table object to be supplied as a parameter
        to the query, additional parameters can be supplied as positional
        arguments.

        Parameters:
            query: A str object indicating the query to be executed, this
                must contain a required argument $1 associated with the
                table being queried.
            *args: Any additional positional arguments required for query
                execution.

        Returns:
            The queried table associated with the SQL statement

        Examples:

        Query a keyed table supplying no additional arguments

        ```python
        >>> tab = kx.Table(
        ...     data = {'x': [1, 2, 3], 'x1': ['a', 'b', 'a']}
        ...     ).set_index('x')
        >>> tab.sql("select * from $1 where x1='a'")
        pykx.Table(pykx.q('
        x x1
        ----
        1 a
        3 a
        '))
        ```

        Query a keyed table supplying multiple arguments

        ```python
        >>> tab = kx.Table(
        ...     data = {'x': [1, 2, 3], 'x1': ['a', 'b', 'a']}
        ...     ).set_index('x')
        >>> tab.sql("select * from $1 where x1=$2 and x=$3", 'a', 1)
        pykx.Table(pykx.q('
        x x1
        ----
        1 a
        '))
        ```
        """
        if not isinstance(query, str):
            raise TypeError('Supplied query is not of type "str"')
        if '$1' not in query:
            raise QError('Supplied query does not contain argument $1')
        return q.sql(query, self, *args)

    def select(self, columns=None, where=None, by=None, inplace=False):
        """Apply a q style select statement on the supplied keyed table defined within the process.

        This implementation follows the q functional select syntax with limitations on
        structures supported for the various clauses a result of this.

        Parameters:
            columns: A dictionary mapping the name to be given to a column and the logic to be
                applied in aggregation to that column both as strings.
            where: Conditional filtering used to select subsets of the data on which by-clauses and
                appropriate aggregations are to be applied.
            by: A dictionary mapping the names to be assigned to the produced columns and the
                columns whose results are used to construct the groups of the by clause.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects.
                See [here](https://code.kx.com/q/basics/qsql/#result-and-side-effects).

        Examples:

        Define a q table in python, and give it a name in q memory

        ```python
        >>> import pykx as kx
        >>> qtab = kx.Table(data = {
        ...     'col1': kx.random.random(100, ['a', 'b', 'c']),
        ...     'col2': kx.random.random(100, 1.0),
        ...     'col3': kx.random.random(100, False),
        ...     'col4': kx.random.random(100, 10.0)}
        ...     ).set_index('col1')
        ```

        Select all items in the table

        ```python
        >>> qtab.select()
        ```

        Filter table based on various where conditions

        ```python
        >>> qtab.select(where='col2<0.5')
        ```

        Retrieve statistics by grouping data on symbol columns

        ```python
        >>> qtab.select(columns={'maxCol2': 'max col2'}, by={'col1': 'col1'})
        >>> qtab.select(columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'})
        ```

        Retrieve grouped statistics with restrictive where condition

        ```python
        >>> qtab.select(columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'}, where='col3=0b')
        ```
        """ # noqa: E501
        return q.qsql.select(self, columns, where, by, inplace)

    def exec(self, columns=None, where=None, by=None):
        """
        Apply a q style exec statement on the supplied PyKX KeyedTable.

        This implementation follows the q functional exec syntax with limitations on structures
        supported for the various clauses a result of this.

        Parameters:
            columns: A dictionary mapping the name to be given to a column and the logic to be
                applied in aggregation to that column both as strings. A string defining a single
                column to be retrieved from the table as a list.
            where: Conditional filtering used to select subsets of the data on which by clauses and
                appropriate aggregations are to be applied.
            by: A dictionary mapping the names to be assigned to the produced columns and the
                the columns whose results are used to construct the groups of the by clause.

        Examples:

        Define a PyKX KeyedTable

        ```python
        >>> qtab = kx.Table(data = {
        ...     'col1': kx.random.random(100, ['a', 'b', 'c']),
        ...     'col2': kx.random.random(100, 1.0),
        ...     'col3': kx.random.random(100, False),
        ...     'col4': kx.random.random(100, 10.0)}
        ...     ).set_index('col1')
        ```

        Select last item of the table

        ```python
        qtab.exec()
        ```

        Retrieve a column from the table as a list

        ```python
        qtab.exec('col3')
        ```

        Retrieve a set of columns from a table as a dictionary

        ```python
        qtab.exec({'symcol': 'col1'})
        qtab.exec({'symcol': 'col1', 'boolcol': 'col3'})
        ```

        Filter columns from a table based on various where conditions

        ```python
        qtab.exec('col3', where='col1=`a')
        qtab.exec({'symcol': 'col1', 'maxcol4': 'max col4'}, where=['col1=`a', 'col2<0.3'])
        ```

        Retrieve data grouping by data on symbol columns

        ```python
        qtab.exec('col2', by={'col1': 'col1'})
        qtab.exec(columns={'maxCol2': 'max col2'}, by={'col1': 'col1'})
        qtab.exec(columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'})
        ```

        Retrieve grouped statistics with restrictive where condition

        ```python
        qtab.exec(columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'}, where='col3=0b')
        ```
        """ # noqa: E501
        return q.qsql.exec(self, columns, where, by)

    def update(self, columns=None, where=None, by=None, inplace=False):
        """
        Apply a q style update statement on a PyKX KeyedTable.

        This implementation follows the q functional update syntax with limitations on
        structures supported for the various clauses a result of this.

        Parameters:
            columns: A dictionary mapping the name of a column present in the table or one to be
                added to the contents which are to be added to the column, this content can be a
                string denoting q data or the equivalent Python data.
            where: Conditional filtering used to select subsets of the data on which by-clauses and
                appropriate aggregations are to be applied.
            by: A dictionary mapping the names to be assigned to the produced columns and the
                columns whose results are used to construct the groups of the by clause.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects.
                See [here](https://code.kx.com/q/basics/qsql/#result-and-side-effects).

        Examples:

        Define a q table in python and named in q memory

        ```python
        >>> qtab = kx.Table(data={
        ...         'name': ['tom', 'dick', 'harry'],
        ...         'age': [28, 29, 35],
        ...         'hair': ['fair', 'dark', 'fair'],
        ...         'eye': ['green', 'brown', 'gray']}
        ...     )
        ```

        Update all the contents of a column

        ```python
        >>> qtab.update({'eye': '`blue`brown`green'})
        ```

        Update the content of a column restricting scope using a where clause

        ```python
        >>> qtab.update({'eye': ['blue']}, where='hair=`fair')
        ```

        Define a q table suitable for by clause example

        ```python
        >>> bytab = kx.Table(data={
        ...         'name': kx.random.random(100, ['nut', 'bolt', 'screw']),
        ...         'color': kx.random.random(100, ['red', 'green', 'blue']),
        ...         'weight': 0.5 * kx.random.random(100, 20),
        ...         'city': kx.random.random(100, ['london', 'paris', 'rome'])}
        ...         ).set_index('city')
        ```

        Apply an update grouping based on a by phrase

        ```python
        >>> bytab.update({'weight': 'avg weight'}, by={'city': 'city'})
        ```

        Apply an update grouping based on a by phrase and persist the result using the inplace keyword

        ```python
        >>> bytab.update(columns={'weight': 'avg weight'}, by={'city': 'city'}, inplace=True)
        ```
        """ # noqa: E501
        return q.qsql.update(self, columns, where, by, inplace)

    def delete(self, columns=None, where=None, inplace=False):
        """
        Apply a q style delete statement a PyKX keyed table defined.

        This implementation follows the q functional delete syntax with limitations on
        structures supported for the various clauses a result of this.

        Parameters:
            columns: Denotes the columns to be deleted from a table.
            where: Conditional filtering used to select subsets of the data which are to be
                deleted from the table.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects.
                See [here](https://code.kx.com/q/basics/qsql/#result-and-side-effects).

        Examples:

        Define a PyKX Table against which to run the examples

        ```python
        >>> qtab = kx.Table(data = {
        ...             'name': ['tom', 'dick', 'harry'],
        ...             'age': [28, 29, 35],
        ...             'hair': ['fair', 'dark', 'fair'],
        ...             'eye': ['green', 'brown', 'gray']}
        ...             ).set_index('name')
        ```

        Delete all the contents of the table

        ```python
        >>> qtab.delete()
        ```

        Delete single and multiple columns from the table

        ```python
        >>> qtab.delete('age')
        >>> qtab.delete(['age', 'eye'])
        ```

        Delete rows of the dataset based on where condition

        ```python
        >>> qtab.delete(where='hair=`fair')
        >>> qtab.delete(where=['hair=`fair', 'age=28'])
        ```

        Delete a column from the dataset named in q memory and persist the result using the
        inplace keyword

        ```python
        >>> qtab.delete('age', inplace=True)
        ```
        """ # noqa: E501
        return q.qsql.delete(self, columns, where, inplace)

    def _repr_html_(self):
        if not licensed:
            return self.__repr__()
        keys = q('{cols key x}', self).py()
        console = q.system.console_size.py()
        if detect_bad_columns(q('0!', self)):
            return self.__repr__()
        qtab=q('''{[c;t]
               n:count t;
               cls:{x!x}$[c[1]<ct:count cls:cols t;((c[1]-1)sublist cls),last cls;cls];
               .j.j
                 .pykx.util.html.extendcols[c[1];ct;]
                 .pykx.util.html.extendrows[c[0];n;]
                 .pykx.util.html.stringify
                 .pykx.util.html.filteridxs[cls;c 0;n;t]
               }''', console, self)
        df = pd.read_json(StringIO(qtab.py().decode("utf-8")), orient='records',
                          convert_dates=False, dtype=False)
        columns = q('cols', self).py()
        if len(keys)>=console[1]:
            keys = keys[:console[1]-1]
        if len(df) == 0:
            if len(columns)>console[1]:
                columns = (lambda x: x[:console[1]-1] + ['...'] + [x[-1]])(columns)
            df = pd.DataFrame(columns=columns)
        df.set_index(keys, inplace=True)
        ht = CharVector(df.to_html())
        ht = q('.pykx.util.html.rowcols', console, self, ht).py().decode("utf-8")
        return ht


class GroupbyTable(PandasAPI):
    def __init__(self, tab, as_index, was_keyed, as_vector=None):
        self.tab = tab
        self.as_index = as_index
        self.was_keyed = was_keyed
        self.as_vector = as_vector

    def q(self):
        return self.tab

    def ungroup(self):
        return q.ungroup(self.tab)

    def apply(self, func, *args, **kwargs):
        tab = self.q()
        key = q.key(tab)
        data = q.value(tab)
        return q('{[k; t; f]'
                 '  ff:flip t;'
                 '  d:value ff;'
                 '  agg:{[row; f] f each row}[;f] each d;'
                 'k!((key ff)!/:(flip agg))}',
                 key,
                 data,
                 func
        )

    def __getitem__(self, item):
        return GroupbyTable(
            self.tab[item],
            True,
            False,
            as_vector=item
        )


GTable_init(GroupbyTable)


class Function(Atom):
    """Base type for all q functions.

    `Function` objects can be called as if they were Python functions. All provided arguments will
    be converted to q using [`pykx.toq`][pykx.toq], and the execution of the function will happen
    in q.

    `...` can be used to omit an argument, resulting in a [function projection][pykx.Projection].

    [Refer to chapter 6 of Q for Mortals](https://code.kx.com/q4m3/6_Functions/) for more
    information about q functions.
    """
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)

    def __bool__(self):
        return True

    @property
    def is_null(self) -> bool:
        return False

    @cached_property
    def params(self):
        return ()

    @cached_property
    def args(self):
        return ()

    @cached_property
    def func(self):
        return self

    def __call__(self, *args, **kwargs):
        if not licensed:
            raise LicenseException('call a q function in a Python process')
        args = {i: K(x) for i, x in enumerate(args)}
        if kwargs:
            for param_name, param_value in kwargs.items():
                try:
                    i = self.params.index(param_name)
                except ValueError:
                    raise TypeError(
                        f"Function {self!r} got an unexpected keyword argument '{param_name}'"
                    ) from None
                else:
                    if i in args:
                        raise TypeError(
                            f"Function {self!r} got multiple values for parameter '{param_name}'"
                        ) from None
                args[i] = K(param_value)
        no_gil = True
        if args: # Avoid calling `max` on an empty sequence
            no_gil = not any([type(x) is Foreign for x in args.values()])
            args = {**{x: ... for x in range(max(args))}, **args}
        args = K([x for _, x in sorted(args.items())]) if len(args) else q('enlist(::)')
        return _wrappers.function_call(self, args, no_gil)

    @cached_property
    def each(self):
        return q("{x'}", self)

    @cached_property
    def over(self):
        return q('{x/}', self)

    @cached_property
    def scan(self):
        return q('{x\\}', self)

    @cached_property
    def each_prior(self):
        return q("{x':}", self)

    prior = each_prior

    @cached_property
    def each_right(self):
        return q('{x/:}', self)

    sv = each_right

    @cached_property
    def each_left(self):
        return q('{x\\:}', self)

    vs = each_left


bs4_spec = importlib.util.find_spec("bs4")
md2_spec = importlib.util.find_spec("markdown2")
if bs4_spec is not None and md2_spec is not None:
    Function.scan.__doc__ = help.qhelp('scan')


class Lambda(Function):
    """Wrapper for q lambda functions.

    Lambda's are the most basic kind of function in q. They can take between 0 and 8 parameters
    (inclusive), which all must be q objects themselves. If the provided parameters are not
    [`pykx.K`][pykx.K] objects, they will be converted into them using [`pykx.toq`][pykx.toq].

    Unlike other [`pykx.Function`][pykx.Function] subclasses, `Lambda` objects can be called with
    keyword arguments, using the names of the parameters from q.
    """
    t = 100
    _name = ''

    @property
    def __name__(self):
        return 'pykx.Lambda'

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)

    @cached_property
    def params(self):
        # Strip "PyKXParam" from all param names if it is a prefix for all
        return tuple(
            str(x) for x in q('k){x:.:x;$[min (x:x[1]) like "PyKXParam*"; `$9_\'$x; x]}', self)
        )

    def __new__(cls, x: Any, *args, cast: bool = None, **kwargs):
        if isinstance(x, str):
            x = q(x)
            if not isinstance(x, Lambda):
                raise TypeError("String passed is not in the correct lambda form")
        return toq(x, ktype=None if cls is K else cls, cast=cast)

    @property
    def string(self):
        return q.string(self)

    @property
    def value(self):
        return q.value(self)


class UnaryPrimitive(Function):
    """Wrapper for q unary primitive functions, including `::`, and other built-ins.

    Unary primitives are a class of built-in q functions which take exactly one parameter. New ones
    cannot be defined by a user through any normal means.

    See Also:
        [`pykx.Identity`][pykx.Identity]
    """
    t = 101
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)

    @property
    def __name__(self):
        return 'pykx.UnaryPrimitive'

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TypeError('Cannot use kwargs on a unary primitive')
        return super().__call__(*args, **kwargs)

    @classmethod
    def __instancecheck__(cls, instance):
        return isinstance(instance, UnaryPrimitive) or isinstance(instance, UnaryPrimative)


# TODO: Remove this when UnaryPrimative is fully deprecated
UnaryPrimative = UnaryPrimitive


class Identity(UnaryPrimitive):
    """Wrapper for the q identity function, also known as generic null (`::`).

    Most types in q have their own null value, but `::` is used as a generic/untyped null in
    contexts where a non-null q object would otherwise be, e.g. as a null value for a
    [generic list][pykx.List], or as a null value in a [dictionary][pykx.Dictionary].

    `::` is also the identity function. It takes a single q object as a parameter, which it returns
    unchanged.
    """
    def __repr__(self):
        # For clarity, represents Identity objects with ::
        return f"pykx.{type(self).__name__}(pykx.q('::'))"

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return None

    def __bool__(self):
        return False

    def __str__(self):
        return '::'

    @property
    def is_null(self) -> bool:
        return True


class ProjectionNull(Identity):
    """Wrapper for the q projection null.

    Projection null is a special q object which may initially seem to be
    [generic null (`::`)][pykx.Identity], but is actually a magic value used to create projections.

    Danger: Projection nulls are unwieldy.
        There are very few scenarios in which a typical user needs to work with a projection null
        directly, and doing so can be very error-prone. Instead of using them directly, operate
        on / work with [projections][pykx.Projection], which use them implicitly.

    When a projection null is provided as an argument to a q function, the result is a
    [function projection][pykx.Projection].
    """
    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return ...

    def __lt__(self, other):
        return K(False)

    def __gt__(self, other):
        return K(False)

    def __eq__(self, other):
        try:
            return q('104h ~ type {1b}@', other)
        except TypeError:
            return q('0b')

    def __ne__(self, other):
        return K(not self.__eq__(other))

    __ge__ = __eq__
    __le__ = __eq__


class Operator(Function):
    """Wrapper for q operator functions.

    Operators include `@`, `*`, `+`, `!`, `:`, `^`, and many more. They are documented on the q
    reference page: [here](https://code.kx.com/q/ref/#operators).
    """
    t = 102
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TypeError('Cannot use kwargs on an operator')
        return super().__call__(*args, **kwargs)

    def __new__(self, op): # noqa:B902
        if not isinstance(op, str):
            raise QError('Supplied operator must be a str')
        self.repr = op
        if licensed:
            gen_op = q(op)
            if gen_op.t != 102:
                raise QError('Generation of operator did not return correct type')
            return q(op)
        else:
            raise QError('Unsupported operation in unlicensed mode')

    def __init__(self, op):
        pass


class Iterator(Function):
    """Wrappers for q iterator functions.

    Iterators include the mapping iterators (`'`, `':`, `/:`, and `\\:`), and the accumulating
    iterators (`/`, and `\\`). They are documented on the q reference page:
    [here](https://code.kx.com/q/ref/#iterators)
    """
    t = 103

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TypeError('Cannot use kwargs on an iterator')
        return super().__call__(*args, **kwargs)


class Projection(Function):
    """Wrapper for q function projections.

    Similar to [`functools.partial`](https://docs.python.org/3/library/functools.html),
     q functions can have some of their parameters fixed in
    advance, resulting in a new function, which is a projection. When this projection is called,
    the fixed parameters are no longer required, and cannot be provided.

    If the original function had `n` parameters, and it had `m` of them provided, the result would
    be a function (projection) that has `m` parameters.

    In PyKX, the special Python singleton `...` is used to represent
    [projection null][pykx.ProjectionNull]
    """
    t = 104
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)

    @cached_property
    def params(self) -> Tuple[SymbolAtom]:
        """The param names from the base function that have not been set."""
        if not self.func.params:
            return self.func.params
        # Find the projection nulls
        xi = q('{(104h~type {1b}@) each value x}', self)[1:]
        return tuple(str(x) for i, x in enumerate(self.func.params) if i >= len(xi) or xi[i])

    @cached_property
    def args(self) -> Tuple[K]:
        """The supplied arguments to the function being projected.

        The arguments in the tuple are ordered as they were applied to the function, with
        [projection nulls][pykx.ProjectionNull] to fill the empty spaces before and in-between the
        supplied arguments. The tuple may either end with the last supplied argument, or have some
        trailing projection nulls depending on how the projection was created.

        Examples:

        ```python
        >>> f = kx.q('{x+y+z}')
        >>> f.args
        ()
        >>> f(..., 2)
        pykx.Projection(pykx.q('{x+y+z}[;2]'))
        >>> f(..., 2).args
        (pykx.ProjectionNull(pykx.q('::')), pykx.LongAtom(pykx.q('2')))
        >>> f(..., 2, ...)
        pykx.Projection(pykx.q('{x+y+z}[;2;]'))
        >>> f(..., 2, ...).args
        (pykx.ProjectionNull(pykx.q('::')), pykx.LongAtom(pykx.q('2')), pykx.ProjectionNull(pykx.q('::')))
        ```
        """ # noqa: E501
        return tuple(q('value', self)[1:])

    @cached_property
    def func(self) -> Function:
        """The original function with no fixed parameters.

        With the original function, a new projection can be created, or it can simply be called
        with every parameter set.
        """
        return q('{value[x]0}', self)

    def __call__(self, *args, **kwargs):
        if kwargs and not self.params:
            raise TypeError(
                'This projection does not support kwargs, as it is a '
                f'projection of a {type(self.func).__name__}')
        return super().__call__(*args, **kwargs)


class Composition(Function):
    """Wrapper for q function compositions.

    Functions in q can be directly composed, as opposed to creating a new lambda function that
    applies a chain of functions. Direct composition of functions lends itself to a style which is
    referred to as "point-free" or "tacit" programming.
    """
    t = 105
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)

    @property
    def __name__(self):
        return 'pykx.Composition'

    @property
    def params(self):
        if q('{.pykx.util.isw x}', self).py():
            return q('.pykx.unwrap', self).params
        return self.func.params

    @property
    def py(self):
        if q('{.pykx.util.isw x}', self).py():
            return q('{x[`]}', self).py()

    @property
    def np(self):
        if q('{.pykx.util.isw x}', self).py():
            return q('{x[`]}', self).np()

    @property
    def pd(self):
        if q('{.pykx.util.isw x}', self).py():
            return q('{x[`]}', self).pd()

    @property
    def pa(self):
        if q('{.pykx.util.isw x}', self).py():
            return q('{x[`]}', self).pa()

    @cached_property
    def args(self):
        return tuple(q("k){$[(@)~*v:. x;*|v;x]}'{,/({::;$[105h~@y;x y;y]}.z.s)@'. x }@", self))

    @cached_property
    def func(self):
        return self.args[len(self.args) - 1]

    def __call__(self, *args, **kwargs):
        if not licensed:
            raise LicenseException('call a q function in a Python process')
        if q('{.pykx.util.isw x}', self).py():
            args = {i: K(x) for i, x in enumerate(args)}
            if args: # Avoid calling `max` on an empty sequence
                args = {**{x: ... for x in range(max(args))}, **args}
            args = K([x for _, x in sorted(args.items())])
            params = [x for x in self.params if '**' not in x and '=' not in x]
            if len(args) > len(params):
                kwargs = args[len(params):]
                args = args[:len(params)]
            if isinstance(kwargs, Dictionary) or isinstance(kwargs, Table):
                kwargs = {k: v[0] for k, v in kwargs.py().items()}
            if len(args):
                if issubclass(type(args), Vector) and len(args) == 1 \
                   and issubclass(type(args[0]), Vector):
                    args = q('{raze x}', args)
                if issubclass(type(args), Table):
                    args = [args[x] for x in range(len(args))]
                else:
                    args = [K(x) for x in args]
                return K(
                    q('.pykx.unwrap', self).py()(
                        *args,
                        **{k: K(v) for k, v in kwargs.items()}
                    )
                )
            return K(q('.pykx.unwrap', self).py()(**{k: K(v) for k, v in kwargs.items()}))

        if kwargs and not self.params:
            raise TypeError(
                'This composition does not support kwargs, as the first '
                f'function applied is a {type(self.func).__name__}')
        return super().__call__(*args, **kwargs)


class AppliedIterator(Function):
    """Base type for all q functions that have had an iterator applied to them.

    Iterators, also known as adverbs, are like Python decorators, in that they are functions which
    take a function as their argument, and return a modified version of it. The iterators
    themselves are of the type [`pykx.Iterator`][pykx.Iterator], but when applied to a function a
    new type (which is a subclass of `AppliedIterator`) is created depending on what iterator was
    used.
    """
    _name = ''

    @property
    def __name__(self):
        return 'pykx.AppliedIterator'

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)

    @cached_property
    def params(self):
        return self.func.params

    @cached_property
    def args(self):
        return self.func.args

    @cached_property
    def func(self):
        return q('value', self)

    def __call__(self, *args, **kwargs):
        if kwargs and not self.params:
            raise TypeError(
                'This applied iterator does not support kwargs, as the base'
                f'function is a {type(self.func).__name__}'
            )
        return super().__call__(*args, **kwargs)


class Each(AppliedIterator):
    """Wrapper for functions with the 'each' iterator applied to them."""
    t = 106
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)


class Over(AppliedIterator):
    """Wrapper for functions with the 'over' iterator applied to them."""
    t = 107
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)


class Scan(AppliedIterator):
    """Wrapper for functions with the 'scan' iterator applied to them."""
    t = 108
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)


class EachPrior(AppliedIterator):
    """Wrapper for functions with the 'each-prior' iterator applied to them."""
    t = 109
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)


class EachRight(AppliedIterator):
    """Wrapper for functions with the 'each-right' iterator applied to them."""
    t = 110
    _name = ''

    @property
    def __doc__(self):
        if self._name != '':
            return help.qhelp(self._name)


class EachLeft(AppliedIterator):
    """Wrapper for functions with the 'each-left' iterator applied to them."""
    t = 111


class Foreign(Atom):
    """Wrapper for foreign objects, i.e. wrapped pointers to regions outside of q memory."""
    t = 112

    def __reduce__(self):
        raise TypeError('Unable to serialize pykx.Foreign objects')

    def py(self, stdlib=None, raw=None):
        """Turns the pointer stored within the Foreign back into a Python Object.

        Note: The resulting object is a reference to the same memory location as the initial object.
            This can result in unexpected behavior and it is recommended to only modify the
            original python object passed into the `Foreign`
        """
        return _wrappers.from_foreign_to_pyobject(self)

    def __call__(self, *args, **kwargs):
        if not licensed:
            raise LicenseException('call a q function in a Python process')
        if q('.pykx.util.isf', self).py():
            return q('{.pykx.wrap[x][<]}', self)(
                *[K(x) for x in args],
                **{k: K(v) for k, v in kwargs.items()}
            )
        return q('{x . y}', self, [*[K(x) for x in args]])

    @property
    def params(self):
        params = [str(x) for x in list(signature(self.py()).parameters.values())]
        params = [x.split('=')[0] if '=' in x else x for x in params]
        return tuple(params)


class SymbolicFunction(Function, SymbolAtom):
    """Special wrapper type representing a symbol atom that can be used as a function."""
    @property
    def __name__(self):
        return 'pykx.SymbolicFunction'

    def __init__(self, *args, **kwargs):
        self.execution_ctx = q # Default to executing in the embedded q instance.
        super().__init__(*args, **kwargs)

    # TODO: symbolic function projections
    def __call__(self, *args) -> K:
        return self.execution_ctx(bytes(self), *(args if args else (None,)))

    def __lt__(self, other):
        return K(False)

    def __gt__(self, other):
        return K(False)

    def __eq__(self, other):
        if not isinstance(other, SymbolicFunction):
            return K(False)
        return K(str(self.sym) == str(other.sym) and self.execution_ctx == other.execution_ctx)

    def __ne__(self, other):
        return K(not self.__eq__(other))

    __ge__ = __eq__
    __le__ = __eq__

    # SymbolAtom methods are generally prioritized, but we want the conversions to act as if it is
    # a function.
    py = Function.py
    np = Function.np
    pd = Function.pd
    pa = Function.pa

    @cached_property
    def params(self):
        return self.func.params

    @cached_property
    def args(self):
        return self.func.args

    @cached_property
    def sym(self) -> SymbolAtom:
        """The symbolic function as a plain symbol."""
        return SymbolAtom._from_addr(self._addr)

    @cached_property
    def func(self) -> Function:
        """The symbolic function as a regular function, obtained by dereferencing the symbol."""
        return self.execution_ctx(bytes(self))

    def with_execution_ctx(self, execution_ctx) -> 'SymbolicFunction':
        """Get a new symbolic function that will be evaluated within the provided q instance."""
        x = SymbolicFunction._from_addr(self._addr)
        x.execution_ctx = execution_ctx
        return x


class ParseTree:
    """Special wrapper for a list which will be treated as a ParseTree.
    For use with the Query API
    """
    def __init__(self, tree):
        if isinstance(tree, str):
            tree = q.parse(CharVector(tree))
        elif isinstance(tree, Column):
            tree = tree._value
        elif isinstance(tree, QueryPhrase):
            tree = tree._phrase
        elif isinstance(tree, Variable):
            tree = tree._name
        self._tree = tree

    def __repr__(self):
        preamble = f'pykx.{type(self).__name__}'
        return f"{preamble}({self._tree.__repr__()})"

    def enlist(self):
        cpy = copy.deepcopy(self)
        if isinstance(self._tree, K):
            cpy._tree = q.enlist(self._tree)
        else:
            cpy._tree = [self._tree]
        return cpy

    def first(self):
        cpy = copy.deepcopy(self)
        self._tree = q.first(self._tree)
        return cpy

    def eval(self):
        return q.eval(self._tree)

    def reval(self):
        return q.reval(self._tree)

    def append(self, other):
        if isinstance(other, ParseTree):
            self._tree.append(other._tree)
        else:
            self._tree.append(other)

    def extend(self, other):
        if isinstance(other, ParseTree):
            self._tree.extend(other._tree)
        else:
            self._tree.extend(other)

    @staticmethod
    def table(contents):
        """Helper function to create a ParseTree for the creation of a Table.
        If a dict is passed creates: `(flip;(!;contents.keys();enlist,contents.values()))`
        Else creates: `(flip;(!;contents;enlist,contents))`
        For use with the Query API, particauly for `fby` queries.
        """
        if isinstance(contents, (Dictionary, dict)):
            names = list(contents.keys())
            values = list(contents.values())
        else:
            names = contents
            values = contents
        return ParseTree([q.flip, [q('!'), [names], ParseTree.list(values)]])

    @staticmethod
    def list(values):
        """Helper function to create a ParseTree for the creation of a List.
        Creates: `(enlist;value0;value1...valueN)`
        """
        pt = [q.enlist]
        pt.extend(values)
        return ParseTree(pt)

    @staticmethod
    def value(contents, eval=False):
        """Helper function to create a ParseTree which calls `value` on it's contents.
        Creates: ``(`.q.value;contents)``"""
        if eval and licensed:
            return q(CharVector(contents))
        else:
            return ParseTree(['.q.value', CharVector(contents)])

    @staticmethod
    def fby(by, aggregate, data, by_table=False, data_table=False):
        """Helper function to create a ParseTree of an `fby` call
        Creates: `(fby;(enlist;aggregate;data);by)`
        `data_table` and `by_table` can be set to True to create Table ParseTree of their input"""
        if isinstance(aggregate, str):
            aggregate = q.value(CharVector(aggregate))
        if data_table or isinstance(data, (dict, Dictionary)):
            data = ParseTree.table(data)
        if by_table or isinstance(by, (dict, Dictionary)):
            by = ParseTree.table(by)
        return ParseTree([q.fby, ParseTree.list([aggregate, data]), by])


class Variable:
    """Helper class for passing Variable names through the Query API"""
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        preamble = f'pykx.{type(self).__name__}'
        return f"{preamble}('{self._name}')"

    def get(self):
        return q.get(self._name)

    def value(self):
        return q.value(self._name)

    def exists(self):
        q('{@[{get x;1b};x;{0b}]}', self._name)


class Column:
    """Helper class creating queries for the Query API"""
    def __init__(self, column=None, name=None, value=None, is_tree=False):
        if not licensed:
            raise LicenseException("use kx.Column objects")
        if name is not None:
            self._name = name
        else:
            self._name = column
        if value is not None:
            self._value = value
        else:
            self._value = column
        self._is_tree = is_tree

    def __repr__(self):
        preamble = f'pykx.{type(self).__name__}'
        return f"{preamble}(name='{self._name}', value={type(self._value)})"

    """Function for building up a function call off a Column"""
    def call(self, op, *other, iterator=None, col_arg_ind=0, project_args=None):
        params = []
        for param in other:
            if isinstance(param, Column):
                param = param._value
            elif isinstance(param, Variable):
                param = param._name
            else:
                param = toq(param)
                if (
                    isinstance(param, SymbolAtom)
                    or isinstance(param, SymbolVector)
                ):
                    param = [param]
            params.append(param)
        params.insert(col_arg_ind, self._value)
        if isinstance(op, (str, bytes, CharVector)):
            op = op.encode() if isinstance(op, str) else op
            cmd = [q(op)]
        else:
            cmd=[op]
        if project_args is not None:
            project=[]
            for i in range(len(params)):
                project.append(params[i] if i in project_args else q("(value(1;))2"))
            cmd.extend(project)
            cmd = [cmd]
            for i in sorted(project_args, reverse=True):
                params.pop(i)
        cmd.extend(params)

        if iterator is not None:
            id = {'/': '(/)',
                  '\\': '(\\)',
                  '/:': '(/:)',
                  '\\:': '(\\:)',
                  '\\:/:': ['(\\:)', '(/:)'],
                  '/:\\:': ['(/:)', '(\\:)'],
                  }
            if iterator in ['each', 'peach', 'over', 'scan', 'prior']:
                i = [ParseTree.value(iterator, eval=True)]
                i.extend(cmd)
            elif iterator in ["':", "'"]:
                i = [[ParseTree.value(iterator, eval=True), cmd[0]]]
                i.extend(cmd[1:])
            elif iterator in ['/:', '\\:', '/', '\\']:
                i = [[ParseTree.value(id[iterator], eval=True), cmd[0]]]
                i.extend(cmd[1:])
            elif iterator in ['/:\\:', '\\:/:']:
                iterator = id[iterator]
                i = [[ParseTree.value(iterator[1], eval=True),
                     [ParseTree.value(iterator[0], eval=True), cmd[0]]]]
                i.extend(cmd[1:])
            else:
                i = iterator
                i.extend(cmd)
            cmd = i
        cpy = copy.deepcopy(self)
        cpy._value = cmd
        return cpy

    def __add__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('+', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    __radd__ = __add__

    def __sub__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('-', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __rsub__(self, other, iterator=None, col_arg_ind=1, project_args=None):
        return self.call('-', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __mul__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('*', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    __rmul__ = __mul__

    def __floordiv__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('div', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __rfloordiv__(self, other, iterator=None, col_arg_ind=1, project_args=None):
        return self.call('div', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __truediv__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('%', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __rtruediv__(self, other, iterator=None, col_arg_ind=1, project_args=None):
        return self.call('%', other, col_arg_ind=col_arg_ind)

    def __mod__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('mod', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __pow__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('xexp', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __eq__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('=', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __ne__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('<>', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __gt__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('>', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __ge__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('>=', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __lt__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('<', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __le__(self, other, iterator=None, col_arg_ind=0, project_args=None):
        return self.call('<=', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def __pos__(self, iterator=None):
        return self.call('abs', iterator=iterator)

    def __abs__(self, iterator=None):
        return self.call('abs', iterator=iterator)

    def __neg__(self, iterator=None):
        return self.call('neg', iterator=iterator)

    def __floor__(self, iterator=None):
        return self.call('floor', iterator=iterator)

    def __ceil__(self, iterator=None):
        return self.call('ceiling', iterator=iterator)

    def __invert__(self, iterator=None):
        return self.call('not', iterator=iterator)

    def __and__(self, other):
        wp = QueryPhrase(self)
        if isinstance(other, Column):
            wp.append(other)
        elif isinstance(other, QueryPhrase):
            wp.extend(other)
        else:
            raise TypeError(
                f"Supplied object type '{type(other)}' cannot `&` off a `pykx.Column`.")
        return wp

    def __or__(self, other):
        if isinstance(other, Column):
            other = other._value
        elif isinstance(other, QueryPhrase):
            raise TypeError("Cannot | off a Column with a QueryPhrase")
        elif isinstance(other, ParseTree):
            other = other._tree
        cpy = copy.deepcopy(self)
        cpy._value = [q(b'or'), self._value, other]
        return cpy

    def abs(self, iterator=None):
        """
        Return the absolute value of items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the absolute value for all elements in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], 2, -1]
        ...     })
        >>> tab.select(kx.Column('a').abs())
        pykx.Table(pykx.q('
        a
        -
        1
        1
        0
        '))
        ```
        """
        return self.call('abs', iterator=iterator)

    def acos(self, iterator=None):
        """
        Calculate arccos for a column or items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the arccos value for all elements in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').acos())
        pykx.Table(pykx.q('
        a
        --------
        0
        3.141593
        1.570796
        '))
        ```

        Calculate the arccos value for each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').acos(iterator='each'))
        pykx.Table(pykx.q('
        b
        ------------
        3.141593   0
        1.570796 0
        0
        '))
        ```
        """
        return self.call('acos', iterator=iterator)

    def _and(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return the lesser of the underlying boolean values between two columns.
        This should only be used in specific needed use cases
        as it can come at a performance penalty.
        Use `&` or `[]` in general to build queries.

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Return the rows from the table where both conditions are satisfied :

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [1, 2, 3]
        ...     })
        >>> tab.select(where=(kx.Column('a') > 0)._and(kx.Column('b') > 0))
        pykx.Table(pykx.q('
        a b
        ----
        1 1
        '))
        ```
        """
        return self.call('and', other, iterator=iterator,
                         col_arg_ind=col_arg_ind, project_args=project_args)

    def asc(self, iterator=None):
        """
        Sort the values within a column in ascending order

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Sort the values in a column ascending

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').asc())
        pykx.Table(pykx.q('
        a
        --
        -1
        0
        1
        '))
        ```

        Sort each row in a column ascending:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [3, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').asc(iterator='each'))
        pykx.Table(pykx.q('
        b
        ------
        -1 1 2
        1  2 3
        1  2 3
        '))
        ```
        """
        return self.call('asc', iterator=iterator)

    def asin(self, iterator=None):
        """
        Calculate arcsin for a column or items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the arcsin value for all elements in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').asin())
        pykx.Table(pykx.q('
        a
        ---------
        1.570796
        -1.570796
        0
        '))
        ```

        Calculate the arcsin value for each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').asin(iterator='each'))
        pykx.Table(pykx.q('
        b
        ---------------------------
        -1.570796          1.570796
        0         1.570796
        1.570796
        '))
        ```
        """
        return self.call('asin', iterator=iterator)

    def atan(self, iterator=None):
        """
        Calculate arctan for a column or items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the arctan value for all elements in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').atan())
        pykx.Table(pykx.q('
        a
        ----------
        0.7853982
        -0.7853982
        0
        '))
        ```

        Calculate the arctan value for each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').atan(iterator='each'))
        pykx.Table(pykx.q('
        b
        ------------------------------
        -0.7853982 1.107149  0.7853982
        0          0.7853982 1.107149
        0.7853982  1.107149  1.249046
        '))
        ```
        """
        return self.call('atan', iterator=iterator)

    def avg(self, iterator=None):
        """
        Calculate the average value for a column or items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the value for all elements in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0.5],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').avg())
        pykx.Table(pykx.q('
        a
        ---------
        0.1666667
        '))
        ```

        Calculate average value for each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').avg(iterator='each'))
        pykx.Table(pykx.q('
        b
        ---------
        0.6666667
        1
        2
        '))
        ```
        """
        return self.call('avg', iterator=iterator)

    def avgs(self, iterator=None):
        """
        Calculate a running average value for a column or items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the running average across all elements in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0.5],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').avgs())
        pykx.Table(pykx.q('
        a
        ---------
        1
        0
        0.1666667
        '))
        ```

        Calculate average value for each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').avgs(iterator='each'))
        pykx.Table(pykx.q('
        b
        ----------------
        -1 0.5 0.6666667
        0  0.5 1
        1  1.5 2
        '))
        ```
        """
        return self.call('avgs', iterator=iterator)

    def ceiling(self, iterator=None):
        """
        Calculate a nearest integer greater than or equal to items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the ceiling of all elements in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [0.1, 0.4, 3.6],
        ...     'b': [[-1.1, 2.2, 1.6], [0.3, 1.4, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').ceiling())
        pykx.Table(pykx.q('
        a
        -
        1
        1
        4
        '))
        ```

        Calculate the ceiling for all values in each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [0.1, 0.4, 3.6],
        ...     'b': [[-1.1, 2.2, 1.6], [0.3, 1.4, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').ceiling(iterator='each'))
        pykx.Table(pykx.q('
        b
        ------
        -1 3 2
        1  2 2
        1  2 3
        '))
        ```
        """
        return self.call('ceiling', iterator=iterator)

    def cor(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Calculate the correlation between a column and one of:

        - Another column
        - A vector of equal length to the column
        - A PyKX variable in q memory

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the correlation between two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.exec(kx.Column('a').cor(kx.Column('b')))
        pykx.FloatAtom(pykx.q('-0.9946109'))
        ```

        Calculate the correlation between a column and variable in q memory:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> kx.q('custom_var:100?1f')
        >>> tab.exec(kx.Column('a').cor(kx.Variable('custom_var')))
        pykx.FloatAtom(pykx.q('-0.1670133'))
        ```

        Calculate the correlation between a column and a Python variable:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> kx.q('custom_var:100?1f')
        >>> tab.exec(kx.Column('a').cor(kx.random.random(100, 10.0)))
        pykx.FloatAtom(pykx.q('-0.01448725'))
        ```
        """
        return self.call('cor', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def cos(self, iterator=None):
        """
        Calculate cosine for a column or items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the cosine value for all elements in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').cos())
        pykx.Table(pykx.q('
        a
        ---------
        0.5403023
        0.5403023
        1
        '))
        ```

        Calculate the cosine value for each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').cos(iterator='each'))
        pykx.Table(pykx.q('
        b
        -------------------------------
        0.5403023 -0.4161468 0.5403023
        1         0.5403023  -0.4161468
        0.5403023 -0.4161468 -0.9899925
        '))
        ```
        """
        return self.call('cos', iterator=iterator)

    def count(self, iterator=None):
        """
        Calculate the count of the number of elements in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the count of the number of elements in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.exec(kx.Column('a').count())
        pykx.LongAtom(pykx.q('3'))
        ```

        Count the number of elements in each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 2], 1]
        ...     })
        >>> tab.exec(kx.Column('b').count(iterator='each')))
        pykx.LongVector(pykx.q('3 2 1'))
        ```
        """
        return self.call('count', iterator=iterator)

    def cov(self, other, sample=False, iterator=None, col_arg_ind=0, project_args=None):
        """
        Calculate the covariance/sample covariance between a column and one of:

        - Another column
        - A vector of equal length to the column
        - A PyKX variable in q memory

        Parameters:
            other: The second column or variable (Python/q) to be used
            sample: Should calculations of covariance return the
                sample covariance (set True) covariance (set False {default})
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the covariance between two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.exec(kx.Column('a').cov(kx.Column('b')))
        pykx.FloatAtom(pykx.q('-7.87451'))
        ```

        Calculate the sample covariance between a column and variable in q memory:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> kx.q('custom_var:100?1f')
        >>> tab.exec(kx.Column('a').cov(kx.Variable('custom_var'), sample=True))
        pykx.FloatAtom(pykx.q('-0.1670133'))
        ```

        Calculate the covariance between a column and a Python object:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.exec(kx.Column('a').cov(kx.random.random(100, 10.0)))
        pykx.FloatAtom(pykx.q('-0.1093116'))
        ```
        """
        fn = 'scov' if sample else 'cov'
        return self.call(fn, other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def cross(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return the cross product (all possible combinations) between a column and:

            - Another column
            - A vector of items
            - A PyKX variable in q memory

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Generate the cross product of all values in two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').cross(kx.Column('b')))
        pykx.Table(pykx.q('
        a
        ------------------
        0.1392076 9.996082
        0.1392076 9.797281
        0.1392076 9.796094
        ..
        '))
        ```

        Calculate the cross product between a column and list in in q memory:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> kx.q('custom_var:til 3')
        >>> tab.select(kx.Column('a').cross(kx.Variable('custom_var')))
        pykx.Table(pykx.q('
        a
        -----------
        0.1392076 0
        0.1392076 1
        0.1392076 2
        0.2451336 0
        ..
        '))
        ```

        Calculate the cross product between a column and a Python object:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').cross([1, 2, 3]))
        pykx.Table(pykx.q('
        a
        -----------
        0.1392076 1
        0.1392076 2
        0.1392076 3
        0.2451336 1
        ..
        '))
        ```
        """
        return self.call('cross', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def deltas(self, iterator=None):
        """
        Calculate the difference between consecutive elements in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the difference between consecutive values in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.exec(kx.Column('a').deltas())
        pykx.LongVector(pykx.q('1 -2 1'))
        ```

        Calculate the difference between consecutive values in each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').deltas(iterator='each'))
        pykx.Table(pykx.q('
        b
        -------
        -1 3 -1
        0  1 1
        1  1 1
        '))
        ```
        """
        return self.call('deltas', iterator=iterator)

    def desc(self, iterator=None):
        """
        Sort the values within a column in descending order

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Sort the values in a column descending

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').desc())
        pykx.Table(pykx.q('
        a
        --
        1
        0
        -1
        '))
        ```

        Sort each row in a column descending:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [3, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').desc(iterator='each'))
        pykx.Table(pykx.q('
        b
        ------
        2 1 -2
        3 2 1
        3 2 1
        '))
        ```
        """
        return self.call('desc', iterator=iterator)

    def dev(self, sample=False, iterator=None):
        """
        Calculate the standard deviation or sample standard deviation
        for items in a column or rows in a column

        Parameters:
            sample: Should calculations of standard deviation return the
                square root of the sample variance (set True) or the square
                root of the variance (set False {default})
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the standard deviation of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').dev())
        pykx.FloatAtom(pykx.q('2.749494'))
        ```

        Calculate the sample standard deviation for each row in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.select(kx.Column('b').dev(sample=True, iterator='each'))
        pykx.Table(pykx.q('
        b
        ---------
        3.068428
        3.832719
        2.032402
        2.553458
        2.527216
        1.497015
        ..
        '))
        ```
        """
        fn = 'sdev' if sample else 'dev'
        return self.call(fn, iterator=iterator)

    def differ(self, iterator=None):
        """
        Find locations where items in a column or rows in a column
            change value from one item to the next

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Determine if consecutive rows in a column are different values

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 2),
        ...     'b': kx.random.random([100, 3], 2)
        ...     })
        >>> tab.select(kx.Column('a').differ())
        pykx.Table(pykx.q('
        a
        -
        1
        1
        0
        1
        ..
        '))
        ```

        Determine if consecutive values in vectors within a row have different values

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 2),
        ...     'b': kx.random.random([100, 3], 2)
        ...     })
        >>> tab.select(kx.Column('b').differ(iterator='each'))
        pykx.Table(pykx.q('
        b
        ----
        110b
        101b
        110b
        110b
        ..
        '))
        ```
        """
        return self.call('differ', iterator=iterator)

    def distinct(self, iterator=None):
        """
        Find unique items in a column or rows in a column
            change value from one item to the next

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find all unique items in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').distinct())
        pykx.LongVector(pykx.q('0 1 2 3 4'))
        ```

        Find all unique items in each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').distinct(iterator='each'))
        pykx.Table(pykx.q('
        b
        -----
        0 2 3
        1 3 4
        4 2
        1 4
        1 0
        ,2
        ..
        '))
        ```
        """
        return self.call('distinct', iterator=iterator)

    def div(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return the greatest whole number divisor that does not exceed x%y between a column and:

            - Another column
            - An integer
            - A vector of items equal in length to the column
            - A PyKX variable in q memory

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the greatest whole number divisor between two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 5)),
        ...     'b': kx.q.desc(kx.random.random(100, 5))
        ...     })
        >>> tab.select(kx.Column('a').div(kx.Column('b')))
        pykx.Table(pykx.q('
        a
        -
        0
        0
        0
        1
        ..
        '))
        ```

        Calculate the greatest whole number divisor between a column and an integer:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 5)),
        ...     'b': kx.q.desc(kx.random.random(100, 5))
        ...     })
        >>> tab.select(kx.Column('a').div(2))
        pykx.Table(pykx.q('
        a
        -
        1
        0
        2
        0
        ..
        '))
        ```
        """
        return self.call('div', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def exp(self, iterator=None):
        """
        Raise the expentional constant `e` to a power determined by the elements of a column
            or rows of the column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Raise the exponential constant `e` to the power of values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').exp())
        pykx.FloatVector(pykx.q('1 2.718282 7.389056 20.08554..'))
        ```

        Raise the exponential constant `e` to the power of values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').exp(iterator='each'))
        pykx.Table(pykx.q('
        b
        --------------------------
        1        7.389056 20.08554
        2.718282 20.08554 54.59815
        54.59815 7.389056 7.389056
        2.718282 54.59815 2.718282
        ..
        '))
        ```
        """
        return self.call('exp', iterator=iterator)

    @class_or_instancemethod
    def fby(int_or_class, by, aggregate, data, by_table=False, data_table=False): # noqa B902
        """Helper function to create an `fby` inside a Column object
        Creates: `(fby;(enlist;aggregate;data);by)`
        `data_table` and `by_table` can be set to True to create Table ParseTree of their input"""
        if not isinstance(int_or_class, type):
            raise RuntimeError('Please use pykx.Column.fby() instead of running .fby() on a created Column object.') # noqa E501
        if by_table or isinstance(by, (dict, Dictionary)):
            if isinstance(by, dict):
                name = list(by.keys())[0]
            elif isinstance(by, Dictionary):
                name = by.keys()[0]
            else:
                name = by[0]
        elif isinstance(by, Column):
            name = by._name
            by = by._value
        elif isinstance(by, QueryPhrase):
            name = by._names[0]
            by = by.to_dict()
        else:
            name = by
        if isinstance(data, QueryPhrase):
            data = data.to_dict()
        pt = ParseTree.fby(by, aggregate, data, by_table, data_table)
        return Column(name=name, value=pt)

    def fills(self, iterator=None):
        """
        Replace null values with the preceding non-null value within a column or
            vector within rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Replace nulls in a column with preceding null values

        ```python
        >>> import pykx as kx
        >>> value_list = kx.q.til(2)
        >>> value_list.append(kx.LongAtom.null)
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, value_list),
        ...     'b': kx.random.random([100, 3], value_list)
        ...     })
        >>> tab.select(kx.Column('a').fills())
        pykx.Table(pykx.q('
        a
        -

        0
        0
        1
        1
        ..
        '))
        ```

        Replace null values within each row of a column

        ```python
        >>> import pykx as kx
        >>> value_list = kx.q.til(2)
        >>> value_list.append(kx.LongAtom.null)
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, value_list),
        ...     'b': kx.random.random([100, 3], value_list)
        ...     })
        >>> tab.select(kx.Column('b').fills(iterator='each'))
        pykx.Table(pykx.q('
        b
        -----
        1 1 0
        0 1 1
        1 1 1
        1 1 1
        0 1 1
        ..
        '))
        ```
        """
        return self.call('fills', iterator=iterator)

    def first(self, iterator=None):
        """
        Retrieve the first item of a column or first item of each row
            in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Retrieve the first element of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.exec(kx.Column('a').first())
        pykx.LongAtom(pykx.q('1'))
        ```

        Retrieve the first element of each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [3, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').first(iterator='each'))
        pykx.Table(pykx.q('
        b
        --
        -1
        0
        1
        '))
        ```
        """
        return self.call('first', iterator=iterator)

    def floor(self, iterator=None):
        """
        Calculate a nearest integer less than or equal to items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the floor of all elements in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [0.1, 0.4, 3.6],
        ...     'b': [[-1.1, 2.2, 1.6], [0.3, 1.4, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').floor())
        pykx.Table(pykx.q('
        a
        -
        0
        0
        3
        '))
        ```

        Calculate the floor for all values in each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [0.1, 0.4, 3.6],
        ...     'b': [[-1.1, 2.2, 1.6], [0.3, 1.4, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').floor(iterator='each'))
        pykx.Table(pykx.q('
        b
        ------
        -2 2 1
        0  1 2
        1  2 3
        '))
        ```
        """
        return self.call('floor', iterator=iterator)

    def null(self, iterator=None):
        """
        Determine if a value in a column or row of a column is a null value

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find null values within a column

        ```python
        >>> import pykx as kx
        >>> value_list = kx.q.til(2)
        >>> value_list.append(kx.LongAtom.null)
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, value_list),
        ...     'b': kx.random.random([100, 3], value_list)
        ...     })
        >>> tab.select(kx.Column('a').null())
        pykx.Table(pykx.q('
        a
        -
        1
        0
        0
        0
        1
        ..
        '))
        ```

        Calculate the floor for all values in each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> value_list = kx.q.til(2)
        >>> value_list.append(kx.LongAtom.null)
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, value_list),
        ...     'b': kx.random.random([100, 3], value_list)
        ...     })
        >>> tab.select(kx.Column('b').null(iterator='each'))
        pykx.Table(pykx.q('
        b
        ----
        000b
        110b
        011b
        000b
        ..
        '))
        ```
        """
        return self.call('null', iterator=iterator)

    def iasc(self, iterator=None):
        """
        Return the indexes needed to sort the values in a column/row in
            ascending order

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the indices needed to sort values in a column in ascending order

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.q.til(10)),
        ...     'b': kx.random.random([100, 3], kx.q.til(10))
        ...     })
        >>> tab.select(kx.Column('a').iasc())
        pykx.Table(pykx.q('
        a
        --
        19
        25
        30
        40
        50
        ..
        '))
        ```

        Find the indices needed to sort each row in a column in ascending order

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.q.til(10)),
        ...     'b': kx.random.random([100, 3], kx.q.til(10))
        ...     })
        >>> tab.select(kx.Column('b').iasc(iterator='each'))
        pykx.Table(pykx.q('
        b
        ----
        2 0 1
        1 0 2
        0 1 2
        0 1 2
        ..
        '))
        ```
        """
        return self.call('iasc', iterator=iterator)

    def idesc(self, iterator=None):
        """
        Return the indexes needed to sort the values in a column/row in
            descending order

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the indices needed to sort values in a column in descending order

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.q.til(10)),
        ...     'b': kx.random.random([100, 3], kx.q.til(10))
        ...     })
        >>> tab.select(kx.Column('a').idesc())
        pykx.Table(pykx.q('
        a
        --
        39
        43
        45
        56
        60
        ..
        '))
        ```

        Find the indices needed to sort each row in a column in descending order

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.q.til(10)),
        ...     'b': kx.random.random([100, 3], kx.q.til(10))
        ...     })
        >>> tab.select(kx.Column('b').idesc(iterator='each'))
        pykx.Table(pykx.q('
        b
        ----
        1 0 2
        2 0 1
        1 2 0
        2 1 0
        ..
        '))
        ```
        """
        return self.call('idesc', iterator=iterator)

    def inter(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return the intersection between a column and:

            - Another column
            - A Python list/numpy array
            - A PyKX variable in q memory

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Return the distinct intersection of values between two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 5)),
        ...     'b': kx.q.desc(kx.random.random(100, 10))
        ...     })
        >>> tab.exec(kx.Column('a').inter(kx.Column('b')).distinct())
        pykx.LongVector(pykx.q('2 3 1 4 0'))
        ```

        Return the distinct intersection of values between a column and variable in q memory:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 5)),
        ...     'b': kx.q.desc(kx.random.random(100, 10))
        ...     })
        >>> kx.q('custom_var:100?6')
        >>> tab.exec(kx.Column('b').inter(kx.Variable('custom_var')).distinct())
        pykx.LongVector(pykx.q('5 2 1 4 3 0'))
        ```
        """
        return self.call('inter', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def isin(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return a list of booleans indicating if the items in a column are in a specified:

            - Column
            - Python list/numpy array
            - A PyKX variable in q memory

        Most commonly this function is used in where clauses to filter data

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Query a table for anywhere where the column contains the element 'AAPL':

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, ['AAPL', 'GOOG', 'MSFT']),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(where=kx.Column('a').isin(['AAPL']))
        pykx.Table(pykx.q('
        a    b
        ------
        AAPL 7
        AAPL 4
        AAPL 2
        ..
        '))
        ```

        Return the distinct intersection of values between a column and variable in q memory:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, ['AAPL', 'GOOG', 'MSFT']),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> kx.q('custom_var:1 2 3')
        >>> tab.select(where=kx.Column('b').isin(kx.Variable('custom_var')))
        pykx.Table(pykx.q('
        a    b
        ------
        GOOG 2
        MSFT 1
        MSFT 2
        GOOG 3
        ..
        '))
        ```
        """
        return self.call('in', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def last(self, iterator=None):
        """
        Retrieve the last item of a column or last item of each row
            in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Retrieve the last element of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.exec(kx.Column('a').last())
        pykx.LongAtom(pykx.q('0'))
        ```

        Retrieve the last element of each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [3, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').last(iterator='each'))
        pykx.Table(pykx.q('
        b
        -
        1
        2
        3
        '))
        ```
        """
        return self.call('last', iterator=iterator)

    def like(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return a list of booleans indicating whether an item in a column matches a
            supplied regex pattern. Most commonly this function is used in where
            clauses to filter data.

        Parameters:
            other: A string/byte array defining a regex pattern to be used for query
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Query a table for anywhere where the column contains the element 'AAPL':

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, ['TEST', 'tEsTing', 'string']),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(where=kx.Column('a').like('[tT]E*'))
        pykx.Table(pykx.q('
        a    b
        ------
        TEST    7
        TEST    8
        tEsTing 4
        tEsTing 9
        ..
        '))
        ```
        """
        if isinstance(other, str):
            other = other.encode('utf-8')
        return self.call('like', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def log(self, iterator=None):
        """
        Calculate the natural logarithm of values in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the natural log of values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').log())
        pykx.FloatVector(pykx.q('0 1.386294 1.386294 1.098612..'))
        ```

        Calculate the natural log of values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').log(iterator='each'))
        pykx.Table(pykx.q('
        b
        ----------------------------
        1.386294  -0w       0
        -0w       -0w       1.098612
        1.098612  0         0
        1.386294  1.386294  1.098612
        ..
        '))
        ```
        """
        return self.call('log', iterator=iterator)

    def lower(self, iterator=None):
        """
        Change the case of string/symbol objects within a column to be all
            lower case

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Lower all values within a symbol list

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': ['TeStiNG', 'lOwER', 'FuncTion'],
        ...     'b': [1, 2, 3]
        ...     })
        >>> tab.select(kx.Column('a').lower())
        pykx.Table(pykx.q('
        a
        --------
        testing
        lower
        function
        '))
        ```
        """
        return self.call('lower', iterator=iterator)

    def ltrim(self, iterator=None):
        """
        Remove whitespace at the start of character vectors(strings) within items
            in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Remove leading whitespace from all values in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [b'  test ', b' values  ', b'trim  '],
        ...     'b': [1, 2, 3]
        ...     })
        >>> tab.select(kx.Column('a').ltrim())
        pykx.Table(pykx.q('
        a
        ----------
        "test "
        "values  "
        "trim  "
        '))
        ```
        """
        return self.call('ltrim', iterator=iterator)

    def mavg(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Calculate the simple moving average of items in a column for a specified
            window length. Any nulls after the first item are replaced by zero.
            The results are returned as a floating point.

        Parameters:
            other: An integer denoting the window to be used for calculation of
                the moving average
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Query a table for anywhere where the column contains the element 'AAPL':

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, ['TEST', 'tEsTing', 'string']),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(kx.Column('b').mavg(3))
        pykx.Table(pykx.q('
        b
        --------
        7
        7.5
        6.333333
        5.333333
        ..
        '))
        ```
        """
        return self.call('mavg', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def max(self, iterator=None):
        """
        Find the maximum value in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the maximum values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').max())
        pykx.LongAtom(pykx.q('4'))
        ```

        Find the maximum values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').max(iterator='each'))
        pykx.Table(pykx.q('
        b
        -
        2
        3
        4
        4
        ..
        '))
        ```
        """
        return self.call('max', iterator=iterator)

    def maxs(self, iterator=None):
        """
        Find the running maximum value in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the running maximum values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('a').maxs())
        pykx.Table(pykx.q('
        a
        -
        0
        1
        2
        3
        3
        ..
        '))
        ```

        Find the running maximum values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').maxs(iterator='each'))
        pykx.Table(pykx.q('
        b
        -----
        2 2 2
        3 3 3
        4 4 4
        0 3 4
        ..
        '))
        ```
        """
        return self.call('maxs', iterator=iterator)

    def mcount(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Calculate the moving count of non-null items in a column for a specified
            window length. The first 'other' items of the result are the counts
            so far, thereafter the result is the moving average

        Parameters:
            other: An integer denoting the window to be used for calculation of
                the moving count
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the moving count of non-null values within a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, [1, kx.LongAtom.null, 2]),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(kx.Column('a').mcount(3))
        pykx.Table(pykx.q('
        a
        -
        0
        1
        1
        2
        1
        ..
        '))
        ```
        """
        return self.call('mcount', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def md5(self, iterator=None):
        """
        Apply MD5 hash algorithm on columns/rows within a column, it is
            suggested that this function should be used on rows rather than
            columns if being applied

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Apply the MD5 hash algorithm on each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [b'  test ', b' values  ', b'trim  ']
        ...     })
        >>> tab.select(kx.Column('a').md5(iterator='each'))
        pykx.Table(pykx.q('
        a
        ----------------------------------
        0x5609a772b21a22d88f3fb3d21f564eab
        0xbb24d929a28559cc0aa65cb326d7662e
        0xdeafa2fe0c90bcf8c722003bfdeb7c78
        '))
        ```
        """
        return self.call('md5', iterator=iterator)

    def mdev(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Calculate the moving standard deviation for items in a column over a specified
            window length. The first 'other' items of the result are the standard deviation
            of items so far, thereafter the result is the moving standard deviation

        Parameters:
            other: An integer denoting the window to be used for calculation of
                the moving standard deviation
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the moving standard deviation of values within a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, [1, kx.LongAtom.null, 2]),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.exec(kx.Column('b').mdev(3))
        pykx.FloatVector(pykx.q('0 1.5 1.699673 1.247219..'))
        ```
        """
        return self.call('mdev', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def med(self, iterator=None):
        """
        Find the median value in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the median value of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').med())
        pykx.FloatAtom(pykx.q('2f'))
        ```

        Find the median value for each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').med(iterator='each'))
        pykx.Table(pykx.q('
        b
        -
        3
        2
        3
        3
        ..
        '))
        ```
        """
        return self.call('med', iterator=iterator)

    def min(self, iterator=None):
        """
        Find the minimum value in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the minimum values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').max())
        pykx.LongAtom(pykx.q('0'))
        ```

        Find the minimum values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').min(iterator='each'))
        pykx.Table(pykx.q('
        b
        -
        0
        0
        2
        0
        ..
        '))
        ```
        """
        return self.call('min', iterator=iterator)

    def mins(self, iterator=None):
        """
        Find the running minimum value in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the running minimum values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('a').mins())
        pykx.Table(pykx.q('
        a
        -
        0
        0
        0
        0
        0
        ..
        '))
        ```

        Find the running minimum values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').mins(iterator='each'))
        pykx.Table(pykx.q('
        b
        -----
        0 0 0
        0 0 0
        2 2 2
        2 2 0
        ..
        '))
        ```
        """
        return self.call('mins', iterator=iterator)

    def mmax(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Calculate the moving maximum  for items in a column over a specified
            window length. The first 'other' items of the result are the maximum
            of items so far, thereafter the result is the moving maximum

        Parameters:
            other: An integer denoting the window to be used for calculation of
                the moving maximum
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the moving maximum of values within a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, [1, kx.LongAtom.null, 2]),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.exec(kx.Column('b').mmax(3))
        pykx.LongVector(pykx.q('4 4 4 3 7..'))
        ```
        """
        return self.call('mmax', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def mmin(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Calculate the moving minumum for items in a column over a specified
            window length. The first 'other' items of the result are the minimum
            of items so far, thereafter the result is the moving minimum

        Parameters:
            other: An integer denoting the window to be used for calculation of
                the moving minimum
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the moving minimum of values within a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, [1, kx.LongAtom.null, 2]),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.exec(kx.Column('b').mmin(3))
        pykx.LongVector(pykx.q('4 1 0 0 0 2..'))
        ```
        """
        return self.call('mmin', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def mod(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Calculate the modulus of items in a column for a given value.

        Parameters:
            other: An integer denoting the divisor to be used when calculating the modulus
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the modulus for items within a column for a value 3:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, [1, kx.LongAtom.null, 2]),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.exec(kx.Column('b').mod(3))
        pykx.LongVector(pykx.q('1 2 1 1 0..'))
        ```
        """
        return self.call('mod', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def msum(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Calculate the moving sum of items in a column over a specified
            window length. The first 'other' items of the result are the sum
            of items so far, thereafter the result is the moving sum

        Parameters:
            other: An integer denoting the window to be used for calculation of
                the moving sum
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the moving sum of values within a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, [1, kx.LongAtom.null, 2]),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.exec(kx.Column('b').msum(3))
        pykx.LongVector(pykx.q('4 5 5 4 10 12..'))
        ```
        """
        return self.call('msum', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def neg(self, iterator=None):
        """
        Compute the negative value for all items in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Compute the negative value for all items in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('a').neg())
        pykx.Table(pykx.q('
        a
        --
        0
        -3
        -4
        -2
        0
        ..
        '))
        ```
        """
        return self.call('neg', iterator=iterator)

    def _not(self, iterator=None):
        """
        Return rows where the condition does not evaluate to True.

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Return the rows that do not satisfy the condition

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [1, 2, 3]
        ...     })
        >>> tab.select(where=(kx.Column('a') > 0)._not())
        pykx.Table(pykx.q('
        a  b
        ----
        -1 2
        0  3
        '))
        ```
        """
        return self.call('not', iterator=iterator)

    def _or(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return the larger of the underlying boolean values between two columns:

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Return the rows from the table where either condition is True:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [1, 2, 3]
        ...     })
        >>> tab.select(where=(kx.Column('a') > 0)._or(kx.Column('b') > 0))
        pykx.Table(pykx.q('
        a b
        ----
        1 1
        -1 2
        0 3
        '))
        ```
        """
        return self.call('or', other, iterator=iterator,
                         col_arg_ind=col_arg_ind, project_args=project_args)

    def prd(self, iterator=None):
        """
        Calculate the product of all values in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the product of values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').prd())
        pykx.FloatAtom(pykx.q('9.076436e+25'))
        ```

        Find the product of values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').prd(iterator='each'))
        pykx.Table(pykx.q('
        b
        -
        0
        0
        32
        0
        0
        48
        ..
        '))
        ```
        """
        return self.call('prd', iterator=iterator)

    def prds(self, iterator=None):
        """
        Find the running product of values in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the running product of values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('a').prds())
        pykx.Table(pykx.q('
        a
        ---------
        0.8276359
        3.833871
        2.317464
        3.940125
        ..
        '))
        ```

        Find the running product of values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').prds(iterator='each'))
        pykx.Table(pykx.q('
        b
        -------
        0 0  0
        0 0  0
        2 8  32
        2 6  0
        0 0  0
        3 12 48
        ..
        '))
        ```
        """
        return self.call('prds', iterator=iterator)

    def prev(self, iterator=None):
        """
        Retrieve the immediately preceding item in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Shift the values in column 'a' within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.update(kx.Column('a').prev())
        pykx.Table(pykx.q('
        a         b
        ---------------
                  0 4 4
        0.8276359 0 2 3
        4.632315  2 4 4
        0.6044712 2 3 0
        '))
        ```
        """
        return self.call('prev', iterator=iterator)

    def rank(self, iterator=None):
        """
        Retrieve the positions items would take in a sorted list from a column
            or items in a column, this is equivalent of calling `iasc` twice`

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the rank of items in a list

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 1000),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.update(kx.Column('a').prev())
        pykx.Table(pykx.q('
        a  b
        --------
        89 3 4 4
        31 1 2 1
        57 4 4 0
        25 4 4 2
        ..
        '))
        ```
        """
        return self.call('rank', iterator=iterator)

    def ratios(self, iterator=None):
        """
        Calculate the ratio between consecutive elements in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the difference between consecutive values in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 1000),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').ratios())
        pykx.FloatVector(pykx.q('908 0.3964758 1.45..'))
        ```
        """
        return self.call('ratios', iterator=iterator)

    def reciprocal(self, iterator=None):
        """
        Calculate the reciprocal of all elements in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the reciprocal of items in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 1000),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').reciprocal())
        pykx.FloatVector(pykx.q('0.001101322 0.002777778 0.001915709..'))
        ```
        """
        return self.call('reciprocal', iterator=iterator)

    def reverse(self, iterator=None):
        """
        Reverse the elements of a column or contents of rows of the column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the reverse the items in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        >>> tab = kx.Table(data={
        ...     'a': kx.q.til(100),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').reverse())
        pykx.LongVector(pykx.q('99 98 97..'))
        ```
        """
        return self.call('reverse')

    def rotate(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Shift the items in a column "left" or "right" by an integer amount denoted
            by the parameter other.

        Parameters:
            other: An integer denoting the number of elements left(positve) or right(negative)
                which the column list will be shifted
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Shift the items in column b by 2 left:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, ['TEST', 'tEsTing', 'string']),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(kx.Column('b') & kx.Column('b').rotate(2).name('rot_b'))
        pykx.Table(pykx.q('
        b rot_b
        -------
        7 4
        8 4
        4 6
        4 9
        6 9
        9 2
        ..
        '))
        ```

        Round a column of times to 15 minute buckets

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.q('100?0t')),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(kx.Column('b').avg(), by=kx.Column('a').minute.xbar(15))
        pykx.KeyedTable(pykx.q('
        a    | b
        -----| --------
        00:00| 5.666667
        00:15| 3
        00:45| 1
        01:00| 4.5
        '))
        ```
        """
        return self.call('rotate', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def rtrim(self, iterator=None):
        """
        Remove whitespace from the end of character vectors(strings) within items
            in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Remove trailing whitespace from all values in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [b'  test ', b' values  ', b'trim  '],
        ...     'b': [1, 2, 3]
        ...     })
        >>> tab.select(kx.Column('a').ltrim())
        pykx.Table(pykx.q('
        a
        ---------
        "  test"
        " values"
        "trim"
        '))
        ```
        """
        return self.call('rtrim', iterator=iterator)

    def scov(self, iterator=None):
        """
        Calculate the sample covariance for items in a column or rows in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the sample covariance of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').scov())
        pykx.FloatAtom(pykx.q('8.983196'))
        ```

        Calculate the sample covariance for each row in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.select(kx.Column('b').scov(iterator='each'))
        pykx.Table(pykx.q('
        b
        ---------
        0.3333333
        0.3333333
        5.333333
        1.333333
        ..
        '))
        ```
        """
        return self.call('scov', iterator=iterator)

    def sdev(self, iterator=None):
        """
        Calculate the sample deviation for items in a column or rows in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the sample deviation of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').sdev())
        pykx.FloatAtom(pykx.q('8.983196'))
        ```

        Calculate the sample deviation for each row in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.select(kx.Column('b').sdev(iterator='each'))
        pykx.Table(pykx.q('
        b
        ---------
        0.3333333
        0.3333333
        5.333333
        1.333333
        ..
        '))
        ```
        """
        return self.call('sdev', iterator=iterator)

    def signum(self, iterator=None):
        """
        Determine if the elements in a column or items in the row of a column is

            - null or negative, returns -1i
            - zero, returns 0i
            - positive, returns 1i

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Determine if values are positive, null, zero or positive in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.exec(kx.Column('a').signum())
        pykx.IntVector(pykx.q('1 -1 0i'))
        ```

        Find if values are positive, null, zero or positive in each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').signum(iterator='each'))
        pykx.Table(pykx.q('
        b
        ------
        -1 1 1
        0  1 1
        1  1 1
        '))
        ```
        """
        return self.call('signum', iterator=iterator)

    def sin(self, iterator=None):
        """
        Calculate sine for a column or items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the sine value for all elements in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').sin())
        pykx.Table(pykx.q('
        a
        ---------
        0.841471
        -0.841471
        0
        '))
        ```

        Calculate the sine value for each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').sin(iterator='each'))
        pykx.Table(pykx.q('
        b
        -----------------------------
        -0.841471 0.9092974 0.841471
        0         0.841471  0.9092974
        0.841471  0.9092974 0.14112
        '))
        ```
        """
        return self.call('sin', iterator=iterator)

    def sqrt(self, iterator=None):
        """
        Calculate the square root each element of a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the square root of each value within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').sqrt())
        pykx.FloatVector(pykx.q('1.152283 1.717071 1.253352..'))
        ```

        Find the square root of each value within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').sqrt(iterator='each'))
        pykx.Table(pykx.q('
        b
        --------------------------
        1.732051 1.414214 1.732051
        2        1.732051 2
        1.732051 1.732051 1.414214
        0        0        1.414214
        1.732051 1.414214 1.414214
        ..
        '))
        ```
        """
        return self.call('sqrt', iterator=iterator)

    def string(self, iterator=None):
        """
        Convert all elements of a column to a PyKX string (CharVector)

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Convert all elements of a column to strings

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1234, 1.01, 12142],
        ...     'b': [1, 2, 3]
        ...     })
        >>> tab.select(kx.Column('a').string())
        pykx.Table(pykx.q('
        a
        -------
        "1234"
        "1.01"
        "12142"
        '))
        ```
        """
        return self.call('string', iterator=iterator)

    def sum(self, iterator=None):
        """
        Calculate the sum of all values in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the sum of values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').sum())
        pykx.FloatAtom(pykx.q('249.3847'))
        ```

        Find the sum of values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').sum(iterator='each'))
        pykx.Table(pykx.q('
        b
        -
        6
        4
        6
        6
        ..
        '))
        ```
        """
        return self.call('sum', iterator=iterator)

    def sums(self, iterator=None):
        """
        Find the running sum of values in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the running sum of values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('a').sums())
        pykx.Table(pykx.q('
        a
        ---------
        4.396227
        8.42457
        8.87813
        11.26718
        ..
        '))
        ```

        Find the running sum of values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').sums(iterator='each'))
        pykx.Table(pykx.q('
        b
        -------
        0 3 6
        3 4 4
        1 3 6
        3 3 6
        ..
        '))
        ```
        """
        return self.call('sums', iterator=iterator)

    def svar(self, iterator=None):
        """
        Calculate the sample variance for items in a column or rows in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the sample variance of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').svar())
        pykx.FloatAtom(pykx.q('8.394893'))
        ```

        Calculate the sample variance for each row in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.select(kx.Column('b').svar(iterator='each'))
        pykx.Table(pykx.q('
        b
        ---------
        6.023586
        29.48778
        6.318229
        0.1609426
        5.241295
        ..
        '))
        ```
        """
        return self.call('svar', iterator=iterator)

    def tan(self, iterator=None):
        """
        Calculate tan for a column or items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the tan value for all elements in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').tan())
        pykx.Table(pykx.q('
        a
        ---------
        1.557408
        -1.557408
        0
        '))
        ```

        Calculate the tan value for each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').tan(iterator='each'))
        pykx.Table(pykx.q('
        b
        -----------------------------
        -1.557408 -2.18504 1.557408
        0         1.557408 -2.18504
        1.557408  -2.18504 -0.1425465
        '))
        ```
        """
        return self.call('tan', iterator=iterator)

    def trim(self, iterator=None):
        """
        Remove whitespace from the start and end of character vectors(strings)
            within items in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Remove trailing and following whitespace from all values in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [b'  test ', b' values  ', b'trim  '],
        ...     'b': [1, 2, 3]
        ...     })
        >>> tab.select(kx.Column('a').trim())
        pykx.Table(pykx.q('
        a
        ---------
        "test"
        "values"
        "trim"
        '))
        ```
        """
        return self.call('trim', iterator=iterator)

    def union(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return the union between a column and:

            - Another column
            - A Python list/numpy array
            - A PyKX variable in q memory

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Return the distinct union of values between two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 5)),
        ...     'b': kx.q.desc(kx.random.random(100, 10))
        ...     })
        >>> tab.exec(kx.Column('a').union(kx.Column('b')).distinct())
        pykx.LongVector(pykx.q('0 1 2 3 4 9 8 7 6 5'))
        ```

        Return the distinct union of values between a column and variable in q memory:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 5)),
        ...     'b': kx.q.desc(kx.random.random(100, 10))
        ...     })
        >>> kx.q('custom_var:100?6')
        >>> tab.exec(kx.Column('b').union(kx.Variable('custom_var')).distinct())
        pykx.LongVector(pykx.q('9 8 7 6 5 4 3 2 1 0'))
        ```
        """
        return self.call('union', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def upper(self, iterator=None):
        """
        Change the case of string/symbol objects within a column to be all
            upper case

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Convert all values within a symbol list to be upper case

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': ['TeStiNG', 'UpPer', 'FuncTion'],
        ...     'b': [1, 2, 3]
        ...     })
        >>> tab.select(kx.Column('a').upper())
        pykx.Table(pykx.q('
        a
        --------
        TESTING
        UPPER
        FUNCTION
        '))
        ```
        """
        return self.call('upper', iterator=iterator)

    def var(self, iterator=None, sample=False):
        """
        Calculate the variance or sample variance for items in a
            column or rows in a column

        Parameters:
            sample: Should calculation of variance return the
                sample variance (set True) or the variance (set False {default})
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the variance of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').var())
        pykx.FloatAtom(pykx.q('8.310944'))
        ```

        Calculate the sample sample deviation for each row in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.select(kx.Column('b').var(sample=True, iterator='each'))
        pykx.Table(pykx.q('
        b
        ---------
        6.023586
        29.48778
        6.318229
        0.1609426
        5.241295
        ..
        '))
        ```
        """
        fn = 'svar' if sample else 'var'
        return self.call(fn, iterator=iterator)

    def wavg(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return the weighted average between a column and:

            - Another column
            - A Python list/numpy array
            - A PyKX variable in q memory

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Return the weighted average between two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 5)),
        ...     'b': kx.q.desc(kx.random.random(100, 10))
        ...     })
        >>> tab.exec(kx.Column('a').wavg(kx.Column('b')))
        pykx.FloatAtom(pykx.q('2.456731'))
        ```

        Return the weighted average between a column and variable in q memory:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 5)),
        ...     'b': kx.q.desc(kx.random.random(100, 10))
        ...     })
        >>> kx.q('custom_var:100?6')
        >>> tab.exec(kx.Column('b').wavg(kx.Variable('custom_var')))
        pykx.FloatAtom(pykx.q('2.431111'))
        ```
        """
        return self.call('wavg', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def within(self, lower, upper, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return a boolean list indicating whether the items of a column are within bounds
            of an lower and upper limite.

        Parameters:
            lower: A sortable item defining the lower limit
            upper: A sortable item defining the upper limit
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Return any rows where column a has a value within the range 1, 4:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.q.desc(kx.random.random(100, 10))
        ...     })
        >>> tab.select(where = kx.Column('a').within(1, 4))
        pykx.Table(pykx.q('
        a b
        ---
        1 9
        1 9
        2 9
        2 9
        4 9
        ..
        '))
        ```

        Return any rows where column a has a value within a date range:

        ```python
        >>> import pykx as kx
        >>> today = kx.DateAtom('today')
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, today - range(0, 10)),
        ...     'b': kx.q.desc(kx.random.random(100, 10))
        ...     })
        >>> tab.select(where=kx.Column('a').within(today - 5, today - 3))
        pykx.FloatAtom(pykx.q('2.431111'))
        ```
        """
        return self.call('within', [lower, upper], iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def wsum(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Return the weighted sum between a column and:

            - Another column
            - A Python list/numpy array
            - A PyKX variable in q memory

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Return the weighted sum between two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 5)),
        ...     'b': kx.q.desc(kx.random.random(100, 10))
        ...     })
        >>> tab.exec(kx.Column('a').wsum(kx.Column('b')))
        pykx.FloatAtom(pykx.q('511f'))
        ```

        Return the weighted sum between a column and variable in q memory:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 5)),
        ...     'b': kx.q.desc(kx.random.random(100, 10))
        ...     })
        >>> kx.q('custom_var:100?6')
        >>> tab.exec(kx.Column('b').wsum(kx.Variable('custom_var')))
        pykx.FloatAtom(pykx.q('1094f'))
        ```
        """
        return self.call('wsum', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def xbar(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Round the elements of a column down to the nearest multiple of the supplied
            parameter other.

        Parameters:
            other: An integer denoting the multiple to which all values will be rounded
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Round the items of a column to multiples of 3:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, ['TEST', 'tEsTing', 'string']),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(kx.Column('b').xbar(3))
        pykx.Table(pykx.q('
        b
        -
        3
        6
        9
        6
        ..
        '))
        ```

        Round a column of times to 15 minute buckets

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.q('100?0t')),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(kx.Column('b').avg(), by=kx.Column('a').minute.xbar(15))
        pykx.KeyedTable(pykx.q('
        a    | b
        -----| --------
        00:00| 5.666667
        00:15| 3
        00:45| 1
        01:00| 4.5
        '))
        ```
        """
        return self.call('xbar', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def xexp(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Raise the elements of a column down to power of the value supplied as
            the parameter other.

        Parameters:
            other: An integer denoting the power to which all values will be raised
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Round the items of a column to multiples of 3:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(kx.Column('b').xexp(2))
        pykx.Table(pykx.q('
        b
        ---
        64
        512
        4
        8
        2
        ..
        '))
        ```
        """
        return self.call('xexp', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def xlog(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Return the base-N logarithm for the elements of a column where N is specified
            by the parameter other.

        Parameters:
            other: An integer denoting the logarithmic base to which all values will be set
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Round the items of a column to multiples of 3:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(kx.Column('b').xlog(2))
        pykx.Table(pykx.q('
        b
        --------
        1.584963
        3.169925
        2.321928
        3.169925
        ..
        '))
        ```
        """
        return self.call('xlog', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def xprev(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        For a specified column return for each item in the column the item N elements
            before it. Where N is specified by the parameter other.

        Parameters:
            other: An integer denoting the number of indices before elements in the list
                to retrieve the value of
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Shift the data in a column by 3 indexes:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.select(kx.Column('a') & kx.Column('a').xprev(3).name('lag_3_a'))
        pykx.Table(pykx.q('
        a         lag_3_a
        -------------------
        3.927524
        5.170911
        5.159796
        4.066642  3.927524
        1.780839  5.170911
        3.017723  5.159796
        '))
        ```
        """
        return self.call('xprev', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    @property
    def hour(self):
        """
        Retrieve the hour information from a temporal column


        Examples:

        Retrieve hour information from a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.TimestampAtom.inf),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').hour)
        pykx.IntVector(pykx.q('11 1 13 12..'))
        ```
        """
        return self.call('`hh$')

    @property
    def minute(self):
        """
        Retrieve the minute information from a temporal column


        Examples:

        Retrieve minute information from a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.TimestampAtom.inf),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').minute)
        pykx.MinuteVector(pykx.q('11:55 01:09 13:43..'))
        ```
        """
        return self.call('`minute$')

    @property
    def date(self):
        """
        Retrieve the date information from a temporal column


        Examples:

        Retrieve date information from a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.TimestampAtom.inf),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').date)
        pykx.DateVector(pykx.q('2122.07.05 2120.10.23..'))
        ```
        """
        return self.call('`date$')

    @property
    def year(self):
        """
        Retrieve the year information from a temporal column


        Examples:

        Retrieve year information from a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.TimestampAtom.inf),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').year)
        pykx.IntVector(pykx.q('2122 2120 2185..'))
        ```
        """
        return self.call('`year$')

    @property
    def day(self):
        """
        Retrieve the day of the month information from a temporal column


        Examples:

        Retrieve day of month information from a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.TimestampAtom.inf),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').day)
        pykx.IntVector(pykx.q('7 10 12..'))
        ```
        """
        return self.call('`dd$')

    @property
    def month(self):
        """
        Retrieve the month information from a temporal column


        Examples:

        Retrieve month information from a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.TimestampAtom.inf),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').month)
        pykx.IntVector(pykx.q('7 10 12..'))
        ```
        """
        return self.call('`mm$')

    @property
    def second(self):
        """
        Retrieve the second information from a temporal column


        Examples:

        Retrieve year information from a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.TimestampAtom.inf),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').second)
        pykx.SecondVector(pykx.q('11:55:50 01:09:35..'))
        ```
        """
        return self.call('`second$')

    # Functions below this point are generalisations of q operators or     #
    # expanded function names to improve readability in Python first usage #

    def add(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Add the content of a column to one of:

            - Another column
            - A vector of equal length to the column
            - A PyKX variable in q memory

        Note in it's most basic usage this is equivalent to

        ```python
        >>> kx.Column('x') + kx.Column('y')
        ```

        It is supplied as a named function to allow the use of iterators
            when adding elements.

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Add together two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').add(kx.Column('b')))
        pykx.Table(pykx.q('
        a
        --------
        9.967087
        9.870729
        9.882342
        9.95924
        ..
        '))
        ```

        Add a value of 3 to each element of a column.

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').add(3))
        pykx.Table(pykx.q('
        a
        --------
        3.021845
        3.044166
        3.062797
        3.051352
        ..
        '))
        ```

        For each row in a column add 3 and 4 to the value of the column
            This makes use of each-left and each-right from q:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').add([3, 4], iterator='/:\:'))
        pykx.Table(pykx.q('
        a
        -----------------
        3.021845 4.021845
        3.044166 4.044166
        3.062797 4.062797
        3.166843 4.166843
        ..
        '))
        ```
        """
        return self.call('+', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def name(self, name):
        """
        Rename the resulting column from a calculation

        Parameters:
            name: The name to be given to the column following application of function

        Examples:

        Rename the column 'a' to 'average_a' following application of the function average

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0.5],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').average().name('average_a'))
        pykx.Table(pykx.q('
        average_a
        ---------
        0.1666667
        '))
        ```
        """
        cpy = copy.deepcopy(self)
        cpy._name = name
        return cpy

    def average(self, iterator=None):
        """
        Calculate the average value for a column or items in a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the value for all elements in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0.5],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').average())
        pykx.Table(pykx.q('
        a
        ---------
        0.1666667
        '))
        ```

        Calculate average value for each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('b').average(iterator='each'))
        pykx.Table(pykx.q('
        b
        ---------
        0.6666667
        1
        2
        '))
        ```
        """
        return self.call('avg', iterator=iterator)

    def cast(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Convert the content of a column to another PyKX type

        Parameters:
            other: The name of the type to which your column should be cast
                or the lower case letter used to define it in q, for more information
                see [here](https://code.kx.com/q/ref/cast/).
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Example:

        Cast a column containing PyKX long objects to float objects

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.q.til(10)),
        ...     'b': kx.random.random(100, kx.q.til(10))
        ...     })
        >>> tab.dtypes
        pykx.Table(pykx.q('
        columns datatypes     type
        -----------------------------------
        a       "kx.LongAtom" "kx.LongAtom"
        b       "kx.LongAtom" "kx.LongAtom"
        '))
        >>> tab.select(
        ...     kx.Column('a') &
        ...     kx.Column('a').cast('float').name('a_float') &
        ...     kx.Column('b')).dtypes
        pykx.Table(pykx.q('
        columns datatypes      type
        -------------------------------------
        a       "kx.LongAtom"  "kx.LongAtom"
        a_float "kx.FloatAtom" "kx.FloatAtom"
        b       "kx.LongAtom"  "kx.LongAtom"
        '))
        ```
        """
        if not isinstance(other, str):
            raise QError('Supplied value other must be a str')
        if 1 == len(other):
            other = other.encode('UTF-8')
        return self.call('$', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def correlation(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Calculate the correlation between a column and one of:

            - Another column
            - A vector of equal length to the column
            - A PyKX variable in q memory

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the correlation between two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.exec(kx.Column('a').cor(kx.Column('b')))
        pykx.FloatAtom(pykx.q('-0.9946109'))
        ```

        Calculate the correlation between a column and variable in q memory:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> kx.q('custom_var:100?1f')
        >>> tab.exec(kx.Column('a').correlation(kx.Variable('custom_var')))
        pykx.FloatAtom(pykx.q('-0.1670133'))
        ```

        Calculate the correlation between a column and a Python variable:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> kx.q('custom_var:100?1f')
        >>> tab.exec(kx.Column('a').correlation(kx.random.random(100, 10.0)))
        pykx.FloatAtom(pykx.q('-0.01448725'))
        ```
        """
        return self.call('cor', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def covariance(self, other, sample=False, iterator=None, col_arg_ind=0, project_args=None):
        """
        Calculate the covariance/sample covariance between a column and one of:

            - Another column
            - A vector of equal length to the column
            - A PyKX variable in q memory

        Parameters:
            other: The second column or variable (Python/q) to be used
            sample: Should calculations of covariance return the
                sample covariance (set True) covariance (set False {default})
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the covariance between two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.exec(kx.Column('a').covariance(kx.Column('b')))
        pykx.FloatAtom(pykx.q('-7.87451'))
        ```

        Calculate the sample covariance between a column and variable in q memory:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> kx.q('custom_var:100?1f')
        >>> tab.exec(kx.Column('a').covariance(kx.Variable('custom_var'), sample=True))
        pykx.FloatAtom(pykx.q('-0.1670133'))
        ```

        Calculate the covariance between a column and a Python object:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.exec(kx.Column('a').covariance(kx.random.random(100, 10.0)))
        pykx.FloatAtom(pykx.q('-0.1093116'))
        ```
        """
        fn = 'scov' if sample else 'cov'
        return self.call(fn, other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def divide(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Divide the content of one column by:

            - Another column
            - Python/Numpy list/item
            - A PyKX variable in q memory

        Note in it's most basic usage this is equivalent to

        ```python
        >>> kx.Column('x') % kx.Column('y')
        ```

        It is supplied as a named function to allow the use of iterators
            when adding elements.

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Divide on column by another column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').divide(kx.Column('b')))
        pykx.Table(pykx.q('
        a
        -----------
        0.0021965
        0.004494546
        0.006395103
        0.01703797
        ..
        '))
        ```

        Divide each element of a column by 3.

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').divide(3))
        pykx.Table(pykx.q('
        a
        -----------
        0.007281574
        0.01472198
        0.02093233
        0.05561419
        ..
        '))
        ```

        For each row in a column divide the row by both 3 and 4 independently.
            This makes use of each-left and each-right from q:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').divide([3, 4], iterator='/:\:'))
        pykx.Table(pykx.q('
        a
        ---------------------
        0.06553417 0.08737889
        0.1324978  0.1766638
        0.188391   0.251188
        0.5005277  0.6673703
        ..
        '))
        ```
        """
        return self.call('%', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def drop(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Drop N rows from a column or N elements from items in a column using
            an iterator. Where N is specified by the other parameter.

        Parameters:
            other: An integer defining the number of elements to drop
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Drop 3 rows from a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.q.til(10)),
        ...     'b': kx.random.random([100, 3], kx.q.til(10))
        ...     })
        >>> tab.select(kx.Column('a').drop(3).count())
        pykx.Table(pykx.q('
        a
        --
        10
        12
        24
        27
        ..
        '))
        ```
        """
        return self.call('_', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def fill(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Replace all null values in a column with a specified 'other' parameter

        Parameters:
            other: The value which should replace nulls within a column
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Replace all nulls in column a with a value 0, displaying that only 0, 1 and 2 exist
            in this column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, [1, kx.LongAtom.null, 2]),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.exec(kx.Column('a').fill(0).distinct())
        pykx.LongVector(pykx.q('1 0 2'))
        ```
        """
        return self.call('^', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def index_sort(self, ascend=True, iterator=None):
        """
        Return the indexes needed to sort the values in a column/row in
            ascending order or descending order

        Parameters:
            ascend: A boolean indicating if the index return should be
                retrieved in ascending or descending order
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the indices needed to sort values in a column in ascending order

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.q.til(10)),
        ...     'b': kx.random.random([100, 3], kx.q.til(10))
        ...     })
        >>> tab.select(kx.Column('a').index_sort())
        pykx.Table(pykx.q('
        a
        --
        10
        12
        24
        27
        ..
        '))
        ```

        Find the indices needed to sort values in a column in descending order

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.q.til(10)),
        ...     'b': kx.random.random([100, 3], kx.q.til(10))
        ...     })
        >>> tab.select(kx.Column('a').index_sort(ascend=False))
        pykx.Table(pykx.q('
        a
        --
        1
        15
        44
        50
        ..
        '))
        ```

        """
        fn = 'iasc' if ascend else 'idesc'
        return self.call(fn, iterator=iterator)

    def join(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Join the content of one column with another or using an iterator
            produce complex combinations of items in a column with:

            - Another Column
            - A list/item which is to be joined to the column
            - A variable in q memory

        Parameters:
            other: The Column, list, item or variable to be joined to the
                original column
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Join the content of one column to another column (extend the column)

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> len(tab.select(kx.Column('a').join(kx.Column('b'))))
        200
        ```

        Join the value 3 to each row in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').join(3, iterator="'"))
        pykx.Table(pykx.q('
        a
        ---
        1 3
        2 3
        1 3
        3 3
        ..
        '))
        ```
        """
        return self.call(',', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def len(self, iterator=None):
        """
        Calculate the length of the number of elements in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the length of the number of elements in a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.exec(kx.Column('a').len())
        pykx.LongAtom(pykx.q('3'))
        ```

        Count the length of elements in each row of a specified column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 2], 1]
        ...     })
        >>> tab.exec(kx.Column('b').len(iterator='each')))
        pykx.LongVector(pykx.q('3 3 3'))
        ```
        """
        return self.call('count', iterator=iterator)

    def modulus(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Calculate the modulus of items in a column for a given value.

        Parameters:
            other: An integer denoting the divisor to be used when calculating the modulus
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 1.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Calculate the modulus for items within a column for a value 3:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, [1, kx.LongAtom.null, 2]),
        ...     'b': kx.random.random(100, 10)
        ...     })
        >>> tab.exec(kx.Column('b').mod(3))
        pykx.LongVector(pykx.q('1 2 1 1 0..'))
        ```
        """
        return self.call('mod', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def multiply(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Multiply the content of a column to one of:

            - Another column
            - Python/Numpy list
            - A PyKX variable in q memory

        Note in it's most basic usage this is equivalent to

        ```python
        >>> kx.Column('x') * kx.Column('y')
        ```

        It is supplied as a named function to allow the use of iterators
            when adding elements.

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Multiply together two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').multiply(kx.Column('b')))
        pykx.Table(pykx.q('
        a
        ---------
        0.2172511
        0.4339994
        0.616638
        1.633789
        ..
        '))
        ```

        Multiply each element of a column by 3.

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').multiply(3))
        pykx.Table(pykx.q('
        a
        ----------
        0.06553417
        0.1324978
        0.188391
        0.5005277
        ..
        '))
        ```

        For each row in a column multiply the row by both 3 and 4 independently.
            This makes use of each-left and each-right from q:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').multiply([3, 4], iterator='/:\:'))
        pykx.Table(pykx.q('
        a
        ---------------------
        0.06553417 0.08737889
        0.1324978  0.1766638
        0.188391   0.251188
        0.5005277  0.6673703
        ..
        '))
        ```
        """
        return self.call('*', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def next_item(self, iterator=None):
        """
        Retrieve the immediately following item in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Shift the values in column 'a' within a column forward one

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.update(kx.Column('a').next_item())
        pykx.Table(pykx.q('
        a         b
        ---------------
                  0 4 4
        0.8276359 0 2 3
        4.632315  2 4 4
        0.6044712 2 3 0
        '))
        ```
        """
        return self.call('next', iterator=iterator)

    def previous_item(self, iterator=None):
        """
        Retrieve the immediately preceding item in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Shift the values in column 'a' within a column back one

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.update(kx.Column('a').previous_item())
        pykx.Table(pykx.q('
        a         b
        ---------------
                  0 4 4
        0.8276359 0 2 3
        4.632315  2 4 4
        0.6044712 2 3 0
        '))
        ```
        """
        return self.call('prev', iterator=iterator)

    def product(self, iterator=None):
        """
        Calculate the product of all values in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the product of values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.exec(kx.Column('a').product())
        pykx.FloatAtom(pykx.q('9.076436e+25'))
        ```

        Find the product of values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').product(iterator='each'))
        pykx.Table(pykx.q('
        b
        -
        0
        0
        32
        0
        0
        48
        ..
        '))
        ```
        """
        return self.call('prd', iterator=iterator)

    def products(self, iterator=None):
        """
        Find the running product of values in a column or rows of a column

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Find the running product of values within a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5.0),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('a').products())
        pykx.Table(pykx.q('
        a
        ---------
        0.8276359
        3.833871
        2.317464
        3.940125
        ..
        '))
        ```

        Find the running product of values within each row of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 5),
        ...     'b': kx.random.random([100, 3], 5)
        ...     })
        >>> tab.select(kx.Column('b').products(iterator='each'))
        pykx.Table(pykx.q('
        b
        -------
        0 0  0
        0 0  0
        2 8  32
        2 6  0
        0 0  0
        3 12 48
        ..
        '))
        ```
        """
        return self.call('prds', iterator=iterator)

    def sort(self, ascend=True, iterator=None):
        """
        Sort the values within a column in ascending or descending order

        Parameters:
            ascend: Should the data be sorted in ascending or descending order
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Sort the values in a column ascending

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').sort())
        pykx.Table(pykx.q('
        a
        --
        -1
        0
        1
        '))
        ```

        Sort the values in descending order:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': [1, -1, 0],
        ...     'b': [[-1, 2, 1], [0, 1, 2], [1, 2, 3]]
        ...     })
        >>> tab.select(kx.Column('a').sort(ascend=False))
        pykx.Table(pykx.q('
        a
        --
        1
        0
        -1
        '))
        ```
        """
        if ascend:
            fn = 'asc'
        else:
            fn = 'desc'
        return self.call(fn, iterator=iterator)

    def subtract(self, other, iterator=None, col_arg_ind=0, project_args=None):
        """
        Subtract from a column one of:

            - The values of another column
            - Python/Numpy list/value
            - A PyKX variable in q memory

        Note in it's most basic usage this is equivalent to

        ```python
        >>> kx.Column('x') - kx.Column('y')
        ```

        It is supplied as a named function to allow the use of iterators
            when adding elements.

        Parameters:
            other: The second column or variable (Python/q) to be used
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Subtract the values of two columns:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').subtract(kx.Column('b')))
        pykx.Table(pykx.q('
        a
        ---------
        -9.923397
        -9.782397
        -9.756748
        -9.625555
        ..
        '))
        ```

        Substract 3 from each element of a column.

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').subtract(3))
        pykx.Table(pykx.q('
        a
        ---------
        -2.978155
        -2.955834
        -2.937203
        -2.833157
        ..
        '))
        ```

        For each row in a column subtract 3 and 4 from the row independently.
            This makes use of each-left and each-right from q:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.q.asc(kx.random.random(100, 10.0)),
        ...     'b': kx.q.desc(kx.random.random(100, 10.0))
        ...     })
        >>> tab.select(kx.Column('a').subtract([3, 4], iterator='/:\:'))
        pykx.Table(pykx.q('
        a
        -------------------
        -2.978155 -3.978155
        -2.955834 -3.955834
        -2.937203 -3.937203
        -2.833157 -3.833157
        ..
        '))
        ```
        """
        return self.call('-', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def take(self, other, iterator=None, col_arg_ind=1, project_args=None):
        """
        Retrieve the first N rows from a column or N elements from items
            from a column using an iterator. Where N is specified by the other parameter.

        Parameters:
            other: An integer defining the number of elements to retrieve
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`.
            col_arg_ind: Determines the index within the multivariate function
                where the column parameter will be used. Default 0.
            project_args: The argument indices of a multivariate function which will be
                projected on the function before evocation with use of an iterator.

        Examples:

        Retrieve 3 rows from a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, kx.q.til(10)),
        ...     'b': kx.random.random([100, 3], kx.q.til(10))
        ...     })
        >>> tab.select(kx.Column('a').take(3).count())
        pykx.Table(pykx.q('
        a
        --
        10
        12
        24
        '))
        ```
        """
        return self.call('#', other, iterator=iterator, col_arg_ind=col_arg_ind,
                         project_args=project_args)

    def value(self, iterator=None):
        """
        When passed an EnumVector will return the corresponding SymbolVector

        Parameters:
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the variance of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.q('([] a:`sym?`a`b`c`a)')
        >>> tab.exec(kx.Column('a'))
        pykx.EnumVector(pykx.q('`sym$`a`b`c`a'))
        >>> tab.exec(kx.Column('a').value())
        pykx.SymbolVector(pykx.q('`a`b`c`a'))
        ```
        """
        return self.call('value', iterator=iterator)

    def variance(self, sample=False, iterator=None):
        """
        Calculate the variance or sample variance for items in a
            column or rows in a column

        Parameters:
            sample: Should calculation of variance return the
                sample variance (set True) or the variance (set False {default})
            iterator: What iterator to use when operating on the column
                for example, to execute per row, use `each`

        Examples:

        Calculate the variance of a column

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.exec(kx.Column('a').variance())
        pykx.FloatAtom(pykx.q('8.310944'))
        ```

        Calculate the sample sample deviation for each row in a column:

        ```python
        >>> import pykx as kx
        >>> tab = kx.Table(data={
        ...     'a': kx.random.random(100, 10.0),
        ...     'b': kx.random.random([100, 3], 10.0)
        ...     })
        >>> tab.select(kx.Column('b').variance(sample=True, iterator='each'))
        pykx.Table(pykx.q('
        b
        ---------
        6.023586
        29.48778
        6.318229
        0.1609426
        5.241295
        ..
        '))
        ```
        """
        fn = 'svar' if sample else 'var'
        return self.call(fn, iterator=iterator)


class QueryPhrase:
    """Special wrapper for a list which will be treated as a QueryPhrase.
        For use with the Query API
        """
    def __init__(self, phrase, names=None, are_trees=False):
        if isinstance(phrase, QueryPhrase):
            self._phrase = phrase._phrase
            self._names = phrase._names
            self._are_trees = phrase._are_trees
        elif isinstance(phrase, ParseTree):
            self._phrase = phrase._tree
        elif isinstance(phrase, Column):
            self._phrase = [phrase._value]
            self._names = [phrase._name]
            self._are_trees = [phrase._is_tree]
        elif isinstance(phrase, str):
            self._phrase = ParseTree(phrase).enlist()._tree
        elif isinstance(phrase, dict):
            self._phrase = list(phrase.values())
            self._names = list(phrase.keys())
            self._are_trees = [are_trees] * len(phrase)
        else:
            self._phrase = phrase
            self._names = names
            self._are_trees = are_trees

    def __repr__(self):
        preamble = f'pykx.{type(self).__name__}'
        return (f"{preamble}(names={self._names}, phrase={type(self._phrase)},"
                f"are_trees={self._are_trees})")

    def append(self, other):
        if isinstance(other, ParseTree):
            self._phrase.append(other._tree)
            self._names.append('')
            self._are_trees.append(False)
        elif isinstance(other, Column):
            self._phrase.append(other._value)
            self._names.append(other._name)
            self._are_trees.append(other._is_tree)
        elif isinstance(other, QueryPhrase):
            self._phrase.append(other._phrase)
            self._names.append(other._names)
            self._are_trees.append(other._are_trees)
        else:
            self._phrase.append(other)
            self._names.append('')
            self._are_trees.append(False)

    def extend(self, other):
        if isinstance(other, ParseTree):
            self._phrase.extend(other._tree)
            self._names.extend('')
            self._are_trees.extend(False)
        elif isinstance(other, Column):
            self._phrase.extend(other._value)
            self._names.extend(other._name)
            self._are_trees.extend(other._is_tree)
        elif isinstance(other, QueryPhrase):
            self._phrase.extend(other._phrase)
            self._names.extend(other._names)
            self._are_trees.extend(other._are_trees)
        else:
            self._phrase.extend(other)
            self._names.extend('')
            self._are_trees.extend(False)

    def to_dict(self):
        return dict(map(lambda i, j: (i, j), self._names, self._phrase))

    def __and__(self, other):
        cpy = copy.deepcopy(self)
        if isinstance(other, Column):
            cpy.append(other)
        elif isinstance(other, QueryPhrase):
            cpy.extend(other)
        else:
            raise TypeError(
                f"Supplied object type '{type(other)}' cannot `&` off a `pykx.QueryPhrase`.")
        return cpy


def _internal_k_list_wrapper(addr: int, incref: bool):
    res = list(_wrappers._factory(addr, incref))
    for i in range(len(res)):
        if type(res[i]) is Foreign:
            res[i] = _wrappers.from_foreign_to_pyobject(res[i])
    return tuple(res)


def _internal_is_k_dict(x):
    return 1 if isinstance(x, Dictionary) else 0


def _internal_k_dict_to_py(addr: int):
    return _wrappers._factory(addr, True).py()


vector_to_atom = {
    BooleanVector:   BooleanAtom,
    GUIDVector:      GUIDAtom,
    ByteVector:      ByteAtom,
    ShortVector:     ShortAtom,
    IntVector:       IntAtom,
    LongVector:      LongAtom,
    RealVector:      RealAtom,
    FloatVector:     FloatAtom,
    CharVector:      CharAtom,
    SymbolVector:    SymbolAtom,
    TimestampVector: TimestampAtom,
    MonthVector:     MonthAtom,
    DateVector:      DateAtom,
    DatetimeVector:  DatetimeAtom,
    TimespanVector:  TimespanAtom,
    MinuteVector:    MinuteAtom,
    SecondVector:    SecondAtom,
    TimeVector:      TimeAtom,
    EnumVector:      EnumAtom,
}


atom_to_vector = {v: k for k, v in vector_to_atom.items()}


# tags for types that need special handling
_k_table_type = object()
_k_dictionary_type = object()
_k_unary_primitive = object()


type_number_to_pykx_k_type = {
    -128: QError,
    -20: EnumAtom,
    -19: TimeAtom,
    -18: SecondAtom,
    -17: MinuteAtom,
    -16: TimespanAtom,
    -15: DatetimeAtom,
    -14: DateAtom,
    -13: MonthAtom,
    -12: TimestampAtom,
    -11: SymbolAtom,
    -10: CharAtom,
     -9: FloatAtom,                                                     # noqa
     -8: RealAtom,                                                      # noqa
     -7: LongAtom,                                                      # noqa
     -6: IntAtom,                                                       # noqa
     -5: ShortAtom,                                                     # noqa
     -4: ByteAtom,                                                      # noqa
     -2: GUIDAtom,                                                      # noqa
     -1: BooleanAtom,                                                   # noqa
      0: List,                                                          # noqa
      1: BooleanVector,                                                 # noqa
      2: GUIDVector,                                                    # noqa
      4: ByteVector,                                                    # noqa
      5: ShortVector,                                                   # noqa
      6: IntVector,                                                     # noqa
      7: LongVector,                                                    # noqa
      8: RealVector,                                                    # noqa
      9: FloatVector,                                                   # noqa
     10: CharVector,                                                    # noqa
     11: SymbolVector,                                                  # noqa
     12: TimestampVector,                                               # noqa
     13: MonthVector,                                                   # noqa
     14: DateVector,                                                    # noqa
     15: DatetimeVector,                                                # noqa
     16: TimespanVector,                                                # noqa
     17: MinuteVector,                                                  # noqa
     18: SecondVector,                                                  # noqa
     19: TimeVector,                                                    # noqa
     20: EnumVector,                                                    # noqa
     77: Anymap,                                                        # noqa
     98: _k_table_type,                                                 # noqa
     99: _k_dictionary_type,                                            # noqa
    100: Lambda,
    101: _k_unary_primitive,
    102: Operator,
    103: Iterator,
    104: Projection,
    105: Composition,
    106: Each,
    107: Over,
    108: Scan,
    109: EachPrior,
    110: EachRight,
    111: EachLeft,
    112: Foreign
}


__all__ = [
    'Anymap',
    'AppliedIterator',
    'Atom',
    'BooleanAtom',
    'BooleanVector',
    'ByteAtom',
    'ByteVector',
    'Column',
    'CharAtom',
    'CharVector',
    'Collection',
    'Composition',
    'DateAtom',
    'DateVector',
    'DatetimeAtom',
    'DatetimeVector',
    'Dictionary',
    'Each',
    'EachLeft',
    'EachPrior',
    'EachRight',
    'EnumAtom',
    'EnumVector',
    'FloatAtom',
    'FloatVector',
    'Foreign',
    'Function',
    'GUIDAtom',
    'GUIDVector',
    'Identity',
    'IntAtom',
    'IntVector',
    'IntegralNumericAtom',
    'IntegralNumericVector',
    'Iterator',
    'K',
    'KeyedTable',
    'Lambda',
    'List',
    'LongAtom',
    'LongVector',
    'Mapping',
    'MinuteAtom',
    'MinuteVector',
    'MonthAtom',
    'MonthVector',
    'NonIntegralNumericAtom',
    'NonIntegralNumericVector',
    'NumericAtom',
    'NumericVector',
    'Operator',
    'Over',
    'ParseTree',
    'PartitionedTable',
    'Projection',
    'ProjectionNull',
    'RealAtom',
    'RealVector',
    'Scan',
    'SecondAtom',
    'SecondVector',
    'ShortAtom',
    'ShortVector',
    'SplayedTable',
    'SymbolAtom',
    'SymbolVector',
    'SymbolicFunction',
    'Table',
    'TemporalAtom',
    'TemporalFixedAtom',
    'TemporalFixedVector',
    'TemporalSpanAtom',
    'TemporalSpanVector',
    'TemporalVector',
    'TimeAtom',
    'TimeVector',
    'TimespanAtom',
    'TimespanVector',
    'TimestampAtom',
    'TimestampVector',
    'UnaryPrimative',
    'UnaryPrimitive',
    'Variable',
    'Vector',
    'QueryPhrase',
    '_internal_k_list_wrapper',
    '_internal_is_k_dict',
    '_internal_k_dict_to_py',
]


def __dir__():
    return __all__

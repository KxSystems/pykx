"""Wrappers for q data structures, with conversion functions to Python/Numpy/Pandas/Arrow.

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
   e.g. [`pykx.SecondAtom`][], one can override the defaults.

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
from datetime import datetime, timedelta
from inspect import signature
import math
from numbers import Integral, Number, Real
import operator
from uuid import UUID
from typing import Any, Optional, Tuple, Union
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import pytz

from . import _wrappers
from ._pyarrow import pyarrow as pa
from .config import k_gc, licensed, pandas_2
from .core import keval as _keval
from .constants import INF_INT16, INF_INT32, INF_INT64, NULL_INT16, NULL_INT32, NULL_INT64
from .exceptions import LicenseException, PyArrowUnavailable, PyKXException, QError
from .util import cached_property, classproperty, df_from_arrays, slice_to_range


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


def _key_preprocess(key, n, slice=False):
    if key is not None:
        if key < 0:
            key = n + key
        if (key >= n or key < 0) and not slice:
            raise IndexError('index out of range')
        elif slice:
            if key < 0:
                key = 0
            if key > n:
                key = n
    return(key)


def _rich_convert(x: 'K', stdlib: bool = True):
    if stdlib:
        return x.py(stdlib=stdlib)
    if isinstance(x, Mapping):
        return x.pd()
    return x.np()


def _null_gen(x):
    def null():
        """Generate the pykx null representation associated with an atom type

        Examples:

        ```python
        >>> import pykx as kx
        >>> kx.TimeAtom.null
        pykx.TimeAtom(pykx.q('0Nt'))
        >>> kx.GUIDAtom.null
        pykx.GUIDAtom(pykx.q('00000000-0000-0000-0000-000000000000'))
        ```
        """
        if licensed and x is not None:
            return q(f'{x}')
        elif not licensed:
            raise QError('Generation of null data not supported in unlicensed mode')
        else:
            raise NotImplementedError('Retrieval of null values not supported for this type')
    return null


def _inf_gen(x):
    def inf(neg=False):
        """Generate the pykx infinite value associated with an atom type

        Parameters:
            neg: Should the return value produce the negative infinity value

        Examples:

        ```python
        >>> import pykx as kx
        >>> kx.TimeAtom.inf
        pykx.TimeAtom(pykx.q('0Wt'))
        >>> kx.TimeAtom.inf(neg=True)
        pykx.TimeAtom(pykx.q('-0Wt'))
        ```
        """
        if licensed and x is not None:
            return q('{[p]$[p;neg;]'+f'{x}'+'}', neg)
        elif not licensed:
            raise QError('Generation of infinite data not supported in unlicensed mode')
        else:
            raise NotImplementedError('Retrieval of infinite values not supported for this type')
    return inf


# HACK: This gets overwritten by the toq module to avoid a circular import error.
def toq(*args, **kwargs): # nocov
    raise NotImplementedError


class K:
    """Base type for all q objects.

    Parameters:
        x (Any): An object that will be converted into a `pykx.K` object via [`pykx.toq`][].
    """
    # TODO: `cast` should be set to False at the next major release (KXI-12945)
    def __new__(cls, x: Any, *, cast: bool = None, **kwargs):
        return toq(x, ktype=None if cls is K else cls, cast=cast) # TODO: 'strict' and 'cast' flags

    # TODO: `cast` should be set to False at the next major release (KXI-12945)
    def __init__(self, x: Any, *, cast: bool = None, **kwargs): # Signature must match `__new__`
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


class Atom(K):
    """Base type for all q atoms, including singular basic values, and functions.

    See Also:
        [`pykx.Collection`][]
    """
    @property
    def is_null(self) -> bool:
        return q('null', self).py()

    @property
    def is_inf(self) -> bool:
        if self.t in {-1, -2, -4, -10, -11}:
            return False
        try:
            type_char = ' bg xhijefcspmdznuvts'[abs(self.t)]
        except IndexError:
            return False
        return q(f'{{any -0W 0W{type_char}~\\:x}}')(self).py()

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


class EnumAtom(Atom):
    """Wrapper for q enum atoms."""
    t = -20

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
            epoch_offset = 0 if self.is_inf else self._epoch_offset
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

    # TODO: `cast` should be set to False at the next major release (KXI-12945)
    def __new__(cls, x: Any, *, cast: bool = None, **kwargs):
        if (type(x) == str) and x == 'now': # noqa: E721
            if licensed:
                return q('.z.T')
        return toq(x, ktype=None if cls is K else cls, cast=cast) # TODO: 'strict' and 'cast' flags

    def _prototype(self=None):
        return TimeAtom(np.timedelta64(59789214, 'ms'))

    @classproperty
    def null(cls): # noqa: B902
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls):  # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32


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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32


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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32


class TimespanAtom(TemporalSpanAtom):
    """Wrapper for q timespan atoms."""
    t = -16
    _null = '0Nn'
    _inf = '0Wn'
    _np_type = 'ns'
    _np_dtype = 'timedelta64[ns]'

    def _prototype(self=None):
        return TimespanAtom(np.timedelta64(3796312051664551936, 'ns'))

    @classproperty
    def null(cls): # noqa: B902
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return _wrappers.k_j(self) == NULL_INT64

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_j(self)) == INF_INT64


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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    def __init__(self, *args, **kwargs):
        warnings.warn('The q datetime type is deprecated', DeprecationWarning)
        super().__init__(*args, **kwargs)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        if raw:
            return _wrappers.k_f(self)
        raise TypeError('The q datetime type is deprecated, and can only be accessed with '
                        'the keyword argument `raw=True` in Python or `.pykx.toRaw` in q')

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return _wrappers.k_f(self)
        raise TypeError('The q datetime type is deprecated, and can only be accessed with '
                        'the keyword argument `raw=True` in Python or `.pykx.toRaw` in q')


class DateAtom(TemporalFixedAtom):
    """Wrapper for q date atoms."""
    t = -14
    _np_type = 'D'
    _null = '0Nd'
    _inf = '0Wd'
    _epoch_offset = DATE_OFFSET
    _np_dtype = 'datetime64[D]'

    # TODO: `cast` should be set to False at the next major release (KXI-12945)
    def __new__(cls, x: Any, *, cast: bool = None, **kwargs):
        if (type(x) == str) and x == 'today': # noqa: E721
            if licensed:
                return q('.z.D')
        return toq(x, ktype=None if cls is K else cls, cast=cast) # TODO: 'strict' and 'cast' flags

    def _prototype(self=None):
        return DateAtom(np.datetime64('1972-05-31', 'D'))

    @classproperty
    def null(cls): # noqa: B902
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32


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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32


class TimestampAtom(TemporalFixedAtom):
    """Wrapper for q timestamp atoms."""
    t = -12
    _null = '0Np'
    _inf = '0Wp'
    _np_type = 'ns'
    _epoch_offset = TIMESTAMP_OFFSET
    _np_dtype = 'datetime64[ns]'

    # TODO: `cast` should be set to False at the next major release (KXI-12945)
    def __new__(cls, x: Any, *, cast: bool = None, **kwargs):
        if (type(x) == str) and x == 'now': # noqa: E721
            if licensed:
                return q('.z.P')
        return toq(x, ktype=None if cls is K else cls, cast=cast) # TODO: 'strict' and 'cast' flags

    def _prototype(self=None):
        return TimestampAtom(datetime(2150, 10, 22, 20, 31, 15, 70713))

    @classproperty
    def null(cls): # noqa: B902
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return _wrappers.k_j(self) == NULL_INT64

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_j(self)) == INF_INT64

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
        [`pykx.CharVector`][]
    """
    t = -11
    _null = '`'
    _inf = None
    _np_dtype = None

    def _prototype(self=None):# noqa
        return SymbolAtom('')

    @classproperty
    def null(cls): # noqa: B902
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return str(self) == ''

    @property
    def is_inf(self) -> bool:
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

    def _prototype(self=None):
        return CharAtom(b' ')

    @classproperty
    def null(cls): # noqa: B902
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return 32 == _wrappers.k_g(self)

    @property
    def is_inf(self) -> bool:
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
        return q('first', self)

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return _wrappers.k_g(self)
        return bytes(chr(_wrappers.k_g(self)), 'utf-8')


class NumericAtom(Atom):
    """Base type for all q numeric atoms."""
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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

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
        if not raw and (self.is_null or self.is_inf):
            # By returning the wrapper around the q null/inf when a Python object is requested, we
            # propagate q's behavior around them - for better and for worse. Notably this ensures
            # symmetric conversions.
            return self
        return default

    def _np_null_or_inf(self, default, raw: bool):
        if not raw:
            if self.is_null:
                raise PyKXException('Numpy does not support null atomic integral values')
            if self.is_inf:
                raise PyKXException('Numpy does not support infinite atomic integral values')
        return default


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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return _wrappers.k_j(self) == NULL_INT64

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_j(self)) == INF_INT64

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return self._py_null_or_inf(_wrappers.k_j(self), raw)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return self._np_null_or_inf(np.int64(_wrappers.k_j(self)), raw)


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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return _wrappers.k_i(self) == NULL_INT32

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_i(self)) == INF_INT32

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return self._py_null_or_inf(_wrappers.k_i(self), raw)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return self._np_null_or_inf(np.int32(_wrappers.k_i(self)), raw)


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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return _wrappers.k_h(self) == NULL_INT16

    @property
    def is_inf(self) -> bool:
        return abs(_wrappers.k_h(self)) == INF_INT16

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        return self._py_null_or_inf(_wrappers.k_h(self), raw)

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return self._np_null_or_inf(np.int16(_wrappers.k_h(self)), raw)


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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return False

    @property
    def is_inf(self) -> bool:
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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return self.py(raw=True) == 0j

    @property
    def is_inf(self) -> bool:
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
        return _null_gen(cls._null)()

    @classproperty
    def inf(cls): # noqa: B902
        return _inf_gen(cls._inf)()

    @property
    def is_null(self) -> bool:
        return False

    @property
    def is_inf(self) -> bool:
        return False

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        if raw:
            return _wrappers.k_g(self)
        return bool(_wrappers.k_g(self))


class Collection(K):
    """Base type for all q collections (i.e. non-atoms), including vectors, and mappings.

    See Also:
        [`pykx.Collection`][]
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
            type_char = ' bg xhijefcspmdznuvts'[self.t]
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
        return pa.array(self.np(raw=raw, has_nulls=has_nulls))

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
        if not isinstance(self, List):
            if not q('{(0>type[y])& type[x]=abs type y}', self, data):
                raise QError(f'Appending data of type: {type(K(data))} '
                             f'to vector of type: {type(self)} not supported')
        append_vec = q('{[orig;app]orig,$[0<type app;enlist;]app}', self, data)
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
        for i in slice_to_range(slice(start, end), _wrappers.k_n(self)):
            if self[i] == x:
                return i
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
        warnings.warn('Warning: Attempting to call numpy __array_function__ on a '
                      f'PyKX Vector type. __array_function__: {func}. Support for this method '
                      'is on a best effort basis.')
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


class List(Vector):
    """Wrapper for q lists, which are vectors of K objects of any type.

    Note: The memory layout of a q list is special.
        All other vector types (see: subclasses of [`pykx.Vector`][]) are structured in-memory as a
        K object which contains metadata, followed immediately by the data in the vector. By
        contrast, q lists are a a vector of pointers to K objects, so they are structured in-memory
        as a K object containing metadata, followed immediately by pointers. As a result, the base
        data "contained" by the list is located elsewhere in memory. This has performance and
        ownership implications in q, which carry over to PyKX.
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
        return [_rich_convert(x, stdlib) for x in self]

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        """Provides a Numpy representation of the list."""
        return _wrappers.list_np(self, False, has_nulls)


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
        return [x if x.is_null else x.py() for x in self]

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
        raise NotImplementedError

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
        raise NotImplementedError


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
        [`pykx.SymbolAtom`][]
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

    def py(self, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
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
        return [x if x.is_null else x.py() for x in self]

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
        if raw:
            has_nulls = False
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
            converted_vector[i]=TimestampAtom(None)
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
        warnings.warn('The q datetime type is deprecated', DeprecationWarning)
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
    """Wrapper for q enum vectors."""
    t = 20

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
        - [`pykx.SplayedTable`][]
        - [`pykx.PartitionedTable`][]
    """
    t = 98

    # TODO: `cast` should be set to False at the next major release (KXI-12945)
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
            _map[k] = v._prototype() if (type(v) == type or type(v) == ABCMeta) else v
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
        if raw_guids:
            warnings.warn("Keyword 'raw_guids' is deprecated", DeprecationWarning)
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
        return pa.Table.from_pandas(self.pd(raw=raw, has_nulls=has_nulls, raw_guids=True))

    def np(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        return self.pd(raw=raw, has_nulls=has_nulls).to_records(index=False)

    def insert(
        self,
        row: Union[list, List],
        match_schema: bool = False,
        test_insert: bool = False,
        replace_self: bool = True,
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
            replace_self: `Deprecated` please use `inplace` keyword.
                Causes the underlying Table python object to update itself with the
                resulting Table after the insert.
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
        if not replace_self:
            warnings.warn("Keyword 'replace_self' is deprecated please use 'inplace'",
                          DeprecationWarning)
        if replace_self and inplace:
            self.__dict__.update(res.__dict__)
        q('delete itab from `.pykx.i')
        return res

    def upsert(
        self,
        row: Union[list, List],
        match_schema: bool = False,
        test_insert: bool = False,
        replace_self: bool = True,
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
            replace_self: `Deprecated` please use `inplace` keyword.
                Causes the underlying Table python object to update itself with the
                resulting Table after the upsert.
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
        if not replace_self:
            warnings.warn("Keyword 'replace_self' is deprecated please use 'inplace'",
                          DeprecationWarning)
        if replace_self and inplace:
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
        >>> windows = kx.q('{-2 1+\:x}', trades['time'])
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
        qtab = q('''{[c;t]
                 n:count t;
                 cls:$[c[1]<count cls:cols t;((c[1]-1)sublist cls),last cls;cls];
                 h:.j.j $[c[1]<count cols t;
                 {[t]c:count cls:cols t;#[(c-1) sublist cls;t],'
                 (flip(enlist `$"...")!enlist count[t]#enlist "..."),'#[-1 sublist cls;t]};(::)]
                 $[c[0]<n;{(-2 _ x),(enlist {"..."} each flip 0#x),(-1 sublist x)};::]
                 {flip {{$[11h~type x;
                  string x;
                 (enlist 11h)~distinct type each x;
                 sv[" "]each string x;
                 (enlist 20h)~distinct type each x;
                 sv[" "]each string @[{value each x};x;x];
                 .Q.s1 each x]}$[20h~type x;@[value;x;x];x]}
                  each flip x}
                 {i:til c:count y;if[c;i[c-1]:x];
                 (flip enlist[`pykxTableIndex]!enlist(i)),'y}[$[n=c[0];n-2;-1+n]]
                 ?[t;enlist(<;`i;c[0]-1);0b;{x!x}cls],
                $[n>c[0];?[t;enlist(=;`i;(last;`i));0b;{x!x}cls];()];
                 h
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
        ht = q('''{[c;t;h]
               $[c[0]<n:count t;
                 h,"\n<p>",{reverse "," sv 3 cut reverse string x}[n]," rows × ",
               {reverse "," sv 3 cut reverse string x}[count cols t]," columns</p>";h]
               }''', console, self, ht).py().decode("utf-8")
        return ht


class SplayedTable(Table):
    """Wrapper for q splayed tables."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._splay_dir = _wrappers.k_from_addr(SymbolAtom, self._values._addr, True)
        self._values = None

    def __getitem__(self, key):
        raise NotImplementedError

    def __reduce__(self):
        raise TypeError('Unable to serialize pykx.SplayedTable objects')

    def any(self):
        raise NotImplementedError

    def all(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    @property
    def flip(self):
        raise NotImplementedError

    def pd(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        raise NotImplementedError

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        raise NotImplementedError

    def _repr_html_(self):
        if not licensed:
            return self.__repr__()
        console = q.system.console_size.py()
        qtab = q('''{[c;t]
              n:count t;
              cls:$[c[1]<count cls:cols t;((c[1]-1)sublist cls),last cls;cls];
              h:.j.j $[c[1]<count cols t;
              {[t]c:count cls:cols t;#[(c-1) sublist cls;t],'
              (flip(enlist `$"...")!enlist count[t]#enlist "..."),'#[-1 sublist cls;t]};(::)]
              $[c[0]<n;{(-2 _ x),(enlist {"..."} each flip 0#x),(-1 sublist x)};::]
              {flip {{$[11h~type x;
                  string x;
                 (enlist 11h)~distinct type each x;
                 sv[" "]each string x;
                 (enlist 20h)~distinct type each x;
                 sv[" "]each string @[{value each x};x;x];
                 .Q.s1 each x]}$[20h~type x;@[value;x;x];x]} each flip x}
                 {i:til c:count y;if[c;i[c-1]:x];
                 (flip enlist[`pykxTableIndex]!enlist(i)),'y}[$[n=c[0];n-2;-1+n]]
                 ?[t;enlist(<;`i;c[0]-1);0b;{x!x}cls],
                $[n>c[0];?[t;enlist(=;`i;(last;`i));0b;{x!x}cls];()];
              h
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
        ht = q('''{[c;t;h]
               $[c[0]<n:count t;
                 h,"\n<p>",{reverse "," sv 3 cut reverse string x}[n]," rows × ",
               {reverse "," sv 3 cut reverse string x}[count cols t]," columns</p>";h]
               }''', console, self, ht).py().decode("utf-8")
        return ht


class PartitionedTable(SplayedTable):
    """Wrapper for q partitioned tables."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        raise NotImplementedError

    def __reduce__(self):
        raise TypeError('Unable to serialize pykx.PartitionedTable objects')

    def items(self):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    @property
    def flip(self):
        raise NotImplementedError

    def pd(self, *, raw: bool = False, has_nulls: Optional[bool] = None):
        raise NotImplementedError

    def py(self, *, raw: bool = False, has_nulls: Optional[bool] = None, stdlib: bool = True):
        raise NotImplementedError

    def _repr_html_(self):
        if not licensed:
            return self.__repr__()
        console = q.system.console_size.py()
        qtab = q('''{[c;t]
                t:value flip t;
                if[not count .Q.pn t;.Q.cn get t];
                ps:sums .Q.pn t;n:last ps;
                cls:$[c[1]<count cls:cols t;((c[1]-1)sublist cls),last cls;cls];
                if[c[0]>=n;:.j.j $[c[1]<count cols t;
                 {[t]c:count cls:cols t;#[(c-1) sublist cls;t],'
                 (flip(enlist `$"...")!enlist count[t]#enlist "..."),'#[-1 sublist cls;t]};(::)]
                {flip {{$[11h~type x;
                  string x;
                 (enlist 11h)~distinct type each x;
                 sv[" "]each string x;
                 (enlist 20h)~distinct type each x;
                 sv[" "]each string @[{value each x};x;x];
                 .Q.s1 each x]}$[20h~type x;@[value;x;x];x]}each flip x}
                 {i:til c:count y;if[c;i[c-1]:x];(flip enlist[`pykxTableIndex]!enlist(i)),'y}[-1+n]
                 ?[t;();0b;{x!x}cls]];
                fp:first where ps>=c[0];r:();
                if[fp~0;r:?[t;((=;.Q.pf;first .Q.pv);(<;`i;c[0]-1));0b;{x!x}cls]];
                if[fp>0;
                 r:?[t;enlist(in;.Q.pf;fp#.Q.pv);0b;{x!x}cls];
                 r:r,?[t;((=;.Q.pf;.Q.pv fp);(<;`i;-1+c[0]-ps[fp-1]));0b;{x!x}cls];
                 ];
                if[c[0]<n;
                 r:r,?[t;((=;.Q.pf;last .Q.pv{last where not x=0}.Q.pn t);
                 (=;`i;(last;`i)));0b;{x!x}cls]];
                r:{i:til c:count y;if[c;i[c-1]:x];
                 (flip enlist[`pykxTableIndex]!enlist(i)),'y}[$[n=c[0];n-2;-1+n]]r;
                h:.j.j $[c[1]<count cols t;
                 {[t]c:count cls:cols t;#[(c-1) sublist cls;t],'
                 (flip(enlist `$"...")!enlist count[t]#enlist "..."),'#[-1 sublist cls;t]};(::)]
                 {(-2 _ x),(enlist {"..."} each flip 0#x),(-1 sublist x)}
                {flip {{$[11h~type x;
                  string x;
                 (enlist 11h)~distinct type each x;
                 sv[" "]each string x;
                 (enlist 20h)~distinct type each x;
                 sv[" "]each string @[{value each x};x;x];
                 .Q.s1 each x]}$[20h~type x;@[value;x;x];x]}
                  each flip x}  r;
                h
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
        ht = q('''{[c;t;h]
               $[c[0]<n:count t;
                 h,"\n<p>",{reverse "," sv 3 cut reverse string x}[n]," rows × ",
               {reverse "," sv 3 cut reverse string x}[count cols t]," columns</p>";h]
               }''', console, self, ht).py().decode("utf-8")
        return ht


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
            raise NotImplementedError("pykx.Dictionary objects do not support tuple key assignment")
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
                 $[c[0]<n;
                 {(-1 _ x),(enlist {"..."} each flip 0#x)};::]
                 {flip {{$[11h~type x;
                  string x;
                 (enlist 11h)~distinct type each x;
                 sv[" "]each string x;
                 (enlist 20h)~distinct type each x;
                 sv[" "]each string @[{value each x};x;x];
                 .Q.s1 each x]}$[20h~type x;@[value;x;x];x]}
                  each flip x} ?[t;enlist(<;`i;c[0]);0b;{x!x}cls];
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
    # TODO: `cast` should be set to False at the next major release (KXI-12945)
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
        yield from zip(*self._keys._values)

    def keys(self):
        return list(self)

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
        raise NotImplementedError

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
        replace_self: bool = True,
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
            replace_self: `Deprecated` please use `inplace` keyword.
                Causes the underlying Table python object to update itself with the
                resulting Table after the insert.
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
        if not replace_self:
            warnings.warn("Keyword 'replace_self' is deprecated please use 'inplace'",
                          DeprecationWarning)
        if replace_self and inplace:
            self.__dict__.update(res.__dict__)
        q('delete itab from `.pykx.i')
        return res

    def upsert(
        self,
        row: Union[list, List],
        match_schema: bool = False,
        test_insert: bool = False,
        replace_self: bool = True,
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
            replace_self: `Deprecated` please use `inplace` keyword.
                Causes the underlying Table python object to update itself with the
                resulting Table after the insert.
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
        if not replace_self:
            warnings.warn("Keyword 'replace_self' is deprecated please use 'inplace'",
                          DeprecationWarning)
        if replace_self and inplace:
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

    def _repr_html_(self):
        if not licensed:
            return self.__repr__()
        keys = q('{cols key x}', self).py()
        console = q.system.console_size.py()
        qtab=q('''{[c;t]
               n:count t:t;
               cls:$[c[1]<count cls:cols t;((c[1]-1)sublist cls),last cls;cls];
               h:.j.j $[c[1]<count cols t;
                 {[t]c:count cls:cols t;#[(c-1) sublist cls;t],'
                 (flip(enlist `$"...")!enlist count[t]#enlist "..."),'#[-1 sublist cls;t]};(::)]
               $[c[0]<n;{(-2 _ x),(enlist {"..."} each flip 0#x),(-1 sublist x)};::]
               {flip {{$[11h~type x;
                  string x;
                 (enlist 11h)~distinct type each x;
                 sv[" "]each string x;
                 (enlist 20h)~distinct type each x;
                 sv[" "]each string @[{value each x};x;x];
                 .Q.s1 each x]}$[20h~type x;@[value;x;x];x]}
                  each flip x}
                 ?[t;enlist(<;`i;c[0]-1);0b;{x!x}cls],
                $[n>c[0];?[t;enlist(=;`i;(last;`i));0b;{x!x}cls];()];
               h
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
        ht = q('''{[c;t;h]
               $[c[0]<n:count t;
                 h,"\n<p>",{reverse "," sv 3 cut reverse string x}[n]," rows × ",
               {reverse "," sv 3 cut reverse string x}[count cols t]," columns</p>";h]
               }''', console, self, ht).py().decode("utf-8")
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
        keys = q.keys(self.tab).py()
        if isinstance(item, list):
            keys.extend(item)
        else:
            keys.append(item)
        return GroupbyTable(
            q(f'{len(q.keys(self.tab))}!', self.tab[keys]),
            True,
            False,
            as_vector=item
        )


GTable_init(GroupbyTable)


class Function(Atom):
    """Base type for all q functions.

    `Function` objects can be called as if they were Python functions. All provided arguments will
    be converted to q using [`pykx.toq`][], and the execution of the function will happen in q.

    `...` can be used to omit an argument, resulting in a [function projection][pykx.Projection].

    [Refer to chapter 6 of Q for Mortals](https://code.kx.com/q4m3/6_Functions/) for more
    information about q functions.
    """
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


class Lambda(Function):
    """Wrapper for q lambda functions.

    Lambda's are the most basic kind of function in q. They can take between 0 and 8 parameters
    (inclusive), which all must be q objects themselves. If the provided parameters are not
    [`pykx.K`][] objects, they will be converted into them using [`pykx.toq`][].

    Unlike other [`pykx.Function`][] subclasses, `Lambda` objects can be called with keyword
    arguments, using the names of the parameters from q.
    """
    t = 100

    @property
    def __name__(self):
        return 'pykx.Lambda'

    @cached_property
    def params(self):
        # Strip "PyKXParam" from all param names if it is a prefix for all
        return tuple(
            str(x) for x in q('k){x:.:x;$[min (x:x[1]) like "PyKXParam*"; `$9_\'$x; x]}', self)
        )


class UnaryPrimitive(Function):
    """Wrapper for q unary primitive functions, including `::`, and other built-ins.

    Unary primitives are a class of built-in q functions which take exactly one parameter. New ones
    cannot be defined by a user through any normal means.

    See Also:
        [`pykx.Identity`][]
    """
    t = 101

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
    reference page: https://code.kx.com/q/ref/#operators
    """
    t = 102

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TypeError('Cannot use kwargs on an operator')
        return super().__call__(*args, **kwargs)


class Iterator(Function):
    """Wrappers for q iterator functions.

    Iterators include the mapping iterators (`'`, `':`, `/:`, and `\\:`), and the accumulating
    iterators (`/`, and `\\`). They are documented on the q reference page:
    https://code.kx.com/q/ref/#iterators
    """
    t = 103

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TypeError('Cannot use kwargs on an iterator')
        return super().__call__(*args, **kwargs)


class Projection(Function):
    """Wrapper for q function projections.

    Similar to [`functools.partial`][], q functions can have some of their parameters fixed in
    advance, resulting in a new function, which is a projection. When this projection is called,
    the fixed parameters are no longer required, and cannot be provided.

    If the original function had `n` parameters, and it had `m` of them provided, the result would
    be a function (projection) that has `m` parameters.

    In PyKX, the special Python singleton `...` is used to represent
    [projection null][`pykx.ProjectionNull`]
    """
    t = 104

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
    themselves are of the type [`pykx.Iterator`][], but when applied to a function a new type
    (which is a subclass of `AppliedIterator`) is created depending on what iterator was used.
    """
    @property
    def __name__(self):
        return 'pykx.AppliedIterator'

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


class Over(AppliedIterator):
    """Wrapper for functions with the 'over' iterator applied to them."""
    t = 107


class Scan(AppliedIterator):
    """Wrapper for functions with the 'scan' iterator applied to them."""
    t = 108


class EachPrior(AppliedIterator):
    """Wrapper for functions with the 'each-prior' iterator applied to them."""
    t = 109


class EachRight(AppliedIterator):
    """Wrapper for functions with the 'each-right' iterator applied to them."""
    t = 110


class EachLeft(AppliedIterator):
    """Wrapper for functions with the 'each-left' iterator applied to them."""
    t = 111


class Foreign(Atom):
    """Wrapper for foreign objects, i.e. wrapped pointers to regions outside of q memory."""
    t = 112

    def __reduce__(self):
        raise TypeError('Unable to serialize pykx.Foreign objects')

    def py(self, stdlib=None):
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
    'Vector',
    '_internal_k_list_wrapper',
    '_internal_is_k_dict',
    '_internal_k_dict_to_py',
]


def __dir__():
    return __all__

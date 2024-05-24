"""Converts a Python object to a `pykx.K` object.

`pykx.toq` can be called to automatically select the appropriate `pykx.toq.from_*` function
based on the type of the provided Python object. Refer to the documentation for each of the
`pykx.toq.from_*` functions for more details about how each conversion works.

Note: `pykx.toq.from_*` methods should not be used directly
    `pykx.toq` or one of the `pykx.K` subclassses should instead be used to convert a value into a
    `pykx.K` object. The `pykx.toq.from_*` functions are automatically called by these higher-level
    interfaces. They are documented here to explain what conversions from Python to q are supported,
    and how they are performed.

Instead of calling e.g. `pykx.toq('qwerty', ktype=pykx.CharVector)`, one can instantiate the
desired type directly like so: `pykx.CharVector('qwerty')`.

Examples:

```python
>>> import pykx as kx
>>> import pandas as pd
>>> kx.toq('grok')
pykx.SymbolAtom(pykx.q('`grok'))
>>> kx.toq('grok', ktype=kx.CharVector)
pykx.CharVector(pykx.q('"grok"'))
>>> df = pd.DataFrame.from_dict({'x': [1, 2], 'y': ['a', 'b']})
>>> kx.toq(df)
pykx.Table(pykx.q('
x y
---
1 a
2 b
'))
>>> kx.toq(df, ktype={'x': kx.CharVector})
pykx.Table(pykx.q('
x    y
------
,"1" a
,"2" b
'))
```

**Parameters:**

+---------------+---------------------------+---------------------------------------+-------------+
| **Name**      | **Type**                  | **Description**                       | **Default** |
+===============+===========================+=======================================+=============+
| `x`           | `Any`                     | A Python object which is to be        | *required*  |
|               |                           | converted into a `pykx.K` object.     |             |
+---------------+---------------------------+---------------------------------------+-------------+
| `ktype`       | `Optional[Union[pykx.K,`  | Desired `pykx.K` subclass (or type    | `None`      |
|               |     `int, dict]]`         | number) for the returned value. If    |             |
|               |                           | `None`, the type is inferred from     |             |
|               |                           | `x`. If specified as a dictionary     |             |
|               |                           | will convert tabular data based on    |             |
|               |                           | mapping of column name to type. Note  |             |
|               |                           | that dictionary based conversion is   |             |
|               |                           | only supported when operating in      |             |
|               |                           | licensed mode.                        |             |
+---------------+---------------------------+---------------------------------------+-------------+
| `cast`        | `bool`                    | Cast the input Python object to the   | `False`     |
|               |                           | closest conforming Python type before |             |
|               |                           | converting to a `pykx.K` object.      |             |
+---------------+---------------------------+---------------------------------------+-------------+
| `handle_nulls | `bool`                    | Convert `pd.NaT` to corresponding q   | `False`     |
|               |                           | null values in Pandas dataframes and  |             |
|               |                           | Numpy arrays.                         |             |
+---------------+---------------------------+---------------------------------------+-------------+

**Returns:**

**Type** | **Description**
-------- | -----------------------------------------------------------
`pykx.K` | The provided Python object as an analogous `pykx.K` object.
"""

from libc.stddef cimport *
from libc.stdint cimport *
from libc.string cimport memcpy
from cpython.ref cimport Py_INCREF

cimport numpy as cnp

from pykx cimport core
from pykx._wrappers cimport factory

import datetime
from ctypes import CDLL
from inspect import signature
import math
import os
from pathlib import Path
import pytz
import sys
import re
from types import ModuleType
from typing import Any, Callable, Optional, Union
from uuid import UUID, uuid4 as random_uuid
from warnings import warn

import numpy as np
import pandas as pd

from . import wrappers as k
from ._pyarrow import pyarrow as pa
from .cast import *
from . import config
from .config import find_core_lib, k_allocator, licensed, pandas_2, system
from .constants import INF_INT16, INF_INT32, INF_INT64, NULL_INT16, NULL_INT32, NULL_INT64
from .exceptions import LicenseException, PyArrowUnavailable, PyKXException, QError
from .util import df_from_arrays, slice_to_range


__all__ = [
    'from_arrow',
    'from_bytes',
    'from_callable',
    'from_datetime_date',
    'from_datetime_time',
    'from_datetime_datetime',
    'from_datetime_timedelta',
    'from_dict',
    'from_ellipsis',
    'from_fileno',
    'from_float',
    'from_int',
    'from_pykx_k',
    'from_list',
    'from_numpy_datetime64',
    'from_numpy_ndarray',
    'from_numpy_timedelta64',
    'from_none',
    'from_pandas_dataframe',
    'from_pandas_index',
    'from_pandas_series',
    'from_pathlib_path',
    'from_range',
    'from_slice',
    'from_str',
    'from_uuid_UUID',
    'from_tuple',
]


def __dir__():
    return __all__


def _init(_q):
    global q
    q = _q


if system == 'Windows': # nocov
    import msvcrt # nocov


cdef inline core.K _k(x):
    return <core.K><uintptr_t>x._addr


# nanoseconds between 1970-01-01 and 2000-01-01
DEF TIMESTAMP_OFFSET = 946684800000000000
#      months between 1970-01-01 and 2000-01-01
DEF MONTH_OFFSET = 360
#        days between 1970-01-01 and 2000-01-01
DEF DATE_OFFSET = 10957


if not licensed:
    # Initialize the q memory system
    CDLL(str(find_core_lib('e'))).khp('', -1)


np_int_types = (
    np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
    np.int_, np.uint, np.longlong, np.ulonglong, np.int8, np.int16, np.int32,
    np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.intp, np.uintp
)

# Floating-point types, with less or equal 32 bits, which should be converted to q Real
np_float32_types = (
    np.half, np.float16, np.single, np.float32
)

# 64-bits floating-point types to convert to q Float
np_float64_types = (
    np.double, np.longdouble, np.float64, np.float_
)

np_float_types = np_float32_types + np_float64_types

type_number_to_pykx_k_type = {**k.type_number_to_pykx_k_type, **{
    98: k.Table,
    99: k.Dictionary,
    101: k.UnaryPrimitive,
}}

pykx_type_to_type_number = {v: k for k, v in type_number_to_pykx_k_type.items()}

pykx_ktype_to_np_type = {
    k.BooleanVector: bool,
    k.GUIDVector: np.complex128,
    k.ByteVector: np.uint8,
    k.ShortVector: np.int16,
    k.IntVector: np.int32,
    k.LongVector: np.int64,
    k.RealVector: np.float32,
    k.FloatVector: np.float64,
    k.CharVector: bytes,
    k.SymbolVector: str,
    k.TimestampVector: 'datetime64[ns]',
    k.MonthVector: 'datetime64[M]',
    k.DateVector: 'datetime64[D]',
    k.DatetimeVector: np.float64,
    k.TimespanVector: 'timedelta64[ns]',
    k.MinuteVector: 'timedelta64[m]',
    k.SecondVector: 'timedelta64[s]',
    k.TimeVector: 'timedelta64[ms]',
    k.EnumVector: np.int64
}


_overflow_error_by_size = {
    'long': OverflowError('q long must be in the range [-9223372036854775808, 9223372036854775807]'),
    'int': OverflowError('q int must be in the range [-2147483648, 2147483647]'),
    'short': OverflowError('q short must be in the range [-32768, 32767]'),
    'byte': OverflowError('q byte must be in the range [0, 255]'),
    'boolean': OverflowError('q boolean must be in the range [0, 1]'),
}


def _first_isinstance(x, t):
    try:
        return isinstance(x[0], t)
    except IndexError:
        return False


def _conversion_TypeError(x, input_type, output_type):
    if output_type is None:
        output_type = 'K object'
    x_repr = repr(x) if isinstance(x, str) else f"'{x!r}'"
    return TypeError(f"Cannot convert {input_type} {x_repr} to {output_type}. See pykx.register to register custom conversions.")

KType = Union[k.K, int]


def _resolve_k_type(ktype: KType) -> Optional[k.K]:
    """Resolve the pykx.K type represented by the ktype parameter."""
    if ktype is None:
        return None
    elif isinstance(ktype, dict):
        return ktype
    elif isinstance(ktype, int):
        try:
            return type_number_to_pykx_k_type[ktype]
        except KeyError:
            raise TypeError(f'Numeric ktype {ktype!r} does not exist')
    try:
        if issubclass(ktype, k.K):
            return ktype
    except TypeError:
        pass
    raise TypeError(f'ktype {ktype!r} unrecognized')


def _default_converter(x, ktype: Optional[KType] = None, *, cast: bool = False, handle_nulls: bool = False):
    if os.environ.get('PYKX_UNDER_Q', '').lower() == "true":
        return from_pyobject(x, ktype, cast, handle_nulls)
    raise _conversion_TypeError(x, type(x), ktype)


def from_none(x: None,
              ktype: Optional[KType] = None,
              *,
              cast: bool = False,
              handle_nulls: bool = False,
) -> k.Identity:
    """Converts `None` into a `pykx.Identity` object.

    Parameters:
        x: The `None` singleton object.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            `ktype`         | Returned value
            --------------- | ----------------------------------
            `None`          | Same as for `ktype=pykx.Identity`.
            `pykx.Identity` | Generic null (`::`).
        cast: Unused.
        handle_nulls: Unused.

    Returns:
        A `pykx.Identity` instance, i.e. generic null.
    """
    cdef core.K kx
    cdef core.U null_guid

    if ktype == k.GUIDAtom:
        null_guid.g = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # 16 null bytes
        kx = core.ku(null_guid)
    elif ktype == k.ShortAtom:
        kx = core.kh(NULL_INT16)
    elif ktype == k.IntAtom:
        kx = core.ki(NULL_INT32)
    elif ktype == k.LongAtom:
        kx = core.kj(NULL_INT64)
    elif ktype == k.RealAtom:
        kx = core.ke(math.nan)
    elif ktype == k.FloatAtom:
        kx = core.kf(math.nan)
    elif ktype == k.CharAtom:
        kx = core.kc(b' ')
    elif ktype == k.SymbolAtom:
        kx = core.ks('')
    elif ktype == k.TimestampAtom:
        kx = core.ktj(-12, NULL_INT64)
    elif ktype == k.MonthAtom:
        kx = core.ki(NULL_INT32)
        kx.t = -13
    elif ktype == k.DateAtom:
        kx = core.ki(NULL_INT32)
        kx.t = -14
    elif ktype == k.TimespanAtom:
        kx = core.ktj(-16, NULL_INT64)
    elif ktype == k.MinuteAtom:
        kx = core.ki(NULL_INT32)
        kx.t = -17
    elif ktype == k.SecondAtom:
        kx = core.ki(NULL_INT32)
        kx.t = -18
    elif ktype == k.TimeAtom:
        kx = core.ki(NULL_INT32)
        kx.t = -19
    else:
        if licensed:
            return q('::') # `::` always has the same address in memory, which is occasionally relevant
        kx = core.ka(101)
        kx.g = 0
        kx.j = 0
    return factory(<uintptr_t>kx, False)


_ktype_to_type_number_str = {
    k.List: "0h",

    k.BooleanAtom: "-1h",
    k.ByteAtom: "-4h",
    k.ShortAtom: "-5h",
    k.IntAtom: "-6h",
    k.LongAtom: "-7h",
    k.RealAtom: "-8h",
    k.FloatAtom: "-9h",
    k.CharAtom: "-10h",
    k.SymbolAtom: "-11h",
    k.TimestampAtom: "-12h",
    k.MonthAtom: "-13h",
    k.DateAtom: "-14h",
    k.DatetimeAtom: "-15h",
    k.TimespanAtom: "-16h",
    k.MinuteAtom: "-17h",
    k.SecondAtom: "-18h",
    k.TimeAtom: "-19h",

    k.BooleanVector: "1h",
    k.ByteVector: "4h",
    k.ShortVector: "5h",
    k.IntVector: "6h",
    k.LongVector: "7h",
    k.RealVector: "8h",
    k.FloatVector: "9h",
    k.CharVector: "10h",
    k.SymbolVector: "11h",
    k.TimestampVector: "12h",
    k.MonthVector: "13h",
    k.DateVector: "14h",
    k.DatetimeVector: "15h",
    k.TimespanVector: "16h",
    k.MinuteVector: "17h",
    k.SecondVector: "18h",
    k.TimeVector: "19h",
}


def from_pykx_k(x: k.K,
                ktype: Optional[KType] = None,
                *,
                cast: bool = False,
                handle_nulls: bool = False,
) -> k.K:
    """Converts a `pykx.K` object into a `pykx.K` object.

    Parameters:
        x: A `pykx.K` instance.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +------------------------+-------------------------------------------------------------+
            | `ktype`                | Returned value                                              |
            +========================+=============================================================+
            | `None`                 | The provided `pykx.K` object unchanged.                     |
            +------------------------+-------------------------------------------------------------+
            | `The type of `x`       | Same as for `ktype=None`.                                   |
            +------------------------+-------------------------------------------------------------+
            | `pykx.List`            | The provided `pykx.K` object as a `pykx.List` object        |
            +------------------------+-------------------------------------------------------------+
            | `pykx.BooleanAtom`     | The provided `pykx.K` object as a `pykx.BooleanAtom`        |
            +------------------------+-------------------------------------------------------------+
            | `pykx.ByteAtom`        | The provided `pykx.K` object as a `pykx.ByteAtom`           |
            +------------------------+-------------------------------------------------------------+
            | `pykx.ShortAtom`       | The provided `pykx.K` object as a `pykx.ShortAtom`          |
            +------------------------+-------------------------------------------------------------+
            | `pykx.IntAtom`         | The provided `pykx.K` object as a `pykx.IntAtom`            |
            +------------------------+-------------------------------------------------------------+
            | `pykx.LongAtom`        | The provided `pykx.K` object as a `pykx.LongAtom`           |
            +------------------------+-------------------------------------------------------------+
            | `pykx.RealAtom`        | The provided `pykx.K` object as a `pykx.RealAtom`           |
            +------------------------+-------------------------------------------------------------+
            | `pykx.FloatAtom`       | The provided `pykx.K` object as a `pykx.FloatAtom`          |
            +------------------------+-------------------------------------------------------------+
            | `pykx.CharAtom`        | The provided `pykx.K` object as a `pykx.CharAtom`           |
            +------------------------+-------------------------------------------------------------+
            | `pykx.SymbolAtom`      | The provided `pykx.K` object as a `pykx.SymbolAtom`         |
            +------------------------+-------------------------------------------------------------+
            | `pykx.TimestampAtom`   | The provided `pykx.K` object as a `pykx.TimestampAtom`      |
            +------------------------+-------------------------------------------------------------+
            | `pykx.MonthAtom`       | The provided `pykx.K` object as a `pykx.MonthAtom`          |
            +------------------------+-------------------------------------------------------------+
            | `pykx.DateAtom`        | The provided `pykx.K` object as a `pykx.DateAtom`           |
            +------------------------+-------------------------------------------------------------+
            | `pykx.DatetimeAtom`    | The provided `pykx.K` object as a `pykx.DatetimeAtom`       |
            +------------------------+-------------------------------------------------------------+
            | `pykx.TimespanAtom`    | The provided `pykx.K` object as a `pykx.TimespanAtom`       |
            +------------------------+-------------------------------------------------------------+
            | `pykx.MinuteAtom`      | The provided `pykx.K` object as a `pykx.MinuteAtom`         |
            +------------------------+-------------------------------------------------------------+
            | `pykx.SecondAtom`      | The provided `pykx.K` object as a `pykx.SecondAtom`         |
            +------------------------+-------------------------------------------------------------+
            | `pykx.TimeAtom`        | The provided `pykx.K` object as a `pykx.TimeAtom`           |
            +------------------------+-------------------------------------------------------------+
            | `pykx.BooleanVector`   | The provided `pykx.K` object as a `pykx.BooleanVector`      |
            +------------------------+-------------------------------------------------------------+
            | `pykx.ByteVector`      | The provided `pykx.K` object as a `pykx.ByteVector`         |
            +------------------------+-------------------------------------------------------------+
            | `pykx.ShortVector`     | The provided `pykx.K` object as a `pykx.ShortVector`        |
            +------------------------+-------------------------------------------------------------+
            | `pykx.IntVector`       | The provided `pykx.K` object as a `pykx.IntVector`          |
            +------------------------+-------------------------------------------------------------+
            | `pykx.LongVector`      | The provided `pykx.K` object as a `pykx.LongVector`         |
            +------------------------+-------------------------------------------------------------+
            | `pykx.RealVector`      | The provided `pykx.K` object as a `pykx.RealVector`         |
            +------------------------+-------------------------------------------------------------+
            | `pykx.FloatVector`     | The provided `pykx.K` object as a `pykx.FloatVector`        |
            +------------------------+-------------------------------------------------------------+
            | `pykx.CharVector`      | The provided `pykx.K` object as a `pykx.CharVector`         |
            +------------------------+-------------------------------------------------------------+
            | `pykx.SymbolVector`    | The provided `pykx.K` object as a `pykx.SymbolVector`       |
            +------------------------+-------------------------------------------------------------+
            | `pykx.TimestampVector` | The provided `pykx.K` object as a `pykx.TimestampVector`    |
            +------------------------+-------------------------------------------------------------+
            | `pykx.MonthVector`     | The provided `pykx.K` object as a `pykx.MonthVector`        |
            +------------------------+-------------------------------------------------------------+
            | `pykx.DateVector`      | The provided `pykx.K` object as a `pykx.DateVector`         |
            +------------------------+-------------------------------------------------------------+
            | `pykx.DatetimeVector`  | The provided `pykx.K` object as a `pykx.DatetimeVector`     |
            +------------------------+-------------------------------------------------------------+
            | `pykx.TimespanVector`  | The provided `pykx.K` object as a `pykx.TimespanVector`     |
            +------------------------+-------------------------------------------------------------+
            | `pykx.MinuteVector`    | The provided `pykx.K` object as a `pykx.MinuteVector`       |
            +------------------------+-------------------------------------------------------------+
            | `pykx.SecondVector`    | The provided `pykx.K` object as a `pykx.SecondVector`       |
            +------------------------+-------------------------------------------------------------+
            | `pykx.TimeVector`      | The provided `pykx.K` object as a `pykx.TimeVector`         |
            +------------------------+-------------------------------------------------------------+
            | `pykx.Table`           | The provided `pykx.K` object as a `pykx.Table`              |
            +------------------------+-------------------------------------------------------------+
            | `pykx.Dictionary`      | The provided `pykx.K` object as a `pykx.Dictionary`         |
            +------------------------+-------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        LicenseException: Directly casting between `ktype`s is not possible without a license.
        TypeError: `ktype` is not `None` and does not match the provided `pykx.K` object or any of
            the castable `ktype`s.
        QError: Cannot cast the provided `pykx.K` object to `ktype`.

    Returns:
        The provided `pykx.K` instance.
    """
    if ktype is None or ktype == type(x):
        return x
    else:
        if not licensed:
            raise LicenseException('directly convert between K types.')
        if type(x) is k.Table or type(x) is k.Dictionary:
            if ktype is k.Table or ktype is k.Dictionary:
                try:
                    return q.flip(x)
                except QError:
                    raise QError("'length all rows of a Dictionary must be of equal length to flip")
            else:
                raise TypeError('Can only convert directly between pykx.Table and pykx.Dictionary')
        if ktype in _ktype_to_type_number_str.keys():
            try:
                return q(f'{{{_ktype_to_type_number_str[ktype]}$x}}', x)
            except QError:
                raise QError(f"'type cannot cast {type(x)} into {ktype}")
        else:
            raise TypeError(f'Unknown ktype {ktype!r} in conversion of {k!r}.')


def from_int(x: Any,
             ktype: Optional[KType] = None,
             *,
             cast: bool = False,
             handle_nulls: bool = False,
) -> k.IntegralNumericAtom:
    """Converts an `int` into an instance of a subclass of `pykx.IntegralNumericAtom`.

    Parameters:
        x: The `int` that will be converted into a `pykx.IntegralNumericAtom` object.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +--------------------+----------------------------------------------------------------+
            |`ktype`             | Returned value                                                 |
            +====================+================================================================+
            | `None`             | Same as for `ktype=pykx.LongAtom`, except when `x` is a        |
            |                    | `bool`, in which case it is the same as for                    |
            |                    | `ktype=pykx.BooleanAtom`.                                      |
            +--------------------+----------------------------------------------------------------+
            | `pykx.LongAtom`    | The `int` as a signed 64 bit integer in q.                     |
            +--------------------+----------------------------------------------------------------+
            | `pykx.IntAtom`     | The `int` as a signed 32 bit integer in q.                     |
            +--------------------+----------------------------------------------------------------+
            | `pykx.ShortAtom`   | The `int` as a signed 16 bit integer in q.                     |
            +--------------------+----------------------------------------------------------------+
            | `pykx.ByteAtom`    | The `int` as a unsigned 8 bit integer in q.                    |
            +--------------------+----------------------------------------------------------------+
            | `pykx.BooleanAtom` | The `int` as a boolean in q.                                   |
            +--------------------+----------------------------------------------------------------+
        cast: Apply cast to Python `int` before converting to an `pykx.IntegralNumericAtom` object.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for an `int` or unsupported cast between Python types.
        OverflowError: The provided `int` is out of bounds for the selected
            `pykx.IntegralNumericAtom` subclass.

    Warning: The edges of the bounds have special meanings in q.
        For `pykx.LongAtom`, `pykx.IntAtom`, and `pykx.ShortAtom`, the lower bound of the
        range is the null value for that type, one larger than that is the negative infinity for
        that type, and the upper bound of the range is positive infinity for that type. An
        exception will not be raised if `int` objects with these values are provided. Refer to
        the nulls and infinities PyKX documentation for more information about q nulls and
        infinities, and how to handle them with PyKX.

    Returns:
        An instance of a `pykx.IntegralNumericAtom` subclass.
    """
    if hasattr(x, 'fileno'):
        return from_fileno(x, ktype, cast=cast, handle_nulls=handle_nulls)
    elif ktype is None and isinstance(x, (bool, np.bool_)):
        ktype = k.BooleanAtom

    if cast and type(x) is not int:
        x = cast_to_python_int(x)

    cdef core.K kx
    if ktype is None or ktype is k.LongAtom:
        if not NULL_INT64 <= x <= INF_INT64:
            raise _overflow_error_by_size['long']
        kx = core.kj(x)
    elif ktype is k.IntAtom:
        if not NULL_INT32 <= x <= INF_INT32:
            raise _overflow_error_by_size['int']
        kx = core.ki(x)
    elif ktype is k.ShortAtom:
        if not NULL_INT16 <= x <= INF_INT16:
            raise _overflow_error_by_size['short']
        kx = core.kh(x)
    elif ktype is k.ByteAtom:
        if not 0 <= x <= 255:
            raise _overflow_error_by_size['byte']
        kx = core.kg(x)
    elif ktype is k.BooleanAtom:
        if not 0 <= x <= 1:
            raise _overflow_error_by_size['boolean']
        kx = core.kb(x)
    else:
        raise _conversion_TypeError(x, 'Python int', ktype)
    return factory(<uintptr_t>kx, False)


def from_float(x: Any,
               ktype: Optional[KType] = None,
               *,
               cast: bool = False,
               handle_nulls: bool = False,
) -> k.NonIntegralNumericAtom:
    """Converts a `float` into an instance of a subclass of `pykx.NonIntegralNumericAtom`.

    Parameters:
        x: The `float` that will be converted into a `pykx.NonIntegralNumericAtom` object.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            `ktype`          | Returned value
            ---------------- | ------------------------------------------
            `None`           | Same as for `ktype=pykx.FloatAtom`.
            `pykx.FloatAtom` | The `float` as a signed 64 bit float in q.
            `pykx.RealAtom`  | The `float` as a signed 32 bit float in q.
        cast: Apply a cast to a Python `float` before converting to a `pykx.NonIntegralNumericAtom`
            object.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for a `float`.

    Returns:
        An instance of a `pykx.NonIntegralNumericAtom` subclass.
    """
    if cast and type(x) is not float:
        x = cast_to_python_float(x)

    if ktype is None:
        if isinstance(x, np_float32_types):
            ktype = k.RealAtom
        elif isinstance(x, (float, *np_float_types)):
            ktype = k.FloatAtom
    cdef core.K kx
    if ktype is None or ktype is k.FloatAtom:
        kx = core.kf(x)
    elif ktype is k.RealAtom:
        kx = core.ke(x)
    else:
        raise _conversion_TypeError(x, 'Python float', ktype)
    return factory(<uintptr_t>kx, False)


def from_str(x: str,
             ktype: Optional[KType] = None,
             *,
             cast: bool = False,
             handle_nulls: bool = False,
) -> Union[k.CharAtom, k.CharVector, k.SymbolAtom]:
    """Converts a `str` into an instance of a string-like subclass of `pykx.K`.

    Parameters:
        x: The `str` that will be converted into a `pykx.K` object.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            `ktype`                 | Returned value
            ----------------------- | ---------------------------------------------------------
            `None`                  | Same as for `ktype=pykx.SymbolAtom`.
            `pykx.SymbolAtom`       | The `str` as a q symbol.
            `pykx.SymbolicFunction` | The `str` as a q symbol that can be called as a function.
            `pykx.CharVector`       | The `str` as a q character vector.
            `pykx.CharAtom`         | The `str` as a q character atom.
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for a `str`.
        ValueError: `ktype=pykx.CharAtom`, but `x` is not 1 character long.

    Returns:
        An instance of a `pykx.SymbolAtom`, `pykx.CharVector`, or `pykx.CharAtom`.
    """
    cdef core.K kx
    cdef bytes as_bytes = x.encode('utf-8')
    if ktype is None or issubclass(ktype, k.SymbolAtom):
        kx = core.ks(as_bytes)
    elif ktype is k.CharAtom:
        if len(as_bytes) != 1:
            raise ValueError(
                f"'pykx.CharAtom' can only be created from a single character, not {x!r}"
            )
        kx = core.kc(ord(as_bytes) if x else 32)
    elif ktype is k.CharVector:
        kx = core.kpn(as_bytes, len(as_bytes))
    else:
        raise _conversion_TypeError(x, 'Python str', ktype)
    wrapped = factory(<uintptr_t>kx, False)
    if ktype is not None and ktype is k.SymbolicFunction:
        return k.SymbolicFunction._from_addr(wrapped._addr)
    return wrapped


def from_bytes(x: bytes,
               ktype: Optional[KType] = None,
               *,
               cast: bool = False,
               handle_nulls: bool = False,
) -> Union[k.SymbolAtom, k.SymbolVector, k.CharAtom]:
    """Converts a `bytes` object into an instance of a string-like subclass of `pykx.K`.

    Parameters:
        x: The `bytes` object that will be converted into a `pykx.K` object.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +-------------------+---------------------------------------------------------+
            | `ktype`           | Returned value                                          |
            +===================+=========================================================+
            | `None`            | Same as for `ktype=pykx.CharVector` when `len(x) != 1`. |
            |                   | Same as for `ktype=pykx.CharAtom` when `len(x) == 1`.   |
            +-------------------+---------------------------------------------------------+
            | `pykx.CharVector` | The `str` as a q character vector.                      |
            +-------------------+---------------------------------------------------------+
            | `pykx.SymbolAtom` | The `str` as a q symbol.                                |
            +-------------------+---------------------------------------------------------+
            | `pykx.CharAtom`   | The `str` as a q character atom.                        |
            +-------------------+---------------------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `bytes`.
        ValueError: `ktype=pykx.CharAtom`, but `x` is not 1 character long.

    Returns:
        An instance of a `pykx.CharVector`, `pykx.SymbolAtom`, or `pykx.CharAtom`.
    """
    cdef core.K kx
    if (ktype is None and len(x) == 1) or (ktype is not None and ktype is k.CharAtom):
        if len(x) != 1:
            raise ValueError(
                f"'pykx.CharAtom' can only be created from a single character, not {x!r}"
            )
        kx = core.kc(ord(x) if x else 32)
    elif ktype is None or ktype is k.CharVector:
        kx = core.kpn(x, len(x))
    elif ktype is k.SymbolAtom:
        kx = core.ks(x)
    else:
        raise _conversion_TypeError(x, 'Python bytes', ktype)  # nocov
    return factory(<uintptr_t>kx, False)


def from_uuid_UUID(x: UUID,
                   ktype: Optional[KType] = None,
                   *,
                   cast: bool = False,
                   handle_nulls: bool = False,
) -> k.GUIDAtom:
    """Converts a `uuid.UUID` into a `pykx.GUIDAtom`.

    Parameters:
        x: The `uuid.UUID` that will be converted into a `pykx.GUIDAtom`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            `ktype`         | Returned value
            --------------- | ----------------------------------
            `None`          | Same as for `ktype=pykx.GUIDAtom`.
            `pykx.GUIDAtom` | The `uuid.UUID` as a q GUID atom.
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `uuid.UUID`.

    Returns:
        An instance of `pykx.GUIDAtom`.
    """
    if not isinstance(x, UUID):
        raise _conversion_TypeError(x, type(x), ktype)

    cdef core.K kx
    if ktype is not None and not ktype is k.GUIDAtom:
        raise _conversion_TypeError(x, repr('uuid.UUID'), ktype)

    u = x.bytes
    cdef core.U guid
    for i in range(len(u)):
        guid.g[i] = u[i]
    return factory(<uintptr_t>core.ku(guid), False)


def from_list(x: list,
              ktype: Optional[KType] = None,
              *,
              cast: bool = False,
              handle_nulls: bool = False,
) -> k.Vector:
    """Converts a `list` into an instance of a subclass of `pykx.Vector`.

    Note: q lists are vectors
        In q, a list (also called a "general list") is a vector of k objects. Non-list vectors
        (also called "typed vectors" or simply "vectors") are vectors of one particular type.

    Parameters:
        x: The `list` that will be converted into a `pykx.Vector`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +----------------------+--------------------------------------------------------------+
            | `ktype`              | Returned value                                               |
            +======================+==============================================================+
            | `None`               | Same as for `ktype=pykx.List`.                               |
            +----------------------+--------------------------------------------------------------+
            | `pykx.List`          | The `list` as a q list. Each element of the Python list will |
            |                      | be converted independently into a `pykx.K` object, and added |
            |                      | to the q list.                                               |
            +----------------------+--------------------------------------------------------------+
            | `pykx.BooleanVector` | The `list` as a `pykx.BooleanVector`. This is equivalent to  |
            |                      | converting a Numpy array `array(x, dtype=bool)` to q.        |
            +----------------------+--------------------------------------------------------------+
            | `pykx.ByteVector`    | The `list` as a `pykx.ByteVector`. This is equivalent to     |
            |                      | converting a Numpy array `array(x, dtype=np.uint8)` to q.    |
            +----------------------+--------------------------------------------------------------+
            | `pykx.ShortVector`   | The `list` as a `pykx.ShortVector`. This is equivalent to    |
            |                      | converting a Numpy array `array(x, dtype=np.int16)` to q.    |
            +----------------------+--------------------------------------------------------------+
            | `pykx.IntVector`     | The `list` as a `pykx.IntVector`. This is equivalent to      |
            |                      | converting a Numpy array `array(x, dtype=np.int32)` to q.    |
            +----------------------+--------------------------------------------------------------+
            | `pykx.LongVector`    | The `list` as a `pykx.LongVector`. This is equivalent to     |
            |                      | converting a Numpy array `array(x, dtype=np.int64)` to q.    |
            +----------------------+--------------------------------------------------------------+
            | `pykx.RealVector`    | The `list` as a `pykx.RealVector`. This is equivalent to     |
            |                      | converting a Numpy array `array(x, dtype=np.float32)` to q.  |
            +----------------------+--------------------------------------------------------------+
            | `pykx.FloatVector`   | The `list` as a `pykx.FloatVector`. This is equivalent to    |
            |                      | converting a Numpy array `array(x, dtype=np.float64)` to q.  |
            +----------------------+--------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `list`.
        ValueError: Could not convert some elements of `x` in accordance with the `ktype`.

    Returns:
        An instance of a subclass of `pykx.Vector`.
    """
    if ktype is not None and not ktype is k.List:
        try:
            np_type = pykx_ktype_to_np_type.get(ktype) if ktype != k.GUIDVector else object

            if ktype is k.TimestampVector and config.keep_local_times:

                x = [y.replace(tzinfo=None) for y in x]
            return from_numpy_ndarray(np.array(x, dtype=np_type), ktype, cast=cast, handle_nulls=handle_nulls)
        except TypeError as ex:
            raise _conversion_TypeError(x, 'Python list', ktype) from ex
    cdef core.K kx = core.ktn(0, len(x))
    for i, item in enumerate(x):
        # No good way to specify the ktype for nested types
        kk = toq(item, cast=cast, handle_nulls=handle_nulls)
        (<core.K*>kx.G0)[i] = core.r1(_k(kk))
    res = factory(<uintptr_t>kx, False)
    if licensed:
        try:
            q
        except NameError:
            pass #Once on import q does not exist
        else:
            res = q('{value $[9h~type first x;count[x]#0x0;x]!x}',res, skip_debug=True)
    return res


def from_tuple(x: tuple,
               ktype: Optional[KType] = None,
               *,
               cast: bool = False,
               handle_nulls: bool = False,
) -> k.Vector:
    """Converts a `tuple` into an instance of a subclass of `pykx.Vector`.

    See Also:
        [`from_list`][pykx.toq.from_list]

    Parameters:
        x: The `tuple` that will be converted into a `pykx.Vector`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +----------------------+--------------------------------------------------------------+
            | `ktype`              | Returned value                                               |
            +======================+==============================================================+
            | `None`               | Same as for `ktype=pykx.List`.                               |
            +----------------------+--------------------------------------------------------------+
            | `pykx.List`          | The `tuple` as a q list. Each element of the Python list     |
            |                      | will be converted independently into a `pykx.K` object, and  |
            |                      | added to the q list.                                         |
            +----------------------+--------------------------------------------------------------+
            | `pykx.BooleanVector` | The `tuple` as a `pykx.BooleanVector`. This is equivalent to |
            |                      | converting a Numpy array `array(x, dtype=bool)` to q.        |
            +----------------------+--------------------------------------------------------------+
            | `pykx.ByteVector`    | The `tuple` as a `pykx.ByteVector`. This is equivalent to    |
            |                      | converting a Numpy array `array(x, dtype=np.uint8)` to q.    |
            +----------------------+--------------------------------------------------------------+
            | `pykx.ShortVector`   | The `tuple` as a `pykx.ShortVector`. This is equivalent to   |
            |                      | converting a Numpy array `array(x, dtype=np.int16)` to q.    |
            +----------------------+--------------------------------------------------------------+
            | `pykx.IntVector`     | The `tuple` as a `pykx.IntVector`. This is equivalent to     |
            |                      | converting a Numpy array `array(x, dtype=np.int32)` to q.    |
            +----------------------+--------------------------------------------------------------+
            | `pykx.LongVector`    | The `tuple` as a `pykx.LongVector`. This is equivalent to    |
            |                      | converting a Numpy array `array(x, dtype=np.int64)` to q.    |
            +----------------------+--------------------------------------------------------------+
            | `pykx.RealVector`    | The `tuple` as a `pykx.RealVector`. This is equivalent to    |
            |                      | converting a Numpy array `array(x, dtype=np.float32)` to q.  |
            +----------------------+--------------------------------------------------------------+
            | `pykx.FloatVector`   | The `tuple` as a `pykx.FloatVector`. This is equivalent to   |
            |                      | converting a Numpy array `array(x, dtype=np.float64)` to q.  |
            +----------------------+--------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `tuple`.
        ValueError: Could not convert some elements of `x` in accordance with the `ktype`.

    Returns:
        An instance of a subclass of `pykx.Vector`.
    """
    if ktype is not None and not issubclass(ktype, k.Vector):
        raise _conversion_TypeError(x, 'Python tuple', ktype)
    return from_list(list(x), ktype=ktype, cast=cast, handle_nulls=handle_nulls)


def from_dict(x: dict,
              ktype: Optional[KType] = None,
              *,
              cast: bool = False,
              handle_nulls: bool = False,
) -> k.Dictionary:
    """Converts a `dict` into a `pykx.Dictionary`.

    Parameters:
        x: The `dict` that will be converted into a `pykx.Dictionary`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +-------------------+-----------------------------------------------------------------+
            | `ktype`           | Returned value                                                  |
            +===================+=================================================================+
            | `None`            | Same as for `ktype=pykx.Dictionary`.                            |
            +-------------------+-----------------------------------------------------------------+
            | `pykx.Dictionary` | The `dict` as a q dictionary. Each key and value of the Python  |
            |                   |  dictionary will be converted independently into a `pykx.K`     |
            |                   |  object, and added to the q dictionary.                         |
            +-------------------+-----------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Convert `pd.NaT` to corresponding q null values in Pandas dataframes and
            Numpy arrays.

    Raises:
        TypeError: Unsupported `ktype` for `dict`.

    Returns:
        An instance of `pykx.Dictionary`.
    """
    if ktype is not None and ktype != k.Dictionary:
        raise _conversion_TypeError(x, 'Python dict', ktype)
    cdef core.K kx
    if len(x) == 0:
        k_keys = from_list([])
    elif all(isinstance(key, (str, k.SymbolAtom)) for key in x.keys()):
        k_keys = from_numpy_ndarray(np.array([str(key) for key in x.keys()], dtype='U'),
                                    cast=cast, handle_nulls=handle_nulls)
    else:
        k_keys = from_list(list(x.keys()), cast=cast, handle_nulls=handle_nulls)
    k_values = from_list(list(x.values()), cast=cast, handle_nulls=handle_nulls)
    kx = core.xD(core.r1(_k(k_keys)), core.r1(_k(k_values)))
    return factory(<uintptr_t>kx, False)


# keys are supported ndarray k types, values are # of bytes per element
supported_np_temporal_types = {
    k.TimestampVector: 8,
    k.MonthVector: 4,
    k.DateVector: 4,
    k.TimespanVector: 8,
    k.MinuteVector: 4,
    k.SecondVector: 4,
    k.TimeVector: 4,
}


supported_np_nontemporal_types = {
    k.List: 8,
    k.BooleanVector: 1,
    k.GUIDVector: 16,
    k.ByteVector: 1,
    k.ShortVector: 2,
    k.IntVector: 4,
    k.LongVector: 8,
    k.RealVector: 4,
    k.FloatVector: 8,
    k.CharVector: 1,
    k.SymbolVector: 8,
}


supported_ndarray_k_types = {**supported_np_temporal_types, **supported_np_nontemporal_types}


def _listify(x: np.ndarray):
    """Convert all arrays except the lowest level into lists."""
    if len(x.shape) > 1:
        return [_listify(y) for y in list(x)]
    return np.array(x) if k_allocator else x


_dtype_to_ktype = {
    np.dtype('bool'): k.BooleanVector,
    np.dtype('uint8'): k.ByteVector,
    np.dtype('int16'): k.ShortVector,
    np.dtype('int32'): k.IntVector,
    np.dtype('int64'): k.LongVector,
    np.dtype('float32'): k.RealVector,
    np.dtype('float64'): k.FloatVector,
    np.dtype('datetime64[s]'): k.TimestampVector,
    np.dtype('datetime64[ms]'): k.TimestampVector,
    np.dtype('datetime64[us]'): k.TimestampVector,
    np.dtype('datetime64[ns]'): k.TimestampVector,
    np.dtype('datetime64[M]'): k.MonthVector,
    np.dtype('datetime64[D]'): k.DateVector,
    np.dtype('timedelta64[ns]'): k.TimespanVector,
    np.dtype('timedelta64[m]'): k.MinuteVector,
    np.dtype('timedelta64[s]'): k.SecondVector,
    np.dtype('timedelta64[ms]'): k.TimeVector,
}


_ktype_to_dtype = {v: k for k, v in _dtype_to_ktype.items()}


def _resolve_ndarray_k_type(x, ktype):
    if ktype is not None:
        return ktype
    else:
        try:
            return _dtype_to_ktype[x.dtype]
        except KeyError:
            pass

    if x.dtype.char == 'U' or (x.dtype == object and _first_isinstance(x, str)):
        return k.SymbolVector
    elif x.dtype.char == 'S':
        return k.CharVector
    elif (x.dtype == object and _first_isinstance(x, UUID)) or x.dtype == complex:
        return k.GUIDVector
    elif x.dtype == object or len(x) == 0:
        return k.List
    else:
        raise TypeError(f'ktype cannot be inferred from Numpy dtype {x.dtype}')


cdef extern from 'include/foreign.h':
    object pyobject_to_long_addr(object x)


def from_numpy_ndarray(x: np.ndarray,
                       ktype: Optional[KType] = None,
                       *,
                       cast: bool = False,
                       handle_nulls: bool = False,
) -> k.Vector:
    """Converts a `numpy.ndarray` into a `pykx.Vector`.

    Warning: Data is always copied when converting from Numpy to q if not using the PyKX Allocator.
        While many conversions from q to Python types can be performed without incurring a copy,
        the reverse is not true. As mentioned in
        [the q C API documentation](https://code.kx.com/q/interfaces/capiref/#k-object), vectors in
        q store their metadata in a fixed memory location relative to their data, so the data for
        an array in Python's memory cannot be shared by q, since q would have to store the metadata
        in a memory location it has no ownership of.

    If the `pykx.k_allocator` is enabled with the environment variable `PYKX_ALLOCATOR` or if the
    `QARGS` environment variable contains `--pykxalloc`, then 0 copy numpy array conversions
    will be enabled for certain `numpy.dtypes`s. This will allow for 0 copy conversions from `numpy`
    arrays of a corresponding `numpy.dtype` to `pykx.NumericVector`s, and `pykx.TimespanVector`s.
    You can find the corresponding `numpy.dtype` of a `pykx.Vector` type by checking the
    `pykx.Vector._np_dtype` property of the type you want to create.
    Note that when these 0 copy conversions are done both the numpy array and the resulting
    `pykx.Vector` refer to the same area of memory and changes made to one will be present in the
    other object.

    Because q only supports 1-dimensional arrays (vectors), Numpy arrays with more than 1 dimension
    will have all every level converted into a list of lists except for the lowest, for which each
    vector is converted to a q vector independently.

    For example, in the following case, even though Numpy stores all of the data contiguously in
    memory with a single `numpy.ndarray` object wrapping it, q will store it as 3 vectors, with a
    list containing them:

    ```python
    >>> np.arange(12).reshape(3, 4)
    array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    >>> kx.K(np.arange(12).reshape(3, 4))
    pykx.List(pykx.q('
    0 1 2  3
    4 5 6  7
    8 9 10 11
    '))
    ```

    Numpy masked arrays can be used to indicate null data for integer types.

    Parameters:
        x: The `numpy.ndarray` that will be converted into a `pykx.Vector`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +------------------------+------------------------------------------------------------+
            | `ktype`                | Returned value                                             |
            +========================+============================================================+
            | `None`                 | The returned vector is as if the `ktype` had been set      |
            |                        | based off of the Numpy `dtype`, roughly following this     |
            |                        | mapping:                                                   |
            |                        |                                                            |
            |                        | +----------------------+------------------------+          |
            |                        | | `x.dtype`            | Inferred `ktype`       |          |
            |                        | +======================+========================+          |
            |                        | | `object`             | `pykx.List`            |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `object` (uuid.UUID) | `pykx.GUIDVector`      |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `object` (str)       | `pykx.SymbolVector`    |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'U'`                | `pykx.SymbolVector`    |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'S'`                | `pykx.CharVector`      |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `np.float64`         | `pykx.FloatVector`     |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `np.float32`         | `pykx.RealVector`      |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `np.int64`           | `pykx.LongVector`      |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `np.int32`           | `pykx.IntVector`       |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `np.int16`           | `pykx.ShortVector`     |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'timedelta64[ms]'`  | `pykx.TimeVector`      |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'timedelta64[ns]'`  | `pykx.TimespanVector`  |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'datetime64[ns]'`   | `pykx.TimestampVector` |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'datetime64[us]'`   | `pykx.TimestampVector` |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'datetime64[ms]'`   | `pykx.TimestampVector` |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'datetime64[s]'`    | `pykx.TimestampVector` |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'timedelta64[s]'`   | `pykx.SecondVector`    |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'datetime64[M]'`    | `pykx.MonthVector`     |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'datetime64[D]'`    | `pykx.DateVector`      |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `'timedelta64[m]'`   | `pykx.MinuteVector`    |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `np.uint8`           | `pykx.ByteVector`      |          |
            |                        | +----------------------+------------------------+          |
            |                        | | `bool`               | `pykx.BooleanVector`   |          |
            |                        | +----------------------+------------------------+          |
            +------------------------+------------------------------------------------------------+
            | `pykx.List`            | The Numpy array as a q general list. Equivalent to         |
            |                        | `pykx.List(x.tolist())`.                                   |
            +------------------------+------------------------------------------------------------+
            | `pykx.BooleanVector`   | The Numpy array as a vector of q booleans, which each take |
            |                        | up 8 bits in memory.                                       |
            +------------------------+------------------------------------------------------------+
            | `pykx.ByteVector`      | The Numpy array as a vector of q unsigned 8 bit intergers. |
            +------------------------+------------------------------------------------------------+
            | `pykx.GUIDVector`      | The Numpy array as a vector of q GUIDs, which each take up |
            |                        | 128 bits in memory.                                        |
            +------------------------+------------------------------------------------------------+
            | `pykx.ShortVector`     | The Numpy array as a vector of q signed 16 bit integers.   |
            +------------------------+------------------------------------------------------------+
            | `pykx.IntVector`       | The Numpy array as a vector of q signed 32 bit integers.   |
            +------------------------+------------------------------------------------------------+
            | `pykx.LongVector`      | The Numpy array as a vector of q signed 64 bit integers.   |
            +------------------------+------------------------------------------------------------+
            | `pykx.RealVector`      | The Numpy array as a vector of q 32 bit floats.            |
            +------------------------+------------------------------------------------------------+
            | `pykx.FloatVector`     | The Numpy array as a vector of q 64 bit floats.            |
            +------------------------+------------------------------------------------------------+
            | `pykx.CharVector`      | The Numpy array as a vector of q characters. Each          |
            |                        | characters in q is a single byte, which has implications   |
            |                        | for multi-byte characters in e.g. Unicode strings.         |
            +------------------------+------------------------------------------------------------+
            | `pykx.SymbolVector`    | The Numpy array as a vector of q symbols. Each symbol is   |
            |                        | permanently interned into q's memory.                      |
            +------------------------+------------------------------------------------------------+
            | `pykx.TimestampVector` | The Numpy array as a vector of q timestamp, i.e. signed 64 |
            |                        | bit integers which represent the number of nanoseconds     |
            |                        | since the q epoch: `2000-01-01T00:00:00.000000000`. The    |
            |                        | data from the Numpy array will be incremented to adjust    |
            |                        | its epoch from the standard epoch                          |
            |                        | (`1970-01-01T00:00:00.000000000`) to the q epoch.          |
            +------------------------+------------------------------------------------------------+
            | `pykx.MonthVector`     | The Numpy array as a vector of q months, i.e. signed 32    |
            |                        | bit intergers which represent the number of months since   |
            |                        | the q epoch: `2000-01`. The data from the Numpy array will |
            |                        | be incremented to adjust its epoch from the standard epoch |
            |                        | (`1970-01`) to the q epoch.                                |
            +------------------------+------------------------------------------------------------+
            | `pykx.DateVector`      | The Numpy array as a vector of q dates, i.e. signed 32 bit |
            |                        | intergers which represent the number of days since the q   |
            |                        | epoch: `2000-01-01`. The data from the Numpy array will be |
            |                        | incremented to adjust its epoch from the standard epoch    |
            |                        | (`1970-01-01`) to the q epoch.                             |
            +------------------------+------------------------------------------------------------+
            | `pykx.TimespanVector`  | The Numpy array as a vector of q timespans, i.e. signed 64 |
            |                        | bit integers which represent a number of nanoseconds.      |
            +------------------------+------------------------------------------------------------+
            | `pykx.MinuteVector`    | The Numpy array as a vector of q minutes, i.e. signed 32   |
            |                        | bit integers which represent a number of minutes.          |
            +------------------------+------------------------------------------------------------+
            | `pykx.SecondVector`    | The Numpy array as a vector of q seconds, i.e. signed 32   |
            |                        | bit integers which represent a number of seconds.          |
            +------------------------+------------------------------------------------------------+
            | `pykx.TimeVector`      | The Numpy array as a vector of q times, i.e. signed 32 bit |
            |                        | integers which represent a number of milliseconds.         |
            +------------------------+------------------------------------------------------------+
        cast: Apply a cast before converting to a `pykx.Vector` object. It will try to cast the input
            Numpy array to the Numpy type which best conforms to the target `ktype`.
        handle_nulls: Convert `pd.NaT` to corresponding q null values in Pandas dataframes and
            Numpy arrays.


    Raises:
        TypeError: Unsupported `ktype` for `numpy.ndarray`.
        ValueError: Numpy array is not C-contiguous, which is required for `ktype`.

    Returns:
        An instance of a subclass of `pykx.Vector`.
    """
    ktype = _resolve_ndarray_k_type(x, ktype)

    if cast:
        try:
            if _dtype_to_ktype[x.dtype] != ktype:
                x = cast_numpy_ndarray_to_dtype(x, _ktype_to_dtype[ktype])
        except KeyError:
            if x.dtype == np.dtype('object') or x.dtype.char == 'U' or x.dtype.char == 'S':
                x = cast_numpy_ndarray_to_dtype(x, _ktype_to_dtype[ktype])

    if ktype not in supported_ndarray_k_types:
        raise _conversion_TypeError(x, repr('numpy.ndarray'), ktype)

    # q doesn't support n-dimensional vectors, so we treat them as lists to preserve the shape
    if len(x.shape) > 1:
        return from_list(_listify(x), ktype=k.List, cast=cast, handle_nulls=handle_nulls)

    elif isinstance(x, np.ma.MaskedArray):
        if x.dtype.kind != 'i':
            raise TypeError(
                'Non-integral masked array conversions to q are not yet implemented'
            )
        x = np.ma.MaskedArray(x, copy=False, fill_value=-2 ** (x.itemsize * 8 - 1)).filled()

    elif ktype is k.List:
        return from_list(x.tolist(), ktype=k.List, cast=cast, handle_nulls=handle_nulls)

    elif ktype is k.CharVector:
        if str(x.dtype).endswith('U1'):
            return from_bytes(''.join(x).encode())
        elif str(x.dtype).endswith('S1'):
            return from_bytes(b''.join(x))
        elif 'S' == x.dtype.char:
            return from_list(x.tolist(), ktype=k.List, cast=None, handle_nulls=None)
        raise _conversion_TypeError(x, repr('numpy.ndarray'), ktype)

    cdef long long n = x.size
    cdef core.K kx = NULL
    cdef core.U guid
    cdef bytes as_bytes
    cdef uintptr_t data
    cdef long int i

    if ktype is k.GUIDVector and x.dtype == object:
        kx = core.ktn(ktype.t, n)
        for i in range(n):
            guid_bytes = x[i].bytes
            for j in range(len(guid_bytes)):
                guid.g[j] = guid_bytes[j]
            (<core.U*>kx.G0)[i] = guid
        return factory(<uintptr_t>kx, False)

    elif ktype is k.SymbolVector:
        kx = core.ktn(ktype.t, n)
        for i in range(n):
            if x[i] is None:
                as_bytes = b''
            else:
                as_bytes = str(x[i]).encode('utf-8')
            (<char**>kx.G0)[i] = core.sn(as_bytes, len(as_bytes))
        return factory(<uintptr_t>kx, False)

    elif ktype in supported_np_temporal_types:
        if ktype is k.TimestampVector or ktype is k.TimespanVector:
            offset = TIMESTAMP_OFFSET if ktype is k.TimestampVector else 0
            dtype = x.dtype
            x = x.view(np.int64)
            mul = None
            if dtype == np.dtype('<M8[us]'):
                mul = 1000
            elif dtype == np.dtype('<M8[ms]'):
                mul = 1000000
            elif dtype == np.dtype('<M8[s]'):
                mul = 1000000000
            if mul is not None or handle_nulls:
                x = x.copy()
            if ktype is k.TimestampVector:
                if handle_nulls:
                    mask = (x != NULL_INT64)
                    if mask.all():
                        x = (x if mul is None else (x*mul)) - offset
                    else:
                        x[mask] = (x[mask] if mul is None else (x[mask]*mul)) - offset
                else:
                    x = (x if mul is None else (x*mul)) - offset
            else:
                x = x if mul is None else x*mul
            itemsize = supported_ndarray_k_types[k.LongVector]
            if itemsize != x.itemsize:
                core.r0(kx)
                raise TypeError('Item size mismatch when converting Numpy ndarray to q: q item size '
                                f'({itemsize}) != Numpy item size ({x.itemsize})')
            if not k_allocator:
                kx = core.ktn(ktype.t, n)
                data = x.__array_interface__['data'][0]
                memcpy(<void *> kx.G0, <void *> data, n * itemsize)
                return factory(<uintptr_t>kx, False)
        else:
            kx = core.ktn(ktype.t, n)
            x = x.view(np.int64)

            offset = None
            if ktype is k.MonthVector:
                offset = MONTH_OFFSET
            elif ktype is k.DateVector:
                offset = DATE_OFFSET

            mask = None
            if handle_nulls:
                mask = (x != NULL_INT64)
                x[~mask] = NULL_INT32

            if offset:
                if handle_nulls:
                    x[mask] = x[mask] - offset
                else:
                    x = x - offset

            x = x.astype(np.int32, copy=False)
            itemsize = supported_ndarray_k_types[k.IntVector]
            if itemsize != x.itemsize:
                core.r0(kx)
                raise TypeError('Item size mismatch when converting Numpy ndarray to q: q item size '
                                f'({itemsize}) != Numpy item size ({x.itemsize})')
            # I've benchmarked this and it seems to be consistently faster not using nep-49 tricks,
            # I'm not sure why.
            data = x.__array_interface__['data'][0]
            memcpy(<void *> kx.G0, <void *> data, n * itemsize)
            return factory(<uintptr_t>kx, False)
    elif ktype in supported_np_nontemporal_types:
        if hasattr(x.data, 'c_contiguous') and not x.data.c_contiguous:
            x = np.ascontiguousarray(x)

        if not np.can_cast(x.dtype, pykx_ktype_to_np_type[ktype], casting='no'):
            raise _conversion_TypeError(x, repr('numpy.ndarray'), ktype)

        itemsize = supported_ndarray_k_types[ktype]
        if itemsize != x.itemsize:
            core.r0(kx)
            raise TypeError('Item size mismatch when converting Numpy ndarray to q: q item size '
                            f'({itemsize}) != Numpy item size ({x.itemsize})')
        if not k_allocator:
            kx = core.ktn(ktype.t, n)
            data = x.__array_interface__['data'][0]
            memcpy(<void *> kx.G0, <void *> data, n * itemsize)
            return factory(<uintptr_t>kx, False)
    if not k_allocator:
        return factory(<uintptr_t>kx, False) # nocov
    Py_INCREF(x)
    data = x.__array_interface__['data'][0]
    kx = <core.K>(data - 16)
    kx.t = pykx_type_to_type_number[ktype]
    kx.n = n
    res = factory(<uintptr_t>kx, True)
    setattr(res, '_numpy_allocated', pyobject_to_long_addr(x))
    return res


_size_to_nan = {
    2: NULL_INT16,
    4: NULL_INT32,
    8: NULL_INT64,
}

_float_size_to_class = {
    4: np.float32,
    8: np.float64  
}

_int_size_to_class = {
    2: np.int16,
    4: np.int32 ,
    8: np.int64
}


def _to_numpy_or_categorical(x, col_name=None, df=None):
    # Arrays taken from Pandas can come in many forms. If they come from Pandas series, arrays, or
    # indexes, then we need to set an appropriate value to fill in for `pd.NA`.
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (pd.Series, pd.Index)):
        if isinstance(x.values, pd.Categorical):
            return from_pandas_categorical(
                x.values,
                name=col_name if pandas_2 and col_name is not None else x.name
            )
        elif isinstance(x.values, pd.core.arrays.ExtensionArray):
            if x.dtype.kind != 'f' and hasattr(x, 'isnull') and x.isnull().values.any():
                if not x.dtype.kind in ['i', 'M', 'm']:
                    raise TypeError(
                        'Non-integral masked array conversions to q are not yet implemented'
                    )
                if x.dtype.kind == 'i':
                    dtype = _int_size_to_class[x.dtype.itemsize]
                elif x.dtype.kind == 'm':
                    dtype = np.timedelta64()
                elif x.dtype.kind == 'M':
                    dtype =  np.datetime64()               
                if k_allocator:
                    return np.array(x.to_numpy(copy=False, na_value=_size_to_nan[x.dtype.itemsize], dtype=dtype))
                return x.to_numpy(copy=False, na_value=_size_to_nan[x.dtype.itemsize], dtype=dtype)
            elif x.dtype.kind == 'f' and hasattr(x, 'isnull') and x.isnull().values.any():
                float_class = _float_size_to_class[x.dtype.itemsize]
                if k_allocator:
                    return np.array(x.to_numpy(copy=False, dtype=float_class))
                return x.to_numpy(copy=False, dtype=float_class)
            else:
                return np.array(x) if k_allocator else x.to_numpy(copy=False)
        else:
            return np.array(x.values) if k_allocator else x.values
    else:
        return np.array(x) if k_allocator else np.array(x, copy=False)


def from_pandas_dataframe(x: pd.DataFrame,
                          ktype: Optional[KType] = None,
                          *,
                          cast: bool = False,
                          handle_nulls: bool = False,
) -> Union[k.Table, k.KeyedTable]:
    """Converts a `pandas.DataFrame` into a `pykx.Table` or `pykx.KeyedTable` as appropriate.

    The index and columns of the dataframe are each converted independently, and then the results
    of those conversions are used to assemble the resulting q table or keyed table.

    If the dataframe does not have column names they will be generated: x, x1, .., x(n-1).

    See Also:
        - [`from_pandas_index`][pykx.toq.from_pandas_index]
        - [`from_pandas_series`][pykx.toq.from_pandas_series]

    Parameters:
        x: The `pandas.DataFrame` that will be converted into a `pykx.Table` or `pykx.KeyedTable`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +-------------------+-----------------------------------------------------------------+
            | `ktype`           | Returned value                                                  |
            +===================+=================================================================+
            | `None`            | The returned value is as if the `ktype` had been set based off  |
            |                   | of the index of the dataframe. If the index has an integer      |
            |                   | type, starts at 0, and increases by 1 for every row, then a     |
            |                   | `pykx.Table` is returned. Otherwise, a `pykx.KeyedTable` is     |
            |                   | returned. Note that `pykx.KeyedTable` is a subclass of          |
            |                   | `pykx.Dictionary`, not `pykx.Table`. Both are subclasses of     |
            |                   | `pykx.Mapping`.                                                 |
            +-------------------+-----------------------------------------------------------------+
            | `pykx.Table`      | The `pandas.DataFrame` as a q table.                            |
            +-------------------+-----------------------------------------------------------------+
            | `pykx.KeyedTable` | The `pandas.DataFrame` as a q keyed table.                      |
            +-------------------+-----------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Convert `pd.NaT` to corresponding q null values in Pandas dataframes and
            Numpy arrays.

    Raises:
        TypeError: Unsupported `ktype` for `pandas.DataFrame`.

    Returns:
        An instance of `pykx.Table` or `pykx.KeyedTable`.
    """
    cdef core.K kx

    if not x.columns.values.dtype == np.object_:
        col_names = ['x']
        cols_length =  len(x.columns)-1
        for i in range(cols_length):
            col_names.append('x'+str(i+1))
        x.columns = col_names

    convert_dict = None
    if type(ktype) == dict:
        convert_dict = ktype
        ktype = None

    if ktype is None or type(ktype)==dict or len(x)==0:
        if not x.index.name is None:
            ktype = k.KeyedTable
        elif pd.Index(np.arange(0, len(x))).equals(x.index):
            ktype = k.Table
        else:
            ktype = k.KeyedTable
    if ktype is k.Table:
        kk = from_dict(
            {k: _to_numpy_or_categorical(x[k], k, x) for k in x.columns},
            cast=cast,
            handle_nulls=handle_nulls
        )
        kx = core.xT(core.r1(_k(kk)))
        if kx == NULL:
            raise PyKXException('Failed to create table from k dictionary')
    elif ktype is k.KeyedTable:
        idx = None
        if isinstance(x.index, pd.MultiIndex):
            idx = x.index
        else:
            # The trick below helps create a pd.MultiIndex from another base Index
            idx = pd.DataFrame(index=[x.index]).index
        k_keys = from_pandas_index(idx, cast=cast, handle_nulls=handle_nulls)
        k_values = from_pandas_dataframe(x.reset_index(drop=True), cast=cast, handle_nulls=handle_nulls)
        kx = core.xD(core.r1(_k(k_keys)), core.r1(_k(k_values)))
        if kx == NULL:
            raise PyKXException('Failed to create k dictionary (keyed table)')
    else:
        raise _conversion_TypeError(x, repr('pandas.DataFrame'), ktype)
    if convert_dict == None:
        return factory(<uintptr_t>kx, False)
    return factory(<uintptr_t>kx, False).astype(convert_dict)


def from_pandas_series(x: pd.Series,
                       ktype: Optional[KType] = None,
                       *,
                       cast: bool = False,
                       handle_nulls: bool = False,
) -> k.Vector:
    """Converts a `pandas.Series` into an instance of a subclass of `pykx.Vector`.

    The given series is converted to a Numpy array, and then converted to a q vector as Numpy
    arrays generally are.

    See Also:
        [`from_numpy_ndarray`][pykx.toq.from_numpy_ndarray]

    Parameters:
        x: The `pandas.Series` that will be converted into a `pykx.Vector`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`,
            the type is inferred from `x` using the same logic as `from_numpy_ndarray` applied
            to `x.to_numpy()`.
        cast: Unused.
        handle_nulls: Convert `pd.NaT` to corresponding q null values in Pandas dataframes and
            Numpy arrays.

    Raises:
        TypeError: Unsupported `ktype` for `pandas.Series`.

    Returns:
        An instance of a subclass of `pykx.Vector`.
    """
    arr = _to_numpy_or_categorical(x)
    if isinstance(arr, np.ndarray):
        return toq(arr[0] if (1,) == arr.shape else arr, ktype=ktype)
    else:
        return arr


if not pandas_2:
    _supported_pandas_index_types_via_numpy = (
        pd.core.indexes.base.Index,
        pd.core.indexes.numeric.NumericIndex,
        pd.core.indexes.extension.NDArrayBackedExtensionIndex,
    )
else:
    _supported_pandas_index_types_via_numpy = (
        pd.core.indexes.base.Index,
        pd.core.indexes.extension.NDArrayBackedExtensionIndex,
    )


def from_pandas_index(x: pd.Index,
                      ktype: Optional[KType] = None,
                      *,
                      cast: bool = False,
                      handle_nulls: bool = False,
) -> Union[k.Vector, k.Table]:
    """Converts a `pandas.Index` into a `pykx.Vector` or `pykx.Table` as appropriate.

    See Also:
        [`from_numpy_ndarray`][pykx.toq.from_numpy_ndarray]

    Parameters:
        x: The `pandas.Index` that will be converted into a `pykx.Table` or `pykx.KeyedTable`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +---------+---------------------------------------------------------------------------+
            | `ktype` | Returned value                                                            |
            +=========+===========================================================================+
            | `None`  | The resulting value is determined based on the kind of Pandas index       |
            |         | provided. If `x` is an instance of                                        |
            |         | `pandas.core.indexes.numeric.NumericIndex`, then it will be converted     |
            |         | into a Numpy array, then that will be converted into a `pykx.Vector` via  |
            |         | `from_numpy_ndarray`. If `x` is an instance of `pandas.MultiIndex`, then  |
            |         | each of its levels will be converted into a Numpy array, and assembled    |
            |         | into a `pykx.Table` with column names equal to the names of the levels of |
            |         | the index if they are named. For any level which is not named, its index  |
            |         | within `x.levels` is used as its name.                                    |
            +---------+---------------------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Convert `pd.NaT` to corresponding q null values in Pandas dataframes and
            Numpy arrays.

    Raises:
        TypeError: Unsupported `ktype` for `pandas.Index`.

    Returns:
        An instance of `pykx.Vector` or `pykx.Table`.
    """
    if isinstance(x, pd.CategoricalIndex):
        return from_pandas_categorical(x.values, ktype, x.name)
    elif isinstance(x, pd.MultiIndex):
        d = {(level.name if level.name else str(i)): level[x.codes[i]]
             for i, level in enumerate(x.levels)}
        index_dict = from_dict(d)
        return factory(<uintptr_t>core.xT(core.r1(_k(index_dict))), False)
    elif isinstance(x, _supported_pandas_index_types_via_numpy):
        return from_numpy_ndarray(x.to_numpy(), cast=cast, handle_nulls=handle_nulls)
    else:
        raise _conversion_TypeError(x, 'Pandas index', ktype)


ENUM_COUNT = 0
ENUMS = []

def from_pandas_categorical(x: pd.Categorical,
                            ktype: Optional[KType] = None,
                            name: Optional[str] = None,
                            *,
                            cast: bool = False,
                            handle_nulls: bool = False,
) -> k.Vector:
    """Converts a `pandas.Categorical` into a `pykx.EnumVector`.

    Parameters:
        x: The `pandas.Categorical` that will be converted into a `pykx.EnumVector` object.
        ktype: Ignored.
        name: The name of the resulting q enumeration.
        cast: Unused.
        handle_nulls: Unused.

    Returns:
        An instance of pykx.EnumVector.
    """
    global ENUM_COUNT
    global ENUMS
    if name is None:
        name = 'enum%d' % ENUM_COUNT
        ENUM_COUNT += 1
    try:
        q(f'{name}')
        if name not in ENUMS:
            ENUMS.append(name)
    except QError:
        pass
    if name not in ENUMS:
        res = q(f'{{{name}::y;`{name}!x}}',
                     k.IntVector(x.codes.astype('int32')),
                     x.categories)
        ENUMS.append(name)
    else:
        res = q(f"{{if[any not y in {name}; `cast]; `{name}$y@x}}", 
                x.codes.astype('int32'), 
                x.categories)
    return res


def from_pandas_nat(x: type(pd.NaT),
                    ktype: Optional[KType] = None,
                    *,
                    cast: bool = False,
                    handle_nulls: bool = False,
) -> k.TemporalAtom:
    """Converts a `pandas.NaT` into an instance of a subclass of `pykx.TemporalAtom`.

    Parameters:
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is a `pykx.TimestampAtom`. The following values are supported:

            `ktype`              | Returned value
            -------------------- | --------------------------------------------
            `None`               | Same as for `ktype=pykx.TimestampAtom`.
            `pykx.TimestampAtom` | The `pandas.NaT` as a null q timestamp atom.
            `pykx.MonthAtom`     | The `pandas.NaT` as a null q month atom.
            `pykx.DateAtom`      | The `pandas.NaT` as a null q date atom.
            `pykx.TimespanAtom`  | The `pandas.NaT` as a null q timespan atom.
            `pykx.MinuteAtom`    | The `pandas.NaT` as a null q minute atom.
            `pykx.SecondAtom`    | The `pandas.NaT` as a null q second atom.
            `pykx.TimeAtom`      | The `pandas.NaT` as a null q time atom.
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `Pandas.NaT`.
        NotImplementedError: The q datetime type is deprecated, so support for converting to it
            from Python has not been implemented.

    Returns:
        An instance of a subclass of `pykx.TemporalAtom`.
    """
    if ktype is None or ktype is k.TimestampAtom:
        kx = core.ktj(-12, k.TimestampVector._base_null_value)
    elif ktype is k.MonthAtom:
        kx = core.ki(k.MonthVector._base_null_value)
        kx.t = -13
    elif ktype is k.DateAtom:
        kx = core.kd(k.DateVector._base_null_value)
    elif ktype is k.DatetimeAtom:
        raise NotImplementedError
    elif ktype is k.TimespanAtom:
        kx = core.ktj(-16, k.TimespanVector._base_null_value)
    elif ktype is k.MinuteAtom:
        kx = core.ki(k.MinuteVector._base_null_value)
        kx.t = -17
    elif ktype is k.SecondAtom:
        kx = core.ki(k.SecondVector._base_null_value)
        kx.t = -18
    elif ktype is k.TimeAtom:
        kx = core.kt(k.TimeVector._base_null_value)
    else:
        raise _conversion_TypeError(x, repr('pandas.NaT'), ktype)
    return factory(<uintptr_t>kx, False)

_timedelta_resolution_str_map = {
    'timedelta64[ns]': k.TimespanAtom,
    'timedelta64[ms]': k.TimeAtom,
    'timedelta64[s]': k.SecondAtom,
}

def from_pandas_timedelta(
    x: Any,
    ktype: Optional[KType] = None,
    *,
    cast: bool = False,
    handle_nulls: bool = False,
) -> k.K:
    x = x.to_numpy()
    if ktype is None:
        ktype = _timedelta_resolution_str_map[str(x.dtype)]
    return from_numpy_timedelta64(x, ktype=ktype, cast=cast, handle_nulls=handle_nulls)


def from_arrow(x: Union['pa.Array', 'pa.Table'],
               ktype: Optional[KType] = None,
               *,
               cast: bool = False,
               handle_nulls: bool = False,
) -> Union[k.Vector, k.Table]:
    """Converts PyArrow arrays/tables into PyKX vectors/tables, respectively.

    Conversions from PyArrow to q are performed by converting the PyArrow array/table to pandas
    first, which avoids copying data when possible, then converting the resulting Pandas
    data structure to q using `from_pandas_series` or `from_pandas_dataframe` as appropriate.

    See Also:
        - [`from_pandas_dataframe`][pykx.toq.from_pandas_dataframe]
        - [`from_pandas_series`][pykx.toq.from_pandas_series]

    Warning: Memory usage may spike during this function.
        Normally converting from Python to q results in one copy of the data being made, but
        because this function first converts to Pandas, it's possible for there to be a temporary
        third copy made. `See the PyArrow docs about converting to Pandas for more details
        <https://arrow.apache.org/docs/python/pandas.html#memory-usage-and-zero-copy>`_.

    Parameters:
        x: The `pyarrow.Array`  that will be converted into a `pykx.Vector`, or the
            `pyarrow.Table` that will be converted into a `pykx.Table`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`,
            the type is inferred from `x`. This argument is propagated to the Pandas conversion
            functions (which in turn propagate it to the Numpy conversion functions).
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Cannot convert PyArrow extension array to `ktype`.

    Returns:
        An instance of `pykx.Vector` or `pykx.Table`.
    """
    # To convert Arrow data structures to q, we first convert it to Pandas. This conversion avoids
    # copies where possible, but often results in some amount of data being copied.
    # https://arrow.apache.org/docs/python/pandas.html#memory-usage-and-zero-copy
    if pa is None:
        raise PyArrowUnavailable
    if isinstance(x, pa.ExtensionArray):
        raise _conversion_TypeError(x, 'Arrow extension array', ktype)
    return toq(x.to_pandas(), ktype=ktype, cast=cast, handle_nulls=handle_nulls)


def from_datetime_date(x: Any,
                       ktype: Optional[KType] = None,
                       *,
                       cast: bool = False,
                       handle_nulls: bool = False,
) -> k.TemporalFixedAtom:
    """Converts a `datetime.date` into an instance of a subclass of `pykx.TemporalFixedAtom`.

    Parameters:
        x: The `datetime.date` that will be converted into a `pykx.TemporalFixedAtom`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +----------------------+--------------------------------------------------------------+
            | `ktype`              | Returned value                                               |
            +======================+==============================================================+
            | `None`               | Same as for `ktype=pykx.DateAtom`.                           |
            +----------------------+--------------------------------------------------------------+
            | `pykx.TimestampAtom` | The `datetime.date` as a q timestamp atom, which is a signed |
            |                      | 64 bit value representing the number of nanoseconds since    |
            |                      | the q epoch: `2000-01-01T00:00:00.000000000`.                |
            +----------------------+--------------------------------------------------------------+
            | `pykx.MonthAtom`     | The `datetime.date` as a q month atom, which is a signed 32  |
            |                      | bit value representing the number of months since the q      |
            |                      | epoch: `2000-01`.                                            |
            +----------------------+--------------------------------------------------------------+
            | `pykx.DateAtom`      | The `datetime.date` as a q date atom, which is a signed 32   |
            |                      | bit value representing the number of days since the q        |
            |                      | epoch: `2000-01-01`.                                         |
            +----------------------+--------------------------------------------------------------+
        cast: Apply a cast to a `datetime.date` object before converting to a `pykx.TemporalFixedAtom` object.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `datetime.date`.
        NotImplementedError: The q datetime type is deprecated, so support for converting to it
            from Python has not been implemented.

    Returns:
        An instance of a subclass of `pykx.TemporalFixedAtom`.
    """
    # TODO: the `cast is None` should be removed at the next major release (KXI-12945)
    if (cast is None or cast) and type(x) is not datetime.date:
        x = cast_to_python_date(x)

    return from_datetime_datetime(datetime.datetime.combine(x, datetime.time.min),
                                  ktype=k.DateAtom if ktype is None else ktype,
                                  cast=cast,
                                  handle_nulls=handle_nulls)


def from_datetime_time(x: Any,
                           ktype: Optional[KType] = None,
                           *,
                           cast: bool = False,
                           handle_nulls: bool = False,
) -> k.TemporalFixedAtom:
    if (cast is None or cast) and type(x) is not datetime.time:
        x = cast_to_python_time(x)

    return k.toq(datetime.datetime.combine(datetime.date.min, x) - datetime.datetime.min)


def from_datetime_datetime(x: Any,
                           ktype: Optional[KType] = None,
                           *,
                           cast: bool = False,
                           handle_nulls: bool = False,
) -> k.TemporalFixedAtom:
    """Converts a `datetime.datetime` into an instance of a subclass of `pykx.TemporalFixedAtom`.

    Note: Setting environment variable `KEEP_LOCAL_TIMES` will result in the use of local time zones not UTC time.
        By default this function will convert any `datetime.datetime` objects with time zone
        information to UTC before converting it to `q`. If you set the environment vairable to 1,
        true or True, then the objects with time zone information will not be converted to UTC and
        instead will be converted to `q` with no changes.

    Parameters:
        x: The `datetime.datetime` that will be converted into a `pykx.TemporalFixedAtom`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +----------------------+--------------------------------------------------------------+
            | `ktype`              | Returned value                                               |
            +======================+==============================================================+
            | `None`               | Same as for `ktype=pykx.TimestampAtom`.                      |
            +----------------------+--------------------------------------------------------------+
            | `pykx.TimestampAtom` | The `datetime.datetime` as a q timestamp atom, which is a    |
            |                      | signed 64 bit value representing the number of nanoseconds   |
            |                      | since the q epoch: `2000-01-01T00:00:00.000000000`.          |
            +----------------------+--------------------------------------------------------------+
            | `pykx.MonthAtom`     | The `datetime.datetime` as a q month atom, which is a signed |
            |                      | 32 bit value representing the number of months since the q   |
            |                      | epoch: `2000-01`.                                            |
            +----------------------+--------------------------------------------------------------+
            | `pykx.DateAtom`      | The `datetime.datetime` as a q date atom, which is a signed  |
            |                      | 32 bit value representing the number of days since the q     |
            |                      | epoch: `2000-01-01`.                                         |
            +----------------------+--------------------------------------------------------------+
        cast: Apply a cast to a `datetime.datetime` object before converting to a `pykx.TemporalFixedAtom`
            object.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `datetime.datetime`.
        NotImplementedError: The q datetime type is deprecated, so support for converting to it
            from Python has not been implemented.

    Returns:
        An instance of a subclass of `pykx.TemporalFixedAtom`.
    """
    # TODO: the `cast is None` should be removed at the next major release (KXI-12945)
    if (cast is None or cast) and type(x) is not datetime.datetime:
        x = cast_to_python_datetime(x)

    cdef core.K kx
    epoch = datetime.datetime(2000, 1, 1)
    if ktype is None or ktype is k.TimestampAtom:
        if isinstance(x, datetime.datetime) and x.tzinfo is not None:
            if config.keep_local_times:
                x = x.replace(tzinfo=None)
            else:
                epoch = epoch.replace(tzinfo=pytz.utc)
                x = x.astimezone(pytz.utc)
        d = x - epoch
        t = 1000 * (d.microseconds + 1000000 * (d.seconds + d.days * 86400))
        kx = core.ktj(-12, t)
    elif ktype is k.MonthAtom:
        kx = core.ki(12 * (x.year - epoch.year) + x.month - epoch.month)
        kx.t = -13
    elif ktype is k.DateAtom:
        kx = core.kd((x - epoch).days)
    elif ktype is k.DatetimeAtom:
        raise NotImplementedError
    else:
        raise _conversion_TypeError(x, repr('datetime.datetime'), ktype)
    return factory(<uintptr_t>kx, False)


def from_datetime_timedelta(x: Any,
                            ktype: Optional[KType] = None,
                            *,
                            cast: bool = False,
                            handle_nulls: bool = False,
) -> k.TemporalSpanAtom:
    """Converts a `datetime.timedelta` into an instance of a subclass of `pykx.TemporalSpanAtom`.

    Parameters:
        x: The `datetime.timedelta` that will be converted into a `pykx.TemporalSpanAtom`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +---------------------+---------------------------------------------------------------+
            | `ktype`             | Returned value                                                |
            +=====================+===============================================================+
            | `None`              | Same as for `ktype=pykx.TimespanAtom`.                        |
            +---------------------+---------------------------------------------------------------+
            | `pykx.TimespanAtom` | The `datetime.timedelta` as a q timespan atom, which is a     |
            |                     | signed 64 bit value representing a number of nanoseconds.     |
            +---------------------+---------------------------------------------------------------+
            | `pykx.MinuteAtom`   | The `datetime.timedelta` as a q minute atom, which is a       |
            |                     | signed 32 bit value representing a number of minutes.         |
            +---------------------+---------------------------------------------------------------+
            | `pykx.SecondAtom`   | The `datetime.timedelta` as a q second atom, which is a       |
            |                     | signed 32 bit value representing a number of seconds.         |
            +---------------------+---------------------------------------------------------------+
            | `pykx.TimeAtom`     | The `datetime.timedelta` as a q date atom, which is a signed  |
            |                     | 32 bit value representing a number of milliseconds.           |
            +---------------------+---------------------------------------------------------------+
        cast: Apply a cast to a `datetime.timedelta` object before converting to a `pykx.TemporalSpanAtom`
            object.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `datetime.timedelta`.

    Returns:
        An instance of a subclass of `pykx.TemporalSpanAtom`.
    """
    # TODO: the `cast is None` should be removed at the next major release (KXI-12945)
    if (cast is None or cast) and type(x) is not datetime.timedelta:
        x = cast_to_python_timedelta(x)

    cdef core.K kx
    if ktype is None or ktype is k.TimespanAtom:
        t = 1000 * (x.microseconds + 1000000 * (x.seconds + x.days * 86400))
        kx = core.ktj(-16, t)
    elif ktype is k.MinuteAtom:
        kx = core.ki(x.total_seconds() // 60)
        kx.t = -17
    elif ktype is k.SecondAtom:
        kx = core.ki(x.total_seconds())
        kx.t = -18
    elif ktype is k.TimeAtom:
        kx = core.kt(x.days * 86400000 + x.seconds * 1000 + x.microseconds // 1000)
    else:
        raise _conversion_TypeError(x, repr('datetime.timedelta'), ktype)
    return factory(<uintptr_t>kx, False)


def from_numpy_datetime64(x: np.datetime64,
                          ktype: Optional[KType] = None,
                          *,
                          cast: bool = False,
                          handle_nulls: bool = False,
) -> k.TemporalFixedAtom:
    """Converts a `numpy.datetime64` into an instance of a subclass of `pykx.TemporalFixedAtom`.

    Parameters:
        x: The `numpy.datetime64` that will be converted into a `pykx.TemporalFixedAtom`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +----------------------+--------------------------------------------------------------+
            | `ktype`              | Returned value                                               |
            +======================+==============================================================+
            | `None`               | Same as for `ktype=pykx.TimestampAtom`.                      |
            +----------------------+--------------------------------------------------------------+
            | `pykx.TimestampAtom` | The `numpy.datetime64` as a q timestamp atom, which is a     |
            |                      | signed 64 bit value representing the number of nanoseconds   |
            |                      | since the q epoch: `2000-01-01T00:00:00.000000000`.          |
            +----------------------+--------------------------------------------------------------+
            | `pykx.MonthAtom`     | The `numpy.datetime64` as a q month atom, which is a signed  |
            |                      | 32 bit value representing the number of months since the q   |
            |                      | epoch: `2000-01`.                                            |
            +----------------------+--------------------------------------------------------------+
            | `pykx.DateAtom`      | The `numpy.datetime64` as a q date atom, which is a signed   |
            |                      | 32 bit value representing the number of days since the q     |
            |                      | epoch: `2000-01-01`.                                         |
            +----------------------+--------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `numpy.datetime64`.
        NotImplementedError: The q datetime type is deprecated, so support for converting to it
            from Python has not been implemented.

    Returns:
        An instance of a subclass of `pykx.TemporalFixedAtom`.
    """
    cdef core.K kx
    if isinstance(x, pd._libs.tslibs.timestamps.Timestamp):
        x = x.to_datetime64()
    if ktype is None or ktype is k.TimestampAtom:
        kx = core.ktj(-12, x.astype(np.dtype('datetime64[ns]')).astype(np.int64) - TIMESTAMP_OFFSET) # noqa
    elif ktype is k.MonthAtom:
        kx = core.ki(x.astype(np.dtype('datetime64[M]')).astype(int) - MONTH_OFFSET)
        kx.t = -13
    elif ktype is k.DateAtom:
        kx = core.kd(x.astype(np.dtype('datetime64[D]')).astype(int) - DATE_OFFSET)
    elif ktype is k.DatetimeAtom:
        raise NotImplementedError
    else:
        raise _conversion_TypeError(x, repr('numpy.datetime64'), ktype)
    return factory(<uintptr_t>kx, False)


def from_numpy_timedelta64(x: np.timedelta64,
                           ktype: Optional[KType] = None,
                           *,
                           cast: bool = False,
                           handle_nulls: bool = False,
) -> k.TemporalSpanAtom:
    """Converts a `numpy.timedelta64` into an instance of a subclass of `pykx.TemporalSpanAtom`.

    Parameters:
        x: The `numpy.timedelta64` that will be converted into a `pykx.TemporalSpanAtom`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +---------------------+---------------------------------------------------------------+
            | `ktype`             | Returned value                                                |
            +=====================+===============================================================+
            | `None`              | Same as for `ktype=pykx.TimespanAtom`.                        |
            +---------------------+---------------------------------------------------------------+
            | `pykx.TimespanAtom` | The `numpy.timedelta64` as a q timespan atom, which is a      |
            |                     | signed 64 bit value representing a number of nanoseconds.     |
            +---------------------+---------------------------------------------------------------+
            | `pykx.MinuteAtom`   | The `numpy.timedelta64` as a q minute atom, which is a signed |
            |                     | 32 bit value representing a number of minutes.                |
            +---------------------+---------------------------------------------------------------+
            | `pykx.SecondAtom`   | The `numpy.timedelta64` as a q second atom, which is a signed |
            |                     | 32 bit value representing a number of seconds.                |
            +---------------------+---------------------------------------------------------------+
            | `pykx.TimeAtom`     | The `numpy.timedelta64` as a q time atom, which is a signed   |
            |                     | 32 bit value representing a number of milliseconds.           |
            +---------------------+---------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `numpy.timedelta64`.

    Returns:
        An instance of a subclass of `pykx.TemporalSpanAtom`.
    """
    cdef core.K kx
    if ktype is None or ktype is k.TimespanAtom:
        kx = core.ktj(-16, x.astype(np.dtype('timedelta64[ns]')).astype(np.int64))
    elif ktype is k.MinuteAtom:
        kx = core.ki(x.astype(np.dtype('timedelta64[m]')).astype(int))
        kx.t = -17
    elif ktype is k.SecondAtom:
        kx = core.ki(x.astype(np.dtype('timedelta64[s]')).astype(int))
        kx.t = -18
    elif ktype is k.TimeAtom:
        kx = core.kt(x.astype(np.dtype('timedelta64[ms]')).astype(int))
    else:
        raise _conversion_TypeError(x, repr('numpy.timedelta64'), ktype)
    return factory(<uintptr_t>kx, False)


def from_slice(x: slice,
               ktype: Optional[KType] = None,
               *,
               cast: bool = False,
               handle_nulls: bool = False,
) -> k.IntegralNumericVector:
    """Converts a `slice` into an instance of a subclass of `pykx.IntegralNumericVector`.

    Slice objects are used by Python for indexing. In q, indexing is done by applying a numeric
    atom or vector to the object being indexed. As such, `from_slice` works by converting the
    given `slice` into a `pykx.IntegralNumericVector`, which can then be applied to an
    indexable q object.

    Parameters:
        x: The `slice` that will be converted into a `pykx.IntegralNumericVector`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +----------------------+--------------------------------------------------------------+
            | `ktype`              | Returned value                                               |
            +======================+==============================================================+
            | `None`               | Same as for `ktype=pykx.LongVector`.                         |
            +----------------------+--------------------------------------------------------------+
            | `pykx.LongVector`    | The `slice` as a q vector of signed 64 bit integers.         |
            +----------------------+--------------------------------------------------------------+
            | `pykx.IntVector`     | The `slice` as a q vector of signed 32 bit integers.         |
            +----------------------+--------------------------------------------------------------+
            | `pykx.ShortVector`   | The `slice` as a q vector of signed 16 bit integers.         |
            +----------------------+--------------------------------------------------------------+
            | `pykx.ByteVector`    | The `slice` as a q vector of unsigned 8 bit integers.        |
            +----------------------+--------------------------------------------------------------+
            | `pykx.BooleanVector` | The `slice` as a q vector of unsigned 1 bit integers.        |
            +----------------------+--------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `slice`.
        OverflowError: The minimum or maximum value in the slice is out of bounds for the selected
            `ktype`.

    Returns:
        An instance of a subclass of `pykx.IntegralNumericVector`.
    """
    if x.stop is None:
        raise ValueError(f'Cannot convert endless slice {x!r} to vector')
    return from_range(slice_to_range(x, x.stop), ktype=ktype, cast=cast, handle_nulls=handle_nulls)


_range_k_type_map = {
    k.LongVector: ('long', np.int64, (NULL_INT64, INF_INT64)),
    k.IntVector: ('int', np.int32, (NULL_INT32, INF_INT32)),
    k.ShortVector: ('short', np.int16, (NULL_INT16, INF_INT16)),
    k.ByteVector: ('byte', np.uint8, (0, 255)),
    k.BooleanVector: ('boolean', bool, (0, 1)),
}


def from_range(x: range,
               ktype: Optional[KType] = None,
               *,
               cast: bool = False,
               handle_nulls: bool = False,
) -> k.IntegralNumericVector:
    """Converts a `range` into an instance of a subclass of `pykx.IntegralNumericVector`.

    Parameters:
        x: The `range` that will be converted into a `pykx.IntegralNumericVector`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +----------------------+--------------------------------------------------------------+
            | `ktype`              | Returned value                                               |
            +======================+==============================================================+
            | `None`               | Same as for `ktype=pykx.LongVector`.                         |
            +----------------------+--------------------------------------------------------------+
            | `pykx.LongVector`    | The `slice` as a q vector of signed 64 bit integers.         |
            +----------------------+--------------------------------------------------------------+
            | `pykx.IntVector`     | The `slice` as a q vector of signed 32 bit integers.         |
            +----------------------+--------------------------------------------------------------+
            | `pykx.ShortVector`   | The `slice` as a q vector of signed 16 bit integers.         |
            +----------------------+--------------------------------------------------------------+
            | `pykx.ByteVector`    | The `slice` as a q vector of unsigned 8 bit integers.        |
            +----------------------+--------------------------------------------------------------+
            | `pykx.BooleanVector` | The `slice` as a q vector of unsigned 1 bit integers.        |
            +----------------------+--------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `range`.
        OverflowError: The minimum or maximum value in the range is out of bounds for the selected
            `ktype`.

    Returns:
        An instance of a subclass of `pykx.IntegralNumericVector`.
    """
    if ktype is None:
        ktype = k.LongVector
    if not issubclass(ktype, k.IntegralNumericVector):
        raise _conversion_TypeError(x, 'Python range', ktype)
    name, dtype, (lower_bound, upper_bound) = _range_k_type_map[ktype]
    if lower_bound - 1 in x or upper_bound + 1 in x:
        raise _overflow_error_by_size[name]
    return from_numpy_ndarray(np.arange(x.start, x.stop, x.step, dtype=dtype),
                              ktype=ktype,
                              cast=cast,
                              handle_nulls=handle_nulls)


def from_pathlib_path(x: Path,
                      ktype: Optional[KType] = None,
                      *,
                      cast: bool = False,
                      handle_nulls: bool = False,
) -> k.SymbolAtom:
    """Converts a `pathlib.Path` into a q handle symbol.

    In q a symbol atom that begin with a `:` is known as an "hsym", or handle symbol. These
    symbols represent local or remote paths or resources.

    Windows paths are reformatted as POSIX paths.

    This function does not make the path absolute.

    Parameters:
        x: The `pathlib.Path` that will be converted into an hsym.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +-------------------+-----------------------------------------------------------------+
            | `ktype`           | Returned value                                                  |
            +===================+=================================================================+
            | `None`            | Same as for `ktype=pykx.SymbolAtom`.                            |
            +-------------------+-----------------------------------------------------------------+
            | `pykx.SymbolAtom` | The `pathlib.Path` as a q handle symbol, which will be the path |
            |                   | as text, prefixed by `:`, as a POSIX path.                      |
            +-------------------+-----------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `pathlib.Path`.

    Returns:
        An instance of a subclass of `pykx.IntegralNumericVector`.
    """
    return from_str(f'{"" if str(x)[:1] == ":" else ":"}{x.as_posix()}')


def from_ellipsis(x: Ellipsis,
                  ktype: Optional[KType] = None,
                  *,
                  cast: bool = False,
                  handle_nulls: bool = False,
) -> k.ProjectionNull:
    """Converts an `Ellipsis` (`...`) into a q projection null.

    Warning: Projection nulls are unwieldy.
        If you aren't sure you know what you are doing, then you probably don't need to use a
        projection null.

    PyKX uses Python's `...` singleton to represent the q projection null, which is similar to
    generic null (see [`from_none`][pykx.toq.from_none]), but indicates to q not just that there
    is missing data, but that it should be filled in with the next q object that is applied onto
    it.

    This is how q creates function projections with later parameters specified. For instance,
    the q function projection `{x+y}[;7]` has an argument list of 2 q objects: a projection null,
    followed by a long atom with a value of 7.

    Because PyKX treats `...` as projection null, q projections can be created in Python like so:

    ```python
    >>> f = pykx.K(lambda x, y: x + y)
    >>> projection = f(..., 7)
    >>> projection(2)
    pykx.LongAtom(pykx.q('9'))
    ```

    Parameters:
        x: The `Ellipsis` singleton object.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +-----------------------+------------------------------------------+
            | `ktype`               | Returned value                           |
            +=======================+==========================================+
            | `None`                | Same as for `ktype=pykx.ProjectionNull`. |
            +-----------------------+------------------------------------------+
            | `pykx.ProjectionNull` | An instance of `pykx.ProjectionNull`.    |
            +-----------------------+------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for `Ellipsis`.

    Returns:
        An instance of a subclass of `pykx.ProjectionNull`.
    """
    if ktype is not None and not ktype is k.ProjectionNull:
        raise _conversion_TypeError(x, 'Ellipsis', ktype)
    return q('value[(;)]1')


# TODO: After Python 3.7 support is dropped, use a `typing.Protocol` for `x` (KXI-9158)
def from_fileno(x: Any,
                ktype: Optional[KType] = None,
                *,
                cast: bool = False,
                handle_nulls: bool = False,
) -> k.IntAtom:
    """Converts an object with a `fileno` attribute to a `pykx.IntAtom`.

    In q, int atoms that match open file descriptors can be called as if they were functions. Refer
    to https://code.kx.com/q/basics/handles/ for more information.

    The `fileno` attribute of `x` can either be a Python `int`, or a function that returns
    an `int` when called. The `int` should represent an open file descriptor.

    Parameters:
        x: An object with a `fileno` attribute.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +----------------+-----------------------------------------------+
            | `ktype`        | Returned value                                |
            +================+===============================================+
            | `None`         | Same as for `ktype=pykx.IntAtom`.             |
            +----------------+-----------------------------------------------+
            | `pykx.IntAtom` | The open file descriptor as a `pykx.IntAtom`. |
            +----------------+-----------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for an object with the `fileno` attribute.

    Returns:
        A `pykx.IntAtom` that represents an open file descriptor.
    """
    if ktype is not None and not ktype is k.IntAtom:
        raise _conversion_TypeError(x, "object with 'fileno' attribute", ktype)
    fd = x.fileno() if callable(x.fileno) else x.fileno
    if system == 'Windows':
        try:
            fd = msvcrt.get_osfhandle(fd)
        except OSError:
            pass
    return from_int(fd, ktype=k.IntAtom)


def from_callable(x: Callable,
                  ktype: Optional[KType] = None,
                  *,
                  cast: bool = False,
                  handle_nulls: bool = False,
) -> k.Composition:
    """Converts a callable object into a q composition.

    The resulting `pykx.Composition` object works by keeping a reference to the provided callable
    object. When the composition is called, it calls the Python callable with the arguments it received
    as `pykx.K` objects. Therefore, Python functions that are converted to q functions must be
    able to handle receiving `pykx.K` objects as arguments.

    The return value of the provided Python callable will automatically be converted to a
    `pykx.K` object by having `pykx.K` called on it. You can control how this conversion
    happens by manually ensuring the return value of the function is a `pykx.K` instance.

    Because q does not have keyword arguments, all of the parameters of the Python callable are
    treated as position parameters. `*args` and `**kwargs` are likewise treated as individual
    parameters, which can be provided a `pykx.Vector` and a `pykx.Mapping` respectively.

    Functions in q can have at most 8 parameters, so Python callables being converted to q must
    abide by this limit. Parameters which have defaults set still count towards this limit. An
    `*args` parameters and a `**kwargs` each count as a single parameter towards this limit.

    Note: Inspection of the callable must be possible.
        In order to determine how many parameters a provided callable has, PyKX calls
        `inspect.signature` on it. Some callables may not be introspectable in certain
        implementations of Python. For example, in CPython, some built-in functions defined in C
        provide no metadata about their arguments.

    Parameters:
        x: A callable Python object that can take `pykx.K` as arguments, and returns an object
            that can be converted into a q object via `pykx.K`.
        ktype: Desired `pykx.K` subclass (or type number) for the returned value. If `None`, the
            type is inferred from `x`. The following values are supported:

            +--------------------+----------------------------------------------------------+
            | `ktype`            | Returned value                                           |
            +====================+==========================================================+
            | `None`             | Same as for `ktype=pykx.Composition`.                    |
            +--------------------+----------------------------------------------------------+
            | `pykx.Composition` | A q composition that wraps the provided callable object. |
            +---------------+---------------------------------------------------------------+
        cast: Unused.
        handle_nulls: Unused.

    Raises:
        TypeError: Unsupported `ktype` for a callable object.
        ValueError: `x` accepts too many parameters - no more than 8 can be accepted.
        ValueError: `inspect.signature(x)` failed - could not introspect `x`.

    Returns:
        A `pykx.Composition` that calls the provided Python callable object when it is calls.
    """
    if not licensed:
        raise LicenseException('convert a Python callable to q')
    params = list(signature(x).parameters.values())
    if len(params) > 8:
        raise ValueError('Too many parameters - q functions cannot have more than 8 parameters')
    return q('{.pykx.wrap[x][<]}', k.Foreign(x))


cdef extern from 'include/foreign.h':
    uintptr_t py_to_pointer(object x)
    void py_destructor(core.K x)


cpdef from_pyobject(p: object,
                    ktype: Optional[KType] = None,
                    cast: bool = False,
                    handle_nulls: bool = False,
):
    # q foreign objects internally are a 2 value list, where the type number has been set to 112
    # The first value is a destructor function to be called when q drops the object
    # The second value is a pointer to the object being wrapped
    cdef core.K x = core.knk(2, py_destructor, py_to_pointer(p))
    x.t = 112
    Py_INCREF(p)
    return factory(<uintptr_t>x, False)


def _from_iterable(x: Any,
                   ktype: Optional[KType] = None,
                   *,
                   cast: bool = False,
                   handle_nulls: bool = False,
                   ):
    if type(x) is np.ndarray:
        return from_numpy_ndarray(x, ktype, cast=cast, handle_nulls=handle_nulls)
    elif type(x) is list:
        return from_list(x, ktype, cast=cast, handle_nulls=handle_nulls)
    elif type(x) is tuple:
        return from_tuple(x, ktype, cast=cast, handle_nulls=handle_nulls)
    elif type(x) is dict:
        return from_dict(x, ktype, cast=cast, handle_nulls=handle_nulls)
    elif type(x) is range:
        return from_range(x, ktype, cast=cast, handle_nulls=handle_nulls)
    elif type(x) is slice:
        return from_slice(x, ktype, cast=cast, handle_nulls=handle_nulls)
    else:
        raise _conversion_TypeError(x, type(x), ktype)


def _from_str_like(x: Any,
                   ktype: Optional[KType] = None,
                   *,
                   cast: bool = False,
                   handle_nulls: bool = False,
                   ):
    if type(x) is str:
        return from_str(x, ktype)
    elif type(x) is bytes:
        return from_bytes(x, ktype)
    elif type(x) is np.ndarray:
        return from_numpy_ndarray(x, ktype, cast=cast)
    elif type(x) is list:
        return from_list(x, ktype, cast=cast)
    elif isinstance(x, Path):
        return from_pathlib_path(x, ktype)
    else:
        raise _conversion_TypeError(x, type(x), ktype)


_converter_from_ktype = {
    k.List: _from_iterable,

    k.BooleanAtom: from_int,
    k.GUIDAtom: from_uuid_UUID,
    k.ByteAtom: from_int,
    k.ShortAtom: from_int,
    k.IntAtom: from_int,
    k.LongAtom: from_int,
    k.RealAtom: from_float,
    k.FloatAtom: from_float,
    k.CharAtom: _from_str_like,
    k.SymbolAtom: _from_str_like,
    k.TimestampAtom: from_datetime_datetime,
    k.MonthAtom: from_datetime_datetime,
    k.DateAtom: from_datetime_date,

    k.DatetimeAtom: from_datetime_datetime,
    k.TimespanAtom: from_datetime_timedelta,
    k.MinuteAtom: from_datetime_timedelta,
    k.SecondAtom: from_datetime_timedelta,
    k.TimeAtom: from_datetime_timedelta,

    k.BooleanVector: _from_iterable,
    k.GUIDVector: _from_iterable,
    k.ByteVector: _from_iterable,
    k.ShortVector: _from_iterable,
    k.IntVector: _from_iterable,
    k.LongVector: _from_iterable,
    k.RealVector: _from_iterable,
    k.FloatVector: _from_iterable,
    k.CharVector: _from_str_like,
    k.SymbolVector: _from_iterable,
    k.TimestampVector: _from_iterable,
    k.MonthVector: _from_iterable,
    k.DateVector: _from_iterable,
    k.TimespanVector: _from_iterable,
    k.MinuteVector: _from_iterable,
    k.SecondVector: _from_iterable,
    k.TimeVector: _from_iterable,

    k.SymbolicFunction: from_str,

    k.Foreign: from_pyobject,
    122: from_pyobject,
}


_converter_from_python_type = {
    bool: from_int,
    np.bool_: from_int,
    int: from_int,
    np.byte: from_int,
    np.ubyte: from_int,
    np.short: from_int,
    np.ushort: from_int,
    np.intc: from_int,
    np.uintc: from_int,
    np.int_: from_int,
    np.uint: from_int,
    np.longlong: from_int,
    np.ulonglong: from_int,
    np.int8: from_int,
    np.int16: from_int,
    np.int32: from_int,
    np.int64: from_int,
    np.uint8: from_int,
    np.uint16: from_int,
    np.uint32: from_int,
    np.uint64: from_int,
    np.intp: from_int,
    np.uintp: from_int,

    np.half: from_float,
    np.float16: from_float,
    np.single: from_float,
    np.float32: from_float,
    float: from_float,
    np.double: from_float,
    np.longdouble: from_float,
    np.float64: from_float,
    np.float_: from_float,

    str: from_str,
    bytes: from_bytes,
    UUID: from_uuid_UUID,

    datetime.date: from_datetime_date,
    datetime.time: from_datetime_time,
    datetime.datetime: from_datetime_datetime,
    datetime.timedelta: from_datetime_timedelta,
    np.datetime64: from_numpy_datetime64,
    pd._libs.tslibs.timestamps.Timestamp: from_numpy_datetime64,
    np.timedelta64: from_numpy_timedelta64,
    type(pd.NaT): from_pandas_nat,

    list: from_list,
    tuple: from_tuple,
    dict: from_dict,
    range: from_range,
    slice: from_slice,

    np.ndarray: from_numpy_ndarray,
    np.ma.MaskedArray: from_numpy_ndarray,

    pd.DataFrame: from_pandas_dataframe,
    pd.Series: from_pandas_series,
    pd.Index: from_pandas_index,
    pd.core.indexes.range.RangeIndex: from_pandas_index,
    pd.core.indexes.datetimes.DatetimeIndex: from_pandas_index,
    pd.core.indexes.multi.MultiIndex: from_pandas_index,
    pd.core.indexes.category.CategoricalIndex: from_pandas_index,
    pd.Categorical: from_pandas_categorical,
}


if not pandas_2:
    _converter_from_python_type[pd.core.indexes.numeric.Int64Index] = from_pandas_index
    _converter_from_python_type[pd.core.indexes.numeric.Float64Index] = from_pandas_index
else:
    _converter_from_python_type[pd._libs.tslibs.timedeltas.Timedelta] = from_pandas_timedelta

class ToqModule(ModuleType):
    # TODO: `cast` should be set to False at the next major release (KXI-12945)
    def __call__(self, x: Any, ktype: Optional[KType] = None, *, cast: bool = None, handle_nulls: bool = False) -> k.K:
        ktype = _resolve_k_type(ktype)

        check_ktype = False
        try:
            check_ktype = ktype is not None \
                and issubclass(ktype, k.Vector)
        except TypeError:
            check_ktype = False

        if x is not None and check_ktype   \
            and not hasattr(x, '__iter__') \
            and type(x) is not slice:
            x = [x]

        ktype_conversion=False
        try:
            if ktype is not None and ktype in _converter_from_ktype:
                converter = _converter_from_ktype[ktype]
                ktype_conversion = True
        except BaseException:
            pass

        if x is None:
            converter = from_none
        elif isinstance(x, type(pd.NaT)):
            converter = from_pandas_nat
        elif isinstance(x, k.K):
            converter = from_pykx_k
        elif ktype_conversion:
            pass
        else:
            if type(x) in _converter_from_python_type:
                converter = _converter_from_python_type[type(x)]

            elif isinstance(x, Path):
                converter = from_pathlib_path
            elif x is Ellipsis:
                converter = from_ellipsis
            elif pa is not None and type(x).__module__.startswith('pyarrow') and hasattr(x, 'to_pandas'):
                converter = from_arrow
            elif hasattr(x, 'fileno'):
                converter = from_fileno
            elif callable(x): # Check this last because many Python objects are incidentally callable.
                converter = from_callable
            elif isinstance(x, k.GroupbyTable):
                return self(x.tab, ktype=ktype, cast=cast, handle_nulls=handle_nulls)
            else:
                converter = _default_converter
        if type(ktype)==dict:
            if not licensed:
                raise PyKXException("Use of dictionary based conversion unsupported in unlicensed mode")
            if pa is not None:
                if not type(x) in [pd.DataFrame, pa.Table]:
                    raise TypeError(f"'ktype' not supported as dictionary for {type(x)}")
            else:
                if not type(x) == pd.DataFrame:
                    raise TypeError(f"'ktype' not supported as dictionary for {type(x)}")
        return converter(x, ktype, cast=cast, handle_nulls=handle_nulls)


# Set the module type for this module to `ToqModule` so that it can be called via `__call__`.
toq = sys.modules[__name__]
toq.__class__ = ToqModule

# HACK: inject in the definition to avoid a circular import error
k.toq = toq

"""Cast between Python types.

When `pykx.toq` is called with the argument `cast=True`, a helper functions in this file
will be called to cast the input Python object to the target Python type before converting to a
`pykx.K` object. These helper functions are not intended to be called used by end users
directly and their underlying API may change.

"""

import datetime

import numpy as np

from .constants import INF_INT16, INF_INT32, INF_INT64, NULL_INT16, NULL_INT32, NULL_INT64


def _cast_TypeError(x, input_type, output_type, exc=None):
    reason = f"Can not cast {input_type} {x!r} to {output_type}"
    if exc is not None:
        reason += f" because of following exception: {exc!r}"
    return TypeError(reason)


_overflow_error_by_dtype = {
    np.int64: OverflowError('numpy.int64 must be in the range [-9223372036854775808, 9223372036854775807]'),  # noqa
    np.int32: OverflowError('numpy.int32 must be in the range [-2147483648, 2147483647]'),
    np.int16: OverflowError('numpy.int16 short must be in the range [-32768, 32767]'),
    np.uint8: OverflowError('numpy.uint8 byte must be in the range [0, 255]'),
    np.bool_: OverflowError('numpy.bool must be in the range [0, 1]'),
}


def cast_numpy_ndarray_to_dtype(x, dtype):  # noqa
    _serr_const = '%s of %s'
    try:
        if x.dtype.char == 'U' or x.dtype.char == 'S':
            if dtype.kind == 'i' or dtype.kind == 'u':
                x = x.astype('int64')
            elif dtype.kind == 'f':
                x = x.astype(float)
            else:
                raise _cast_TypeError(x, _serr_const % (type(x).__name__, x.dtype), dtype)
    except Exception as e:
        if type(e) is TypeError:
            raise e
        else:
            raise _cast_TypeError(x, _serr_const % (type(x).__name__, x.dtype), dtype, e)

    try:
        if dtype == np.int16:
            if (x < NULL_INT16).any() or (x > INF_INT16).any():
                raise _overflow_error_by_dtype[np.int16]
        elif dtype == np.int32:
            if (x < NULL_INT32).any() or (x > INF_INT32).any():
                raise _overflow_error_by_dtype[np.int32]
        elif dtype == np.int64:
            if (x < NULL_INT64).any() or (x > INF_INT64).any():
                raise _overflow_error_by_dtype[np.int64]
    except TypeError as e:
        raise _cast_TypeError(x, _serr_const % (type(x).__name__, x.dtype), dtype, e)

    try:
        casted = x.astype(dtype)
    except Exception as e:
        raise _cast_TypeError(x, _serr_const % (type(x).__name__, x.dtype), dtype, e)

    return casted


def cast_to_python_date(x):
    if type(x) is datetime.datetime:
        return x.date()
    if type(x) is np.datetime64:
        return x.astype(datetime.date)
    else:
        raise _cast_TypeError(x, type(x), datetime.date)


def cast_to_python_time(x):
    if type(x) is datetime.datetime:
        return x.time()
    if type(x) is np.datetime64:
        return x.astype(datetime.time).time()
    else:
        raise _cast_TypeError(x, type(x), datetime.time)


def cast_to_python_datetime(x):
    if type(x) is datetime.date:
        return datetime.datetime.combine(x, datetime.datetime.min.time())
    elif type(x) is np.datetime64:
        return x.astype(datetime.datetime)
    else:
        raise _cast_TypeError(x, type(x), datetime.datetime)


def cast_to_python_float(x):
    try:
        return float(x)
    except ValueError:
        raise _cast_TypeError(x, type(x), float)


def cast_to_python_int(x):
    try:
        return int(x)
    except ValueError:
        raise _cast_TypeError(x, type(x), int)


def cast_to_python_timedelta(x):
    if type(x) is np.timedelta64:
        if x.dtype == np.dtype('timedelta64[ns]'):
            ns = x.astype(datetime.timedelta)
            days = (ns // 1000000000) // (24*3600)
            secs = (ns // 1000000000) % (24 * 3600)
            usecs = (ns - (days * 24 * 3600000000000) - (secs*1000000000)) // 1000
            return datetime.timedelta(days, secs, usecs)
        else:
            return x.astype(datetime.timedelta)
    else:
        raise _cast_TypeError(x, datetime.date, datetime.datetime)


__all__ = [
    'cast_numpy_ndarray_to_dtype',
    'cast_to_python_date',
    'cast_to_python_time',
    'cast_to_python_datetime',
    'cast_to_python_float',
    'cast_to_python_int',
    'cast_to_python_timedelta',
]


def __dir__():
    return __all__

from libc.stdint cimport *

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.ref cimport Py_INCREF
cimport cython
cimport numpy as np # Imported ambiguously as "np" by convention for using Numpy with Cython.
cimport numpy as cnp # Imported unambiguously as "cnp" to avoid a Cython issue.

cnp.import_array()

from pykx cimport core

import os
import threading
from uuid import UUID

import numpy as np

from ._pyarrow import pyarrow as pa
from .config import licensed, release_gil
from .core import q_lock
from .exceptions import PyArrowUnavailable, QError

from . import numpy_conversions


wrappers = None


def _init(py_module):
    global wrappers
    wrappers = py_module


NPY_BOOL = cnp.NPY_BOOL
NPY_UINT8 = cnp.NPY_UINT8
NPY_INT16 = cnp.NPY_INT16
NPY_INT32 = cnp.NPY_INT32
NPY_INT64 = cnp.NPY_INT64
NPY_FLOAT32 = cnp.NPY_FLOAT32
NPY_FLOAT64 = cnp.NPY_FLOAT64
NPY_DATETIME64_NS = cnp.dtype('datetime64[ns]')
NPY_DATETIME64_M = cnp.dtype('datetime64[M]')
NPY_DATETIME64_D = cnp.dtype('datetime64[D]')
NPY_TIMEDELTA64_NS = cnp.dtype('timedelta64[ns]')
NPY_TIMEDELTA64_M = cnp.dtype('timedelta64[m]')
NPY_TIMEDELTA64_S = cnp.dtype('timedelta64[s]')
NPY_TIMEDELTA64_MS = cnp.dtype('timedelta64[ms]')


NPY_TYPE_ITEMSIZE = {
    NPY_BOOL: np.bool_().itemsize,
    NPY_UINT8: np.uint8().itemsize,
    NPY_INT16: np.int16().itemsize,
    NPY_INT32: np.int32().itemsize,
    NPY_INT64: np.int64().itemsize,
    NPY_FLOAT32: np.float32().itemsize,
    NPY_FLOAT64: np.float64().itemsize
}


def k_vec_to_array(k_vec, np_type: int) -> np.ndarray:
    """Creates a Numpy array that references the K object until the array's data is deallocated."""
    cdef np.npy_intp n = (<core.K><uintptr_t>k_vec._addr).n
    cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, &n, np_type, <void*>_k(k_vec).G0)
    # Increment the Python ref count of k_vec because PyArray_SetBaseObject will steal a reference
    Py_INCREF(k_vec)
    PyArray_SetBaseObject(arr, k_vec)
    return arr


cdef inline uint64_t byteswap64(uint64_t x):
    x = (x & 0x00000000FFFFFFFF) << 32 | (x & 0xFFFFFFFF00000000) >> 32
    x = (x & 0x0000FFFF0000FFFF) << 16 | (x & 0xFFFF0000FFFF0000) >> 16
    x = (x & 0x00FF00FF00FF00FF) << 8  | (x & 0xFF00FF00FF00FF00) >> 8 # noqa
    return x


# A cdef class is used to store the reference in order to guarantee r0 is called
cdef class _K:
    cdef core.K k

    def __cinit__(self, uintptr_t k, bint incref):
        if incref:
            core.r1(<core.K>k)
        self.k = <core.K>k

    def __dealloc__(self):
        core.r0(self.k)

    @property
    def r(self):
        return self.k.r


def k_from_addr(cls, uintptr_t addr, bint incref):
    instance = object.__new__(cls)
    instance._addr = addr
    instance._k = _K(addr, incref)
    instance.__init__(None) # placeholder argument
    return instance


def k_str(self):
    if not licensed:
        return repr(self)
    cdef core.K x = core.k(0, <char* const>'.Q.s', core.r1(_k(self)), NULL)
    if x.n == 0:
        s = ''
    else:
        s = np.asarray(<char[:x.n]><char*>x.G0).tobytes().decode()
    core.r0(x)
    if len(s) and s.endswith(os.linesep):
        # Use `len(s) - X` instead of `-X` because wraparound is disabled here
        s = s[:len(s) - len(os.linesep)] # strip trailing newline appened by .Q.s
    return s


cpdef inline deserialize(x):
    cdef core.K buff = core.kpn(<char*>x, len(x))
    if 0 == core.okx(buff):
        core.r0(buff)
        raise QError('Failed to deserialize supplied non PyKX IPC serialized format object')
    cdef core.K kx = core.ee(core.d9(buff))
    core.r0(buff)
    return factory(<uintptr_t>kx, False)


cpdef inline k_t(x):
    return (<core.K><uintptr_t>x._addr).t


cpdef inline int k_r(x):
    return (<core.K><uintptr_t>x._addr).r


cpdef inline unsigned char k_g(x):
    return (<core.K><uintptr_t>x._addr).g


cpdef inline short k_h(x):
    return (<core.K><uintptr_t>x._addr).h


cpdef inline int k_i(x):
    return (<core.K><uintptr_t>x._addr).i


cpdef inline long long k_j(x):
    return (<core.K><uintptr_t>x._addr).j


cpdef inline float k_e(x):
    return (<core.K><uintptr_t>x._addr).e


cpdef inline double k_f(x):
    return (<core.K><uintptr_t>x._addr).f


def k_s(x) -> bytes:
    return bytes((<core.K><uintptr_t>x._addr).s)


cpdef inline uintptr_t k_k(x):
    return <uintptr_t>(<core.K><uintptr_t>x._addr).k


cpdef inline long long k_n(x):
    return (<core.K><uintptr_t>x._addr).n


cdef inline core.K _k(x):
    return <core.K><uintptr_t>x._addr


cpdef k_unpickle(x):
    cdef bytes as_bytes = x.tobytes()
    cdef core.K kx = core.kpn(as_bytes, len(x))
    kx.t = 4 # Convert from char vector to byte vector
    return factory(<uintptr_t>core.d9(kx), False)


# We pickle to a Numpy array instead of bytes to benefit from Numpy's highly performant pickling.
cpdef k_pickle(x):
    cdef core.K k_serialized = core.b9(6, <core.K><uintptr_t>x._addr)
    serialized = factory(<uintptr_t>k_serialized, False)
    cdef np.npy_intp n = k_serialized.n
    cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, &n, np.NPY_UINT8, <void*>k_serialized.G0)
    # Increment the Python ref count because PyArray_SetBaseObject will steal a reference
    Py_INCREF(serialized)
    PyArray_SetBaseObject(arr, serialized)
    return arr


cpdef k_hash(x):
    cdef core.K serialized = core.b9(6, <core.K><uintptr_t>x._addr)
    return hash(PyBytes_FromStringAndSize(<char*>serialized.G0, <Py_ssize_t>serialized.n))


def pandas_uuid_type_from_arrow(self, array):
    if pa is None:
        raise PyArrowUnavailable # nocov
    if isinstance(array, pa.lib.ChunkedArray):
        if len(array.chunks) > 1:
            raise ValueError(
                "Cannot convert multiple chunks from 'pyarrow.lib.ChunkedArray' to Pandas"
            )
        array = array.chunks[0]
    buf = array.buffers()[1]
    np_array = np.asarray(
        <np.complex128_t[:buf.size // 16]><np.complex128_t*>(<uintptr_t>buf.address)
    ) # XXX: this probably isn't safe...
    return wrappers.PandasUUIDArray(np_array)


def list_unlicensed_getitem(self, Py_ssize_t index):
    cdef long long n = (<core.K><uintptr_t>self._addr).n
    cdef uintptr_t[:] addrs = <uintptr_t[:n]><uintptr_t*>_k(self).G0
    return factory(addrs[index], True)


def guid_vector_unlicensed_getitem(self, Py_ssize_t index):
    cdef long long n = (<core.K><uintptr_t>self._addr).n
    cdef char[:] vector_as_chars = <char[:n*16]><char*>_k(self).G0
    item_as_bytes = bytes(vector_as_chars[index*16:(index+1)*16])
    return wrappers.GUIDAtom(UUID(bytes=item_as_bytes))


def char_vector_unlicensed_getitem(self, Py_ssize_t index):
    cdef long long n = (<core.K><uintptr_t>self._addr).n
    cdef uint8_t[:] x = <uint8_t[:n]><uint8_t*>_k(self).G0
    cdef core.K kx = core.kc(ord(x[index]))
    return factory(<uintptr_t>kx, False)


def symbol_vector_unlicensed_getitem(self, Py_ssize_t index):
    cdef long long n = (<core.K><uintptr_t>self._addr).n
    cdef uintptr_t[:] addrs = <uintptr_t[:n]><uintptr_t*>_k(self).G0
    return wrappers.SymbolAtom(str(<char*>addrs[index], 'utf-8'))


cdef extern from 'include/unlicensed_getitem.h':
    float get_float(core.K k, size_t index)
    double get_double(core.K k, size_t index)
    long long get_long(core.K k, size_t index)
    int get_int(core.K k, size_t index)
    short get_short(core.K k, size_t index)
    unsigned char get_byte(core.K k, size_t index)


def vector_unlicensed_getitem(self, ssize_t index):
    cdef core.K kx = <core.K><uintptr_t>self._addr
    cdef core.K res
    cdef int value
    ktype = type(self)
    if ktype == wrappers.RealVector:
        res = core.ke(get_float(<core.K><uintptr_t>self._addr, index))
    elif ktype == wrappers.FloatVector:
        res = core.kf(get_double(<core.K><uintptr_t>self._addr, index))
    elif ktype == wrappers.DatetimeVector:
        res = core.kz(get_double(<core.K><uintptr_t>self._addr, index))
    elif ktype == wrappers.ShortVector:
        res = core.kh(get_short(<core.K><uintptr_t>self._addr, index))
    elif ktype == wrappers.BooleanVector:
        res = core.kb(get_byte(<core.K><uintptr_t>self._addr, index))
    elif ktype == wrappers.ByteVector:
        res = core.kg(get_byte(<core.K><uintptr_t>self._addr, index))
    elif ktype == wrappers.LongVector:
        res = core.kj(get_long(<core.K><uintptr_t>self._addr, index))
    elif ktype == wrappers.TimespanVector:
        res = core.ktj(-16, get_long(<core.K><uintptr_t>self._addr, index))
    elif ktype == wrappers.TimestampVector:
        res = core.ktj(-12, get_long(<core.K><uintptr_t>self._addr, index))
    else: # int type
        value = get_int(<core.K><uintptr_t>self._addr, index)
        if ktype == wrappers.IntVector:
            res = core.ki(value)
        elif ktype == wrappers.TimeVector:
            res = core.kt(value)
        else: # last of temporal types
            res = core.kd(value)
            if ktype == wrappers.MonthVector:
                res.t = -13
            elif ktype == wrappers.MinuteVector:
                res.t = -17
            elif ktype == wrappers.SecondVector:
                res.t = -18
            # else it is a date and no change is needed
    return factory(<uintptr_t>res, False)


def guid_atom_py(self, bint raw, bint has_nulls, bint stdlib):
    if raw:
        return np.asarray(<np.complex128_t[:1]><np.complex128_t*>_k(self).G0)[0]
    return UUID(bytes=(_k(self).G0)[:16])


def list_np(self, bint raw, bint has_nulls):
    cdef uintptr_t[:] addrs, razed_addrs
    cdef Py_ssize_t i
    cdef long long n = (<core.K><uintptr_t>self._addr).n
    cdef long long n0 = 1
    if raw:
        # Should use `np.NPY_UINTP`, but that isn't defined for whatever reason. It's just a
        # typedef for `np.NPY_UINT`, so we'll use that instead.
        return k_vec_to_array(self, np.NPY_UINT).astype(np.uintp, copy=False)

    if n == 0:
        return np.empty((0,), dtype=object)

    addrs = <uintptr_t[:n]><uintptr_t*>_k(self).G0
    n0 = (<core.K><uintptr_t>addrs[0]).n
    ty = (<core.K><uintptr_t>addrs[0]).t

    # Optimization for nested vectors:
    if licensed and n0 >= 1 and ty >= 0 and ty < 21 and ty != 10:
        # q query to check if should raze then return razed vector or null
        # (should raze if all element are of same size and same vector type except CharVector)
        query = \
            b'{if[1=count distinct count each x;'          \
            b'    if[1=count types: distinct type each x;' \
            b'        if[types[0] in "h"$(1+til 20)_9;'    \
            b'            : raze x]]];'                    \
            b'    0N}'
        razed = factory(<uintptr_t>core.k(0, query, core.r1(_k(self)), NULL), False)
        if isinstance(razed, wrappers.Vector):
            razed = razed.np()
            razed.shape = (n, -1)
            arr = np.empty(n, dtype=object)
            for i in range(n):
                arr[i] = razed[i].view()
            return arr
    # Fallback:
    arr = np.empty(n, dtype=object)
    for i in range(n):
        if 10 == (<core.K>(addrs[i])).t:
            arr[i] = wrappers._rich_convert(k_from_addr(wrappers.CharVector, addrs[i], True))
        else:
            arr[i] = wrappers._rich_convert(factory(addrs[i], True), stdlib=False)
    return arr


def guid_vector_np(self, bint raw, bint has_nulls):
    if raw:
        # XXX: NPY_COMPLEX128 is the only 128 bit wide Numpy type, and GUIDs are 128 bits wide
        return k_vec_to_array(self, np.NPY_COMPLEX128)

    cdef np.npy_intp n = (<core.K><uintptr_t>self._addr).n
    arr = np.PyArray_New(np.ndarray, 1, &n, np.NPY_VOID, NULL, _k(self).G0, 16, 0, None)
    return np.array([UUID(bytes=u.tobytes()) for u in arr], object)


def symbol_vector_np(self, bint raw, bint has_nulls):
    return numpy_conversions.symbol_vector_to_np(self._addr, raw)


def table_init(self):
    cdef core.K kx = _k(self)
    self._keys = factory(<uintptr_t>(<core.K*>kx.k.G0)[0], True)
    self._values = factory(<uintptr_t>(<core.K*>kx.k.G0)[1], True)


def dictionary_init(self):
    cdef core.K kx = _k(self)
    self._keys = factory(<uintptr_t>(<core.K*>kx.G0)[0], True)
    self._values = factory(<uintptr_t>(<core.K*>kx.G0)[1], True)


cdef core.K _function_call_dot(core.K self, core.K args, bint no_gil) except *:
    try:
        with q_lock:
            if no_gil and release_gil:
                with nogil:
                    return core.dot(self, args)
            return core.dot(self, args)
    except BaseException as err:
        raise err


cpdef function_call(self, args, no_gil):
    return factory(<uintptr_t>core.ee(_function_call_dot(_k(self), _k(args), no_gil)), False)


cdef object q_table_type(core.K val):
    cdef core.K x
    x = core.k(0, <char* const>'.Q.qp', core.r1(val), NULL)
    if x.t == -1 and x.g == 1:      # x == 1b
        wrapper = wrappers.PartitionedTable
    elif x.t == -1 and x.g == 0:    # x == 0b
        wrapper = wrappers.SplayedTable
    else:                           # x == 0
        wrapper = wrappers.Table
    core.r0(x)
    return wrapper


cdef extern from 'include/vector_conversion.h':
    object symbol_vector_to_py(core.K s, int raw)
    object char_vector_to_py(core.K s)


def get_symbol_list(s, raw):
    return symbol_vector_to_py(<core.K><uintptr_t>s._addr, 1 if raw else 0)


def get_char_list(s):
    return char_vector_to_py(<core.K><uintptr_t>s._addr)


cdef extern from 'include/foreign.h':
    object foreign_to_python(core.K x)
    object get_numpy_array(object x)
    object _to_bytes_(core.K x, char wait)


cpdef object from_foreign_to_pyobject(x: Foreign):
    return foreign_to_python(<core.K><uintptr_t>x._addr)


cpdef object _to_bytes(mode, x, wait):
    cdef core.K ser
    ser = core.b9(mode, <core.K><uintptr_t>x._addr)
    if ser == NULL:
        return factory(<uintptr_t>core.ee(ser), False, 1)
    res = _to_bytes_(ser, <char>wait)
    return (<unsigned long long><uintptr_t>ser, res)


cpdef decref_numpy_allocated_data(x):
    # this gets the backing numpy array without incrementing the refcount and then drops it
    # decrementing the refcount.
    get_numpy_array(x)


cpdef decref(x):
    core.r0(<core.K><uintptr_t>x)


cdef bint is_pnull(core.K val):
    cdef core.K x
    x = core.k(0, <char* const>'104h ~ type {1b}@', core.r1(val), NULL)
    cdef bint r = x.g
    core.r0(x)
    return r


cdef inline object select_wrapper(core.K k):
    cdef signed char ktype = k.t
    cdef signed char key_ktype
    wrapper = wrappers.type_number_to_pykx_k_type.get(ktype, wrappers.K)
    if wrapper is wrappers._k_unary_primitive:
        if k.g:
            wrapper = wrappers.UnaryPrimitive
        elif licensed and is_pnull(k):
            wrapper = wrappers.ProjectionNull
        else:
            wrapper = wrappers.Identity
    elif wrapper is wrappers._k_table_type:
        wrapper = q_table_type(k) if licensed else wrappers.Table
    elif wrapper is wrappers._k_dictionary_type:
        # XXX: it's possible to have a dictionary that is not a keyed table, but still uses tables
        # as its keys
        key_ktype = (<core.K*>k.G0)[0].t
        value_ktype = (<core.K*>k.G0)[1].t
        wrapper = wrappers.KeyedTable if key_ktype == 98 and value_ktype == 98 else wrappers.Dictionary
    return wrapper


# This dictionary holds a Python exception object raised from Python code run under q, if one
# exists. It maps thread IDs to exception objects. Its purpose is to be used in a
# `raise ex from cause` statement. See `pykx.c::k_py_error` for more about how we handle Python
# exceptions raised from under q.
_current_exception = {}


cdef inline factory(uintptr_t addr, bint incref, bint err_preamble=0):
    wrapper = select_wrapper(<core.K>addr)
    if wrapper is QError:
        q_exception = wrapper(('Failed to serialize IPC message: ' if err_preamble else '') + str((<core.K>addr).s, 'utf-8'))
        # `pop` the exception object out to prevent it from being handled multiple times.
        _current_exception_in_thread = _current_exception.pop(threading.get_ident(), None)
        if _current_exception_in_thread is None:
            # `raise ex from None` is different from `raise ex`, so we can't write this line as
            # `raise q_exception from _current_exception_in_thread`.
            raise q_exception
        else:
            raise q_exception from _current_exception_in_thread
    return k_from_addr(wrapper, addr, incref)


def _factory(addr: int, incref: bool):
    return factory(addr, incref)


def _pyfactory(addr: int, incref: bool, typenum: int, raw: bool = False):
    k_object = factory(addr, incref)
    if typenum != 0:
        if typenum == 1:
            return k_object.py(raw=raw)
        elif typenum == 2:
            return k_object.np(raw=raw)
        elif typenum == 3:
            return k_object.pd(raw=raw)
        elif typenum == 4:
            return k_object.pa(raw=raw)
        elif typenum == 5:
            return k_object
    k_dir = dir(k_object)
    if 'np' in k_dir: # nocov
        return k_object.np(raw=raw) # nocov
    elif 'py' in k_dir: # nocov
        return k_object.py(raw=raw) # nocov
    elif 'pd' in k_dir: # nocov
        return k_object.pd(raw=raw) # nocov
    elif 'pa' in k_dir: # nocov
        return k_object.pa(raw=raw) # nocov


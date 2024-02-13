from libc.stdint cimport *

cimport numpy as np

from uuid import UUID


cdef extern from 'numpy/arrayobject.h':
    int PyArray_SetBaseObject(np.ndarray, object) except -1


cpdef deserialize(x)


cdef factory(uintptr_t addr, bint incref, bint err_preamble=*)

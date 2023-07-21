from libc.stdint cimport uintptr_t

cimport numpy as np

from pykx cimport core


cdef extern from 'k.h':
    cdef core.I khpunc(core.S hostname,
                       core.I port,
                       core.S credentials,
                       core.I timeout,
                       core.I capability)
    cdef core.V kclose(core.I x)

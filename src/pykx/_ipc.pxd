from libc.stdint cimport uintptr_t

cimport numpy as np

from pykx cimport core


cdef extern from 'k.h':
    cdef int khpunc(char* hostname, int port, char* credentials, int timeout, int capability)
    cdef void kclose(int x)

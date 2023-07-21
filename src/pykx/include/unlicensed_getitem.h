#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "k.h"


static F get_double(K k, size_t index) {
    return kF(k)[index];
}


static E get_float(K k, size_t index) {
    return kE(k)[index];
}


static J get_long(K k, size_t index) {
    return kJ(k)[index];
}


static I get_int(K k, size_t index) {
    return kI(k)[index];
}


static H get_short(K k, size_t index) {
    return kH(k)[index];
}


static G get_byte(K k, size_t index) {
    return kG(k)[index];
}


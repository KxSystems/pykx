#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "k.h"


static double get_double(K k, size_t index) {
    return kF(k)[index];
}


static float get_float(K k, size_t index) {
    return kE(k)[index];
}


static long long get_long(K k, size_t index) {
    return kJ(k)[index];
}


static int get_int(K k, size_t index) {
    return kI(k)[index];
}


static short get_short(K k, size_t index) {
    return kH(k)[index];
}


static unsigned char get_byte(K k, size_t index) {
    return kG(k)[index];
}


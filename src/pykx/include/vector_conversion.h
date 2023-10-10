#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "k.h"

static PyObject* symbol_vector_to_py(K s, int raw) {
    PyObject* res = PyList_New(s->n);
    char** sl = kS(s);
    if (raw == 1) {
        for (int i = 0; i < s->n; i++) {
            PyObject* py_str = NULL;
            char* str = sl[i];
            py_str = PyBytes_FromString(str);
            PyList_SET_ITEM(res, i, py_str);
        }
    } else {
        for (int i = 0; i < s->n; i++) {
            PyObject* py_str = NULL;
            char* str = sl[i];
            py_str = PyUnicode_FromString(str);
            PyList_SET_ITEM(res, i, py_str);
        }
    }
    return res;
}

static PyObject* char_vector_to_py(K s) {
    PyObject* py_str = PyBytes_FromStringAndSize(kC(s), s->n);
    return py_str;
}


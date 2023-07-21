#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "k.h"

static PyObject* get_py_ptr(K f) {
    return (PyObject*)kK(f)[1];
}


static void py_destructor(K x) {
    int g = PyGILState_Ensure();
    Py_XDECREF((PyObject*)kK(x)[1]);
    PyGILState_Release(g);
}


static uintptr_t py_to_pointer(PyObject* x) {
    return (uintptr_t)x;
}

static PyObject* pyobject_to_long_addr(PyObject* x) {
    return PyLong_FromSize_t((size_t)x);
}


static PyObject* get_numpy_array(PyObject* np_addr) {
    return (PyObject*)PyLong_AsSize_t(np_addr);
}


static PyObject* foreign_to_python(K x) {
    Py_INCREF((PyObject*)kK(x)[1]);
    return (PyObject*)kK(x)[1];
}

static PyObject* _to_bytes_(K k, char wait) {
    // Index 1 of the serialized message is used to let q know if the message should be responded to immediately
    // or if the message should be considered async and no reply is necessary
    // A value of 2 at index 1 implies that the message is a response
    kG(k)[1] = wait;
    return PyMemoryView_FromMemory((S)kG(k), (Py_ssize_t)k->n, PyBUF_READ);
}

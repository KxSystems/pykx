#define PY_SSIZE_T_CLEAN
#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "k.h"


static PyObject* symbol_vector_to_np(PyObject* self, PyObject* args) {
    long long addr;
    int raw;
    if (!PyArg_ParseTuple(args, "Li", &addr, &raw)) Py_RETURN_NONE;
    K symbol_vector = (K)(uintptr_t)addr;
    npy_intp const dims[] = {symbol_vector->n};
    PyArray_Descr* descr = PyArray_DescrFromType(NPY_OBJECT);
    PyArrayObject* arr = PyArray_Zeros(1, dims, descr, 0);

    PyObject** data = (PyObject**)PyArray_DATA(arr);
    char** sl = kS(symbol_vector);
    if (raw == 1) {
        for (int i = 0; i < dims[0]; i++){
            PyObject* py_str = NULL;
            char* str = sl[i];
            py_str = PyBytes_FromString(str);
            data[i] = py_str;
        }
    } else {
        for (int i = 0; i < dims[0]; i++){
            PyObject* py_str = NULL;
            char* str = sl[i];
            py_str = PyUnicode_FromString(str);
            data[i] = py_str;
        }
    }
    return (PyObject*)arr;
}


static PyMethodDef _NumpyMethods[] = {
    {"symbol_vector_to_np", symbol_vector_to_np, METH_VARARGS, "Convert a K SymbolVector into a numpy array."},
    {NULL, NULL, 0, NULL} // Sentinel
};


static struct PyModuleDef _numpymodule = {
    PyModuleDef_HEAD_INIT,
    "numpy_conversions",       // Module name.
    NULL,           // Module documentation; may be NULL.
    -1,             // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    _NumpyMethods,  // Module methods.
    NULL,           // Module slots.
    NULL,           // A traversal function to call during GC traversal of the module object.
    NULL,           // A clear function to call during GC clearing of the module object.
    NULL            // A function to call during deallocation of the module object.
};


PyMODINIT_FUNC PyInit_numpy_conversions(void) {
    import_array();
    return PyModule_Create(&_numpymodule);
}

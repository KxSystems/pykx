#define PY_SSIZE_T_CLEAN
#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <Python.h>


static PyObject* clear_all_errors(PyObject* self, PyObject* args) {
    PyErr_Clear();
    Py_RETURN_NONE;
}


static PyMethodDef _HelperMethods[] = {
    {"clean_errors", clear_all_errors, METH_NOARGS, "Clean the python error state."},
    {NULL, NULL, 0, NULL} // Sentinel
};


static struct PyModuleDef _helpermodule = {
    PyModuleDef_HEAD_INIT,
    "_pykx_helpers",// Module name.
    NULL,           // Module documentation; may be NULL.
    -1,             // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    _HelperMethods,  // Module methods.
    NULL,           // Module slots.
    NULL,           // A traversal function to call during GC traversal of the module object.
    NULL,           // A clear function to call during GC clearing of the module object.
    NULL            // A function to call during deallocation of the module object.
};


PyMODINIT_FUNC PyInit__pykx_helpers(void) {
    return PyModule_Create(&_helpermodule);
}

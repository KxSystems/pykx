#define KXVER 3
#define PY_SSIZE_T_CLEAN

#include <dlfcn.h>
#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "k.h"

#define buf_from_k(x) (uintptr_t)(x->G0)
#define k_from_buf(x) (K)(x - offsetof(struct k0, G0))

static void (*r0_ptr)(K);
static K (*k_ptr)(int, const char*, ...);
static K (*ktn_ptr)(int, long long);
void* q_lib;


uint8_t k_type_to_size[] = {(uint8_t)sizeof(K), 1, 16, 0, 1, 2, 4, 8, 4, 8, 1, 8, 8, 4, 4, 8, 8, 4, 4, 4};

long gc_enabled = -1;


typedef struct {
    uintptr_t(*malloc)(size_t);
    uintptr_t(*calloc)(size_t, size_t);
    uintptr_t(*realloc)(uintptr_t, size_t);
    void (*free)(uintptr_t);
} Allocator;


uintptr_t k_malloc(Allocator* ctx, size_t sz) {
    uintptr_t r = buf_from_k(ktn_ptr(KG, sz));
    return r;
}


uintptr_t k_calloc(Allocator* ctx, size_t sz) {
    uintptr_t r = memset(buf_from_k(ktn_ptr(KG, sz)), 0, sz);
    return r;
}


void k_free(Allocator* ctx, uintptr_t p, npy_uintp sz) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    if (p) {
        K x = k_from_buf(p);
        r0_ptr(x);
        int should_dealloc = x->r == 0;
        if (should_dealloc) {
            if (gc_enabled == -1) {
                gc_enabled = PyLong_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("pykx.config")), "k_gc"))
                    && PyLong_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("pykx.core")), "licensed"));
            }
            if (gc_enabled) {
                k_ptr(0, ".Q.gc[]", NULL);
            }
        }
    }
    PyGILState_Release(gstate);
}


uintptr_t k_realloc(Allocator* ctx, uintptr_t p, npy_uintp sz) {
    K k = k_from_buf(p);

    // Temporarily set the type to bytes.
    signed char t = k->t;
    k->t = KG;

    // Calculate the size of the vector in bytes, and temporarily set k->n to that.
    long long prev_size;
    assert(t < 20 && 0 <= t);
    prev_size = k->n * k_type_to_size[t];
    k->n = prev_size;

    int diff = sz - prev_size;

    if (diff != 0) {
        K prev = k;
        k = ktn_ptr(KG, sz);
        memcpy(k->G0, prev->G0, diff > 0 ? prev_size : sz);
        r0_ptr(prev);
    }

    k->t = t;
    k->n = sz / k_type_to_size[t];
    uintptr_t r = buf_from_k(k);
    return r;
}


static Allocator pykx_handler_ctx = {
    malloc,
    calloc,
    realloc,
    free
};


static PyDataMem_Handler pykx_handler = {
    "pykx_allocator",
    1,
    {
        &pykx_handler_ctx,
        k_malloc,
        k_calloc,
        k_realloc,
        k_free
    }
};


static PyObject* original_handler = NULL;


static PyObject* numpy_activate_pykx_allocators(PyObject* self, PyObject* args) {
    PyObject* pykx_handler_capsule = PyCapsule_New(&pykx_handler, "mem_handler", NULL);
    original_handler = PyDataMem_SetHandler(pykx_handler_capsule);
    Py_DECREF(pykx_handler_capsule);
    Py_RETURN_NONE;
}


static PyObject* numpy_deactivate_pykx_allocators(PyObject* self, PyObject* args) {
    if (original_handler != NULL) PyDataMem_SetHandler(original_handler);
    Py_RETURN_NONE;
}


static PyObject* init_numpy_ctx(PyObject* self, PyObject* args) {
    Py_ssize_t r0_addr;
    Py_ssize_t k_addr;
    Py_ssize_t ktn_addr;
    PyArg_ParseTuple(args, "nnn", &r0_addr, &k_addr, &ktn_addr);
    r0_ptr = (void*)(uintptr_t)r0_addr;
    ktn_ptr = (K)(uintptr_t)ktn_addr;
    k_ptr = (K)(uintptr_t)k_addr;
    Py_RETURN_NONE;
}


static PyMethodDef _NumpyMethods[] = {
    {"activate_pykx_allocators", numpy_activate_pykx_allocators, METH_NOARGS, "See `pykx.activate_numpy_allocator` for details."},
    {"deactivate_pykx_allocators", numpy_deactivate_pykx_allocators, METH_NOARGS, "See `pykx.deactivate_numpy_allocator` for details."},
    {"init_numpy_ctx", init_numpy_ctx, METH_VARARGS, "See `pykx.init_numpy_ctx` for details."},
    {NULL, NULL, 0, NULL} // Sentinel
};


static struct PyModuleDef _numpymodule = {
    PyModuleDef_HEAD_INIT,
    "_numpy",       // Module name.
    NULL,           // Module documentation; may be NULL.
    -1,             // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    _NumpyMethods,  // Module methods.
    NULL,           // Module slots.
    NULL,           // A traversal function to call during GC traversal of the module object.
    NULL,           // A clear function to call during GC clearing of the module object.
    NULL            // A function to call during deallocation of the module object.
};


PyMODINIT_FUNC PyInit__numpy(void) {
    import_array();
    return PyModule_Create(&_numpymodule);
}

#define KXVER 3

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#include <dlfcn.h>
#include <stdio.h>
#include <stdbool.h>
#include "k.h"
#include "Python.h"
#include "foreign.h"


static void (*r0_ptr)(K);
static K (*r1_ptr)(K);
static K (*ki_ptr)(long long);
static K (*kj_ptr)(long long);
static K (*ks_ptr)(char*);
static K (*ktn_ptr)(int, long long);
static K (*knk_ptr)(int, ...);
static K (*kpn_ptr)(char*, long long);
static K (*kp_ptr)(char*);
static K (*k_ptr)(int, char*, ...);
void* q_lib;

static PyObject* sys;
static PyObject* builtins;
static PyObject* toq_module;
static PyObject* toq;
static PyObject* wrappers_module;
static PyObject* py_wrappers_module;
static PyObject* exception_tracker;
static PyObject* factory;
static PyObject* pyfactory;
static PyObject* k_factory;
static PyObject* k_dict_converter;
static PyObject* UUID;
static PyObject* thread_get_ident;
static PyObject* POSITIONAL_ONLY;
static PyObject* POSITIONAL_OR_KEYWORD;
static PyObject* VAR_POSITIONAL;
static PyObject* KEYWORD_ONLY;
static PyObject* VAR_KEYWORD;
static PyObject* error_preamble;


int pykx_flag = -1;
int init_ptrs = 0;
bool pykx_threading = false;


EXPORT K k_pykx_init(K k_q_lib_path, K _pykx_threading) {
    if (_pykx_threading->g)
        pykx_threading = true;
    q_lib = dlopen(k_q_lib_path->s, RTLD_NOW | RTLD_GLOBAL);
    r0_ptr = (void (*)(K))dlsym(q_lib, "r0");
    r1_ptr = (K (*)(K))dlsym(q_lib, "r1");
    k_ptr = (K (*)(int, char*, ...))dlsym(q_lib, "k");
    ki_ptr = (K (*)(long long))dlsym(q_lib, "ki");
    kj_ptr = (K (*)(long long))dlsym(q_lib, "kj");
    ks_ptr = (K (*)(char*))dlsym(q_lib, "ks");
    ktn_ptr = (K (*)(int, long long))dlsym(q_lib, "ktn");
    knk_ptr = (K (*)(int, ...))dlsym(q_lib, "knk");
    kpn_ptr = (K (*)(char*, long long))dlsym(q_lib, "kpn");
    kp_ptr = (K (*)(char*))dlsym(q_lib, "kp");

    Py_Initialize();
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    sys = PyModule_GetDict(PyImport_ImportModule("sys"));
    builtins = PyModule_GetDict(PyImport_ImportModule("builtins"));
    toq_module = PyModule_GetDict(PyImport_AddModule("pykx.toq"));
    toq = PyDict_GetItemString(toq_module, "toq");
    wrappers_module = PyModule_GetDict(PyImport_AddModule("pykx._wrappers"));
    py_wrappers_module = PyModule_GetDict(PyImport_AddModule("pykx.wrappers"));
    exception_tracker = PyDict_GetItemString(wrappers_module, "_current_exception");
    factory = PyDict_GetItemString(wrappers_module, "_factory");
    pyfactory = PyDict_GetItemString(wrappers_module, "_pyfactory");
    k_factory = PyDict_GetItemString(py_wrappers_module, "_internal_k_list_wrapper");
    k_dict_converter = PyDict_GetItemString(py_wrappers_module, "_internal_k_dict_to_py");
    UUID = PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("uuid")), "UUID");
    thread_get_ident = PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("threading")), "get_ident");
    PyObject* Parameter = PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("inspect")), "Parameter");
    POSITIONAL_ONLY = PyObject_GetAttrString(Parameter, "POSITIONAL_ONLY");
    POSITIONAL_OR_KEYWORD = PyObject_GetAttrString(Parameter, "POSITIONAL_OR_KEYWORD");
    VAR_POSITIONAL = PyObject_GetAttrString(Parameter, "VAR_POSITIONAL");
    KEYWORD_ONLY = PyObject_GetAttrString(Parameter, "KEYWORD_ONLY");
    VAR_KEYWORD = PyObject_GetAttrString(Parameter, "VAR_KEYWORD");
    error_preamble = PyUnicode_FromString("Error in pykx q extension library: ");

    init_ptrs = 1;

    PyGILState_Release(gstate);
    return (K)0;
}


K raise_k_error(char* error_str) {
    K z = ks_ptr(error_str);
    z->t = -128;
    return z;
}


static K create_foreign(PyObject* p) {
    K x = knk_ptr(2, py_destructor, (uintptr_t)p);
    x->t = 112;
    Py_INCREF(p);
    return x;
}

static int check_py_foreign(K x){return x->t==112 && x->n==2 && *kK(x)==(K)py_destructor;}

EXPORT K k_check_python(K x){return kb(check_py_foreign(x));}

void flush_stdout() {
    PyObject* out = PyDict_GetItemString(sys, "stdout");
    if ( PyObject_HasAttrString(out, "flush") ) {
        PyObject_CallMethod(out, "flush", NULL);
    }
}

K k_py_error() {
    if (!PyErr_Occurred()) return (K)0;

    PyObject* ex_type;
    PyObject* ex_value;
    PyObject* ex_traceback;
    PyErr_Fetch(&ex_type, &ex_value, &ex_traceback);
    PyErr_NormalizeException(&ex_type, &ex_value, &ex_traceback);
    if (ex_traceback) PyException_SetTraceback(ex_value, ex_traceback);

    // Build a q error object with the repr of the exception value as its message. The full
    // traceback is provided as the cause to the QError that will be raised.
    PyObject* ex_repr = PyObject_CallMethod(ex_value, "__repr__", NULL);
    K k = ks_ptr((char*)PyUnicode_AsUTF8(ex_repr));
    k->t = -128;
    Py_XDECREF(ex_repr);

    // Store the Python exception in `k._current_exception` so that when the k module raises the
    // QError it can provide the Python exception as cause via
    // `raise QError from _current_exception[threading.getident()]`.
    // We use a dict to track exceptions on a per-thread basis. Once the exception has been used
    // by the k module, it is set to `None` in the `exception_tracker` dictionary.
    PyObject* thread_id = PyObject_CallFunction(thread_get_ident, NULL);
    if (PyDict_SetItem(exception_tracker, thread_id, ex_value) && PyErr_Occurred()) {
        PyErr_WriteUnraisable(ex_value);
    }
    if (ex_value)
        Py_XDECREF(ex_value);
    if (ex_traceback)
        Py_XDECREF(ex_traceback);
    if (thread_id)
        Py_XDECREF(thread_id);

    return k;
}


static PyObject* k_to_py_cast(K x, K typenum, K israw) {
    if (pykx_threading)
        return Py_None;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    if (x->t == 112) {
        PyGILState_Release(gstate);
        return get_py_ptr(x);
    }

    PyObject* factory_args = PyTuple_New(4);
    PyTuple_SetItem(factory_args, 0, Py_BuildValue("K", (unsigned long long)x));
    PyTuple_SetItem(factory_args, 1, Py_True);
    PyTuple_SetItem(factory_args, 2, Py_BuildValue("l", (long)typenum->j));
    if (israw->g) {
        PyTuple_SetItem(factory_args, 3, Py_True);
        Py_INCREF(Py_True);
    } else {
        PyTuple_SetItem(factory_args, 3, Py_False);
        Py_INCREF(Py_False);
    }
    Py_INCREF(Py_True);

    PyObject* new_val = PyObject_CallObject(pyfactory, factory_args);
    Py_XDECREF(factory_args);

    PyGILState_Release(gstate);
    return new_val;
}


static PyObject* k_to_py_list(K x) {
    if (pykx_threading)
        return Py_None;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    if (x->t == 112) {
        PyGILState_Release(gstate);
        return get_py_ptr(x);
    }

    PyObject* factory_args = PyTuple_New(2);
    PyTuple_SetItem(factory_args, 0, Py_BuildValue("K", (unsigned long long)x));
    PyTuple_SetItem(factory_args, 1, Py_True);
    Py_INCREF(Py_True);

    PyObject* new_val = (PyObject*)PyObject_CallObject(k_factory, factory_args);
    Py_XDECREF(factory_args);
    PyGILState_Release(gstate);
    return new_val;
}


EXPORT K k_to_py_foreign(K x, K typenum, K israw) {
    K k;
    if (pykx_threading)
        return raise_k_error("pyForeign is not supported when using PYKX_THREADING");
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyObject* p = k_to_py_cast(x, typenum, israw);
    if ((k = k_py_error())) {
        PyGILState_Release(gstate);
        return k;
    }
    k = create_foreign(p);
    Py_XDECREF(p);
    PyGILState_Release(gstate);
    return k;
}


void construct_args_kwargs(PyObject* params, PyObject** args, PyObject** kwargs, size_t num_args, PyObject* py_k_args[8]) {
    PyObject* param;
    PyObject* kind;
    PyObject* name;
    PyObject* tmp; // temp var to aid with proper reference management
    *args = PyList_New(0);
    *kwargs = PyDict_New();
    for (size_t i = 0; i < num_args - 1; ++i) {
        param = PyList_GetItem(params, i);
        Py_INCREF(param);
        kind = PyObject_GetAttrString(param, "kind");
        if (PyObject_RichCompareBool(kind, POSITIONAL_ONLY, Py_EQ) || PyObject_RichCompareBool(kind, POSITIONAL_OR_KEYWORD, Py_EQ)) {
            PyList_Append(*args, py_k_args[i]);
        }
        else if (PyObject_RichCompareBool(kind, VAR_POSITIONAL, Py_EQ)) {
            for (Py_ssize_t j = 0; j < PySequence_Size(py_k_args[i]); ++j) {
                tmp = PySequence_GetItem(py_k_args[i], j);
                PyList_Append(*args, tmp);
                Py_XDECREF(tmp);
            }
        }
        else if (PyObject_RichCompareBool(kind, KEYWORD_ONLY, Py_EQ)) {
            name = PyObject_GetAttrString(param, "name");
            PyDict_SetItem(*kwargs, name, py_k_args[i]);
            Py_XDECREF(name);
        }
        else if (PyObject_RichCompareBool(kind, VAR_KEYWORD, Py_EQ)) {
            // turn the keys into strings, because the keys of a kwargs dict must be strings, not pykx.SymbolAtom objects
            PyObject* d = PyObject_CallFunctionObjArgs(PyDict_GetItemString(builtins, "dict"), py_k_args[i], NULL);
            PyObject* tmp = PyObject_CallMethod(d, "keys", NULL);
            PyObject* keys = PyObject_CallFunctionObjArgs(PyDict_GetItemString(builtins, "list"), tmp, NULL);
            Py_XDECREF(tmp);
            for (Py_ssize_t j = 0; j < PySequence_Size(keys); ++j) {
                tmp = PySequence_GetItem(keys, j);
                PyObject* key = PyObject_CallMethod(tmp, "py", NULL);
                PyObject* value = PyObject_CallMethod(d, "pop", "(O)", tmp);
                Py_XDECREF(tmp);
                PyDict_SetItem(d, key, value);
                Py_XDECREF(value);
                Py_XDECREF(key);
            }
            Py_XDECREF(keys);
            PyDict_Update(*kwargs, d);
            Py_XDECREF(d);
        }
        Py_XDECREF(param);
        Py_XDECREF(kind);
    }
    tmp = *args;
    *args = PyList_AsTuple(*args);
    Py_XDECREF(tmp);
}


// k_eval_or_exec == 0 -> eval the code string
// k_eval_or_exec == 1 -> exec the code string
EXPORT K k_pyrun(K k_ret, K k_eval_or_exec, K as_foreign, K k_code_string) {

    if (pykx_threading)
        return raise_k_error("pyrun is not supported when using PYKX_THREADING");
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    K k;

    // If a char atom is provided instead of a char vector, make it into a char vector:
    if (k_code_string->t == -10) {
        char str[1] = {k_code_string->g};
        k_code_string = kpn_ptr(str, 1);
    }

    if (k_code_string->t != 10) {
        PyGILState_Release(gstate);
        return raise_k_error("String input expected for code evaluation/execution.");
    }

    void* code_string = PyMem_Calloc(k_code_string->n + 1, 1);
    strncpy(code_string, (const char *)k_code_string->G0, k_code_string->n);
    PyObject* ctx = PyModule_GetDict(PyImport_AddModule("__main__"));
    PyObject* py_ret = PyRun_String(
        code_string,
        k_eval_or_exec->g ? Py_file_input : Py_eval_input,
        ctx,
        ctx
    );
    PyMem_Free(code_string);

    if (!k_ret->g) {
        if ((k = k_py_error())) {
            flush_stdout();
            Py_XDECREF(py_ret);
            PyGILState_Release(gstate);
            return k;
        } else Py_XDECREF(py_ret);
        flush_stdout();
        PyGILState_Release(gstate);
        return (K)0;
    }

    if ((k = k_py_error())) {
        flush_stdout();
        Py_XDECREF(py_ret);
        PyGILState_Release(gstate);
        return k;
    }
    if (as_foreign->g) {
        k = (K)create_foreign(py_ret);
        flush_stdout();
        Py_XDECREF(py_ret);
        PyGILState_Release(gstate);
        return k;
    }
    PyObject* py_k_ret = PyObject_CallFunctionObjArgs(toq, py_ret, NULL);
    Py_XDECREF(py_ret);
    if ((k = k_py_error())) {
        flush_stdout();
        Py_XDECREF(py_k_ret);
        PyGILState_Release(gstate);
        return k;
    }
    PyObject* py_addr = PyObject_GetAttrString(py_k_ret, "_addr");
    Py_XDECREF(py_k_ret);
    k = (K)PyLong_AsLongLong(py_addr);
    Py_XDECREF(py_addr);
    flush_stdout();
    PyGILState_Release(gstate);
    return k;
}


inline long long modpow(long long base, long long exp, long long mod) {
    long long result = 1;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}


EXPORT K k_modpow(K k_base, K k_exp, K k_mod_arg) {
    K result;
    K k_mod;

    if (k_mod_arg->t == 101 && k_mod_arg->g == 0) {
        k_mod = kj_ptr(9223372036854775807); // default modulo is 2^63-1
    }
    else {
        k_mod = r1_ptr(k_mod_arg);
    }

    if (k_base->t >= 0 && k_exp->t >= 0 && k_mod->t >= 0) {
        if (k_base->n != k_exp->n || k_exp->n != k_mod->n) {
            result = ks_ptr("length");
            result->t = -128;
        } else {
            result = ktn_ptr(7, k_base->n);
            for (long long x = 0; x < k_base->n; ++x) kJ(result)[x] = modpow(kJ(k_base)[x], kJ(k_exp)[x], kJ(k_mod)[x]);
        }
    } else if (k_base->t >= 0 && k_exp->t >= 0) {
        if (k_base->n != k_exp->n) {
            result = ks_ptr("length");
            result->t = -128;
        }
        else {
            result = ktn_ptr(7, k_base->n);
            for (long long x = 0; x < k_base->n; ++x) kJ(result)[x] = modpow(kJ(k_base)[x], kJ(k_exp)[x], k_mod->j);
        }
    } else if (k_base->t >= 0 && k_mod->t >= 0) {
        if (k_base->n != k_mod->n) {
            result = ks_ptr("length");
            result->t = -128;
        }
        result = ktn_ptr(7, k_base->n);
        for (long long x = 0; x < k_base->n; ++x) kJ(result)[x] = modpow(kJ(k_base)[x], k_exp->j, kJ(k_mod)[x]);
    } else if (k_base->t >= 0) {
        result = ktn_ptr(7, k_base->n);
        for (long long x = 0; x < k_base->n; ++x) kJ(result)[x] = modpow(kJ(k_base)[x], k_exp->j, k_mod->j);
    } else if (k_exp->t >= 0 && k_mod->t >= 0) {
        if (k_exp->n != k_mod->n) {
            result = ks_ptr("length");
            result->t = -128;
        }
        else {
            result = ktn_ptr(7, k_exp->n);
            for (long long x = 0; x < k_exp->n; ++x) kJ(result)[x] = modpow(k_base->j, kJ(k_exp)[x], kJ(k_mod)[x]);
        }
    } else if (k_exp->t >= 0) {
        result = ktn_ptr(7, k_exp->n);
        for (long long x = 0; x < k_exp->n; ++x) kJ(result)[x] = modpow(k_base->j, kJ(k_exp)[x], k_mod->j);
    } else if (k_mod->t >= 0) {
        result = ktn_ptr(7, k_mod->n);
        for (long long x = 0; x < k_mod->n; ++x) kJ(result)[x] = modpow(k_base->j, k_exp->j, kJ(k_mod)[x]);
    } else {
        result = kj_ptr(modpow(k_base->j, k_exp->j, k_mod->j));
    }
    r0_ptr(k_mod);
    return result;
}


EXPORT K foreign_to_q(K f, K b) {
    if (pykx_threading)
        return raise_k_error("foreignToq is not supported when using PYKX_THREADING");
    if (f->t != 112)
        return raise_k_error("Expected foreign object for call to .pykx.toq");
    if (!check_py_foreign(f))
        return raise_k_error("Provided foreign object is not a Python object");
    K k;
    int gstate = PyGILState_Ensure();

    PyObject* pyobj = get_py_ptr(f);
    Py_INCREF(pyobj);

    PyObject* toq_args = PyTuple_New(2);
    PyTuple_SetItem(toq_args, 0, pyobj);
    PyTuple_SetItem(toq_args, 1, Py_BuildValue(""));

    PyObject* _kwargs = PyDict_New();
    PyDict_SetItemString(_kwargs, "strings_as_char", PyBool_FromLong((long)b->g));

    PyObject* qpy_val = PyObject_Call(toq, toq_args, _kwargs);
    if ((k = k_py_error())) {
        PyGILState_Release(gstate);
        return k;
    }

    PyObject* k_addr = PyObject_GetAttrString(qpy_val, "_addr");
    if ((k = k_py_error())) {
        Py_XDECREF(toq_args);
        Py_XDECREF(_kwargs);
        Py_XDECREF(k_addr);
        Py_XDECREF(qpy_val);
        PyGILState_Release(gstate);
        return k;
    }
    long long _addr = PyLong_AsLongLong(k_addr);
    K res = (K)(uintptr_t)_addr;
    r1_ptr(res);
    Py_XDECREF(toq_args);
    Py_XDECREF(_kwargs);
    Py_XDECREF(qpy_val);
    Py_XDECREF(k_addr);

    PyGILState_Release(gstate);
    return res;
}


EXPORT K repr(K as_repr, K f) {
    if (pykx_threading)
        return raise_k_error("repr is not supported when using PYKX_THREADING");
    K k;
    if (f->t != 112) {
        if (as_repr->g){
            if (f->t == 105) {
                return raise_k_error("Expected a foreign object for .pykx.repr, try unwrapping the foreign object with `.");
            }
            return raise_k_error("Expected a foreign object for .pykx.repr");
        } else {
            if (f->t == 105) {
                return raise_k_error("Expected a foreign object for .pykx.print, try unwrapping the foreign object with `.");
            }
            return raise_k_error("Expected a foreign object for .pykx.print");
        }
    }
    int gstate = PyGILState_Ensure();
    PyObject* p = get_py_ptr(f);
    PyObject* repr = PyObject_Repr(p);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    Py_XDECREF(repr);
    if (!as_repr->g) {
        const char *bytes = PyBytes_AS_STRING(str);
        PySys_WriteStdout("%s\n", bytes);
        flush_stdout();
        PyGILState_Release(gstate);
        Py_XDECREF(str);
        return (K)0;
    }
    if ((k = k_py_error())) {
        flush_stdout();
        PyGILState_Release(gstate);
        Py_XDECREF(str);
        return k;
    }
    const char *chars = PyBytes_AS_STRING(str);
    flush_stdout();
    PyGILState_Release(gstate);
    return kp_ptr(chars);
}


EXPORT K get_attr(K f, K attr) {
    if (pykx_threading)
        return raise_k_error("getattr is not supported when using PYKX_THREADING");
    K k;
    if (f->t != 112) {
        if (f->t == 105) {
            return raise_k_error("Expected foreign object for call to .pykx.getattr, try unwrapping the foreign object with `.");
        }
        return raise_k_error("Expected foreign object for call to .pykx.getattr");
    }
    if (attr->t != -11) {
        return raise_k_error("Expected a SymbolAtom for the attribute to get in .pykx.getattr");
    }
    int gstate = PyGILState_Ensure();
    PyObject* p = get_py_ptr(f);
    PyObject* _attr = Py_BuildValue("s", attr->s);
    PyObject* pres = PyObject_GetAttr(p, _attr);
    if ((k = k_py_error())) {
        Py_XDECREF(pres);
        PyGILState_Release(gstate);
        return k;
    }
    K res = create_foreign(pres);
    Py_XDECREF(_attr);
    PyGILState_Release(gstate);
    return res;
}


EXPORT K get_global(K attr) {
    if (pykx_threading)
        return raise_k_error("getGlobal is not supported when using PYKX_THREADING");
    K k;
    if (attr->t != -11) {
        return raise_k_error("Expected a SymbolAtom for the attribute to get in .pykx.get");
    }
    int gstate = PyGILState_Ensure();
    PyObject* p = PyImport_AddModule("__main__");
    if ((k = k_py_error())) {
        PyGILState_Release(gstate);
        return k;
    }
    PyObject* _attr = Py_BuildValue("s", attr->s);
    PyObject* pres = PyObject_GetAttr(p, _attr);
    if ((k = k_py_error())) {
        Py_XDECREF(pres);
        PyGILState_Release(gstate);
        return k;
    }
    K res = create_foreign(pres);
    Py_XDECREF(_attr);
    PyGILState_Release(gstate);
    return res;
}


EXPORT K set_global(K attr, K val) {
    if (pykx_threading)
        return raise_k_error("setGlobal is not supported when using PYKX_THREADING");
    K k;
    int gstate = PyGILState_Ensure();

    PyObject* p = PyImport_AddModule("__main__");
    if ((k = k_py_error())) {
        PyGILState_Release(gstate);
        return k;
    }
    PyObject* v = get_py_ptr(val);
    if ((k = k_py_error())) {
        PyGILState_Release(gstate);
        return k;
    }
    PyObject_SetAttrString(p, attr->s, v);
    if ((k = k_py_error())) {
        PyGILState_Release(gstate);
        return k;
    }
    PyGILState_Release(gstate);
    return NULL;
}


EXPORT K set_attr(K f, K attr, K val) {
    if (pykx_threading)
        return raise_k_error("setattr is not supported when using PYKX_THREADING");
    if (f->t != 112) {
        if (f->t == 105) {
            return raise_k_error("Expected foreign object for call to .pykx.setattr, try unwrapping the foreign object with `.");
        }
        return raise_k_error("Expected foreign object for call to .pykx.setattr");
    }
    if (attr->t != -11) {
        return raise_k_error("Expected a SymbolAtom for the attribute to set in .pykx.setattr");
    }
    int gstate = PyGILState_Ensure();
    K k;
    PyObject* p = get_py_ptr(f);
    Py_INCREF(p);
    PyObject* v = get_py_ptr(val);
    if ((k = k_py_error())) {
        PyGILState_Release(gstate);
        return k;
    }
    PyObject_SetAttrString(p, attr->s, v);
    if ((k = k_py_error())) {
        PyGILState_Release(gstate);
        return k;
    }
    PyGILState_Release(gstate);
    return NULL;
}

EXPORT K import(K module) {
    if (pykx_threading)
        return raise_k_error("import is not supported when using PYKX_THREADING");
    K k;
    K res;
    if (module->t != -11)
        return raise_k_error("Module to be imported must be a symbol");
    int gstate = PyGILState_Ensure();

    PyObject* p = PyImport_ImportModule(module->s);
    if ((k = k_py_error())) {
       PyGILState_Release(gstate);
       return k;
    }
    res = create_foreign(p);
    PyGILState_Release(gstate);
    return res;
}



EXPORT K call_func(K f, K has_no_args, K args, K kwargs) {

    if (pykx_threading)
        return raise_k_error("import is not supported when using PYKX_THREADING");
    K k;
    PyObject* pyf = NULL;

    int gstate = PyGILState_Ensure();
    if ((k = k_py_error())) {
        PyGILState_Release(gstate);
        return k;
    }
    pyf = get_py_ptr(f);
    Py_INCREF(pyf);
    if (!PyCallable_Check(pyf)) {
        return raise_k_error("Attempted to call non callable python foreign object");
    }
    int len = (has_no_args->j==0)?0:(int)args->n;
    PyObject* py_params = NULL;
    PyObject* py_kwargs = NULL;
    if (len != 0) {
        py_params = k_to_py_list(args);
        if ((k = k_py_error())) {
            Py_XDECREF(py_params);
            PyGILState_Release(gstate);
            return k;
        }
    } else{
        py_params = PyTuple_New(0);
    }
      

    if ((kK(kwargs)[0])->n != 0) {
        PyObject* factory_args = PyTuple_New(1);
        PyTuple_SetItem(factory_args, 0, Py_BuildValue("K", (unsigned long long)kwargs));
        if ((k = k_py_error())) {
            Py_XDECREF(py_params);
            Py_XDECREF(py_kwargs);
            Py_XDECREF(factory_args);
            PyGILState_Release(gstate);
            return k;
        }
        py_kwargs = PyObject_CallObject(k_dict_converter, factory_args);
        Py_XDECREF(factory_args);

        if ((k = k_py_error())) {
            Py_XDECREF(py_params);
            Py_XDECREF(py_kwargs);
            PyGILState_Release(gstate);
            return k;
        }
    }

    PyObject* pyres = PyObject_Call(pyf, py_params, py_kwargs);
    Py_XDECREF(pyf);
    Py_XDECREF(py_params);
    Py_XDECREF(py_kwargs);

    if ((k = k_py_error())) {
        if (pyres)
            Py_XDECREF(pyres);
        flush_stdout();
        PyGILState_Release(gstate);
        return k;
    }

    K res;
    if (pyres == NULL) {
        pyres = Py_BuildValue("");
    }

    res = create_foreign(pyres);
    Py_XDECREF(pyres);
    flush_stdout();
    PyGILState_Release(gstate);

    return res;
}



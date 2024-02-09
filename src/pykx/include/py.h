#include<stdio.h>
#include<wchar.h>
#include<string.h>
#include<stdlib.h>
#if _WIN32
#include<windows.h>
typedef LONG_PTR L;
#define __thread __declspec(thread)
#define EXP __declspec(dllexport)
#else
#include<dlfcn.h>
typedef ssize_t L;
#define EXP
#endif
#define KXVER 3
#include"k.h"
#define K3(f) K f(K x,K y,K z)

typedef struct _p _p,*P;
struct _p{
    L r;
    P t;
    L n;
    union{
        P* p;
        P v[1];
    };
};
typedef struct{
    L r;
    P t;
    unsigned char* g;
    int n;
    L*c,*s;
    P*b;
    struct{
        L r;
        P t;
        P*o;
        char* k,c,b,f;
        int n,e;
    }*d;
    int f;
}*A;
typedef struct{
    char* n;
    void* m;
    int f;
    char* d;
}D;
#define ZP static P
#define PyGILState_STATE int
#define Py_ssize_t size_t
#define Py_EQ 2
#define Py_file_input 257
#define Py_eval_input 258
#define Py_PRINT_RAW 1
#define Py_INCREF Py_IncRef
#define Py_XDECREF Py_DecRef
#define PyBytes_AS_STRING PyBytes_AsString

//https://docs.python.org/3/c-api/ https://github.com/python/cpython/blob/3.6/PC/python3.def
#define PF \
 X(void,Py_InitializeEx,(int))\
 X(void,Py_Finalize,())\
 X(void,Py_DecRef,(P))\
 X(void,Py_IncRef,(P))\
 X(void,PyErr_Clear,())\
 X(void,PyErr_Fetch,(P*,P*,P*))\
 X(void,PyErr_NormalizeException,(P*,P*,P*))\
 X(P,PyErr_BadArgument,())\
 X(P,PyErr_SetString,(P,char*))\
 X(int,PyGILState_Ensure,())\
 X(void,PyGILState_Release,(int))\
 X(int,PyGILState_Check,())\
 X(void,PyEval_InitThreads,())\
 X(int,PyEval_ThreadsInitialized,())\
 X(void*,PyGILState_GetThisThreadState,())\
 X(void*,PyEval_SaveThread,())\
 X(void,PyEval_RestoreThread,(void*))\
 X(_p,PyExc_RuntimeError,)\
 X(P,PyObject_Str,(P))\
 X(wchar_t*,Py_DecodeLocale,(char*,void*))\
 X(void,Py_SetPythonHome,(wchar_t*))\
 X(void,Py_SetProgramName,(wchar_t*))\
 X(P,PyImport_AddModule,(char*))\
 X(P,PyImport_ImportModule,(char*))\
 X(P,PyObject_GetAttrString,(P,char*))\
 X(P,PyObject_Type,(P))\
 X(P,PyModule_GetDict,(P))\
 X(P,PyDict_GetItemString,(P,char*))\
 X(P,PyDict_SetItemString,(P,char*,P))\
 X(P,PyEval_EvalCode,(P,P,P))\
 X(P,Py_CompileString,(char*,char*,int))\
 X(P,PyCapsule_New,(void*,char*,void*))\
 X(void*,PyCapsule_GetPointer,(P,char*))\
 X(P,PyCFunction_NewEx,(D*,P,P))\
 X(_p,_Py_TrueStruct,)\
 X(_p,_Py_FalseStruct,)\
 X(_p,_Py_NoneStruct,)\
 X(_p,PyLong_Type,)\
 X(_p,PyFloat_Type,)\
 X(_p,PyTuple_Type,)\
 X(_p,PyList_Type,)\
 X(_p,PyDict_Type,)\
 X(_p,PyUnicode_Type,)\
 X(_p,PyBytes_Type,)\
 X(int,PyType_IsSubtype,(P,P))\
 X(long long,PyLong_AsLongLongAndOverflow,(P,int*))\
 X(double,PyFloat_AsDouble,(P))\
 X(int,PyObject_RichCompareBool,(P,P,int))\
 X(char*,PyUnicode_AsUTF8AndSize,(P,L*))\
 X(int,PyBytes_AsStringAndSize,(P,char**,L*))\
 X(char*,PyBytes_AsString,(P))\
 X(P,PyBool_FromLong,(long))\
 X(P,PyErr_Occurred,())\
 X(void,PyErr_WriteUnraisable,(P))\
 X(int,PyException_SetTraceback,(P,P))\
 X(void*,PyMem_Calloc,(size_t,size_t))\
 X(void,PyMem_Free,(void*))\
 X(void,PyErr_SetObject,(P,P))\
 X(void,PyErr_Print,())\
 X(P,PyObject_Repr,(P))\
 X(int,PyObject_Print,(P,FILE*,int))\
 X(P,PyLong_FromLongLong,(long long))\
 X(long long,PyLong_AsLongLong,(P))\
 X(P,PyLong_FromSize_t,(size_t))\
 X(P,PyFloat_FromDouble,(double))\
 X(P,PyUnicode_FromStringAndSize,(char*,L))\
 X(P,PyUnicode_FromString,(char*))\
 X(P,PyUnicode_AsEncodedString,(P,char*,char*))\
 X(P,PyBytes_FromStringAndSize,(char*,L))\
 X(P,PySequence_List,(P))\
 X(P,Py_BuildValue,(char*,...))\
 X(P,PyTuple_New,(L))\
 X(P,PyList_New,(L))\
 X(P,PyDict_New,())\
 X(size_t,PySequence_Size,(P))\
 X(P,PySequence_GetItem,(P, size_t))\
 X(int,PyList_Append,(P,P))\
 X(P,PyList_GetItem,(P,size_t))\
 X(P,PyTuple_GetItem,(P,size_t))\
 X(P,PyDict_GetItemWithError,(P,P))\
 X(P,PyDict_SetItem,(P,P,P))\
 X(int,PyTuple_SetItem,(P,size_t,P))\
 X(P,PyDict_Keys,(P))\
 X(P,PyDict_Values,(P))\
 X(int,PyDict_Update,(P,P))\
 X(P,PyList_AsTuple,(P))\
 X(P,PyObject_CallFunctionObjArgs,(P,...))\
 X(P,PyObject_CallFunction,(P,char*,...))\
 X(P,PyObject_CallMethod,(P,char*,char*,...))\
 X(P,PyObject_Call,(P,P,P))\
 X(P,PyObject_CallObject,(P,P))\
 X(int,PyObject_HasAttr,(P,P))\
 X(int,PyObject_HasAttrString,(P,char*))\
 X(P,PyObject_GetAttr,(P,P))\
 X(int,PyObject_SetAttrString,(P,char*,P))\
 X(char*,PyUnicode_AsUTF8,(P))\
 X(int,PyCallable_Check,(P))\
 X(P,PyRun_String,(char*,int,P,P))\
 X(P,PyImport_Import,(P))\
 X(int,Py_IsInitialized,())\
 X(int,PySys_WriteStdout,(char*,...))\

//https://docs.scipy.org/doc/numpy/reference/c-api.html https://github.com/numpy/numpy/blob/master/numpy/core/code_generators/numpy_api.py
#undef PyCFunction_New
#define PyCFunction_New(x,y) PyCFunction_NewEx(x,y,NULL)
#define NF \
 X(_p,PyArray_Type,,2)\
 X(_p,PyGenericArrType_Type,,10)\
 X(A,PyArray_NewCopy,(A,int),85)\
 X(A,PyArray_New,(P,int,L*,int,L*,void*,int,int,P),93)\
 X(int,PyArray_SetBaseObject,(A,P),282)\
 X(A,PyArray_FromScalar,(P,P),61)\

#define X(r,n,a,...) typedef r T##n;static T##n(*n)a;
PF NF
#undef X

static int ta(A a){int c[]={KB,-1,KG,KH,-1,KI,-1,4==sizeof(long)?KI:KJ,-1,KJ,-1,KE,KF,-1,-1,-1,-1,0,-1,-1,-1},t=a->d->n;return t>20?-1:c[t];}
static int tk(K x){int c[]={-1,0,-1,-1,-1,3,5,9,11,12,-1,-1,9,5,5,12,9,5,5,5},t=xt>0?xt:-xt;return t>19?-1:c[t];}

static int pyl(char* l){
#if _WIN32
 HMODULE d=LoadLibrary(l);
#define X(r,n,a) U(n=(T##n(*)a)GetProcAddress(d,#n))
#else
 void*d=dlopen(l,RTLD_NOW|RTLD_GLOBAL);
#define X(r,n,a) U(n=dlsym(d,#n))
#endif
 P(!d,0)PF
#undef X
 return 1;}

#define BUFFSIZE 4096

static int pyn(void**v){
#define X(r,n,a,i) U(n=(T##n(*)a)v[i])
 NF
 return 1;}
#ifdef __clang__
#pragma clang diagnostic ignored "-Wincompatible-pointer-types-discards-qualifiers"
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif __GNUC__
#ifndef __arm__
#if __GNUC__ >= 5
#pragma GCC diagnostic ignored "-Wdiscarded-qualifiers"
#endif
#endif
#endif

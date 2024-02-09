cdef extern from 'stdarg.h':
    ctypedef struct va_list:
        pass


cdef extern from 'dlfcn.h':
    void* dlopen(const char* filename, int flag)
    char* dlerror()
    void* dlsym(void* handle, const char* symbol)
    int dlclose(void* handle)

    unsigned int RTLD_LAZY
    unsigned int RTLD_NOW
    unsigned int RTLD_GLOBAL
    unsigned int RTLD_LOCAL
    unsigned int RTLD_NODELETE
    unsigned int RTLD_NOLOAD
    unsigned int RTLD_DEEPBIND
    unsigned int RTLD_DEFAULT
    long unsigned int RTLD_NEXT


cdef extern from 'k.h':
    cdef struct k0:
        signed char m   # internal (?)
        signed char a   # internal (?)
        signed char t   # type
        char u             # attributes
        int r             # reference count
        unsigned char g             # char
        short h             # short
        int i             # int
        long long j             # long
        float e             # real
        double f             # float
        char* s             # symbol
        k0* k           # dictionary representation of a table (and more?)
        long long n             # number of elements in a vector
        unsigned char G0[1]         # vector data (extends past the end of this struct)

    cdef struct _U:
        unsigned char g[16]
    ctypedef _U U

    ctypedef k0* K

cdef int (*qinit)(int, char**, char*, char*, char*)

cdef unsigned char* (*kG)(K x)
cdef unsigned char* (*kC)(K x)
cdef U* (*kU)(K x)
cdef char** (*kS)(K x)
cdef short* (*kH)(K x)
cdef int* (*kI)(K x)
cdef long long* (*kJ)(K x)
cdef float* (*kE)(K x)
cdef double* (*kF)(K x)
cdef K* (*kK)(K x)

cdef void (*_shutdown_thread)()
cdef K (*b9)(int mode, K x)
cdef K (*d9)(K x)
cdef int (*dj)(int date)
cdef K (*dl)(void* f, long long n)
cdef K (*dot)(K x, K y) nogil
cdef K (*ee)(K x)
cdef K (*ja)(K* x, void*)
cdef K (*jk)(K* x, K y)
cdef K (*js)(K* x, char* s)
cdef K (*jv)(K* x, K y)
cdef K (*k)(int handle, const char* s, ...) nogil
cdef K (*knogil)(void* x, char* code, void* a1, void* a2, void* a3, void* a4, void* a5, void* a6, void* a7, void* a8) nogil
cdef K (*ka)(int t)
cdef K (*kb)(int x)
cdef K (*kc)(int x)
cdef void (*kclose)(int x)
cdef K (*kd)(int x)
cdef K (*ke)(double x)
cdef K (*kf)(double x)
cdef K (*kg)(int x)
cdef K (*kh)(int x)
cdef int (*khpunc)(char* v, int w, char* x, int y, int z)
cdef K (*ki)(int x)
cdef K (*kj)(long long x)
cdef K (*knk)(int n, ...)
cdef K (*knt)(long long n, K x)
cdef K (*kp)(char* x)
cdef K (*kpn)(char* x, long long n)
cdef K (*krr)(const char* s)
cdef K (*ks)(char* x)
cdef K (*kt)(int x)
cdef K (*ktd)(K x)
cdef K (*ktj)(short _type, long long x)
cdef K (*ktn)(int _type, long long length)
cdef K (*ku)(U x)
cdef K (*kz)(double x)
cdef void (*m9)()
cdef int (*okx)(K x)
cdef K (*orr)(const char*)
cdef void (*r0)(K k)
cdef K (*r1)(K k)
cdef void (*sd0)(int d)
cdef void (*sd0x)(int d, int f)
cdef K (*sd1)(int d, f)
cdef K (*sd1)(int d, f)
cdef char* (*sn)(char* s, long long n)
cdef char* (*ss)(char* s)
cdef K (*sslInfo)(K x)
cdef K (*vak)(int x, const char* s, va_list l)
cdef K (*vaknk)(int, va_list l)
cdef int (*ver)()
cdef K (*xD)(K x, K y)
cdef K (*xT)(K x)
cdef int (*ymd)(int year, int month, int day)

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
    ctypedef char* S
    ctypedef char C
    ctypedef unsigned char G
    ctypedef short H
    ctypedef int I
    ctypedef long long J
    ctypedef float E
    ctypedef double F
    ctypedef void V

    cdef struct k0:
        signed char m   # internal (?)
        signed char a   # internal (?)
        signed char t   # type
        C u             # attributes
        I r             # reference count
        G g             # char
        H h             # short
        I i             # int
        J j             # long
        E e             # real
        F f             # float
        S s             # symbol
        k0* k           # dictionary representation of a table (and more?)
        J n             # number of elements in a vector
        G G0[1]         # vector data (extends past the end of this struct)

    cdef struct _U:
        G g[16]
    ctypedef _U U

    ctypedef k0* K

cdef I (*qinit)(I, C**, C*, C*, C*)

cdef G* (*kG)(K x)
cdef G* (*kC)(K x)
cdef U* (*kU)(K x)
cdef S* (*kS)(K x)
cdef H* (*kH)(K x)
cdef I* (*kI)(K x)
cdef J* (*kJ)(K x)
cdef E* (*kE)(K x)
cdef F* (*kF)(K x)
cdef K* (*kK)(K x)

cdef K (*b9)(I mode, K x)
cdef K (*d9)(K x)
cdef I (*dj)(I date)
cdef K (*dl)(V* f, J n)
cdef K (*dot)(K x, K y) nogil
cdef K (*ee)(K x)
cdef K (*ja)(K* x, V*)
cdef K (*jk)(K* x, K y)
cdef K (*js)(K* x, S s)
cdef K (*jv)(K* x, K y)
cdef K (*k)(I handle, const S s, ...) nogil
cdef K (*knogil)(void* x, char* code, void* a1, void* a2, void* a3, void* a4, void* a5, void* a6, void* a7, void* a8) nogil
cdef K (*ka)(I t)
cdef K (*kb)(I x)
cdef K (*kc)(I x)
cdef V (*kclose)(I x)
cdef K (*kd)(I x)
cdef K (*ke)(F x)
cdef K (*kf)(F x)
cdef K (*kg)(I x)
cdef K (*kh)(I x)
cdef I (*khpunc)(S v, I w, S x, I y, I z)
cdef K (*ki)(I x)
cdef K (*kj)(J x)
cdef K (*knk)(I n, ...)
cdef K (*knt)(J n, K x)
cdef K (*kp)(S x)
cdef K (*kpn)(S x, J n)
cdef K (*krr)(const S s)
cdef K (*ks)(S x)
cdef K (*kt)(I x)
cdef K (*ktd)(K x)
cdef K (*ktj)(H _type, J x)
cdef K (*ktn)(I _type, J length)
cdef K (*ku)(U x)
cdef K (*kz)(F x)
cdef V (*m9)()
cdef I (*okx)(K x)
cdef K (*orr)(const S)
cdef V (*r0)(K k)
cdef K (*r1)(K k)
cdef V (*sd0)(I d)
cdef V (*sd0x)(I d, I f)
cdef K (*sd1)(I d, f)
cdef K (*sd1)(I d, f)
cdef S (*sn)(S s, J n)
cdef S (*ss)(S s)
cdef K (*sslInfo)(K x)
cdef K (*vak)(I x, const S s, va_list l)
cdef K (*vaknk)(I, va_list l)
cdef I (*ver)()
cdef K (*xD)(K x, K y)
cdef K (*xT)(K x)
cdef I (*ymd)(I year, I month, I day)

#define KXVER 3

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#include <dlfcn.h>
#include "k.h"
#include "Python.h"


// Equivalent to starting Python with the `-S` flag. Allows us to edit some global config variables
// before `site.main()` is called.
int Py_NoSiteFlag = 1;


EXPORT K k_init_python(K k_libpython_path) {
    void* libpython = dlopen(k_libpython_path->s, RTLD_NOW | RTLD_GLOBAL);
    ((void (*)())dlsym(libpython, "Py_Initialize"))();
    return (K)0;
}

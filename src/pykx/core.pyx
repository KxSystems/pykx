import os, platform
from pathlib import Path
from threading import RLock
from typing import List, Tuple

from .util import num_available_cores
from .config import _is_enabled


def _normalize_qargs(user_args: List[str]) -> Tuple[bytes]:
    skip_indexes = [] # The indexes of user args to skip because they were already processed.
    normalized_args = [b'q'] # First arg is always the name of the program.

    # Add the -q flag if the user does not already provide it:
    if '-q' not in user_args:
        normalized_args.append(b'-q')

    # Add the -s flag with the number of cores the user specified, or default to all cores:
    try:
        s_index = user_args.index('-s')
    except ValueError: # -s was not specified
        num_cores_as_bytes = str(num_available_cores()).encode()
    else:
        try:
            num_cores_user_arg = user_args[s_index + 1]
            num_cores_as_bytes = str(int(num_cores_user_arg)).encode()
            skip_indexes.extend([s_index, s_index + 1])
        except IndexError as ex:
            raise ValueError(
                "Missing argument for '-s' (number of secondary threads) in $QARGS"
            ) from ex
        except ValueError as ex:
            raise ValueError(
                "Invalid argument for '-s' (number of secondary threads) in $QARGS: "
                f'{num_cores_user_arg!r}'
            ) from ex
    normalized_args.extend([b'-s', num_cores_as_bytes])

    return (
        *normalized_args,
        *(x.encode() for i, x in enumerate(user_args) if i not in skip_indexes)
    )


cdef int _qinit(int (*qinit)(int, char**, char*, char*, char*), qhome_str: str, qlic_str: str, args: List[str]) except *:
    normalized_args = _normalize_qargs(args)
    cdef int argc = len(normalized_args)
    cdef char** argv = <char**>PyMem_Malloc(sizeof(char*) * argc)
    for i, arg in enumerate(normalized_args):
        argv[i] = strncpy(<char*>PyMem_Malloc(len(arg) + 1), arg + b'\0', len(arg) + 1)
    qhome_bytes = bytes(Path(__file__).parent.absolute()/'lib')
    qlic_bytes = qlic_str.encode()
    init_code = qinit(argc, argv, qhome_bytes, qlic_bytes, NULL)
    os.environ['QHOME'] = qhome_str
    return init_code


cdef char* _libq_path
cdef void* _q_handle


# The `PYKX_QINIT_CHECK` env var indicates that the current process is a Python subprocess made to
# check if qinit can be initialized without error. The error might be something relatively harmless
# like a license error, but could also cause a segfault. The subprocess allows us to safely attempt
# calling `_qinit`, and return the result for the parent to decide if it should use licensed mode,
# or fallback to unlicensed mode.
# Data is passed in via the `PYKX_QINIT_CHECK` env var to reduce PyKX's import time - why do the
# work to gather that data when the parent process has already done it?
qinit_check_data = os.environ.get('PYKX_QINIT_CHECK')
if qinit_check_data is not None:                                                   # nocov
    _core_q_lib_path, _qhome_str, _qlic_str, _qargs = qinit_check_data.split(';')  # nocov
    import shlex                                                                   # nocov
    _qargs = list(shlex.split(_qargs))                                             # nocov
    _libq_path_py = _core_q_lib_path.encode()                                      # nocov
    _libq_path = _libq_path_py                                                     # nocov
    _q_handle = dlopen(_libq_path, RTLD_NOW | RTLD_GLOBAL)                         # nocov
    qinit = <int (*)(int, char**, char*, char*, char*)>dlsym(_q_handle, 'qinit')                   # nocov
    os._exit(_qinit(qinit, _qhome_str, _qlic_str, _qargs))                         # nocov


from libc.string cimport strncpy
from cpython.mem cimport PyMem_Malloc
from libc.stdint cimport *

from warnings import warn
import subprocess
import sys

from .config import find_core_lib, ignore_qhome, k_gc, qargs, qhome, qlic, pykx_lib_dir, \
    release_gil, _set_licensed, under_q, use_q_lock
from .exceptions import PyKXException, PyKXWarning


if '--licensed' in qargs and '--unlicensed' in qargs:
    raise PyKXException("$QARGS includes mutually exclusive flags '--licensed' and '--unlicensed'")


class QLock:
    def __init__(self, timeout):
        if not timeout:
            self._q_lock = None
            self.blocking = False
            self.timeout = 0
        else:
            # Use a RLock (Reentrant lock) so that if a single thread re-requests the lock it can
            # be received again. Only a separate thread running q at the same time is unsafe.
            self._q_lock = RLock()
            self.blocking = False
            try:
                self.timeout = float(timeout)
                if self.timeout > 0.0 or self.timeout == -1.0:
                    self.blocking = True
            except BaseException:
                pass

    def __enter__(self):
        if self._q_lock is not None:
            acquired = False
            if not self.blocking:
                acquired = self._q_lock.acquire(blocking=False)
            else:
                acquired = self._q_lock.acquire(timeout=self.timeout)
            if not acquired:
                raise PyKXException('Attempted to acquire lock on already locked call into q.')
        return self

    def __exit__(self, *exc):
        if self._q_lock is not None:
            self._q_lock.release()
        return False


q_lock = QLock(use_q_lock)


cdef inline K r1k(x):
    return r1(<K><uintptr_t>x._addr)


cdef inline char is_foreign(K k) nogil:
    cdef char ret_val = 0
    if k != NULL and k.t == 112:
        ret_val = 1
    return ret_val


cdef inline uintptr_t _keval(const char* code, K k1, K k2, K k3, K k4, K k5, K k6, K k7, K k8, int handle) except *:
    cdef bint not_safe_to_drop_GIL = \
        is_foreign(k1) or            \
        is_foreign(k2) or            \
        is_foreign(k3) or            \
        is_foreign(k4) or            \
        is_foreign(k5) or            \
        is_foreign(k6) or            \
        is_foreign(k7) or            \
        is_foreign(k8)
    try:
        with q_lock:
            if not not_safe_to_drop_GIL and release_gil and handle == 0:
                # with nogil ensures the gil is dropped during the call into k
                with nogil:
                    return <uintptr_t>knogil(<void*> k, <char* const>code, k1, k2, k3, k4, k5, k6, k7, k8)
            return <uintptr_t>k(handle, <char* const>code, k1, k2, k3, k4, k5, k6, k7, k8, NULL)
    except BaseException as err:
        raise err


def keval(code: bytes, k1=None, k2=None, k3=None, k4=None, k5=None, k6=None, k7=None, k8=None, handle=0):
    # This code is ugly, but Cython can turn it into a switch statement.
    if k1 is None:
        return _keval(code, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, handle)
    elif k2 is None:
        return _keval(code, r1k(k1), NULL, NULL, NULL, NULL, NULL, NULL, NULL, handle)
    elif k3 is None:
        return _keval(code, r1k(k1), r1k(k2), NULL, NULL, NULL, NULL, NULL, NULL, handle)
    elif k4 is None:
        return _keval(code, r1k(k1), r1k(k2), r1k(k3), NULL, NULL, NULL, NULL, NULL, handle)
    elif k5 is None:
        return _keval(code, r1k(k1), r1k(k2), r1k(k3), r1k(k4), NULL, NULL, NULL, NULL, handle)
    elif k6 is None:
        return _keval(code, r1k(k1), r1k(k2), r1k(k3), r1k(k4), r1k(k5), NULL, NULL, NULL, handle)
    elif k7 is None:
        return _keval(code, r1k(k1), r1k(k2), r1k(k3), r1k(k4), r1k(k5), r1k(k6), NULL, NULL, handle)
    elif k8 is None:
        return _keval(code, r1k(k1), r1k(k2), r1k(k3), r1k(k4), r1k(k5), r1k(k6), r1k(k7), NULL, handle)
    else:
        return _keval(code, r1k(k1), r1k(k2), r1k(k3), r1k(k4), r1k(k5), r1k(k6), r1k(k7), r1k(k8), handle)

def _link_qhome():
    update_marker = pykx_lib_dir/'_update_marker'
    subdirs = ('', 'l64', 'm64', 'w64')
    try:
        mark = os.path.getmtime(update_marker)
    except FileNotFoundError:
        pass
    else:
        for subdir in subdirs:
            qhome_subdir = qhome/subdir
            if qhome_subdir.exists() and mark < os.path.getmtime(qhome_subdir):
                break
        else:
            # QHOME has not had any files/directories added/removed since we last updated links.
            return
    # Avoid recursion, but allow for the effective merger via symlinks of the directories under the
    # lib dir that come with PyKX.
    for subdir in subdirs:
        # Remove old symlinks:
        with os.scandir(pykx_lib_dir/subdir) as dir_iter:
            for dir_entry in dir_iter:
                if dir_entry.is_symlink():
                    os.unlink(dir_entry)
        # Add new symlinks:
        try:
            with os.scandir(qhome/subdir) as dir_iter:
                for dir_entry in dir_iter:
                    try:
                        os.symlink(
                            dir_entry,
                            pykx_lib_dir/subdir/dir_entry.name,
                            target_is_directory=dir_entry.is_dir()
                        )
                    except FileExistsError:
                        pass # Skip files/dirs that would overwrite those that come with PyKX.
                    except OSError as ex:  #nocov
                        # Making this a warning instead of an error is particularly important for
                        # Windows, which essentially only lets admins create symlinks.
                        warn('Unable to connect user QHOME to PyKX QHOME via symlinks\n' # nocov
                             f'{ex}',     # nocov
                             PyKXWarning) # nocov
                        return            # nocov
        except FileNotFoundError:
            pass # Skip subdirectories of $QHOME that don't exist.
    update_marker.touch()


if under_q: # nocov
    if '--unlicensed' in qargs: # nocov
        warn("The '--unlicensed' flag has no effect when running under a q process", # nocov
             PyKXWarning) # nocov
    _q_handle = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL) # nocov
    licensed = True # nocov
else:
    # To make Cython happy, we indirectly assign Python values to `_libq_path`
    if '--unlicensed' in qargs:
        _libq_path_py = bytes(find_core_lib('e'))
        _libq_path = _libq_path_py
        _q_handle = dlopen(_libq_path, RTLD_NOW | RTLD_GLOBAL)
        licensed = False
    else:
        if platform.system() == 'Windows': # nocov
            from ctypes.util import find_library # nocov
            if find_library("msvcr100.dll") is None: # nocov
                msvcrMSG = "Needed dependency msvcr100.dll missing. See: https://code.kx.com/pykx/getting-started/installing.html" # nocov
                if '--licensed' in qargs: # nocov
                    raise PyKXException(msvcrMSG)  # nocov
                else: # nocov
                    warn(msvcrMSG, PyKXWarning) # nocov
        _core_q_lib_path = find_core_lib('q')
        licensed = True
        if not _is_enabled('PYKX_UNSAFE_LOAD', '--unsafeload'):
            _qinit_check_proc = subprocess.run(
                (str(Path(sys.executable).as_posix()), '-c', 'import pykx'),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={
                    **os.environ,
                    'PYKX_QINIT_CHECK': ';'.join((
                        str(_core_q_lib_path),
                        str(pykx_lib_dir if ignore_qhome is None else qhome),
                        str(qlic),
                        # Use the env var directly because `config.qargs` has already split the args.
                        os.environ.get('QARGS', ''),
                    )),
                }
            )
            _qinit_output = '    ' + '    '.join(_qinit_check_proc.stdout.strip().splitlines(True))
            if _qinit_check_proc.returncode: # Fallback to unlicensed mode
                if _qinit_output != '    ':
                    _capout_msg = f' Captured output from initialization attempt:\n{_qinit_output}'
                else:
                    _capout_msg = '' # nocov - this can only occur under extremely weird circumstances.
                if '--licensed' in qargs:
                    raise PyKXException(f'Failed to initialize embedded q.{_capout_msg}')
                else:
                    warn(f'Failed to initialize PyKX successfully with the following error: {_capout_msg}', PyKXWarning)
                _libq_path_py = bytes(find_core_lib('e'))
                _libq_path = _libq_path_py
                _q_handle = dlopen(_libq_path, RTLD_NOW | RTLD_GLOBAL)
                licensed = False
        if licensed: # Start in licensed mode
            if 'QHOME' in os.environ and not ignore_qhome:
                # Only link the user's QHOME to PyKX's QHOME if the user actually set $QHOME.
                # Note that `pykx.qhome` has a default value of `./q`, as that is the behavior
                # employed by q.
                try:
                    _link_qhome()
                except BaseException:
                    warn('Failed to link user QHOME directory contents to allow access to PyKX.\n'
                        'To suppress this warning please set the configuration option "PYKX_IGNORE_QHOME" as outlined at:\n'
                        'https://code.kx.com/pykx/user-guide/configuration.html')
            _libq_path_py = bytes(_core_q_lib_path)
            _libq_path = _libq_path_py
            _q_handle = dlopen(_libq_path, RTLD_NOW | RTLD_GLOBAL)
            qinit = <int (*)(int, char**, char*, char*, char*)>dlsym(_q_handle, 'qinit')
            qinit_return_code = _qinit(qinit, str(qhome if ignore_qhome else pykx_lib_dir), str(qlic), list(qargs))
            if qinit_return_code:    # nocov
                dlclose(_q_handle)   # nocov
                licensed = False     # nocov
                raise PyKXException( # nocov
                    f'Non-zero qinit return code {qinit_return_code} despite successful pre-check') # nocov
_set_licensed(licensed)


if k_gc and not licensed:
    raise PyKXException('Early garbage collection requires a valid q license.')



kG = <unsigned char* (*)(K x)>dlsym(_q_handle, 'kG')
kC = <unsigned char* (*)(K x)>dlsym(_q_handle, 'kC')
kU = <U* (*)(K x)>dlsym(_q_handle, 'kU')
kS = <char** (*)(K x)>dlsym(_q_handle, 'kS')
kH = <short* (*)(K x)>dlsym(_q_handle, 'kH')
kI = <int* (*)(K x)>dlsym(_q_handle, 'kI')
kJ = <long long* (*)(K x)>dlsym(_q_handle, 'kJ')
kE = <float* (*)(K x)>dlsym(_q_handle, 'kE')
kF = <double* (*)(K x)>dlsym(_q_handle, 'kF')
kK = <K* (*)(K x)>dlsym(_q_handle, 'kK')

b9 = <K (*)(int mode, K x)>dlsym(_q_handle, 'b9')
d9 = <K (*)(K x)>dlsym(_q_handle, 'd9')
dj = <int (*)(int date)>dlsym(_q_handle, 'dj')
dl = <K (*)(void* f, long long n)>dlsym(_q_handle, 'dl')
dot = <K (*)(K x, K y) nogil>dlsym(_q_handle, 'dot')
ee = <K (*)(K x)>dlsym(_q_handle, 'ee')
ja = <K (*)(K* x, void*)>dlsym(_q_handle, 'ja')
jk = <K (*)(K* x, K y)>dlsym(_q_handle, 'jk')
js = <K (*)(K* x, char* s)>dlsym(_q_handle, 'js')
jv = <K (*)(K* x, K y)>dlsym(_q_handle, 'jv')
k = <K (*)(int handle, const char* s, ...) nogil>dlsym(_q_handle, 'k')
cdef extern from 'include/foreign.h':
    K k_wrapper(void* x, char* code, void* a1, void* a2, void* a3, void* a4, void* a5, void* a6, void* a7, void* a8) nogil
knogil = k_wrapper
ka = <K (*)(int t)>dlsym(_q_handle, 'ka')
kb = <K (*)(int x)>dlsym(_q_handle, 'kb')
kc = <K (*)(int x)>dlsym(_q_handle, 'kc')
kclose = <void (*)(int x)>dlsym(_q_handle, 'kclose')
kd = <K (*)(int x)>dlsym(_q_handle, 'kd')
ke = <K (*)(double x)>dlsym(_q_handle, 'ke')
kf = <K (*)(double x)>dlsym(_q_handle, 'kf')
kg = <K (*)(int x)>dlsym(_q_handle, 'kg')
kh = <K (*)(int x)>dlsym(_q_handle, 'kh')
khpunc = <int (*)(char* v, int w, char* x, int y, int z)>dlsym(_q_handle, 'khpunc')
ki = <K (*)(int x)>dlsym(_q_handle, 'ki')
kj = <K (*)(long long x)>dlsym(_q_handle, 'kj')
knk = <K (*)(int n, ...)>dlsym(_q_handle, 'knk')
knt = <K (*)(long long n, K x)>dlsym(_q_handle, 'knt')
kp = <K (*)(char* x)>dlsym(_q_handle, 'kp')
kpn = <K (*)(char* x, long long n)>dlsym(_q_handle, 'kpn')
krr = <K (*)(const char* s)>dlsym(_q_handle, 'krr')
ks = <K (*)(char* x)>dlsym(_q_handle, 'ks')
kt = <K (*)(int x)>dlsym(_q_handle, 'kt')
ktd = <K (*)(K x)>dlsym(_q_handle, 'ktd')
ktj = <K (*)(short _type, long long x)>dlsym(_q_handle, 'ktj')
ktn = <K (*)(int _type, long long length)>dlsym(_q_handle, 'ktn')
ku = <K (*)(U x)>dlsym(_q_handle, 'ku')
kz = <K (*)(double x)>dlsym(_q_handle, 'kz')
m9 = <void (*)()>dlsym(_q_handle, 'm9')
okx = <int (*)(K x)>dlsym(_q_handle, 'okx')
orr = <K (*)(const char*)>dlsym(_q_handle, 'orr')
r0 = <void (*)(K k)>dlsym(_q_handle, 'r0')
r1 = <K (*)(K k)>dlsym(_q_handle, 'r1')
sd0 = <void (*)(int d)>dlsym(_q_handle, 'sd0')
sd0x = <void (*)(int d, int f)>dlsym(_q_handle, 'sd0x')
sd1 = <K (*)(int d, f)>dlsym(_q_handle, 'sd1')
sd1 = <K (*)(int d, f)>dlsym(_q_handle, 'sd1')
sn = <char* (*)(char* s, long long n)>dlsym(_q_handle, 'sn')
ss = <char* (*)(char* s)>dlsym(_q_handle, 'ss')
sslInfo = <K (*)(K x)>dlsym(_q_handle, 'sslInfo')
vak = <K (*)(int x, const char* s, va_list l)>dlsym(_q_handle, 'vak')
vaknk = <K (*)(int, va_list l)>dlsym(_q_handle, 'vaknk')
ver = <int (*)()>dlsym(_q_handle, 'ver')
xD = <K (*)(K x, K y)>dlsym(_q_handle, 'xD')
xT = <K (*)(K x)>dlsym(_q_handle, 'xT')
ymd = <int (*)(int year, int month, int day)>dlsym(_q_handle, 'ymd')

_r0_ptr = int(<size_t><uintptr_t>r0)
_k_ptr = int(<size_t><uintptr_t>k)
_ktn_ptr = int(<size_t><uintptr_t>ktn)

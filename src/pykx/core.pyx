from cython import NULL
import os, platform
from pathlib import Path
from platform import system
from threading import RLock
from typing import List, Tuple
import re
import sys

from . import beta_features
from .util import num_available_cores
from .config import tcore_path_location, _is_enabled, _license_install, pykx_threading, _check_beta, _get_config_value, pykx_lib_dir, ignore_qhome, lic_path


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


cdef int _qinit(int (*qinit)(int, char**, char*, char*, char*), qhome_str: str, qlic_str: str, ignore_qhome: bool, args: List[str]) except *:
    normalized_args = _normalize_qargs(args)
    cdef int argc = len(normalized_args)
    cdef char** argv = <char**>PyMem_Malloc(sizeof(char*) * argc)
    for i, arg in enumerate(normalized_args):
        argv[i] = strncpy(<char*>PyMem_Malloc(len(arg) + 1), arg + b'\0', len(arg) + 1)
    qhome_bytes = bytes(pykx_lib_dir) if ignore_qhome else qhome_str.encode()
    qlic_bytes = qlic_str.encode()
    init_code = qinit(argc, argv, qhome_bytes, qlic_bytes, NULL)
    os.environ['QHOME'] = qhome_str
    return init_code


cdef char* _libq_path
cdef char* _tcore_path
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
    os._exit(_qinit(qinit, _qhome_str, _qlic_str, ignore_qhome, _qargs))                         # nocov


from libc.string cimport strncpy
from cpython.mem cimport PyMem_Malloc
from libc.stdint cimport *

from warnings import warn
import subprocess
import sys

from .config import find_core_lib, k_gc, qargs, qhome, qlic, pykx_lib_dir, \
    release_gil, _set_licensed, under_q, use_q_lock
from .exceptions import PyKXException, PyKXWarning

final_qhome = str(qhome if ignore_qhome else pykx_lib_dir)

if '--licensed' in qargs and '--unlicensed' in qargs:
    raise PyKXException("$QARGS includes mutually exclusive flags '--licensed' and '--unlicensed'")
elif ('--unlicensed' in qargs or _is_enabled('PYKX_UNLICENSED', '--unlicensed')) & \
     ('--licensed' in qargs or _is_enabled('PYKX_LICENSED', '--licensed')):
    raise PyKXException("User specified options for setting 'licensed' and 'unlicensed' behaviour "
                        "resulting in conflicts")

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
            if pykx_threading:
                with nogil:
                    return <uintptr_t>k(handle, <char* const>code, k1, k2, k3, k4, k5, k6, k7, k8, NULL)
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

cdef void (*init_syms)(char* x)

if not pykx_threading:
    if under_q: # nocov
        if '--unlicensed' in qargs: # nocov
            warn("The '--unlicensed' flag has no effect when running under a q process", # nocov
                 PyKXWarning) # nocov
        _q_handle = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL) # nocov
        licensed = True # nocov
    else:
        # To make Cython happy, we indirectly assign Python values to `_libq_path`
        if '--unlicensed' in qargs or _is_enabled('PYKX_UNLICENSED', '--unlicensed'):
            _libq_path_py = bytes(find_core_lib('e'))
            _libq_path = _libq_path_py
            _q_handle = dlopen(_libq_path, RTLD_NOW | RTLD_GLOBAL)
            licensed = False
        else:
            if platform.system() == 'Windows': # nocov
                from ctypes.util import find_library # nocov
                if find_library("msvcr100.dll") is None: # nocov
                    msvcrMSG = "Needed dependency msvcr100.dll missing. See: https://code.kx.com/pykx/getting-started/installing.html" # nocov
                    if '--licensed' in qargs or _is_enabled('PYKX_LICENSED', '--licensed'): # nocov
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
                            final_qhome,
                            str(qlic),
                            # Use the env var directly because `config.qargs` has already split the args.
                            os.environ.get('QARGS', ''),
                        )),
                    }
                )
                _qinit_output = '    ' + '    '.join(_qinit_check_proc.stdout.strip().splitlines(True))
                _license_message = False
                if _qinit_check_proc.returncode: # Fallback to unlicensed mode
                    if _qinit_output != '    ':
                        _capout_msg = f'Captured output from initialization attempt:\n{_qinit_output}'
                        _lic_location = f'License location used:\n{lic_path}'
                    else:
                        _capout_msg = '' # nocov - this can only occur under extremely weird circumstances.
                        _lic_location = '' # nocov - this additional line is to ensure this code path is covered.
                    if hasattr(sys, 'ps1'):
                        if re.compile('exp').search(_capout_msg):
                            _exp_license = 'Your PyKX license has now expired.\n\n'\
                                           f'{_capout_msg}\n\n'\
                                           f'{_lic_location}\n\n'\
                                           'Would you like to renew your license? [Y/n]: '
                            _license_message = _license_install(_exp_license, True, True, 'exp')
                        elif re.compile('embedq').search(_capout_msg):
                            _ce_license = 'You appear to be using a non kdb Insights license.\n\n'\
                                          f'{_capout_msg}\n\n'\
                                           f'{_lic_location}\n\n'\
                                          'Running PyKX in the absence of a kdb Insights license '\
                                          'has reduced functionality.\nWould you like to install '\
                                          'a kdb Insights personal license? [Y/n]: '
                            _license_message = _license_install(_ce_license, True)
                        elif re.compile('upd').search(_capout_msg):
                            _upd_license = 'Your installed license is out of date for this version'\
                                           ' of PyKX and must be updated.\n\n'\
                                           f'{_capout_msg}\n\n'\
                                           f'{_lic_location}\n\n'\
                                           'Would you like to install an updated kdb '\
                                           'Insights personal license? [Y/n]: '
                            _license_message = _license_install(_upd_license, True)
                if (not _license_message) and _qinit_check_proc.returncode:
                    if '--licensed' in qargs or _is_enabled('PYKX_LICENSED', '--licensed'):
                        raise PyKXException(f'Failed to initialize embedded q.{_capout_msg}\n\n{_lic_location}')
                    else:
                        warn('Failed to initialize PyKX successfully with '
                             f'the following error: {_capout_msg}\n\n{_lic_location}', PyKXWarning)
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
                    except BaseException as e:
                        warn('Failed to link user QHOME directory contents to allow access to PyKX.\n'
                            'To suppress this warning please set the configuration option "PYKX_IGNORE_QHOME" as outlined at:\n'
                            'https://code.kx.com/pykx/user-guide/configuration.html')
                _libq_path_py = bytes(_core_q_lib_path)
                _libq_path = _libq_path_py
                _q_handle = dlopen(_libq_path, RTLD_NOW | RTLD_GLOBAL)
                qinit = <int (*)(int, char**, char*, char*, char*)>dlsym(_q_handle, 'qinit')
                qinit_return_code = _qinit(qinit, final_qhome, str(qlic), ignore_qhome, list(qargs))
                if qinit_return_code:    # nocov
                    dlclose(_q_handle)   # nocov
                    licensed = False     # nocov
                    raise PyKXException( # nocov
                        f'Non-zero qinit return code {qinit_return_code} despite successful pre-check') # nocov
else:
    _check_beta('PYKX Threading')
    beta_features.append('PyKX Threading')
    _libq_path_py = bytes(str(find_core_lib('q')), 'utf-8')
    _tcore_path = tcore_path_location
    _libq_path = _libq_path_py
    _q_handle = dlopen(_tcore_path, RTLD_NOW | RTLD_GLOBAL)

    init_syms = <void (*)(char* x)>dlsym(_q_handle, 'sym_init')
    init_syms(_libq_path)
    qinit = <int (*)(int, char**, char*, char*, char*)>dlsym(_q_handle, 'q_init')

    qinit_return_code = _qinit(qinit, final_qhome, str(qlic), ignore_qhome, list(qargs))
    if qinit_return_code:    # nocov
        dlclose(_q_handle)   # nocov
        licensed = False     # nocov
        if qinit_return_code == 1: # nocov
            raise PyKXException( # nocov
                f'qinit failed because of an invalid license file, please ensure you have a valid'
                'q license installed before using PYKX_THREADING.'
            ) # nocov
        else: # nocov
            raise PyKXException( # nocov
                f'Non-zero qinit return code {qinit_return_code}, failed to initialize '
                'PYKX_THREADING.'
            ) # nocov
    os.environ['QHOME'] = final_qhome
    licensed = True
_set_licensed(licensed)


if k_gc and not licensed:
    raise PyKXException('Early garbage collection requires a valid q license.')


sym_name = lambda x: bytes('_' + x, 'utf-8') if pykx_threading else bytes(x, 'utf-8')

if not pykx_threading:
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

_shutdown_thread = <void (*)()>dlsym(_q_handle, 'shutdown_thread')

cpdef shutdown_thread():
    if pykx_threading:
        _shutdown_thread()


b9 = <K (*)(int mode, K x)>dlsym(_q_handle, sym_name('b9'))
d9 = <K (*)(K x)>dlsym(_q_handle, sym_name('d9'))
dj = <int (*)(int date)>dlsym(_q_handle, sym_name('dj'))
dl = <K (*)(void* f, long long n)>dlsym(_q_handle, sym_name('dl'))
dot = <K (*)(K x, K y) nogil>dlsym(_q_handle, sym_name('dot'))
ee = <K (*)(K x)>dlsym(_q_handle, sym_name('ee'))
ja = <K (*)(K* x, void*)>dlsym(_q_handle, sym_name('ja'))
jk = <K (*)(K* x, K y)>dlsym(_q_handle, sym_name('jk'))
js = <K (*)(K* x, char* s)>dlsym(_q_handle, sym_name('js'))
jv = <K (*)(K* x, K y)>dlsym(_q_handle, sym_name('jv'))
k = <K (*)(int handle, const char* s, ...) nogil>dlsym(_q_handle, sym_name('k'))
cdef extern from 'include/foreign.h':
    K k_wrapper(void* x, char* code, void* a1, void* a2, void* a3, void* a4, void* a5, void* a6, void* a7, void* a8) nogil
knogil = k_wrapper
ka = <K (*)(int t)>dlsym(_q_handle, sym_name('ka'))
kb = <K (*)(int x)>dlsym(_q_handle, sym_name('kb'))
kc = <K (*)(int x)>dlsym(_q_handle, sym_name('kc'))
kclose = <void (*)(int x)>dlsym(_q_handle, sym_name('kclose'))
kd = <K (*)(int x)>dlsym(_q_handle, sym_name('kd'))
ke = <K (*)(double x)>dlsym(_q_handle, sym_name('ke'))
kf = <K (*)(double x)>dlsym(_q_handle, sym_name('kf'))
kg = <K (*)(int x)>dlsym(_q_handle, sym_name('kg'))
kh = <K (*)(int x)>dlsym(_q_handle, sym_name('kh'))
khpunc = <int (*)(char* v, int w, char* x, int y, int z)>dlsym(_q_handle, sym_name('khpunc'))
ki = <K (*)(int x)>dlsym(_q_handle, sym_name('ki'))
kj = <K (*)(long long x)>dlsym(_q_handle, sym_name('kj'))
knk = <K (*)(int n, ...)>dlsym(_q_handle, sym_name('knk'))
knt = <K (*)(long long n, K x)>dlsym(_q_handle, sym_name('knt'))
kp = <K (*)(char* x)>dlsym(_q_handle, sym_name('kp'))
kpn = <K (*)(char* x, long long n)>dlsym(_q_handle, sym_name('kpn'))
krr = <K (*)(const char* s)>dlsym(_q_handle, sym_name('krr'))
ks = <K (*)(char* x)>dlsym(_q_handle, sym_name('ks'))
kt = <K (*)(int x)>dlsym(_q_handle, sym_name('kt'))
ktd = <K (*)(K x)>dlsym(_q_handle, sym_name('ktd'))
ktj = <K (*)(short _type, long long x)>dlsym(_q_handle, sym_name('ktj'))
ktn = <K (*)(int _type, long long length)>dlsym(_q_handle, sym_name('ktn'))
ku = <K (*)(U x)>dlsym(_q_handle, sym_name('ku'))
kz = <K (*)(double x)>dlsym(_q_handle, sym_name('kz'))
m9 = <void (*)()>dlsym(_q_handle, sym_name('m9'))
okx = <int (*)(K x)>dlsym(_q_handle, sym_name('okx'))
orr = <K (*)(const char*)>dlsym(_q_handle, sym_name('orr'))
r0 = <void (*)(K k)>dlsym(_q_handle, sym_name('r0'))
r1 = <K (*)(K k)>dlsym(_q_handle, sym_name('r1'))
sd0 = <void (*)(int d)>dlsym(_q_handle, sym_name('sd0'))
sd0x = <void (*)(int d, int f)>dlsym(_q_handle, sym_name('sd0x'))
sd1 = <K (*)(int d, f)>dlsym(_q_handle, sym_name('sd1'))
sd1 = <K (*)(int d, f)>dlsym(_q_handle, sym_name('sd1'))
sn = <char* (*)(char* s, long long n)>dlsym(_q_handle, sym_name('sn'))
ss = <char* (*)(char* s)>dlsym(_q_handle, sym_name('ss'))
sslInfo = <K (*)(K x)>dlsym(_q_handle, sym_name('sslInfo'))
vak = <K (*)(int x, const char* s, va_list l)>dlsym(_q_handle, sym_name('vak'))
vaknk = <K (*)(int, va_list l)>dlsym(_q_handle, sym_name('vaknk'))
ver = <int (*)()>dlsym(_q_handle, sym_name('ver'))
xD = <K (*)(K x, K y)>dlsym(_q_handle, sym_name('xD'))
xT = <K (*)(K x)>dlsym(_q_handle, sym_name('xT'))
ymd = <int (*)(int year, int month, int day)>dlsym(_q_handle, sym_name('ymd'))

_r0_ptr = int(<size_t><uintptr_t>r0)
_k_ptr = int(<size_t><uintptr_t>k)
_ktn_ptr = int(<size_t><uintptr_t>ktn)

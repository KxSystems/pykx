import os
from pathlib import Path
import platform
import shlex
import sys

from .exceptions import PyKXWarning


system = platform.system()
# q expects (m|l|w)64 to exist under whatever QHOME was when qinit was executed.
q_lib_dir_name = {
    'Darwin': 'm64',
    'Linux': 'l64',
    'Windows': 'w64',
}[system]
if 'Darwin' in system and 'arm' in platform.machine():
    q_lib_dir_name = 'm64arm'
if 'Linux' in system and ('arm' in platform.machine() or 'aarch64' in platform.machine()):
    q_lib_dir_name = 'l64arm'
pykx_dir = Path(__file__).parent.resolve(strict=True)
os.environ['PYKX_DIR'] = str(pykx_dir)
pykx_lib_dir = Path(os.getenv('PYKX_Q_LIB_LOCATION', pykx_dir/'lib'))
pykx_platlib_dir = pykx_lib_dir/q_lib_dir_name
lib_prefix = '' if system == 'Windows' else 'lib'
lib_ext = {
    'Darwin': 'dylib',
    'Linux': 'so',
    'Windows': 'dll',
}[system]

try:
    qhome = Path(os.environ.get('QHOME', Path().home()/'q')).resolve(strict=True)
except FileNotFoundError: # nocov
    # If QHOME and its fallback weren't set/valid, then q/Python must be
    # running in the same directory as q.k (and presumably other stuff one
    # would expect to find in QHOME).
    qhome = Path().resolve(strict=True)

for lic in ('kx.lic', 'kc.lic', 'k4.lic'): # nocov
    try:
        lic_path = Path(lic).resolve(strict=True)
    except FileNotFoundError:
        continue
    else:
        qlic = lic_path.parent
        break
else:
    qlic = Path(os.environ.get('QLIC', qhome)).resolve(strict=True)

qargs = tuple(shlex.split(os.environ.get('QARGS', '')))
licensed = False


def _is_enabled(envvar, cmdflag=None):
    return os.getenv(envvar, '').lower() in ('1', 'true') or (cmdflag and cmdflag in qargs)


def _is_set(envvar):
    return os.getenv(envvar, None)


under_q = _is_enabled('PYKX_UNDER_Q')
qlib_location = Path(os.getenv('PYKX_Q_LIB_LOCATION', pykx_dir/'lib'))
no_sigint = _is_enabled('PYKX_NO_SIGINT')


enable_pandas_api = _is_enabled('PYKX_ENABLE_PANDAS_API', '--pandas-api')
ignore_qhome = _is_enabled('IGNORE_QHOME', '--ignore-qhome')
keep_local_times = _is_enabled('KEEP_LOCAL_TIMES', '--keep-local-times')
max_error_length = int(os.getenv('PYKX_MAX_ERROR_LENGTH', 256))

if _is_enabled('PYKX_ALLOCATOR', '--pykxalloc'):
    if sys.version_info[1] <= 7:
        raise PyKXWarning('A python version of at least 3.8 is required to use the PyKX allocators') # noqa nocov
        k_allocator = False  # nocov
    else:
        k_allocator = True
else:
    k_allocator = False

k_gc = _is_enabled('PYKX_GC', '--pykxgc')
release_gil = _is_enabled('PYKX_RELEASE_GIL', '--release-gil')
use_q_lock = os.getenv('PYKX_Q_LOCK', False)
skip_under_q = _is_enabled('SKIP_UNDERQ', '--skip-under-q')
no_qce = _is_enabled('PYKX_NOQCE', '--no-qce')
load_pyarrow_unsafe = _is_enabled('PYKX_LOAD_PYARROW_UNSAFE', '--load-pyarrow-unsafe')


def find_core_lib(name: str) -> Path:
    suffix = '.dll' if system == 'Windows' else '.so'
    path = pykx_platlib_dir/f'{lib_prefix}{name}'
    try:
        return path.with_suffix(suffix).resolve(strict=True)
    except FileNotFoundError: # nocov
        if system == 'Darwin' and suffix == '.so':
            return path.with_suffix('.dylib').resolve(strict=True)
        raise


def _set_licensed(licensed_):
    global licensed
    licensed = licensed_


def _set_keep_local_times(keep_local_times_):
    global keep_local_times
    keep_local_times = keep_local_times_


__all__ = [
    'system',
    'q_lib_dir_name',
    'pykx_dir',
    'pykx_lib_dir',
    'pykx_platlib_dir',
    'lib_prefix',
    'lib_ext',

    'qhome',
    'qlic',
    'qargs',
    'licensed',
    'under_q',
    'qlib_location',

    'enable_pandas_api',
    'ignore_qhome',
    'keep_local_times',
    'max_error_length',

    'k_allocator',
    'k_gc',
    'release_gil',
    'use_q_lock',
    'skip_under_q',
    'no_qce',
    'load_pyarrow_unsafe',

    'find_core_lib',
]


def __dir__():
    return sorted(__all__)

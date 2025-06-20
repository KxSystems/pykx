from contextlib import contextmanager
from functools import wraps
import inspect
import io
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Union
from zipfile import ZipFile
from warnings import warn

import pandas as pd
from pandas.core.internals import BlockManager, make_block
import requests
import toml

from .config import (
    _executable, _get_qexecutable, _get_qhome, _pykx_config_location, _pykx_profile_content,
    allocator, beta_features, ignore_qhome, jupyterq, k_gc, keep_local_times,
    load_pyarrow_unsafe, max_error_length, no_pykx_signal, no_qce, pykx_4_1,
    pykx_config_location, pykx_config_profile, pykx_debug_insights, pykx_dir, pykx_lib_dir,
    pykx_qdebug, pykx_threading, q_executable, qargs, qhome, qlic, release_gil,
    skip_under_q, suppress_warnings, use_q_lock)
from ._version import version as __version__
from .exceptions import PyKXException
from .reimporter import PyKXReimport


try:
    import psutil
    _psutil_available = True
except ImportError:
    _psutil_available = False


__all__ = [
    'num_available_cores',
    'BlockManagerUnconsolidated',
    'ClassPropertyDescriptor',
    'classproperty',
    'attr_as',
    'cached_property',
    'class_or_instancemethod',
    'debug_environment',
    'delete_q_variable',
    'df_from_arrays',
    'get_default_args',
    'normalize_to_bytes',
    'normalize_to_str',
    'once',
    'detect_bad_columns',
    'slice_to_range',
    'subclasses',
    'jupyter_qfirst_enable',
    'jupyter_qfirst_disable',
    'kill_q_process'
]


def _init(_q):
    global q
    q = _q


def __dir__():
    return sorted(__all__)


def num_available_cores() -> int:
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    elif platform.system() == 'Windows': # nocov
        import ctypes # nocov
        import ctypes.wintypes # nocov
        kernel32 = ctypes.WinDLL('kernel32') # nocov
        DWORD_PTR = ctypes.wintypes.WPARAM # nocov
        PDWORD_PTR = ctypes.POINTER(DWORD_PTR) # nocov
        GetCurrentProcess = kernel32.GetCurrentProcess # nocov
        GetCurrentProcess.restype = ctypes.wintypes.HANDLE # nocov
        GetProcessAffinityMask = kernel32.GetProcessAffinityMask # nocov
        GetProcessAffinityMask.argtypes = (ctypes.wintypes.HANDLE, PDWORD_PTR, PDWORD_PTR) # nocov
        mask = DWORD_PTR() # nocov
        if not GetProcessAffinityMask(GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(DWORD_PTR())): # noqa nocov
            raise PyKXException("Call to 'GetProcessAffinityMask' failed") # nocov
        return bin(mask.value).count('1') # nocov
    # No way to determine the number of cores available in this case. The CPU count will work in
    # most scenarios, and for those in which it does not there is no known workaround.
    return os.cpu_count() # nocov


# TODO: Once Python 3.7 support is dropped, we can replace this with `functools.cached_property`
class cached_property:
    """Property-like descriptor which overwrites itself with the first obtained property value."""
    def __init__(self, func):
        self.func = func
        self.attrname = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = self.func(instance)
        instance.__dict__[self.attrname] = value
        return value


class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


def slice_to_range(s: slice, n: int) -> range:
    """Converts a slice and collection size into a range whose indices match the slice.

    Parameters:
        s: The slice to be converted into a range of indices.
        n: The size of the container the slice was going to be applied to.

    Returns:
        A range whose indices match the slice
    """
    return range(
        s.start or 0,
        n if s.stop is None else min(s.stop, n),
        s.step or 1
    )


def subclasses(cls: type) -> set:
    """Given a class, returns the set of all subclasses of that class (including itself)."""
    def _subclasses(cls):
        yield cls
        for subclass in cls.__subclasses__():
            yield subclass
            yield from _subclasses(subclass)
    return set(_subclasses(cls))


def normalize_to_bytes(s: Union[str, bytes], name: str = 'Argument') -> bytes:
    """Given text as a `str` or `bytes` object, returns that text as a `bytes` object.

    Parameters:
        s: The text as a `str` or `bytes` object.
        name: An identifiable name for `s` for the purpose of producing clear error messages.

    Returns:
        The given text as a `bytes` object.
    """
    if isinstance(s, str):
        return s.encode()
    elif isinstance(s, bytes):
        return s
    raise TypeError(f"{name} must be of type 'str' or 'bytes'")


def normalize_to_str(s: Union[str, bytes], name: str = 'Argument') -> str:
    """Given text as a `str` or `bytes` object, returns that text as a `str` object.

    Parameters:
        s: The text as a `str` or `bytes` object.
        name: An identifiable name for `s` for the purpose of producing clear error messages.

    Returns:
        The given text as a `str` object.
    """
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode()
    raise TypeError(f"{name} must be of type 'str' or 'bytes'")


@contextmanager
def attr_as(x, name: str, val) -> None:
    """Temporarily replaces attribute `name` of `x` with `val`."""
    already_exists = hasattr(x, name)
    if already_exists:
        prev_val = getattr(x, name)
    setattr(x, name, val)
    try:
        yield
    finally:
        if already_exists:
            setattr(x, name, prev_val)
        else:
            delattr(x, name)


def once(f):
    """Decorator that makes it so that a function can only be run once.

    The return value is cached permanently. Repeated calls return that cached value.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.was_run:
            wrapper.was_run = True
            wrapper.result = f(*args, **kwargs)
        return wrapper.result
    wrapper.was_run = False
    return wrapper


class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return ClassPropertyDescriptor(func)


class BlockManagerUnconsolidated(BlockManager):
    """BlockManager that does not consolidate blocks, thus avoiding copying."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_consolidated = False
        self._known_consolidated = False

    def __reduce__(self):
        # This works in CI but fails in some other esoteric situations
        # return (BlockManager, super().__reduce__()[1])
        # Manually reassigning the class attribute to the pandas BlockManager fixes this function
        # in those situations.
        c = self.copy()
        c.__class__ = BlockManager
        return c.__reduce__()

    def _consolidate_inplace(self): # nocov
        pass

    def _consolidate(self): # nocov
        return self.blocks


def df_from_arrays(columns, arrays, index):
    """Create a DataFrame from Numpy arrays without copying."""
    blocks = tuple(
        make_block(values=a.reshape((1, len(a))), placement=(i,))
        for i, a in enumerate(arrays)
    )
    return pd.DataFrame(
        BlockManagerUnconsolidated(axes=[columns, index], blocks=blocks),
        copy=False
    )

    # TODO: The following code probably ought to work, but some parts of Pandas still rely on the
    # assumption that a BlockManager is being used. Next time we update the minimum Pandas version
    # required by PyKX we should check if we can use an ArrayManager (KXI-9722).

    # return pd.DataFrame(
    #     ArrayManager(arrays=arrays, axes=[index, columns]),
    #     copy=False
    # )


def get_default_args(f: Callable) -> Dict[str, Any]:
    """Returns a dictionary mapping each parameter name to its default argument.

    Parameters:
        f: An function which can be inspected by `inspect.signature`.

    Returns:
        A dictionary mapping each parameter name to its default arguments. Parameters which do not
        have default arguments are not added to this dictionary.
    """
    return {
        k: v.default for k, v in inspect.signature(f).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def debug_environment(detailed: bool = False, return_info: bool = False) -> Union[str, None]:
    """
    Functionality for the retrieval of information about a users environment

    Parameters:
        detailed: When returning information about a users license print the content of both
            `QHOME` and `QLIC` directories
        return_info: Should the information returned from the function be printed to console
            (default) or provided as a str

    Returns:
        Returns `None` if return information is printed to console otherwise
            returns a `str` representation

    Examples:

    ```python
    >>> import pykx as kx
    >>> kx.util.debug_environment()
    **** PyKX information ****
    pykx.args: ()
    pykx.qhome: /usr/local/anaconda3/envs/qenv/q
    pykx.qlic: /usr/local/anaconda3/envs/qenv/q
    pykx.licensed: True
    pykx.__version__: 3.1.1
    pykx.file: /Library/Versions/3.12/lib/python3.12/site-packages/pykx/util.py

    **** Python information ****
    sys.version: 3.12.3 (v3.12.3:f6650f9ad7, Apr  9 2024, 08:18:48) ..
    pandas: 1.5.3
    numpy: 1.26.2
    pytz: 2024.1
    which python: /usr/local/bin/python
    which python3: /Library/Versions/3.12/bin/python3
    find_libpython: /Library/Versions/3.12/Python

    **** Platform information ****
    platform.platform: macOS-13.0.1-x86_64-i386-64bit

    **** PyKX Configuration File ****
    File location: /usr/local/.pykx-config
    Used profile: default
    Profile content: {'PYKX_Q_EXECUTABLE': '/usr/local/anaconda3/envs/qenv/q/m64/q'}

    **** PyKX Configuration Variables ****
    PYKX_IGNORE_QHOME: False
    PYKX_KEEP_LOCAL_TIMES: False
    PYKX_ALLOCATOR: False
    PYKX_GC: False
    PYKX_LOAD_PYARROW_UNSAFE: False
    PYKX_MAX_ERROR_LENGTH: 256
    PYKX_NOQCE: False
    PYKX_RELEASE_GIL: False
    PYKX_Q_LIB_LOCATION: /Library/Versions/3.12/lib/python3.12/site-packages/pykx/lib
    PYKX_Q_LOCK: False
    PYKX_SKIP_UNDERQ: False
    PYKX_Q_EXECUTABLE: /usr/local/anaconda3/envs/qenv/q/m64/q
    PYKX_THREADING: False
    PYKX_4_1_ENABLED: False
    PYKX_QDEBUG: False
    PYKX_DEBUG_INSIGHTS_LIBRARIES: False
    PYKX_CONFIGURATION_LOCATION: .
    PYKX_NO_SIGNAL: False
    PYKX_CONFIG_PROFILE: default
    PYKX_BETA_FEATURES: True
    PYKX_JUPYTERQ: False
    PYKX_SUPPRESS_WARNINGS: False
    PYKX_DEFAULT_CONVERSION:
    PYKX_EXECUTABLE: /Library/Versions/3.12/bin/python3.12
    PYKX_PYTHON_LIB_PATH:
    PYKX_PYTHON_BASE_PATH:
    PYKX_PYTHON_HOME_PATH:
    PYKX_DIR: /Library/Versions/3.12/lib/python3.12/site-packages/pykx
    PYKX_USE_FIND_LIBPYTHON:
    PYKX_UNLICENSED:
    PYKX_LICENSED:
    PYKX_4_1_ENABLED:

    **** q Environment Variables ****
    QARGS:
    QHOME: /Library/Versions/3.12/lib/python3.12/site-packages/pykx/lib
    QLIC: /usr/local/anaconda3/envs/qenv/q
    QINIT:

    **** License information ****
    pykx.qlic directory: True
    pykx.qhome writable: True
    pykx.qhome lics: ['k4.lic']
    pykx.qlic lics: ['k4.lic']

    **** q information ****
    which q: /usr/local/bin/q
    q info:
    (`m64;4.1;2024.10.16)
    "insights.lib.embedq insights.lib.pykx insights.lib.sql insights.lib.qlog insights.lib.kurl"

    **** pykx startup information ****
    secondary threads: 8
    ```
    """
    debug_info = ""
    debug_info += pykx_information()
    debug_info += python_information()
    debug_info += platform_information()
    debug_info += config_information()
    debug_info += env_information()
    debug_info += lic_information(detailed=detailed)
    debug_info += q_information()
    debug_info += pykx_startup_information()
    if return_info:
        return debug_info
    print(debug_info)
    return None


def pykx_information():
    from .core import _is_licensed
    pykx_info = "**** PyKX information ****\n"
    pykx_info += f"pykx.args: {qargs}\n"
    pykx_info += f"pykx.qhome: {qhome}\n"
    pykx_info += f"pykx.qlic: {qlic}\n"

    pykx_info += f"pykx.licensed: {_is_licensed()}\n"
    pykx_info += f"pykx.__version__: {__version__}\n"
    pykx_info += f"pykx.file: {__file__}\n"
    return pykx_info


def python_information():
    py_info = '\n**** Python information ****\n'
    try:
        py_info += f"sys.version: {sys.version}\n"

        import importlib.metadata
        py_info += f"pandas: {importlib.metadata.version('pandas')}\n"
        py_info += f"numpy: {importlib.metadata.version('numpy')}\n"
        py_info += f"pytz: {importlib.metadata.version('pytz')}\n"

        import shutil
        py_info += f"which python: {shutil.which('python')}\n"
        py_info += f"which python3: {shutil.which('python3')}\n"
    except Exception:
        pass
    try:
        import find_libpython
        py_info += f"find_libpython: {find_libpython.find_libpython()}\n"
    except BaseException:
        pass
    return py_info


def platform_information():
    platform_info = '\n**** Platform information ****\n'
    platform_info += f"platform.platform: {platform.platform()}\n"
    return platform_info


def config_information():
    config_info = '\n**** PyKX Configuration File ****\n'
    config_info += f"File location: {_pykx_config_location}\n"
    config_info += f"Used profile: {pykx_config_profile}\n"
    config_info += f"Profile content: {_pykx_profile_content}\n"
    return config_info


def env_information():
    env_info = '\n**** PyKX Configuration Variables ****\n'

    global_config = {'PYKX_IGNORE_QHOME': ignore_qhome, 'PYKX_KEEP_LOCAL_TIMES': keep_local_times,
                     'PYKX_ALLOCATOR': allocator, 'PYKX_GC': k_gc,
                     'PYKX_LOAD_PYARROW_UNSAFE': load_pyarrow_unsafe,
                     'PYKX_MAX_ERROR_LENGTH': max_error_length, 'PYKX_NOQCE': no_qce,
                     'PYKX_RELEASE_GIL': release_gil, 'PYKX_Q_LIB_LOCATION': pykx_lib_dir,
                     'PYKX_Q_LOCK': use_q_lock, 'PYKX_SKIP_UNDERQ': skip_under_q,
                     'PYKX_Q_EXECUTABLE': q_executable, 'PYKX_THREADING': pykx_threading,
                     'PYKX_4_1_ENABLED': pykx_4_1, 'PYKX_QDEBUG': pykx_qdebug,
                     'PYKX_DEBUG_INSIGHTS_LIBRARIES': pykx_debug_insights,
                     'PYKX_CONFIGURATION_LOCATION': pykx_config_location,
                     'PYKX_NO_SIGNAL': no_pykx_signal,
                     'PYKX_CONFIG_PROFILE': pykx_config_profile,
                     'PYKX_BETA_FEATURES': beta_features, 'PYKX_JUPYTERQ': jupyterq,
                     'PYKX_SUPPRESS_WARNINGS': suppress_warnings}

    env_only = ['PYKX_DEFAULT_CONVERSION',
                'PYKX_EXECUTABLE', 'PYKX_PYTHON_LIB_PATH',
                'PYKX_PYTHON_BASE_PATH', 'PYKX_PYTHON_HOME_PATH', 'PYKX_DIR',
                'PYKX_USE_FIND_LIBPYTHON', 'PYKX_UNLICENSED', 'PYKX_LICENSED',
                'PYKX_4_1_ENABLED'
                ]

    for k, v in global_config.items():
        env_info += f"{k}: {v}\n"

    for x in env_only:
        env_info += f"{x}: {os.getenv(x, '')}\n"

    env_info += '\n**** q Environment Variables ****\n'
    qenv = ['QARGS', 'QHOME', 'QLIC', 'QINIT']

    for x in qenv:
        env_info += f"{x}: {os.getenv(x, '')}\n"
    return env_info


def lic_information(detailed=False):
    lic_info = '\n**** License information ****\n'

    lic_info += f"pykx.qlic directory: {os.path.isdir(qlic)}\n"
    lic_info += f"pykx.qhome writable: {os.access(qhome, os.W_OK)}\n"

    if detailed:
        lic_info += f"pykx.qhome contents: {os.listdir(qhome)}\n"
        lic_info += f"pykx.lic contents: {os.listdir(qlic)}\n"

    try:
        import re
        klic = re.compile('k.\\.lic').match
        qhomelics = list(filter(klic, os.listdir(qhome)))
        lic_info += f"pykx.qhome lics: {qhomelics}\n"
        qliclics = list(filter(klic, os.listdir(qlic)))
        lic_info += f"pykx.qlic lics: {qliclics}\n"
    except Exception:
        pass
    return lic_info


def q_information():
    q_info = '\n**** q information ****\n'

    try:
        import shutil
        import subprocess
        whichq = shutil.which('q')
        q_info += f"which q: {whichq}\n"
        if whichq is not None:
            q_info += ('q info: \n')
            if platform.system() == 'Windows':
                cmd = "powershell -NoProfile -ExecutionPolicy ByPass \"echo \\\"-1 .Q.s1 (.z.o;.z.K;.z.k);-1 .Q.s1 .z.l 4;\\\" | q -c 200 200\"" # noqa: E501
            else:
                cmd = "echo \"-1 .Q.s1 (.z.o;.z.K;.z.k);-1 .Q.s1 .z.l 4;\" | q -c 200 200"
            with PyKXReimport():
                out = subprocess.run(cmd, shell=True, capture_output=True)
            if out.returncode == 0:
                q_info += (out.stdout).decode(encoding='utf-8')
            else:
                q_info += "Failed to gather q information: " + (out.stderr).decode(encoding='utf-8')
    except Exception as e:
        q_info += f"Failed to gather q information: {e}"
    return q_info


def pykx_startup_information():
    pykx_start_info = '\n**** pykx startup information ****\n'

    from .core import _is_licensed
    if _is_licensed():
        sec_threads = q('string system"s"')
        pykx_start_info += f"secondary threads: {sec_threads}\n"
    else:
        pykx_start_info += "Gathering PyKX startup information only available in licensed mode\n"
    return pykx_start_info


def _run_all_cell_with_magics(lines):
    if "%%py" == lines[0].strip():
        return lines[1:]
    elif "%%q" in lines[0].strip():
        return lines
    else:
        return (["%%q \n"]+lines)


def jupyter_qfirst_enable():
    qfirst_modify("q")


def jupyter_qfirst_disable():
    qfirst_modify("python")


def qfirst_modify(state):
    try:
        ipython = get_ipython()
        if _run_all_cell_with_magics in ipython.input_transformers_cleanup and state == "python":
            ipython.input_transformers_cleanup.remove(_run_all_cell_with_magics)
            print("""PyKX now running in 'python' mode (default). All cells by default will be run as python code. 
Include '%%q' at the beginning of each cell to run as q code. """) # noqa 
        elif _run_all_cell_with_magics not in ipython.input_transformers_cleanup and state == "q":
            ipython.input_transformers_cleanup.append(_run_all_cell_with_magics)
            print("""PyKX now running in 'jupyter_qfirst' mode. All cells by default will be run as q code. 
Include '%%py' at the beginning of each cell to run as python code. """) # noqa 
        else:
            print(f"PyKX already running in '{state}' mode")
    except NameError:
        print("Not running under IPython/Jupyter")


def add_to_config(config, folder='~'):
    """
    Add configuration options to the file '.pykx-config' in a specified folder

    Parameters:
        config: A dictionary mapping the configuration options to their associated value
        folder: The folder where the users '.pykx-config' file is to be updated

    Examples:

    ```python
    >>> import pykx as kx
    >>> kx.util.add_to_config({'PYKX_GC': 'True', 'PYKX_BETA_FEATURES': 'True'})
    Configuration updated at: /usr/local/.pykx-config.
    Profile updated: default.
    Successfully added:
        - PYKX_GC = True
        - PYKX_BETA_FEATURES = True
    ```
    """
    if not isinstance(config, dict):
        raise TypeError(f'Supplied config must be of type dict, supplied type: {type(config)}')
    fpath = str(Path(os.path.expanduser(folder)) / '.pykx-config')
    try:
        os.access(fpath, os.W_OK)
    except FileNotFoundError:
        pass
    except PermissionError:
        raise PermissionError(f"You do not have sufficient permissions to write to: {fpath}")
    print_config = f"\nConfiguration updated at: {fpath}.\nProfile updated: "\
                   f"{pykx_config_profile}.\nSuccessfully added:\n"
    if os.path.exists(fpath):
        with open(fpath, 'r') as file:
            data = toml.load(file)
    else:
        data = {pykx_config_profile: {}}
    for k, v in config.items():
        data[pykx_config_profile][k] = v
        print_config += f'\t- {k} = {v}\n'
        if isinstance(v, (int, bool)):
            v = str(v)
        os.environ[k] = v
    with open(fpath, 'w') as file:
        toml.dump(data, file)
        print(print_config)


_user_os = {'Linux': 'l64', 'Darwin': 'm64', 'Windows': 'w64'}

_user_arch = {'x86_64': '', 'aarch64': 'arm'}

_kdb_url = 'https://portal.dl.kx.com/assets/raw/kdb+/4.0'


def install_q(location: str = '~/q',
              date: str = '2024.07.08'):
    """
    Install q to a specified location.

    Parameters:
        location: The location to which q will be installed
        date: The dated version of kdb+ 4.0 which is to be installed
    """
    global qhome
    my_os = _user_os[platform.uname()[0]]
    if my_os == 'l64':
        my_os += _user_arch[platform.uname()[4]]
    location = Path(os.path.expanduser(location))
    url = f'{_kdb_url}/{date}/{my_os}.zip'
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f'Request for download of q unsuccessful with code: {r.status_code}')
    zf = ZipFile(io.BytesIO(r.content), 'r')
    zf.extractall(location)
    try:
        executable = 'q.exe' if my_os == 'w64' else 'q'
        executable_loc = location/my_os/executable
        os.chmod(executable_loc, 0o777)
    except BaseException:
        raise RuntimeError(f"Unable to set execute permissions on file: {executable_loc}")
    shutil.copy(pykx_dir/'pykx.q', location/'pykx.q')
    shutil.copy(pykx_dir/'lib/s.k_', location/'s.k_')

    add_to_config({
        'PYKX_Q_EXECUTABLE': str(executable_loc),
        'QHOME': str(location)})
    print('Please restart your process to use this executable.')


def start_q_subprocess(port: int,
                       load_file: str = '',
                       init_args: list = None,
                       process_logs: bool = True):
    """
    Initialize a q subprocess using a supplied path to an executable on a specified port

    Parameters:
        port: The port on which the q process will be started.
        load_file: A file for the process to load
        init_args: A list denoting any arguments to be passed when starting
            the q process.
        process_logs: Should stdout/stderr be printed to in the parent process

    Returns:
        The subprocess object which was generated on initialisation
    """
    q_executable = _get_qexecutable()
    if q_executable is None:
        my_os = _user_os[platform.uname()[0]]
        if my_os == 'l64':
            my_os += _user_arch[platform.uname()[4]]
        qhome = _get_qhome()
        loc = qhome / my_os / _executable
        if loc.is_file():
            q_executable = str(loc)
        else:
            raise RuntimeError(
                'Unable to locate an appropriate q executable\n'
                'Please install q using the function "kx.util.install_q" or following the '
                'instructions at:\nhttps://code.kx.com/pykx/getting-started/installing.html'
            )
    with PyKXReimport():
        qinit = [q_executable, load_file, '-p', f'{port}']
        if init_args is not None:
            if not isinstance(init_args, list):
                raise TypeError('Supplied additional startup arguments must be a list')
            if not all(isinstance(s, str) for s in init_args):
                raise TypeError('All supplied arguments to init_args must be str type objects')
            qinit.extend(init_args)
        server = subprocess.Popen(
            qinit,
            stdin=subprocess.PIPE,
            stdout=None if process_logs else subprocess.DEVNULL,
            stderr=None)
        time.sleep(2)
    return server


def kill_q_process(port: int) -> bool:
    """
    Kill a q process running on a specified port, this allows users to
    to kill sub-processes running q in the case access to the port has been
    lost due to parent process

    Parameters:
        port: The port which is to be killed

    Returns:
        Kill a process and return None
    """
    if not _psutil_available:
        raise ImportError(
            'psutil library not available, install psutil with pip/conda as follows :\n'
            '    pip -> pip install psutil\n'
            '    conda -> conda install conda-forge::psutil'
        )
    processes = [proc for proc in psutil.process_iter() if proc.name()
                 == 'q']
    for p in processes:
        for c in p.connections():
            if c.status == 'LISTEN' and c.laddr.port == port:
                try:
                    os.kill(p.pid, signal.SIGKILL)
                    return True
                except BaseException:
                    return False
    return False


def detect_bad_columns(table, return_cols: bool = False):
    """
    Validate that the columns of a table conform to expected naming conventions for kdb+
    and do not contain duplicates.

    Parameters:
        table: The `pykx.Table`, `pykx.KeyedTable`, `pykx.SplayedTable` or
            `pykx.PartitionedTable` object which is to be checked
        return_cols: Should the invalid columns from the table be returned

    Returns:
        Raises a warning indicating the issue with the column(s) and returns `True` or `False`
        if the columns are invalid (`True`) or not (`False`).
    """
    cols = []
    bad_cols = q('.pykx.util.html.detectbadcols', table).py()
    hasDups, hasInvalid = [len(x) for x in bad_cols.values()]
    if hasDups or hasInvalid:
        warn_string = '\nDuplicate columns or columns with reserved characters detected:'
        if hasDups:
            warn_string += f'\n\tDuplicate columns: {bad_cols["dup"]}'
        if hasInvalid:
            warn_string += f'\n\tInvalid columns: {bad_cols["invalid"]}'
        warn_string += '\nSee https://code.kx.com/pykx/help/troubleshooting.html to learn more about updating your table' # noqa
        warn(warn_string, RuntimeWarning)
        if return_cols:
            for i in bad_cols.values():
                cols.extend(i)
            return cols
        else:
            return True
    if return_cols:
        return cols
    return False


def delete_q_variable(variable: str, namespace: str = '', garbage_collect: bool = False):
    """
    Deletes a variable from q memory.

    Parameters:
        variable: The name of the variable to delete
        namespace: The name of the namespace which the variable is in
        garbage_collect: Control whether to run gargabeg collection after variable deletion

    Returns:
        None
    """
    ns = '.' + namespace
    q('{![x;();0b;enlist y]}', ns, variable)
    if garbage_collect:
        q.Q.gc()

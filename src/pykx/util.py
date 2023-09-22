from contextlib import contextmanager
from functools import wraps
import inspect
import os
import platform
from typing import Any, Callable, Dict, Union

import pandas as pd
from pandas.core.internals import BlockManager, make_block

from .config import qargs, qhome, qlic
from ._version import version as __version__
from .exceptions import PyKXException


__all__ = [
    'num_available_cores',
    'BlockManagerUnconsolidated',
    'attr_as',
    'cached_property',
    'debug_environment',
    'df_from_arrays',
    'get_default_args',
    'normalize_to_bytes',
    'normalize_to_str',
    'once',
    'slice_to_range',
    'subclasses',
]


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


def debug_environment(detailed=False, return_info=False):
    """Displays information about your environment to help debug issues."""
    debug_info = ""
    debug_info += pykx_information()
    debug_info += python_information()
    debug_info += platform_information()
    debug_info += env_information()
    debug_info += lic_information(detailed=detailed)
    debug_info += q_information()
    if return_info:
        return debug_info
    print(debug_info)
    return None


def pykx_information():
    pykx_info = "**** PyKX information ****\n"
    pykx_info += f"pykx.args: {qargs}\n"
    pykx_info += f"pykx.qhome: {qhome}\n"
    pykx_info += f"pykx.qlic: {qlic}\n"

    from .config import licensed
    pykx_info += f"pykx.licensed: {licensed}\n"
    pykx_info += f"pykx.__version__: {__version__}\n"
    pykx_info += f"pykx.file: {__file__}\n"
    return pykx_info


def python_information():
    py_info = '\n**** Python information ****\n'
    try:
        import sys
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
    return py_info


def platform_information():
    platform_info = '\n**** Platform information ****\n'
    platform_info += f"platform.platform: {platform.platform()}\n"
    return platform_info


def env_information():
    env_info = '\n**** Environment Variables ****\n'

    envs = ['IGNORE_QHOME', 'PYKX_IGNORE_QHOME', 'PYKX_KEEP_LOCAL_TIMES', 'PYKX_ALLOCATOR',
            'PYKX_GC', 'PYKX_LOAD_PYARROW_UNSAFE', 'PYKX_MAX_ERROR_LENGTH',
            'PYKX_NOQCE', 'PYKX_Q_LIB_LOCATION', 'PYKX_RELEASE_GIL', 'PYKX_Q_LOCK',
            'QARGS', 'QHOME', 'QLIC',
            'PYKX_DEFAULT_CONVERSION', 'PYKX_SKIP_UNDERQ', 'PYKX_UNSET_GLOBALS',
            'SKIP_UNDERQ', 'UNSET_PYKX_GLOBALS'    # Deprecated
            ]

    for x in envs:
        env_info += f"{x}: {os.getenv(x, '')}\n"
    return env_info


def lic_information(detailed=False):
    lic_info = '\n**** License information ****\n'

    lic_info += f"pykx.qlic directory: {os.path.isdir(qlic)}\n"
    lic_info += f"pykx.lic writable: {os.access(qhome, os.W_OK)}\n"

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
            if platform.system() == 'Windows': # nocov:
                q_info += subprocess.check_output("powershell -NoProfile -ExecutionPolicy ByPass \"echo \\\"-1 .Q.s1 (.z.o;.z.K;.z.k);-1 .Q.s1 .z.l 4;\\\" | q -c 200 200\"", shell=True).decode(encoding='utf-8') # noqa: E501
            else: # nocov:
                q_info += subprocess.check_output("echo \"-1 .Q.s1 (.z.o;.z.K;.z.k);-1 .Q.s1 .z.l 4;\" | q -c 200 200", shell=True).decode(encoding='utf-8') # noqa: E501
    except Exception:
        pass
    return q_info

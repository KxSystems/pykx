"""
An interface between Python and q.
"""

import os
import sys

if os.getenv('PYKX_LOADED_UNDER_Q') == 'True':
    sys.exit(0)

# Attempt to import PyArrow early to get ahead of others (e.g. Pandas) who would try to import it
# without guarding against segfaults. Skip this if we're just doing a qinit check.
if os.environ.get('PYKX_QINIT_CHECK') is None:
    try:
        from ._pyarrow import pyarrow
    except ImportError: # nocov
        pass
else: # nocov
    pass


from . import reimporter
# Importing core initializes q if in licensed mode, and loads the q C API symbols. This should
# happen early on so that if the qinit check is currently happening then no time is wasted.
# The current process will exit while the core module is loading if the qinit check is happening.
from . import core

from abc import ABCMeta, abstractmethod
import itertools as it
from pathlib import Path
import platform
import shutil
import signal
from typing import Any, List, Optional, Union
from warnings import warn
from weakref import proxy

from .config import k_allocator, licensed, pykx_platlib_dir
from . import util

if platform.system() == 'Windows': # nocov
    os.environ['PATH'] += f';{pykx_platlib_dir}'
    if platform.python_version_tuple()[:2] >= ('3', '8'):
        os.add_dll_directory(pykx_platlib_dir)


def _first_resolved_path(possible_paths: List[Union[str, Path]]) -> Path:
    """Returns the resolved version of the first path that exists."""
    for path in possible_paths:
        try:
            return Path(path).resolve(strict=True)
        except FileNotFoundError:
            pass
    unfound_paths = '\n'.join(str(path) for path in possible_paths)
    raise FileNotFoundError(f'Could not find any of the following files:\n{unfound_paths}')


class Q(metaclass=ABCMeta):
    """Abstract base class for all interfaces between Python and q.

    See Also:
        - [`pykx.EmbeddedQ`][]
        - [`pykx.QConnection`][]
    """
    reserved_words = {
        'abs', 'acos', 'aj', 'aj0', 'ajf', 'ajf0', 'all', 'and', 'any', 'asc',
        'asin', 'asin', 'asof', 'atan', 'attr', 'avg', 'avgs', 'bin',
        'binr', 'ceiling', 'cols', 'cor', 'cos', 'count', 'cov', 'cross',
        'csv', 'cut', 'delete', 'deltas', 'desc', 'dev', 'differ', 'distinct',
        'div', 'do', 'dsave', 'each', 'ej', 'ema', 'enlist', 'eval',
        'except', 'exec', 'exit', 'exp', 'fby', 'fills', 'first', 'fkeys',
        'flip', 'floor', 'get', 'getenv', 'group', 'gtime', 'hclose', 'hcount', 'hdel',
        'hopen', 'hsym', 'iasc', 'idesc', 'if', 'ij', 'ijf', 'in',
        'inter', 'inv', 'key', 'keys', 'last', 'like', 'lj', 'ljf',
        'load', 'log', 'lower', 'lsq', 'ltime', 'ltrim', 'mavg', 'max', 'maxs',
        'mcount', 'md5', 'mdev', 'med', 'meta', 'min', 'mins', 'mmax', 'mmin', 'mmu', 'mod',
        'msum', 'neg', 'next', 'not', 'null', 'or', 'over', 'parse', 'peach', 'pj', 'prd',
        'prds', 'prev', 'prior', 'rand', 'rank', 'ratios', 'raze', 'read0', 'read1', 'reciprocal',
        'reval', 'reverse', 'rload', 'rotate', 'rsave', 'rtrim', 'save', 'scan', 'scov', 'sdev',
        'select', 'set', 'setenv', 'show', 'signum', 'sin',
        'sqrt', 'ss', 'ssr', 'string', 'sublist', 'sum', 'sums', 'sv', 'svar',
        'system', 'tables', 'tan', 'til', 'trim', 'type', 'uj', 'ujf', 'ungroup',
        'union', 'update', 'upper', 'value', 'var', 'view', 'views',
        'vs', 'wavg', 'wavg', 'where', 'while', 'within', 'wj', 'wj1',
        'wsum', 'xasc', 'xbar', 'xcol', 'xcols', 'xdesc', 'xexp', 'xgroup', 'xkey', 'xlog',
        'xprev', 'xrank'
    }

    def __init__(self):
        self.paths = default_paths
        # Start with an empty set to avoid errors during initialization.
        object.__setattr__(self, '_q_ctx_keys', set())
        # HACK: `'_connection_info' in self.__dict__` is used as a proxy for a type-check to avoid a
        # cyclic import.
        if licensed or '_connection_info' in self.__dict__:
            if '_connection_info' in self.__dict__ and self._connection_info['no_ctx']:
                object.__setattr__(self, 'ctx', QContext(proxy(self), '', None, no_ctx=True))
                pass
            else:
                object.__setattr__(self, 'ctx', QContext(proxy(self), '', None))
                object.__setattr__(self, '_q_ctx_keys', {
                    *self.ctx.q._context_keys,
                    *self.reserved_words,
                })
        object.__setattr__(self, 'console', QConsole(proxy(self)))
        object.__setattr__(self, 'insert', Insert(proxy(self)))
        object.__setattr__(self, 'upsert', Upsert(proxy(self)))
        object.__setattr__(self, 'query', None) # placeholder
        object.__setattr__(self, 'qsql', QSQL(proxy(self)))
        object.__setattr__(self, 'sql', SQL(proxy(self)))
        object.__setattr__(self, 'read', QReader(proxy(self)))
        object.__setattr__(self, 'write', QWriter(proxy(self)))
        object.__setattr__(self, 'system', SystemCommands(proxy(self)))

    @abstractmethod
    def __call__(self, query: str, *args, wait: Optional[bool] = None):
        pass # nocov

    def __getattr__(self, key):
        # Elevate the q context to the global context, as is done in q normally
        ctx = self.__getattribute__('ctx')
        if key in self.__getattribute__('_q_ctx_keys'):
            # if-statement used instead of try-block for performance
            return ctx.q.__getattr__(key)
        try:
            return ctx.__getattr__(key)
        except AttributeError as attribute_error:
            try:
                self.__getattribute__('_register')(name=key)
                ctx._invalidate_cache()
                return ctx.__getattr__(key)
            except Exception as inner_error:
                raise attribute_error from inner_error

    # __setattr__ takes precedence over data descriptors, so we implement a custom one that uses
    # the default behaviour for select keys
    def __setattr__(self, key, value):
        if key in self.__dict__ or key in Q.__dict__:
            object.__setattr__(self, key, value)
        else:
            if key in self.reserved_words or key in self.q:
                raise exceptions.PyKXException(
                    'Cannot assign to reserved word or overwrite q namespace.'
                )
            self._call('set', f'.{key}', value, wait=True)

    def __delattr__(self, key):
        self.ctx.__delattr__(key)

    def __getitem__(self, key):
        return self._call('get', key, wait=True)

    def __setitem__(self, key, value):
        if key in self.reserved_words or key in self.q:
            raise exceptions.PyKXException(
                'Cannot assign to reserved word or overwrite q namespace.'
            )
        self._call('set', key, value, wait=True)

    def __delitem__(self, key):
        key = str(key)
        if key.startswith('.'):
            raise exceptions.PyKXException('Cannot delete from the global context.')
        elif key not in self._call('key `.', wait=True).py():
            raise KeyError(key)
        self._call('{![`.;();0b;enlist x]}', key, wait=True) # delete key from `.

    def __dir__(self):
        if hasattr(self, 'ctx'): # Temporary workaround to disable the context interface over IPC.
            return sorted({
                *super().__dir__(),
                *dir(self.ctx),
                *dir(self.ctx.__getattr__('q')),
            })
        else:
            return super().__dir__()

    def _register(self,
                  name: Optional[str] = None,
                  path: Optional[Union[Path, str]] = None,
    ) -> str:
        """Obtain the definitions from a q/k script.

        Parameters:
            name: Name of the context to be loaded. If `path` is not provided, [a file whose name
                matches will be searched for](#script-search-logic), and loaded if found. If `path`
                is provided, `name` will be used as the name of the context that the script at
                `path` is executed in.
            path: Path to the script to load. If `name` is not provided, it will default to the
                filename sans extension.

        Returns:
            The attribute name for the newly loaded module.
        """
        if name is None and path is None:
            raise ValueError('Module name or path must be provided.')
        if path is not None:
            path = Path(path).resolve(strict=True)
        else:
            path = _first_resolved_path([''.join(x) for x in it.product(
                # `str(Path/@)[:-1]` gets the path with a trailing path separator
                (str(x/'@')[:-1] for x in self.paths),
                ('.', ''),
                (name,),
                ('.q', '.k'),
                ('', '_')
            )])
        if name is None: # defaults to filename at end of path sans extension
            name = path.stem
        prev_ctx = self._call('string system"d"', wait=True)
        try:
            self._call(
                f'{"" if name[0] == "." else "."}{name}:(enlist`)!enlist(::);'
                f'system "d {"" if name[0] == "." else "."}{name}";'
                f'system "l {path.as_posix()}"',
                wait=True,
            )
            return name[1:] if name[0] == '.' else name
        finally:
            self._call('{system"d ",x}', prev_ctx, wait=True)

    @property
    def paths(self):
        """List of locations for the context interface to find q scripts in.

        Defaults to the current working directory and `$QHOME`.

        If you change directories, the current working directory stored in this list will
        automatically reflect that change.
        """
        return object.__getattribute__(self, '_paths')

    @paths.setter
    def paths(self, paths: List[Union[str, Path]]):
        resolved = []
        for path in paths:
            if not isinstance(path, Path):
                path = Path(path)
            try:
                resolved.append(path.resolve(strict=True))
            except FileNotFoundError:
                warn(f"Module path '{path!r}' not found", RuntimeWarning)
        object.__setattr__(self, '_paths', resolved)

    @property
    def max_num_threads(self):
        """The maximum number of secondary threads available to q.

        This value can be set using the `-s` command-line flag, provided to q via the $QARGS
        environment variable. For example: `QARGS='-s 8' python`, or `QARGS='-s 0' python`.
        """
        warn('The q.max_num_threads property has been deprecated - use q.system.num_threads '
             'instead.',
             DeprecationWarning
        )
        return self.system.max_num_threads

    @max_num_threads.setter
    def max_num_threads(self, value):
        warn('The q.max_num_threads property has been deprecated - use q.system.num_threads '
             'instead.',
             DeprecationWarning
        )
        self.system.max_num_threads = value

    @property
    def num_threads(self):
        """The current number of secondary threads being used by q.

        Computations in q will automatically be parallelized across this number of threads as q
        deems appropriate.

        This property is meant to be modified on `EmbeddedQ` and `QConnection` instances.

        Examples:

        Set the number of threads for embedded q to use to 8.

        ```
        kx.q.num_threads = 8
        ```

        Set the number of threads for a q process being connected to over IPC to 8.

        ```
        q = kx.SyncQConnection('localhost', 5001)
        q.num_threads = 8
        ```
        """
        warn('The q.num_threads property has been deprecated - use q.system.num_threads instead.',
             DeprecationWarning
        )
        return self.system.num_threads

    @num_threads.setter
    def num_threads(self, value):
        warn('The q.num_threads property has been deprecated - use q.system.num_threads instead.',
             DeprecationWarning
        )
        self.system.num_threads = value


# Import order matters here, so the imports are not ordered conventionally.
from .console import QConsole
from .ctx import default_paths, QContext
from .query import Insert, QSQL, SQL, Upsert
from .read import QReader
from .system import SystemCommands
from .write import QWriter
from .reimporter import PyKXReimport

from . import config
from . import console
from . import exceptions
from . import wrappers
from . import schema

from ._wrappers import _init as _wrappers_init
_wrappers_init(wrappers)

from .embedded_q import EmbeddedQ, q
from ._version import version as __version__
from .exceptions import *

from ._ipc import _init as _ipc_init
_ipc_init(q)

from .ipc import AsyncQConnection, QConnection, QFuture, RawQConnection, SecureQConnection, SyncQConnection # noqa
from .config import qargs, qhome, qlic
from .wrappers import *
from .wrappers import CharVector, K

from ._ipc import ssl_info

from .schema import _init as _schema_init
_schema_init(q)

if k_allocator:
    from . import _numpy as _pykx_numpy_cext

if config.enable_pandas_api:
    def merge_asof(left, *args, **kwargs):
        return left.merge_asof(*args, **kwargs)


if sys.version_info[1] < 8:
    warn(
        'Python 3.7 has reach its end of life period and is no longer supported.'
        'Please consider upgrading to Python 3.8, as PyKX will no longer support issues for this '
        'Python version.',
        exceptions.PyKXWarning
    )


def install_into_QHOME(overwrite_embedpy=False) -> None:
    """Copies the embedded Python functionality of PyKX into `$QHOME`.

        Parameters:
            overwrite_embedpy: If embedPy had previously been installed replace it otherwise
                save functionality as pykx.q

        Returns:
            None
    """
    p = Path(qhome)/'p.k'
    if not p.exists() or overwrite_embedpy:
        shutil.copy(Path(__file__).parent/'p.k', p)
    shutil.copy(Path(__file__).parent/'pykx.q', qhome/'p.q' if overwrite_embedpy else qhome)
    shutil.copy(Path(__file__).parent/'pykx_init.q_', qhome)
    if platform.system() == 'Windows':
        shutil.copy(Path(__file__).parent/'lib/w64/q.dll', qhome)


def activate_numpy_allocator() -> None:
    """Sets the allocator used for Numpy array data to one optimzied for use with PyKX.

    This will only change the default allocator if the environment variable `PYKX_ALLOCATOR` is set
    to 1 or if the flag `--pykxalloc` is present in the QARGS environment variable.

    The name of the allocator set by this function is `'pykx_allocator'`.

    The name of the active Numpy allocator can be retrieved by running
    `numpy.core.multiarray.get_handler_name()`.

    Refer to NEP 49 for details about custom Numpy allocators: https://numpy.org/neps/nep-0049.html

    The custom allocator set by PyKX allocates Numpy array data into the embedded q memory space.
    Numpy arrays created with this allocator can be converted into a q vector without copying the
    data.

    Because q objects must have their metadata immediately preceeding the data, only a single
    q vector can be created using this approach. Repeated conversions of the Numpy array into a q
    vector will yield the same q vector with its reference count incremented by 1 each time.

    Since Numpy stores its metadata separately from its data, multiple views can be taken over the
    data without incurring a copy. The same is not true for q vectors. If a Numpy array is
    converted into a q vector, then a view is taken of the Numpy array, this results in the view
    being converted into a q vector and a copy of the data being incurred. Regardless of whether
    this custom Numpy allocator is activated.

    The Numpy array data object holds a reference to the q vector. When the Numpy array data object
    is deallocated, it decrements the refcount of the q vector.

    If the installed version of Numpy support custom allocators (i.e. is greater than or equal to
    version `1.22.0`), and the active Numpy allocator when PyKX is imported is the default
    allocator, PyKX will automatically call this function.

    Note that Numpy arrays created with an allocator other than `'pykx_allocator'` (e.g. before
    PyKX has been imported) will not gain the benefits enabled by this allocator. The name of the
    allocator used by a given array `x` can be retrieved by running
    `numpy.core.multiarray.get_handler_name(x)`.
    """
    if k_allocator:
        _pykx_numpy_cext.activate_pykx_allocators()


import numpy as np
if k_allocator:
    _pykx_numpy_cext.init_numpy_ctx(core._r0_ptr, core._k_ptr, core._ktn_ptr)
    if np.core.multiarray.get_handler_name() == 'default_allocator':
        activate_numpy_allocator()


def deactivate_numpy_allocator():
    """Deactivate the PyKX Numpy allocator.

    If the PyKX Numpy allocator has not been activated then this function has no effect. Otherwise,
    it replaces the PyKX Numpy allocator with the allocator that was in use when it was first
    activated.
    """
    if k_allocator:
        _pykx_numpy_cext.deactivate_pykx_allocators()


try:
    # If we are running under IPython/Jupyter...
    ipython = get_ipython()  # noqa
    # Load the PyKX extension for Jupyter Notebook.
    ipython.extension_manager.load_extension('pykx.nbextension')
except NameError:
    # Not running under IPython/Jupyter...
    pass


__all__ = sorted([
    'AsyncQConnection',
    'EmbeddedQ',
    'Q',
    'qargs',
    'QConnection',
    'QConsole',
    'QContext',
    'QFuture',
    'qhome',
    'QReader',
    'QWriter',
    'qlic',
    'SyncQConnection',
    'RawQConnection',
    'activate_numpy_allocator',
    'console',
    'ctx',
    'deactivate_numpy_allocator',
    'exceptions',
    'install_into_QHOME',
    'licensed',
    'toq',
    'schema',
    'config',
    'util',
    'q',
    'PyKXReimport',
    *exceptions.__all__,
    *wrappers.__all__,
])

try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except Exception:
    import logging
    logging.exception('Failed to set SIGINT handler...')


def __dir__():
    return __all__

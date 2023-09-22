from abc import ABCMeta
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union
from warnings import warn

from . import ipc
from . import pandas_api
from . import Q
from . import toq
from . import wrappers
from . import schema
from .config import find_core_lib, licensed, no_qce, pykx_dir, skip_under_q
from .core import keval as _keval
from .exceptions import FutureCancelled, LicenseException, NoResults, PyKXException, PyKXWarning, QError # noqa
from ._wrappers import _factory as factory


__all__ = [
    'EmbeddedQ',
    'EmbeddedQFuture',
    'q',
]


class EmbeddedQFuture():
    _result = None
    _exception = None

    def __init__(self):
        self._done = False # nocov
        self._cancelled = False # nocov
        self._cancelled_message = '' # nocov

    def __await__(self) -> Any:
        if self.done(): # nocov
            return self.result() # nocov
        while not self.done(): # nocov
            pass # nocov
        yield from self # nocov
        return self.result() # nocov

    async def __async_await__(self) -> Any:
        return await self # nocov

    def _await(self) -> Any:
        if self.done(): # nocov
            return self.result() # nocov
        while not self.done(): # nocov
            pass # nocov
        return self.result() # nocov

    def set_result(self, val: Any) -> None:
        self._result = val # nocov
        self._done = True # nocov

    def set_exception(self, err: Exception) -> None:
        self._done = True # nocov
        self._exception = err # nocov

    def result(self) -> Any:
        if self._exception is not None: # nocov
            raise self._exception # nocov
        if self._cancelled: # nocov
            raise FutureCancelled(self._cancelled_message) # nocov
        if self._result is not None: # nocov
            return self._result # nocov
        raise NoResults() # nocov

    def done(self) -> bool:
        return self._done or self._cancelled # nocov

    def cancelled(self) -> bool:
        return self._cancelled # nocov

    def cancel(self, msg: str = '') -> None:
        self._cancelled = True # nocov
        self._cancelled_message = msg # nocov

    def exception(self) -> BaseException:
        if self._cancelled: # nocov
            return FutureCancelled(self._cancelled_message) # nocov
        if not self._done: # nocov
            return NoResults() # nocov
        return self._exception # nocov

    def get_loop(self):
        raise PyKXException('QFutures do not rely on an event loop to drive them, ' # nocov
                            'and therefore do not have one.') # nocov

    __iter__ = __await__ # nocov


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def __dir__():
    return __all__


# The metaclass of a derived class (EmbeddedQ) must be a (non-strict) subclass of the metaclasses of
# all its bases (ABCMeta from Q).
class ABCMetaSingleton(ABCMeta):
    def __init__(self, name, bases, _dict):
        super().__init__(name, bases, _dict)
        self._singleton_instance = None

    def __call__(self):
        if self._singleton_instance is None:
            self._singleton_instance = super().__call__()
        return self._singleton_instance


class EmbeddedQ(Q, metaclass=ABCMetaSingleton):
    """Interface for q within the current process; can be called to execute q code."""
    def __init__(self): # noqa
        if licensed:
            kxic_path = (pykx_dir/'lib'/'kxic.k').as_posix()
            pykx_qlib_path = (pykx_dir/'pykx').as_posix()
            # This q code is run as a single call into q to improve startup performance:
            code = ''
            if not no_qce:
                code += f'if[not `comkxic in key `;system"l {kxic_path}"];'
            if os.getenv('PYKX_UNDER_Q') is None:
                code += 'setenv[`UNDER_PYTHON;"true"];'
                code += f'2:[`$"{pykx_qlib_path}";(`k_pykx_init; 1)][`$"{find_core_lib("q").as_posix()}"];'  # noqa: E501
                code += f'`.pykx.i.pyfunc set (`$"{pykx_qlib_path}") 2: (`k_pyfunc; 2);'
                code += f'`.pykx.modpow set {{((`$"{pykx_qlib_path}") 2: (`k_modpow; 3))["j"$x;"j"$y;$[z~(::);(::);"j"$z]]}};'  # noqa: E501
            else:
                code += f'2:[`$"{pykx_qlib_path}q";(`k_pykx_init; 1)][`$"{find_core_lib("q").as_posix()}"];'  # noqa: E501
                code += f'`.pykx.modpow set {{((`$"{pykx_qlib_path}q") 2: (`k_modpow; 3))["j"$x;"j"$y;$[z~(::);(::);"j"$z]]}};'  # noqa: E501
            code += '@[get;`.pykx.i.kxic.loadfailed;{()!()}]'
            kxic_loadfailed = self._call(code).py()
            if not no_qce:
                sql = self._call('$["insights.lib.sql" in " " vs .z.l 4; @[system; "l s.k_";{x}];::]').py()  # noqa: E501
                if sql is not None:
                    kxic_loadfailed['s.k'] = sql
            for lib, msg in kxic_loadfailed.items():
                if os.getenv('PYKX_DEBUG_INSIGHTS_LIBRARIES'):
                    warn(f'Failed to load KX Insights Core library {lib!r}: {msg.decode()}',
                         PyKXWarning)
                else:
                    eprint(f'WARN: Failed to load KX Insights Core library {lib!r}.')
            if (not skip_under_q and os.getenv('PYKX_Q_LOADED_MARKER') != 'loaded'
                and os.getenv('PYKX_UNDER_Q') is None
            ):
                os.environ['PYKX_Q_LOADED_MARKER'] = 'loaded'
                self._call('setenv[`PYKX_Q_LOADED_MARKER; "loaded"]')
                try:
                    self._call('.Q.ld')
                except QError as err:
                    if '.Q.ld' in str(err):
                        # .Q.ld is not defined on the server so we define it here
                        with open(Path(__file__).parent.absolute()/'lib'/'q.k', 'r') as f:
                            lines = f.readlines()
                        for line in lines:
                            if 'pykxld:' in line:
                                self._call('k).Q.' + line)
                                break
                    else:
                        raise err
                pykx_qini_path = (Path(__file__).parent.absolute()/'pykx_init.q_')
                self._call(f'\l {pykx_qini_path}') # noqa
                pykx_q_path = (Path(__file__).parent.absolute()/'pykx.q')
                with open(pykx_q_path, 'r') as f:
                    code = f.read()
                code = [wrappers.CharVector(x) for x in code.split('\n')][:-1]
                self._call(
                    "{[code;file] value (@';last file;enlist[file],/:.Q.pykxld code)}",
                    code,
                    b'pykx.q'
                )
                self._call('.pykx.setdefault[enlist"k"]')
        super().__init__()

    def __repr__(self):
        return 'pykx.q'

    def __call__(self,
                 query: Union[str, bytes, wrappers.CharVector],
                 *args: Any,
                 wait: Optional[bool] = None,
                 sync: Optional[bool] = None
    ) -> wrappers.K:
        """Run code in the q instance.

        Parameters:
            query: The code to run in the q instance.
            *args: Arguments to the q query. Each argument will be converted into a `pykx.K` object.
                Up to 8 arguments can be provided, as that is the maximum supported by q.
            wait: Keyword argument provided solely for conformity with `pykx.QConnection`. All
                queries against the embedded q instance are synchronous regardless of what this
                parameter is set to. Setting this keyword argument to `False` results in q generic
                null (`::`) being returned, so as to conform with `pykx.QConnection`. This
                conformity enables one to call any `pykx.Q` instance the same way regardless of
                whether it is a `pykx.EmbeddedQ` or `pykx.QConnection` instance. For cases where
                the query executing asynchronously (and returning after it has been issued, rather
                than after is is done executing) is actually important, one can discriminate
                between `pykx.Q` instances using `isinstance` as normal.

        Returns:
            The value obtained by evaluating the `query` within the current process.

        Raises:
            LicensedException: Attempted to execute q code outside of licensed mode.
            TypeError: Too many arguments were provided - q queries cannot have more than 8
                parameters.
        """
        if not licensed:
            raise LicenseException("run q code via 'pykx.q'")
        if len(args) > 8:
            raise TypeError('Too many arguments - q queries cannot have more than 8 parameters')
        result = _keval(bytes(wrappers.CharVector(query)), *[wrappers.K(x) for x in args])
        if wait is None or wait:
            return factory(result, False)
        return self('::', wait=True)

    # Asynchronous q calls internally use a _call method to run q code synchronously, so this has
    # been added in order to provide a consistent interface across all subclasses of `Q`.
    _call = __call__


q = EmbeddedQ()

# HACK: some modules are reliant on the EmbeddedQ instance, so we provide it to them here
ipc._init(q)
toq._init(q)
wrappers._init(q)
pandas_api._init(q)
schema._init(q)

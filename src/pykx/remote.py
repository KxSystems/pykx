"""
Functionality for the generation and management of remote Python function
execution.

!!! Warning

        This functionality is provided in it's present form as a BETA
        Feature and is subject to change. To enable this functionality
        for testing please following configuration instructions
        [here](../user-guide/configuration.md) setting `PYKX_BETA_FEATURES='true'`
"""
import inspect
from typing import Union

from . import beta_features
from .config import _check_beta
from .ipc import SyncQConnection


try:
    import dill
    import_success = True
except BaseException:
    import_success = False


__all__ = [
    'session',
    'function'
]

beta_features.append('Remote Functions')


def _init(_q):
    global q
    q = _q


def __dir__():
    return __all__


class session():
    """
    A session refers to a connection to a remote kdb+/q process against which
    users are defining/registering Python Functions which will return results
    to a Python session.
    """
    def __init__(self):
        _check_beta('Remote Functions')
        if not import_success:
            raise ImportError("Failed to load Python package: 'dill',"
                              " please install dependency using 'pip install pykx[beta]'")
        self.valid = False
        self._libraries = []
        self._session = None

    def add_library(self, *args):
        """
        Add a list of Python libraries which will be imported prior to definition
        of a remote Python function, this allows users for example to import numpy
        and use it as a defined library within a remote function.

        Parameters:
            *args: A list of strings denoting the packages which are to be imported
                for use by a remote function.

        Returns:
            Returns a `None` type object on successful invocation.

        Example:

        ```python
        >>> from pykx.remote import session
        >>> remote_session = session()
        >>> remote_session.add_library('numpy', 'pandas')
        ```
        """
        if self._session is None:
            raise Exception("Unable to add packages in the absence of a session")
        for i in args:
            if not isinstance(i, str):
                raise Exception(f'Supplied library argument {i} is not a str like object, '
                                f'supplied object is of type: {type(i)}')
            self._libraries.append(i)

    def create(self,
               host: Union[str, bytes] = 'localhost',
               port: int = None,
               *,
               username: Union[str, bytes] = '',
               password: Union[str, bytes] = '',
               timeout: float = 0.0,
               large_messages: bool = True,
               tls: bool = False):
        """
        Populate a session for use when generating a function for remote execution. This
        session will be backed by a SyncQConnection instance, note that only one session
        can be associated with a given instance of a `session` class.

        Parameters:
            host: The host name to which a connection is to be established.
            port: The port to which a connection is to be established.
            username: Username for q connection authorization.
            password: Password for q connection authorization.
            timeout: Timeout for blocking socket operations in seconds. If set to `0`, the socket
                will be non-blocking.
            large_messages: Whether support for messages >2GB should be enabled.
            tls: Whether TLS should be used.

        Returns:
            Returns a `None` type object on successful connection creation

        Example:

        - Connect to a q session on localhost at port 5050

            ```python
            >>> from pykx.remote import session
            >>> remote_session = session()
            >>> remote_session.create(port = 5050)
            ```

        - Connect to a user-password protected q session at a defined port

            ```python
            >>> from pykx.remote import session
            >>> remote_session = session()
            >>> remote_session.create_session(port=5001, username='username', password='password')
            ```
        """
        if self._session is not None:
            raise Exception("Active session in progress")
        self._session = SyncQConnection(host, port,
                                        username=username,
                                        password=password,
                                        timeout=timeout,
                                        large_messages=large_messages,
                                        tls=tls,
                                        no_ctx=True)

    def clear(self):
        """
        Reset/clear the session and libraries associated with a defined session information

        Example:

        ```python
        >>> from pykx.remote import session
        >>> remote_session = session()
        >>> remote_session.create(port = 5050)
        >>> remote_session.add_library('numpy')
        >>> {'session': session._session, 'libraries': session._libraries}
        {'session': pykx.QConnection(port=5001), 'libraries': ['numpy']}
        >>> remote_session.clear()
        >>> {'session': session._session, 'libraries': session._libraries}
        {'session': None, 'libraries': []}
        ```
        """
        self._session = None
        self._libraries = []


def function(remote_session, *args):
    """
    This decorator allows users to tag functions which will be executed
    on a remote server defined by a `kx.remote.session` instance.

    Parameters:
        remote_session: Valid `kx.remote.session` object used to interact with external q process
        *args: When invoked the decorated function will be passed supplied arguments

    Returns:
        When invoked the decorated function will return the result as a PyKX object to the
        calling process

    Examples:

    - Call a basic decorated function on a remote process

        ```python
        >>> from pykx.remote import session, function
        >>> remote_session = session()
        >>> session.create(port = 5050)
        >>> @function(session)
        ... def func(x):
        ...    return x+1
        >>> func(1)
        pykx.LongAtom(pykx.q('2'))
        ```

    - Apply a function making use of a named library

        ```python
        >>> from pykx.remote import session, function
        >>> remote_session = session()
        >>> session.create(port = 5050)
        >>> session.add_library('numpy')
        >>> @function(session)
        ... def func(start, stop, count):
        ...     return numpy.linspace(start, stop, count)
        >>> func(0, 10, 5)
        pykx.FloatVector(pykx.q('0 2.5 5 7.5 10'))
        ```
    """
    def inner_decorator(func):
        def pykx_func(*args, _function=func):
            _check_beta('Remote Functions')
            if not isinstance(remote_session, session):
                raise Exception("Supplied remote_session instance must "
                                "be a kx.remote.session object")
            if remote_session._session is None:
                raise Exception("User session must be generated using "
                                "the 'create_session' function")
            if not remote_session.valid:
                pykx_loaded = remote_session._session('`pykx in key `')
                if not pykx_loaded:
                    print("PyKX not loaded on remote server, attempting to load PyKX")
                    remote_session._session("@[system\"l \",;\"pykx.q\";"
                                            "{'\"Failed to load PyKX with error: \",x}]")
                remote_session.valid = True
            if remote_session._libraries is not None:
                for i in remote_session._libraries:
                    remote_session._session('{x:string x;'
                                            '  @[.pykx.pyexec;'
                                            '"import ",x;{\'"Failed to load package: ",'
                                            'x," with: ",y}[x]]}',
                                            i)
            try:
                src = dill.source.getsource(_function)
            except BaseException:
                src = inspect.getsource(_function)
            return remote_session._session('{.pykx.pyexec "\n" sv 1_"\n" vs x; .pykx.get[y;<] . z}',
                                           bytes(src, 'UTF-8'),
                                           _function.__name__,
                                           list(args))
        return pykx_func
    return inner_decorator

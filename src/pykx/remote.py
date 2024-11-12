"""
_This page documents the API for generation and management of remote Python function execution._
"""
import inspect
from typing import Union

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


def _init(_q):
    global q
    q = _q


def __dir__():
    return __all__


class session:
    def __init__(self,
                 host: Union[str, bytes] = 'localhost',
                 port: int = None,
                 libraries: dict = None,
                 *,
                 username: Union[str, bytes] = '',
                 password: Union[str, bytes] = '',
                 timeout: float = 0.0,
                 large_messages: bool = True,
                 tls: bool = False,
                 reconnection_attempts: int = -1
    ) -> None:
        """
        Initialise a session object, opening a connection to the specified remote q process. Users
        can specify the Python libraries to load into the remote process. Once the connection is
        successful, pykx will be loaded if it is not, then the requested libraries will be imported.

        Parameters:
            host: The host name running the remote process.
            port: The port of the remote process.
            libraries: A dictionary mapping the desired name of the imported Python library to
                its library which is being imported
            username: Username for q connection authorization.
            password: Password for q connection authorization.
            timeout: Number of seconds to set the timeout for blocking socket operations. Input 0
                to set the socket to non-blocking.
            large_messages: Boolean flag to enable/disable messages >2GB in size.
            tls: Boolean flag to enable/disable TLS.
            reconnection_attempts: The number of attempts to reconnect to the q server when there is
                a disconnect. Input a negative value to disable reconnect attempts. A value of 0
                indicates no limit on reconnect attempts, with each attempt applying an exponential
                backoff on the time between successive attempts. Input a positive number to
                specify the maximum number of reconnect attempts. Hitting the maximum without a
                successful reconnect will throw an error.

        Examples:

        - Generate a session connecting to a process running locally

            ```python
            >>> import pykx as kx
            >>> remote_session = kx.remote.session(port=5050)
            ```

        - Generate a session connecting to a remote q process, providing required Python libraries,
          a username and password

            ```python
            >>> import pykx as kx
            >>> remote_session = kx.remote.session(
            ...     port = 5050,
            ...     username = 'user',
            ...     password = 'pass',
            ...     libraries = {'kx': 'pykx', 'np': 'numpy'})
            ```
        """
        if not import_success:
            raise ImportError("Failed to load Python package: 'dill',"
                              " please install dependency using 'pip install pykx[remote]'")
        if not (isinstance(libraries, dict) or libraries is None):
            raise TypeError("libraries must be supplied as a dictionary or None")

        self.valid = False
        self._libraries = libraries
        self._session = SyncQConnection(host, port,
                                        username=username,
                                        password=password,
                                        timeout=timeout,
                                        large_messages=large_messages,
                                        tls=tls,
                                        no_ctx=True)
        pykx_loaded = self._session('`pykx in key `')
        if not pykx_loaded:
            print("PyKX not loaded on remote server, attempting to load PyKX")
            self._session('@[system"l ",;"pykx.q";{\'"Failed to load PyKX with error: ",x}]')
            self.valid = True
        if self._libraries is not None:
            self.libraries(self._libraries)

    def libraries(self, libs: dict = None) -> None:
        """
        Send a list of libraries to the remote process and load them into that process.

        Parameters:
            libs: A dictionary mapping the desired name of the imported Python library to
                its library which is being imported

        Returns:
            `#!python None` if successful.

        Example:

        ```python
        >>> import pykx as kx
        >>> remote_session = kx.remote.session(port=5050)
        >>> remote_session.libraries({'np': 'numpy', 'pd': 'pandas', 'kx': 'pykx'})
        ```
        """
        if not isinstance(libs, dict):
            raise TypeError("libs must be provided as a dictionary")
        for key, value in libs.items():
            self._session('''
                {[alias;library]
                  alias:string alias;
                  library:string library;
                  @[.pykx.pyexec;
                    "import ",library," as ",alias;
                    {'"Failed to load library '",x,
                      "' with alias '",y,"' with error: ",z}[library;alias]
                    ]}
                ''', key, value)

    def close(self) -> None:
        """
        Close the connection.

        Example:

        ```python
        >>> from pykx.remote import session
        >>> remote_session = session(port=5050)
        >>> remote_session.close()
        ```
        """
        if self._session is not None:
            self._session.close()


def function(remote_session: session, *args) -> None:
    """
    This decorator allows users to tag functions which will be executed
    on a remote server defined by a `#!python kx.remote.session` instance.

    Parameters:
        remote_session: Valid `#!python kx.remote.session` object used to interact with external
        q process
        *args: Arguments that will be passed to the decorated function when it is invoked

    Returns:
        A PyKX converted type of the result returned from the execution of the decorated function
        on the remote process

    Examples:

    - Call a basic decorated function on a remote process

        ```python
        >>> import pykx as kx
        >>> session = kx.remote.session(port=5050)
        >>> @kx.remote.function(session)
        ... def func(x):
        ...    return x+1
        >>> func(1)
        pykx.LongAtom(pykx.q('2'))
        ```

    - Initialize a remote session object with a named library then decorate a function which uses
      that session to call functions from that library

        ```python
        >>> import pykx as kx
        >>> session = kx.remote.session(port=5050, libraries={'np': 'numpy'})
        >>> @kx.remote.function(session)
        ... def func(start, stop, count):
        ...     return numpy.linspace(start, stop, count)
        >>> func(0, 10, 5)
        pykx.FloatVector(pykx.q('0 2.5 5 7.5 10'))
        ```

    - Initialize a remote session object. Once created have that session import a new library.

        ```python
        >>> import pykx as kx
        >>> session = kx.remote.session(port=5050)
        >>> session.libraries({'kx': 'pykx'})
        >>> @kx.remote.function(session)
        ... def func(start, stop):
        ...     return start + kx.q.til(stop)
        >>> func(10, 5)
        pykx.LongVector(pykx.q('10 11 12 13 14'))
        ```
    """
    def inner_decorator(func):
        def pykx_func(*args, _function=func):
            if not isinstance(remote_session, session):
                raise Exception("Supplied remote_session instance must "
                                "be a kx.remote.session object")
            try:
                src = dill.source.getsource(_function)
            except BaseException:
                src = inspect.getsource(_function)
            return remote_session._session('''
                                           {[code;func_name;args;lenargs]
                                             .pykx.pyexec trim "\n" sv 1_"\n" vs code;
                                             func:.pykx.get[func_name;<];
                                             $[lenargs;func . args;func[]]
                                             }
                                           ''',
                                           bytes(src, 'UTF-8'),
                                           _function.__name__,
                                           list(args),
                                           len(args))
        return pykx_func
    return inner_decorator

import warnings

from . import beta_features
from .config import _check_beta, pykx_threading, system
from .exceptions import QError
from .ipc import SyncQConnection

beta_features.append('Streamlit Integration')


# This class is required to ensure that in the absence
# of the streamlit dependency PyKX can be imported
class _dummy_class(object):
    def __getattr__(self, item):
        return self

    def __call__(self, *args, **kwargs):
        return self


try:
    from streamlit.connections import BaseConnection
    _streamlit_unavailable = False
except ImportError:
    # This base connection object is to ensure the streamlit
    # class can be initialized correctly
    BaseConnection = {SyncQConnection: _dummy_class}
    _streamlit_unavailable = True


def _check_streamlit():
    if _streamlit_unavailable:
        raise QError('Use of streamlit functionality requires access to '
                     'of streamlit as a dependency, this can be installed '
                     ' using:\n\npip install pykx[streamlit]')


class PyKXConnection(BaseConnection[SyncQConnection]):
    """
    A connection to q/kdb+ processes from streamlit. Initialize using:

    ```python
    st.connection("<name>", type = pykx.streamlit.PyKXConnection, *args)
    ```

    PyKX Connection supports the application of queries using Syncronous IPC
    connections to q/kdb+ processes or Python processes running PyKX as a
    server.

    This is supported through the ``query()`` method, this method allows
    users to run `sql`, `qsql` or `q` queries against these processes returning
    PyKX data.

    !!! Warning
            Streamlit integration is not presently supported for Windows as for
            full utilization it requires use of `PYKX_THREADING` functionality

    Parameters:
        host: The host name to which a connection is to be established.
        port: The port to which a connection is to be established.
        username: Username for q connection authorization.
        password: Password for q connection authorization.
        timeout: Timeout for blocking socket operations in seconds. If set to `0`, the socket
            will be non-blocking.
        large_messages: Whether support for messages >2GB should be enabled.
        tls: Whether TLS should be used.
        unix: Whether a Unix domain socket should be used instead of TCP. If set to `True`, the
            host parameter is ignored. Does not work on Windows.
        wait: Whether the q server should send a response to the query (which this connection
            will wait to receive). Can be overridden on a per-call basis. If `True`, Python will
            wait for the q server to execute the query, and respond with the results. If
            `False`, the q server will respond immediately to every query with generic null
            (`::`), then execute them at some point in the future.

    Note: The `username` and `password` parameters are not required.
        The `username` and `password` parameters are only required if the q server requires
        authorization. Refer to [ssl documentation](https://code.kx.com/q/kb/ssl/) for more
        information.

    Note: The `timeout` argument may not always be enforced when making succesive querys.
        When making successive queries if one query times out the next query will wait until a
        response has been recieved from the previous query before starting the timer for its own
        timeout. This can be avioded by using a seperate `SyncQConnection` instance for each
        query.

    Examples:

    Connect to a q process at `localhost` on port `5050` as a streamlit connection,
    querying using q

    ```python
    >>> import streamlit as st
    >>> import pykx as kx
    >>> conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
    ...                      host = 'localhost', port = 5050)
    >>> df = conn.query('select from tab').pd()
    >>> st.dataframe(df)
    ```
    """
    _connection = None
    _connection_kwargs = {}

    def _connect(self, **kwargs) -> None:
        _check_beta('Streamlit Integration')
        _check_streamlit()
        if system == 'Windows':
            raise QError('Streamlit integration currently unsupported for Windows')
        if not pykx_threading:
            warnings.warn("Streamlit caching requires execution on secondary threads, "
                          "to utilize this fully please consider setting PYKX_THREADING "
                          "= 'True'")
        self._connection = SyncQConnection(no_ctx=True, **kwargs)
        self._connection_kwargs = kwargs

    def reset(self, **kwargs) -> None:
        """
        Reset an existing Streamlit Connection object, this can be used to manually
        reconnect to a datasource which was disconnected. This will use the connection
        details provided at initialisation of the original class.

        Example:

        Reset a connection if deemed to no longer be valid

        ```python
        >>> import streamlit as st
        >>> import pykx as kx
        >>> conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
        ...                      host = 'localhost', port = 5050)
        >>> if not conn.is_healthy():
        ...     conn.reset()
        >>>
        ```
        """
        _check_beta('Streamlit Integration')
        _check_streamlit()
        if not isinstance(self._connection, SyncQConnection):
            raise QError('Unable to reset uninitialized connection')
        self._connection.close()
        self._connect(**self._connection_kwargs)

    def is_healthy(self) -> bool:
        """
        Check if an existing streamlit connection is 'healthy' and
        available for query.

        Returns:
            A boolean indicating if the connection being used is in a
                'healthy' state

        Example:

        ```python
        >>> import streamlit as st
        >>> import pykx as kx
        >>> conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
        ...                      host = 'localhost', port = 5050)
        >>> conn.is_healthy()
        True
        ```
        """
        _check_beta('Streamlit Integration')
        _check_streamlit()
        if not isinstance(self._connection, SyncQConnection):
            raise QError('Unable to validate uninitialized connection')
        if self._connection.closed:
            warnings.warn('Connection closed')
            return False
        try:
            self.query('::')
            return True
        except BaseException as err:
            warnings.warn('Unhealthy connection detected with error: ' + str(err))
            return False

    def query(self, query: str, *args, format='q', **kwargs):
        """
        Evaluate a query on the connected q process over IPC.

        Parameters:
            query: A q expression to be evaluated.
            *args: Arguments to the q query. Each argument will be converted into a `pykx.K`
                object. Up to 8 arguments can be provided, as that is the maximum
                supported by q.
            format: What execution format is to be used, should the function use the `qsql`
                interface, execute a `sql` query or run `q` code.

        Raises:
            RuntimeError: A closed IPC connection was used.
            QError: Query timed out, may be raised if the time taken to make or receive a query
                goes over the timeout limit.
            TypeError: Too many arguments were provided - q queries cannot have more than 8
                parameters.
            ValueError: Attempted to send a Python function over IPC.

        Examples:

        Connect to a q process at `localhost` on port `5050` as a streamlit connection,
        querying using q

        ```python
        >>> import streamlit as st
        >>> import pykx as kx
        >>> conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
        ...                      host = 'localhost', port = 5050)
        >>> df = conn.query('select from tab').pd()
        >>> st.dataframe(df)
        ```

        Connect to a q process at `localhost` on port `5050` as a streamlit connection,
        querying using qsql

        ```python
        >>> import streamlit as st
        >>> import pykx as kx
        >>> conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
        ...                      host = 'localhost', port = 5050)
        >>> df = conn.query('tab', where='x>0.5', format='qsql').pd()
        >>> st.dataframe(df)
        ```

        Connect to a q process at `localhost` on port `5050` as a streamlit connection,
        querying using sql

        ```python
        >>> import streamlit as st
        >>> import pykx as kx
        >>> conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
        ...                      host = 'localhost', port = 5050)
        >>> df = conn.query('select * from tab where x>0.5', format='sql').pd()
        >>> st.dataframe(df)
        ```
        """
        _check_beta('Streamlit Integration')
        _check_streamlit()

        def _query(query: str, format, args, kwargs):
            if format == 'sql':
                try:
                    res = self._connection.sql(query, *args)
                except QError as err:
                    if '.s.sp' in str(err):
                        raise QError('SQL functionality not loaded on connected server, error: ' + str(err)) # noqa: E501
                    raise QError(err)
                return res
            elif format == 'q':
                return self._connection(query, *args, **kwargs)
            if format == 'qsql':
                return self._connection.qsql.select(query, *args, **kwargs)
            else:
                raise QError("Unsupported format provided for query, must be one of 'q', 'qsql' or 'sql'") # noqa: E501
        return _query(query, format, args, kwargs)

from warnings import warn

from .config import pykx_threading, suppress_warnings, system
from .exceptions import QError
from .ipc import SyncQConnection


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
    A connection to a q server from Streamlit. Initialise using:

    ```python
    st.connection("<name>", type = pykx.streamlit.PyKXConnection, *args)
    ```

    !!! Warning
            Streamlit integration is not supported for Windows.
            Full utilization requires `#!bash PYKX_THREADING` which is not supported on windows.

    Parameters:
        host: Server host name.
        port: Server port number.
        username: Username for q connection.
        password: Password for q connection.
        timeout: The number of seconds before a blocking operation times out. A value of 0 creates
            a non-blocking connection.
        large_messages: Boolean flag to enable/disable support for messages >2GB.
        tls: Boolean flag to enable/disable TLS.
        unix: Boolean flag to enable Unix domain socket connection. Host parameter is ignored if
            `#!python True`. Does not work on Windows.
        wait: Boolean to enable/disable waiting for the q server to complete executing the query
            and return the result. `#!python False` emulates async queries, causing the q server
            to respond immediately with the generic null `#!q ::` and perform calculations at
            another time.
        reconnection_attempts: The number of maximum attempts to reconnect when a connection is
            lost. A negative number prevents any attempts to reconnect. A value of 0 will cause
            continuous reconnect attempts until a connection is established. Positive values are
            the number of times to attempt. Successive reconnect attempts are run at exponentially
            increasing backoff times. Hitting the maximum number of limits with unsuccessful
            attempts will throw an error.

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

    Open a streamlit connection to a locally running q process on port 5050.

    ```python
    >>> import streamlit as st
    >>> import pykx as kx
    >>> conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
    ...                      host = 'localhost', port = 5050)
    >>>
    ```
    """
    _connection = None
    _connection_kwargs = {}

    def _connect(self, **kwargs) -> None:
        _check_streamlit()
        if system == 'Windows':
            raise QError('Streamlit integration currently unsupported for Windows')
        if (not pykx_threading) and (not suppress_warnings):
            warn("Streamlit caching requires execution on secondary threads, "
                 "to utilize this fully please consider setting PYKX_THREADING "
                 "= 'True'. To suppress this warning please consider setting "
                 "PYKX_SUPPRESS_WARNINGS = 'True'")
        self._connection = SyncQConnection(no_ctx=True, **kwargs)
        self._connection_kwargs = kwargs

    def reset(self, **kwargs) -> None:
        """
        Close and reopen an existing Streamlit connection.

        Example:

        Open a connection to a locally running process on port 5050 and check if it is a healthy
        connection. If it is not, reset the connection.

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
        _check_streamlit()
        if not isinstance(self._connection, SyncQConnection):
            raise QError('Unable to reset uninitialized connection')
        self._connection.close()
        self._connect(**self._connection_kwargs)

    def is_healthy(self) -> bool:
        """
        Check if an existing streamlit connection is 'healthy' and available for query.

        Returns:
            A boolean indicating if the connection being used is in a 'healthy' state

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
        _check_streamlit()
        if not isinstance(self._connection, SyncQConnection):
            raise QError('Unable to validate uninitialized connection')
        if self._connection.closed:
            warn('Connection closed')
            return False
        try:
            self.query('::')
            return True
        except BaseException as err:
            warn('Unhealthy connection detected with error: ' + str(err))
            return False

    def query(self, query: str, *args, format='q', **kwargs):
        """
        Query the connected q process over IPC.

        Parameters:
            query: A q expression to be evaluated. This must be valid q, qSQL or SQL in the KX
                Insights style.
            *args: Arguments to the query. Each argument will be converted into a `#!python pykx.K`
                object. Up to 8 arguments can be provided (maximum supported by q functions).
            format: Description of query format for internal pre-processing before the query is sent
                to the server. This must be one of 'q', 'qsql' or 'sql'.

        Raises:
            RuntimeError: A closed IPC connection was used.
            QError: Query timed out, may be raised if the time taken to make or receive a query
                goes over the timeout limit.
            TypeError: Too many arguments were provided - q queries cannot have more than 8
                parameters.
            ValueError: Attempted to send a Python function over IPC.

        Examples:

        Open a connection to a locally running q process on port 5050 and query using 'q' format.

        ```python
        >>> import streamlit as st
        >>> import pykx as kx
        >>> conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
        ...                      host = 'localhost', port = 5050)
        >>> df = conn.query('select from tab').pd()
        >>> st.dataframe(df)
        ```

        Open a connection to a locally running q process on port 5050 and query using 'qsql'
        format.

        ```python
        >>> import streamlit as st
        >>> import pykx as kx
        >>> conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
        ...                      host = 'localhost', port = 5050)
        >>> df = conn.query('tab', where='x>0.5', format='qsql').pd()
        >>> st.dataframe(df)
        ```

        Connect to a locally running q process on port 5050 and query using 'sql'
        format.

        ```python
        >>> import streamlit as st
        >>> import pykx as kx
        >>> conn = st.connection('pykx', type=kx.streamlit.PyKXConnection,
        ...                      host = 'localhost', port = 5050)
        >>> df = conn.query('select * from tab where x>0.5', format='sql').pd()
        >>> st.dataframe(df)
        ```
        """
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

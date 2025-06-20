"""
_This page documents the API functions for using q IPC within PyKX._
"""

from enum import Enum
from abc import abstractmethod
import asyncio
from contextlib import nullcontext
from multiprocessing import Lock as multiprocessing_lock, RawValue
from pathlib import Path
import selectors
import socket
from threading import Lock as threading_lock
from time import monotonic_ns, sleep
from typing import Any, Callable, Optional, Union
import warnings
from weakref import finalize, WeakMethod
import warnings
import sys

from . import deserialize, serialize, Q
from .config import max_error_length, pykx_lib_dir, pykx_qdebug, system
from .core import licensed
from .exceptions import FutureCancelled, NoResults, PyKXException, QError, UninitializedConnection
from .util import get_default_args, normalize_to_bytes, normalize_to_str
from .wrappers import CharVector, Composition, Foreign, Function, K, List, SymbolAtom, SymbolicFunction, Table # noqa : E501
from . import _wrappers
from . import _ipc


__all__ = [
    'AsyncQConnection',
    'QConnection',
    'QFuture',
    'RawQConnection',
    'SecureQConnection',
    'SyncQConnection',
]


def __dir__():
    return __all__


def _init(_q):
    global q
    q = _q

_ipc_err_warning = {
  'reconnect_warn': 'WARNING: Connection lost attempting to reconnect.',
  'delay_type': 'reconnection_delay must be either int/float',
  'reconnected': 'Connection successfully reestablished.',
  'closed': 'Attempted to use a closed IPC connection',
  'cannot_send': 'Cannot send object of passed type over IPC: ',
  'timeout': 'Query timed out'
}

def reconnection_function(reconnection_delay):
    return reconnection_delay * 2

class MessageType(Enum):
    """
    The message types available to q.

      - 0 = async message
      - 1 = sync message
      - 2 = response message
    """
    async_msg = 0
    sync_msg = 1
    resp_msg = 2


class QFuture(asyncio.Future):
    """
    A Future object to be returned by calls to q from an instance of
    [pykx.AsyncQConnection][pykx.AsyncQConnection] or [pykx.RawQConnection][pykx.RawQConnection].

    This object can be awaited to receive the resulting value.

    Examples:

    Await an instance of this class to receive the return value of an
    `#!python AsyncQConnection` call.

    ```python
    async with pykx.AsyncQConnection('localhost', 5001) as q:
        q_future = q('til 10') # returns a QFuture object
        q_result = await q_future
    ```
    """
    _result = None
    _exception = None
    _callbacks = []

    def __init__(self, q_connection, timeout, debug, poll_recv=None):
        self.q_connection = q_connection
        self._done = False
        self._cancelled = False
        self._cancelled_message = ''
        self._timeout = timeout
        self.poll_recv = poll_recv
        self._debug = debug
        super().__init__()

    def __await__(self) -> Any:
        """Await the result of the `#!python QFuture`.

        Returns:
            The result of the `#!python QFuture`.

        Raises:
            FutureCancelled: This QFuture instance has been cancelled and cannot be awaited.
            BaseException: If the future has an exception set it will be raised upon awaiting it.
        """
        async def closure():
            await self.q_connection._recv2(acceptAsync=True, fut=self)
            await asyncio.sleep(0)
            return self

        if self.done():
            return self.result()
        while not self.done():
            if self.done():
                return self.result()
            if self.poll_recv is not None:
                try:
                    res = self.q_connection.poll_recv()
                    if res is not None:
                        self.set_result(res)
                except BaseException as e:
                    self.set_exception(QError(str(e)))
            else:
                try:
                    return closure().__await__()
                except BaseException as e:
                    if isinstance(e, QError):
                        raise e
                    if self.q_connection._connection_info['reconnection_attempts'] != -1:
                        self.q_connection._cancel_all_futures()
                        print(_ipc_err_warning['reconnect_warn'], file=sys.stderr)
                        loops = self.q_connection._connection_info['reconnection_attempts']
                        reconnection_delay = self.q_connection._connection_info['reconnection_delay']
                        reconnection_function = self.q_connection._connection_info['reconnection_function']
                        while True:
                            try:
                                self.q_connection._create_connection_to_server()
                            except BaseException as err:
                                # attempts = 0 is infinite attempts as it will go to -1
                                # before the check to break
                                loops -= 1
                                if loops == 0:
                                    print(
                                        'WARNING: Could not reconnect to server within '
                                        f'{self.q_connection._connection_info["reconnection_attempts"]} attempts.',
                                        file=sys.stderr
                                    ) # noqa
                                    raise err
                                print(
                                    f'Failed to reconnect, trying again in {reconnection_delay} '
                                    'seconds.',
                                    file=sys.stderr
                                )
                                if not isinstance(reconnection_delay, (int, float)):
                                    raise TypeError(
                                        _ipc_err_warning['delay_type']
                                    )
                                sleep(reconnection_delay)
                                reconnection_delay = reconnection_function(reconnection_delay)
                                continue
                            print(_ipc_err_warning['reconnected'], file=sys.stderr)
                            break
                    else:
                        raise e
        yield from self
        super().__await__()
        return self.result()

    async def __async_await__(self) -> Any:
        if self.done():
            return self.result()

        while not self.done():
            await asyncio.sleep(0)
            if self.done():
                return self.result()
            if self.poll_recv is not None:
                try:
                    res = await self.q_connection.poll_recv2(fut=self)
                    if res is not None:
                        self.set_result(res)
                        return res
                except BaseException as e:
                    self.set_exception(QError(str(e)))
            else:
                try:
                    await self.q_connection._recv2(acceptAsync=True, fut=self)
                except BaseException as e:
                    if isinstance(e, QError):
                        raise e
                    if self.q_connection._connection_info['reconnection_attempts'] != -1:
                        self.q_connection._cancel_all_futures()
                        print(_ipc_err_warning['reconnect_warn'], file=sys.stderr)
                        loops = self.q_connection._connection_info['reconnection_attempts']
                        reconnection_delay = self.q_connection._connection_info['reconnection_delay']
                        reconnection_function = self.q_connection._connection_info['reconnection_function']
                        while True:
                            try:
                                self.q_connection._create_connection_to_server()
                            except BaseException as err:
                                # attempts = 0 is infinite attempts as it will go to -1 before the
                                # check to break
                                loops -= 1
                                if loops == 0:
                                    print(
                                        'WARNING: Could not reconnect to server within '
                                        f'{self.q_connection._connection_info["reconnection_attempts"]} attempts.',
                                        file=sys.stderr
                                    ) # noqa
                                    raise err
                                print(
                                    f'Failed to reconnect, trying again in {reconnection_delay} '
                                    'seconds.',
                                    file=sys.stderr
                                )
                                if not isinstance(reconnection_delay, (int, float)):
                                    raise TypeError(
                                        _ipc_err_warning['delay_type']
                                    )
                                sleep(reconnection_delay)
                                reconnection_delay = reconnection_function(reconnection_delay)
                                continue
                            print(_ipc_err_warning['reconnected'], file=sys.stderr)
                            break
                    else:
                        raise e
        if self.done():
            return self.result()
        return await self

    def _await(self) -> Any:
        if self.done():
            return self.result()
        try:
            while not self.done():
                self.q_connection._recv(locked=True, acceptAsync=True)
        except BaseException as e:
            if isinstance(e, QError):
                raise e
            if self.q_connection._connection_info['reconnection_attempts'] != -1:
                # TODO: Clear call stack futures
                print(_ipc_err_warning['reconnect_warn'], file=sys.stderr)
                loops = self._connection_info['reconnection_attempts']
                reconnection_delay = self.q_connection._connection_info['reconnection_delay']
                reconnection_function = self.q_connection._connection_info['reconnection_function']
                while True:
                    try:
                        self._create_connection_to_server()
                    except BaseException as err:
                        # attempts = 0 is infinite attempts as it will go to -1 before the check
                        # to break
                        loops -= 1
                        if loops == 0:
                            print(
                                'WARNING: Could not reconnect to server within '
                                f'{self._connection_info["reconnection_attempts"]} attempts.',
                                file=sys.stderr
                            )
                            raise err
                        print(
                            f'Failed to reconnect, trying again in {reconnection_delay} seconds.',
                            file=sys.stderr
                        )
                        if not isinstance(reconnection_delay, (int, float)):
                            raise TypeError(
                                _ipc_err_warning['delay_type']
                            )
                        sleep(reconnection_delay)
                        reconnection_delay = reconnection_function(reconnection_delay)
                        continue
                    print(_ipc_err_warning['reconnected'], file=sys.stderr)
                    break
            else:
                raise e
        return self.result()

    def set_result(self, val: Any) -> None:
        """Set the result of the `#!python QFuture` and mark it as done.

        The result is set first, then any functions in the callback list will execute
        with this `#!python Qfuture` as the only parameter input.

        Parameters:
            val: The value to set as the result of the `#!python QFuture`.
        """
        self._result = val
        for _ in self._callbacks:
            callback = self._callbacks.pop(0)
            callback(self)
        self._done = True

    def set_exception(self, err: Exception) -> None:
        """Set the exception of the `#!python QFuture` and mark it as done.

        Parameters:
            err: The exception to set as the exception of the `#!python QFuture`.
        """
        self._done = True
        self._exception = err

    def result(self) -> Any:
        """Get the result of the `#!python QFuture`.

        Returns:
            The result of the `#!python QFuture`.

        Raises:
            FutureCancelled: This `#!python QFuture` instance has been cancelled and
                cannot be awaited.
            NoResults: The result is not ready.
        """
        if self._exception is not None:
            raise self._exception
        if self._cancelled:
            raise FutureCancelled(self._cancelled_message)
        if self._result is not None:
            if self._cancelled_message != '':
                print('Connection was lost no result', file=sys.stderr)
                return None
            if self._debug or pykx_qdebug:
                if self._result._unlicensed_getitem(0).py() == True:
                    print((self._result._unlicensed_getitem(1).py()).decode(), file=sys.stderr)
                    raise QError(self._result._unlicensed_getitem(2).py().decode())
                else:
                    return self._result._unlicensed_getitem(1)
            return self._result
        raise NoResults()
    
    def _disconnected(self):
        if object.__getattribute__(self.q_connection, '_loop') is not None:
            self.add_done_callback(
                lambda x: print(f'Connection was lost no result', file=sys.stderr)
            )
        self._result = 0
        self._cancelled_message = ' '
        self._done = True

    def done(self) -> bool:
        """
        Returns:
            `#!python True` if the `#!python QFuture` is done or if it has been cancelled.
        """
        return self._done or self._cancelled

    def cancelled(self) -> bool:
        """
        Returns:
            `#!python True` if the `#!python QFuture` has been cancelled.
        """
        return self._cancelled

    def cancel(self, msg: str = '') -> None:
        """Cancel the `#!python QFuture`.

        Parameters:
            msg: An optional message to append to the end of the
                `#!python pykx.FutureCancelled` exception.
        """
        self._cancelled = True
        self._cancelled_message = msg

    def exception(self) -> None:
        """Get the exception of the `#!python QFuture`.

        Returns:
            The excpetion of the `#!python QFuture` object.
        """
        if self._cancelled:
            return FutureCancelled(self._cancelled_message)
        if not self._done:
            return NoResults()
        return self._exception

    def add_done_callback(self, callback: Callable):
        """Add a callback function to the list of callback functions which will be executed after
           the `#!python QFuture` result is set.

        Parameters:
            callback: A callback function to append to the list of callback functions which will be
                executed after the `#!python QFuture` result is set.

        Note: The callback parameter must accept one parameter.
            When it is executed the callback function will be passed the current instance of this
            class. The callback function is executed after the result of the future is set,
            allowing the use and modification of the result itself.
        """
        self._callbacks.append(callback)

    def remove_done_callback(self, callback: Callable) -> int:
        """Remove a callback from the list of callbacks contained within the class.

        All matching callbacks will be removed.

        Parameters:
            callback: The callback function to be removed from the list of callback functions to
                call.

        Returns:
            The number of functions removed.
        """
        new_callbacks = [c for c in self._callbacks if c != callback]
        removed = len(self._callbacks) - len(new_callbacks)
        self._callbacks = new_callbacks
        return removed

    def get_loop(self):
        """
        Raises:
            PyKXException: QFutures do not rely on an event loop to drive them, and therefore do not
                have one.
        """
        raise PyKXException('QFutures do not rely on an event loop to drive them, '
                            'and therefore do not have one.')

    __iter__ = __await__


class QConnection(Q):
    _ipc_errors = {
        0: 'Authentication error',
        -1: 'Connection error',
        -2: 'Timeout error',
        -3: 'OpenSSL initialization failed',
    }

    # 65536 is the read size the the c/e libs use internally for IPC requests
    _socket_buffer_size = 65536

    def __new__(cls, *args, **kwargs):
        if cls is QConnection:
            if 'tls' in kwargs.keys() and kwargs['tls']:
                return SecureQConnection(*args, **kwargs)
            return SyncQConnection(*args, **kwargs)
        return object.__new__(cls)

    def __init__(self,
                 host: Union[str, bytes] = 'localhost',
                 port: int = None,
                 *args,
                 username: Union[str, bytes] = '',
                 password: Union[str, bytes] = '',
                 timeout: float = 0.0,
                 large_messages: bool = True,
                 tls: bool = False,
                 unix: bool = False,
                 wait: bool = True,
                 lock: Optional[Union[threading_lock, multiprocessing_lock]] = None,
                 no_ctx: bool = False,
                 reconnection_attempts: int = -1,
                 reconnection_delay: float = 0.5,
                 reconnection_function: callable = reconnection_function
    ):
        """Interface with a q process using the q IPC protocol.

        Users are recommended to instantiate an object of
        [pykx.SyncQConnection][pykx.SyncQConnection] instead of using this class
        directly.

        Parameters:
            host: The hostname to connect to.
            port: The port to connect to.
            username: Username for q connection authorization.
            password: Password for q connection authorization.
            timeout: Timeout for blocking socket operations in seconds. If set to 0,
                the socket will be non-blocking.
            large_messages: Flag to enable support for messages >2GB.
            tls: Flag to enable tls.
            unix: Flag to enable Unix domain socket instead of TCP socket. If set to
                `#!python True`, the `#!python host` parameter is ignored. Does not work on Windows.
            wait: Whether the q server should send a response to the query (which this connection
                will wait to receive). Can be overridden on a per-call basis. If `#!python True`,
                Python will wait for the q server to execute the query, and respond with the
                results. If `#!python False`, the q server will respond immediately to every query
                with generic null(`#!q ::`), then execute them at some point in the future.
            no_ctx: Flag to disable the context interface. Disabling the context interface will not
                stop extra q queries being sent, but will disable the extra features around the
                context interface.
            reconnection_attempts: The number of attempts to reconnect to the q server when there is
                a disconnect. A negative value will disable reconnect attempts.
                A value of 0 indicates no limit on reconnect attempts, with each attempt applying
                `#!python reconnection_function`. Positive integers specify the maximum number of
                attempts to reconnect. Hitting the maximum without a reconnect will throw an error.
            reconnection_function: A function to execute on each attempt to reconnect. This function
                must take one parameter that must be a `#!python float` type. When this function is
                executed it will be passed the `#!python reconnection_delay` parameter. The default
                implementation is a function which modifies `#!python reconnection_delay` to
                increase its value exponentially (delay*2).
            reconnection_delay: A `#!python float` for the initial delay between reconnect attempts
                (in seconds). This is passed to the provided `#!python reconnection_function` that
                is executed on reconnect attempt.

        Note: The `#!python username` and `#!python password` parameters are not required.
            The `#!python username` and `#!python password` parameters are only required if the
            q server requires authorization. Refer to
            [ssl documentation](https://code.kx.com/q/kb/ssl/) for more information.

        Note: The `#!python timeout` argument may not always be enforced.
            When making successive queries if one query times out the next query will wait until
            a response has been received from the previous query before starting the timer for its
            own timeout. This can be avoided by using a separate `#!python QConnection` instance
            for each query.

        Note: When querying KX Insights the `#!python no_ctx=True` keyword argument must be used.

        Raises:
            PyKXException: Using both tls and unix is not possible with a QConnection.
        """
        super().__init__()

    def _create_connection_to_server(self):
        object.__setattr__(
            self,
            '_handle',
            _ipc.init_handle(
                self._connection_info['host'],
                self._connection_info['port'],
                self._connection_info['credentials'],
                self._connection_info['unix'],
                self._connection_info['tls'],
                self._connection_info['timeout'],
                self._connection_info['large_messages']
            )
        )
        if not isinstance(self, SecureQConnection):
            object.__setattr__(
                self,
                '_sock',
                socket.fromfd(
                    self._handle,
                    socket.AF_INET,
                    socket.SOCK_STREAM
                )
            )
            self._sock.setblocking(0)
            object.__setattr__(self, '_reader', selectors.DefaultSelector())
            self._reader.register(self._sock, selectors.EVENT_READ, (WeakMethod(self._recv_socket), WeakMethod(self._recv_socket2)))
            object.__setattr__(self, '_writer', selectors.DefaultSelector())
            self._writer.register(self._sock, selectors.EVENT_WRITE, (WeakMethod(self._send_sock), WeakMethod(self._send_sock)))

    def _init(self,
              host: Union[str, bytes] = 'localhost',
              port: int = None,
              *,
              username: Union[str, bytes] = '',
              password: Union[str, bytes] = '',
              timeout: float = 0.0,
              large_messages: bool = True,
              tls: bool = False,
              unix: bool = False,
              wait: bool = True,
              lock: Optional[Union[threading_lock, multiprocessing_lock]] = None,
              no_ctx: bool = False,
              as_server: bool = False,
              conn_gc_time: float = 0.0,
              reconnection_attempts: int = -1,
              reconnection_delay: float = 0.5,
              reconnection_function: callable = reconnection_function
    ):
        credentials = f'{normalize_to_str(username, "Username")}:' \
                      f'{normalize_to_str(password, "Password")}'
        object.__setattr__(self, '_connection_info', {
            'host': host,
            'port': port,
            'username': username,
            'password': password,
            'timeout': timeout,
            'large_messages': large_messages,
            'credentials': credentials,
            'tls': tls,
            'unix': unix,
            'wait': wait,
            'lock': lock,
            'no_ctx': no_ctx,
            'as_server': as_server,
            'conn_gc_time': conn_gc_time,
            'reconnection_attempts': reconnection_attempts,
            'reconnection_delay': reconnection_delay,
            'reconnection_function': reconnection_function
        })
        if system == 'Windows' and unix: # nocov
            raise TypeError('Unix domain sockets cannot be used on Windows')
        if port is None or not isinstance(port, int):
            raise TypeError('IPC port must be provided')
        object.__setattr__(self, '_lock', lock)
        object.__setattr__(self, 'closed', False)
        if isinstance(self, RawQConnection) and as_server:
            server_sock = socket.create_server(("", port), family=socket.AF_INET)
            server_sock.listen()
            object.__setattr__(self, '_sock', server_sock)
            object.__setattr__(self, '_handle', server_sock.fileno())
            object.__setattr__(self, '_finalizer', lambda: server_sock.close())
        else:
            try:
               handle = _ipc.init_handle(host,
                                         port,
                                         credentials,
                                         unix,
                                         tls,
                                         timeout,
                                         large_messages)
            except BaseException as e:
                if isinstance(e, QError):
                    if 'access' == str(e):
                        raise QError('access: Failed to connect to server with invalid username/password')
                raise e
            object.__setattr__(self, '_handle', handle)
            if licensed:
                object.__setattr__(
                    self,
                    '_finalizer',
                    finalize(self, _ipc._close_handle, self._handle)
                )
            else:
                if self._handle <= 0: # nocov
                    raise PyKXException(self._ipc_errors.get(self._handle, 'Unknown IPC error'))
                object.__setattr__(
                    self,
                    '_finalizer',
                    finalize(self, _ipc._close_handle, self._handle)
                )
            if not isinstance(self, SecureQConnection):
                object.__setattr__(self, '_sock', socket.fromfd(self._handle,
                                                                socket.AF_INET,
                                                                socket.SOCK_STREAM))
        if not isinstance(self, SecureQConnection):
            self._sock.setblocking(0)
            object.__setattr__(self, '_reader', selectors.DefaultSelector())
            self._reader.register(self._sock, selectors.EVENT_READ, (WeakMethod(self._recv_socket), WeakMethod(self._recv_socket2)))
            object.__setattr__(self, '_writer', selectors.DefaultSelector())
            self._writer.register(self._sock, selectors.EVENT_WRITE, (WeakMethod(self._send_sock), WeakMethod(self._send_sock)))
        object.__setattr__(self, '_timeouts', 0)
        object.__setattr__(self, '_initialized', True)
        super().__init__()
        if no_ctx:
            object.__setattr__(self, '_q_ctx_keys', q._q_ctx_keys)

    def __repr__(self):
        kwargs = get_default_args(type(self))
        if 'event_loop' in kwargs:
            del kwargs['event_loop']
        for param_name in tuple(kwargs):
            if param_name == 'as_server':
                continue
            if param_name == 'conn_gc_time':
                continue
            arg = self._connection_info[param_name]
            if arg == kwargs[param_name]: # Remove if equal to default value
                del kwargs[param_name]
            elif param_name == 'password': # Prevent the password from being logged
                kwargs[param_name] = '********'
            else:
                kwargs[param_name] = arg
        if 'host' not in kwargs and self._connection_info['host'] != 'localhost':
            kwargs['host'] = self._connection_info['host']
        if 'port' not in kwargs and self._connection_info['port'] is not None:
            kwargs['port'] = self._connection_info['port']
        return (f'pykx.{"Async" if isinstance(self, AsyncQConnection) else ""}'
                f'QConnection({", ".join(f"{k}={v!r}" for k, v in kwargs.items())})')

    @abstractmethod
    def __call__(self,
                 query: Union[str, bytes, CharVector],
                 *args: Any,
                 wait: Optional[bool] = None,
                 debug: bool = False,
    ) -> K:
        pass # nocov

    def _send(self,
              query,
              *params,
              wait: Optional[bool] = None,
              error=False,
              debug=False,
              skip_debug=False
    ):
        if self.closed:
            raise RuntimeError(_ipc_err_warning['closed'])
        tquery = type(query)
        debugging = (not skip_debug) and (debug or pykx_qdebug)
        if issubclass(tquery, SymbolicFunction):
            if licensed:
                query = query.func
                tquery = type(query)
        if not (issubclass(tquery, K) or isinstance(query, (str, bytes))):
            raise ValueError(_ipc_err_warning['cannot_send'] + str(tquery))
        if debugging:
            if not issubclass(tquery, Function):
                query = CharVector(query)
        start_time = monotonic_ns()
        timeout = self._connection_info['timeout']
        while True:
            if timeout > 0 and monotonic_ns() - start_time >= (timeout * 1000000000):
                break
            events = self._writer.select(timeout)
            for key, _mask in events:
                callback = key.data
                if debugging:
                    return callback[0]()(
                        key.fileobj,
                        bytes(CharVector(
                            '{[pykxquery] .Q.trp[{[x] (0b; value x)}; pykxquery;'
                            '{(1b;"backtrace:\n",.Q.sbt y;x)}]}'
                        )),
                        query if len(params) == 0 else List((query, *params)),
                        wait=wait,
                        error=error,
                        debug=debug
                    )
                else:
                    return callback[0]()(key.fileobj, query, *params, wait=wait, error=error, debug=debug)

    def _ipc_query_builder(self, query, *params):
        data = bytes(query, 'utf-8') if isinstance(query, str) else query
        if params:
            data = [data]
            prev_types = [type(data)]
            prev_types.extend([type(x) for x in params])
            data.extend(params)
            data = [K(x) if not isinstance(x, type(None)) else CharVector(x) for x in data]
            for a, b in zip(prev_types, data):

                if not issubclass(a, type(None))\
                   and (isinstance(b, Foreign)
                        or (isinstance(b, Composition) and q('{.pykx.util.isw x}', b))
                   )\
                   and not issubclass(a, Function)\
                   or issubclass(type(b), Function) and\
                        isinstance(b, Composition) and q('{.pykx.util.isw x}', b):
                    raise ValueError(_ipc_err_warning['cannot_send']  + str(type(b)))
        return data

    def _send_sock(self,
                   sock,
                   query,
                   *params,
                   wait: Optional[bool] = None,
                   error=False,
                   debug=False
    ):
        if len(params) > 8:
            raise TypeError('Too many parameters - q queries cannot have more than 8 parameters')
        query = self._ipc_query_builder(query, *params)
        # The second parameter `1 if wait else 0` sets the value of the second byte of the message
        # to 1 if the message should be sent and a result waited for or a 0 if the message is to be
        # considered async and no result is expected to be returned from the q process.
        # Find more on IPC and serialization here:
        # - https://code.kx.com/q/basics/ipc/
        # - https://code.kx.com/q/kb/serialization/
        k_query = K(query)
        msg_view = serialize(k_query, mode=6, wait=2 if error else 1 if wait else 0)
        msg_len = len(msg_view)
        if error:
            msg_view = list(msg_view.copy())
            msg_view[8] = 128
            msg_view = memoryview(bytes(msg_view))
            wait=False
        sent = 0
        while sent < msg_len:
            try:
                sent += sock.send(msg_view[sent:min(msg_len, sent + self._socket_buffer_size)])
            except BlockingIOError: # nocov
                # The only way to get here is if we send too much data to the socket before it
                # can be sent elsewhere, we just need to wait a moment until more data can be
                # sent to the sockets buffer
                pass
            except BaseException as e:  # nocov
                raise RuntimeError(f"Failed to send query on IPC socket: '{e}'")
        if isinstance(self, SyncQConnection) or isinstance(self, RawQConnection):
            return
        if wait:
            q_future = QFuture(self, self._connection_info['timeout'], debug)
            self._call_stack.append(q_future)
            return q_future
        else:
            q_future = QFuture(self, self._connection_info['timeout'], debug)
            q_future.set_result(K(None))
            return q_future

    # flake8: noqa: C901
    async def _recv2(self, locked=False, acceptAsync=False, fut: Optional[QFuture]=None):
        timeout = self._connection_info['timeout']
        while self._timeouts > 0:
            events = self._reader.select(timeout)
            for key, _ in events:
                key.data()(key.fileobj)
                self._timeouts -= 1
        if isinstance(self, RawQConnection):
            if len(self._send_stack) != 0:
                self.poll_send(0)
        start_time = monotonic_ns()
        with self._lock if self._lock is not None and not locked else nullcontext():
            while True:
                if fut is not None and fut.done():
                    return fut.result()
                if (timeout > 0) and monotonic_ns() - start_time >= (timeout * 1000000000):
                    self._timeouts += 1
                    raise QError(_ipc_err_warning['timeout'])
                events = self._reader.select(timeout)
                for key, _ in events:
                    callback = key.data
                    msg_type, res = callback[0]()(key.fileobj)
                    if MessageType.sync_msg.value == msg_type:
                        print("WARN: Discarding unexpected sync message from handle: "
                              + str(self.fileno()), file=sys.stderr)
                        try:
                            self._send(SymbolAtom("PyKX cannot receive queries in client mode"),
                                       error=True)
                        except BaseException:
                            pass
                    elif MessageType.async_msg.value == msg_type and not acceptAsync:
                        print("WARN: Discarding unexpected async message from handle: "
                              + str(self.fileno()), file=sys.stderr)
                    elif MessageType.resp_msg.value == msg_type or \
                            MessageType.async_msg.value == msg_type:
                        return res
                    else:
                        raise RuntimeError('MessageType unknown')
                    return
                await asyncio.sleep(0.0)


    # flake8: noqa: C901
    def _recv(self, locked=False, acceptAsync=False):
        timeout = self._connection_info['timeout']
        while self._timeouts > 0:
            events = self._reader.select(timeout)
            for key, _ in events:
                key.data[0]()(key.fileobj)
                self._timeouts -= 1
        if isinstance(self, RawQConnection):
            if len(self._send_stack) == len(self._call_stack) and len(self._send_stack) != 0:
                self.poll_send()
        start_time = monotonic_ns()
        with self._lock if self._lock is not None and not locked else nullcontext():
            while True:
                if timeout > 0 and monotonic_ns() - start_time >= (timeout * 1000000000):
                    self._timeouts += 1
                    raise QError(_ipc_err_warning['timeout'])
                events = self._reader.select(timeout)
                for key, _ in events:
                    callback = key.data
                    msg_type, res = callback[0]()(key.fileobj)
                    if MessageType.sync_msg.value == msg_type:
                        print("WARN: Discarding unexpected sync message from handle: "
                              + str(self.fileno()), file=sys.stderr)
                        try:
                            self._send(SymbolAtom("PyKX cannot receive queries in client mode"),
                                       error=True)
                        except BaseException:
                            pass
                    elif MessageType.async_msg.value == msg_type and not acceptAsync:
                        print("WARN: Discarding unexpected async message from handle: "
                              + str(self.fileno()), file=sys.stderr)
                    elif MessageType.resp_msg.value == msg_type or \
                            MessageType.async_msg.value == msg_type:
                        return res
                    else:
                        raise RuntimeError('MessageType unknown')

    async def _recv_socket2(self, sock):
        tot_bytes = 0
        chunks = []
        # message header
        a = await self._loop.sock_recv(sock, 8)
        chunks = list(a)
        tot_bytes += 8
        if len(chunks) == 0:
            try:
                if self._connection_info['reconnection_attempts'] == -1:
                    self.close()
            except BaseException:
                self.close()
            raise RuntimeError(_ipc_err_warning['closed'])
        elif len(chunks) <8:
            try:
                if self._connection_info['reconnection_attempts'] == -1:
                    self.close()
            except BaseException:
                self.close()
            raise RuntimeError("PyKX attempted to process a message containing less than "
                               "the expected number of bytes, connection closed."
                               f"\nReturned bytes: {chunks}.\n"
                               "If you have a reproducible use-case please raise an "
                               "issue at https://github.com/kxsystems/pykx/issues with "
                               "the use-case provided.")

        # The last 5 bytes of the header contain the size and the first byte contains information
        # about whether the message is encoded in big-endian or little-endian form
        endianness = chunks[0]
        if endianness == 1: # little-endian
            size = chunks[3]
            for i in range(7, 3, -1):
                size = size << 8
                size += chunks[i]
        else: # nocov
            # big-endian
            size = chunks[3]
            for i in range(4, 8):
                size = size << 8
                size += chunks[i]

        buff = bytearray(size)
        chunks = bytearray(chunks)
        for i in range(8):
            buff[i] = chunks[i]
        view = memoryview(buff)[8:]
        # message body
        while tot_bytes < size:
            try:
                to_read = min(self._socket_buffer_size, size - tot_bytes)
                read, _ = await self._loop.sock_recvfrom_into(sock, view, to_read)
                view = view[read:]
                tot_bytes += read
            except BlockingIOError: # nocov
                # The only way to get here is if we start processing a message before all the data
                # has been received by the socket
                pass
        res = chunks[1], self._create_result(buff)
        return res

    def _recv_socket(self, sock):
        tot_bytes = 0
        chunks = []
        # message header
        a = sock.recv(8)
        chunks = list(a)
        tot_bytes += 8
        if len(chunks) == 0:
            try:
                if self._connection_info['reconnection_attempts'] == -1:
                    self.close()
            except BaseException:
                self.close()
            raise RuntimeError(_ipc_err_warning['closed'])
        elif len(chunks) <8:
            try:
                if self._connection_info['reconnection_attempts'] == -1:
                    self.close()
            except BaseException:
                self.close()
            raise RuntimeError("PyKX attempted to process a message containing less than "
                               "the expected number of bytes, connection closed."
                               f"\nReturned bytes: {chunks}.\n"
                               "If you have a reproducible use-case please raise an "
                               "issue at https://github.com/kxsystems/pykx/issues with "
                               "the use-case provided.")

        # The last 5 bytes of the header contain the size and the first byte contains information
        # about whether the message is encoded in big-endian or little-endian form
        endianness = chunks[0]
        if endianness == 1: # little-endian
            size = chunks[3]
            for i in range(7, 3, -1):
                size = size << 8
                size += chunks[i]
        else: # nocov
            # big-endian
            size = chunks[3]
            for i in range(4, 8):
                size = size << 8
                size += chunks[i]

        buff = bytearray(size)
        chunks = bytearray(chunks)
        for i in range(8):
            buff[i] = chunks[i]
        view = memoryview(buff)[8:]
        # message body
        while tot_bytes < size:
            try:
                to_read = min(self._socket_buffer_size, size - tot_bytes)
                read = sock.recv_into(view, to_read)
                view = view[read:]
                tot_bytes += read
            except BlockingIOError: # nocov
                # The only way to get here is if we start processing a message before all the data
                # has been received by the socket
                pass
        res = chunks[1], self._create_result(buff)
        return res

    def _create_error(self, buff):
        try:
            err_msg = [chr(x) for x in buff[9:-1]]
            if len(err_msg) > max_error_length:
                err_msg = err_msg[:max_error_length]
            return QError(''.join(err_msg))
        except BaseException:
            return QError('An unknown exception occured.')

    def _create_result(self, buff):
        if isinstance(self, SyncQConnection) or\
           (isinstance(self, RawQConnection) and len(self._call_stack) == 0):
            # buff[8] contains the responses type and 128 is the type for an error response
            if int(buff[2]) == 0 and int(buff[8]) == 128:
                raise self._create_error(buff)
            else:
                return deserialize(memoryview(buff).obj)
        if int(buff[2]) == 0 and int(buff[8]) == 128:
            q_future = self._call_stack.pop(0)
            q_future.set_exception(self._create_error(buff))
        else:
            q_future = self._call_stack.pop(0)
            q_future.set_result(deserialize(memoryview(buff).obj))

    def upd(self, table: str, data:Union[list, Table]) -> None:
        """
        Execute `#!q .u.upd` on a remote q process. This function assumes the definition of
            `#!q .u.upd` on the remote q process takes the same count and data type of arguments
            as the default implementation (q keyword `#!q insert`). The `#!python data` argument
            will be converted to a list if it is a PyKX `#!python Table`.

        Parameters:
            table: The name of the global variable on the q process to update.
            data: The contents of the update.

        Returns:
            On successful execution this function will return None

        Example:

        Successfully execute `#!q .u.upd` on connected process

        ```python
        >>> import pykx as kx
        >>> with kx.SyncQConnection(port=5050) as q:
        ...     q.upd('trade', [kx.TimespanAtom('now') 'AAPL', 1.0])
        >>> trades = kx.Table(data = {
        ...     'time': kx.TimespanAtom('now'),
        ...     'sym': kx.random.random(N, ['AAPL', 'MSFT', 'GOOG']),
        ...     'price': kx.random.random(N, 10.0)})
        >>> with kx.SyncQConnection(port=5050) as q:
        ...     q.upd('trade', trades)
        ```
        """
        if isinstance(data, Table):
            data = data._values
        try:
            self(b'.u.upd', table, data)
        except QError as err:
            if '.u.upd' in str(err):
                raise QError("Update function '.u.upd' not defined on connected process")
            raise err

    def file_execute(
        self,
        file_path: str,
        *,
        return_all: bool = False,
    ):
        """Functionality for the execution of the content of a local file on a remote server

        Parameters:
            file_path: Path to the file which is to be executed on a remote server
            return_all: Return the execution result from all lines within the executed script

        Raises:
            PyKXException: Will raise error associated with failure to execute the code on
                the server associated with the given file.

        Examples:

        Connect to a q process on localhost and execute a file based on relative path.

        ```python
        conn = pykx.QConnection('localhost', 5000)
        conn.file_execute('file.q')
        ```

        Connect to a q process using an asynchronous QConnection at IP address 127.0.0.1,
            on port 5000 and execute a file based on absolute path.

        ```python
        conn = pykx.QConnection('127.0.0.1', 5000, wait=False)
        conn.file_execute('/User/path/to/file.q')
        ```
        """
        wlist = ['k', 'q', 'p', 'py']
        with open(pykx_lib_dir/'q.k', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'pykxld:' in line:
                    ld = line[7:].encode()
        if isinstance(file_path, str):
            path_stem = Path(file_path).suffix[1:]
            if not path_stem in wlist:
                raise QError(f"Provided file type '{path_stem}' unsupported")
        with open(file_path) as f:
            lines = f.readlines()
        lines = [CharVector(i.rstrip('\n')) for i in lines]
        return self("""
                    {[fn;code;file;stem]
                        $[any stem~/:("p";"py");
                            $[`pykx in key `;
                              .pykx.pyexec "\n" sv code;
                              '"PyKX must be loaded on remote server"];
                            value (@';last file;enlist[file],/:value[\"k)\",fn]code)
                            ]
                        }
                    """,
                    ld,
                    lines,
                    bytes(file_path, 'utf-8'),
                    bytes(path_stem, 'utf-8'),
                    wait=return_all)

    def fileno(self) -> int:
        return self._handle


class SyncQConnection(QConnection):
    def __new__(cls, *args, **kwargs):
        if 'tls' in kwargs.keys() and kwargs['tls']:
            return SecureQConnection(*args, **kwargs)
        return object.__new__(cls)

    def __init__(self,
                 host: Union[str, bytes] = 'localhost',
                 port: int = None,
                 *args,
                 username: Union[str, bytes] = '',
                 password: Union[str, bytes] = '',
                 timeout: float = 0.0,
                 large_messages: bool = True,
                 tls: bool = False,
                 unix: bool = False,
                 wait: bool = True,
                 lock: Optional[Union[threading_lock, multiprocessing_lock]] = None,
                 no_ctx: bool = False,
                 reconnection_attempts: int = -1,
                 reconnection_delay: float = 0.5,
                 reconnection_function: callable = reconnection_function
    ):
        """Interface with a q process using the q IPC protocol.

        Instances of this class represent an open connection to a q process, which can be sent
        messages synchronously or asynchronously by calling it as a function.

        Parameters:
            host: The host name to which a connection is to be established.
            port: The port to which a connection is to be established.
            username: Username for q connection authorization.
            password: Password for q connection authorization.
            timeout: Timeout for blocking socket operations in seconds. If set to 0, the socket
                will be non-blocking.
            large_messages: Whether support for messages >2GB should be enabled.
            tls: Whether TLS should be used.
            unix: Whether a Unix domain socket should be used instead of TCP. If set to
                `#!python True`, the host parameter is ignored. Does not work on Windows.
            wait: Whether the q server should send a response to the query (which this connection
                will wait to receive). Can be overridden on a per-call basis. If `#!python True`,
                Python will wait for the q server to execute the query, and respond with
                the results. If `#!python False`, the q server will respond immediately to every
                query with generic null(`#!q ::`), then execute them at some point in the future.
            no_ctx: This parameter determines whether or not the context interface will be disabled.
                disabling the context interface will stop extra q queries being sent but will
                disable the extra features around the context interface.
            reconnection_attempts: This parameter specifies how many attempts will be made to
                reconnect to the server if the connection is lost. The query will be resent if the
                reconnection is successful. The default is -1 which will not attempt to
                reconnect, 0 will continuously attempt to reconnect to the server using the backoff
                `#!python reconnection_function`. Any positive integer will specify the maximum
                number of tries to reconnect before throwing an error if a connection can not
                be made.
            reconnection_delay: This parameter outlines the initial delay between reconnection
                attempts, by default this is set to 0.5 seconds and is passed to the function
                defined by the `#!python reconnection_function` parameter which takes this delay as
                it's only parameter.
            reconnection_function: This parameter defines the function which is used to modify the
                `#!python reconnection_delay` on successive attempts to reconnect to the server. By
                default this is an exponential backoff where the `#!python reconnection_delay` is
                multiplied by two on each invocation.

        Note: The `#!python username` and `#!python password` parameters are not required.
            The `#!python username` and `#!python password` parameters are only required if the
            q server requires authorization. Refer to
            [ssl documentation](https://code.kx.com/q/kb/ssl/) for more information.

        Note: The `#!python timeout` argument may not always be enforced when making successive
            queries. When making successive queries if one query times out the next query will
            wait until a response has been received from the previous query before starting the
            timer for its own timeout. This can be avoided by using a separate
            `#!python SyncQConnection` instance for each query.

        Note: When querying KX Insights the `#!python no_ctx=True` keyword argument must be used.

        Raises:
            PyKXException: Using both tls and unix is not possible with a QConnection.

        Examples:

        Connect to a q process on localhost with a required username and password.

        ```python
        pykx.SyncQConnection('localhost', 5001, 'username', 'password')
        ```

        Connect to a q process at IP address 127.0.0.0, on port 5000 with a timeout of 2 seconds
        and TLS enabled.

        ```python
        pykx.SyncQConnection('127.0.0.1', 5001, timeout=2.0, tls=True)
        ```

        Connect to a q process via a Unix domain socket on port 5001

        ```python
        pykx.SyncQConnection(port=5001, unix=True)
        ```

        Automatically reconnect to a q server after a disconnect.

        ```python
        >>> conn = kx.SyncQConnection(port=5001, reconnection_attempts=0)
        >>> conn('til 10')
        pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
        # server connection is lost here
        >>> conn('til 10')
        WARNING: Connection lost attempting to reconnect.
        Failed to reconnect, trying again in 0.5 seconds.
        Connection successfully reestablished.
        pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
        ```
        """
        self._init(host,
                   port,
                   *args,
                   username=username,
                   password=password,
                   timeout=timeout,
                   large_messages=large_messages,
                   tls=tls,
                   unix=unix,
                   wait=wait,
                   lock=lock,
                   no_ctx=no_ctx,
                   reconnection_attempts=reconnection_attempts,
                   reconnection_delay=reconnection_delay,
                   reconnection_function=reconnection_function
        )
        super().__init__()

    def __call__(self,
                 query: Union[str, bytes, CharVector],
                 *args: Any,
                 wait: Optional[bool] = None,
                 debug: bool = False,
                 skip_debug: bool = False,
    ) -> K:
        """Evaluate a query on the connected q process over IPC.

        Parameters:
            query: A q expression to be evaluated.
            *args: Arguments to the q query. Each argument will be converted into a
                `#!python pykx.K` object. Up to 8 arguments can be provided, as that is the
                maximum supported by q.
            wait: Whether the q server should execute the query before responding.
                If `#!python True`, Python will wait for the q server to execute the query, and
                respond with the results.
                If `#!python False`, the q server will respond immediately to the query with
                generic null (`#!q ::`), then execute them at some point in the future. Defaults to
                whatever the `#!python wait` keyword argument was for the `#!python SyncQConnection`
                instance (i.e. this keyword argument overrides the instance-level default).


        Raises:
            RuntimeError: A closed IPC connection was used.
            QError: Query timed out, may be raised if the time taken to make or receive a query goes
                over the timeout limit.
            TypeError: Too many arguments were provided - q queries cannot have more than 8
                parameters.
            ValueError: Attempted to send a Python function over IPC.

        Examples:

        ```python
        q = pykx.SyncQConnection(host='localhost', port=5002)
        ```

        Call an anonymous function with 2 parameters

        ```python
        q('{y+til x}', 10, 5)
        ```

        Execute a q query with no parameters

        ```python
        q('til 10')
        ```

        Call an anonymous function with 3 parameters and don't wait for a response

        ```python
        q('{x set y+til z}', 'async_query', 10, 5, wait=False)
        ```

        Call an anonymous function with 3 parameters and don't wait for a response by default

        ```python
        q = pykx.SyncQConnection(host='localhost', port=5002, wait=False)
        # Because `wait=False`, all calls on this q instance are not responded to by default:
        q('{x set y+til z}', 'async_query', 10, 5)
        # But we can issue calls and wait for results by overriding the `wait` option on a per-call
        # basis:
        q('{x set y+til z}', 'async_query', 10, 5, wait=True)
        ```

        Call a PyKX Operator function with supplied parameters

        ```python
        q(kx.q.sum, [1, 2, 3])
        ```

        Call a PyKX Keyword function with supplied paramters

        ```python
        q(kx.q.floor, [5.2, 10.4])
        ```
        """
        if wait is None:
            wait = self._connection_info['wait']
        with self._lock if self._lock is not None else nullcontext():
            return self._call(query, *args, wait=wait, debug=debug, skip_debug=skip_debug)

    def _call(self,
              query: Union[str, bytes],
              *args: Any,
              wait: Optional[bool] = None,
              debug: bool = False,
              skip_debug: bool = False,
    ) -> K:
        try:
            self._send(query, *args, wait=wait, debug=debug, skip_debug=skip_debug)
            if not wait:
                return K(None)
            res = self._recv(locked=True)
            if skip_debug or not (debug or pykx_qdebug):
                return res
            if res._unlicensed_getitem(0).py() == True:
                print((res._unlicensed_getitem(1).py()).decode(), file=sys.stderr)
                raise QError(res._unlicensed_getitem(2).py().decode())
            else:
                return res._unlicensed_getitem(1)
        except BaseException as e:
            if isinstance(e, QError):
                raise e
            if self._connection_info['reconnection_attempts'] != -1:
                print(_ipc_err_warning['reconnect_warn'], file=sys.stderr)
                loops = self._connection_info['reconnection_attempts']
                reconnection_delay = self._connection_info['reconnection_delay']
                reconnection_function = self._connection_info['reconnection_function']
                while True:
                    try:
                        self._create_connection_to_server()
                    except BaseException as err:
                        # attempts = 0 is infinite attempts as it will go to -1 before the check
                        # to break
                        loops -= 1
                        if loops == 0:
                            print(
                                'WARNING: Could not reconnect to server within '
                                f'{self._connection_info["reconnection_attempts"]} attempts.',
                                file=sys.stderr
                            )
                            raise err
                        print(
                            f'Failed to reconnect, trying again in {reconnection_delay} seconds.',
                            file=sys.stderr
                        )
                        if not isinstance(reconnection_delay, (int, float)):
                            raise TypeError(
                                _ipc_err_warning['delay_type']
                            )
                        sleep(reconnection_delay)
                        reconnection_delay = reconnection_function(reconnection_delay)
                        continue
                    print(_ipc_err_warning['reconnected'], file=sys.stderr)
                    return self._call(query, *args, wait=wait, debug=debug)
            else:
                raise e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Close the connection.

        Examples:

        Open and subsequently close a connection to a q process on localhost:

        ```python
        q = pykx.SyncQConnection('localhost', 5001)
        q.close()
        ```

        Using this class with a with-statement should be preferred:

        ```python
        with pykx.SyncQConnection('localhost', 5001) as q:
            # do stuff with q
            pass
        # q is closed automatically
        ```
        """
        if not self.closed:
            object.__setattr__(self, 'closed', True)
            self._reader.unregister(self._sock)
            self._writer.unregister(self._sock)
            self._reader.close()
            self._writer.close()
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
                self._finalizer()
            except BaseException:
                pass

    def fileno(self) -> int:
        """The file descriptor or handle of the connection."""
        return super().fileno()


class AsyncQConnection(QConnection):
    def __init__(self,
                 host: Union[str, bytes] = 'localhost',
                 port: int = None,
                 *args,
                 username: Union[str, bytes] = '',
                 password: Union[str, bytes] = '',
                 timeout: float = 0.0,
                 large_messages: bool = True,
                 tls: bool = False,
                 unix: bool = False,
                 wait: bool = True,
                 lock: Optional[Union[threading_lock, multiprocessing_lock]] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None,
                 no_ctx: bool = False,
                 reconnection_attempts: int = -1,
                 reconnection_delay: float = 0.5,
                 reconnection_function: callable = reconnection_function
    ):
        """Interface with a q process using the q IPC protocol.

        Instances of this class represent an open connection to a q process, which can be sent
        messages asynchronously by calling it as a function.

        Parameters:
            host: The host name to which a connection is to be established.
            port: The port to which a connection is to be established.
            username: Username for q connection authorization.
            password: Password for q connection authorization.
            timeout: Timeout is not supported when using `AsyncQConnection` objects.
            large_messages: Whether support for messages >2GB should be enabled.
            tls: Whether TLS should be used.
            unix: Whether a Unix domain socket should be used instead of TCP. If set to
                `#!python True`, the host parameter is ignored. Does not work on Windows.
            wait: Whether the q server should send a response to the query (which this connection
                will wait to receive). Can be overridden on a per-call basis. If `#!python True`,
                Python will wait for the q server to execute the query, and respond with
                the results. If `#!python False`, the q server will respond immediately to every
                query with generic null (`::`), then execute them at some point in the future.
            event_loop: If running an event loop that supports the `#!python create_task()`
                method then you can provide the event loop here and the returned future object will
                be an instance of the loops future type. This will allow the current event loop
                to manage awaiting `#!python QFuture` objects as well as any other async tasks that
                may be running. If no event loop is provided the default result of
                `aysncio.get_event_loop()` will be used.
            no_ctx: This parameter determines whether or not the context interface will be disabled.
                disabling the context interface will stop extra q queries being sent but will
                disable the extra features around the context interface.
            reconnection_attempts: This parameter specifies how many attempts will be made to
                reconnect to the server if the connection is lost. The query will be resent if the
                reconnection is successful. The default is -1 which will not attempt to reconnect, 0
                will continuously attempt to reconnect to the server using the backoff
                `#!python reconnection_function`. Any positive integer will specify the maximum
                number of tries to reconnect before throwing an error if a connection
                cannot be made.
            reconnection_delay: This parameter outlines the initial delay between reconnection
                attempts, by default this is set to 0.5 seconds and is passed to the function
                defined by the `#!python reconnection_function` parameter which takes this delay
                as it's only parameter.
            reconnection_function: This parameter defines the function which is used to modify the 
                `#!python reconnection_delay` on successive attempts to reconnect to the server. By 
                default this is an exponential backoff where the `#!python reconnection_delay` is
                multiplied by two on each invocation

        Note: The `#!python username` and `#!python password` parameters are not required.
            The `#!python username` and `#!python password` parameters are only required if
            the q server requires authorization. Refer to
            [ssl documentation](https://code.kx.com/q/kb/ssl/) for more information.

        Note: When querying KX Insights the `#!python no_ctx=True` keyword argument must be used.

        Warning: AsyncQConnections will not resend queries that have not completed on reconnection.
            When using the `#!python reconnection_attempts` key word argument any queries that were
            not complete before the connection was lost will have to be manually sent again after
            the automatic reconnection.

        Raises:
            PyKXException: Using both tls and unix is not possible with a QConnection.

        Examples:

        Connect to a q process on localhost with a required username and password.

        ```python
        await pykx.AsyncQConnection('localhost', 5001, 'username', 'password')
        ```

        Connect to a q process at IP address 127.0.0.0, on port 5000 with a timeout of 2 seconds
        and TLS enabled.

        ```python
        await pykx.AsyncQConnection('127.0.0.1', 5001, timeout=2.0, tls=True)
        ```

        Connect to a q process via a Unix domain socket on port 5001

        ```python
        await pykx.AsyncQConnection(port=5001, unix=True)
        ```

        Automatically reconnect to a q server after a disconnect.

        ```python
        async def main():
            conn = await kx.AsyncQConnection(
                port=5001,
                event_loop=asyncio.get_event_loop(),
                reconnection_attempts=0
            )
            print(await conn('til 10'))
            # Connection lost here
            # All unfinished futures are cancelled on connection loss
            print(await conn('til 10')) # First call only causes a reconnection but wont send the query and returns none
            print(await conn('til 10')) # Second one completes

            print(await conn('til 10'))
        asyncio.run(main())

        # Outputs
        0 1 2 3 4 5 6 7 8 9
        WARNING: Connection lost attempting to reconnect.
        Connection successfully reestablished.
        Connection was lost no result
        None
        0 1 2 3 4 5 6 7 8 9
        ```
        """
        if timeout > 0.0:
            warnings.warn('Timeout is not supported when using AsyncQConnection objects.')
        # TODO: Remove this once TLS support is fixed
        if tls:
            raise PyKXException('TLS is currently only supported for SyncQConnections')
        loop = event_loop
        if loop is None:
            # `asyncio.get_event_loop()` used to do this automatically but it is deprecated as of
            #  Python 3.12, so we do this weird try - except to future proof.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
        object.__setattr__(self, '_stored_args', {
            'host': host,
            'port': port,
            'args': args,
            'username': username,
            'password': password,
            'timeout': 0.0,
            'large_messages': large_messages,
            'tls': tls,
            'unix': unix,
            'wait': wait,
            'lock': lock,
            'loop': loop,
            'no_ctx': no_ctx,
            'reconnection_attempts':reconnection_attempts,
            'reconnection_delay': reconnection_delay,
            'reconnection_function': reconnection_function,
        })
        object.__setattr__(self, '_initialized', False)

    async def _async_init(self,
                          host: Union[str, bytes] = 'localhost',
                          port: int = None,
                          *args,
                          username: Union[str, bytes] = '',
                          password: Union[str, bytes] = '',
                          timeout: float = 0.0,
                          large_messages: bool = True,
                          tls: bool = False,
                          unix: bool = False,
                          wait: bool = True,
                          lock: Optional[Union[threading_lock, multiprocessing_lock]] = None,
                          event_loop: Optional[asyncio.AbstractEventLoop] = None,
                          no_ctx: bool = False,
                          reconnection_attempts: int = -1,
                          reconnection_delay: float = 0.5,
                          reconnection_function: callable = reconnection_function,
    ):
        object.__setattr__(self, '_call_stack', [])
        self._init(host,
                   port,
                   *args,
                   username=username,
                   password=password,
                   timeout=timeout,
                   large_messages=large_messages,
                   tls=tls,
                   unix=unix,
                   wait=wait,
                   lock=lock,
                   no_ctx=no_ctx,
                   reconnection_attempts=reconnection_attempts,
                   reconnection_delay=reconnection_delay,
                   reconnection_function=reconnection_function,
        )
        object.__setattr__(self, '_loop', event_loop)
        con_info = object.__getattribute__(self, '_connection_info')
        con_info['event_loop'] = None
        object.__setattr__(self, '_connection_info', con_info)
        super().__init__()

    async def _initobj(self): # nocov
        """Crutch used for `__await__` after spawning."""
        if not self._initialized:
            await self._async_init(
                self._stored_args['host'],
                self._stored_args['port'],
                *self._stored_args['args'],
                username=self._stored_args['username'],
                password=self._stored_args['password'],
                timeout=self._stored_args['timeout'],
                large_messages=self._stored_args['large_messages'],
                tls=self._stored_args['tls'],
                unix=self._stored_args['unix'],
                wait=self._stored_args['wait'],
                lock=self._stored_args['lock'],
                event_loop=self._stored_args['loop'],
                no_ctx=self._stored_args['no_ctx'],
                reconnection_attempts=self._stored_args['reconnection_attempts'],
                reconnection_delay=self._stored_args['reconnection_delay'],
                reconnection_function=self._stored_args['reconnection_function'],
            )
        return self

    def _cancel_all_futures(self):
        [x._disconnected() for x in self._call_stack]
        self._call_stack = []

    def __await__(self):
        return self._initobj().__await__()

    def __call__(self,
                 query: Union[str, bytes, CharVector],
                 *args: Any,
                 wait: bool = True,
                 reuse: bool = True,
                 debug: bool = False,
                 async_response: bool = False,
    ) -> QFuture:
        """Evaluate a query on the connected q process over IPC.

        Parameters:
            query: A q expression to be evaluated.
            *args: Arguments to the q query. Each argument will be converted into a
                `#!python pykx.K` object. Up to 8 arguments can be provided, as that is
                the maximum supported by q.
            wait: Whether the q server should execute the query before responding.
                If `#!python True`, Python will wait for the q server to execute the query,
                and respond with the results. If `#!python False`, the q server will respond
                immediately to the query with generic null (`#!q ::`), then execute them at some
                point in the future. Defaults to whatever the `#!python wait` keyword argument
                was for the `ASyncQConnection` instance (i.e. this keyword argument overrides the
                instance-level default).
            reuse: Whether the AsyncQConnection instance should be reused for subsequent queries,
                if using q queries that respond in a deferred/asynchronous manner this should be set
                to `#!python False` so the query can be made in a dedicated
                `#!python AsyncQConnection` instance.
            async_response: When using `reuse=False` and `wait=False` if an asynchronous response is
                expected you can use this argument to keep the connection alive until an
                asynchronous response has been received. Awaiting the inital returned future object
                will return a second future that you can await upon to recieve the asynchronous
                response.

        Returns:
            A QFuture object that can be awaited on to get the result of the query.

        Raises:
            RuntimeError: A closed IPC connection was used.
            QError: Query timed out, may be raised if the time taken to make or receive a query goes
                over the timeout limit.
            TypeError: Too many arguments were provided - q queries cannot have more than 8
                parameters.
            ValueError: Attempted to send a Python function over IPC.

        Examples:

        ```python
        q = await pykx.AsyncQConnection(host='localhost', port=5002)
        ```

        Call an anonymous function with 2 parameters

        ```python
        await q('{y+til x}', 10, 5)
        ```

        Execute a q query with no parameters

        ```python
        await q('til 10')
        ```

        Call an anonymous function with 3 parameters and don't wait for a response

        ```python
        await q('{x set y+til z}', 'async_query', 10, 5, wait=False)
        ```

        Call an anonymous function with 3 parameters and don't wait for a response by default

        ```python
        q = await pykx.AsyncQConnection(host='localhost', port=5002, wait=False)
        # Because `wait=False`, all calls on this q instance are not responded to by default:
        await q('{x set y+til z}', 'async_query', 10, 5)
        # But we can issue calls and wait for results by overriding the `wait` option on a per-call
        # basis:
        await q('{x set y+til z}', 'async_query', 10, 5, wait=True)
        ```

        Call a PyKX Operator function with supplied parameters
      
        ```python
        await q(kx.q.sum, [1, 2, 3])
        ```

        Call a PyKX Keyword function with supplied paramters
        
        ```python
        await q(kx.q.floor, [5.2, 10.4])
        ```
        """
        if async_response and reuse:
            warnings.warn('Cannot use async_response=True without reuse=False.')
        if async_response and wait:
            warnings.warn('Cannot use async_response=True without wait=False.')
        if not reuse:
            conn = _DeferredQConnection(self._stored_args['host'],
                                        self._stored_args['port'],
                                        *self._stored_args['args'],
                                        username=self._stored_args['username'],
                                        password=self._stored_args['password'],
                                        timeout=self._stored_args['timeout'],
                                        large_messages=self._stored_args['large_messages'],
                                        tls=self._stored_args['tls'],
                                        unix=self._stored_args['unix'],
                                        wait=self._stored_args['wait'],
                                        no_ctx=self._stored_args['no_ctx'])
            q_future = conn(query, *args, wait=wait, debug=debug)
            if async_response and not wait:
                q_future2 = QFuture(conn, conn._connection_info['timeout'], debug)
                conn._call_stack.append(q_future2)
                q_future2 = self._loop.create_task(q_future2.__async_await__())
                q_future2.add_done_callback(lambda x: conn.close())
                q_future.set_result(q_future2)
                return q_future
            q_future.add_done_callback(lambda x: conn.close())
            if self._loop is None:
                return q_future
            return self._loop.create_task(q_future.__async_await__())
        else:
            if not self._initialized:
                raise UninitializedConnection()
            try:
                with self._lock if self._lock is not None else nullcontext():
                    q_future = self._send(query, *args, wait=wait, debug=debug)
                    if self._loop is None:
                        return q_future
                    return self._loop.create_task(q_future.__async_await__())
            except BaseException as e:
                if isinstance(e, QError):
                    raise e
                if self._connection_info['reconnection_attempts'] != -1:
                    self._cancel_all_futures()
                    print(_ipc_err_warning['reconnect_warn'], file=sys.stderr)
                    loops = self._connection_info['reconnection_attempts']
                    reconnection_delay = self._connection_info['reconnection_delay']
                    reconnection_function = self._connection_info['reconnection_function']
                    while True:
                        try:
                            self._create_connection_to_server()
                        except BaseException as err:
                            # attempts = 0 is infinite attempts as it will go to -1 before the check
                            # to break
                            loops -= 1
                            if loops == 0:
                                print(
                                    'WARNING: Could not reconnect to server within '
                                    f'{self._connection_info["reconnection_attempts"]} attempts.',
                                    file=sys.stderr
                                )
                                raise err
                            print(
                                f'Failed to reconnect, trying again in {reconnection_delay} seconds.',
                                file=sys.stderr
                            )
                            if not isinstance(reconnection_delay, (int, float)):
                                raise TypeError(
                                    _ipc_err_warning['delay_type']
                                )
                            sleep(reconnection_delay)
                            reconnection_delay = reconnection_function(reconnection_delay)
                            continue
                        print(_ipc_err_warning['reconnected'], file=sys.stderr)
                        break

                    q_future = QFuture(self, self._connection_info['timeout'], debug)
                    q_future.set_result(K(None))
                    return q_future
                else:
                    raise e

    def _call(self,
              query: Union[str, bytes],
              *args: Any,
              wait: Optional[bool] = None,
              debug: bool = False,
              skip_debug: bool = False
    ):
        try:
            with self._lock if self._lock is not None else nullcontext():
                return self._send(query, *args, wait=wait, debug=debug)._await()
        except BaseException as e:
            if isinstance(e, QError):
                raise e
            if self._connection_info['reconnection_attempts'] != -1:
                self._cancel_all_futures()
                print(_ipc_err_warning['reconnect_warn'], file=sys.stderr)
                loops = self._connection_info['reconnection_attempts']
                reconnection_delay = self._connection_info['reconnection_delay']
                reconnection_function = self._connection_info['reconnection_function']
                while True:
                    try:
                        self._create_connection_to_server()
                    except BaseException as err:
                        # attempts = 0 is infinite attempts as it will go to -1 before the check
                        # to break
                        loops -= 1
                        if loops == 0:
                            print(
                                'WARNING: Could not reconnect to server within '
                                f'{self._connection_info["reconnection_attempts"]} attempts.',
                                file=sys.stderr
                            )
                            raise err
                        print(
                            f'Failed to reconnect, trying again in {reconnection_delay} seconds.',
                            file=sys.stderr
                        )
                        if not isinstance(reconnection_delay, (int, float)):
                            raise TypeError(
                                _ipc_err_warning['delay_type']
                            )
                        sleep(reconnection_delay)
                        reconnection_delay = reconnection_function(reconnection_delay)
                        continue
                    print(_ipc_err_warning['reconnected'], file=sys.stderr)
                    break
            else:
                raise e

    async def __aenter__(self):
        return await self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close the connection.

        Examples:

        Open and subsequently close a connection to a q process on localhost:

        ```python
        q = await pykx.AsyncQConnection('localhost', 5001)
        await q.close()
        ```

        Using this class with a with-statement should be preferred:

        ```python
        async with pykx.AsyncQConnection('localhost', 5001) as q:
            # do stuff with q
            pass
        # q is closed automatically
        ```
        """
        if not self._initialized:
            raise UninitializedConnection()
        if not self.closed:
            while self._call_stack != []:
                events = self._reader.select()
                for key, _mask in events:
                    callback = key.data
                    callback[0]()(key.fileobj)
            object.__setattr__(self, 'closed', True)
            self._reader.unregister(self._sock)
            self._writer.unregister(self._sock)
            self._reader.close()
            self._writer.close()
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
                self._finalizer()
            except BaseException:
                pass

    def fileno(self) -> int:
        """The file descriptor or handle of the connection."""
        if not self._initialized:
            raise UninitializedConnection()
        return self._handle


class _DeferredQConnection(QConnection):
    def __init__(self,
                 host: Union[str, bytes] = 'localhost',
                 port: int = None,
                 *args,
                 username: Union[str, bytes] = '',
                 password: Union[str, bytes] = '',
                 timeout: float = 0.0,
                 large_messages: bool = True,
                 tls: bool = False,
                 unix: bool = False,
                 wait: bool = True,
                 no_ctx: bool = False
    ):
        object.__setattr__(self, '_call_stack', [])
        self._init(host,
                   port,
                   *args,
                   username=username,
                   password=password,
                   timeout=timeout,
                   large_messages=large_messages,
                   tls=tls,
                   unix=unix,
                   wait=wait,
                   no_ctx=no_ctx,
        )
        super().__init__()

    def __call__(self,
                 query: Union[str, bytes, CharVector],
                 *args: Any,
                 wait: Optional[bool] = None,
                 debug: bool = False,
    ) -> K:
        return self._send(query, *args, wait=wait, debug=debug)

    def _call(self,
              query: Union[str, bytes],
              *args: Any,
              wait: Optional[bool] = None,
              debug: bool = False,
              skip_debug: bool = False
    ):
        return self._send(query, *args, wait=wait, debug=debug)._await()

    def close(self) -> None:
        if not self.closed: # nocov
            while self._call_stack != []:
                events = self._reader.select()
                for key, _mask in events:
                    callback = key.data
                    callback[0]()(key.fileobj)
            object.__setattr__(self, 'closed', True)
            self._reader.unregister(self._sock)
            self._writer.unregister(self._sock)
            self._reader.close()
            self._writer.close()
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
                self._finalizer()
            except BaseException:
                pass


def handshake(conn: socket.socket):
    val = []
    last = None
    while last != b'\x00':
        last = conn.recv(1)
        val.append(last)
    try:
        if str(q.z.pw) != '::':
            login = (b''.join(val[:-2])).decode()
            if ':' in login:
                login = login.split(':')
                user = login[0]
                password = login[1]
                if q.z.pw(user, password):
                    conn.send(val[-2])
                    return int(str(val[-2])[-2])
                else:
                    return -1
            else:
                if q.z.pw(login, None):
                    conn.send(val[-2])
                    return int(str(val[-2])[-2])
                else:
                    return -1
        else:
            conn.send(val[-2])
            return int(str(val[-2])[-2])
    except BaseException:
        return -1


class RawQConnection(QConnection):
    open_cons = []

    def __init__(self,
                 host: Union[str, bytes] = 'localhost',
                 port: int = None,
                 *args,
                 username: Union[str, bytes] = '',
                 password: Union[str, bytes] = '',
                 timeout: float = 0.0,
                 large_messages: bool = True,
                 tls: bool = False,
                 unix: bool = False,
                 wait: bool = True,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None,
                 no_ctx: bool = False,
                 as_server: bool = False,
                 conn_gc_time: float = 0.0,
    ):
        """Interface with a q process using the q IPC protocol.

        Instances of this class represent an open connection to a q process, which can be sent
        messages asynchronously by calling it as a function, the send and receive selector queues
        can also, be polled directly using this class.

        Parameters:
            host: The host name to which a connection is to be established.
            port: The port to which a connection is to be established.
            username: Username for q connection authorization.
            password: Password for q connection authorization.
            timeout: Timeout is not supported when using `RawQConnection` objects.
            large_messages: Whether support for messages >2GB should be enabled.
            tls: Whether TLS should be used.
            unix: Whether a Unix domain socket should be used instead of TCP. If set to
                `#!python True`, the host parameter is ignored. Does not work on Windows.
            wait: Whether the q server should send a response to the query (which this connection
                will wait to receive). Can be overridden on a per-call basis. If `#!python True`,
                Python will wait for the q server to execute the query, and respond with the
                results. If `#!python False`, the q server will respond immediately to every query
                with generic null (`#!q ::`), then execute them at some point in the future.
            event_loop: If running an event loop that supports the `#!python create_task()`
                method then you can provide the event loop here and the returned future object will
                be an instance of the loops future type. This will allow the current event loop
                to manage awaiting `#!python QFuture` objects as well as any other async tasks that
                may be running. If no event loop is provided the default result of
                `aysncio.get_event_loop()` will be used.
            no_ctx: This parameter determines whether or not the context interface will be disabled.
                disabling the context interface will stop extra q queries being sent but will
                disable the extra features around the context interface.
            as_server: If this parameter is set to True the QConnection will act as a q server,
                that other processes can connect to, and will not create a connection. this
                functionality is licensed only.
            conn_gc_time: When running as a server this will determine the number of seconds between
                going through the list of opened connections and closing any that the clients have
                closed. If not set the default of 0.0 will cause any old connections to never be
                closed unless `#!python self.clean_open_connections()` is manually called.

        Note: The `#!python username` and `#!python password` parameters are not required.
            The `#!python username` and `#!python password` parameters are only required if the q
            server requires authorization. Refer to
            [ssl documentation](https://code.kx.com/q/kb/ssl/) for more information.

        Note: The overhead of calling `#!python clean_open_connections` is large.
            When running as a server you should ensure that `#!python clean_open_connections` is
            called fairly infrequently as the overhead of clearing all the dead connections can be
            quite large. It is recommended to have a large delay on successive clears or manage it
            manually.

        Note: When querying KX Insights the `#!python no_ctx=True` keyword argument must be used.

        Note: 3.1 Upgrade considerations
            As of PyKX version 3.1 all QFuture objects returned from calls to `RawQConnection`
            objects must be awaited to recieve their results. Previously you could use just
            `conn.poll_recv()` and then directly get the result with `future.result()`.

        Raises:
            PyKXException: Using both tls and unix is not possible with a QConnection.

        Examples:

        Connect to a q process on localhost with a required username and password.

        ```python
        await pykx.RawQConnection('localhost', 5001, 'username', 'password')
        ```

        Connect to a q process at IP address 127.0.0.0, on port 5000 with a timeout of 2 seconds
        and TLS enabled.

        ```python
        await pykx.RawQConnection('127.0.0.1', 5001, timeout=2.0, tls=True)
        ```

        Connect to a q process via a Unix domain socket on port 5001

        ```python
        await pykx.RawQConnection(port=5001, unix=True)
        ```
        """
        if timeout > 0.0:
            warnings.warn('Timeout is not supported when using AsyncQConnection objects.')
        # TODO: Remove this once TLS support is fixed
        if tls:
            raise PyKXException('TLS is currently only supported for SyncQConnections')
        loop = event_loop
        if loop is None:
            # `asyncio.get_event_loop()` used to do this automatically but it is deprecated as of
            #  Python 3.12, so we do this weird try - except to future proof.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
        object.__setattr__(self, '_stored_args', {
            'host': host,
            'port': port,
            'args': args,
            'username': username,
            'password': password,
            'timeout': 0.0,
            'large_messages': large_messages,
            'tls': tls,
            'unix': unix,
            'wait': wait,
            'loop': loop,
            'no_ctx': True if as_server else no_ctx,
            'as_server': as_server,
            'conn_gc_time': conn_gc_time,
        })
        object.__setattr__(self, '_initialized', False)

    async def _async_init(self,
                          host: Union[str, bytes] = 'localhost',
                          port: int = None,
                          *args,
                          username: Union[str, bytes] = '',
                          password: Union[str, bytes] = '',
                          timeout: float = 0.0,
                          large_messages: bool = True,
                          tls: bool = False,
                          unix: bool = False,
                          wait: bool = True,
                          event_loop: Optional[asyncio.AbstractEventLoop] = None,
                          no_ctx: bool = False,
                          as_server: bool = False,
                          conn_gc_time: float = 0.0,
    ):
        object.__setattr__(self, '_call_stack', [])
        object.__setattr__(self, '_send_stack', [])
        self._init(host,
                   port,
                   *args,
                   username=username,
                   password=password,
                   timeout=timeout,
                   large_messages=large_messages,
                   tls=tls,
                   unix=unix,
                   wait=wait,
                   no_ctx=no_ctx,
                   as_server=as_server,
                   conn_gc_time=conn_gc_time,
        )
        object.__setattr__(self, '_loop', event_loop)
        con_info = object.__getattribute__(self, '_connection_info')
        con_info['event_loop'] = None
        object.__setattr__(self, '_connection_info', con_info)
        super().__init__()

    async def _initobj(self): # nocov
        """Crutch used for `__await__` after spawning."""
        if not self._initialized:
            await self._async_init(self._stored_args['host'],
                                   self._stored_args['port'],
                                   *self._stored_args['args'],
                                   username=self._stored_args['username'],
                                   password=self._stored_args['password'],
                                   timeout=self._stored_args['timeout'],
                                   large_messages=self._stored_args['large_messages'],
                                   tls=self._stored_args['tls'],
                                   unix=self._stored_args['unix'],
                                   wait=self._stored_args['wait'],
                                   event_loop=self._stored_args['loop'],
                                   no_ctx=self._stored_args['no_ctx'],
                                   as_server=self._stored_args['as_server'],
                                   conn_gc_time=self._stored_args['conn_gc_time'],
                                   )
        return self

    def __await__(self):
        return self._initobj().__await__()

    def __call__(self,
                 query: Union[str, bytes, CharVector],
                 *args: Any,
                 wait: bool = True,
                 debug: bool = False,
    ) -> QFuture:
        """Evaluate a query on the connected q process over IPC.

        Parameters:
            query: A q expression to be evaluated.
            *args: Arguments to the q query. Each argument will be converted into a
                `#!python pykx.K` object. Up to 8 arguments can be provided, as that is the maximum
                supported by q.
            wait: Whether the q server should execute the query before responding. If
                `#!python True`, Python will wait for the q server to execute the query, and respond
                with the results. If `#!python False`, the q server will respond immediately to the
                query with generic null (`#!q ::`), then execute them at some point in the future.
                Defaults to whatever the `#!python wait` keyword argument was for the
                `#!python ASyncQConnection` instance (i.e. this keyword argument overrides the
                instance-level default).

        Returns:
            A QFuture object that can be awaited on to get the result of the query.

        Raises:
            RuntimeError: A closed IPC connection was used.
            QError: Query timed out, may be raised if the time taken to make or receive a query goes
                over the timeout limit.
            TypeError: Too many arguments were provided - q queries cannot have more than 8
                parameters.
            ValueError: Attempted to send a Python function over IPC.

        Note: Queries are not sent until a response has been awaited or the send queue is polled.

        Note: When querying KX Insights the `#!python no_ctx=True` keyword argument must be used.

        Examples:

        ```python
        q = await pykx.RawQConnection(host='localhost', port=5002)
        ```

        Call an anonymous function with 2 parameters

        ```python
        await q('{y+til x}', 10, 5)
        ```

        Execute a q query with no parameters

        ```python
        await q('til 10')
        ```

        Call an anonymous function with 3 parameters and don't wait for a response

        ```python
        await q('{x set y+til z}', 'async_query', 10, 5, wait=False)
        ```

        Call an anonymous function with 3 parameters and don't wait for a response by default

        ```python
        q = await pykx.RawQConnection(host='localhost', port=5002, wait=False)
        # Because `wait=False`, all calls on this q instance are not responded to by default:
        await q('{x set y+til z}', 'async_query', 10, 5)
        # But we can issue calls and wait for results by overriding the `wait` option on a per-call
        # basis:
        await q('{x set y+til z}', 'async_query', 10, 5, wait=True)
        ```
        """
        if not self._initialized:
            raise UninitializedConnection()
        fut = QFuture(self, self._connection_info['timeout'], debug)
        res = self._loop.create_task(
            fut.__async_await__()
        )
        self._send_stack.append({'query': query, 'args': args, 'wait': wait, 'debug': debug})
        self._call_stack.append(fut)
        return res

    def _call(self,
              query: Union[str, bytes],
              *args: Any,
              wait: Optional[bool] = None,
              debug: bool = False,
              skip_debug: bool = False,
    ):
        conn = _DeferredQConnection(self._stored_args['host'],
                                    self._stored_args['port'],
                                    *self._stored_args['args'],
                                    username=self._stored_args['username'],
                                    password=self._stored_args['password'],
                                    timeout=self._stored_args['timeout'],
                                    large_messages=self._stored_args['large_messages'],
                                    tls=self._stored_args['tls'],
                                    unix=self._stored_args['unix'],
                                    wait=self._stored_args['wait'],
                                    no_ctx=self._stored_args['no_ctx'])
        q_future = conn(query, *args, wait=wait, debug=debug)
        q_future.add_done_callback(lambda x: conn.close())
        return q_future._await()

    def poll_send(self, amount: int = 1):
        """Send queued queries to the process connected to over IPC.

        Parameters:
            amount: The number of send requests to handle, defaults to one, if 0 is used then all
                currently waiting queries will be sent.

        Raises:
            QError: Query timed out, may be raised if the time taken to make or receive a query goes
                over the timeout limit.

        Examples:

        ```python
        q = await pykx.RawQConnection(host='localhost', port=5002)
        ```

        Send a single queued message.

        ```python
        q_fut = q('til 10') # not sent yet
        q.poll_send() # 1 message is sent
        ```

        Send two queued messages.

        ```python
        q_fut = q('til 10') # not sent yet
        q_fut2 = q('til 10') # not sent yet
        q.poll_send(2) # 2 messages are sent
        ```

        Send all queued messages.

        ```python
        q_fut = q('til 10') # not sent yet
        q_fut2 = q('til 10') # not sent yet
        q.poll_send(0) # 2 messages are sent
        ```
        """
        count = amount
        if count == 0:
            count = len(self._send_stack)
        while count > 0:
            if not self._initialized:
                raise UninitializedConnection()
            if len(self._send_stack) == 0:
                return
            to_send = self._send_stack.pop(0)
            self._send(to_send['query'], *to_send['args'], wait=to_send['wait'], debug=to_send['debug'])
            count -= 1

    def _serialize_response(self, response, level):
        error = isinstance(response, tuple)
        if error:
            response = response[1]
        try:
            msg = serialize(response, mode=level, wait=2)
        except QError as e:
            error = True
            response = SymbolAtom(f"{e}")
            msg = serialize(response, mode=level, wait=2)
        if error:
            msg_view = list(msg.copy())
            msg_view[8] = 128
            return memoryview(bytes(msg_view))
        return msg

    def _send_sock_server(self, sock, response, level):
        try:
            msg_view = self._serialize_response(response, level)
            msg_len = len(msg_view)
            sent = 0
            while sent < msg_len:
                try:
                    sent += sock.send(msg_view[sent:min(msg_len, sent + self._socket_buffer_size)])
                except BlockingIOError: # nocov
                    # The only way to get here is if we send too much data to the socket before it
                    # can be sent elsewhere, we just need to wait a moment until more data can be
                    # sent to the sockets buffer
                    pass
                except BaseException as e:  # nocov
                    raise RuntimeError(f"Failed to send query on IPC socket: '{e}'")
        except BaseException:  # nocov
            pass

    def _recv_socket_server(self, sock): # noqa
        tot_bytes = 0
        chunks = []
        # message header
        try:
            a = sock.recv(8)
            chunks = list(a)
            tot_bytes += 8
            if len(chunks) == 0:
                return
            elif len(chunks) <8:
                self.close()
                raise RuntimeError("PyKX attempted to process a message containing less than "
                                   "the expected minimum number of bytes, connection closed."
                                   f"\nReturned bytes: {chunks}.\n"
                                   "If you have a reproducible use-case please raise an "
                                   "issue at https://github.com/kxsystems/pykx/issues with "
                                   "the use-case provided.")

            # The last 5 bytes of the header contain the size and the first byte contains
            # information about whether the message is encoded in big-endian or little-endian form
            endianness = chunks[0]
            if endianness == 1: # little-endian
                size = chunks[3]
                for i in range(7, 3, -1):
                    size = size << 8
                    size += chunks[i]
            else: # nocov
                # big-endian
                size = chunks[3]
                for i in range(4, 8):
                    size = size << 8
                    size += chunks[i]

            buff = bytearray(size)
            chunks = bytearray(chunks)
            for i in range(8):
                buff[i] = chunks[i]
            view = memoryview(buff)[8:]
            # message body
            while tot_bytes < size:
                try:
                    to_read = min(self._socket_buffer_size, size - tot_bytes)
                    read = sock.recv_into(view, to_read)
                    view = view[read:]
                    tot_bytes += read
                except BlockingIOError: # nocov
                    # The only way to get here is if we send too much data to the socket before it
                    # can be sent elsewhere, we just need to wait a moment until more data can be
                    # sent to the sockets buffer
                    pass

            return chunks[1], deserialize(memoryview(buff).obj)
        except ConnectionResetError:
            pass

    def _poll_server(self, amount: int = 1): # noqa
        # This is gross and hacky but the ctx interface has to be disabled when running as a
        # server, so we need to do some weird things to access class members
        count = amount
        timeout = self._connection_info['timeout']
        try:
            self._stored_args["last_gc"]
        except KeyError:
            self._stored_args["last_gc"] = 0.0
        for (reader, writer, _, level) in self.open_cons:
            events = reader.select(timeout)
            for key, _ in events:
                callback = key.data
                res = callback[0]()(key.fileobj)
                if res is None:
                    count -= 1
                    if count > 1:
                        return
                    continue
                msg_type, res = res
                wevents = writer.select(timeout)
                for key, _ in wevents:
                    callback = key.data
                    try:
                        if MessageType.sync_msg.value == msg_type:
                            handler = q.z.pg
                        elif MessageType.async_msg.value == msg_type:
                            handler = q.z.ps
                        elif MessageType.resp_msg.value == msg_type:
                            raise RuntimeError('MessageType.resp_msg not supported')
                        else:
                            raise RuntimeError('MessageType unknown')
                        if isinstance(handler, Composition) and q('{.pykx.util.isw x}', handler):
                            # if handler was overriden to use a python func we must enlist the
                            # query or it will be passed through as CharAtom's
                            res = q('enlist', res)
                        res = handler(res)
                    except QError as e:
                        if MessageType.sync_msg.value == msg_type:
                            res = (True, SymbolAtom(f"{e}"))
                        elif MessageType.async_msg.value == msg_type:
                            print(e)
                    if MessageType.sync_msg.value == msg_type:
                        callback[0]()(key.fileobj, res, level)
                    count -= 1
                    if count > 1:
                        return
        if (self._stored_args["conn_gc_time"] > 0.0
            and monotonic_ns() / 1000000000 - self._stored_args["last_gc"]
            > self._stored_args["conn_gc_time"]
        ):
            self._stored_args["last_gc"] = monotonic_ns() / 1000000000
            self.clean_open_connections()
        try:
            conn, _ = self._sock.accept()
            conn.setblocking(0)
            level = handshake(conn)
            if level != -1:
                reader = selectors.DefaultSelector()
                reader.register(conn, selectors.EVENT_READ, WeakMethod(self._recv_socket_server))
                writer = selectors.DefaultSelector()
                writer.register(conn, selectors.EVENT_WRITE, WeakMethod(self._send_sock_server))
                self.open_cons.append((reader, writer, conn, level))
        except BaseException:
            pass

    def clean_open_connections(self):
        for i in range(len(self.open_cons) - 1, -1, -1):
            try:
                msg = serialize(q('::'), mode=self.open_cons[i][3], wait=0)
                self.open_cons[i][2].send(msg.copy())
            except BaseException:
                self.open_cons[i][0].unregister(self.open_cons[i][2])
                self.open_cons[i][1].unregister(self.open_cons[i][2])
                self.open_cons[i][2].close()
                self.open_cons.pop(i)

    def poll_recv_async(self):
        """Asynchronously receive a query from the process connected to over IPC.

        Raises:
            QError: Query timed out, may be raised if the time taken to make or receive a query goes
                over the timeout limit.

        Examples:

        Receive a single queued message.

        ```python
        q = await pykx.RawQConnection(host='localhost', port=5002)
        q_fut = q('til 10') # not sent yet
        q.poll_send() # message is sent
        await q.poll_recv_async() # message response is received
        ```
        """
        q_future = QFuture(self, self._connection_info['timeout'], False, poll_recv=True)
        if self._loop is not None:
            return self._loop.create_task(q_future.__async_await__())
        return q_future

    async def poll_recv2(self, amount: int = 1, fut: Optional[QFuture] = None):
        """Recieve queries from the process connected to over IPC.

        Parameters:
            amount: The number of receive requests to handle, defaults to one, if 0 is used then
                all currently waiting responses will be received.

        Raises:
            QError: Query timed out, may be raised if the time taken to make or receive a query goes
                over the timeout limit.

        Examples:

        ```python
        q = await pykx.RawQConnection(host='localhost', port=5002)
        ```

        Receive a single queued message.

        ```python
        q_fut = q('til 10') # not sent yet
        q.poll_send() # message is sent
        q.poll_recv() # message response is received
        ```

        Receive two queued messages.

        ```python
        q_fut = q('til 10') # not sent yet
        q_fut2 = q('til 10') # not sent yet
        q.poll_send(2) # messages are sent
        q.poll_recv(2) # message responses are received
        ```

        Receive all queued messages.

        ```python
        q_fut = q('til 10') # not sent yet
        q_fut2 = q('til 10') # not sent yet
        q.poll_send(0) # all messages are sent
        q.poll_recv(0) # all message responses are received
        ```
        """
        if fut is not None and fut.done():
            return fut.result()
        count = amount
        timeout = self._connection_info['timeout']
        if self._stored_args['as_server']:
            self._poll_server(amount)
        else:
            last = None
            if count == 0:
                count = len(self._call_stack) if len(self._call_stack) > 0 else 1
            while count >= 0:
                if fut is not None and fut.done():
                    return fut.result()
                start_time = monotonic_ns()
                with self._lock if self._lock is not None else nullcontext():
                    if (timeout > 0.0) and monotonic_ns() - start_time >= (timeout * 1000000000):
                        self._timeouts += 1
                        raise QError(_ipc_err_warning['timeout'])
                    events = self._reader.select(timeout)
                    if len(events) != 0:
                        for key, _ in events:
                            callback = key.data
                            res = callback[0]()(key.fileobj)
                            res = res[1] if isinstance(res, tuple) else res
                            if count == 1:
                                return res
                            count -= 1
                            last = res
                    else:
                        if count == 1:
                            return last
                        count -= 1
            return last

    def poll_recv(self, amount: int = 1):
        """Recieve queries from the process connected to over IPC.

        Parameters:
            amount: The number of receive requests to handle, defaults to one, if 0 is used then
                all currently waiting responses will be received.

        Raises:
            QError: Query timed out, may be raised if the time taken to make or receive a query goes
                over the timeout limit.

        Examples:

        ```python
        q = await pykx.RawQConnection(host='localhost', port=5002)
        ```

        Receive a single queued message.

        ```python
        q_fut = q('til 10') # not sent yet
        q.poll_send() # message is sent
        q.poll_recv() # message response is received
        ```

        Receive two queued messages.

        ```python
        q_fut = q('til 10') # not sent yet
        q_fut2 = q('til 10') # not sent yet
        q.poll_send(2) # messages are sent
        q.poll_recv(2) # message responses are received
        ```

        Receive all queued messages.

        ```python
        q_fut = q('til 10') # not sent yet
        q_fut2 = q('til 10') # not sent yet
        q.poll_send(0) # all messages are sent
        q.poll_recv(0) # all message responses are received
        ```
        """
        count = amount
        timeout = self._connection_info['timeout']
        if self._stored_args['as_server']:
            self._poll_server(amount)
        else:
            last = None
            if count == 0:
                count = len(self._call_stack) if len(self._call_stack) > 0 else 1
            while count >= 0:
                start_time = monotonic_ns()
                with self._lock if self._lock is not None else nullcontext():
                    if timeout > 0.0 and monotonic_ns() - start_time >= (timeout * 1000000000):
                        self._timeouts += 1
                        raise QError(_ipc_err_warning['timeout'])
                    events = self._reader.select(timeout)
                    if len(events) != 0:
                        for key, _ in events:
                            callback = key.data
                            res = callback[0]()(key.fileobj)
                            res = res[1] if isinstance(res, tuple) else res
                            if count == 1:
                                return res
                            count -= 1
                            last = res
                    else:
                        if count == 1:
                            return last
                        count -= 1
            return last

    async def __aenter__(self):
        return await self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close the connection.

        Examples:

        Open and subsequently close a connection to a q process on localhost:

        ```python
        q = await pykx.RawQConnection('localhost', 5001)
        await q.close()
        ```

        Using this class with a with-statement should be preferred:

        ```python
        async with pykx.RawQConnection('localhost', 5001) as q:
            # do stuff with q
            pass
        # q is closed automatically
        ```
        """
        if not self._initialized:
            raise UninitializedConnection()
        if not self.closed:
            object.__setattr__(self, 'closed', True)
            while self._call_stack != []:
                events = self._reader.select()
                for key, _mask in events:
                    callback = key.data
                    callback[0](key.fileobj)
            self._reader.unregister(self._sock)
            self._writer.unregister(self._sock)
            self._reader.close()
            self._writer.close()
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
                self._finalizer()
            except BaseException:
                pass


class SecureQConnection(QConnection):
    def __init__(self,
                 host: Union[str, bytes] = 'localhost',
                 port: int = None,
                 *args,
                 username: Union[str, bytes] = '',
                 password: Union[str, bytes] = '',
                 timeout: float = 0.0,
                 large_messages: bool = True,
                 tls: bool = False,
                 unix: bool = False,
                 wait: bool = True,
                 lock: Optional[Union[threading_lock, multiprocessing_lock]] = None,
                 no_ctx: bool = False,
                 reconnection_attempts: int = -1,
                 reconnection_delay: float = 0.5,
                 reconnection_function: callable = reconnection_function,
    ):
        """Interface with a q process using the q IPC protocol.

        Instances of this class represent an open connection to a q process, which can be sent
        messages synchronously or asynchronously by calling it as a function. This class is
        automatically created when using TLS to encrypt your queries.

        Parameters:
            host: The host name to which a connection is to be established.
            port: The port to which a connection is to be established.
            username: Username for q connection authorization.
            password: Password for q connection authorization.
            timeout: Timeout for blocking socket operations in seconds. If set to 0, the socket
                will be non-blocking.
            large_messages: Whether support for messages >2GB should be enabled.
            tls: Whether TLS should be used.
            unix: Whether a Unix domain socket should be used instead of TCP. If set to
                `#!python True`, the host parameter is ignored. Does not work on Windows.
            wait: Whether the q server should send a response to the query (which this connection
                will wait to receive). Can be overridden on a per-call basis. If `#!python True`,
                Python will wait for the q server to execute the query, and respond with the
                results. If `#!python False`, the q server will respond immediately to every query
                with generic null (`#!q ::`), then execute them at some point in the future.
            no_ctx: This parameter determines whether or not the context interface will be disabled.
                disabling the context interface will stop extra q queries being sent but will
                disable the extra features around the context interface.
            reconnection_attempts: This parameter specifies how many attempts will be made to
                reconnect to the server if the connection is lost. The query will be resent if the
                reconnection is successful. The default is -1 which will not attempt to reconnect, 0
                will continuously attempt to reconnect to the server using the backoff
                `#!python reconnection_function`. Any positive integer will specify the maximum
                number of tries to reconnect before throwing an error if a connection can not be made.
            reconnection_delay: This parameter outlines the initial delay between reconnection
                attempts, by default this is set to 0.5 seconds and is passed to the function
                defined by the `#!python reconnection_function` parameter which takes this delay
                as it's only parameter
            reconnection_function: This parameter defines the function which is used to modify the
                `#!python reconnection_delay` on successive attempts to reconnect to the server. By
                default this is an exponential backoff where the `#!python reconnection_delay` is
                multiplied by two on each invocation

        Note: The `#!python username` and `#!python password` parameters are not required.
            The `#!python username` and `#!python password` parameters are only required if
            the q server requires authorization. Refer to
            [ssl documentation](https://code.kx.com/q/kb/ssl/) for more information.

        Note: The `#!python timeout` argument may not always be enforced when making successive
            queries. When making successive queries if one query times out the next query will wait
            until a response has been received from the previous query before starting the timer for
            its own timeout. This can be avoided by using a separate `#!python SecureQConnection`
            instance for each query.

        Note: When querying KX Insights the `#!python no_ctx=True` keyword argument must be used.

        Raises:
            PyKXException: Using both tls and unix is not possible with a QConnection.

        Examples:

        Connect to a q process at IP address 127.0.0.0, on port 5000 with a timeout of 2 seconds
        and TLS enabled.

        ```python
        pykx.SecureQConnection('127.0.0.1', 5001, timeout=2.0, tls=True)
        ```
        """
        self._init(host,
                   port,
                   *args,
                   username=username,
                   password=password,
                   timeout=timeout,
                   large_messages=large_messages,
                   tls=tls,
                   unix=unix,
                   wait=wait,
                   lock=lock,
                   no_ctx=no_ctx,
                   reconnection_attempts=reconnection_attempts,
                   reconnection_delay=reconnection_delay,
                   reconnection_function=reconnection_function,
        )
        super().__init__()

    @staticmethod
    def _licensed_call(handle: int, query: bytes, parameters: List, wait: bool) -> K:
        ret = q(f'{{{handle} x}}', [query, *parameters] if parameters else query)
        if not wait:
            q(f'{handle}[]')
        return ret

    # TODO: can we switch over to exclusively using this approach instead of `_licensed_call`?
    # It would involve making `cls._lib` be either libq or libe depending on if we're licensed.
    @classmethod
    def _unlicensed_call(cls, handle: int, query, parameters: List, wait: bool) -> K:
        return _ipc._unlicensed_call(handle, query, parameters, wait)

    def __call__(self,
                 query: Union[str, bytes, CharVector, K],
                 *args: Any,
                 wait: Optional[bool] = None,
                 debug: bool = False,
    ) -> K:
        """Evaluate a query on the connected q process over IPC.

        Parameters:
            query: A q expression to be evaluated.
            *args: Arguments to the q query. Each argument will be converted into a
                `#!python pykx.K` object. Up to 8 arguments can be provided, as that is the maximum
                supported by q.
            wait: Whether the q server should execute the query before responding. If
                `#!python True`, Python will wait for the q server to execute the query, and
                respond with the results. If `#!python False`, the q server will respond immediately
                to the query with generic null (`#!q ::`), then execute them at some point in the
                future. Defaults to whatever the `wait` keyword argument was for the
                `#!python SecureQConnection` instance (i.e. this keyword argument overrides the
                instance-level default).

        Raises:
            RuntimeError: A closed IPC connection was used.
            QError: Query timed out, may be raised if the time taken to make or receive a query goes
                over the timeout limit.
            TypeError: Too many arguments were provided - q queries cannot have more than 8
                parameters.
            ValueError: Attempted to send a Python function over IPC.

        Examples:

        ```python
        q = pykx.SecureQConnection(host='localhost', port=5002, tls=True)
        ```

        Call an anonymous function with 2 parameters

        ```python
        q('{y+til x}', 10, 5)
        ```

        Execute a q query with no parameters

        ```python
        q('til 10')
        ```

        Call an anonymous function with 3 parameters and don't wait for a response

        ```python
        q('{x set y+til z}', 'async_query', 10, 5, wait=False)
        ```

        Call an anonymous function with 3 parameters and don't wait for a response by default

        ```python
        q = pykx.SecureQConnection(host='localhost', port=5002, wait=False, tls=True)
        # Because `wait=False`, all calls on this q instance are not responded to by default:
        q('{x set y+til z}', 'async_query', 10, 5)
        # But we can issue calls and wait for results by overriding the `wait` option on a per-call
        # basis:
        q('{x set y+til z}', 'async_query', 10, 5, wait=True)
        ```

        Call a PyKX Operator function with supplied parameters
      
        ```python
        q(kx.q.sum, [1, 2, 3])
        ```

        Call a PyKX Keyword function with supplied paramters
        
        ```python
        q(kx.q.floor, [5.2, 10.4])
        ```

        Automatically reconnect to a q server after a disconnect.

        ```python
        >>> conn = kx.SecureQConnection(port=5001, reconnection_attempts=0)
        >>> conn('til 10')
        pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
        >>> conn('til 10')
        WARNING: Connection lost attempting to reconnect.
        Failed to reconnect, trying again in 0.5 seconds.
        Connection successfully reestablished.
        pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
        ```
        """
        return self._call(query, *args, wait=wait, debug=debug)

    def _call(self,
              query: Union[K, str, bytes],
              *args: Any,
              wait: Optional[bool] = None,
              debug: bool = False,
              skip_debug: bool = False
    ) -> K:
        if wait is None:
            wait = self._connection_info['wait']
        if self.closed:
            raise RuntimeError(_ipc_err_warning['closed'])
        tquery = type(query)
        if not (issubclass(tquery, K) or isinstance(query, (str, bytes))):
            raise ValueError(_ipc_err_warning['cannot_send']  + str(tquery))
        if not issubclass(tquery, Function):
            if isinstance(query, str):
                query = query.encode()
        if len(args) > 8:
            raise TypeError('Too many parameters - q queries cannot have more than 8 parameters')
        prev_types = [type(x) for x in args]
        handle = self._handle if wait else -self._handle
        args = [K(x) for x in args]
        handler = self._licensed_call if licensed else self._unlicensed_call

        try:
            with self._lock if self._lock is not None else nullcontext():
                if debug or pykx_qdebug:
                    res = handler(
                        handle, 
                        normalize_to_bytes(
                            '{[pykxquery] .Q.trp[{[x] (0b; value x)}; pykxquery;'
                            '{(1b; "backtrace:\n",.Q.sbt y; x)}]}',
                            'Query'
                        ),
                        [K(query)] if len(args) == 0 else [List((K(query), *args))],
                        wait,
                    )
                    if res._unlicensed_getitem(0).py() == True:
                        print((res._unlicensed_getitem(1).py()).decode(), file=sys.stderr)
                        raise QError(res._unlicensed_getitem(2).py().decode())
                    else:
                        return res._unlicensed_getitem(1)
                return handler(handle, query, args, wait)
        except BaseException as e:
            if isinstance(e, QError) and 'snd handle' not in str(e) and 'write to handle' not in str(e) and 'close handle' not in str(e):
                raise e
            if self._connection_info['reconnection_attempts'] != -1:
                print(_ipc_err_warning['reconnect_warn'], file=sys.stderr)
                loops = self._connection_info['reconnection_attempts']
                reconnection_delay = self._connection_info['reconnection_delay']
                reconnection_function = self._connection_info['reconnection_function']
                while True:
                    try:
                        self._create_connection_to_server()
                        if not licensed and self._handle == -1:
                            raise ConnectionError('Could not connect to q server')
                    except BaseException as err:
                        # attempts = 0 is infinite attempts as it will go to -1 before the check
                        # to break
                        loops -= 1
                        if loops == 0:
                            print(
                                'WARNING: Could not reconnect to server within '
                                f'{self._connection_info["reconnection_attempts"]} attempts.',
                                file=sys.stderr
                            )
                            raise err
                        print(
                            f'Failed to reconnect, trying again in {reconnection_delay} seconds.',
                            file=sys.stderr
                        )
                        if not isinstance(reconnection_delay, (int, float)):
                            raise TypeError(
                                _ipc_err_warning['delay_type']
                            )
                        sleep(reconnection_delay)
                        reconnection_delay = reconnection_function(reconnection_delay)
                        continue
                    print(_ipc_err_warning['reconnected'], file=sys.stderr)
                    return self._call(query, *args, wait=wait, debug=debug)
            else:
                raise e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Close the connection.

        Examples:

        Open and subsequently close a connection to a q process on localhost:

        ```python
        q = pykx.SecureQConnection('localhost', 5001, tls=True)
        q.close()
        ```

        Using this class with a with-statement should be preferred:

        ```python
        with pykx.SecureQConnection('localhost', 5001, tls=True) as q:
            # do stuff with q
            pass
        # q is closed automatically
        ```
        """
        if not self.closed:
            object.__setattr__(self, 'closed', True)
            self._finalizer()

    def fileno(self) -> int:
        """The file descriptor or handle of the connection."""
        return super().fileno()

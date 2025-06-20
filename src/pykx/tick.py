"""
Functionality for the generation and management of streaming infrastructures using PyKX.
Fully described [here](../user-guide/advanced/streaming/index.md), this allows users to ingest,
persist and query vast amounts of real-time and historical data in a unified data-format.
"""

from .exceptions import QError
from .util import start_q_subprocess
from .ipc import SyncQConnection
from . import wrappers as k

import inspect
import os
import time
from typing import Callable, Union

import dill

__all__ = [
    'TICK',
    'RTP',
    'HDB',
    'GATEWAY',
    'BASIC'
]


def _init(_q):
    global q
    q = _q


def __dir__():
    return __all__


class STREAMING:
    """
    The `STREAMING` class acts as a base parent class for the TICK, RTP, HDB and GATEWAY
    class objects. Each of these child classes inherit and may modify the logic of this parent.
    In all cases the functions [`libraries`](#pykx.tick.STREAMING.libraries) and
    [`register_api`](#pykx.tick.STREAMING.register_api) for example have the same definition
    and are available to all process types.

    Unless provided with a separate definition as is the case for `start` in all class types
    a user should assume that the logic used for use of `register_api` is consistent across
    process types.
    """
    def __init__(self,
                 port: int = 5010,
                 *,
                 process_logs: Union[str, bool] = False,
                 libraries: dict = None,
                 apis: dict = None,
                 init_args=None
                 ) -> None:
        self._port = port
        self._libraries = libraries
        self._apis = apis
        self._init_args = init_args
        self._init_config = None
        self._connection = None
        try:
            if not self._initialized:
                self._initalized=False
        except BaseException:
            self._initialized=False
        self.server = start_q_subprocess(self._port, load_file='pykx.q', init_args=init_args)
        if self.server is None:
            try:
                self.server = start_q_subprocess(self._port,
                                                 load_file='pykx.q',
                                                 init_args=init_args)
            except BaseException as err:
                raise QError(f'Unable to initialize q process with error {str(err.value)}')
        try:
            connection = SyncQConnection(port=port)
            self._connection = connection
            self._process_logs = process_logs
            if isinstance(process_logs, str):
                connection('{system"1 ",string[x];system"2 ",string x}', process_logs)
            if libraries is not None:
                self.libraries(libraries)
            if isinstance(apis, dict):
                for key, value in apis.items():
                    self.register_api(key, value)
        except BaseException as err:
            self.stop()
            raise err

    def __call__(self, *args) -> k.K:
        """
        Execute a synchronous call against the connected process

        Parameters:
            *args: Pass supplied arguments to the `pykx.SyncQConnection`
                object

        Returns:
            The result of the executed call on the connection object

        Example:

        ```python
        >>> import pykx as kx
        >>> tick = kx.tick.TICK(port=5030)
        >>> tick('1+1').py()
        2
        ```
        """
        return self._connection(*args)

    def start(self,
              config: dict = None,
              print_init: bool = True,
              custom_start: str = '') -> None:
        """
        Start/initialise processing of messages on the associated sub-process.
        This allows users to split the process initialisation from processing
        of data to allow additional configuration/setup to be completed before
        messages begin to be processed.

        Parameters:
            config: A dictionary passed to the sub-process which is used by
                the function `.tick.init` when the process is started, the
                supported parameters for this function will be different
                depending on process type.
            print_init: A boolean indicating if during initialisation
                we should print a message stating that the process is being
                initialised successfully.

        Returns:
            On successful start this functionality will return None,
                otherwise will raise an error

        Example:

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
        Initialising Tickerplant process on port: 5030
        Tickerplant initialised successfully on port: 5030
        >>> tick.start()
        Starting Tickerplant data processing on port: 5030
        Tickerplant process successfully started on port: 5030
        ```
        """
        if print_init:
            print(f'Starting {self._name} {custom_start} on port: {self._port}')
        if config is None:
            config={}
        if not isinstance(config, dict):
            raise TypeError('Supplied configuration must be a "dict" object')
        self._init_config=config
        self._connection('.pykx.setdefault["k"]')
        self._connection('.tick.init', config)
        if print_init:
            print(f'{self._name} {custom_start} successfully started on port: {self._port}\n')
        self._initialized=True

    def stop(self):
        """
        Stop processing on the sub-process and kill the process.
            This allows the port on which the process is deployed to be reclaimed
            and the process to be restarted if appropriate.

        Example:

        ```python
        >>> import pykx as kx
        >>> tick = kx.tick.TICK(port=5030)
        Initialising Tickerplant process on port: 5030
        Tickerplant initialised successfully on port: 5030
        >>> tick.stop()
        Tickerplant process on port 5030 being stopped
        Tickerplant successfully shutdown on port 5030
        ```
        """
        print(f'{self._name} process on port {self._port} being stopped')
        self.server.stdin.close()
        self.server.kill()
        time.sleep(1)
        print(f'{self._name} successfully shutdown on port {self._port}\n')

    def libraries(self, libs: dict = None) -> None:
        """
        Specify and load the Python libraries which should be available on a
            process, the libraries should be supplied as a dictionary mapping
            the alias used for calling the library to the library name.

        Parameters:
            libs: A dictionary mapping the alias by which a Python library will be
                referred to the name of library

        Example:

        - In the following example we denote that the process should have access
          to the Python libraries `numpy` and `pykx` which when called by a user
          will be referred to as `np` and `kx` respectively

            ```python
            >>> import pykx as kx
            >>> tick = kx.tick.TICK(port=5030)
            Initialising Tickerplant process on port: 5030
            Tickerplant initialised successfully on port: 5030
            >>> tick.libraries({'np': 'numpy', 'kx': 'pykx'})
            ```
        """
        if libs is None:
            raise ValueError('No libraries provided')
        if not isinstance(libs, dict):
            raise TypeError('Provided libraries not of type dict')
        for key, value in libs.items():
            self._connection('{.pykx.pyexec"import ",string[y]," as ",string x}',
                             key,
                             value)

    def register_api(self, api_name: str, function: Callable) -> None:
        """
        Define a registered API to be callable by name on a process,
            this API can be a Python function or a PyKX
            lambda/projection.

        Parameters:
            api_name: The name by which the provided function will be called
                on the process
            function: The function which is to be defined as a callable API on
                the process, in the case of a Python function this must be a
                single independent function which is callable using libraries
                available on the process

        Example:

        ```python
        >>> import pykx as kx
        >>> def custom_func(num_vals, added_value):
        ...    return added_value + kx.q.til(num_vals)
        >>> hdb = kx.tick.HDB(port=5031)
        >>> hdb.libraries({'kx': 'pykx'})
        >>> hdb.register_api('custom_api', custom_func)
        >>> hdb('custom_api', 5, 10)
        pykx.LongVector(pykx.q('10 11 12 13 14'))
        ```
        """
        print(f"Registering callable function '{api_name}' on port {self._port}")
        if isinstance(function, k.Function):
            self._connection('set', api_name, function)
        else:
            try:
                src = dill.source.getsource(function)
            except BaseException:
                src = inspect.getsource(function)
            self._connection('{.pykx.pyexec x;z set .pykx.get[y;<]}',
                             bytes(src, 'UTF-8'),
                             function.__name__,
                             api_name)
        print(f"Successfully registed callable function '{api_name}' on port {self._port}")

    def set_timer(self, timer: int = 1000) -> None:
        """
        Set a timer on the connected process, this allows users to configure
        the intervals at which data is published for example.

        Parameters:
            timer: The interval at which the timer is triggered in milliseconds.

        Returns:
            On successful execution this will return None
        """
        self._connection('{system"t ",string[x]}', timer)

    def set_tables(self, tables: dict, tick: bool = False) -> None:
        """
        Define the tables to be available to the process being initialized.

        Parameters:
            tables: A dictionary mapping the name of a table to be defined on
                the process to the table schema
            tick: Is the process you are setting the table on a tickerplant?

        Returns:
            On a process persist the table schema as the supplied name

        Example:

        Set a table 'trade' with a supplied schema on a tickerplant process

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030)
        >>> tick.set_tables({'trade': trade})
        >>> tick('trade')
        pykx.Table(pykx.q('
        time sym exchange sz px
        -----------------------
        '))
        ```
        """
        for key, value in tables.items():
            if not isinstance(key, str):
                raise QError('Provided table name must be an "str"')
            if not isinstance(value, k.Table):
                raise QError('Provided table schema must be an "kx.Table"')
            if tick and not q('~', ['time', 'sym'], value.columns[:2]):
                raise QError("'time' and 'sym' must be first two columns "
                             f"in Table: {key}")
            self._connection('.tick.set_tables', key, value)


class TICK(STREAMING):
    """
    Initialise a tickerplant subprocess establishing a communication connection.
    This can either be a process which publishes data to subscribing processes only
    (chained) or a process which logs incoming messages for replay and triggers
    end-of-day events on subscribing processes.

    Parameters:
        port: The port on which the tickerplant process will be established
        process_logs: Should the logs of the generated tickerplant process be published
            to standard-out of the Python process (True), suppressed (False) or
            published to a supplied file-name
        tables: A dictionary mapping the names of tables and their schemas which are
            used to denote the tables which the tickerplant will process
        hard_reset: Reset logfiles for the current date when starting tickerplant
        log_directory: The location of the directory to which logfiles will be published
        chained: If the tickerplant is 'chained' or not, if chained the process will
            not log messages or run end-of-day processing
        init_args: A list of arguments passed to the initialized q process at startup
            denoting the command line options to be used for the initialized q process
            see [here](https://code.kx.com/q/basics/cmdline/) for a full breakdown.

    Returns:
        On successful initialisation will initialise the tickerplant process and set
            appropriate configuration

    Examples:

    Initialise a tickerplant on port 5030, defining a trade table.

    ```python
    >>> import pykx as kx
    >>> trade = kx.schema.builder({
    ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
    ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    ...     'px': kx.FloatAtom})
    >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
    Initialising Tickerplant process on port: 5030
    Tickerplant initialised successfully on port: 5030
    ```

    Initialise a chained tickerplant on port 5031 receiving messages from an upstream
        tickerplant on port 5030. Publish stdout/stderr from the process to a file
        'test.log'.

    ```python
    >>> import pykx as kx
    >>> trade = kx.schema.builder({
    ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
    ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    ...     'px': kx.FloatAtom})
    >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade}, process_logs='test.log')
    Initialising Tickerplant process on port: 5030
    Tickerplant initialised successfully on port: 5030
    >>> tick.start()
    Starting Tickerplant data processing on port: 5030
    Tickerplant process successfully started on port: 5030
    >>>
    >>> tick_chained = kx.tick.TICK(port=5031, chained=True)
    Initialising Tickerplant process on port: 5031
    Tickerplant initialised successfully on port: 5031
    >>> tick_chained.start({'tickerplant': 'localhost:5030'})
    Starting Tickerplant data processing on port: 5031
    Tickerplant process successfully started on port: 5031
    ```
    """
    def __init__(self,
                 port: int = 5010,
                 *,
                 process_logs: Union[bool, str] = True,
                 tables: dict = None,
                 log_directory: str = None,
                 hard_reset: bool = False,
                 chained: bool = False,
                 init_args: list = None) -> None:
        self._chained = chained
        self._tables=tables
        self._name = 'Tickerplant'

        print(f'Initialising {self._name} process on port: {port}')
        super().__init__(port, process_logs=process_logs, init_args=init_args)
        self._log_directory = os.getcwd() if log_directory is None else log_directory
        try:
            self._connection('{.tick.hardReset:x}', hard_reset)
            self._connection('{.tick.logdir:$[x~(::);();string[x]]}', log_directory)
            if chained:
                self._connection('.pykx.loadExtension["chained_tick"]')
            else:
                self._connection('.pykx.loadExtension["plant"]')
                if isinstance(tables, dict):
                    self.set_tables(tables)
        except BaseException as err:
            print(f'{self._name} failed to initialise on port: {port}\n')
            if self._connection is not None:
                self.server.stop()
            raise err
        print(f'{self._name} initialised successfully on port: {port}\n')

    def start(self, config: dict = None) -> None:
        """
        Start/initialise processing of messages on the associated tickerplant sub-process.
        This allows users to split the process initialisation from processing
        of data to allow additional configuration/setup to be completed before
        messages begin to be processed.

        Parameters:
            config: A dictionary passed to the sub-process which is used by
                the function `.tick.init` when the process is started, the
                supported parameters for this function will be different
                depending on process type.

        Returns:
            On successful start this functionality will return None,
                otherwise will raise an error

        Example:

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
        >>> tick.start()
        ```
        """
        print(f'Starting {self._name} data processing on port: {self._port}')
        if not self._chained:
            if self._connection('.tick.tabs').py() == []:
                raise QError('Unable to initialise TICKERPLANT without tables '
                             'set using "set_tables"')
        super().start(config, print_init=False, custom_start='data processing')
        print(f'{self._name} process successfully started on port: {self._port}\n')

    def restart(self) -> None:
        """
        Restart and re-initialise the Tickerplant, this will
           start the processes with all tables defined on the expected port

        Example:

        Restart a Tickerplant validating that the expected tables are
        appropriately defined

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
        Initialising Tickerplant process on port: 5030
        Tickerplant initialised successfully on port: 5030
        >>> tick.start()
        Starting Tickerplant data processing on port: 5030
        Tickerplant process successfully started on port: 5030
        >>> tick.restart()
        Restarting Tickerplant on port 5030

        Tickerplant process on port 5030 being stopped
        Tickerplant successfully shutdown on port 5030

        Initialising Tickerplant process on port: 5030
        Tickerplant initialised successfully on port: 5030

        Tickerplant on port 5030 successfully restarted
        >>> tick('trade')
        pykx.Table(pykx.q('
        time sym exchange sz px
        -----------------------
        '))
        ```
        """
        print(f'Restarting {self._name} on port {self._port}\n')
        self.stop()
        self.__init__(port=self._port,
                      process_logs=self._process_logs,
                      tables=self._tables,
                      log_directory=self._log_directory,
                      chained=self._chained)
        if self._init_config is not None:
            self.init(config=self._init_config)
        print(f'{self._name} on port {self._port} successfully restarted\n')

    def set_tables(self, tables: dict) -> None:
        """
        Define the tables to be used for consuming and serving messages on
            the tickerplant process.

        Parameters:
            tables: A dictionary mapping the name of a table to be defined on
                the process to the table schema

        Returns:
            On the tickerplant persist the table schema as the supplied name

        Example:

        Set a table 'trade' with a supplied schema on a tickerplant process

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030)
        >>> tick.set_tables({'trade': trade})
        >>> tick('trade')
        pykx.Table(pykx.q('
        time sym exchange sz px
        -----------------------
        '))
        ```
        """
        super().set_tables(tables, tick=True)

    def set_snap(self, snap_function: Callable) -> None:
        """
        Define a 'snap' function used by KX Dashboards UI to manage the data
        presented to a Dashboard process when subscribing to data from a
        Tickerplant process.

        Parameters:
            snap_function: A Python function or callable PyKX Lambda which takes
                a single argument and returns the expected tabular dataset for
                display

        Returns:
            On successful execution will set the streaming function `.u.snap` and
                return None

        Example:

        Implement a ring-buffer to provide the most recent 1,000 datapoints

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030)
        >>> def buffer_ring(x):
        ...     if 1000 < len(kx.q['trade']):
        ...         return trade
        ...     else:
        ...         kx.q['trade'][-1000:]
        >>> tick.set_
        ```
        """
        if not isinstance(snap_function, Callable):
            raise QError('Provided snap_function is not a callable function')
        self.register_api('.u.snap', snap_function)


# The below is named real-time processing to allow for a distinction between an RDB and RTE
# to not be required at initialisation ... naming is hard
class RTP(STREAMING):
    """
    Initialise a Real-Time Processor (RTP), establishing a communication connection to this
    process. An RTP at it's most fundamental level comprises the following actions and is
    known as a 'vanilla' RTP:

    1. Receives messages from an upstream tickerplant process via subscription.
    2. Inserts data into an in-memory table which will be written to disk at a defined
        time interval.
    3. Triggers end-of-day processing which writes the data to disk and telling connected
        historical databases to reload if needed.

    In a more complex case an RTP will run analytics on data prior to and post data insert
    as noted in step 2 above. These analytics can either be Python or q/PyKX functions.
    Additionally users can define 'apis' on the server which can be called explicitly
    by users.

    Parameters:
        port: The port on which the RTP process will be established
        process_logs: Should the logs of the generated RTP process be published
            to standard-out of the Python process (True), suppressed (False) or
            published to a supplied file-name
        libraries: A dictionary mapping the alias by which a Python library will be
            referred to the name of library
        subscriptions: A list of tables (str) from which to receive updates, if None
            the RTP will receive updates from all tables
        apis: A dictionary mapping the names to be used by users when calling a
            defined API to the callable Python functions or PyKX lambdas/projections
            which will be called.
        vanilla: In the case that the RTP is defined as 'vanilla' data received
            from a downstream tickerplant will be inserted into an in-memory table.
            If vanilla is False then a 'pre_processor' and 'post_processor' function
            can be defined using the below parameters to modify data prior to and post
            insert.
        pre_processor: A function taking the name of a table and message as parameters,
            this function should/can modify the message prior to insertion into an
            in-memory table. If this function returns `None` the processing of that
            message will be terminated and the data will not be inserted to the table.
        post_processor: A function taking the name of a table and message as parameters,
            this function can publish data to other processes, update global variables etc.
            In most examples post_processor functions are used to publish data to a
            tickerplant or persist derived analytics for use by other users.
        init_args: A list of arguments passed to the initialized q process at startup
            denoting the command line options to be used for the initialized q process
            see [here](https://code.kx.com/q/basics/cmdline/) for a full breakdown.
        tables: A dictionary mapping the names of tables and their schemas which can be
            used to define the tables available to the real-time processor.

    Returns:
        On successful initialisation will initialise the RTP process and set
            appropriate configuration

    Examples:

    Initialise a vanilla Real-Time Processor on port 5032 subscribing to all messages
    from a tickerplant on port 5030.

    ```python
    >>> import pykx as kx
    >>> trade = kx.schema.builder({
    ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
    ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    ...     'px': kx.FloatAtom})
    >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade}, process_logs='test.log')
    Initialising Tickerplant process on port: 5030
    Tickerplant initialised successfully on port: 5030
    >>> tick.start()
    Starting Tickerplant data processing on port: 5030
    Tickerplant process successfully started on port: 5030
    >>>
    >>> rdb = kx.tick.RTP(port=5032)
    Initialising Real-time processor on port: 5032
    Real-time processor initialised successfully on port: 5032
    >>> rdb.start({'tickerplant': 'localhost:5030'})
    Starting Real-time processing on port: 5032
    Real-time processing successfully started on port: 5032
    ```

    Initialise a vanilla Real-Time Processor on port 5032 logging process logs to 'test.log',
    subscribing to a table `trade` only and defining a query API named `custom_query`.

    ```python
    >>> import pykx as kx
    >>> trade = kx.schema.builder({
    ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
    ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    ...     'px': kx.FloatAtom})
    >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
    >>> tick.start()
    >>>
    >>> def query_api(table):
    ...     return kx.q.qsql.select(table)
    >>> rdb = kx.tick.RTP(
    ...     port=5032,
    ...     process_logs='test.log',
    ...     libraries = {'kx': 'pykx'},
    ...     api={'custom_query': query_api}
    ...     )
    Initialising Real-time processor on port: 5032
    Registering callable function 'custom_query' on port 5032
    Successfully registed callable function 'custom_query' on port 5032
    Real-time processor initialised successfully on port: 5032
    >>> rdb.start({'tickerplant': 'localhost:5030'})
    Starting Real-time processing on port: 5032
    Real-time processing successfully started on port: 5032
    ```

    Initialise a complex Real-Time Processor which includes data pre-processing
    prior to insertion of data into the Real-Time Database and which contains a
    post-processing step to derive analytics after data has been inserted into the
    in-memory table.

    ```python
    >>> import pykx as kx
    >>> trade = kx.schema.builder({
    ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
    ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    ...     'px': kx.FloatAtom})
    >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
    >>> tick.start()
    >>> def pre_process(table, message):
    ...     if table in ['trade', 'quote']:
    ...         return message
    ...     else:
    ...         return None
    >>> def post_process(table, message):
    ...     tradeagg = kx.q.qsql.select('trade',
    ...                                 columns={'trdvol': 'sum px*sz',
    ...                                          'maxpx': 'max px',
    ...                                          'minpx': 'min px'},
    ...                                 by='sym')
    ...      quoteagg = kx.q.qsql.select('quote',
    ...                                  columns={'maxbpx': 'max bid',
    ...                                           'minapx': 'min ask',
    ...                                           'baspread': 'max[bid]-min[ask]'},
    ...                                  by='sym')
    ...      tab = tradeagg.merge(quoteagg, how='left', q_join=True).reset_index()
    ...      tab['time'] = kx.TimespanAtom('now')
    ...      kx.q['aggregate'] = kx.q.xcols(['time', 'sym'], tab)
    ...      return None
    >>> rte = kx.tick.RTP(port=5031,
    ...                   libraries = {'kx': 'pykx'},
    ...                   subscriptions = ['trade', 'quote'],
    ...                   pre_processor = q_preproc,
    ...                   post_processor = rte_post_analytic,
    ...                   vanilla=False)
    >>> rte.start({'tickerplant': 'localhost:5030'})
    ```
    """
    def __init__(self,
                 port: int = 5011,
                 *,
                 process_logs: Union[bool, str] = True,
                 libraries: dict = None,
                 subscriptions: str = None,
                 apis: dict = None,
                 vanilla: bool = True,
                 pre_processor: Callable = None,
                 post_processor: Callable = None,
                 init_args: list = None,
                 tables: dict = None) -> None:
        self._subscriptions=subscriptions
        self._pre_processor=pre_processor
        self._post_processor=post_processor
        self._tables = tables
        self._vanilla = vanilla
        self._name = 'Real-time'

        print(f'Initialising {self._name} processor on port: {port}')
        try:
            super().__init__(port,
                             process_logs=process_logs,
                             libraries=libraries,
                             apis=apis,
                             init_args=init_args)
            self._connection('{.tick.vanilla:x}', vanilla)
            self._connection('.pykx.loadExtension["rdb"]')
            if pre_processor is not None:
                self.pre_processor(pre_processor)
            if post_processor is not None:
                self.post_processor(post_processor)
            if subscriptions is not None:
                self.subscriptions(subscriptions)
            if isinstance(tables, dict):
                self.set_tables(tables)
        except BaseException as err:
            print(f'{self._name} processor failed to initialise on port: {port}\n')
            if self._connection is not None:
                self.server.stop()
            raise err
        print(f'{self._name} processor initialised successfully on port: {port}\n')

    def start(self, config: dict = None) -> None:
        """
        Start/initialise processing of messages on the Real-Time Processor.
        This splits the process initialisation from processing of data to allow
        additional configuration/setup to be completed before messages begin to
        be processed.

        Parameters:
            config: A dictionary passed to the sub-process which is used by
                the function `.tick.init` when the process is started. The following
                are the supported config options for RTP processes

                    1. `tickerplant`: a string denoting the host+port of the
                        tickerplant from which messages are received. By default
                        port 5010 will be used
                    2. `hdb`: a string denoting the host+port of the HDB
                        which will be re-loaded at end-of-day
                    3. `database: a string denoting the directory where your current
                        days data will be persisted. This should be the same directory
                        as the `database` keyword for your HDB process should it be used.
                        By default the location "db" will be used in the directory PyKX was
                        imported.

        Returns:
            On successful start this functionality will return None,
                otherwise will raise an error

        Example:

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
        >>> tick.start()
        >>> rdb = kx.tick.RTP(port=5032,
        ...     subscriptions = ['trade', 'quote']
        ...     )
        >>> rdb.start({
        ...     'tickerplant': 'localhost:5030',
        ...     'hdb': 'localhost:5031',
        ...     'database': 'db'})
        ```
        """
        super().start(config, custom_start='processing')

    def restart(self) -> None:
        """
        Restart and re-initialise the Real-Time Processor, this will
           start the processes with all subscriptions, processing functions
           etc as defined in the initial configuration of the processes.

        Example:

        Restart an RTP process validating that defined API's in the restarted
        process are appropriately defined

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
        Initialising Tickerplant process on port: 5030
        Tickerplant initialised successfully on port: 5030
        >>> tick.start()
        Starting Tickerplant data processing on port: 5030
        Tickerplant process successfully started on port: 5030
        >>>
        >>> def query_api(table):
        ...     return kx.q.qsql.select(table)
        >>> rdb = kx.tick.RTP(
        ...     port=5032,
        ...     process_logs='test.log',
        ...     libraries = {'kx': 'pykx'},
        ...     api={'custom_query': query_api}
        ...     )
        Initialising Real-time processor on port: 5032
        Registering callable function 'custom_query' on port 5032
        Successfully registed callable function 'custom_query' on port 5032
        Real-time processor initialised successfully on port: 5032
        >>> rdb.start({'tickerplant': 'localhost:5030'})
        Starting Real-time processing on port: 5032
        Real-time processing successfully started on port: 5032
        >>> rdb.restart()
        Restarting Real-time processor on port 5032

        Real-time processor process on port 5032 being stopped
        Real-time processor successfully shutdown on port 5032

        Initialising Real-time processor on port: 5032
        Registering callable function 'custom_query' on port 5032
        Successfully registed callable function 'custom_query' on port 5032
        Real-time processor initialised successfully on port: 5032

        Starting Real-time processing on port: 5032
        Real-time processing successfully started on port: 5032

        Real-time processor on port 5032 successfully restarted
        >>> rdb('tab:([]5?1f;5?1f)')
        >>> rdb('custom_query', 'tab')
        pykx.Table(pykx.q('
        x         x1
        -------------------
        0.3017723 0.3927524
        0.785033  0.5170911
        0.5347096 0.5159796
        0.7111716 0.4066642
        0.411597  0.1780839
        '))
        ```
        """
        print(f'Restarting {self._name} processor on port {self._port}\n')
        self.stop()
        self.__init__(port=self._port,
                      process_logs=self._process_logs,
                      libraries=self._libraries,
                      subscriptions=self._subscriptions,
                      apis=self._apis,
                      vanilla=self._vanilla,
                      pre_processor=self._pre_processor,
                      post_processor=self._post_processor,
                      tables=self._tables)
        if self._init_config is not None:
            self.init(config=self._init_config)
        print(f'{self._name} processor on port {self._port} successfully restarted\n')

    def pre_processor(self, function: Callable) -> None:
        """
        Define a pre-processing function on the RTP process which is
        called prior to inserting data into the Real-Time Database.

        This function must take two parameters:

        1. table: The name of the table to which data will be inserted
        2. message: The data which is to be inserted into the table

        If this function returns `None` or `kx.q('::')` then data processing
        will not continue for that message and it will not be inserted into
        the database.

        The pre-processing function should return

        Parameters:
            function: A callable function or PyKX Lambda taking 2 arguments
                the name of the table as a `str` and the message to be processed

        Returns:
            On successful execution of this method the data pre-processing function
                defined on the RTP server will be updated

        Example:

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
        >>> tick.start()
        >>> def preprocess(table, message):
        ...     if table in ['trade', 'quote']:
        ...         return message
        ...     else:
        ...         return None
        >>> rte = kx.tick.RTP(port=5034,
        ...                   libraries = {'kx': 'pykx'},
        ...                   subscriptions = ['trade', 'quote'],
        ...                   vanilla=False)
        >>> rte.pre_processor(preprocess)
        ```
        """
        if self._vanilla:
            raise QError('Pre-processing of incoming message not '
                         'supported in vanilla real-time processor')
        if isinstance(function, k.Function):
            self._connection('set', '.tick.RTPPreProc', function)
            return None
        try:
            src = dill.source.getsource(function)
        except BaseException:
            src = inspect.getsource(function)
        self._connection('{.pykx.pyexec x;z set .pykx.get[y;<]}',
                         bytes(src, 'UTF-8'),
                         function.__name__,
                         '.tick.RTPPreProc')

    def post_processor(self, function: Callable) -> None:
        """
        Define a post-processing function on the RTP process which is
        called after inserting data into the Real-Time Database.

        This function must take two parameters:

        1. table: The name of the table to which data will be inserted
        2. message: The data which is to be inserted into the table

        This function can have side-effects and does not expect a return

        Parameters:
            function: A callable function or PyKX Lambda taking 2 arguments
                the name of the table as a `str` and the message to be processed

        Returns:
            On successful execution of this method the data pre-processing function
                defined on the RTP server will be updated

        Example:

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
        >>> tick.start()
        >>> def postprocess(table, message):
        ...     tradeagg = kx.q.qsql.select('trade',
        ...                                 columns={
        ...                                   'trdvol': 'sum px*sz',
        ...                                   'maxpx': 'max px',
        ...                                   'minpx': 'min px'},
        ...                                 by='sym')
        ...     quoteagg = kx.q.qsql.select('quote',
        ...                                 columns={
        ...                                   'maxbpx': 'max bid',
        ...                                   'minapx': 'min ask',
        ...                                   'baspread': 'max[bid]-min[ask]'},
        ...                                 by='sym')
        ...     kx.q['aggregate'] = kx.q.xcols(['time', 'sym'], tab)
        ...     return None
        >>> rte = kx.tick.RTP(port=5034,
        ...                   libraries = {'kx': 'pykx'},
        ...                   subscriptions = ['trade', 'quote'],
        ...                   vanilla=False)
        >>> rte.post_processor(postprocess)
        ```
        """
        if self._vanilla:
            raise QError('Post-processing of incoming message not '
                         'supported in vanilla real-time processor')
        if isinstance(function, k.Function):
            self._connection('set', '.tick.RTPPostProc', function)
            return None
        try:
            src = dill.source.getsource(function)
        except BaseException:
            src = inspect.getsource(function)
        self._connection('{.pykx.pyexec x;z set .pykx.get[y;<]}',
                         bytes(src, 'UTF-8'),
                         function.__name__,
                         '.tick.RTPPostProc')

    def set_tables(self, tables: dict) -> None:
        """
        Define tables to be available on the RTP processes.

        Parameters:
            tables: A dictionary mapping the name of a table to be defined on
                the process to the table schema

        Returns:
            On the RTP persist the table schemas as the supplied name

        Example:

        Set a table 'trade' with a supplied schema on a tickerplant process

        ```python
        >>> import pykx as kx
        >>> prices = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> rte = kx.tick.RTP(port=5034,
        ...                   subscriptions = ['trade', 'quote'],
        ...                   vanilla=False)
        >>> rte.set_tables({'prices': prices})
        >>> rte('prices')
        pykx.Table(pykx.q('
        time sym exchange sz px
        -----------------------
        '))
        ```
        """
        super().set_tables(tables)

    def subscriptions(self, sub_list):
        self._connection('{.tick.subscriptions:x}', sub_list)


class HDB(STREAMING):
    """
    Initialise a Historical Database (HDB) subprocess establishing a communication connection.
    This process may contain a loaded database and APIs used for analytic transformations on
    historical data

    Parameters:
        port: The port on which the tickerplant process will be established
        process_logs: Should the logs of the generated tickerplant process be published
            to standard-out of the Python process (True), suppressed (False) or
            published to a supplied file-name
        libraries: A dictionary mapping the alias by which a Python library will be
            referred to the name of library
        apis: A dictionary mapping the names to be used by users when calling a
            defined API to the callable Python functions or PyKX lambdas/projections
            which will be called.
        init_args: A list of arguments passed to the initialized q process at startup
            denoting the command line options to be used for the initialized q process
            see [here](https://code.kx.com/q/basics/cmdline/) for a full breakdown.
         tables: A dictionary mapping the names of tables and their schemas which can be
            used to define the tables available to the HDB.

    Returns:
        On successful initialisation will initialise the HDB process and set
            appropriate configuration

    Examples:

    Initialise a HDB on port 5035

    ```python
    >>> import pykx as kx
    >>> hdb = kx.tick.HDB(port=5035)
    Initialising HDB process on port: 5035
    HDB initialised successfully on port: 5035
    ```

    Initialise a HDB on port 5035, defining a custom api on the process
        and stating that the library `pykx` must be available.

    ```python
    >>> import pykx as kx
    >>> def custom_api(values):
    ...     return kx.q(values)
    >>> hdb = kx.tick.HDB(
    ...     port=5035,
    ...     libraries={'kx': 'pykx'},
    ...     apis={'hdb_query': custom_api}
    ...     )
    Initialising HDB process on port: 5035
    Registering callable function 'hdb_query' on port 5035
    Successfully registed callable function 'hdb_query' on port 5035
    HDB initialised successfully on port: 5035
    >>> hdb('hdb_query', '1+1')
    pykx.LongAtom(pykx.q('2'))
    ```
    """
    def __init__(self,
                 port: int = 5012,
                 *,
                 process_logs: Union[str, bool] = True,
                 libraries: dict = None,
                 apis: dict = None,
                 init_args: list = None,
                 tables: dict = None):
        self._name = 'HDB'
        self._libraries = libraries
        self._apis = apis
        self._tables = tables
        print(f'Initialising {self._name} process on port: {port}')
        try:
            super().__init__(port,
                             process_logs=process_logs,
                             apis=apis,
                             libraries=libraries,
                             init_args=init_args)
            self._connection('.pykx.loadExtension["hdb"]')
            if isinstance(tables, dict):
                super().set_tables(tables)
        except BaseException as err:
            print(f'{self._name} failed to initialise on port: {port}\n')
            if self._connection is not None:
                self.server.stop()
            raise err
        print(f'{self._name} initialised successfully on port: {port}\n')

    def start(self, database: str = None, config: dict = None) -> None:
        """
        Start the Historical Database (HDB) process for analytic/query availability.
        This command allows for the loading of the Database to be used by the process.

        Parameters:
            database: The path to the database which is to be loaded on the process.
            config: A dictionary passed to the sub-process which can be used by
                the function `.tick.init` when the process is started.

        Returns:
            On successful start this functionality will return None and load
                the specified database, otherwise will raise an error.

        Example:

        ```python
        >>> import pykx as kx
        >>> hdb = kx.tick.HDB(port=5031)
        >>> hdb.start(database='/tmp/db')
        ```
        """
        print(f'Starting {self._name} process to allow historical query')
        if config is None:
            config = {}
        self._database=database
        if database is None:
            raise QError(f"{self._name} initialisation requires defined 'database'")
        config['database'] = database
        super().start(config, print_init=False, custom_start='load')
        print(f'{self._name} process successfully started\n')

    def restart(self) -> None:
        """
        Restart and re-initialise the HDB Process, this will
           start the processes with validation and api functions
           etc as defined in the initial configuration of the processes.

        Example:

        Restart a HDB process validating that defined API's in the restarted
        process are appropriately defined

        ```python
        >>> import pykx as kx
        >>> def hdb_api(value):
        ...     return kx.q(value)
        >>> hdb = kx.tick.HDB(
        ...     port=5035,
        ...     libraries={'kx': 'pykx'},
        ...     apis={'custom_api': gateway_api})
        Initialising HDB process on port: 5035
        Registering callable function 'custom_api' on port 5035
        Successfully registed callable function 'custom_api' on port 5035
        HDB process initialised successfully on port: 5035
        >>> hdb('custom_api', '1+1')
        pykx.LongAtom(pykx.q('2'))
        >>> hdb.restart()
        Restarting HDB on port 5035

        HDB process on port 5035 being stopped
        HDB successfully shutdown on port 5035

        Initialising HDB process on port: 5035
        Registering callable function 'custom_api' on port 5035
        Successfully registed callable function 'custom_api' on port 5035
        HDB process initialised successfully on port: 5035

        HDB process on port 5035 successfully restarted
        >>> hdb('custom_api', '1+1')
        pykx.LongAtom(pykx.q('2'))
        ```
        """
        print(f'Restarting {self._name} on port {self._port}\n')
        self.stop()
        self.__init__(port=self._port,
                      process_logs=self._process_logs,
                      libraries=self._libraries,
                      apis=self._apis,
                      tables=self._tables)
        if self._init_config is not None:
            self.init(self._database, self._init_config)
        print(f'{self._name} on port {self._port} successfully restarted\n')

    def set_tables(self, tables: dict) -> None:
        """
        Define tables to be available on the HDB processes.

        Parameters:
            tables: A dictionary mapping the name of a table to be defined on
                the process to the table schema

        Returns:
            On the HDB persist the table schemas as the supplied name

        Example:

        Set a table 'prices' with a supplied schema on a HDB process

        ```python
        >>> import pykx as kx
        >>> prices = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> hdb = kx.tick.HDB(port=5035)
        Initialising HDB process on port: 5035
        HDB process initialised successfully on port: 5035
        >>> hdb.set_tables({'prices': prices})
        >>> hdb('prices')
        pykx.Table(pykx.q('
        time sym exchange sz px
        -----------------------
        '))
        ```
        """
        super().set_tables(tables)


class GATEWAY(STREAMING):
    """
    Initialise a Gateway subprocess establishing a communication connection.
    A gateway provides a central location for external users to query named
    API's within a streaming infrastructure which retrieves data from multiple
    processes within the infrastructure.

    A gateway within this implementation provides helper functions for the
    application of basic user validation and functionality to allow custom
    API's to call named process connections.

    Parameters:
        port: The port on which the tickerplant process will be established
        process_logs: Should the logs of the generated tickerplant process be published
            to standard-out of the Python process (True), suppressed (False) or
            published to a supplied file-name
        libraries: A dictionary mapping the alias by which a Python library will be
            referred to the name of library
        apis: A dictionary mapping the names to be used by users when calling a
            defined API to the callable Python functions or PyKX lambdas/projections
            which will be called.
        connections: A dictionary passed to the sub-process which is used by
                maps a key denoting the 'name' to be assigned
                to a process with the connection string as follows.
                `<host>:<port>:<username>:<password>` where `username` and
                `password` are optional.
        connection_validator: A function taking username and password which returns
            `True` or `False depending on whether connecting user should be
            allowed to connect or not.
        init_args: A list of arguments passed to the initialized q process at startup
            denoting the command line options to be used for the initialized q process
            see [here](https://code.kx.com/q/basics/cmdline/) for a full breakdown.

    Returns:
        On successful initialisation will initialise the Gateway process and set
            appropriate configuration

    Examples:

    Initialise a Gateway defining a callable API against a HDB and RDB process.
    This will allow free-form function calls on both processes.

    ```python
    >>> import pykx as kx
    >>> trade = kx.schema.builder({
    ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
    ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    ...     'px': kx.FloatAtom})
    >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
    >>> tick.start()
    >>> hdb = kx.tick.HDB(port=5031)
    >>> hdb.start(database='/tmp/db')
    >>> rdb = kx.tick.RTP(port=5032)
    >>> rdb.start({'tickerplant': 'localhost:5030'})
    >>> def gateway_func(x):
    ...     # The 'module' gateway is a populated class
    ...     # on the PyKX Gateway processes
    ...     rdb_data = gateway.call_port('rdb', b'{x+1}', x)
    ...     hdb_data = gateway.call_port('hdb', b'{x+2}', x)
    ...     return([rdb_data, hdb_data])
    >>> gw = kx.tick.GATEWAY(
    ...     port=5033,
    ...     connections={'rdb': 'localhost:5032', 'hdb: 'localhost:5031'},
    ...     apis={'custom_api': gateway_func}
    ...     )
    >>> gw.start()
    >>> with kx.SyncQConnection(port=5033) as q:
    ...     print(q('custom_api', 2))
    ```
    """
    def __init__(self,
                 port: int = 5010,
                 *,
                 process_logs: Union[str, bool] = False,
                 libraries: dict = None,
                 apis: dict = None,
                 connections: dict = None,
                 connection_validator: Callable = None,
                 init_args: list = None) -> None:
        self._name = 'Gateway'
        self._connections=connections
        self._connection_validator=connection_validator

        print(f'Initialising {self._name} process on port: {port}')
        super().__init__(port,
                         process_logs=process_logs,
                         libraries=libraries,
                         apis=apis,
                         init_args=init_args)
        try:
            self._connection('.pykx.loadExtension["gateway"]')
            if connection_validator is not None:
                self.connection_validation(connection_validator)
        except BaseException as err:
            print(f'{self._name} failed to initialise on port: {port}\n')
            if self._connection is not None:
                self.server.stop()
            raise err
        if connections is not None:
            self._connection('{.gw.ports:x}', connections)
        print(f'{self._name} process initialised successfully on port: {port}\n')

    def start(self, config: dict = None) -> None:
        """
        Start the gateway processes connections to external processes.
        This supplied configuration will be used to create 'named'
        inter-process connections with remote processes which can
        be called by users in their gateway functions.

        Parameters:
            config: UNUSED

        Returns:
            On successful start this functionality will return None,
                otherwise will raise an error

        Example:

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
        >>> def gateway_api(value):
        ...     gw.call('tp', b'{x+1}', value)
        >>> gw = kx.tick.GATEWAY(
        ...     port=5031,
        ...     connections={'tp': 'localhost:5030'},
        ...     apis={'custom_api': gateway_api})
        >>> gw.start()
        ```
        """
        super().start(config, custom_start='access')

    def add_connection(self, connections: dict = None):
        """
        Add additional callable named connections to a gateway process
            this functionality is additive to the connections (if established)
            when configuring a `GATEWAY` process. If the same name is used for
            two connections the last added connection will be used in function
            execution.

        Parameters:
            connections: A dictionary which maps a key denoting the 'name' to
                be assigned to a process with the connection string containing the
                host/port information as follows:
                `<host>:<port>:<username>:<password>` where `username` and
                `password` are optional.

        Example:

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> tick = kx.tick.TICK(port=5030, tables={'trade': trade})
        >>> def gateway_api(value):
        ...     gw.call('tp', b'{x+1}', value)
        >>> gw = kx.tick.GATEWAY(
        ...     port=5031,
        ...     apis={'custom_api': gateway_api})
        >>> gw.add_connection({'tp': 'localhost:5030'})
        ```
        """
        if (connections is None) or not isinstance(connections, dict):
            raise TypeError('connections must be supplied as a dict object')
        self._connection('.tick.addConnection', connections)

    def restart(self) -> None:
        """
        Restart and re-initialise the Gateway Process, this will
           start the processes with validation and api functions
           etc as defined in the initial configuration of the processes.

        Example:

        Restart a Gateway process validating that defined API's in the restarted
        process are appropriately defined

        ```python
        >>> import pykx as kx
        >>> def gateway_api(value):
        ...     return kx.q(value)
        >>> gateway = kx.tick.GATEWAY(
        ...     port=5035,
        ...     libraries={'kx': 'pykx'},
        ...     apis={'custom_api': gateway_api})
        Initialising Gateway process on port: 5035
        Registering callable function 'custom_function' on port 5035
        Successfully registed callable function 'custom_function' on port 5035
        Gateway process initialised successfully on port: 5035
        >>> gateway.start()
        >>> gateway('gateway_api', '1+1')
        pykx.LongAtom(pykx.q('2'))
        >>> gateway.restart()
        Restarting Gateway on port 5035

        Gateway process on port 5035 being stopped
        Gateway successfully shutdown on port 5035

        Initialising Gateway process on port: 5035
        Registering callable function 'custom_function' on port 5035
        Successfully registed callable function 'custom_function' on port 5035
        Gateway process initialised successfully on port: 5035

        Gateway process on port 5035 successfully restarted
        >>> gateway('gateway_api', '1+1')
        pykx.LongAtom(pykx.q('2'))
        ```
        """
        print(f'Restarting {self._name} on port {self._port}\n')
        self.stop()
        self.__init__(port=self._port,
                      process_logs=self._process_logs,
                      libraries=self._libraries,
                      apis=self._apis,
                      connection_validator=self._connection_validator)
        if self._init_config is not None:
            self.init(self._init_config)
        print(f'{self._name} on port {self._port} successfully restarted\n')

    def connection_validation(self, function: Callable) -> None:
        """
        Define a function to be used on the Gateway process which validates
        users connecting to the process. This function should take two
        inputs, username and password and validate a user connecting is
        allowed to do so.

        This function should return `True` if a user is permitted to establish
        a connection and `False` if they are not.

        Parameters:
            function: A function taking two parameters (username and password) which
                validates that a user connecting to the process is permitted or not
                to establish a callable connection.

        Example:

        Define a function on the gateway process to only accept users with the name
            'new_user'.

        ```python
        >>> import pykx as kx
        >>> def custom_validation(username, password):
        ...     if username != 'new_user':
        ...         return False
        ...     else:
        ...         return True
        >>> gateway = kx.tick.GATEWAY(port=5034, connection_validator=custom_validation)
        >>> with kx.SyncQConnection(port=5034, username='user') as q:
        ...     q('1+1')
        QError: access
        >>> with kx.SyncQConnection(port=5034, username='new_user') as q:
        ...     q('1+1')
        pykx.LongAtom(pykx.q('2'))
        ```
        """
        if isinstance(function, k.Function):
            self._connection('set', '.z.pw', function)
            return None
        try:
            src = dill.source.getsource(function)
        except BaseException:
            src = inspect.getsource(function)
        self._connection('{.pykx.pyexec x;z set .pykx.get[y;<]}',
                         bytes(src, 'UTF-8'),
                         function.__name__,
                         '.z.pw')


_default_ports = {'tickerplant': 5010,
                  'rdb': 5011,
                  'hdb': 5012}


class BASIC:
    """
    Initialise a configuration for a basic PyKX streaming workflow.

    This configuration will be used to (by default) start the following processes:

    1. A Tickerplant process on port 5010 to which messages can be published
        for logging and consumption by down-stream subscribers.
    2. A Real-Time Database process (RDB) on port 5011 which subscribes to the
        tickerplant and maintains an in-memory representation of all the data
        consumed that day.
    3. If a database is denoted at initialisation initialise a Historical Database (HDB)
        process which loads the database and makes available historical data to a user.

    With this basic infrastructure users can then add functionality to increase overall
        complexity of their system.

    Parameters:
        tables: A dictionary mapping the names of tables and their schemas which are
            used to denote the tables which the tickerplant will process
        log_directory: The location of the directory to which logfiles will be published
        database: The path to the database which is to be loaded on the HDB process and
            the working directory of the RDB process
        hard_reset: Reset logfiles for the current date when starting tickerplant
        ports: A dictionary mapping the process type to the IPC communication port on which it
            should be made available. Dictionary "Values" must be supplied as integers denoting
            the desired port while "Keys" should be a str object of value "tickerplant", "rdb"
            and "hdb".

    Returns:
        On successful initialisation will initialise the Tickerplant, RDB and HDB processes,
            setting appropriate configuration

    Examples:

    Configure a Tickerplant and RDB process using default parameters

    ```python
    >>> import pykx as kx
    >>> trade = kx.schema.builder({
    ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
    ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    ...     'px': kx.FloatAtom})
    >>> basic = kx.tick.BASIC(tables={'trade': trade})
    ```

    Configure a Tickerplant, RDB and HDB process architecture loading a database
        at the location `'/tmp/db'` and persisting the tickerplant logs to
        the folder `logs`

    ```python
    >>> import pykx as kx
    >>> trade = kx.schema.builder({
    ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
    ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    ...     'px': kx.FloatAtom})
    >>> basic = kx.tick.BASIC(
    ...     tables={'trade': trade},
    ...     database='/tmp/db',
    ...     log_directory='logs')
    ```

    Configure a Tickerplant, RDB and HDB process setting these processes on the
        ports 5030, 5031 and 5032 respectively

    ```python
    >>> import pykx as kx
    >>> trade = kx.schema.builder({
    ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
    ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
    ...     'px': kx.FloatAtom})
    >>> basic = kx.tick.BASIC(
    ...     tables={'trade': trade},
    ...     ports={'tickerplant': 5030, 'rdb': 5031, 'hdb': 5032}
    ```
    """
    def __init__(
            self,
            tables,
            *,
            log_directory='.',
            hard_reset=False,
            database=None,
            ports=_default_ports):
        self._ports = ports
        self._tables = tables
        self._log_directory = log_directory,
        self._database = database
        self._hard_reset = hard_reset
        self.tick = None
        self.rdb = None
        self.hdb = None

    def start(self) -> None:
        """
        Start a basic streaming architecture configured using `kx.tick.BASIC`

        With this basic infrastructure users can then add functionality to increase overall
            complexity of their system.

        Returns:
            On successful initialisation will start the Tickerplant, RDB and HDB processes,
                setting appropriate configuration

        Examples:

        Configure and start a Tickerplant and RDB process using default parameters

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> basic = kx.tick.BASIC(tables={'trade': trade})
        >>> basic.start()
        ```

        Configure and start a Tickerplant, RDB and HDB process architecture loading a database
            at the location `'/tmp/db'` and persisting the tickerplant logs to
            the folder `logs`

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> basic = kx.tick.BASIC(
        ...     tables={'trade': trade},
        ...     database='/tmp/db',
        ...     log_directory='logs')
        >>> basic.start()
        ```

        Configure and start a Tickerplant, RDB and HDB process setting these processes on the
            ports 5030, 5031 and 5032 respectively

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> basic = kx.tick.BASIC(
        ...     tables={'trade': trade},
        ...     ports={'tickerplant': 5030, 'rdb': 5031, 'hdb': 5032}
        ```
        """
        # Initialise tickerplant
        try:
            tick = TICK(
                port=self._ports['tickerplant'],
                tables=self._tables,
                hard_reset=self._hard_reset,
                log_directory=self._log_directory)
            self.tick = tick
            self.tick.start()
        except BaseException as err:
            if self.tick is not None:
                self.tick.stop()
            raise err

        # Initialise HDB
        if self._database is not None:
            try:
                hdb = HDB(port=self._ports['hdb'])
                self.hdb = hdb
                self.hdb.start(database=self._database)
            except BaseException as err:
                self.tick.stop()
                if self.hdb is not None:
                    self.hdb.stop()
                raise err

        # Initialise RDB
        try:
            rdb = RTP(port=self._ports['rdb'])
            self.rdb = rdb
            rdb_config = {
                'tickerplant': f'localhost:{self._ports["tickerplant"]}',
                'hdb': f'localhost:{self._ports["hdb"]}'}
            if self._database is not None:
                rdb_config['database'] = self._database
            self.rdb.start(rdb_config)
        except BaseException as err:
            self.tick.stop()
            if self.hdb is not None:
                self.hdb.stop()
            if self.rdb is not None:
                self.rdb.stop()
            raise err

    def stop(self):
        """
        Stop processing and kill all processes within the streaming workflow.
            This allows the port on which the process is deployed to be reclaimed
            and the process to be restarted if appropriate.

        Example:

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> basic = kx.tick.BASIC(
        ...     tables={'trade': trade},
        ...     database='/tmp/db',
        ...     log_directory='logs')
        >>> basic.start()
        >>> basic.stop()
        ```
        """
        self.tick.stop()
        if self.hdb is not None:
            self.hdb.stop()
        self.rdb.stop()

    def restart(self):
        """
        Restart and re-initialise a Basic streaming infrastructure, this will
           start the processes with the configuration initially supplied.

        Example:

        ```python
        >>> import pykx as kx
        >>> trade = kx.schema.builder({
        ...     'time': kx.TimespanAtom  , 'sym': kx.SymbolAtom,
        ...     'exchange': kx.SymbolAtom, 'sz': kx.LongAtom,
        ...     'px': kx.FloatAtom})
        >>> basic = kx.tick.BASIC(
        ...     tables={'trade': trade},
        ...     database='/tmp/db',
        ...     log_directory='logs')
        >>> basic.start()
        >>> basic.restart()
        ```
        """
        self.tick.restart()
        if self.hdb is not None:
            self.hdb.restart()
        self.rdb.restart()

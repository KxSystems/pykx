"""Functionality for the interaction with and management of databases.

!!! Warning

        This functionality is provided in it's present form as a BETA
        Feature and is subject to change. To enable this functionality
        for testing please following configuration instructions
        [here](../user-guide/configuration.md) setting `PYKX_BETA_FEATURES='true'`
"""

from .exceptions import QError
from . import wrappers as k
from . import beta_features
from .config import _check_beta
from .compress_encrypt import Compress, Encrypt

import os
from pathlib import Path
from typing import Union
from warnings import warn

__all__ = [
    'DB',
]

beta_features.append('Database Management')


def _init(_q):
    global q
    q = _q


def __dir__():
    return __all__


def _check_loading(cls, table, err_msg):
    if not cls.loaded:
        raise QError("No database referenced/loaded")
    if table is not None:
        if table not in cls.tables:
            raise QError(err_msg + " not possible as specified table not available")


def _check_column(cls, table, column):
    table_cols = cls.list_columns(table)
    if column not in table_cols:
        raise QError("Specified column '" + column + "' not present in table '" + table + "'")


_ktype_to_conversion = {
    k.GUIDAtom: "guid",
    k.BooleanAtom: "boolean",
    k.ByteAtom: "byte",
    k.ShortAtom: "short",
    k.IntAtom: "int",
    k.LongAtom: "long",
    k.RealAtom: "real",
    k.FloatAtom: "float",
    k.CharAtom: "char",
    k.SymbolAtom: "symbol",
    k.TimestampAtom: "timestamp",
    k.MonthAtom: "month",
    k.DateAtom: "date",
    k.DatetimeAtom: "datetime",
    k.TimespanAtom: "timespan",
    k.MinuteAtom: "minute",
    k.SecondAtom: "second",
    k.TimeAtom: "time",
}

_func_mapping = {
    'dpt': '{[d;p;f;t;s] .Q.dpt[d;p;t]}',
    'dpft': '{[d;p;f;t;s] .Q.dpft[d;p;f;t]}',
    'dpfs': '{[d;p;f;t;s] .Q.dpfs[d;p;f;s]}',
    'dpfts': '{[d;p;f;t;s] .Q.dpfts[d;p;f;t;s]}'
}


class _TABLES:
    pass


class DB(_TABLES):
    """Singleton class used for the management of kdb+ Databases"""
    _instance = None
    _init_tabs = None
    path = None
    tables = None
    table = _TABLES
    loaded = False

    def __new__(cls, *, path=None):
        if cls._instance is None:
            cls._instance = super(DB, cls).__new__(cls)
        return cls._instance

    def __init__(self, *, path=None):
        _check_beta('Database Management')
        if path is not None:
            try:
                self.load(path)
            except BaseException:
                self.path = Path(os.path.abspath(path))
        pass

    def create(self, table, table_name, partition, *, # noqa: C901
               by_field=None, sym_enum=None, log=True,
               compress=None, encrypt=None):
        """
        Create an on-disk partitioned table within a kdb+ database from a supplied
            `pykx.Table` object. Once generated this table will be accessible
            as an attribute of the `DB` class or a sub attribute of `DB.table`.

        Parameters:
            table: The `pykx.Table` object which is to be persisted to disk
            table_name: The name with which the table will be persisted and accessible
                once loaded and available as a `pykx.PartitionedTable`
            partition: The name of the column which is to be used to partition the data if
                supplied as a `str` or if supplied as non string object this will be used as
                the partition to which all data is persisted
            by_field: A field of the table to be used as a by column, this column will be
                the second column in the table (the first being the virtual column determined
                by the partitioning column)
            sym_enum: The name of the symbol enumeration table to be associated with the table
            log: Print information about status of partitioned datab
            compress: `pykx.Compress` initialized class denoting the
                compression settings to be used when persisting a partition/partitions
            encrypt: `pykx.Encrypt` initialized class denoting the encryption setting to be used
                when persisting a partition/partitions


        Returns:
            A `None` object on successful invocation, the database class will be
                updated to contain attributes associated with the available created table

        Examples:

        Generate a partitioned table from a table containing multiple partitions

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = '/tmp/newDB')
        >>> N = 1000
        >>> qtab = kx.Table(data = {
        ...     'date': kx.q.asc(kx.random.random(N, kx.q('2020.01 2020.02 2020.03m'))),
        ...     'sym': kx.random.random(N, ['AAPL', 'GOOG', 'MSFT']),
        ...     'price': kx.random.random(N, 10.0),
        ...     'size': kx.random.random(N, 100)
        ... })
        >>> db.create(qtab, 'stocks', 'date', by_field = 'sym', sym_enum = 'symbols')
        >>> db.tables
        ['stocks']
        >>> db.stocks
        pykx.PartitionedTable(pykx.q('
        month   sym  price     size
        ---------------------------
        2020.01 AAPL 7.979004  85
        2020.01 AAPL 5.931866  55
        2020.01 AAPL 5.255477  49
        2020.01 AAPL 8.15255   74
        2020.01 AAPL 4.771067  80
        ..
        '))
        ```

        Add a table as a partition to an on-disk database, in the example below we are adding
            a partition to the table generated above

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = '/tmp/newDB')
        >>> N = 333
        >>> qtab = kx.Table(data = {
        ...     'sym': kx.random.random(N, ['AAPL', 'GOOG', 'MSFT']),
        ...     'price': kx.random.random(N, 10.0),
        ...     'size': kx.random.random(N, 100)
        ... })
        >>> db.create(qtab, 'stocks', kx.q('2020.04m'), by_field = 'sym', sym_enum = 'symbols')
        >>> db.tables
        ['stocks']
        >>> db.stocks
        pykx.PartitionedTable(pykx.q('
        month   sym  price     size
        ---------------------------
        2020.01 AAPL 7.979004  85
        2020.01 AAPL 5.931866  55
        2020.01 AAPL 5.255477  49
        2020.01 AAPL 8.15255   74
        2020.01 AAPL 4.771067  80
        ..
        '))
        ```

        Add a table as a partition to an on-disk database, in the example below we are
            additionally applying gzip compression to the persisted table

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = '/tmp/newDB')
        >>> N = 333
        >>> qtab = kx.Table(data = {
        ...     'sym': kx.random.random(N, ['AAPL', 'GOOG', 'MSFT']),
        ...     'price': kx.random.random(N, 10.0),
        ...     'size': kx.random.random(N, 100)
        ... })
        >>> compress = kx.Compress(kx.CompressionAlgorithm.gzip, level=2)
        >>> db.create(qtab, 'stocks', kx.q('2020.04m'), compress=compress)
        >>> kx.q('{-21!hsym x}', '/tmp/newDB/2020.04/stocks/price')
        pykx.Dictionary(pykx.q('
        compressedLength  | 2064
        uncompressedLength| 2680
        algorithm         | 2i
        logicalBlockSize  | 17i
        zipLevel          | 2i
        '))
        ```
        """
        save_dir = self.path
        func_name = 'dpfts'
        if type(table) != k.Table:
            raise QError('Supplied table must be of type pykx.Table')
        if by_field is None:
            func_name = func_name.replace('f', '')
        if sym_enum is None:
            func_name = func_name.replace('s', '')
        compression_cache = q.z.zd
        if encrypt is not None:
            if not isinstance(encrypt, Encrypt):
                raise ValueError('Supplied encrypt object not an instance of pykx.Encrypt')
            if not encrypt.loaded:
                encrypt.load_key()
            if compress is None:
                compress = Compress()
        if compress is not None:
            if not isinstance(compress, Compress):
                raise ValueError('Supplied compress parameter is not a pykx.Compress object')
            compress.global_init(encrypt=encrypt)
        qfunc = q(_func_mapping[func_name])
        try:
            if type(partition) == str:
                if partition not in table.columns:
                    raise QError(f'Partition column {partition} not in supplied table')
                if type(table[partition]).t not in [5, 6, 7, 13, 14]:
                    raise QError(f'Unsupported type: {type(table[partition])} '
                                 'not supported for table partitioning')
                parts = q.distinct(table[partition])
                for i in parts:
                    if log:
                        print(f'Writing Database Partition {i} to table {table_name}')
                    q[table_name] = q('{?[x;enlist y;0b;()]}', table, [q('='), partition, i])
                    q[table_name] = q('{![x;();0b;enlist y]}', q[table_name], partition)
                    qfunc(save_dir, i, by_field, table_name, sym_enum)
            else:
                q[table_name] = table
                if log:
                    print(f'Writing Database Partition {partition} to table {table_name}')
                qfunc(save_dir, partition, by_field, table_name, sym_enum)
        except QError as err:
            q('{![`.;();0b;enlist x]}', table_name)
            q.z.zd = compression_cache
            raise QError(err)
        q('{![`.;();0b;enlist x]}', table_name)
        q.z.zd = compression_cache
        self.load(self.path, overwrite=True)
        return None

    def load(self, path: Union[Path, str], *, overwrite=False, encrypt=None):
        """
        Load the tables associated with a kdb+ Database, once loaded a table
            is accessible as an attribute of the `DB` class or a sub attribute
            of `DB.table`. Note that can alternatively be called when providing a path
            on initialisation of the DB class.

        Parameters:
            path: The file system path at which your database is located
            overwrite: Should loading of the database overwrite any currently
                loaded databases

        Returns:
            A `None` object on successful invocation, the database class will be
                updated to contain attributes associated with available tables

        Examples:

        Load an on-disk database

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testData')
        >>> db.tables
        ['testData']
        >>> db.testData
        pykx.PartitionedTable(pykx.q('
        month   sym  time         price    size
        ---------------------------------------
        2020.01 FDP  00:00:00.004 90.94738 12
        2020.01 FDP  00:00:00.005 33.81127 15
        2020.01 FDP  00:00:00.027 88.89853 16
        2020.01 FDP  00:00:00.035 78.33244 9
        2020.01 JPM  00:00:00.055 68.65177 1
        ..
        '))
        >>> db.table.testData
        pykx.PartitionedTable(pykx.q('
        month   sym  time         price    size
        ---------------------------------------
        2020.01 FDP  00:00:00.004 90.94738 12
        2020.01 FDP  00:00:00.005 33.81127 15
        2020.01 FDP  00:00:00.027 88.89853 16
        2020.01 FDP  00:00:00.035 78.33244 9
        2020.01 JPM  00:00:00.055 68.65177 1
        ..
        '))
        ```

        Load an on-disk database when initialising the class

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = 'testData')
        >>> db.tables
        ['testData']
        >>> db.testData
        pykx.PartitionedTable(pykx.q('
        month   sym  time         price    size
        ---------------------------------------
        2020.01 FDP  00:00:00.004 90.94738 12
        2020.01 FDP  00:00:00.005 33.81127 15
        2020.01 FDP  00:00:00.027 88.89853 16
        2020.01 FDP  00:00:00.035 78.33244 9
        2020.01 JPM  00:00:00.055 68.65177 1
        ..
        '))
        >>> db.table.testData
        pykx.PartitionedTable(pykx.q('
        month   sym  time         price    size
        ---------------------------------------
        2020.01 FDP  00:00:00.004 90.94738 12
        2020.01 FDP  00:00:00.005 33.81127 15
        2020.01 FDP  00:00:00.027 88.89853 16
        2020.01 FDP  00:00:00.035 78.33244 9
        2020.01 JPM  00:00:00.055 68.65177 1
        ..
        '))
        ```
        """
        load_path = Path(os.path.abspath(path))
        if not overwrite and self.path == load_path:
            raise QError("Attempting to reload existing database. Please pass "
                         "the keyword overwrite=True to complete database reload")
        if not overwrite and self.loaded:
            raise QError("Only one kdb+ database can be loaded within a process. "
                         "Please use the 'overwrite' keyword to load a new database.")
        if not load_path.is_dir():
            if load_path.is_file():
                err_info = 'Provided path is a file'
            else:
                err_info = 'Unable to find object at specified path'
            raise QError('Loading of kdb+ databases can only be completed on folders: ' + err_info)
        if encrypt is not None:
            if not isinstance(encrypt, Encrypt):
                raise ValueError('Supplied encrypt object not an instance of pykx.Encrypt')
            if not encrypt.loaded:
                encrypt.load_key()
        q('''
          {[dbpath]
            @[system"l ",;
              1_string dbpath;
              {'"Failed to load Database with error: ",x}
              ]
            }
          ''', load_path)
        self.path = load_path
        self.loaded = True
        self.tables = q.Q.pt.py()
        for i in self.tables:
            if hasattr(self, i):
                warn(f'A database table "{i}" would overwrite one of the pykx.DB() methods, please access your table via the table attribute') # noqa: E501
            else:
                setattr(self, i, q[i])
            setattr(self.table, i, q[i])
        return None

    def _reload(self):
        _check_loading(self, None, None)
        return self.load(self.path, overwrite=True)

    def rename_column(self, table, original_name, new_name):
        """
        Rename a column within a loaded kdb+ Database

        Parameters:
            table: The name of the table within which a column is to be renamed
            original_name: Name of the column which is to be renamed
            new_name: Column name which will be used as the new column name

        Returns:
            A `None` object on successful invocation, the database class will be
                updated and column rename actioned.

        Examples:

        Rename the column of a table

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> db.list_columns('testTable')
        ['month', 'sym', 'time', 'price', 'size']
        >>> db.rename_column('testTable', 'sym', 'symbol')
        >>> db.testTable
        pykx.PartitionedTable(pykx.q('
        month   symbol  time         price    size
        ---------------------------------------
        2020.01 FDP     00:00:00.004 90.94738 12
        2020.01 FDP     00:00:00.005 33.81127 15
        2020.01 FDP     00:00:00.027 88.89853 16
        2020.01 FDP     00:00:00.035 78.33244 9
        2020.01 JPM     00:00:00.055 68.65177 1
        ..
        '))
        ```
        """
        _check_loading(self, table, 'Column rename')
        _check_column(self, table, original_name)
        q.dbmaint.renamecol(self.path, table, original_name, new_name)
        self._reload()
        return None

    def delete_column(self, table, column):
        """
        Delete the column of a loaded kdb+ Database

        Parameters:
            table: The name of the table within which a column is to be deleted
            column: Column which is to be deleted from the database

        Returns:
            A `None` object on successful invocation, the database class will be
                updated and specified column deleted

        Examples:

        Delete the column of a table

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> db.list_columns('testTable')
        ['month', 'sym', 'time', 'price', 'size']
        >>> db.delete_column('testTable', 'size')
        >>> db.testTable
        pykx.PartitionedTable(pykx.q('
        month   symbol  time         price
        -------------------------------------
        2020.01 FDP     00:00:00.004 90.94738
        2020.01 FDP     00:00:00.005 33.81127
        2020.01 FDP     00:00:00.027 88.89853
        2020.01 FDP     00:00:00.035 78.33244
        2020.01 JPM     00:00:00.055 68.65177
        ..
        '))
        ```
        """
        _check_loading(self, table, 'Column deletion')
        _check_column(self, table, column)
        q.dbmaint.deletecol(self.path, table, column)
        self._reload()
        return None

    def rename_table(self, original_name, new_name):
        """
        Rename a table within a loaded kdb+ Database

        Parameters:
            original_name: The name of the table which is to be renamed
            new_name: Updated table name

        Returns:
            A `None` object on successful invocation, the database class will be
                updated, original table name deleted from q memory and new table
                accessible

        Examples:

        Rename a database table

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> db.rename_table('testTable', 'updated_table')
        >>> db.tables
        ['updated_table']
        ```
        """
        _check_loading(self, original_name, 'Table rename')
        q.dbmaint.rentable(self.path, original_name, new_name)
        # Remove the original table, without this it persists as an accessible table
        q('{![`.;();0b;enlist x]`}', original_name)
        self._reload()
        return None

    def list_columns(self, table):
        """
        List the columns of a table within a loaded kdb+ Database

        Parameters:
            table: The name of the table whose columns are listed

        Returns:
            A list of strings defining the columns of a table

        Examples:

        List the columns of a table in a database

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> db.list_columns('testTable')
        ['month', 'sym', 'time', 'price', 'size']
        ```
        """
        _check_loading(self, table, 'Column listing')
        return q.dbmaint.listcols(self.path, table).py()

    def add_column(self, table, column_name, default_value):
        """
        Add a column to a table within a loaded kdb+ Database

        Parameters:
            table: The name of the table to which a column is to be added
            column_name: Name of the column to be added
            default_value: The default value to be used for all existing partitions

        Returns:
            A `None` object on successful invocation, the database class will be
                updated and the new column available for use/access

        Examples:

        Add a column to a table within a partitioned database

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> db.list_columns('testTable')
        ['month', 'sym', 'time', 'price', 'size']
        >>>  db.add_column('testTable', 'test', kx.IntAtom.null)
        >>> db.list_columns('testTable')
        ['month', 'sym', 'time', 'price', 'size']
        ```
        """
        _check_loading(self, table, 'Column addition')
        q.dbmaint.addcol(self.path, table, column_name, default_value)
        self._reload()
        return(None)

    def find_column(self, table, column_name):
        """
        Functionality for finding a column across partitions within a loaded kdb+ Database

        Parameters:
            table: The name of the table within which columns are to be found
            column_name: The name of the column to be found within a table

        Returns:
            A `None` object on successful invocation printing search status per partition,
            if a column does not exist in a specified partition an error will be raised
            and the logs will indicate which columns did not have the specified column.

        Examples:

        Find a column that exists

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> db.list_columns('testTable')
        ['month', 'sym', 'time', 'price', 'size']
        >>> db.find_column('price')
        2023.11.10 16:48:57 column price (type 0) in `:/usr/pykx/db/2015.01.01/testTable
        2023.11.10 16:48:57 column price (type 0) in `:/usr/pykx/db/2015.01.02/testTable
        ```

        Attempt to find a column that does not exist

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> db.list_columns('testTable')
        ['month', 'sym', 'time', 'price', 'size']
        >>> db.find_column('side')
        2023.11.10 16:49:02 column side *NOT*FOUND* in `:/usr/pykx/db/2015.01.01/testTable
        2023.11.10 16:49:02 column side *NOT*FOUND* in `:/usr/pykx/db/2015.01.02/testTable
        Traceback (most recent call last):
        ...
        pykx.exceptions.QError: Requested column not found in all partitions, see log output above
        ```
        """
        _check_loading(self, table, 'Finding columns')
        return q.dbmaint.findcol(self.path, table, column_name).py()

    def reorder_columns(self, table, new_order):
        """
        Reorder the columns of a persisted kdb+ database

        Parameters:
            table: The name of the table within which columns will be rearranged
            new_order: The ordering of the columns following update

        Returns:
            A `None` object on successfully updating the columns of the database

        Examples:

        Update the order of columns for a persisted kdb+ database

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> col_list = db.list_columns('testTable')
        >>> col_list
        ['month', 'sym', 'time', 'price', 'size']
        >>> col_list.reverse()
        >>> col_list
        ['size', 'price', 'time', 'sym', 'month']
        >>> db.reorder_columns('testTable', col_list)
        2023.11.13 17:56:17 reordering columns in `:/usr/pykx/2015.01.01/testTable
        2023.11.13 17:56:17 reordering columns in `:/usr/pykx/2015.01.02/testTable
        ['month', 'sym', 'time', 'price', 'size']
        ```
        """
        _check_loading(self, table, 'Column reordering')
        q.dbmaint.reordercols(self.path, table, new_order)
        return None

    def set_column_attribute(self, table, column_name, new_attribute):
        """
        Set an attribute associated with a column for an on-disk database

        Parameters:
            table: The name of the table within which an attribute will be set
            column_name: Name of the column to which the attribute will be applied
            new_attribute: The attribute which is to be applied, this can be one of
                'sorted'/'u', 'partitioned'/'p', 'unique'/'u' or 'grouped'/'g'.

        Returns:
            A `None` object on successfully setting the attribute for a column

        Examples:

        Add an attribute to a column of a persisted database

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> kx.q.meta(db.testTable)
        pykx.KeyedTable(pykx.q('
        c   | t f a
        ----| -----
        date| d
        test| j
        p   | f
        sym | s
        '))
        >>> db.set_column_attribute('testTable', 'sym', 'grouped')
        >>> kx.q.meta(db.testTable)
        pykx.KeyedTable(pykx.q('
        c   | t f a
        ----| -----
        date| d
        test| j
        p   | f
        sym | s   g
        '))
        ```
        """
        _check_loading(self, table, 'Attribute setting')
        _check_column(self, table, column_name)
        if new_attribute not in ['s', 'g', 'p', 'u', 'sorted',
                                 'grouped', 'partitioned', 'unique']:
            raise QError("new_attribute must be one of "
                         "'s', 'g', 'p', 'u', 'sorted', 'grouped' or 'unique'")
        if new_attribute not in ['s', 'g', 'p', 'u']:
            new_attribute = {'sorted': 's',
                             'grouped': 'g',
                             'partitioned': 'p',
                             'unique': 'u'}[new_attribute]
        q.dbmaint.setattrcol(self.path, table, column_name, new_attribute)
        return None

    def set_column_type(self, table, column_name, new_type):
        """
        Convert/set the type of a column to a specified type

        Parameters:
            table: The name of the table within which a column is to be converted
            column_name: Name of the column which is to be converted
            new_type: PyKX type to which a column is to be converted

        Returns:
            A `None` object on successfully updating the type of the column

        Examples:

        Convert the type of a column within a database table

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> kx.q.meta(db.testTable)
        pykx.KeyedTable(pykx.q('
        c   | t f a
        ----| -----
        date| d
        test| j
        p   | f
        sym | s
        '))
        >>> db.set_column_type('testTable', 'test', kx.FloatAtom)
        >>> kx.q.meta(db.testTable)
        pykx.KeyedTable(pykx.q('
        c   | t f a
        ----| -----
        date| d
        test| f
        p   | f
        sym | s
        '))
        ```
        """
        _check_loading(self, table, 'Column casting')
        _check_column(self, table, column_name)
        if new_type not in _ktype_to_conversion:
            raise QError("Unable to find user specified conversion type: " + str(new_type))
        col_type = _ktype_to_conversion[new_type]
        try:
            q.dbmaint.castcol(self.path, table, column_name, col_type)
        except QError as err:
            if str(err) == 'type':
                raise QError("Unable to convert specified column '" + column_name + "' to type: " + str(new_type)) # noqa: E501
            raise QError(err)
        self._reload()
        return None

    def clear_column_attribute(self, table, column_name):
        """
        Clear an attribute associated with a column of an on-disk database

        Parameters:
            table: The name of the table within which the attribute of a column will be removed
            column_name: Name of the column from which an attribute will be removed

        Returns:
            A `None` object on successful removal of the attribute of a column

        Examples:

        Remove an attribute of a column of a persisted database

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.tables
        ['testTable']
        >>> kx.q.meta(db.testTable)
        pykx.KeyedTable(pykx.q('
        c   | t f a
        ----| -----
        date| d
        test| j
        p   | f
        sym | s    g
        '))
        >>> db.clear_column_attribute('testTable', 'sym')
        >>> kx.q.meta(db.testTable)
        pykx.KeyedTable(pykx.q('
        c   | t f a
        ----| -----
        date| d
        test| j
        p   | f
        sym | s
        '))
        ```
        """
        _check_loading(self, table, 'Attribute clearing')
        _check_column(self, table, column_name)
        q.dbmaint.clearattrcol(self.path, table, column_name)
        return None

    def copy_column(self, table, original_column, new_column):
        """
        Create a copy of a column within a table

        Parameters:
            table: Name of the table
            original_column: Name of the column to be copied
            new_column: Name of the copied column

        Returns:
            A `None` object on successful column copy, reloading the
            database following column copy

        Examples:

        Copy a column within a kdb+ database

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.list_columns('testTable')
        ['month', 'sym', 'time', 'price', 'size']
        >>> db.copy_column('testTable', 'size', 'dup_size')
        ['month', 'sym', 'time', 'price', 'size', 'dup_size']
        ```
        """
        _check_loading(self, table, 'Column copying')
        _check_column(self, table, original_column)
        q.dbmaint.copycol(self.path, table, original_column, new_column)
        self._reload()
        return None

    def apply_function(self, table, column_name, function):
        """
        Apply a function per partition on a column of a persisted kdb+ database

        Parameters:
            table: Name of the table
            column_name: Name of the column on which the function is to be applied
            function: Callable function to be applied on a column vector per column

        Returns:
            A `None` object on successful application of a function to the column
            and the reloading of the database

        Examples:

        Apply a q function to a specified column per partition

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.testTable
        pykx.PartitionedTable(pykx.q('
        month   symbol  time         price
        -------------------------------------
        2020.01 FDP     00:00:00.004 90.94738
        2020.01 FDP     00:00:00.005 33.81127
        2020.01 FDP     00:00:00.027 88.89853
        2020.01 FDP     00:00:00.035 78.33244
        2020.01 JPM     00:00:00.055 68.65177
        ..
        '))
        >>> db.apply_function('testTable', 'price', kx.q('2*'))
        >>> db.testTable
        pykx.PartitionedTable(pykx.q('
        month   symbol  time         price
        -------------------------------------
        2020.01 FDP     00:00:00.004 181.8948
        2020.01 FDP     00:00:00.005 67.62254
        2020.01 FDP     00:00:00.027 177.7971
        2020.01 FDP     00:00:00.035 156.6649
        2020.01 JPM     00:00:00.055 137.3035
        ..
        '))
        ```

        Apply a Python function to the content of a specified column per partition

        ```python
        >>> import pykx as kx
        >>> db = kx.DB()
        >>> db.load('testDB')
        >>> db.testTable
        pykx.PartitionedTable(pykx.q('
        month   symbol  time         price
        -------------------------------------
        2020.01 FDP     00:00:00.004 90.94738
        2020.01 FDP     00:00:00.005 33.81127
        2020.01 FDP     00:00:00.027 88.89853
        2020.01 FDP     00:00:00.035 78.33244
        2020.01 JPM     00:00:00.055 68.65177
        ..
        '))
        >>> db.apply_function('testTable', 'price', lambda x:2*x.np())
        >>> db.testTable
        pykx.PartitionedTable(pykx.q('
        month   symbol  time         price
        -------------------------------------
        2020.01 FDP     00:00:00.004 181.8948
        2020.01 FDP     00:00:00.005 67.62254
        2020.01 FDP     00:00:00.027 177.7971
        2020.01 FDP     00:00:00.035 156.6649
        2020.01 JPM     00:00:00.055 137.3035
        ..
        '))
        ```
        """
        _check_loading(self, table, 'Function application')
        _check_column(self, table, column_name)
        if not callable(function):
            raise RuntimeError("Provided 'function' is not callable")
        q.dbmaint.fncol(self.path, table, column_name, function)
        self._reload()
        return None

    def fill_database(self):
        """
        Fill missing tables from partitions within a database using the
            most recent partition as a template, this will report the
            partitions but not the tables which are being filled.

        Returns:
            A `None` object on successful filling of missing tables in
                partitioned database

        Examples:

        Fill missing tables from a database

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = 'newDB')
        >>> db.fill_database()
        Successfully filled missing tables to partition: :/usr/newDB/2020.04
        Successfully filled missing tables to partition: :/usr/newDB/2020.03
        Successfully filled missing tables to partition: :/usr/newDB/2020.02
        Successfully filled missing tables to partition: :/usr/newDB/2020.01
        ```
        """
        fill_parts = []
        try:
            fill_parts = q.raze(q.Q.chk(self.path)).py()
        except QError as err:
            if 'No such file or directory' in str(err):
                raise QError("Unable to apply database filling due to write permission issues") # noqa: E501
            raise QError(err)
        if 0<len(fill_parts):
            for i in fill_parts:
                print(f'Successfully filled missing tables to partition: {i}')
        self._reload()
        return None

    def partition_count(self, *, subview=None):
        """
        Count the number of rows per partition for the presently loaded database.
            Use of the parameter `subview` can allow users to count only the rows
            in specifies partitions.

        Parameters:
            subview: An optional list of partitions from which to retrieve the per partition
                count

        Returns:
            A `pykx.Dictionary` object showing the count of data in each table within
                the presently loaded partioned database.

        !!! Warning

                Using this function will result in any specified `subview` of the data being reset,
                if you require the use of a subview for queries please reset using the database
                `subview` command.

        Examples:

        Copy a column within a kdb+ database

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = 'newDB')
        >>> db.partition_count()
        pykx.Dictionary(pykx.q('
               | trades quotes
        -------| -------------
        2020.01| 334    0
        2020.02| 324    0
        2020.03| 342    1000
        '))
        ```

        Copy a column within a kdb+ database

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = 'newDB')
        >>> db.partition_count(sub_view = kx.q('2020.01 2020.02m'))
        pykx.Dictionary(pykx.q('
               | trades quotes
        -------| -------------
        2020.01| 334    0
        2020.02| 324    0
        '))
        ```
        """
        qtables = self.tables
        if subview==None: # noqa: E711
            q.Q.view()
        else:
            q.Q.view(subview)
        for i in qtables:
            q.Q.cn(getattr(self.table, i))
        res = q('.Q.pv!flip .Q.pn')
        q.Q.view()
        return res

    def subview(self, view=None):
        """
        Specify the subview to be used when querying a partitioned table

        Parameters:
            view: A list of partition values which will serve as a filter
                for all queries against any partitioned table within the
                database. If view is supplied as `None` this will reset
                the query view to all partitions

        Returns:
            A `None` object on successful setting of the view state

        Examples:

        Set the subview range to include only `2020.02` and `2020.03`

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = 'newDB')
        >>> db.subview(kx.q('2020.02 2020.03m')
        >>> kx.q.qsql.select(db.trades, 'month')
        pykx.Table(pykx.q('
        month
        -------
        2020.02
        2020.03
        '))
        ```

        Reset the database subview to include a fully specified range

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = 'newDB')
        >>> db.subview()
        >>> kx.q.qsql.select(db.trades, 'month')
        pykx.Table(pykx.q('
        month
        -------
        2020.01
        2020.02
        2020.03
        2020.04
        2020.05
        '))
        ```
        """
        if view==None: # noqa: E711
            q.Q.view()
        else:
            q.Q.view(view)
        return None

    def enumerate(self, table, *, sym_file=None):
        """
        Perform an enumeration on a user specified table against the
            current sym files associated with the database

        Parameters:
            path: The folder location to which your table will be persisted
            table: The `pykx.Table` object which is to be persisted to disk
                and which is to undergo enumeration
            sym_file: The name of the sym file contained in the folder specified by
                the `path` parameter against which enumeration will be completed

        Returns:
            The supplied table with enumeration applied

        Examples:

        Enumerate the symbol columns of a table without specifying the `sym` file

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = 'newDB')
        >>> N = 1000
        >>> tab = kx.Table(data = {
        ...     'x': kx.random.random(N, ['a', 'b', 'c']),
        ...     'x1': kx.random.random(N, 1.0),
        ...     'x2': kx.random.random(N, 10)
        ... }
        >>> tab = db.enumerate(tab)
        >>> tab['x']
        pykx.EnumVector(pykx.q('`sym$`a`b`a`c`b..'))
        ```

        Enumerate the symbol columns of a table specifying the `sym` file used

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = 'newDB')
        >>> N = 1000
        >>> tab = kx.Table(data = {
        ...     'x': kx.random.random(N, ['a', 'b', 'c']),
        ...     'x1': kx.random.random(N, 1.0),
        ...     'x2': kx.random.random(N, 10)
        ... }
        >>> tab = db.enumerate(tab, sym_file = 'mysym')
        >>> tab['x']
        pykx.EnumVector(pykx.q('`mysym$`a`b`a`c`b..'))
        ```
        """
        load_path = Path(os.path.abspath(self.path))
        if sym_file is None:
            return q.Q.en(load_path, table)
        else:
            return q.Q.ens(load_path, table, sym_file)

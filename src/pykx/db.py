"""
_This page documents the API for managing kdb+ databases using PyKX._
"""

from .exceptions import DBError, QError
from . import wrappers as k
from .config import pykx_4_1
from .compress_encrypt import Compress, Encrypt

import os
from pathlib import Path
from typing import Any, Optional, Union
from warnings import warn

__all__ = [
    'DB',
]


def _init(_q):
    global q
    q = _q


def __dir__():
    return __all__


def _check_loading(cls, table, err_msg):
    if not cls.loaded:
        raise QError("No database referenced/loaded")
    if (table is not None) and (table not in cls.tables):
        raise QError(err_msg + " not possible as specified table not available")


def _get_type(cls, table):
    type_str = str(type(getattr(cls, table)))
    if 'SplayedTable' in type_str:
        return 'splayed'
    elif 'PartitionedTable' in type_str:
        return 'partitioned'
    else:
        raise QError(f'Unsupported type {type_str} passed to _get_type')


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
    'dpts': '{[d;p;f;t;s] .Q.dpts[d;p;t;s]}',
    'dpfts': '{[d;p;f;t;s] .Q.dpfts[d;p;f;t;s]}'
}


class _TABLES:
    pass


class DB(_TABLES):
    _instance = None
    _init_tabs = None
    _dir_cache = None
    _change_dir = True
    _load_script = True
    path = None
    tables = None
    table = _TABLES
    loaded = False

    def __new__(cls,
                *,
                path: Optional[Union[str, Path]] = None,
                change_dir: Optional[bool] = True,
                load_scripts: Optional[bool] = True
    ) -> None:
        if cls._dir_cache is None:
            cls._dir_cache = dir(cls)
        if cls._instance is None:
            cls._instance = super(DB, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 *,
                 path: Optional[Union[str, Path]] = None,
                 change_dir: Optional[bool] = True,
                 load_scripts: Optional[bool] = True
    ) -> None:
        """
        Initialize a database class used within your process. This is a singleton class from
            which all interactions with your database will be made. On load if supplied
            with a 'path' this functionality will attempt to load the database at this location.
            If no database exists at this location the path supplied will be used when a new
            database is created.

        Parameters:
            path: The location at which your database is/will be located.
            change_dir: Should the working directory be changed to the location of the
                loaded database, for q 4.0 this is the only supported behavior, please
                set `PYKX_4_1_ENABLED` to allow use if this functionality.
            load_scripts: Should any q scripts find in the database directory be loaded,
                for q 4.0 this is the only supported behavior, please set
                `PYKX_4_1_ENABLED` to allow use if this functionality.

        Returns:
            A database class which can be used to interact with a partitioned database.

        Examples:

        Load a partitioned database at initialization

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = '/tmp/db')
        >>> db.tables
        ['quote', 'trade']
        ```

        Define the path to be used for a database which does not initially exist

        ```python
        >>> import pykx as kx
        >>> db = kx.DB(path = 'db')
        >>> db.tables
        >>> db.path
        PosixPath('/usr/projects/pykx/db')
        ```
        """
        self._change_dir = change_dir
        self._load_scripts = load_scripts
        if not pykx_4_1:
            if not change_dir:
                raise QError("'change_dir' behavior only supported with PYKX_4_1_ENABLED")
            if not load_scripts:
                raise QError("'load_scripts' behavior only supported with PYKX_4_1_ENABLED")
        if path is not None:
            try:
                self.load(path, change_dir=self._change_dir, load_scripts=self._load_scripts)
            except DBError:
                self.path = Path(os.path.abspath(path))

    def create(self,
               table: k.Table,
               table_name: str,
               partition: Union[int, str, k.DateAtom] = None,
               *, # noqa: C901
               format: Optional[str] = 'partitioned',
               by_field: Optional[str] = None,
               sym_enum: Optional[str] = None,
               log: Optional[bool] = True,
               compress: Optional[Compress] = None,
               encrypt: Optional[Encrypt] = None,
               change_dir: Optional[bool] = True,
               load_scripts: Optional[bool] = True
    ) -> None:
        """
        Create an on-disk partitioned table within a kdb+ database from a supplied
            `#!python pykx.Table` object. Once generated this table will be accessible
            as an attribute of the `#!python DB` class or a sub attribute of `#!python DB.table`.

        Parameters:
            table: The `#!python pykx.Table` object which is to be persisted to disk
            table_name: The name with which the table will be persisted and accessible
                once loaded and available as a `#!python pykx.PartitionedTable`
            partition: The name of the column which is to be used to partition the data if
                supplied as a `#!python str` or if supplied as non string object this is
                used as the partition to which all data is persisted.
            format: Is the table that's being created a 'splayed' or 'partitioned' table
            by_field: A field of the table to be used as a by column, this column will be
                the second column in the table (the first being the virtual column determined
                by the partitioning column)
            sym_enum: The name of the symbol enumeration table to be associated with the table
            log: Print information about status while persisting the partitioned database
            compress: `#!python pykx.Compress` initialized class denoting the compression settings
                to be used when persisting a partition/partitions
            encrypt: `#!python pykx.Encrypt` initialized class denoting the encryption setting
                to be used when persisting a partition/partitions


        Returns:
            A `#!python None` object on successful invocation, the database class is
                updated to contain attributes associated with the available created table

        Examples:

        Generate a partitioned database from a table containing multiple partitions.

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

        Add a table as a partition to an on-disk database.

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

        Add a table as a partition to an on-disk database and apply gzip
            compression to the persisted table

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
        table = k.toq(table)
        if type(table) != k.Table:
            raise QError('Supplied table must be of type "pykx.Table" or can be converted to this type') # noqa: E501
        if format not in ['splayed', 'partitioned']:
            raise QError("'format' must be one of 'splayed'/'partitioned', supplied value: "
                         f'{format}')
        if (format == 'partitioned') & (partition == None): # noqa: E711
            raise QError("Creation of partitioned format table requires a supplied "
                         "'partition' parameter, currently set as default None")
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
            if format == 'splayed':
                table = q.Q.en(save_dir, table)
                q('{.Q.dd[x;`] set y}', save_dir/table_name, table)
            else:
                if isinstance(partition, str):
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
        if change_dir is None:
            change_dir = self._change_dir
        if load_scripts is None:
            load_scripts = self._load_scripts
        self.load(self.path, overwrite=True, change_dir=change_dir, load_scripts=load_scripts)
        return None

    def load(self,
             path: Union[Path, str],
             *,
             change_dir: Optional[bool] = True,
             load_scripts: Optional[bool] = True,
             overwrite: Optional[bool] = False,
             encrypt: Optional[Encrypt] = None
    ) -> None:
        """
        Load the tables associated with a kdb+ database. Once loaded, a table
            is accessible as an attribute of the `#!python DB` class or a sub-attribute
            of `#!python DB.table`. Note this can alternatively be called when providing a path
            on initialisation of the DB class.

        Parameters:
            path: The file system path at which your database is located
            change_dir: Should the working directory be changed to the location of the
                loaded database, for q 4.0 this is the only supported behavior, please
                set `PYKX_4_1_ENABLED` to allow use if this functionality.
            load_scripts: Should any q scripts find in the database directory be loaded,
                for q 4.0 this is the only supported behavior, please set
                `PYKX_4_1_ENABLED` to allow use if this functionality.
            overwrite: Should loading of the database overwrite any currently
                loaded databases
            encrypt: The encryption key object to be loaded prior to database load

        Returns:
            A `#!python None` object on successful invocation, the database class is
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
            raise DBError("Attempting to reload existing database. Please pass "
                          "the keyword overwrite=True to complete database reload")
        if not overwrite and self.loaded:
            raise DBError("Only one kdb+ database can be loaded within a process. "
                          "Please use the 'overwrite' keyword to load a new database.")
        if not load_path.is_dir():
            if load_path.is_file():
                err_info = 'Provided path is a file'
            else:
                err_info = 'Unable to find object at specified path'
            raise DBError('Loading of kdb+ databases can only be completed on folders: ' + err_info)
        if encrypt is not None:
            if not isinstance(encrypt, Encrypt):
                raise ValueError('Supplied encrypt object not an instance of pykx.Encrypt')
            if not encrypt.loaded:
                encrypt.load_key()
        if pykx_4_1:
            q('''
              {[path;cd;ld]
                .[.Q.lo;
                  (`$1_string path;cd;ld);
                  {x:$[x like "*.DS_Store";
                       "Invalid MacOS metadata file '.DS_Store' stored in Database";
                       x];
                   '"Failed to load Database with error: ",x
                  }
                  ]
                }
              ''', load_path, change_dir, load_scripts)
        else:
            if not change_dir:
                raise QError("'change_dir' behavior only supported with PYKX_4_1_ENABLED")
            if not load_scripts:
                raise QError("'load_scripts' behavior only supported with PYKX_4_1_ENABLED")
            db_path = load_path.parent
            db_name = os.path.basename(load_path)
            q('''
              {[dbpath;dbname]
                .[.pykx.util.loadfile;
                  (1_string dbpath;string dbname);
                  {x:$[x like "*.DS_Store";
                       "Invalid MacOS metadata file '.DS_Store' stored in Database";
                       x];
                   '"Failed to load Database with error: ",x
                  }
                  ]
                }
              ''', db_path, db_name)
        self.path = load_path
        self.loaded = True
        self.tables = q('{x where {$[-1h=type t:.Q.qp tab:get x;$[t;1b;in[`$last vs["/";-1_string value flip tab]; key y]];0b]}[;y]each x}', q.tables(), load_path).py() # noqa: E501
        for i in self.tables:
            if i in self._dir_cache:
                warn(f'A database table "{i}" would overwrite one of the pykx.DB() methods, please access your table via the table attribute') # noqa: E501
            else:
                setattr(self, i, q[i])
            setattr(self.table, i, q[i])
        return None

    def _reload(self):
        _check_loading(self, None, None)
        return self.load(self.path,
                         overwrite=True,
                         change_dir=self._change_dir,
                         load_scripts=self._load_scripts)

    def rename_column(self,
                      table: str,
                      original_name: str,
                      new_name: str
    ) -> None:
        """
        Rename a column within a loaded kdb+ Database

        Parameters:
            table: The name of the table containing the column to be renamed
            original_name: Name of the column which is to be renamed
            new_name: Updated column name

        Returns:
            A `#!python None` object on successful invocation, the database class is```
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
        >>> db.list_columns('testTable')
        ['month', 'symbol', 'time', 'price', 'size']
        ```
        """
        _check_loading(self, table, 'Column rename')
        _check_column(self, table, original_name)
        table_type = _get_type(self, table)
        if table_type == 'splayed':
            q.dbmaint.rename1col(self.path / table, original_name, new_name)
        else:
            q.dbmaint.renamecol(self.path, table, original_name, new_name)
        self._reload()
        return None

    def delete_column(self, table: str, column: str) -> None:
        """
        Delete a column from a loaded kdb+ Database.

        Parameters:
            table: The name of the table containing the column to be deleted
            column: Name of the column which is to be deleted from the table

        Returns:
            A `#!python None` object on successful invocation, the database class is
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
        >>> db.list_columns('testTable')
        ['month', 'sym', 'time', 'price']
        ```
        """
        _check_loading(self, table, 'Column deletion')
        _check_column(self, table, column)
        table_type = _get_type(self, table)
        if 'splayed' == table_type:
            q.dbmaint.delete1col(self.path / table, column)
        else:
            q.dbmaint.deletecol(self.path, table, column)
        self._reload()
        return None

    def rename_table(self, original_name: str, new_name: str) -> None:
        """
        Rename a table within a loaded kdb+ Database

        Parameters:
            original_name: The name of the table which is to be renamed
            new_name: Updated table name

        Returns:
            A `#!python None` object on successful invocation, the database class is
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
        table_type = _get_type(self, original_name)
        if 'splayed' == table_type:
            q.dbmaint.ren1table(self.path / original_name, self.path / new_name)
        else:
            q.dbmaint.rentable(self.path, original_name, new_name)
        # Remove the original table, without this it persists as an accessible table
        q('{![`.;();0b;enlist x]`}', original_name)
        self._reload()
        return None

    def list_columns(self, table: str) -> None:
        """
        List the columns of a table within a loaded kdb+ Database

        Parameters:
            table: The name of the table whose columns are to be listed

        Returns:
            A list of strings defining the columns of the table

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
        table_type = _get_type(self, table)
        if table_type == 'splayed':
            return q('get', self.path / table / '.d').py()
        return q.dbmaint.listcols(self.path, table).py()

    def add_column(self,
                   table: str,
                   column_name: str,
                   default_value: Any
    ) -> None:
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

        Add a column to a table within a partitioned database where all items are
            an integer null

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
        table_type = _get_type(self, table)
        if table_type == 'splayed':
            q.dbmaint.add1col(self.path / table, column_name, default_value)
        else:
            q.dbmaint.addcol(self.path, table, column_name, default_value)
        self._reload()
        return(None)

    def find_column(self, table: str, column_name: str) -> None:
        """
        Functionality for finding a column across partitions within a loaded kdb+ Database

        Parameters:
            table: The name of the table within which columns are to be found
            column_name: The name of the column to be found within a table

        Returns:
            A `#!python None` object on successful invocation printing search status per partition.
            If a column does not exist in a specified partition, an error is raised
            and the logs indicate which columns did not contain the specified column.

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
        >>> db.find_column('testTable', 'price')
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
        >>> db.find_column('testTable', 'side')
        2023.11.10 16:49:02 column side *NOT*FOUND* in `:/usr/pykx/db/2015.01.01/testTable
        2023.11.10 16:49:02 column side *NOT*FOUND* in `:/usr/pykx/db/2015.01.02/testTable
        Traceback (most recent call last):
        ...
        pykx.exceptions.QError: Requested column not found in all partitions, see log output above
        ```
        """
        _check_loading(self, table, 'Finding columns')
        table_type = _get_type(self, table)
        if 'splayed' == table_type:
            return q.dbmaint.find1col(self.path / table, column_name).py()
        return q.dbmaint.findcol(self.path, table, column_name).py()

    def reorder_columns(self, table: str, new_order: list) -> None:
        """
        Reorder the columns of a persisted kdb+ database

        Parameters:
            table: The name of the table within which columns will be rearranged
            new_order: The ordering of the columns following update

        Returns:
            A `#!python None` object on successfully updating the columns of the database

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
        table_type = _get_type(self, table)
        if 'splayed' == table_type:
            q.dbmaint.reordercols0(self.path / table, new_order)
        else:
            q.dbmaint.reordercols(self.path, table, new_order)
        self._reload()
        return None

    def set_column_attribute(self, table: str, column_name: str, new_attribute: str) -> None:
        """
        Set an attribute associated with a column for an on-disk database

        Parameters:
            table: The name of the table within which an attribute will be set
            column_name: Name of the column to which the attribute will be applied
            new_attribute: The attribute which is to be applied, this can be one of
                `#!python 'sorted'`/`#!python 's'`, `#!python 'partitioned'`/`#!python 'p'`,
                `#!python 'unique'`/`#!python 'u'` or `#!python 'grouped'`/`#!python 'g'`.

        Returns:
            A `#!python None` object on successfully setting the attribute for a column

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
        table_type = _get_type(self, table)
        if new_attribute not in ['s', 'g', 'p', 'u', 'sorted',
                                 'grouped', 'partitioned', 'unique']:
            raise QError("new_attribute must be one of "
                         "'s', 'g', 'p', 'u', 'sorted', 'grouped' or 'unique'")
        if new_attribute not in ['s', 'g', 'p', 'u']:
            new_attribute = {'sorted': 's',
                             'grouped': 'g',
                             'partitioned': 'p',
                             'unique': 'u'}[new_attribute]
        if 'splayed' == table_type:
            q.dbmaint.fn1col(self.path / table, column_name, q('{x#y}', new_attribute))
        else:
            q.dbmaint.setattrcol(self.path, table, column_name, new_attribute)
        self._reload()
        return None

    def set_column_type(self, table: str, column_name: str, new_type: k.K) -> None:
        """
        Convert/set the type of a column to a specified type

        Parameters:
            table: The name of the table within which a column is to be converted
            column_name: Name of the column which is to be converted
            new_type: PyKX type to which a column is to be converted

        Returns:
            A `#!python None` object on successfully updating the type of the column

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
        table_type = _get_type(self, table)
        try:
            if table_type == 'splayed':
                q.dbmaint.fn1col(self.path / table, column_name, q('{x$y}', col_type))
            else:
                q.dbmaint.castcol(self.path, table, column_name, col_type)
        except QError as err:
            if str(err) == 'type':
                raise QError("Unable to convert specified column '" + column_name + "' to type: " + str(new_type)) # noqa: E501
            raise QError(err)
        self._reload()
        return None

    def clear_column_attribute(self, table: str, column_name: str) -> None:
        """
        Clear an attribute associated with a column of an on-disk database

        Parameters:
            table: The name of the table within which the attribute of a column will be removed
            column_name: Name of the column from which an attribute will be removed

        Returns:
            A `#!python None` object on successful removal of the attribute of a column

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
        table_type = _get_type(self, table)
        if 'splayed' == table_type:
            q.dbmaint.fn1col(self.path / table, column_name, q('`#'))
        q.dbmaint.clearattrcol(self.path, table, column_name)
        return None

    def copy_column(self, table: str, original_column: str, new_column: str) -> None:
        """
        Create a copy of a column within a table

        Parameters:
            table: Name of the table
            original_column: Name of the column to be copied
            new_column: Name of the copied column

        Returns:
            A `#!python None` object on successful column copy, reloading the
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
        table_type = _get_type(self, table)
        if 'splayed' == table_type:
            q.dbmaint.copy1col(self.path / table, original_column, new_column)
        else:
            q.dbmaint.copycol(self.path, table, original_column, new_column)
        self._reload()
        return None

    def apply_function(self, table: str, column_name: str, function: callable) -> None:
        """
        Apply a function per partition on a column of a persisted kdb+ database

        Parameters:
            table: Name of the table
            column_name: Name of the column on which the function is to be applied
            function: Callable function to be applied on a column vector per partition

        Returns:
            A `#!python None` object on successful application of a function to the column
            and reloading of the database

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
        table_type = _get_type(self, table)
        if 'splayed' == table_type:
            q.dbmaint.fn1col(self.path / table, column_name, function)
        else:
            q.dbmaint.fncol(self.path, table, column_name, function)
        self._reload()
        return None

    def fill_database(self) -> None:
        """
        Fill missing tables from partitions within a database using the
            most recent partition as a template, this will report the
            partitions but not the tables which are being filled.

        Returns:
            A `#!python None` object on successful filling of missing tables in
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

    def partition_count(self, *, subview: Optional[list] = None) -> k.Dictionary:
        """
        Count the number of rows per partition for the presently loaded database.
            Use of the parameter `#!python subview` can allow users to count only the rows
            in specified partitions.

        Parameters:
            subview: An optional list of partitions from which to retrieve the per partition
                count

        Returns:
            A `#!python pykx.Dictionary` object showing the count of data in each table within
                the presently loaded partioned database.

        !!! Warning

                Using this function results in any specified `#!python subview` of the data
                being reset, if you require the use of a subview for queries please set using
                the database `#!python subview` command.

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
        cache = None
        try:
            cache = q.Q.pv
        except QError:
            pass
        if subview==None: # noqa: E711
            q.Q.view()
        else:
            q.Q.view(subview)
        for i in qtables:
            tab = getattr(self.table, i)
            if isinstance(tab, k.PartitionedTable):
                q.Q.cn(tab)
        res = q('.Q.pv!flip .Q.pn')
        q.Q.view(cache)
        return res

    def subview(self, view: list = None) -> None:
        """
        Specify the subview to be used when querying a partitioned table

        Parameters:
            view: A list of partition values which will serve as a filter
                for all queries against any partitioned table within the
                database. If view is supplied as `#!python None` this resets
                the query view to all partitions

        Returns:
            A `#!python None` object on successful setting of the view state

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

    def enumerate(self, table: str, *, sym_file: Optional[str] = None) -> k.Table:
        """
        Perform an enumeration on a user specified table against the
            current sym files associated with the database

        Parameters:
            table: The `#!python pykx.Table` object which is to be persisted to disk
                and which is to undergo enumeration
            sym_file: The name of the sym file contained in the folder specified by
                the `#!python path` parameter against which enumeration will be completed

        Returns:
            The supplied table with enumeration applied

        Examples:

        Enumerate the symbol columns of a table without specifying the `#!python sym_file`

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

        Enumerate the symbol columns of a table specifying the `#!python sym_file`

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

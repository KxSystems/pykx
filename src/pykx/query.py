"""_This page documents query interfaces for querying q tables using PyKX._"""

from abc import ABCMeta
import asyncio
from typing import Any, Dict, List, Optional, Union

from . import Q
from . import wrappers as k
from .ipc import QFuture
from .exceptions import PyKXException, QError

__all__ = [
    'Insert',
    'QSQL',
    'SQL',
]


def __dir__():
    return __all__


class QSQL:
    """The `#!python QSQL` class provides methods to query or modify q tables.

    The methods [select][pykx.QSQL.select], [exec][pykx.QSQL.exec], [update][pykx.QSQL.update]
    and [delete][pykx.QSQL.delete] generate and execute functional queries on the given table.
    To learn about functionally querying databases see [Chapter 9 Section 12 of Q for
    Mortals](https://code.kx.com/q4m3/9_Queries_q-sql/#912-functional-forms).

    There are a number of advantages to using this query style as opposed to interpolating
    strings to generate simple qSQL queries:

    1. Users that are unfamiliar with q who use the interface are introduced to this more powerful
        version of querying with q, while still operating within a familiar setting in Python.
    2. Using the functional form promotes data-oriented designs for modifying or querying the q
        tables programmatically using data derived from Python:

        ```python
        qtable = pykx.q('([]1 2 3;4 5 6)')
        pykx.q.qsql.update(qtable, {'x': [10, 20, 30]})
        ```

    3. Development and maintenance of this interface is easier with regard to the different
        supported table formats.
    """

    def __init__(self, q: Q):
        self._q = q

    def select(self,
               table: Union[k.Table, str],
               columns: Optional[Union[Dict[str, str], k.Dictionary]] = None,
               where: Optional[Union[List[str], str, k.SymbolAtom, k.SymbolVector]] = None,
               by: Optional[Union[Dict[str, str], k.Dictionary]] = None,
               inplace: bool = False,
    ) -> k.K:
        """
        Execute a q functional select statement on tables defined within the process.

        This implementation follows the q functional select syntax with limited support
        on structures used in the parameters.

        Parameters:
            table: The q table or name of the table to query. The table must be named within
                the q memory space.
            columns: A dictionary where the keys are names assigned for the query's output columns
                and the values are the logic used to compute the column's result.
            where: Filtering logic for reducing the data used in group-bys and
                output column aggregations.
            by: A dictionary where they keys are names assigned for the produced columns and the
                values are aggregation rules used to construct the group-by parameter.
            inplace: Indicates if the result of an update is to be persisted. This applies to
                tables referenced by name in q memory or general table objects.
                See [here](https://code.kx.com/q/basics/qsql/#result-and-side-effects).

        Returns:
            A PyKX Table or KeyedTable object resulting from the executed select query

        Examples:

        Define a q table in python, and give it a name in q memory

        ```python
        qtab = pykx.q('([]col1:100?`a`b`c;col2:100?1f;col3:100?0b;col4:100?10f)')
        pykx.q['qtab'] = qtab
        ```

        Select all items in the table

        ```python
        pykx.q.qsql.select(qtab)
        pykx.q.qsql.select('qtab')
        ```

        Filter table based on various where conditions

        ```python
        pykx.q.qsql.select(qtab, where='col2<0.5')
        pykx.q.qsql.select(qtab, where=['col1=`a', 'col2<0.3'])
        ```

        Retrieve statistics by grouping data on symbol columns

        ```python
        pykx.q.qsql.select(qtab, columns={'maxCol2': 'max col2'}, by={'col1': 'col1'})
        pykx.q.qsql.select(qtab, columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'})
        ```

        Retrieve grouped statistics with restrictive where condition

        ```python
        pykx.q.qsql.select(qtab, columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'}, where='col3=0b')
        ```
        """ # noqa: E501
        return self._seud(table, 'select', columns, where, by, inplace=inplace)

    def exec(self,
             table: Union[k.Table, str],
             columns: Optional[Union[Dict[str, str], k.Dictionary]] = None,
             where: Optional[Union[List[str], str, k.SymbolAtom, k.SymbolVector]] = None,
             by: Optional[Union[Dict[str, str], k.Dictionary]] = None
    ) -> k.K:
        """
        Execute a q functional exec statement on tables defined within the process.

        This implementation follows the q functional exec syntax with limited support on
        structures used for the parameters.

        Parameters:
            table: The q table or name of the table to query. The table must be named within
                the q memory space.
            columns: A dictionary where the keys are names assigned to the query's output columns
                and the values are the logic used to compute the column's result.
            where: Filtering logic for reducing the data used in group-by and
                output column aggregations.
            by: A dictionary where they keys are names assigned to the produced columns and the
                values are aggregation rules used when q functionally applies group-by.

        Returns:
            A PyKX Vector or Dictionary object resulting from the executed exec query

        Examples:

        Define a q table in python and named in q memory

        ```python
        qtab = pykx.Table(data={
            'col1': pykx.random.random(100, ['a', 'b', 'c']),
            'col2': pykx.random.random(100, 100),
            'col3': pykx.random.random(100, [0, 1]),
            'col4': pykx.random.random(100, 100.0)
            })
        pykx.q['qtab'] = qtab
        ```

        Select last item of the table

        ```python
        pykx.q.qsql.exec(qtab)
        pykx.q.qsql.exec('qtab')
        ```

        Retrieve a column from the table as a list

        ```python
        pykx.q.qsql.exec(qtab, 'col3')
        ```

        Retrieve a set of columns from a table as a dictionary

        ```python
        pykx.q.qsql.exec(qtab, {'symcol': 'col1'})
        pykx.q.qsql.exec(qtab, {'symcol': 'col1', 'boolcol': 'col3'})
        ```

        Filter columns from a table based on various where conditions

        ```python
        pykx.q.qsql.exec(qtab, 'col3', where='col1=`a')
        pykx.q.qsql.exec(qtab, {'symcol': 'col1', 'maxcol4': 'max col4'}, where=['col1=`a', 'col2<0.3'])
        ```

        Retrieve data grouping by data on symbol columns

        ```python
        pykx.q.qsql.exec(qtab, 'col2', by={'col1': 'col1'})
        pykx.q.qsql.exec(qtab, columns={'maxCol2': 'max col2'}, by={'col1': 'col1'})
        pykx.q.qsql.exec(qtab, columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'})
        ```

        Retrieve grouped statistics with restrictive where condition

        ```python
        pykx.q.qsql.exec(qtab, columns={'avgCol2': 'avg col2', 'minCol4': 'min col4'}, by={'col1': 'col1'}, where='col3=0b')
        ```
        """ # noqa: E501
        return self._seud(table, 'exec', columns, where, by)

    def update(self,
               table: Union[k.Table, str],
               columns: Optional[Union[Dict[str, str], k.Dictionary]] = None,
               where: Optional[Union[List[str], str, k.SymbolAtom, k.SymbolVector]] = None,
               by: Optional[Union[Dict[str, str], k.Dictionary]] = None,
               inplace: bool = False,
    ) -> k.K:
        """
        Execute a q style update statement on tables defined within the process.

        This implementation follows the q functional update syntax with limited support on
        structures used for the parameters.

        Parameters:
            table: The q table or name of the table to update. The table must be named within
                the q memory space.
            columns: A dictionary where the keys are names assigned to the query's output columns
                and the values are the logic used to compute the column's result.
            where: Filtering logic for reducing the data used in group-bys and
                output column aggregations.
            by: A dictionary where they keys are names assigned to the result columns and the
                values are aggregation rules used to compute the group-by result.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects
                https://code.kx.com/q/basics/qsql/#result-and-side-effects.

        Returns:
            The updated PyKX Table or KeyedTable object resulting from the executed update query

        Examples:

        Define a q table in python and named in q memory

        ```python
        qtab = pykx.Table(
            [
                ['tom', 28, 'fair', 'green'],
                ['dick', 29, 'dark', 'brown'],
                ['harry', 35, 'fair', 'gray']
            ],
            columns=['name', 'age', 'hair', 'eye']
        )
        pykx.q['qtab'] = qtab
        ```

        Update all the contents of a column

        ```python
        pykx.q.qsql.update(qtab, {'eye': '`blue`brown`green'})
        pykx.q.qsql.update(qtab, {'age': [25, 30, 31]})
        ```

        Update the content of a column restricting scope using a where clause

        ```python
        pykx.q.qsql.update(qtab, {'eye': ['blue']}, where='hair=`fair')
        ```

        Define a q table suitable for by clause example

        ```python
        byqtab = pykx.q('([]p:100?`a`b`c;name:100?`nut`bolt`screw;color:100?`red`green`blue;weight:0.5*100?20;city:100?`london`paris`rome)')
        pykx.q['byqtab'] = byqtab
        ```

        Apply an update grouping based on a by phrase

        ```python
        pykx.q.qsql.update(byqtab, {'weight': 'avg weight'}, by={'city': 'city'})
        ```

        Apply an update grouping based on a by phrase and persist the result using the inplace keyword

        ```python
        pykx.q.qsql.update('byqtab', columns={'weight': 'avg weight'}, by={'city': 'city'}, inplace=True)
        pykx.q['byqtab']
            ```
        """ # noqa: E501
        return self._seud(table, 'update', columns, where, by, inplace)

    def delete(self,
               table: Union[k.Table, str],
               columns: Optional[Union[List[str], k.SymbolVector]] = None,
               where: Optional[Union[List[str], str, k.SymbolAtom, k.SymbolVector]] = None,
               inplace: bool = False,
    ) -> k.K:
        """
        Execute a q functional delete statement on tables defined within the process.

        This implementation follows the q functional delete syntax with limited support on
        structures used for the parameters.

        Parameters:
            table: The q table or name of the table (provided the table is named within the q
                memory space) on which the delete statement is to be applied.
            columns: Denotes the columns to be deleted from a table.
            where: Conditional filtering used to select subsets of the data which are to be
                deleted from the table.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects
                https://code.kx.com/q/basics/qsql/#result-and-side-effects.

        Returns:
            The updated PyKX Table or KeyedTable object resulting from the executed delete query

        Examples:

        Define a q table in python and named in q memory

        ```python
        qtab = pykx.q('([]name:`tom`dick`harry;age:28 29 35;hair:`fair`dark`fair;eye:`green`brown`gray)')
        pykx.q['qtab'] = qtab
        ```

        Delete all the contents of the table

        ```python
        pykx.q.qsql.delete(qtab)
        pykx.q.qsql.delete('qtab')
        ```

        Delete single and multiple columns from the table

        ```python
        pykx.q.qsql.delete(qtab, 'age')
        pykx.q.qsql.delete('qtab', ['age', 'eye'])
        ```

        Delete rows of the dataset based on where condition

        ```python
        pykx.q.qsql.delete(qtab, where='hair=`fair')
        pykx.q.qsql.delete('qtab', where=['hair=`fair', 'age=28'])
        ```

        Delete a column from the dataset named in q memory and persist the result using the
        inplace keyword

        ```python
        pykx.q.qsql.delete('qtab', 'age', inplace=True)
        pykx.q['qtab']
        ```
        """ # noqa: E501
        if columns is not None and where is not None:
            raise TypeError("'where' and 'columns' clauses cannot be used simultaneously in a "
                            "delete statement")
        return self._seud(table, 'delete', columns, where, None, inplace)

    def _seud(self, table, query_type, columns=None, where=None, by=None, inplace=False) -> k.K: # noqa: C901, E501
        if not isinstance(table, str):
            table = k.K(table)

        if isinstance(table, (k.SplayedTable, k.PartitionedTable)) and inplace:
            raise QError("Application of 'inplace' updates not "
                         "supported for splayed/partitioned tables")
        select_clause = self._generate_clause(columns, 'columns', query_type)
        by_clause = self._generate_clause(by, 'by', query_type)
        where_clause = self._generate_clause(where, 'where', query_type)
        get = ''
        query_char = '!' if query_type in ('delete', 'update') else '?'
        if isinstance(table, k.K):
            if not isinstance(table, (k.Table, k.KeyedTable)):
                raise TypeError("'table' object provided was not a K tabular object or an "
                                "object which could be converted to an appropriate "
                                "representation")
        elif isinstance(table, str):
            if (not inplace and query_type in ('delete', 'update')):
                get = 'get'
        else:
            raise TypeError("'table' must be a an object which is convertible to a K object "
                            "or a string denoting an item in q memory")
        wv, bv, sv = 'value', 'value', 'value'
        if isinstance(where_clause, k.QueryPhrase):
            wv = ''
            where_clause = where_clause._phrase
        if isinstance(by_clause, (dict, k.ParseTree)):
            bv = ''
        if isinstance(select_clause, (dict, k.ParseTree)):
            sv = ''
        res = self._q(
            f'{{[tab;x;y;z]{query_char}[{get} tab;{wv} x;{bv} y;{sv} z]}}',
            table,
            where_clause,
            by_clause,
            select_clause,
            wait=True,
        )
        if inplace and isinstance(table, k.K):
            if isinstance(res, QFuture) or isinstance(res, asyncio.Task):
                raise QError("'inplace' not supported with asynchronous query")
            if type(table) != type(res):
                raise QError('Returned data format does not match input type, '
                             'cannot perform inplace operation')
            table.__dict__.update(res.__dict__)
        return res

    def _generate_clause(self, clause_value, clause_name, query_type):
        if clause_value is None:
            if clause_name in ('columns', 'where'):
                b = query_type == 'delete' and clause_name == 'columns'
                return [b'{`symbol$()}', None] if b else [b'{x}', []]
            elif clause_name == 'by':
                return [b'{x}', []] if query_type == 'exec' else [b'{x}', False]
        else:
            if clause_name in ('columns', 'by'):
                return self._generate_clause_columns_by(clause_value, clause_name, query_type)
            elif clause_name == 'where':
                return self._generate_clause_where(clause_value)
        raise TypeError(f"Provided clause_name '{clause_name}' not supported")

    def _generate_clause_columns_by(self, clause_value, clause_name, query_type):
        if isinstance(clause_value, dict):
            return self._generate_clause_columns_by_dict(clause_value)
        elif isinstance(clause_value, k.QueryPhrase):
            if clause_value._are_trees[0]:
                return [b'{x!.[y;(0 0);eval]}', clause_value._names, clause_value._phrase]
            elif query_type == 'delete' and clause_name == 'columns':
                return [b'{x}',  clause_value._names]
            else:
                return [b'{x!y}', clause_value._names, clause_value._phrase]
        elif isinstance(clause_value, k.Column):
            if query_type == 'exec':
                if clause_value._is_tree:
                    return [b'.[;enlist 0;eval]', clause_value._value]
                else:
                    return k.ParseTree(clause_value._value)
            elif query_type == 'delete' and clause_name == 'columns':
                return [b'enlist', clause_value._name]
            else:
                if clause_value._is_tree:
                    return [b'{enlist[x]!enlist .[y;enlist 0;eval]}',
                            clause_value._name, clause_value._value]
                else:
                    return {clause_value._name: clause_value._value}
        elif isinstance(clause_value, k.Variable):
            if query_type == 'exec':
                return k.ParseTree(clause_value._name)
            else:
                return {clause_value._name: clause_value._name}
        elif clause_name == 'columns' and query_type == 'delete':
            if isinstance(clause_value, str):
                if clause_value == '':
                    raise ValueError('q query specifying column cannot be empty')
                clause_value = [k.CharVector(clause_value)]
            else:
                clause_value = [k.CharVector(x) for x in clause_value]
            return [b'{parse each x}', clause_value]
        elif (query_type in ['select', 'exec']) and (clause_name in ['columns', 'by']):
            if isinstance(clause_value, k.Column):
                return clause_value
            elif isinstance(clause_value, k.QueryPhrase):
                return [b'{x!y}', clause_value._names, clause_value._phrase]
            elif isinstance(clause_value, list):
                kys=[]
                vls=[]
                for x in clause_value:
                    if isinstance(x, k.Column):
                        kys.append(x._name)
                        vls.append(x._value)
                    else:
                        kys.append(x)
                        vls.append(x)
                return [b'{x!y}', kys, vls]
            elif isinstance(clause_value, str) and query_type == 'select':
                return [b'{x!x}enlist@', clause_value]
            return [b'{x}', k.K(clause_value)]
        elif isinstance(clause_value, k.K):
            return [b'{x}', clause_value]
        raise TypeError(f"Unsupported type for '{clause_name}' clause")

    def _generate_clause_columns_by_dict(self, clause_value):
        clause_dict = {}
        for key, val in clause_value.items():
            if isinstance(val, str):
                if val == '':
                    raise ValueError(f'q query specifying column for key {key!r} cannot be empty')
                clause_dict[key] = [1, k.CharVector(val)]
            elif isinstance(val, k.Column):
                if val._is_tree:
                    clause_dict[key] = [2, val._value]
                else:
                    clause_dict[key] = [0, val._value]
            else:
                clause_dict[key] = [0, val]
        return [b'''{
                key[x]!{$[0=x 0;(::);1=x 0;parse;2=x 0;.[;enlist 0;eval];(::)] x 1}each value x
                }''', clause_dict]

    def _generate_clause_where(self, clause_value) -> k.List:
        if isinstance(clause_value, (k.QueryPhrase, k.ParseTree, k.Column, k.List)):
            return k.QueryPhrase(clause_value)
        if isinstance(clause_value, k.BooleanVector):
            return k.QueryPhrase([clause_value])
        if isinstance(clause_value, str):
            clause_value = [k.CharVector(clause_value)]
        elif all([isinstance(x, str) for x in clause_value]):
            clause_value = [k.CharVector(x) for x in clause_value]
        else:
            wp = k.QueryPhrase(clause_value[0])
            for wc in clause_value[1:]:
                wp.extend(k.QueryPhrase(wc))
            return wp
        return [b'{parse each x}', clause_value]


class SQL:
    """Wrapper around the [KX Insights Core ANSI SQL](https://code.kx.com/insights/core/sql.html) interface.

    Examples within this interface use a table named **trades**, an example of this table is

    ```Python
    >>> trades = kx.Table(data={
            'sym': kx.random.random(100, ['AAPL', 'GOOG', 'MSFT']),
            'date': kx.random.random(100, kx.q('2022.01.01') + [0,1,2]),
            'price': kx.random.random(100, 1000.0)
        })
    >>> kx.q['trades'] = trades
    ```
    """ # noqa: E501

    def __init__(self, q: Q):
        self._q = q

    def __call__(self, query: str, *args: Any) -> k.Table:
        """Compile and run a SQL statement using string interpolation.

        Parameters:
            query: The query to execute formatted in
                [KX Insights SQL style](https://code.kx.com/insights/core/sql.html)
            *args: The arguments for the query, which will be interpolated into the string. Each
                argument will be converted into a [pykx.K][pykx.K] object.

        Returns:
            The result of the evaluation of `#!python query` with `#!python args` interpolated.

        Note: Avoid interpolating the table name into the query when using this with IPC.
            Use the full name of the table in the string.
            When using this class on the embedded q process it is common to interpolate a
            `#!python pykx.Table` object into a query using `#!python '$1'`. When the `#!python Q`
            object used in the initialization of this class is an [IPC connection][pykx.QConnection]
            the entire table will be sent in the message over the connection. If the table is large
            this will significantly impact performance.

        Examples:

        Query a table by name:

        ```python
        >>> q.sql('select * from trades')
        pykx.Table(pykx.q('
        sym  date       price
        ------------------------
        AAPL 2022.01.02 484.4727
        AAPL 2022.01.02 682.7999
        MSFT 2022.01.01 153.227
        MSFT 2022.01.03 535.0923
        ..
        '))
        ```

        Query a [`pykx.Table`][pykx.Table] instance by injecting it as the first argument using `$n`
          syntax:

        ```python
        >>> q.sql('select * from $1', trades) # where `trades` is a `pykx.Table` object
        pykx.Table(pykx.q('
        sym  date       price
        ------------------------
        AAPL 2022.01.02 484.4727
        AAPL 2022.01.02 682.7999
        MSFT 2022.01.01 153.227
        MSFT 2022.01.03 535.0923
        ..
        '))
        ```

        Query a table using multiple injected arguments:

        ```python
        >>> q.sql('select * from trades where date = $1 and price < $2', date(2022, 1, 2), 500.0)
        pykx.Table(pykx.q('
        sym  date       price
        ------------------------
        AAPL 2022.01.02 484.4727
        MSFT 2022.01.02 457.328
        GOOG 2022.01.02 8.062521
        MSFT 2022.01.02 338.0097
        ..
        '))
        ```
        """
        return self._q('.s.sp', k.CharVector(query), args)

    def prepare(self, query: str, *args: Any) -> k.List:
        """Prepare a parametrized query to be executed later.

        Parameters:
            query: The query to parameterize in
                [KX Insights SQL format](https://code.kx.com/insights/core/sql.html).
            *args: The arguments for `#!python query`. The arguments are not used in the query. They
                are used to determine the expected types of the parameters of the parameterization.

        Returns:
            The parametrized query, which can later be used with `#!python q.query.execute()`

        Examples:

        Note: Preparing a query does not require fully constructed K Atom and Vector types.
            Both the value `#!python kx.LongAtom(1)` and the wrapper [pykx.LongAtom][pykx.LongAtom]
            are valid. To determine table type use `#!python pykx.Table.prototype`.

        Prepare a query for later execution that will expect a table with 3 columns a, b, and c with
        ktypes [pykx.SymbolVector][pykx.SymbolVector], [pykx.FloatVector][pykx.FloatVector], and
        [pykx.LongVector][pykx.LongVector] respectively.

        ```Python
        >>> p = q.sql.prepare('select * from $1', kx.q('([] a:``; b: 0n 0n; c: 0N 0N)'))
        ```
        You can also use the `#!python pykx.Table.prototype` helper function to build a table to
        pass into a prepared SQL query.

        ```Python
        >>> p = q.sql.prepare('select * from $1', kx.Table.prototype({
            'a': kx.SymbolVector,
            'b': kx.FloatVector,
            'c': kx.LongVector})
        )
        ```

        Prepare a query for later execution that will take a date and a float as input at execution
        time to query the trades table.

        ```Python
        >>> p = q.sql.prepare('select * from trades where date = $1 and price < $2',
            date(1, 1, 2),
            500.0
        )
        ```

        You can also directly pass in the [pykx.K][pykx.K] types you wish to use instead.

        ```Python
        >>> p = q.sql.prepare('select * from trades where date = $1 and price < $2',
            kx.DateAtom,
            kx.FloatAtom
        )
        ```
        """
        _args = []
        for a in args:
            _args.append(a._prototype() if (isinstance(a, type) or isinstance(a, ABCMeta)) else a)
        return self._q('.s.sq', k.CharVector(query), _args)

    def execute(self, query: k.List, *args: Any) -> k.K:
        """Execute a prepared query. Parameter types must match the types of the arguments
        used when executing the `#!python sql.prepare` function.

        Parameters:
            query: A prepared SQL statement returned by a call to `#!python sql.prepare`.
            *args: The arguments for the query, which will be interpolated into the query. Each
                argument will be converted into a [pykx.K][pykx.K] object.

        Returns:
            The result of the evaluation of `#!python query` with `#!python args` interpolated.

        Note: Avoid interpolating the table name into the query when using this with IPC.
            Use the full name of the table in the string.
            When using this class on the embedded q process it is common to interpolate a
            `#!python pykx.Table` object into a query using `#!python '$1'`. When the `#!python Q`
            object used in the initialization of this class is an [IPC connection][pykx.QConnection]
            the entire table will be sent in the message over the connection. If the table is large
            this will significantly impact performance.

        Examples:

        Execute a prepared query passing in a [pykx.Table][pykx.Table] with 3 columns a, b, and c
        with ktypes [pykx.SymbolVector][pykx.SymbolVector], [pykx.FloatVector][pykx.FloatVector],
        and [pykx.LongVector][pykx.LongVector] respectively.

        ```Python
        >>> p = q.sql.prepare('select * from $1', kx.q('([] a:``; b: 0n 0n; c: 0N 0N)'))
        >>> q.sql.execute(p, kx.q('([] a:`a`b`c`d; b: 1.0 2.0 3.0 4.0; c: 1 2 3 4)'))
        pykx.Table(pykx.q('
        a b c
        -----
        a 1 1
        b 2 2
        c 3 3
        d 4 4
        '))
        ```

        Execute a prepared query that takes a date and a float as input to query the trades table.

        ```python
        >>> p = q.sql.prepare('select * from trades where date = $1 and price < $2',
            date(1, 1, 2),
            500.0
        )
        >>> q.sql.execute(p, date(2022, 1, 2), 500.0)
        pykx.Table(pykx.q('
        sym  date       price
        ------------------------
        AAPL 2022.01.02 484.4727
        MSFT 2022.01.02 457.328
        GOOG 2022.01.02 8.062521
        MSFT 2022.01.02 338.0097
        ..
        '))
        ```
        """
        return self._q('.s.sx', query, args)

    def get_input_types(self, prepared_query: k.List) -> List[str]:
        """Get the [pykx.K][pykx.K] types that are expected to be used with a prepared query.

        Parameters:
            prepared_query: A prepared SQL statement returned by a call to `#!python q.sql.prepare`.

        Returns:
            A Python list object containing the string representations of the expected K types for
                use with the prepared statement.

        Examples:

        ```Python
        >>> p = q.sql.prepare('select * from trades where date = $1 and price < $2',
            date(1, 1, 1),
            0.0
        )
        >>> q.sql.get_input_types(p)
        ['DateAtom/DateVector', 'FloatAtom/FloatVector']
        ```
        """
        sql_to_ktype = {
            'A': 'CharVector/List[CharVector]',
            'S': 'SymbolAtom/SymbolVector',
            'C': 'CharAtom',
            'B': 'BooleanAtom',
            'Q': 'GUIDAtom/GUIDVector',
            'X': 'ByteAtom/ByteVector',
            'H': 'ShortAtom/ShortVector',
            'I': 'IntAtom/IntVector',
            'J': 'LongAtom/LongVector',
            'E': 'RealAtom/RealVector',
            'F': 'FloatAtom/FloatVector',
            'D': 'DateAtom/DateVector',
            'T': 'TimeAtom/TimeVector',
            'P': 'TimestampAtom/TimestampVector',
            'L': 'EnumAtom used for linked table access',
            'W': 'Identity',

        }
        type_dict = prepared_query[0].py()
        type_checking_info = prepared_query[1][2][0]
        type_keys = type_dict['']
        types = []
        for i in range(len(type_checking_info)):
            # This is not very nice but it works, this q object was not really meant to be parsed
            types.append(sql_to_ktype[type_dict[type_keys[type_checking_info[i][1].py()]]])
        return types


class TableAppend:
    """Helper class for the q insert and upsert functions"""

    def __init__(self, _q):
        self._q = _q

    def __typed_row(self, row: Any) -> k.K:
        if isinstance(row, k.Table):
            return row
        if isinstance(row, k.K):
            row = row.py()
        if not isinstance(row, list):
            raise TypeError('Expected list like object to append to table')
        if isinstance(row[0], list):
            k_rows = []
            for v in row:
                n = str(type(k.K(v[0])))
                if 'Atom' in n:
                    t = 'k.' + n.split("'")[1].split('.')[-1][:-4] + 'Vector'
                    cls = eval(t)
                    k_rows.append(cls(v))
                else:
                    k_rows.append(k.K(v))
            k_rows = k.K(k_rows)
        else:
            k_rows = k.K([k.K(x) for x in row])
        return k_rows

    def append(
        self,
        method: str,
        table: Union[str, k.SymbolAtom, k.Table],
        row: Union[List, k.List],
        match_schema: bool = False,
        test_insert: bool = False
    ) -> Union[None, k.Table]:
        k_row = None
        if match_schema:
            s1 = self._q._call(
                '{[tab] {[tab; x] meta[tab][x][`t]}[tab;] each key meta[tab]}',
                table,
                wait=True
            ).py()
            k_row = self.__typed_row(row)
            if type(k_row) == k.Table:
                s2 = self._q._call(
                    '{[tab] {[tab; x] meta[tab][x][`t]}[tab;] each key meta[tab]}',
                    k_row,
                    wait=True
                ).py()
            else:
                s2 = self._q._call('.Q.ty each', k_row, wait=True).py()
            if s1 != s2 and s1 != s2.swapcase():
                raise PyKXException(f'Schema mismatch\n\tAttempted to {method} row: {row}\n'
                                    f'\tBut the row\'s schema: "{str(s2)[2:-1]}" does not match the'
                                    f' tables schema: "{str(s1)[2:-1]}"\n\tNote: The case of the '
                                    'schema is allowed to be inverted.')
        if k_row is None:
            k_row = self.__typed_row(row)
        if test_insert:
            local_table = self._q('{?[x;();0b;();-5]}', table) # select last 5 rows of table
            self._q['.pykx.i.local_tab'] = local_table
            self._q._call(f'`.pykx.i.local_tab {method}', k_row)
            return self._q._call('.pykx.i.local_tab')
        else:
            return self._q._call(method, table, k_row)


class Insert(TableAppend):
    """Helper class for the q insert function"""

    def __init__(self, _q: Q):
        super().__init__(_q)

    def __call__(
        self,
        table: Union[str, k.SymbolAtom],
        row: Union[List, k.List],
        match_schema: bool = False,
        test_insert: bool = False
    ) -> Union[None, k.Table]:
        """Helper function around q's `#!q insert` function which inserts a row or multiple rows into
        a q table object.

        Parameters:
            table: The name of the table for the insert operation.
            row: A list of objects to be inserted as a row, or a list of lists of objects
                to insert multiple rows at once.
            match_schema: Whether the row/rows to be inserted must match the table's current schema.
            test_insert: Causes the function to modify a small local copy of the table and return
                the modified example, this can only be used with embedded q and will not modify the
                source table's contents.

        Returns:
            When `#!python test_insert` is false return a `#!python k.LongVector` denoting the
                index of the rows that were inserted. When `#!python test_insert` is true return the
                last 5 rows of the table with the new rows inserted onto the end leaving
                `#!python table` unmodified.

        Raises:
            PyKXException: If the `#!python match_schema` parameter is used this function may raise
                an error if the row to be inserted does not match the table's schema. The error
                message will contain information about which columns did not match.

        Examples:

        Insert a single row onto a table named `#!python tab` ensuring that the row matches the
        table's schema. This will raise an error if the row does not match.

        ```Python
        >>> q.insert('tab', [1, 2.0, datetime.datetime(2020, 2, 24)], match_schema=True)
        ```

        Insert multiple rows onto a table named `#!python tab` ensuring that each of the rows being
        added match the table's schema.

        ```Python
        >>> q.insert(
            'tab',
            [[1, 2], [2.0, 4.0], [datetime.datetime(2020, 2, 24), datetime.datetime(2020,3 , 19)]],
            match_schema=True
        )
        ```

        Run a test insert to modify a local copy of the table to test what the table would look
        like after inserting the new rows.

        ```Python
        >>> kx.q['tab'] = kx.Table([[1, 1.0, 'a'], [2, 2.0, 'b'], [3, 3.0, 'c']], columns=['a', 'b', 'c'])
        >>> kx.q.insert('tab', [4, 4.0, 'd'], test_insert=True) # example of table after insert
        pykx.Table(pykx.q('
        a b c
        -----
        1 1 a
        2 2 b
        3 3 c
        4 4 d
        '))
        >>> kx.q('tab') # table object was not modified
        pykx.Table(pykx.q('
        a b c
        -----
        1 1 a
        2 2 b
        3 3 c
        '))
        ```
        """ # noqa: E501
        return self.append('insert', table, row, match_schema, test_insert)


class Upsert(TableAppend):
    """Helper class for the q upsert function"""

    def __init__(self, _q: Q):
        super().__init__(_q)

    def __call__(
        self,
        table: Union[str, k.SymbolAtom, k.Table],
        row: Union[List, k.List],
        match_schema: bool = False,
        test_insert: bool = False
    ) -> Union[None, k.Table]:
        """Helper function around q's `#!q upsert` function which inserts a row or multiple rows into
        a q table object.

        Parameters:
            table: A `#!python k.Table` object or the name of the table.
            row: A list of objects to be appended as a row, if the table is within embedded q you
                may also pass in a table object to be upserted.
            match_schema: Whether the row/rows to be appended must match the table's current schema.
            test_insert: Causes the function to modify a small local copy of the table and return
                the modified example, this can only be used with embedded q and will not modify the
                source table's contents.

        Returns:
            When `#!python test_insert` is false and `#!python table` is a `#!python k.Table` return
                the modified table. When `#!python test_insert` is true return the last 5 rows of
                the table with new rows appended to the end. In all other cases `#!python None`
                is returned.

        Raises:
            PyKXException: If the `#!python match_schema` parameter is used this function may raise
                an error if the row to be inserted does not match the table's schema. The error
                message will contain information about which columns did not match.

        Examples:

        Upsert a single row onto a table named `#!python tab` ensuring that the row matches the
        table's schema. This will raise an error if the row does not match.

        ```Python
        >>> q.upsert('tab', [1, 2.0, datetime.datetime(2020, 2, 24)], match_schema=True)
        >>> table = q.upsert(table, [1, 2.0, datetime.datetime(2020, 2, 24)], match_schema=True)
        ```

        Upsert multiple rows onto a table named `#!python tab` ensuring that each of the rows being
        added match the table's schema.

        ```Python
        >>> q.upsert(
            'tab',
            q('([] a: 1 2; b: 1.0 2.0; c: `b`c)'),
            match_schema=True
        )
        >>> table = q.upsert(
            table,
            [[1, 2], [2.0, 4.0], [datetime.datetime(2020, 2, 24), datetime.datetime(2020,3 , 19)]],
            match_schema=True
        )
        ```

        Run a test upsert to modify a local copy of the table to test what the table would look
        like after appending the new rows.

        ```Python
        >>> kx.q['tab'] = kx.Table([[1, 1.0, 'a'], [2, 2.0, 'b'], [3, 3.0, 'c']], columns=['a', 'b', 'c'])
        >>> kx.q.upsert('tab', [4, 4.0, 'd'], test_insert=True) # example of table after insert
        pykx.Table(pykx.q('
        a b c
        -----
        1 1 a
        2 2 b
        3 3 c
        4 4 d
        '))
        >>> kx.q('tab') # table object was not modified
        pykx.Table(pykx.q('
        a b c
        -----
        1 1 a
        2 2 b
        3 3 c
        '))
        ```
        """ # noqa: E501
        return self.append('upsert', table, row, match_schema, test_insert)

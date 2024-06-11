"""Query interfaces for PyKX."""

from abc import ABCMeta
from typing import Any, Dict, List, Optional, Union
import warnings
from uuid import uuid4

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
    """Generates and submits functional q SQL queries.

    Instances of this class can be accessed as the `qsql` attribute of any [`pykx.Q`][pykx.Q]. For
    instance, `pykx.q.qsql`, or `pykx.QConnection(...).qsql`.

    The `QSQL` class provides Python users with a method of querying q simple, keyed, splayed and
    partitioned tables using a single set of functionality.

    This is achieved by wrapping the logic contained within the q functional select, exec, update,
    and delete functionality. For more information on this functionality please refer to [Chapter 9
    Section 12 of Q for Mortals](https://code.kx.com/q4m3/9_Queries_q-sql/#912-functional-forms).

    While it is also conceivable that the interface could compile a qSQL statement to achieve the
    same end goal there are a number of advantages to using the more complex functional form.

    1. Users that are unfamiliar with q who use the interface are introduced to the more powerful
        version of querying with q, while still operating within a familiar setting.
    2. Using the functional form provides the ability when running functional updates to update the
        q tables with data derived from Python:

        ```python
        qtable = pykx.q('([]1 2 3;4 5 6)')
        pykx.q.qsql.update(qtable, {'x': [10, 20, 30]})
        ```

    3. It makes development and maintenance of the interface easier when dealing across the forms
        of supported table within q within which the functional forms of interacting with tables
        are more natural.
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
        """Apply a q style select statement on tables defined within the process.

        This implementation follows the q functional select syntax with limitations on
        structures supported for the various clauses a result of this.

        Parameters:
            table: The q table or name of the table (provided the table is named within the q
                memory space) on which the select statement is to be applied.
            columns: A dictionary mapping the name to be given to a column and the logic to be
                applied in aggregation to that column both as strings.
            where: Conditional filtering used to select subsets of the data on which by-clauses and
                appropriate aggregations are to be applied.
            by: A dictionary mapping the names to be assigned to the produced columns and the
                columns whose results are used to construct the groups of the by clause.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects
                https://code.kx.com/q/basics/qsql/#result-and-side-effects.

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
        Apply a q style exec statement on tables defined within the process.

        This implementation follows the q functional exec syntax with limitations on structures
        supported for the various clauses a result of this.

        Parameters:
            table: The q table or name of the table (provided the table is named within the q
                memory space) on which the exec statement is to be applied.
            columns: A dictionary mapping the name to be given to a column and the logic to be
                applied in aggregation to that column both as strings. A string defining a single
                column to be retrieved from the table as a list.
            where: Conditional filtering used to select subsets of the data on which by clauses and
                appropriate aggregations are to be applied.
            by: A dictionary mapping the names to be assigned to the produced columns and the
                the columns whose results are used to construct the groups of the by clause.

        Examples:

        Define a q table in python and named in q memory

        ```python
        pykx.q['qtab'] = pd.DataFrame.from_dict({
            'col1': [['a', 'b', 'c'][randint(0, 2)] for _ in range(100)],
            'col2': [random() for _ in range(100)],
            'col3': [randint(0, 1) == 1 for _ in range(100)],
            'col4': [random() * 10 for _ in range(100)]
        })
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
               modify: bool = False,
               inplace: bool = False,
    ) -> k.K:
        """
        Apply a q style update statement on tables defined within the process.

        This implementation follows the q functional update syntax with limitations on
        structures supported for the various clauses a result of this.

        Parameters:
            table: The q table or name of the table (provided the table is named within the q
                memory space) on which the update statement is to be applied.
            columns: A dictionary mapping the name of a column present in the table or one to be
                added to the contents which are to be added to the column, this content can be a
                string denoting q data or the equivalent Python data.
            where: Conditional filtering used to select subsets of the data on which by-clauses and
                appropriate aggregations are to be applied.
            by: A dictionary mapping the names to be assigned to the produced columns and the
                columns whose results are used to construct the groups of the by clause.
            modify: `Deprecated`, please use `inplace` instead. Whether the result of an update
                is to be saved. This operates for tables referenced by name in q memory or
                general table objects
                https://code.kx.com/q/basics/qsql/#result-and-side-effects.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects
                https://code.kx.com/q/basics/qsql/#result-and-side-effects.

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

        Apply an update grouping based on a by phrase and persist the result using the modify keyword

        ```python
        pykx.q.qsql.update('byqtab', columns={'weight': 'avg weight'}, by={'city': 'city'}, inplace=True)
        pykx.q['byqtab']
            ```
        """ # noqa: E501
        return self._seud(table, 'update', columns, where, by, modify, inplace)

    def delete(self,
               table: Union[k.Table, str],
               columns: Optional[Union[List[str], k.SymbolVector]] = None,
               where: Optional[Union[List[str], str, k.SymbolAtom, k.SymbolVector]] = None,
               modify: bool = False,
               inplace: bool = False,
    ) -> k.K:
        """
        Apply a q style delete statement on tables defined within the process.

        This implementation follows the q functional delete syntax with limitations on
        structures supported for the various clauses a result of this.

        Parameters:
            table: The q table or name of the table (provided the table is named within the q
                memory space) on which the delete statement is to be applied.
            columns: Denotes the columns to be deleted from a table.
            where: Conditional filtering used to select subsets of the data which are to be
                deleted from the table.
            modify: `Deprecated`, please use `inplace` instead. Whether the result of a delete
                is to be saved. This holds when `table` is the name of a table in q memory,
                as outlined at:
                https://code.kx.com/q/basics/qsql/#result-and-side-effects.
            inplace: Whether the result of an update is to be persisted. This operates for tables
                referenced by name in q memory or general table objects
                https://code.kx.com/q/basics/qsql/#result-and-side-effects.

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
        modify keyword

        ```python
        pykx.q.qsql.delete('qtab', 'age', modify=True)
        pykx.q['qtab']
        ```
        """ # noqa: E501
        if columns is not None and where is not None:
            raise TypeError("'where' and 'columns' clauses cannot be used simultaneously in a "
                            "delete statement")
        return self._seud(table, 'delete', columns, where, None, modify, inplace)

    def _seud(self, table, query_type, columns=None, where=None, by=None, modify=False, inplace=False) -> k.K: # noqa: C901, E501
        if modify and inplace:
            raise RuntimeError("Attempting to use both 'modify' and 'inplace' keywords, please use only 'inplace'") # noqa: E501

        if modify:
            warnings.warn("The 'modify' keyword is now deprecated please use 'inplace'")
            inplace = modify

        if not isinstance(table, str):
            table = k.K(table)

        if isinstance(table, (k.SplayedTable, k.PartitionedTable)) and inplace:
            raise QError("Application of 'inplace' updates not "
                         "supported for splayed/partitioned tables")
        select_clause = self._generate_clause(columns, 'columns', query_type)
        by_clause = self._generate_clause(by, 'by', query_type)
        where_clause = self._generate_clause(where, 'where', query_type)
        original_table = table
        if isinstance(table, k.K):
            if not isinstance(table, (k.Table, k.KeyedTable)):
                raise TypeError("'table' object provided was not a K tabular object or an "
                                "object which could be converted to an appropriate "
                                "representation")
            randguid = str(uuid4())
            self._q(f'''
                    {{@[{{get x}};`.pykx.i.updateCache;{{.pykx.i.updateCache:(`guid$())!()}}];
                    .pykx.i.updateCache["G"$"{randguid}"]:x}}
                    ''', table)
            original_table = table
            table_code = f'.pykx.i.updateCache["G"$"{randguid}"]'
            if not inplace:
                query_char = '!' if query_type in ('delete', 'update') else '?'
            else:
                query_char = table_code + (':!' if query_type in ('delete', 'update') else ':?')
        elif not isinstance(table, str):
            raise TypeError("'table' must be a an object which is convertible to a K object "
                            "or a string denoting an item in q memory")
        else:
            if (not inplace and query_type in ('delete', 'update')):
                table_code = f'get`$"{table}"'
            else:
                table_code = f'`$"{table}"'
            query_char = '!' if query_type in ('delete', 'update') else '?'
        try:
            res = self._q(
                f'{{{query_char}[{table_code};value x;value y;value z]}}',
                where_clause,
                by_clause,
                select_clause,
                wait=True,
            )
            if inplace and isinstance(original_table, k.K):
                res = self._q(table_code)
                if isinstance(res, QFuture):
                    raise QError("'inplace' not supported with asynchronous query")
                if type(original_table) != type(res):
                    raise QError('Returned data format does not match input type, '
                                 'cannot perform inplace operation')
                original_table.__dict__.update(res.__dict__)
            return res
        finally:
            if isinstance(original_table, k.K):
                self._q._call(f'.pykx.i.updateCache _:"G"$"{randguid}"', wait=True)

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
        elif clause_name == 'columns' and query_type == 'delete':
            if isinstance(clause_value, str):
                if clause_value == '':
                    raise ValueError('q query specifying column cannot be empty')
                clause_value = [k.CharVector(clause_value)]
            else:
                clause_value = [k.CharVector(x) for x in clause_value]
            return [b'{parse each x}', clause_value]
        elif (query_type in ['select', 'exec']) and (clause_name in ['columns', 'by']):
            if isinstance(clause_value, list):
                return [b'{v!v:{$[0>type x;x;(0h>v 0)&1~count v:distinct type each x;raze x;x]}x}', clause_value] # noqa: E501
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
                clause_dict[key] = [True, k.CharVector(val)]
            else:
                clause_dict[key] = [False, val]
        return [b'{key[x]!{$[x 0;parse;{$[0>type x;x;(0h>v 0)&1~count v:distinct type each x;raze x;x]}]x 1}each value x}', clause_dict] # noqa: E501

    def _generate_clause_where(self, clause_value) -> k.List:
        if isinstance(clause_value, k.List):
            return [b'{x}', clause_value]
        if isinstance(clause_value, k.BooleanVector):
            return [b'{enlist x}', clause_value]
        if isinstance(clause_value, str):
            clause_value = [k.CharVector(clause_value)]
        else:
            clause_value = [k.CharVector(x) for x in clause_value]
        return [b'{parse each x}', clause_value]


class SQL:
    """Wrapper around the [KX Insights Core ANSI SQL](https://code.kx.com/insights/core/sql.html) interface.

    Lots of examples within this interface use a table named trades, an example of this table is

    ```Python
    >>> kx.q['trades'] = kx.toq(
        pd.DataFrame.from_dict({
            'sym': [['AAPL', 'GOOG', 'MSFT'][randint(0, 2)] for _ in range(100)],
            'date': [[date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)][randint(0, 2)] for _ in range(100)],
            'price': [random() * 1000 for _ in range(100)]
        })
    )
    ```
    """ # noqa: E501

    def __init__(self, q: Q):
        self._q = q

    def __call__(self, query: str, *args: Any) -> k.Table:
        """Compile and run a SQL statement.

        Parameters:
            query: The SQL query, using KX Insights Core SQL, documented at
                https://code.kx.com/insights/core/sql.html
            *args: The arguments for the query, which will be interpolated into the query. Each
                argument will be converted into a [`pykx.K`][] object.

        Returns:
            The result of the evaluation of `query` with `args` interpolated.

        Note: Avoid interpolating the table into the query when running over IPC.
            It's common to interpolate a `pykx.Table` object into the query as `'$1'`. This works
            well when running embedded within the process, but when the `Q` instance is an
            [IPC connection][pykx.QConnection] this will result in the entire table being sent over
            the connection, which will negatively impact performance. Instead, when running over
            IPC, write the name of the table (as defined in the connected q server) directly into
            the query.

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

        Query a [`pykx.Table`][] instance by interpolating it in as the first argument:

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
        Query a table using interpolated conditions:

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
        """Prepare a parametrized query to be executed later, the parameter types are deduced from
        the types of the arguments used to prepare the statement.

        Parameters:
            query: The SQL query, using KX Insights Core SQL, documented at
                https://code.kx.com/insights/core/sql.html
            *args: The arguments for the query, these arguments are not used in the query. They are
                used to determine the types of the parameters that will later be used as parameters
                when executing the query.

        Returns:
            The parametrized query, which can later be used with `q.query.execute()`

        Examples:

        Note: When preparing a query with K types you don't have to fully construct one.
            For example you can pass `kx.LongAtom(1)` as a value to the prepare function as well as
            just [`pykx.LongAtom`][]. This only works for Atom and Vector types. There is also a
            helper function for tables that you can use called `pykx.Table.prototype`.

        Prepare a query for later execution that will expect a table with 3 columns a, b, and c with
        ktypes [`pykx.SymbolVector`][], [`pykx.FloatVector`][], and [`pykx.LongVector`][]
        respectively.

        ```Python
        >>> p = q.sql.prepare('select * from $1', kx.q('([] a:``; b: 0n 0n; c: 0N 0N)'))
        ```
        You can also use the `pykx.Table.prototype` helper function to build a table to pass into a
        prepared SQL query.

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

        You can also directly pass in the [`pykx.K`][] types you wish to use instead.

        ```Python
        >>> p = q.sql.prepare('select * from trades where date = $1 and price < $2',
            kx.DateAtom,
            kx.FloatAtom
        )
        ```
        """
        _args = []
        for a in args:
            _args.append(a._prototype() if (type(a) == type or type(a) == ABCMeta) else a)
        return self._q('.s.sq', k.CharVector(query), _args)

    def execute(self, query: k.List, *args: Any) -> k.K:
        """Execute a prepared query the parameter types must match the types of the arguments
        used in the prepare statement.

        Parameters:
            query: A prepared SQL statement returned by a call to `q.sql.prepare`.
            *args: The arguments for the query, which will be interpolated into the query. Each
                argument will be converted into a [`pykx.K`][] object.

        Returns:
            The result of the evaluation of `query` with `args` interpolated.

        Note: Avoid interpolating the table into the query when running over IPC.
            It's common to interpolate a [`pykx.Table`][] object into the query as `'$1'`. This
            works well when running embedded within the process, but when the `Q` instance is an
            [IPC connection][pykx.QConnection] this will result in the entire table being sent over
            the connection, which will negatively impact performance. Instead, when running over
            IPC, write the name of the table (as defined in the connected q server) directly into
            the query.

        Examples:

        Execute a prepared query passing in a [`pykx.Table`][] with 3 columns a, b, and c with
        ktypes [`pykx.SymbolVector`][], [`pykx.FloatVector`][], and [`pykx.LongVector`][]
        respectively.

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
        """Get the [`pykx.K`][] types that are expected to be used with a prepared query.

        Parameters:
            prepared_query: A prepared SQL statement returned by a call to `q.sql.prepare`.

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
    """Helper class for the q insert function"""

    def __init__(self, _q):
        self._q = _q

    def __typed_row(self, row: Any) -> k.K:
        if isinstance(row, k.Table):
            return row
        if isinstance(row, k.K):
            row = row.py()
        if type(row) != list:
            raise TypeError('Expected list like object to append to table')
        if type(row[0]) == list:
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
    """Helper class for the q insert and upsert functions"""

    def __init__(self, _q: Q):
        super().__init__(_q)

    def __call__(
        self,
        table: Union[str, k.SymbolAtom],
        row: Union[List, k.List],
        match_schema: bool = False,
        test_insert: bool = False
    ) -> Union[None, k.Table]:
        """Helper function around `q`'s `insert` function which inserts a row or multiple rows into
        a q table object.

        Parameters:
            table: The name of the table to be inserted onto.
            row: A list of objects to be inserted as a row, or a list of lists containing objects
                to insert multiple rows at once.
            match_schema: Whether the row/rows to be inserted must match the tables current schema.
            test_insert: Causes the function to modify a small local copy of the table and return
                the modified example, this can only be used with embedded q and will not modify the
                source tables contents.

        Returns:
            A `k.LongVector` denoting the index of the rows that were inserted, unless the
            `test_insert` keyword argument is used in which case it returns the
            last 5 rows of the table with the new rows inserted onto the end, this does not modify
            the actual table object.

        Raises:
            PyKXException: If the `match_schema` parameter is used this function may raise an error
                if the row to be inserted does not match the tables schema. The error message will
                contain information about which columns did not match.

        Examples:

        Insert a single row onto a table named `tab` ensuring that the row matches the tables
        schema. Will raise an error if the row does not match

        ```Python
        >>> q.insert('tab', [1, 2.0, datetime.datetime(2020, 2, 24)], match_schema=True)
        ```

        Insert multiple rows onto a table named `tab` ensuring that each of the rows being added
        match the tables schema.

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
        """Helper function around `q`'s `upsert` function which inserts a row or multiple rows into
        a q table object.

        Parameters:
            table: A `k.Table` object or the name of the table to be inserted onto.
            row: A list of objects to be inserted as a row, if the table is within embedded q you
                may also pass in a table object to be upserted.
            match_schema: Whether the row/rows to be inserted must match the tables current schema.
            test_insert: Causes the function to modify a small local copy of the table and return
                the modified example, this can only be used with embedded q and will not modify the
                source tables contents.

        Returns:
            The modified table if a `k.Table` is passed in, otherwise `None` is returned.
            If the `test_insert` keyword argument is used it returns the last 5 rows of the table
            with the new rows inserted onto the end, this does not modify the actual table object.

        Raises:
            PyKXException: If the `match_schema` parameter is used this function may raise an error
                if the row to be inserted does not match the tables schema. The error message will
                contain information about which columns did not match.

        Examples:

        Upsert a single row onto a table named `tab` ensuring that the row matches the tables
        schema. Will raise an error if the row does not match

        ```Python
        >>> q.upsert('tab', [1, 2.0, datetime.datetime(2020, 2, 24)], match_schema=True)
        >>> table = q.upsert(table, [1, 2.0, datetime.datetime(2020, 2, 24)], match_schema=True)
        ```

        Upsert multiple rows onto a table named `tab` ensuring that each of the rows being added
        match the tables schema. Upserting multiple rows only works within embedded q.

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

        Run a test insert to modify a local copy of the table to test what the table would look
        like after inserting the new rows.

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

"""
Functionality to support the creation and manipulation of schemas.

Generated schemas can be used in combination with both
 [`insert`](https://code.kx.com/pykx/api/pykx-q-data/wrappers.html#pykx.wrappers.Table.insert) and
 [`upsert`](https://code.kx.com/pykx/api/pykx-q-data/wrappers.html#pykx.wrappers.Table.upsert)
 functionality to create populated table and keyed table objects.
"""

from typing import Dict, List, Optional, Union

from . import wrappers as k


__all__ = [
    'builder',
]


def _init(_q):
    global q
    q = _q


def __dir__():
    return __all__


_ktype_to_conversion = {
    k.List: "",
    k.GUIDAtom: "guid", k.GUIDVector: "guid",
    k.BooleanAtom: "boolean", k.BooleanVector: "boolean",
    k.ByteAtom: "byte", k.ByteVector: "byte",
    k.ShortAtom: "short", k.ShortVector: "short",
    k.IntAtom: "int", k.IntVector: "int",
    k.LongAtom: "long", k.LongVector: "long",
    k.RealAtom: "real", k.RealVector: "real",
    k.FloatAtom: "float", k.FloatVector: "float",
    k.CharAtom: "char",
    k.SymbolAtom: "symbol", k.SymbolVector: "symbol",
    k.TimestampAtom: "timestamp", k.TimestampVector: "timestamp",
    k.MonthAtom: "month", k.MonthVector: "month",
    k.DateAtom: "date", k.DateVector: "date",
    k.DatetimeAtom: "datetime", k.DatetimeVector: "datetime",
    k.TimespanAtom: "timespan", k.TimespanVector: "timespan",
    k.MinuteAtom: "minute", k.MinuteVector: "minute",
    k.SecondAtom: "second", k.SecondVector: "sector",
    k.TimeAtom: "time", k.TimeVector: "time",
}


def builder(schema: Dict,
            *,
            key: Optional[Union[str, List[str]]] = None
) -> k.K:
    """Generate an empty schema for a keyed or unkeyed table.

    Parameters:
        schema: The definition of the schema to be created mapping a 'str'
            to a `pykx.*` type object which is one of the types defined in
            `pykx.schema._ktype_to_conversion`.
        key: A `str`-like object or list of `str` objects denoting the columns
            within the table defined by `schema` to be treated as primary keys,
            see [here](https://code.kx.com/q4m3/8_Tables/#841-keyed-table) for
            more information about q keyed tables.

    Returns:
        A `pykx.Table` or `pykx.KeyedTable` matching the provided schema with
            zero rows.

    Examples:

    Create a simple `pykx.Table` with four columns of different types

    ```python
    >>> import pykx as kx
    >>> qtab = kx.schema.builder({
            'col1' : kx.GUIDAtom,
            'col2': kx.TimeAtom,
            'col3': kx.BooleanAtom,
            'col4': kx.FloatAtom
            })
    >>> qtab
    pykx.Table(pykx.q('
    col1 col2 col3 col4
    -------------------
    '))
    >>> kx.q.meta(qtab)
    pykx.KeyedTable(pykx.q('
    c   | t f a
    ----| -----
    col1| g
    col2| t
    col3| b
    col4| f
    '))
    ```

    Create a `pykx.KeyedTable` with a single primary key.

    ```python
    >>> import pykx as kx
    >>> qtab = kx.schema.builder({
            'col1': kx.TimestampAtom,
            'col2': kx.FloatAtom,
            'col3': kx.IntAtom},
            key = 'col1'
            )
    >>> qtab
    pykx.KeyedTable(pykx.q('
    col1| col2 col3
    ----| ---------
    '))
    >>> kx.q.meta(qtab)
    pykx.KeyedTable(pykx.q('
    c   | t f a
    ----| -----
    col1| p
    col2| f
    col3| i
    '))
    ```

    Create a `pykx.KeyedTable` with multiple primary keys.

    ```python
    >>> import pykx as kx
    >>> qtab = kx.schema.builder({
            'col1': kx.TimestampAtom,
            'col2': kx.SymbolAtom,
            'col3': kx.IntAtom,
            'col4': kx.List},
            key = ['col1', 'col2']
            )
    >>> qtab
    pykx.KeyedTable(pykx.q('
    col1 col2| col3 col4
    ---------| ---------
    '))
    >>> kx.q.meta(qtab)
    pykx.KeyedTable(pykx.q('
    c   | t f a
    ----| -----
    col1| p
    col2| s
    col3| i
    col4|
    '))
    ```
    """
    if not isinstance(schema, dict):
        raise Exception("'schema' argument should be a dictionary")
    if set(map(type, schema)) != {str}:
        raise Exception("'schema' keys must be of type 'str'")
    if key:
        if not all(isinstance(i, str) for i in key):
            raise Exception("Supplied 'key' must be a list of 'str' types only")
    columns = list(schema.keys())
    ktypes = list(schema.values())
    mapping = []
    idx=0
    for i in ktypes:
        if i == k.CharVector:
            raise Exception("Error: setting column to 'CharVector' is ambiguous, please use 'List' "
                            "for columns with rows containing multiple characters or 'CharAtom' if "
                            "your rows contain a single character")
        try:
            qconversion = _ktype_to_conversion[i]
        except KeyError as e:
            raise Exception("Error: " + str(e.__class__) + " raised for column " + columns[idx])
        idx+=1
        mapping.append(qconversion)
    qlists = q('{$[x~`;();x$()]}').each(mapping)
    qtab = q('{flip raze[x]!y}', columns, qlists)
    if key is not None:
        qtab = q('{raze[x]xkey y}', key, qtab)
    return(qtab)

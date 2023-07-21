"""Functionality for the manipulation and creation of schemas"""

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


def builder(schema: Dict,
            *,
            key: Optional[Union[str, List[str]]] = None
) -> k.K:
    """Generate an empty schema for a keyed or unkeyed table.

    Parameters:
        schema: The definition of the schema to be created mapping a 'str'
            to a `pykx.*` type object which is one of the types defined in
            `pykx._kytpe_to_conversion`.
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

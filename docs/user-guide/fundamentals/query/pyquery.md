---
title: Querying data using PyKX
description: Introduction to the concept of querying PyKX databases and tables
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, query, historical, SQL, qSQL
---

# Querying data using the query API with PyKX

_This page explains how to query your data with PyKX using the query API._

Before we get started the following dataset will be used throughout the remainder of this page.

Creating a sample table:

```python
>>> import pykx as kx
>>> kx.random.seed(42)
>>> trades = kx.Table(data={
        'sym': kx.random.random(100, ['AAPL', 'GOOG', 'MSFT']),
        'date': kx.random.random(100, kx.q('2022.01.01') + [0,1,2]),
        'price': kx.random.random(100, 1000.0),
        'size': kx.random.random(100, 100) 
    })

# Store the same table in q memory space to be able to demo queries on q variables
>>> kx.q['trades'] = trades
```

## Query basics

The PyKX [query API](../../../api/query.md) provides a Pythonic way to query kdb+ table. This API builds [qSQL](https://code.kx.com/q/basics/qsql/) queries in their [functional](https://code.kx.com/q/basics/funsql/) allowing you to query in-memory and on-disk data.

In the following sections we will introduce the functions, their arguments and how they can be used to perform queries of increasing complexity.

### Query Functions

Query functions describe the operations against in-memory and on-disk data which allow users to retrieve, update or delete data from these tables. Not all operations are supported against all table types, as such the following table provides a guide:

| Function | pykx.Table       | pykx.KeyedTable  | pykx.SplayedTable | pykx.PartitionedTable |
| :------- | :--------------- | :--------------- | :---------------- | :-------------------- |
| select   | :material-check: | :material-check: | :material-check:  | :material-check:      |
| exec     | :material-check: | :material-check: | :material-check:  | :material-close:      |
| update   | :material-check: | :material-check: | :material-minus:  | :material-close:      |
| delete   | :material-check: | :material-check: | :material-minus:  | :material-close:      |

For `pykx.SplayedTable` objects the denoted :material-minus: operations indicate that while applied queries will return a `pykx.Table` object the on-disk data will not be modified directly.

#### select()

[select()](../../../api/query.md#pykx.query.QSQL.select) builds on qSQL [select](https://code.kx.com/q/ref/select/).

Select should be used to query/filter data returning a [pykx.Table](../../../api/pykx-q-data/wrappers.md#pykx.wrappers.Table) or [pykx.KeyedTable](../../../api/pykx-q-data/wrappers.md#pykx.wrappers.KeyedTable).

```python
table.select(columns=None, where=None, by=None, inplace=False)
```

#### exec()

[exec()](../../../api/query.md#pykx.query.QSQL.exec) builds on qSQL [exec](https://code.kx.com/q/ref/exec/).

Exec is used to query tables but unlike Select it does not return tables. Instead this query type will return [pykx.Vector](../../../api/pykx-q-data/wrappers.md##pykx.wrappers.Vector), [pykx.Atom](../../../api/pykx-q-data/wrappers.md##pykx.wrappers.Atom), or [pykx.Dictionary](../../../api/pykx-q-data/wrappers.md##pykx.wrappers.Dictionary) will be returned depending on the query parameters.

For example if querying for data in a single column a vector will be returned, multiple columns will result in a dictionary mapping column name to value and when performing aggregations on a single column you may return an atom.

```python
table.exec(columns=None, where=None, by=None, inplace=False)
```

#### update()

[update()](../../../api/query.md#pykx.query.QSQL.update) builds on qSQL [update](https://code.kx.com/q/ref/update/).

Update returns the modified [pykx.Table](../../../api/pykx-q-data/wrappers.md#pykx.wrappers.Table) or [pykx.KeyedTable](../../../api/pykx-q-data/wrappers.md#pykx.wrappers.KeyedTable).

```python
table.update(columns=None, where=None, by=None, inplace=False)
```

#### delete()

[delete()](../../../api/query.md#pykx.query.QSQL.delete) builds on qSQL [delete](https://code.kx.com/q/ref/delete/).

Delete returns the modified [pykx.Table](../../../api/pykx-q-data/wrappers.md#pykx.wrappers.Table) or [pykx.KeyedTable](../../../api/pykx-q-data/wrappers.md#pykx.wrappers.KeyedTable).

```python
table.delete(columns=None, where=None, by=None, inplace=False)
```

!!! Note

    The following sections makes use of `kx.Column` objects which are only enabled in PyKX licensed mode. For unlicensed query examples using `str` objects see the [query API](../../../api/query.md) page.

### Query arguments

Querying data using this API refers to the four functions outlined above each which can take as arguments the following keyword parameters:

- `columns`
- `where`
- `by`*
- `inplace`

Outlined below these, arguments allow you to manipulate your data to filter for/update specific columns/rows in the case of a `where` clause, apply some analytics in the case of a `columns` clause or group data based on supplied conditions when discussing a `by` clause.

??? Note "by clause restrictions"

	The `by` clause is not supported when used with the `delete` query type

#### columns

The `columns` keyword provides the ability to access columnar data by name or apply analytics to the content of columns. In the following examples we will use various combinations of the `columns` keyword with `select`, `exec`, `update` and `delete` operations.

- `columns` can be passed a single column name without where conditions to retrieve or modify the content of that column:

=== "select"

	```python
	>>> trades.select(columns=kx.Column('sym'))
	pykx.Table(pykx.q('
	sym 
	----
	AAPL
	MSFT
	MSFT
	GOOG
	AAPL
	..
	'))
	>>> trades.select(columns=2 * kx.Column('price'))
	pykx.Table(pykx.q('
	price   
	--------
	291.2518
	1067.837
	34.35393
	1832.257
	280.0766
	..
	'))
	>>> trades.select(columns=kx.Column('price').max())
	pykx.Table(pykx.q('
	price   
	--------
	989.3873
	'))
	```

=== "delete"

	```python
	>>> trades.delete(columns=kx.Column('date'))
	pykx.Table(pykx.q('
	sym  price   
	-------------
	AAPL 145.6259
	MSFT 533.9187
	MSFT 17.17696
	GOOG 916.1286
	AAPL 140.0383
	..
	'))
	```

=== "exec"

	```python
	>>> trades.exec(columns=kx.Column('price'))
	pykx.FloatVector(pykx.q('145.6259 533.91..'))
	>>> trades.exec(columns=kx.Column('price').max())
	pykx.FloatAtom(pykx.q('989.3873'))
	>>> trades.exec(columns=2 * kx.Column('price'))
	pykx.FloatVector(pykx.q('291.2518 1067.83..'))
	```

=== "update"

	```python
	>>> trades.update(columns=(kx.Column('price') * 2).name('dpx'))
        pykx.Table(pykx.q('
        sym  date       price    size dpx
        --------------------------------------
        AAPL 2022.01.01 145.6259 19   291.2518
        MSFT 2022.01.02 533.9187 92   1067.837
        MSFT 2022.01.02 17.17696 7    34.35393
        GOOG 2022.01.03 916.1286 60   1832.257
        AAPL 2022.01.02 140.0383 54   280.0766
        ..
        '))
        >>> trades.update(columns=kx.Column('dpx', value=kx.Column('price') * 2))
        pykx.Table(pykx.q('
        sym  date       price    size dpx
        --------------------------------------
        AAPL 2022.01.01 145.6259 19   291.2518
        MSFT 2022.01.02 533.9187 92   1067.837
        MSFT 2022.01.02 17.17696 7    34.35393
        GOOG 2022.01.03 916.1286 60   1832.257
        AAPL 2022.01.02 140.0383 54   280.0766
        ..
        '))
	```

- Multiple columns can be modified, retrieved or aggregations applied by using queries can be returned and have aggregations/operation performed on them.

=== "select"

	```python
	>>> trades.select(columns=kx.Column('date') & kx.Column('sym'))
	pykx.Table(pykx.q('
	date       sym 
	---------------
	2022.01.01 AAPL
	2022.01.02 MSFT
	2022.01.02 MSFT
	2022.01.03 GOOG
	2022.01.02 AAPL
	..
	'))
	>>> trades.select(columns=kx.Column('price').neg() & kx.Column('date') + 1)
	pykx.Table(pykx.q('
	price     date      
	--------------------
	-145.6259 2022.01.02
	-533.9187 2022.01.03
	-17.17696 2022.01.03
	-916.1286 2022.01.04
	-140.0383 2022.01.03
	..
	'))
	>>> trades.select(columns=kx.Column('price').last() & kx.Column('date').last())
	pykx.Table(pykx.q('
	price    date      
	-------------------
	975.5566 2022.01.01
	'))
	```

=== "delete"

	```python
	>>> trades.delete(columns=kx.Column('date') & kx.Column('sym'))
	pykx.Table(pykx.q('
	price    size
	-------------
	145.6259 19  
	533.9187 92  
	17.17696 7   
	916.1286 60  
	140.0383 54
	..
	'))
	'))
	```

=== "exec"

	```python
	>>> trades.exec(columns=kx.Column('date') & kx.Column('price'))
	pykx.Dictionary(pykx.q('
	date | 2022.01.01 2022.01.02 2020.0..
	price| 145.6259   533.9187   17.176..
	'))
	```

- Columns can be named by using the `name` method on you column objects

=== "select"

	```python
	>>> trades.select(columns=kx.Column('price').max().name('maxPrice'))
	pykx.Table(pykx.q('
	maxPrice
	--------
	989.3873
	'))
	```

=== "exec"

	```python
	>>> trades.exec(columns=(2 * kx.Column('price')).name('multiPrice') &
	...                     kx.Column('sym').name('symName'))
	pykx.Dictionary(pykx.q('
	multiPrice| 291.2518 1067.837 34.35..
	symName   | AAPL     MSFT     MSFT ..
	'))
	```

=== "update"

	In the case of update renaming a column will add a new column with the associated name

	```python
	>>> trades.update(columns=kx.Column('price').name('priceCol'))
	pykx.Table(pykx.q('
	sym  date       price    size priceCol
	--------------------------------------
	AAPL 2022.01.01 145.6259 19   145.6259
	MSFT 2022.01.02 533.9187 92   533.9187
	MSFT 2022.01.02 17.17696 7    17.17696
	GOOG 2022.01.03 916.1286 60   916.1286
	AAPL 2022.01.02 140.0383 54   140.0383
	..
	'))
	```

Finally as an alternative approach for renaming a dictionary can be used to control names of returned columns.

```python
>>> trades.select(columns={'maxPrice':kx.Column('price').max()})
pykx.Table(pykx.q('
maxPrice
--------
993.6284
'))
```

#### where

The [where phrase](https://code.kx.com/q/basics/qsql/#where-phrase) allows you to filter data to retrieve, update, delete or apply functions on rows of a table which meet the specified conditions

By default this parameter has a value `None` which is equivalent to not filtering the data. This parameter is supported for all query types.

- Filter data meeting a specified criteria on one column

=== "select"

	```python
	>>> trades.select(where=kx.Column('price') > 500)
	pykx.Table(pykx.q('
	sym  date       price    size
	-----------------------------
	MSFT 2022.01.02 533.9187 92  
	GOOG 2022.01.03 916.1286 60  
	AAPL 2022.01.02 876.0921 37  
	AAPL 2022.01.03 952.2597 53  
	MSFT 2022.01.02 603.3717 6   
	..
	'))
	>>> trades.select(where=kx.Column('price') < kx.Column('size'))
	pykx.Table(pykx.q('
	sym  date       price    size
	-----------------------------
	MSFT 2022.01.03 46.11964 93  
	GOOG 2022.01.02 16.11913 81  
	AAPL 2022.01.03 28.98133 97  
	AAPL 2022.01.02 44.09906 91  
	GOOG 2022.01.01 12.58364 33  
	'))
	>>> trades.select(where=kx.Column('price') == kx.Column('price').max())
	pykx.Table(pykx.q('
	sym  date       price    size
	-----------------------------
	MSFT 2022.01.01 989.3873 42  
	'))
	```

=== "delete"

	```python
	>>> trades.delete(where=kx.Column('price') > 500)
	pykx.Table(pykx.q('
	sym  date       price    size
	-----------------------------
	AAPL 2022.01.01 145.6259 19  
	MSFT 2022.01.02 17.17696 7   
	AAPL 2022.01.02 140.0383 54  
	MSFT 2022.01.03 282.4291 98  
	MSFT 2022.01.03 46.11964 93 
	..
	'))
	>>> trades.delete(where=kx.Column('price') > kx.Column('size'))
	pykx.Table(pykx.q('
	sym  date       price    size
	-----------------------------
	MSFT 2022.01.03 46.11964 93  
	GOOG 2022.01.02 16.11913 81  
	AAPL 2022.01.03 28.98133 97  
	AAPL 2022.01.02 44.09906 91  
	GOOG 2022.01.01 12.58364 33  
	'))
	```

=== "update"

	```python
	>>> trades.update(columns = 2 * kx.Column('price'),
        ...               where=kx.Column('price') > 500)
	pykx.Table(pykx.q('
	sym  date       price    size
	-----------------------------
	AAPL 2022.01.01 145.6259 19  
	MSFT 2022.01.02 1067.837 92  
	MSFT 2022.01.02 17.17696 7   
	GOOG 2022.01.03 1832.257 60  
	AAPL 2022.01.02 140.0383 54  
	..
	'))
	```

=== "exec"

	```python
	>>> trades.exec(columns = kx.Column('size'), where = kx.Column('price') > 900)
	pykx.LongVector(pykx.q('60 53 61 41 98 12 41 12 23 42 18 76 73 55'))
	```

- Using `&` or passing a list of `pykx.Column` objects will allow multiple filters to be passed

=== "select"
	
	```python
	>>> trades.select(where=(kx.Column('sym') == 'GOOG') & (kx.Column('date') == datetime.date(2022, 1, 1)))
	pykx.Table(pykx.q('
	sym  date       price   
	------------------------
	GOOG 2022.01.01 480.9078
	GOOG 2022.01.01 454.5668
	GOOG 2022.01.01 790.2208
	GOOG 2022.01.01 296.6022
	GOOG 2022.01.01 727.6113
	..
	'))
	>>> trades.select(where=[
	...     kx.Column('sym') == 'GOOG',
        ...     kx.Column('date') == datetime.date(2022, 1, 1)
	...     ])
	>>> from datetime import date
	>>> trades.select(columns=kx.Column('price').wavg(kx.Column('size')),
	...               where=(kx.Column('sym') == 'GOOG') & (kx.Column('date') == date(2022, 1, 1)))
	pykx.Table(pykx.q('
	price  
	-------
	44.7002
	'))
	```

=== "delete"

	```python
	>>> from datetime import date
	>>> trades.delete(where=(kx.Column('sym') == 'AAPL') & (kx.Column('date') == date(2022, 1, 1)))
	pykx.Table(pykx.q('
	sym  date       price    size
	-----------------------------
	MSFT 2022.01.02 533.9187 92  
	MSFT 2022.01.02 17.17696 7   
	GOOG 2022.01.03 916.1286 60  
	AAPL 2022.01.02 140.0383 54  
	MSFT 2022.01.03 282.4291 98
	..
	'))
	```

=== "update"

	```python
	>>> from datetime import date
	>>> trades.update(
	...     columns=2*kx.Column('price'),
	...     where=(kx.Column('sym') == 'AAPL') & (kx.Column('date') == date(2022, 1, 1)))
	pykx.Table(pykx.q('
	sym  date       price    size
	-----------------------------
	AAPL 2022.01.01 291.2518 19  
	MSFT 2022.01.02 533.9187 92  
	MSFT 2022.01.02 17.17696 7   
	GOOG 2022.01.03 916.1286 60  
	AAPL 2022.01.02 140.0383 54  
	..
	'))
	```

=== "exec"

	```python
	>>> from datetime import date
	>>> trades.exec(
	...     columns=kx.Column('price') & kx.Column('date'),
	...     where=(kx.Column('sym') == 'AAPL') & (kx.Column('date') == date(2022, 1, 1)))
	pykx.Dictionary(pykx.q('
	price| 145.6259   636.4009   8..
	date | 2022.01.01 2022.01.01 2..
	'))
	```

#### by

The [by phrase](https://code.kx.com/q/basics/qsql/#aggregates) allows you to apply aggregations or manipulate data grouping the data `by` specific conditions.

By default this parameter has a value `None` which is equivalent to not grouping your data. This parameter is supported for `select`, `exec` and `update` type queries.

When both a `columns` and `by` clause are passed to a select query without use of an aggregation function then each row contains vectors of data related to the `by` columns.

```python
>>> trades.select(columns=kx.Column('price'), by=kx.Column('date') & kx.Column('sym'))
pykx.KeyedTable(pykx.q('
date       sym | price                                                       ..
---------------| ------------------------------------------------------------..
2022.01.01 AAPL| 131.6095 236.3145 140.4332 839.3869 843.3531 641.2171 104.81..
2022.01.01 GOOG| 480.9078 454.5668 790.2208 296.6022 727.6113 341.9665 609.77..
2022.01.01 MSFT| 556.9152 755.6175 865.9657 714.9804 179.5444 149.734 67.0821..
2022.01.02 AAPL| 441.8975 379.1373 659.8286 531.1731 975.3188 613.6512 603.99..
2022.01.02 GOOG| 446.898 664.8273 648.3929 240.1062 119.6 774.3718 449.4149 8..
2022.01.02 MSFT| 699.0336 387.7172 588.2985 725.8795 842.5805 646.37 593.7708..
2022.01.03 AAPL| 793.2503 621.7243 570.4403 626.2866 263.992 153.475 123.7397..
2022.01.03 GOOG| 586.263 777.3633 834.1404 906.9809 617.6205 179.6328 100.041..
2022.01.03 MSFT| 633.3324 39.47309 682.9453 867.1843 483.0873 851.2139 318.93..
'))
```

Adding an aggregation function allows this aggregation to be run on a column within the `by` phrase.

```python
>>> trades.select(columns=kx.Column('price').max(), by=kx.Column('date') & kx.Column('sym'))
pykx.KeyedTable(pykx.q('
date       sym | price   
---------------| --------
2022.01.01 AAPL| 843.3531
2022.01.01 GOOG| 790.2208
2022.01.01 MSFT| 865.9657
2022.01.02 AAPL| 975.3188
2022.01.02 GOOG| 886.0093
2022.01.02 MSFT| 993.6284
2022.01.03 AAPL| 843.9354
2022.01.03 GOOG| 914.6929
2022.01.03 MSFT| 867.1843
'))
```

Using a `by` clause within an update allows you to modify the values of the table conditionally based on your grouped criteria, for example:

```python
>>> trades.update(columns=kx.Column('price').wavg(kx.Column('size')).name('vwap'),
...               by=kx.Column('sym'))
pykx.Table(pykx.q('
sym  date       price    size vwap    
--------------------------------------
AAPL 2022.01.01 145.6259 19   56.09317
MSFT 2022.01.02 533.9187 92   40.46716
MSFT 2022.01.02 17.17696 7    40.46716
GOOG 2022.01.03 916.1286 60   52.721  
AAPL 2022.01.02 140.0383 54   56.09317
..
'))
```


??? Note "What happens without a columns clause"

	Using `by` without an associated `columns` clause will return the last row in the table for each column in the `by` phrase.

	```python
	>>> trades.select(by=kx.Column('sym'))
	pykx.KeyedTable(pykx.q('
	sym | date       price   
	----| -------------------
	AAPL| 2022.01.02 955.4843
	GOOG| 2022.01.02 886.0093
	MSFT| 2022.01.01 719.9879
	'))     
	```

#### inplace

The `inplace` keyword provides the ability for a user to overwrite the representation of the object which they are querying.
This functionality is set to `False` by default but will operate effectively on in-memory table objects for the `select`, `update` and `delete` query types.

If set to `True` the input table can be overwritten as follows

```python
>>> trades.delete(where=kx.Column('sym').isin(['AAPL']), inplace=True)
pykx.Table(pykx.q('
sym  date       price    size
-----------------------------
MSFT 2022.01.02 533.9187 92  
MSFT 2022.01.02 17.17696 7   
GOOG 2022.01.03 916.1286 60  
MSFT 2022.01.03 282.4291 98  
MSFT 2022.01.03 46.11964 93 
..
'))
>>> trades
pykx.Table(pykx.q('
sym  date       price    size
-----------------------------
MSFT 2022.01.02 533.9187 92
MSFT 2022.01.02 17.17696 7
GOOG 2022.01.03 916.1286 60
MSFT 2022.01.03 282.4291 98
MSFT 2022.01.03 46.11964 93
..
'))
```

### Query Types

While this page discusses primarily the Pythonic API for querying kdb+ tables locally. The following describes some of the other ways that queries can be completed

#### Local Queries

qSQL equivalent query for comparison:

```python
>>> kx.q('select from trades where price=max price')
pykx.Table(pykx.q('
sym  date       price   
------------------------
AAPL 2022.01.01 983.0794
'))
```

Access query API off the table object:

```python
>>> trades.select(where=kx.Column('price') == kx.Column('price').max())
pykx.Table(pykx.q('
sym  date       price   
------------------------
AAPL 2022.01.01 983.0794
'))
```

Direct use of the `kx.q.qsql` query APIs taking the table as a parameter:

```python
>>> kx.q.qsql.select(trades, where=kx.Column('price') == kx.Column('price').max())
pykx.Table(pykx.q('
sym  date       price   
------------------------
AAPL 2022.01.01 983.0794
'))
```

Passing a string will query the table of that name in q memory:

```python
>>> kx.q.qsql.select('trades', where=kx.Column('price') == kx.Column('price').max())
pykx.Table(pykx.q('
sym  date       price   
------------------------
AAPL 2022.01.01 983.0794
'))
```

#### Remote Queries

Queries can also be performed over [IPC](../../advanced/ipc.md) to remote servers.

```python
>>> conn = kx.SyncQConnection(port = 5000)
>>> conn.qsql.select('trades', where=kx.Column('price') == kx.Column('price').max())
pykx.Table(pykx.q('
sym  date       price   
------------------------
AAPL 2022.01.01 983.0794
'))
```

## Query Classes

### Column

See [pykx.Column](../../../api/pykx-q-data/wrappers.md#pykx.wrappers.Column) for full documentation on this class.

#### And operator `&`

Using `&` on two `Column` objects will return a `QueryPhrase` which describes the underlying construct which is used to query your table.

```python
>>> qp =(kx.Column('sym') == 'GOOG') & (kx.Column('price') > 500)
>>> type(qp)
<class 'pykx.wrappers.QueryPhrase'>
>>> qp._phrase
[[pykx.Operator(pykx.q('=')), 'sym', [pykx.SymbolAtom(pykx.q('`GOOG'))]], [pykx.Operator(pykx.q('>')), 'price', pykx.LongAtom(pykx.q('500'))]]
>>> trades.select(where=qp)
pykx.Table(pykx.q('
sym  date       price   
------------------------
GOOG 2022.01.03 976.1246
GOOG 2022.01.02 716.2858
GOOG 2022.01.03 872.5027
GOOG 2022.01.02 962.5156
GOOG 2022.01.01 589.7202
..
'))
```

Additional `Column` objects can `&` off a `QueryPhrase` to further build up more complex queries.

#### Or operator `|`

Using `|` on two `Column` objects will return a `Column` object.

```python
>>> c =(kx.Column('price') < 100) | (kx.Column('price') > 500)
>>> type(c)
<class 'pykx.wrappers.Column'>
>>> c._value
[pykx.Operator(pykx.q('|')), [pykx.Operator(pykx.q('<')), 'price', pykx.LongAtom(pykx.q('100'))], [pykx.Operator(pykx.q('>')), 'price', pykx.LongAtom(pykx.q('500'))]]
>>> trades.select(where=c)
pykx.Table(pykx.q('
sym  date       price   
------------------------
AAPL 2022.01.01 542.6371
AAPL 2022.01.01 77.57332
MSFT 2022.01.01 637.4637
GOOG 2022.01.03 976.1246
MSFT 2022.01.03 539.6816
..
'))
```

!!! Note "`or` operator `|` restriction"

	`Column` objects can not apply `|` off a `QueryPhrase`. Presently these are restricted only to operations on two `kx.Column` objects.

#### Python operators

The following Python operators can be used with the `Column` class to perform analysis on your data

| Python operator | q operation  | Magic method    |
| --------------- | ------------ | --------------- |
| `+`             | `+`          | `__add__`       |
| `-`             | `-`          | `__sub__`       |
| `-`             | `-`          | `__rsub__`      |
| `*`             | `*`          | `__mul__`       |
| `/`             | `%`          | `__truediv__`   |
| `/`             | `%`          | `__rtruediv__`  |
| `//`            | `div`        | `__floordiv__`  |
| `//`            | `div`        | `__rfloordiv__` |
| `%`             | `mod`        | `___mod__`      |
| `**`            | `xexp`       | `__pow__`       |
| `==`            | `=`          | `__eq__`        |
| `!=`            | `<>`         | `__ne__`        |
| `>`             | `>`          | `__gt__`        |
| `>=`            | `>=`         | `__ge__`        |
| `<`             | `<`          | `__lt__`        |
| `<=`            | `<=`         | `__le__`        |
| `pos`           | `abs`        | `__pos__`       |
| `neg`           | `neg`        | `__neg__`       |
| `floor`         | `floor`      | `__floor__`     |
| `ceil`          | `ceiling`    | `__ceil__`      |
| `abs`           | `abs`        | `__abs__`       |

The following are a few examples of this various operations in use

1. Finding rows where `price` is greater than or equal to half the average price:

	```python
	>>> trades.select(where=kx.Column('price') >= kx.Column('price').avg() / 2)
	pykx.Table(pykx.q('
	sym  date       price   
	------------------------
	AAPL 2022.01.01 542.6371
	MSFT 2022.01.01 637.4637
	GOOG 2022.01.03 976.1246
	MSFT 2022.01.03 539.6816
	GOOG 2022.01.02 716.2858
	..
	'))
	```

2. Apply the `math` libraries `floor` operation on the column price updating it's value

	```python
	>>> from math import floor
	>>> trades.update(floor(kx.Column('price')))
	pykx.Table(pykx.q('
	sym  date       price size
	--------------------------
	AAPL 2022.01.01 145   19  
	MSFT 2022.01.02 533   92  
	MSFT 2022.01.02 17    7   
	GOOG 2022.01.03 916   60  
	AAPL 2022.01.02 140   54  
	..
	'))
	```

#### PyKX methods

In addition to support for the Python operators outlined above PyKX provides a number of analytic methods and properties for the `kx.Column` objects. In total there are more than 100 analytic methods supported ranging from a basic method to retrieve the maximum value of a column, to more complex analytics for the calculation of the weighted average between two vectors.

The following drop-down provides a list of the supported methods, with full details on the API page [here](../../../api/columns.md).

??? Note "Supported methods"

	[`abs`](../../../api/columns.md#pykx.wrappers.Column.abs), [`acos`](../../../api/columns.md#pykx.wrappers.Column.acos), [`asc`](../../../api/columns.md#pykx.wrappers.Column.asc), [`asin`](../../../api/columns.md#pykx.wrappers.Column.asin), [`atan`](../../../api/columns.md#pykx.wrappers.Column.atan), [`avg`](../../../api/columns.md#pykx.wrappers.Column.avg), [`avgs`](../../../api/columns.md#pykx.wrappers.Column.avgs), [`ceiling`](../../../api/columns.md#pykx.wrappers.Column.ceiling), [`cor`](../../../api/columns.md#pykx.wrappers.Column.cor), [`cos`](../../../api/columns.md#pykx.wrappers.Column.cos), [`count`](../../../api/columns.md#pykx.wrappers.Column.count), [`cov`](../../../api/columns.md#pykx.wrappers.Column.cov), [`cross`](../../../api/columns.md#pykx.wrappers.Column.cross), [`deltas`](../../../api/columns.md#pykx.wrappers.Column.deltas), [`desc`](../../../api/columns.md#pykx.wrappers.Column.desc), [`dev`](../../../api/columns.md#pykx.wrappers.Column.dev), [`differ`](../../../api/columns.md#pykx.wrappers.Column.differ), [`distinct`](../../../api/columns.md#pykx.wrappers.Column.distinct), [`div`](../../../api/columns.md#pykx.wrappers.Column.div), [`exp`](../../../api/columns.md#pykx.wrappers.Column.exp), [`fills`](../../../api/columns.md#pykx.wrappers.Column.fills), [`first`](../../../api/columns.md#pykx.wrappers.Column.first), [`floor`](../../../api/columns.md#pykx.wrappers.Column.floor), [`null`](../../../api/columns.md#pykx.wrappers.Column.null), [`iasc`](../../../api/columns.md#pykx.wrappers.Column.iasc), [`idesc`](../../../api/columns.md#pykx.wrappers.Column.idesc), [`inter`](../../../api/columns.md#pykx.wrappers.Column.inter), [`isin`](../../../api/columns.md#pykx.wrappers.Column.isin), [`last`](../../../api/columns.md#pykx.wrappers.Column.last), [`like`](../../../api/columns.md#pykx.wrappers.Column.like), [`log`](../../../api/columns.md#pykx.wrappers.Column.log), [`lower`](../../../api/columns.md#pykx.wrappers.Column.lower), [`ltrim`](../../../api/columns.md#pykx.wrappers.Column.ltrim), [`mavg`](../../../api/columns.md#pykx.wrappers.Column.mavg), [`max`](../../../api/columns.md#pykx.wrappers.Column.max), [`maxs`](../../../api/columns.md#pykx.wrappers.Column.maxs), [`mcount`](../../../api/columns.md#pykx.wrappers.Column.mcount), [`md5`](../../../api/columns.md#pykx.wrappers.Column.md5), [`mdev`](../../../api/columns.md#pykx.wrappers.Column.mdev), [`med`](../../../api/columns.md#pykx.wrappers.Column.med), [`min`](../../../api/columns.md#pykx.wrappers.Column.min), [`mins`](../../../api/columns.md#pykx.wrappers.Column.mins), [`mmax`](../../../api/columns.md#pykx.wrappers.Column.mmax), [`mmin`](../../../api/columns.md#pykx.wrappers.Column.mmin), [`mod`](../../../api/columns.md#pykx.wrappers.Column.mod), [`msum`](../../../api/columns.md#pykx.wrappers.Column.msum), [`neg`](../../../api/columns.md#pykx.wrappers.Column.neg), [`prd`](../../../api/columns.md#pykx.wrappers.Column.prd), [`prds`](../../../api/columns.md#pykx.wrappers.Column.prds), [`prev`](../../../api/columns.md#pykx.wrappers.Column.prev), [`rank`](../../../api/columns.md#pykx.wrappers.Column.rank), [`ratios`](../../../api/columns.md#pykx.wrappers.Column.ratios), [`reciprocal`](../../../api/columns.md#pykx.wrappers.Column.reciprocal), [`reverse`](../../../api/columns.md#pykx.wrappers.Column.reverse), [`rotate`](../../../api/columns.md#pykx.wrappers.Column.rotate), [`rtrim`](../../../api/columns.md#pykx.wrappers.Column.rtrim), [`scov`](../../../api/columns.md#pykx.wrappers.Column.scov), [`sdev`](../../../api/columns.md#pykx.wrappers.Column.sdev), [`signum`](../../../api/columns.md#pykx.wrappers.Column.signum), [`sin`](../../../api/columns.md#pykx.wrappers.Column.sin), [`sqrt`](../../../api/columns.md#pykx.wrappers.Column.sqrt), [`string`](../../../api/columns.md#pykx.wrappers.Column.string), [`sum`](../../../api/columns.md#pykx.wrappers.Column.sum), [`sums`](../../../api/columns.md#pykx.wrappers.Column.sums), [`svar`](../../../api/columns.md#pykx.wrappers.Column.svar), [`tan`](../../../api/columns.md#pykx.wrappers.Column.tan), [`trim`](../../../api/columns.md#pykx.wrappers.Column.trim), [`union`](../../../api/columns.md#pykx.wrappers.Column.union), [`upper`](../../../api/columns.md#pykx.wrappers.Column.upper), [`var`](../../../api/columns.md#pykx.wrappers.Column.var), [`wavg`](../../../api/columns.md#pykx.wrappers.Column.wavg), [`within`](../../../api/columns.md#pykx.wrappers.Column.within), [`wsum`](../../../api/columns.md#pykx.wrappers.Column.wsum), [`xbar`](../../../api/columns.md#pykx.wrappers.Column.xbar), [`xexp`](../../../api/columns.md#pykx.wrappers.Column.xexp), [`xlog`](../../../api/columns.md#pykx.wrappers.Column.xlog), [`xprev`](../../../api/columns.md#pykx.wrappers.Column.xprev), [`hour`](../../../api/columns.md#pykx.wrappers.Column.hour), [`minute`](../../../api/columns.md#pykx.wrappers.Column.minute), [`date`](../../../api/columns.md#pykx.wrappers.Column.date), [`year`](../../../api/columns.md#pykx.wrappers.Column.year), [`month`](../../../api/columns.md#pykx.wrappers.Column.month), [`day`](../../../api/columns.md#pykx.wrappers.Column.day), [`second`](../../../api/columns.md#pykx.wrappers.Column.second), [`add`](../../../api/columns.md#pykx.wrappers.Column.add), [`name`](../../../api/columns.md#pykx.wrappers.Column.name), [`average`](../../../api/columns.md#pykx.wrappers.Column.average), [`cast`](../../../api/columns.md#pykx.wrappers.Column.cast), [`correlation`](../../../api/columns.md#pykx.wrappers.Column.correlation), [`covariance`](../../../api/columns.md#pykx.wrappers.Column.covariance), [`divide`](../../../api/columns.md#pykx.wrappers.Column.divide), [`drop`](../../../api/columns.md#pykx.wrappers.Column.drop), [`fill`](../../../api/columns.md#pykx.wrappers.Column.fill), [`index_sort`](../../../api/columns.md#pykx.wrappers.Column.index_sort), [`join`](../../../api/columns.md#pykx.wrappers.Column.join), [`len`](../../../api/columns.md#pykx.wrappers.Column.len), [`modulus`](../../../api/columns.md#pykx.wrappers.Column.modulus), [`multiply`](../../../api/columns.md#pykx.wrappers.Column.multiply), [`next_item`](../../../api/columns.md#pykx.wrappers.Column.next_item), [`previous_item`](../../../api/columns.md#pykx.wrappers.Column.previous_item), [`product`](../../../api/columns.md#pykx.wrappers.Column.product), [`products`](../../../api/columns.md#pykx.wrappers.Column.products), [`sort`](../../../api/columns.md#pykx.wrappers.Column.sort), [`subtract`](../../../api/columns.md#pykx.wrappers.Column.subtract), [`take`](../../../api/columns.md#pykx.wrappers.Column.take), [`value`](../../../api/columns.md#pykx.wrappers.Column.value) and [`variance`](../../../api/columns.md#pykx.wrappers.Column.variance).

The following provides a complex example of a user generated query to calculate trade statistics and time-weighted average spread information associated with a Trade and Quote tables making use of the following methods.

- [`distinct`](../../../api/columns.md#pykx.wrappers.Column.distinct)
- [`next_item`](../../../api/columns.md#pykx.wrappers.Column.next_item)
- [`wavg`](../../../api/columns.md#pykx.wrappers.Column.wavg)
- [`max`](../../../api/columns.md#pykx.wrappers.Column.max)
- [`min`](../../../api/columns.md#pykx.wrappers.Column.min)
- [`isin`](../../../api/columns.md#pykx.wrappers.Column.isin)
- [`within`](../../../api/columns.md#pykx.wrappers.Column.within)
- [`avg`](../../../api/columns.md#pykx.wrappers.Column.avg)
- [`name`](../../../api/columns.md#pykx.wrappers.Column.name)

```python
def generate_twap(trade, quote, start_time, end_time, syms = None):
    if syms is None:
        syms = trade.exec(kx.Column('sym').distinct())

    quote_metrics = quote.select(
        columns = (kx.Column('ask') - kx.Column('bid')).avg().name('avg_spread') &
                  (kx.Column('time').next_item() - kx.Column('time')).wavg(kx.Column('ask') - kx.Column('bid')).name('twa_spread') &
                  ((kx.Column('asize') + kx.Column('bsize')).avg().name('avg_size') * 0.5) &
                  (kx.Column('time').next_item() - kx.Column('time')).avg().name('avg_duration'),
        by = kx.Column('sym'),
        where = kx.Column('sym').isin(syms) & kx.Column('time').within(start_time, end_time)
        )

    trade_metrics = trade.select(
        columns = (2 * kx.Column('price').dev()).name('std_dev') &
                  (kx.Column('time').next_item() - kx.Column('time')).wavg(kx.Column('price')).name('std_dev') &
                  kx.Column('price').max().name('max_price') &
                  kx.Column('price').min().name('min_price') &
                  kx.Column('size').wavg(kx.Column('price')).name('vwap'),
        by = kx.Column('sym'),
        where = kx.Column('sym').isin(syms) & kx.Column('time').within(start_time, end_time)
        )

    return kx.q.uj(quote_metrics, trade_metrics)
```

### Variable

See [pykx.Variable](../../../api/pykx-q-data/wrappers.md#pykx.wrappers.Variable) for full documentation on this class.

In some cases when operating at the interface of q and Python analytics you may wish to perform a comparison or analytic which makes use of a named variable from `q`.

The following example shows this in action

```python
>>> kx.q['filter']='GOOG'
>>> trades.select(where=kx.Column('sym') == kx.Variable('filter'))
pykx.Table(pykx.q('
sym  date       price
------------------------
GOOG 2022.01.03 976.1246
GOOG 2022.01.02 716.2858
GOOG 2022.01.03 872.5027
GOOG 2022.01.02 962.5156
GOOG 2022.01.01 589.7202
..
'))
```

## Advanced features

### Custom Functions

While there is an extensive list of functions/analytics that are supported by the API it does not cover all analytics that you, or users of an extension you are writing may need.

To facilitate this you have access to the [pykx.register.column_function](../../../api/pykx-q-data/register.md#pykx.register.column_function), this function provides the ability to define methods off your defined `pykx.Column` objects. This function should take the column on which the function is being performed as it's first argument and the `call` method should be used to apply your analytic.

The `call` method takes as it's first argument the function you wish to apply and can take multiple positional arguments, 

For example take the following cases:

- Define a function applying a min-max scaling against the price column of a table

	```python
	>>> def min_max_scaler(column):
	...     return column.call('{(x-minData)%max[x]-minData:min x}')
	>>> kx.register.column_function('minmax', min_max_scaler)
	>>> trades.update(kx.Column('price').minmax().name('scaled_price'))
	pykx.Table(pykx.q('
	sym  date       price    size scaled_price
	------------------------------------------
	MSFT 2022.01.02 533.9187 92   0.5337153   
	MSFT 2022.01.02 17.17696 7    0.004702399 
	GOOG 2022.01.03 916.1286 60   0.9250016   
	MSFT 2022.01.03 282.4291 98   0.2762535   
	MSFT 2022.01.03 46.11964 93   0.03433238
	..
	'))
	```

- Define a function which multiplies two columns together and calculates the log returning the result

	```python
	>>> def log_multiply(column1, column2):
	...     return column1.call('{log x*y}', column2)
	>>> kx.register.column_function('log_multiply', log_multiply)
	>>> trades.select(kx.Column('price').log_multiply(kx.Column('size')))
	pykx.Table(pykx.q('
	price   
	--------
	10.80203
	4.789479
	10.9145 
	10.22839
	8.363838
	..
	'))
	```

### fby queries

Complex queries often require the application of a function on data grouped by some condition, in many cases the application of a `by` clause will be sufficient to get the information you need, however you will run into cases where you need to filter-by a certain condition.

Take for example the case where you want to find the stock information by symbol where the price is the maximum price

```python
>>> trades.select(where=kx.Column('price') == kx.Column.fby(kx.Column('sym'), kx.q.max, kx.Column('price')))
pykx.Table(pykx.q('
sym  date       price    size
-----------------------------
MSFT 2022.01.03 977.1655 92  
AAPL 2022.01.02 996.8898 20  
GOOG 2022.01.03 971.9498 47  
'))
```

### Using iterators

Not all analytics that you may wish to run on your table will expect to take the full content of a column(s) as input, fo
r example in some cases you may wish to apply an analytic on each row of a column. While operations which rely on iterato
rs may be slower than purely vectorised operations they may be necessary.

PyKX supports the following iterators, a number of examples are provided below

| Iterator | Type        | Link                                                                             |
| :------- | :---------- | :------------------------------------------------------------------------------- |
| `each`   | map         | [Each](https://code.kx.com/q/ref/maps/#each)                                     |
| `peach`  | map         | [Peach](https://code.kx.com/q/ref/maps/#peach-keyword)                           |
| `\:`     | map         | [Each Left](https://code.kx.com/q/ref/maps/#each-left-and-each-right)            |
| `/:`     | map         | [Each Right](https://code.kx.com/q/ref/maps/#each-left-and-each-right)           |
| `\:/:`   | map         | [Each Left-Each Right](https://code.kx.com/q/ref/maps/#each-left-and-each-right) |
| `/:\:`   | map         | [Each Right-Each Left](https://code.kx.com/q/ref/maps/#each-left-and-each-right) |
| `'`      | map         | [Case](https://code.kx.com/q/ref/maps/#case)                                     |
| `':`     | map         | [Each Prior](https://code.kx.com/q/ref/maps/#each-prior)                         |
| `/`      | accumulator | [Over](https://code.kx.com/q/ref/over/)                                          |
| `\`      | accumulator | [Scan](https://code.kx.com/q/ref/scan/)                                          |

#### Example 1

Calculate the maximum value of each row of a column `x`

```python
>>> table = kx.Table(data={'x': [[10, 5, 4], [20, 30, 50], [1, 2, 3]]})
pykx.Table(pykx.q('
x       
--------
10 5  4 
20 30 50
1  2  3 
'))
>>> table.select(kx.Column('x').max(iterator='each'))
pykx.Table(pykx.q('
x 
--
10
50
3 
'))
```

#### Example 2

Join the characters associated from two columns row wise using the `'` iterator

```python
>>> table = kx.Table(data={'x': b'abc', 'y': b'def'})
pykx.Table(pykx.q('
x y
---
a d
b e
c f
'))
>>> table.select(kx.Column('x').join(kx.Column('y'), iterator="'"))
pykx.Table(pykx.q('
x   
----
"ad"
"be"
"cf"
'))
```

#### Example 3

Join the characters `"_xy"` to all rows in a column `x`

```python
>>> table = kx.Table(data={'x': b'abc', 'y': b'def'})
pykx.Table(pykx.q('
x y
---
a d
b e
c f
'))
>>> table.select(kx.Column('x').join(b"_xy", iterator='\\:'))
pykx.Table(pykx.q('
x     
------
"a_xy"
"b_xy"
"c_xy"
'))
```

## Next Steps

Now that you have learnt how to query your data using the Pythonic API you may be interested in other methods for querying your data:

- If you wish to query your data using SQL, you can follow the introduction to this functionality [here](./sql.md).
- If you want to upskill and learn how to query directly using q you can follow [this page](./qquery.md).
- To learn how to make your queries more performant following the tips and tricks [here](./perf.md).

For some further reading, here are some related topics:

- If you don't have a historical database available see [here](../../advanced/database/index.md).
- To learn about creating PyKX Table objects see [here](../../../examples/interface-overview.ipynb).

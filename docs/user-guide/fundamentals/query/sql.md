---
title: Querying data using SQL with PyKX
description: Introduction to querying data using SQL with PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, query, historical, SQL, qSQL
---

# Querying data using SQL with PyKX

_This page explains how to query your data with PyKX using SQL._

PyKX exposes a wrapper around the [KX Insights Core ANSI SQL interface](https://code.kx.com/insights/core/sql.html).
This allows SQL to be used to query in-memory and on-disk data.

The interface is accessed through the `kx.q.sql` class or via the `sql` method on table type objects. Full documentation of the class is included [here](../../../api/query.md#pykx.query.SQL).

## Loading the SQL interface

When you `import pykx as kx` an attempt will be made to load the SQL interface. If this fails you will see:

```python
WARN: Failed to load KX Insights Core library 's.k'.
```

To debug this you can set the [configuration option](../../configuration.md) `PYKX_DEBUG_INSIGHTS_LIBRARIES` before importing PyKX:

```python
import os
os.environ['PYKX_DEBUG_INSIGHTS_LIBRARIES'] = 'true'
import pykx as kx
```

This will print a more detailed error message, for example:

```python
PyKXWarning: Failed to load KX Insights Core library 's.k': s.k_. OS reports: No such file or directory
```

## Querying tables using SQL

Creating a sample table:

```python
>>> import pykx as kx
>>> trades = kx.Table(data={
        'sym': kx.random.random(100, ['AAPL', 'GOOG', 'MSFT']),
        'date': kx.random.random(100, kx.q('2022.01.01') + [0,1,2]),
        'price': kx.random.random(100, 1000.0) 
    })

>>> kx.q['trades'] = trades
```

Query a table by name:

```python
>>> kx.q.sql('select * from trades')
pykx.Table(pykx.q('
sym  date       price   
------------------------
GOOG 2022.01.02 805.0147
AAPL 2022.01.03 847.6275
AAPL 2022.01.03 329.8159
GOOG 2022.01.02 982.5155
MSFT 2022.01.02 724.9456
..
'))
```

Query a [pykx.Table](../../../api/pykx-q-data/wrappers.md#pykx.wrappers.Table) instance by injecting it as the first argument using `$n` syntax:

```python
>>> kx.q.sql('select * from $1', trades)
pykx.Table(pykx.q('
sym  date       price   
------------------------
GOOG 2022.01.02 805.0147
AAPL 2022.01.03 847.6275
AAPL 2022.01.03 329.8159
GOOG 2022.01.02 982.5155
MSFT 2022.01.02 724.9456
..
'))
```

Similarly you can use argument injection when using the `sql` method on your `trades` table object as follows:

```python
>>> trades.sql('select * from $1')
pykx.Table(pykx.q('
sym  date       price   
------------------------
GOOG 2022.01.02 805.0147
AAPL 2022.01.03 847.6275
AAPL 2022.01.03 329.8159
GOOG 2022.01.02 982.5155
MSFT 2022.01.02 724.9456
..
'))
```

Passing multiple arguments using `$n` syntax:

```python
>>> from datetime import date
>>> kx.q.sql('select * from trades where date = $1 and price < $2', date(2022, 1, 2), 500.0)
pykx.Table(pykx.q('
sym  date       price   
------------------------
GOOG 2022.01.02 214.9847
AAPL 2022.01.02 126.2957
AAPL 2022.01.02 184.4151
AAPL 2022.01.02 217.0378
GOOG 2022.01.02 423.6121
..
'))
```

## Next Steps

Now that you have learnt the fundamentals of how to query your data using the SQL API you may be interested in:

- To optimize frequently called SQL queries the [prepare](../../../api/query.md#pykx.query.SQL.prepare) and [execute](../../../api/query.md#pykx.query.SQL.execute) methods can be used to separate SQL parsing from query execution as detailed [here](https://code.kx.com/insights/1.10/core/sql.html#prepare-and-execute).
- If you want to query your data in a more Python-first way follow the guide [here](./pyquery.md).
- If you want to query your data in q follow the guide [here](./qquery.md).
- To learn how to make your queries more performant following the tips and tricks [here](./perf.md).

For some further reading, here are some related topics:

- If you don't have a historical database available see [here](../../advanced/database/index.md).
- To learn about creating PyKX Table objects see [here](../../../examples/interface-overview.ipynb).

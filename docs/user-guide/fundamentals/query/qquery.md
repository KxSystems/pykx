---
title: Querying data using PyKX
description: Introduction to the concept of querying PyKX databases and tables
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, query, historical, SQL, qSQL
---

# Querying data using qSQL with PyKX

_This page explains how to query your data with PyKX using qSQL._

## Querying tables using qSQL

Creating a sample table:

```python
>>> import pykx as kx
>>> trades = kx.Table(data={
        'sym': kx.random.random(100, ['AAPL', 'GOOG', 'MSFT']),
        'date': kx.random.random(100, kx.q('2022.01.01') + [0,1,2]),
        'price': kx.random.random(100, 1000.0) 
    })

>>> # Assign the table to a named object in q memory to allow name based query later
>>> kx.q['trades'] = trades
```

Query a table, by name, using [qSQL](https://code.kx.com/q/basics/qsql/):

```python
>>> kx.q('select from trades')
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

Query a [pykx.Table](../../../api/pykx-q-data/wrappers.md#pykx.wrappers.Table) [passing it as an argument](../../../user-guide/fundamentals/evaluating.md#a2-application-of-functions-taking-multiple-arguments):

```q
>>> kx.q('{select from x}', trades)
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

Passing multiple arguments:

```python
>>> from datetime import date
>>> kx.q('{[x;y] select from trades where date = x, price < y}', date(2022, 1, 2), 500.0)
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

Now that you have learnt how to query your data using q you may be interested in other methods for querying your data:

- If you want to query your data in a more Python-first way follow the guide [here](./pyquery.md).
- If you wish to query your data using SQL, you can follow the introduction to this functionality [here](./sql.md).
- To learn how to make your queries more performant following the tips and tricks [here](./perf.md).

For some further reading, here are some related topics:

- If you don't have a historical database available see [here](../../advanced/database/index.md).
- To learn about creating PyKX Table objects see [here](../../../examples/interface-overview.ipynb).

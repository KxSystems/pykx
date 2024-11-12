---
title: Querying data using PyKX
description: Introduction to the concept of querying PyKX databases and tables
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, query, historical, SQL, qSQL
---

# Query performance considerations using PyKX

_This page explains how to efficiently query your data using PyKX._

## qSQL

The whitepapers detailed below outline optimizations which can be useful for qSQL queries. However, the core lessons/concepts which apply in the [q query](./qquery.md) case also apply to using the [Pythonic query API](./pyquery.md) and [SQL](./sql.md) modes:

- [Columnar database and query optimization](https://code.kx.com/q/wp/columnar-database/)
- [kdb+ query scaling](https://code.kx.com/q/wp/query-scaling/)

The following provides a tangible example of two impactful optimizations

### Parameter ordering

Assume we have a historical database generated using the functionality [here](../../advanced/database/index.md) partitioned on date. The query being performed will query for all data in the database based on date and symbol. The following queries align to those completed [here](https://code.kx.com/q/wp/columnar-database/#query-structure-example)

=== "Optimal query"

	```python
	trade.select(where = (kx.Column('date') == kx.DateAtom(2020, 1, 1)) &
	                     (kx.Column('sym') == 'IBM'))
	```

=== "Non-Optimal query"

	```python
	trade.select(where = (kx.Column('sym') == 'IBM') &
	                     (kx.Column('date') == kx.DateAtom(2020, 1, 1)))
	```

The following shows the scaling of queries based on the number of dates within the database

```q
         |   sym before date  |   date before sym
dates in |  time         size |  time         size
database |  (ms)          (b) |  (ms)          (b) 
---------|--------------------|----------------------
     1   |   470   75,499,920 |    78   75,499,984
     5   |   487   75,878,400 |    78   75,499,984
    10   |   931   75,880,624 |    78   75,499,984
    15   | 1,209   75,882,912 |    78   75,499,984
    20   | 1,438   75,885,072 |    78   75,499,984
```

### Applying Attributes

The following shows the performance difference between the application of a grouped-attribute on the `sym` column of an in-memory table.

```python
rtquote = quote.select(where = kx.Column('date').isin([kx.DateAtom(2020, 1, 1)]).grouped('sym')
rtquote.select(where = kx.Column('sym') == 'IBM')
```

The following shows the scaling of queries based on the number of rows on an in-memory table using only the `sym` column.

```q
            |      no attribute   |  grouped attribute 
    rows in |  time          size |  time          size 
      table |  (ms)           (b) |  (ms)           (b) 
-------------------------------------------------------
 25,000,000 |   119   301,990,304 |     8     2,228,848
 50,000,000 |   243   603,980,192 |    10     4,457,072
 75,000,000 |   326 1,207,959,968 |    14     8,913,520
100,000,000 |   472 1,207,959,968 |    20     8,913,520
125,000,000 |   582 1,207,959,968 |    26     8,913,520
150,000,000 |   711 2,415,919,520 |    30    17,826,416
175,000,000 |   834 2,415,919,520 |    36    17,826,416
200,000,000 |   931 2,415,919,520 |    40    17,826,416
225,000,000 | 1,049 2,415,919,520 |    46    17,826,416
250,000,000 | 1,167 2,415,919,520 |    50    17,826,416
```

## SQL

To optimize frequently called SQL queries you can make use of the [prepare](../../../api/query.md#pykx.query.SQL.prepare) and [execute](../../../api/query.md#pykx.query.SQL.execute) functionality to separate SQL parsing from query execution as detailed [here](https://code.kx.com/insights/1.10/core/sql.html#prepare-and-execute).

## Next Steps

- Learn how to query your data using the PyKX Pythonic Query API [here](pyquery.md).
- If you don't have a historical database available see [here](../../advanced/database/index.md).
- To learn about creating PyKX Table objects see [here](../../../examples/interface-overview.ipynb).

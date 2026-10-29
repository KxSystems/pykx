---
title: pandas API overview
description: Overview of the pandas like API for KDB-X Python tables
date: October 2026
author: KX Systems, Inc.,
tags: KDB-X Python, pandas, table, DataFrame
---

# Pandas like API for KDB-X Python tables

_This page gives an overview of the methods on KDB-X Python tables that follow the pandas `DataFrame` API, and how they compare with pandas and with q._

`pykx.Table`, `pykx.KeyedTable` and the result of `groupby` support a subset of the pandas `DataFrame` methods. The methods run in q on the data where it is, so there is no conversion to pandas and no copy of the table in Python. The focus is on the operations where q can reduce memory use and execution time compared with pandas.

Each method is documented, with its parameters and an example, in the [reference pages](#reference).

## Quick start

A table can be created from Python objects and used much as a `DataFrame` would be.

```python
>>> import pykx as kx
>>> tab = kx.Table(data={
...     'sym': ['a', 'b', 'a', 'c'],
...     'price': [1.5, 2.0, 3.5, 4.0],
...     'size': [10, 20, 30, 40]})
>>> tab.groupby('sym')['size'].sum()
pykx.KeyedTable(pykx.q('
sym| size
---| ----
a  | 40
b  | 20
c  | 40
'))
>>> tab.sort_values('price', ascending=False).head(2)
pykx.Table(pykx.q('
sym price size
--------------
c   4     40
a   3.5   30
'))
>>> tab.mean(numeric_only=True)
pykx.Dictionary(pykx.q('
price| 2.75
size | 25
'))
```

The results are KDB-X Python objects. Use `.pd()` to convert a result to a pandas object.

```python
>>> tab.pd()
  sym  price  size
0   a    1.5    10
1   b    2.0    20
2   a    3.5    30
3   c    4.0    40
```

## Reference

The methods are grouped by purpose. The supported method names are the same as in pandas.

| Section | Methods |
| :------ | :------ |
| [Metadata and aggregations](../../api/pandas-api/metadata.md) | `columns`, `dtypes`, `empty`, `ndim`, `shape`, `size`, `isna`, `isnull`, `notna`, `notnull`, `abs`, `fills`, `ungroup`, `all`, `any`, `count`, `max`, `min`, `idxmax`, `idxmin`, `sum`, `prod`, `mean`, `median`, `mode`, `std`, `sem`, `skew`, `kurt`, `agg` |
| [Indexing and selection](../../api/pandas-api/indexing.md) | `head`, `tail`, `get`, `pop`, `at`, `loc`, `iloc`, `sample`, `drop`, `drop_duplicates`, `rename`, `add_prefix`, `add_suffix`, `set_index`, `index`, `reset_index` |
| [Merging and joining](../../api/pandas-api/merge.md) | `merge`, `merge_asof` |
| [Group by](../../api/pandas-api/groupby.md) | `groupby` |
| [Sorting](../../api/pandas-api/sorting.md) | `sort_values`, `nlargest`, `nsmallest` |
| [Function application](../../api/pandas-api/apply.md) | `apply`, `map`, `applymap` |
| [Data preprocessing](../../api/pandas-api/preprocessing.md) | `astype`, `select_dtypes`, `replace` |

## What is not covered

Coverage means the methods and attributes of the pandas `DataFrame`. It does not include pandas `Series` objects or reading and writing files such as CSV and JSON.

| pandas methods | Supported | Notes |
| :------------- | :-------: | :---- |
| `from_*` | No | Creating tables from other data sources is not supported. |
| `to_*` | No | Converting or saving tables to other formats is not supported. Use `.pd()`, `.np()` or `.pa()` to convert a table. |
| `plot*` | No | Plotting columns or tables is not supported. |
| `sparse*` | No | Sparse data is not supported. |
| `info`, `memory_usage` | No | Not implemented. |

A pandas method which is not listed in the reference is not implemented. Some methods have parameters which are not implemented yet. These are described on the method's reference entry and raise a `NotImplementedError` if used.

To request a method or parameter, open an issue [here](https://github.com/KxSystems/pykx/issues) or open a pull request [here](https://github.com/KxSystems/pykx/pulls).

## How the methods behave

- **Index.** A `Table` has no index, so the row number is used as the index label. The key columns of a `KeyedTable` are its index.
- **Nulls.** The aggregations skip nulls by default, as pandas does. This differs from some q functions of the same name. The [aggregations reference](../../api/pandas-api/metadata.md) lists the differences.
- **Group by.** The aggregations on the result of `groupby` run as a single grouped select in q, so they take about as long as the equivalent `select ... by` query.
- **Results.** Results are `Dictionary`, `Table` or `KeyedTable` objects rather than pandas objects.

## Choosing an API

A table can also be queried with the [query API](../fundamentals/query/pyquery.md), [SQL](../fundamentals/query/sql.md) and [qSQL](../fundamentals/query/qquery.md).
The same query, the total `size` for each `sym` in the table above, looks like this in each.

**pandas API**

```python
>>> tab.groupby('sym')['size'].sum()
pykx.KeyedTable(pykx.q('
sym| size
---| ----
a  | 40
b  | 20
c  | 40
'))
```

**Query API**

```python
>>> tab.select(kx.Column('size').sum(), by=kx.Column('sym'))
pykx.KeyedTable(pykx.q('
sym| size
---| ----
a  | 40
b  | 20
c  | 40
'))
```

**SQL**

```python
>>> tab.sql('select sym, sum(size) as size from $1 group by sym order by sym')
pykx.Table(pykx.q('
sym size
--------
a   40
b   20
c   40
'))
```

**qSQL**

```python
>>> kx.q('{select sum size by sym from x}', tab)
pykx.KeyedTable(pykx.q('
sym| size
---| ----
a  | 40
b  | 20
c  | 40
'))
```

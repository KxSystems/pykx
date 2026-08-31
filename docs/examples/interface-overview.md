---
title: KDB-X Python Fundamentals
description: Learn KDB-X Python data structures, conversions, table methods, q functions, CSV input, and IPC
last_updated: July 2026
author: KX Systems, Inc.,
keywords:
  - KDB-X Python
  - pykx
  - fundamentals
  - q
  - data structures
  - conversions
  - IPC
---

# KDB-X Python Fundamentals

_Learn how to create and convert typed `pykx` objects, analyze vectors and tables, call q functions from Python, and connect to an external q process over IPC._

**Estimated time:** 30 minutes

## Prerequisites

Before you start:

- [Install KDB-X Python](../getting-started/installing.md#install-from-pypi).
- [Install a KDB-X license](../getting-started/installing.md#install-a-kdb-x-license).
- If you are new to KDB-X Python, complete the [quickstart guide](../getting-started/quickstart.md) first.
- Run the examples with [Python 3](../getting-started/installing.md#prerequisites) in an interactive session, a `.py` script, or a Jupyter notebook.

Run each code sample in order because later examples use objects created earlier.

## 1. Import KDB-X Python

Import the `pykx` package with the conventional `kx` alias:

```python
>>> import pykx as kx
```

The remaining examples also use NumPy and pandas:

```python
>>> import numpy as np
>>> import pandas as pd
```

Set the random seed so generated data remains reproducible throughout the guide:

```python
>>> kx.random.seed(42)
```

## 2. Data structures

KDB-X Python wraps values of [q data types](https://code.kx.com/q/basics/datatypes/) in Python classes from the `pykx` package. Pass these objects directly to q functions and KDB-X queries.

This section introduces the structures used throughout the guide:

- 2.1 [Atom](#21-atom)
- 2.2 [Vector](#22-vector)
- 2.3 [List](#23-list)
- 2.4 [Dictionary](#24-dictionary)
- 2.5 [Table](#25-table)
- 2.6 [Function types](#26-function-types)

### 2.1 Atom

An **atom** contains one value of a specific q type. The following examples create `pykx.FloatAtom` and `pykx.DateAtom` objects from equivalent Python values:

```python
>>> float_atom = kx.FloatAtom(1.0)
>>> print(type(float_atom).__name__, float_atom.py())
```

```text
FloatAtom 1.0
```

```python
>>> from datetime import date
>>>
>>> date_atom = kx.DateAtom(date(2020, 1, 1))
>>> print(type(date_atom).__name__, date_atom.py())
```

```text
DateAtom 2020-01-01
```

### 2.2 Vector

A `pykx` vector contains multiple values of one type. Vectors form the columns in dictionaries and tables and support efficient vector operations.

Vectors are one-dimensional and support indexing along one axis. The following examples convert NumPy and pandas objects to typed vectors:

```python
>>> print(kx.IntVector(np.array([1, 2, 3, 4], dtype=np.int32)))
```

```text
1 2 3 4i
```

```python
>>> print(kx.toq(pd.Series([1, 2, 3, 4])))
```

```text
1 2 3 4
```

### 2.3 List

A `pykx.List` can contain values of different types. Use lists for mixed or nested data that cannot use a single typed vector.

Lists can also hold multidimensional and ragged data. Use a typed vector when every item has the same q type.

```python
>>> print(kx.List([[1, 2, 3], [1.0, 1.1, 1.2], ['a', 'b', 'c']]))
```

```text
1 2   3
1 1.1 1.2
a b   c
```

### 2.4 Dictionary

A `pykx.Dictionary` maps keys directly to values. The key list and value list must contain the same number of items.

```python
>>> print(kx.Dictionary({'x': [1, 2, 3], 'x1': np.array([1, 2, 3])}))
```

```text
x | 1 2 3
x1| 1 2 3
```

### 2.5 Table

A `pykx.Table` stores named, typed columns in memory. Each column has a q type, and table methods operate on the columns.

KDB-X Python provides several table types:

- `pykx.Table`
- `pykx.KeyedTable`
- `pykx.SplayedTable`
- `pykx.PartitionedTable`

This section introduces the two in-memory table types.

#### `pykx.Table`

The Quickstart guide creates a table from Python column data. The following example creates a table from rows:

```python
>>> print(
...     kx.Table(
...         [[1, 2, 'a'], [2, 3, 'b'], [3, 4, 'c']],
...         columns=['col1', 'col2', 'col3'],
...     )
... )
```

```text
col1 col2 col3
--------------
1    2    a
2    3    b
3    4    c
```

#### `pykx.KeyedTable`

A `pykx.KeyedTable` assigns one or more columns as keys. It behaves similarly to a pandas `DataFrame` with an index. Call `set_index()` on a table to create one.

[Read the `pykx.KeyedTable` API reference](../api/pykx-q-data/wrappers.md#pykx.wrappers.KeyedTable).

```python
>>> print(
...     kx.Table(
...         data={
...             'x': [1, 2, 3],
...             'x1': [2, 3, 4],
...             'x2': ['a', 'b', 'c'],
...         }
...     ).set_index(['x'])
... )
```

```text
x| x1 x2
-| -----
1| 2  a
2| 3  b
3| 4  c
```

### 2.6 Function types

KDB-X Python wraps q functions so you can call them from Python.

#### `pykx.Lambda`

A `pykx.Lambda` represents a q function with up to eight parameters. You can call it with KDB-X Python or compatible Python values.

```python
>>> pykx_lambda = kx.q('{x+y}')
>>> print(type(pykx_lambda))
```

```text
<class 'pykx.wrappers.Lambda'>
```

```python
>>> print(pykx_lambda(1, 2))
```

```text
3
```

#### `pykx.Projection`

A projection fixes some arguments of a function in advance, similar to [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial). Call the projection with only the remaining arguments.

If a function accepts `n` parameters and you provide `m`, the resulting projection accepts `n - m` parameters.

```python
>>> projection = kx.q('{x+y}')(1)
>>> print(projection)
```

```text
{x+y}[1]
```

```python
>>> print(projection(2))
```

```text
3
```

## 3. Create and access `pykx` objects

This section applies the data structures from the previous section to common tasks:

- 3.1 [Create objects from Python data](#31-create-objects-from-python-data)
- 3.2 [Generate random data](#32-generate-random-data)
- 3.3 [Generate data with q](#33-generate-data-with-q)
- 3.4 [Read a CSV file](#34-read-a-csv-file)
- 3.5 [Optional: query a q process over IPC](#35-optional-query-a-q-process-over-ipc)

### 3.1 Create objects from Python data

KDB-X Python converts data between typed `pykx` objects and common Python formats.

Call `kx.toq()` to convert supported native Python, NumPy, pandas, and PyArrow data to a typed `pykx` object:

```python
>>> pydict = {'a': [1, 2, 3], 'b': ['a', 'b', 'c'], 'c': 2}
>>> print(kx.toq(pydict))
```

```text
a| 1 2 3
b| `a`b`c
c| 2
```

```python
>>> nparray = np.array([1, 2, 3, 4], dtype=np.int32)
>>> q_vector = kx.toq(nparray)
>>> print(q_vector)
```

```text
1 2 3 4i
```

```python
>>> pdframe = pd.DataFrame(data={'a': [1, 2, 3], 'b': ['a', 'b', 'c']})
>>> print(kx.toq(pdframe))
```

```text
a b
---
1 a
2 b
3 c
```

To convert a `pykx` object back to a Python-compatible representation, use:

| **Method** | **Converts to** |
| --- | --- |
| `.py()` | Native Python values |
| `.np()` | NumPy |
| `.pd()` | pandas |
| `.pa()` | PyArrow |

For example:

```python
>>> print(q_vector.py())
>>> print(repr(q_vector.np()))
```

```text
[1, 2, 3, 4]
array([1, 2, 3, 4], dtype=int32)
```

PyArrow is an optional dependency. For more examples and conversion considerations, read [Create and convert KDB-X Python objects](../user-guide/fundamentals/creating.md).

### 3.2 Generate random data

Use `kx.random.random()` to generate typed test data.

The following example generates 10,000 random floating-point values between 0 and 1, then confirms the number of values:

```python
>>> random_values = kx.random.random(10_000, 1.0)
>>> print(len(random_values))
```

```text
10000
```

Pass a list as the second argument to choose randomly from its values:

```python
>>> print(kx.random.random(5, [1, ['a', 'b', 'c'], np.array([1.1, 1.2, 1.3])]))
```

```text
1
1.1 1.2 1.3
1.1 1.2 1.3
1
`a`b`c
```

Pass a list of dimensions as the first argument to create multidimensional data. Typed null and infinity values let you generate data across the range supported by that q type:

```python
>>> print(kx.random.random([2, 5], kx.GUIDAtom.null))
```

```text
cddd3245-b888-e3e6-6b46-238afd6899da 6710ad06-0667-4588-61da-11b89be0d75b 6ac..
88c4ce9d-bebb-8cef-5ad5-7059b07b1136 99016def-c977-34a2-4065-634ee6441b32 af8..
```

```python
>>> print(kx.random.random([2, 3, 4], kx.IntAtom.inf))
```

```text
1297399104 2021175493 1561022346 494454701  1253362152 1017691991 185388196  ..
1508381055 1409761995 429286974  1445067192 1912108436 1867348694 2131953451 ..
```

Set a seed globally or for one call when you need reproducible data:

```python
>>> kx.random.seed(10)
>>> print(kx.random.random(10, 2.0))
```

```text
0.1782082 1.669039 0.7243899 1.999868 0.7675971 1.723838 0.1836728 0.5061767 ..
```

```python
>>> print(kx.random.random(10, 2.0, seed=10))
```

```text
0.1782082 1.669039 0.7243899 1.999868 0.7675971 1.723838 0.1836728 0.5061767 ..
```

### 3.3 Generate data with q

KDB-X Python lets you execute q directly from Python with `kx.q`.

Create a q vector:

```python
>>> print(kx.q('0 1 2 3 4'))
```

```text
0 1 2 3 4
```

```python
>>> print(kx.q('([idx:desc til 5]col1:til 5;col2:5?1f;col3:5?`2)'))
```

```text
idx| col1 col2       col3
---| --------------------
4  | 0    0.8619188  ol
3  | 1    0.09183638 mg
2  | 2    0.2530883  cm
1  | 3    0.2504566  cc
0  | 4    0.7517286  jg
```

Pass arguments to a q function:

```python
>>> print(kx.q('{x+y}', 1, 2))
```

```text
3
```

### 3.4 Read a CSV file

KDB-X Python provides `kx.q.read.csv()` for reading CSV data into typed q columns. The following example creates a small CSV file:

```python
>>> import csv
>>>
>>> with open('pykx.csv', 'w', newline='', encoding='utf-8') as csv_file:
...     writer = csv.writer(csv_file)
...     fields = ["name", "age", "height", "country"]
...
...     writer.writerow(fields)
...     writer.writerow(["Oladele Damilola", "40", "180.0", "Nigeria"])
...     writer.writerow(["Alina Hricko", "23", "179.2", "Ukraine"])
...     writer.writerow(["Isabel Walter", "50", "179.5", "United Kingdom"])
...
```

```python
>>> print(
...     kx.q.read.csv(
...         'pykx.csv',
...         types={'age': kx.LongAtom, 'country': kx.SymbolAtom},
...     )
... )
```

```text
name               age height country
--------------------------------------------
"Oladele Damilola" 40  180    Nigeria
"Alina Hricko"     23  179.2  Ukraine
"Isabel Walter"    50  179.5  United Kingdom
```

```python
>>> import os
>>> os.remove('pykx.csv')
```

### 3.5 Optional: query a q process over IPC

Complete this optional section when you need to query a separate q process. It requires the `q` executable on your system path. [Install KDB-X](https://code.kx.com/kdb-x/get_started/kdb-x-install.html) before you continue.

!!! note "If you run this example on Windows"

    Use Windows Subsystem for Linux (WSL) with KDB-X installed, or connect to an existing q process instead.

The example starts a local q process on port 5000, connects synchronously, and queries it with q, the KDB-X Python query API, and SQL. Skip this section if your environment does not allow subprocesses or local network connections.

```python
>>> import subprocess
>>> import time
>>>
>>> try:
...     with kx.PyKXReimport():
...         proc = subprocess.Popen(('q', '-p', '5000'))
... except OSError as error:
...     raise kx.QError('Unable to create a q process on port 5000') from error
...
>>> conn = None
>>> try:
...     time.sleep(2)
...     conn = kx.SyncQConnection(port=5000)
...
...     conn('system "S 42"')
...     conn('tab:([]col1:100?`a`b`c;col2:100?1f;col3:100?0Ng)')
...     print('q query:')
...     print(conn('select from tab where col1=`a')[:5])
...
...     print('\nQuery API:')
...     print(
...         conn.qsql.select(
...             'tab',
...             where=(kx.Column('col1') == 'a') & (kx.Column('col2') > 0.3),
...         )[:5]
...     )
...
...     print('\nSQL:')
...     print(conn.sql('SELECT * FROM tab where col2>=0.5')[:5])
... finally:
...     if conn is not None:
...         conn.close()
...     if proc.poll() is None:
...         proc.terminate()
...         proc.wait(timeout=5)
...
```

```text
q query:
col1 col2      col3
---------------------------------------------------
a    0.6357471 22371003-8997-eed1-f4df-58fcdedd8376
a    0.1200245 1a43967f-d414-1b0e-ae8c-0c411890e175
a    0.8362442 23befa0d-8324-099e-296f-93e9d7fc70d6
a    0.8223493 2fe5ff57-2d7f-f1d9-024b-7c258e1fcd9a
a    0.5082821 fac43ae3-caf2-1c2a-3951-aa857f3a7b8a

Query API:
col1 col2      col3
---------------------------------------------------
a    0.6357471 22371003-8997-eed1-f4df-58fcdedd8376
a    0.8362442 23befa0d-8324-099e-296f-93e9d7fc70d6
a    0.8223493 2fe5ff57-2d7f-f1d9-024b-7c258e1fcd9a
a    0.5082821 fac43ae3-caf2-1c2a-3951-aa857f3a7b8a
a    0.9455612 418c580d-e1c4-093f-d532-38f2a35686fc

SQL:
col1 col2      col3
---------------------------------------------------
c    0.9805637 84cf32c6-c711-79b4-2f31-6e85923decff
a    0.6357471 22371003-8997-eed1-f4df-58fcdedd8376
a    0.8362442 23befa0d-8324-099e-296f-93e9d7fc70d6
b    0.9044767 3796c7db-e028-e16a-422f-43e2dadbc5b9
c    0.965964  82261eb4-b3a4-67cb-8374-d9c22c68be6d
```

The `finally` block closes the connection and stops the q process if a query raises an error.

---

## 4. Analyze vectors and tables

KDB-X Python provides Python methods and q functions for vector and table calculations.

- 4.1 [Use built-in vector methods](#41-use-built-in-vector-methods)
- 4.2 [Use built-in table methods](#42-use-built-in-table-methods)
- 4.3 [Use q functions](#43-use-q-functions)

### 4.1 Use built-in vector methods

Call methods such as `mean()` and `max()` directly on a typed vector:

```python
>>> q_vector = kx.random.random(1000, 10.0)
```

```python
>>> print(q_vector.mean())
```

```text
4.984157
```

```python
>>> print(q_vector.max())
```

```text
9.998212
```

Use `apply()` when you need a custom Python calculation:

```python
>>> def bespoke_function(x, y):
...     return x*y
...
>>> print(q_vector.apply(bespoke_function, 5))
```

```text
31.74132 38.3376 46.40922 10.17963 38.73944 48.33864 41.12562 45.44382 32.290..
```

### 4.2 Use built-in table methods

KDB-X Python tables provide methods for filtering, grouping, and calculating column values. The [pandas API notebook](../user-guide/advanced/Pandas_API.ipynb) documents the table methods that mirror pandas operations.

The following examples use a typed table with symbol and numeric columns:

```python
>>> N = 10_000
>>> example_table = kx.Table(
...     data={
...         'sym': kx.random.random(N, ['a', 'b', 'c']),
...         'col1': kx.random.random(N, 10.0),
...         'col2': kx.random.random(N, 20),
...     }
... )
>>> print(example_table[:5])
```

```text
sym col1     col2
-----------------
b   5.332563 3
c   1.555556 10
c   7.583763 18
b   9.657423 12
b   4.543002 17
```

Select multiple columns by passing their names as a list:

```python
>>> print(example_table[['sym', 'col1']])
```

```text
sym col1
-------------
b   5.332563
c   1.555556
c   7.583763
b   9.657423
b   4.543002
b   5.587764
c   1.137493
a   1.328115
c   4.453789
c   0.8913583
b   1.720334
a   7.381833
a   6.678592
c   0.4029847
b   1.705896
a   1.209391
a   6.059422
c   1.296014
a   2.615679
b   9.125557
..
```

Filter table rows with `loc`, as you would with a pandas `DataFrame`:

```python
>>> print(example_table.loc[example_table['sym'] == 'a'])
```

```text
sym col1       col2
-------------------
a   1.328115   2
a   7.381833   2
a   6.678592   6
a   1.209391   16
a   6.059422   15
a   2.615679   0
a   2.779952   17
a   3.281611   18
a   6.036149   9
a   0.5906303  11
a   0.06815037 0
a   8.130013   7
a   2.149836   3
a   4.742463   5
a   5.857722   4
a   5.858365   13
a   8.713641   18
a   5.651717   18
a   2.719246   3
a   6.195479   15
..
```

Apply the same Boolean filter with `[]`, which calls `__getitem__()`:

```python
>>> print(example_table[example_table['sym'] == 'b'])
```

```text
sym col1     col2
-----------------
b   5.332563 3
b   9.657423 12
b   4.543002 17
b   5.587764 4
b   1.720334 8
b   1.705896 10
b   9.125557 9
b   5.04443  5
b   8.741315 10
b   7.250155 3
b   2.437429 15
b   7.900336 12
b   9.095591 4
b   5.145199 11
b   7.107319 17
b   8.117879 19
b   5.078329 9
b   8.56087  3
b   9.312045 15
b   8.32909  0
..
```

Call `set_index()` to convert a `pykx.Table` to a `pykx.KeyedTable`:

```python
>>> print(example_table.set_index('sym'))
```

```text
sym| col1      col2
---| --------------
b  | 5.332563  3
c  | 1.555556  10
c  | 7.583763  18
b  | 9.657423  12
b  | 4.543002  17
b  | 5.587764  4
c  | 1.137493  13
a  | 1.328115  2
c  | 4.453789  4
c  | 0.8913583 13
b  | 1.720334  8
a  | 7.381833  2
a  | 6.678592  6
c  | 0.4029847 12
b  | 1.705896  10
a  | 1.209391  16
a  | 6.059422  15
c  | 1.296014  5
a  | 2.615679  0
b  | 9.125557  9
..
```

Apply table analytics such as `mean()` and `median()`:

```python
>>> print('mean:')
>>> print(example_table.mean(numeric_only=True))
>>>
>>> print('median:')
>>> print(example_table.median(numeric_only=True))
```

```text
mean:
col1| 4.941699
col2| 9.5393
median:
col1| 4.927338
col2| 10
```

Group the table by `sym`, then calculate the mean of each numeric column:

```python
>>> print(example_table.groupby('sym').mean())
```

```text
sym| col1     col2
---| -----------------
a  | 4.906693 9.568162
b  | 5.053115 9.494075
c  | 4.867251 9.554758
```

Apply a custom function that converts each group to NumPy before calculating a result:

```python
>>> def apply_func(x):
...     nparray = x.np()
...     return np.sqrt(nparray).mean()
...
>>>
>>> print(example_table.groupby('sym').apply(apply_func))
```

```text
sym| col1     col2
---| -----------------
a  | 2.085255 2.867129
b  | 2.12296  2.863945
c  | 2.071932 2.865273
```

Use `merge_asof()` to join time-series tables. This example creates a trades table and a quotes table with temporal columns:

```python
>>> trades = kx.Table(
...     data={
...         "time": [
...             pd.Timestamp("2016-05-25 13:30:00.023"),
...             pd.Timestamp("2016-05-25 13:30:00.023"),
...             pd.Timestamp("2016-05-25 13:30:00.030"),
...             pd.Timestamp("2016-05-25 13:30:00.041"),
...             pd.Timestamp("2016-05-25 13:30:00.048"),
...             pd.Timestamp("2016-05-25 13:30:00.049"),
...             pd.Timestamp("2016-05-25 13:30:00.072"),
...             pd.Timestamp("2016-05-25 13:30:00.075"),
...         ],
...         "ticker": [
...             "GOOG",
...             "MSFT",
...             "MSFT",
...             "MSFT",
...             "GOOG",
...             "AAPL",
...             "GOOG",
...             "MSFT",
...         ],
...         "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
...         "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
...     }
... )
>>> quotes = kx.Table(
...     data={
...         "time": [
...             pd.Timestamp("2016-05-25 13:30:00.023"),
...             pd.Timestamp("2016-05-25 13:30:00.038"),
...             pd.Timestamp("2016-05-25 13:30:00.048"),
...             pd.Timestamp("2016-05-25 13:30:00.048"),
...             pd.Timestamp("2016-05-25 13:30:00.048"),
...         ],
...         "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
...         "price": [51.95, 51.95, 720.77, 720.92, 98.0],
...         "quantity": [75, 155, 100, 100, 100],
...     }
... )
>>>
>>> print('trades:')
>>> print(trades)
>>> print('\nquotes:')
>>> print(quotes)
```

```text
trades:
time                          ticker bid   ask
-------------------------------------------------
2016.05.25D13:30:00.023000000 GOOG   720.5 720.93
2016.05.25D13:30:00.023000000 MSFT   51.95 51.96
2016.05.25D13:30:00.030000000 MSFT   51.97 51.98
2016.05.25D13:30:00.041000000 MSFT   51.99 52
2016.05.25D13:30:00.048000000 GOOG   720.5 720.93
2016.05.25D13:30:00.049000000 AAPL   97.99 98.01
2016.05.25D13:30:00.072000000 GOOG   720.5 720.88
2016.05.25D13:30:00.075000000 MSFT   52.01 52.03

quotes:
time                          ticker price  quantity
----------------------------------------------------
2016.05.25D13:30:00.023000000 MSFT   51.95  75
2016.05.25D13:30:00.038000000 MSFT   51.95  155
2016.05.25D13:30:00.048000000 GOOG   720.77 100
2016.05.25D13:30:00.048000000 GOOG   720.92 100
2016.05.25D13:30:00.048000000 AAPL   98     100
```

Apply an as-of join by `ticker` so each trade matches the most recent quote for the same ticker:

```python
>>> print(trades.merge_asof(quotes, on='time', by='ticker'))
```

```text
time                          ticker bid   ask    price  quantity
-----------------------------------------------------------------
2016.05.25D13:30:00.023000000 GOOG   720.5 720.93
2016.05.25D13:30:00.023000000 MSFT   51.95 51.96  51.95  75
2016.05.25D13:30:00.030000000 MSFT   51.97 51.98  51.95  75
2016.05.25D13:30:00.041000000 MSFT   51.99 52     51.95  155
2016.05.25D13:30:00.048000000 GOOG   720.5 720.93 720.92 100
2016.05.25D13:30:00.049000000 AAPL   97.99 98.01  98     100
2016.05.25D13:30:00.072000000 GOOG   720.5 720.88 720.92 100
2016.05.25D13:30:00.075000000 MSFT   52.01 52.03  51.95  155
```

### 4.3 Use q functions

Call q functions through `kx.q` when they provide the operation you need. The following examples introduce mathematical, iteration, and table functions:

- 4.3.1 [Mathematical functions](#431-mathematical-functions)
- 4.3.2 [Iteration functions](#432-iteration-functions)
- 4.3.3 [Table functions](#433-table-functions)

#### 4.3.1 Mathematical functions

##### `mavg`

Calculate moving averages with a rolling window:

```python
>>> moving_average = kx.q.mavg(10, kx.random.random(10000, 2.0))
>>> print(moving_average[:10])
```

```text
1.909677 1.326592 1.390974 1.228086 1.006547 0.847965 0.9194783 0.836154 0.86..
```

##### `cor`

Calculate the correlation between two lists:

```python
>>> print(kx.q.cor([1, 2, 3], [2, 3, 4]))
```

```text
1f
```

```python
>>> print(kx.q.cor(kx.random.random(100, 1.0), kx.random.random(100, 1.0)))
```

```text
0.009771912
```

##### `prds`

Calculate cumulative products for a list:

```python
>>> print(kx.q.prds([1, 2, 3, 4, 5]))
```

```text
1 2 6 24 120
```

#### 4.3.2 Iteration functions

##### `each`

Use `each` as a q function or a `pykx.Lambda` method to apply a function to every item:

```python
>>> print(kx.q.each(kx.q('{prd x}'), kx.random.random([5, 5], 10.0, seed=10)))
```

```text
1033.597 377.1784 7126.713 418.3232 89.97531
```

```python
>>> print(kx.q('{prd x}').each(kx.random.random([5, 5], 10.0, seed=10)))
```

```text
1033.597 377.1784 7126.713 418.3232 89.97531
```

#### 4.3.3 Table functions

##### `meta`

Return metadata for a table:

```python
>>> qtab = kx.Table(
...     data={
...         'x': kx.random.random(1000, ['a', 'b', 'c']).grouped(),
...         'y': kx.random.random(1000, 1.0),
...         'z': kx.random.random(1000, kx.TimestampAtom.inf),
...     }
... )
```

```python
>>> print(kx.q.meta(qtab))
```

```text
c| t f a
-| -----
x| s   g
y| f
z| p
```

##### `xasc`

Sort a table in ascending order by one or more columns:

```python
>>> print(kx.q.xasc('z', qtab)[:5])
```

```text
x y          z
------------------------------------------
c 0.2660419  2000.09.17D00:27:33.222932480
b 0.2378591  2001.02.01D19:58:48.496586752
c 0.05802967 2001.05.29D15:29:16.181340160
c 0.9474748  2003.03.24D08:12:02.975653888
b 0.02726729 2004.01.31D07:25:21.959215104
```

## Next steps

Continue with [Objects and attributes](../learn/objects.md) to understand the type information available on KDB-X Python objects.

Or explore by task:

- Explore [data creation and conversion](../user-guide/fundamentals/creating.md).
- Learn how to [index and slice `pykx` objects](../user-guide/fundamentals/indexing.md).
- Use [NumPy functions with `pykx` vectors](../user-guide/advanced/numpy.md).
- Choose a [query interface](../user-guide/fundamentals/query/index.md).
- Learn how to [communicate over IPC](../user-guide/advanced/ipc.md).
- Refer to the [q functions and operators reference](../api/pykx-execution/q.md).
- [Troubleshoot errors](../help/troubleshooting.md).

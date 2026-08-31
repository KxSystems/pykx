---
title: Quickstart for KDB-X Python
description: Create, inspect, and query a q table with KDB-X Python, then convert the result to a pandas DataFrame
last_updated: July 2026
author: KX Systems, Inc.,
keywords:
  - KDB-X Python
  - pykx
  - quickstart
  - q
  - table
  - pandas
  - DataFrame
---

# Quickstart

_Create a small dataset, analyze it with q from Python, and convert the result to a pandas `DataFrame`._

**Estimated time:** 10 minutes

## Prerequisites

Before you start:

- [Install KDB-X Python](installing.md#install-from-pypi).
- [Install a KDB-X license](installing.md#install-a-kdb-x-license).
- Run the examples with [Python 3](installing.md#prerequisites) in an interactive session, a `.py` script, or a Jupyter notebook.

## 1. Import KDB-X Python

Import the `pykx` package using the conventional `kx` alias:

```python
>>> import pykx as kx
```

!!! info "The `kx` alias"

    Documentation examples use this optional alias to keep KDB-X Python code concise.

## 2. Create a table

Create a table of sample trades from native Python lists:

```python
>>> trades = kx.Table(
...     data={
...         'symbol': ['AAPL', 'MSFT', 'AAPL', 'MSFT'],
...         'price': [189.0, 374.0, 191.0, 376.0],
...         'size': [100, 200, 150, 100],
...     }
... )
>>>
>>> print(trades)
```

KDB-X Python creates a typed `pykx.Table` and stores it in q memory:

```text
symbol price size
-----------------
AAPL   189   100
MSFT   374   200
AAPL   191   150
MSFT   376   100
```

KDB-X Python converts the Python strings, floats, and integers into their corresponding q types. Keeping the data in q memory lets q functions operate on it without another conversion.

## 3. Inspect with Python indexing

KDB-X Python tables support familiar Python indexing. Use a slice to select the first two rows:

```python
>>> print(trades[:2])
```

```text
symbol price size
-----------------
AAPL   189   100
MSFT   374   200
```

Pass a list of column names to select specific columns:

```python
>>> print(trades[['symbol', 'price']])
```

```text
symbol price
------------
AAPL   189
MSFT   374
AAPL   191
MSFT   376
```

## 4. Analyze with q

Use `kx.q` to calculate the average price and total size for each symbol:

```python
>>> summary = kx.q(
...     '{select average_price:avg price, total_size:sum size by symbol from x}',
...     trades,
... )
>>>
>>> print(summary)
```

The q function receives `trades` as its `x` argument and returns a keyed table:

```text
symbol| average_price total_size
------| ------------------------
AAPL  | 190           250
MSFT  | 375           300
```

The embedded q runtime executes this query in the same Python process. To query a separate q process, use the [IPC interface](../user-guide/advanced/ipc.md).

## 5. Convert to a pandas `DataFrame`

KDB-X Python results remain typed q objects until you choose to convert them. Call `.pd()` when a downstream Python workflow needs a pandas `DataFrame`:

```python
>>> summary_df = summary.pd()
>>> print(summary_df)
```

The resulting `DataFrame` contains one row per symbol:

```text
        average_price  total_size
symbol
AAPL            190.0         250
MSFT            375.0         300
```

## Next steps

Continue with [KDB-X Python Fundamentals](../examples/interface-overview.md) for a guided tour of q types, conversions, table methods, q functions, and IPC.

Or go directly to a focused guide:

- [Understand objects and attributes](../learn/objects.md)
- [Create and convert KDB-X Python objects](../user-guide/fundamentals/creating.md)
- [Index KDB-X Python objects](../user-guide/fundamentals/indexing.md)
- [Query data](../user-guide/fundamentals/query/index.md)
- [Communicate over IPC](../user-guide/advanced/ipc.md)
- [Troubleshoot errors](../help/troubleshooting.md)

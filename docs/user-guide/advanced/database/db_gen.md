---
title: Generating and extending a database
description: Introduction to the PyKX database creation and management functionality 
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, database, maintenance, management, generation
---


# Generate and extend a database

_This page explains how to create and expand databases using PyKX._

!!! tip "Tip: For the best experience, we recommend reading [Databases in PyKX](index.md) first. If you already have access to a database and only need to load it, you can skip this page and jump right to [load database](db_loading.md)."

Before leveraging the performance of PyKX when querying on-disk data, you need to create a [persisted database](..//..//..//extras/glossary.md#persisted-database). In the following sections we complete the following:

1. [Create a new database](#1-create-database) containing a single table `#!python trade` and multiple days of data.
1. [Add a new day worth of data](#2-add-new-database-partition) for `#!python today` to the database for the `#!python trade` table.
1. [On-board a new table](#3-add-new-table-to-database) (`#!python quote`) which contains data from `#!python today`.
1. Ensure that the [new table is queryable](#4-ensure-new-table-is-queryable).

!!! note "Bring your own data"

	The below example makes use of randomly-generated data using PyKX, where we use `#!python trade` or `#!python quote` tables generated in that manner. You can replace them with an equivalent Pandas/PyArrow table which will be converted to a PyKX table before being persisted.

## 1. Create database

For more information on database structures, see the linked section on [what is a database](index.md#whats-a-pykx-database). With PyKX, use the `#!python pykx.DB` class for all database interactions in Python. This class lets you create, expand, and maintain on-disk partitioned databases. First, we need to create a database.

In the next cell, we create a `#!python trade` table with data from multiple days in the chat.

```python
>>> import pykx as kx
>>> N = 10000000
>>> trade = kx.Table(data={
...     'date': kx.random.random(N, kx.DateAtom('today') - [1, 2, 3, 4]),
...     'time': kx.q.asc(kx.random.random(N, kx.q('1D'))),
...     'sym': kx.random.random(N, ['AAPL', 'GOOG', 'MSFT']),
...     'price': kx.random.random(N, 10.0)
...     })
```

Now that we have generated our trade table, we can persist it to disk at the location `#!python /tmp/db`.

```python
>>> db = kx.DB(path='/tmp/db')
>>> db.create(trade, 'trade', 'date')
```

That's it, you now have a persisted database. To verify the availability of the database and its tables, we can examine the database object:

```python
>>> db.tables
['trade']
>>> type(db.trade)
<class 'pykx.wrappers.PartitionedTable'>
```

The above database persistence uses the default parameters within the `#!python create` function. If you need to compress/encrypt the persisted database partitions or need to define a `#!python by` or specify the symbol enumeration name, you can follow the API documentation [here](../../../api/db.md#pykx.db.DB.create).

## 2. Add new database partition

Now that you have generated a database, you can add extra partitions using the same database class and the `#!python create` function. In this example we will add new data for the current day created in the below cell:

```python
>>> N = 2000000
>>> trade = kx.Table(data={
...     'time': kx.q.asc(kx.random.random(N, kx.q('1D'))),
...     'sym': kx.random.random(N, ['AAPL', 'GOOG', 'MSFT']),
...     'price': kx.random.random(N, 10.0)
...     })
```

Note that in comparison to the original database creation logic, we do not have a `#!python date` column. Instead, we add a date at partition creation. Below we provide a variety of examples of adding new partitions under various conditions:

=== "Generate default partition"

	```python
	>>> db.create(trade, 'trade', kx.DateAtom('today'))
	```

=== "Compress data in a partition"

	In the below example, we compress data within the persisted partition using [`gzip`](https://en.wikipedia.org/wiki/Gzip). For further details on supported compression formats see [here](../compress-encrypt.md) or look at the API reference [here](../../../api/compress.md).

	```python
	>>> gzip = kx.Compress(kx.CompressionAlgorithm.gzip, level=2)
	>>> db.create(trade, 'trade', kx.DateAtom('today'), compress=gzip)
	```

=== "Encrypt persisted data"

	In the below example, we encrypt the data persisted for the added partition. For further details on how encryption works within PyKX see [here](../compress-encrypt.md) or look at the API reference [here](../../../api/compress.md).

	```python
	>>> encrypt = kx.Encrypt('/path/to/mykey.key', 'mySuperSecretPassword')
	>>> db.create(trade, 'trade', kx.DateAtom('today'), encrypt=encrypt)
	```

## 3. Add new table to database

After onboarding your first table to a database, a common question is “How can I add a new table of related data?”. You can use the `#!python database` class and the `#!python create` function to do this. For instance, let’s add a `#!python quote` table for the current day:

```python
>>> N = 1000000
>>> quote = kx.Table(data={
...     'time': kx.q.asc(kx.random.random(N, kx.q('1D'))),
...     'sym': kx.random.random(N, ['AAPL', 'GOOG', 'MSFT']),
...     'ask': kx.random.random(N, 100),
...     'bid': kx.random.random(N, 100)
... })
```

We can now add this as the data for the current day to the `#!python quote` table and see that the table is defined:

```python
>>> db.create(quote, 'quote', kx.DateAtom('today'))
>>> db.tables
['quote', 'trade']
>>> type(db.quote)
<class 'pykx.wrappers.PartitionedTable'>
```

## 4. Ensure new table is queryable

You have now persisted another table to your database, however, you will notice if you access the `#!python quote` table that the return is surprising:

```python
>>> db.quote
pykx.PartitionedTable(pykx.q('+`time`sym`ask`bid!`quote'))
```

The reason for this is that you currently do not have data in each partition of your database for the `#!python quote` table. To rectify this, run the `#!python fill_database` method off the `#!python database` class which adds relevant empty quote data to tables to the partitions from which it's missing:

```python
>>> db.fill_database()
```

Now you should be able to access the `#!python quote` data for query:

```python
>>> db.quote
```

## Next Steps

- [Load an existing database](db_loading.md).
- [Modify the contents of your database](db_mgmt.md)
- [Query your database with Python](../../fundamentals/query/pyquery.md)
- [Compress/encrypt data](../compress-encrypt.md#persisting-database-partitions-with-various-configurations) for persisting database partitions.

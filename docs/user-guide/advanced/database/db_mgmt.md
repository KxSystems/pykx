---
title: Manage a PyKX Database
description: How to modify an existing database
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, database, loading
---

# Manage a database

_This page explains how to modify databases generated in PyKX._

!!! tip "Tip: For the best experience, we recommend reading [Databases in PyKX](index.md), [Generate a database](db_gen.md) and [Load a database](db_loading.md) first."

With PyKX, you can use various methods to modify your on-disk database. These changes can take many forms:

- Add new columns to the database
- Apply functions to existing columns
- Rename columns
- Delete columns

!!! Warning "A cautionary note"

	Operations on persisted databases can lead to changes that are hard to undo. For instance, applying functions that modify row values in a column can result in updated values that make it impossible to retrieve the original data. Before using this functionality for complex tasks, ensure you understand the impact of your changes and have a backup of your data to mitigate any issues.

The next section demonstrates how to edit the `#!python trade` table generated [here](db_gen.md) to extract information from the table columns, sanitize the data, and update the database schema.

## Update your database

Over time, the data you work with will change. This includes the names and types of columns, and even which columns are in the table. These changes can occur as new sensors are introduced in a manufacturing setting or when your data provider updates the information they supply in the financial sector.

To that end, we can take the `#!python trade` table and make the following changes:

1. Rename the column `#!python sym` to `#!python symbol`.
1. Change the type of the `#!python price` column from a `#!python pykx.FloatAtom` to `#!python pykx.RealAtom` to reduce storage requirements.
1. Add a new column `#!python exchange` which initially has an empty `#!python pykx.SymbolAtom` entry under the expectation that newly added partitions will have this column available.

```python
>>> import pykx as kx
>>> db = kx.DB(path='/tmp/db')
>>> db.rename_column('trade', 'sym', 'symbol')
>>> db.set_column_type('trade', 'price', kx.RealAtom)
>>> db.add_column('trade', 'exchange', kx.SymbolAtom.null)
```

Now that we’ve made some basic changes, we can proceed with more detailed modifications to the database. These changes can significantly impact the data since they involve free-form edits to individual columns and partitions. If you’re unsure about the changes or your ability to undo them, it’s a good idea to make a copy of the column first.

In the below cell, we complete the following:

1. Cache the order of columns prior to changes.
1. Make a copy of the column `#!python price` named `#!python price_copy`.
1. Adjust the value of the stock price on the copied column to account for a two-for-one stock split by multiplying the price by half.
1. Delete the original `#!python price` column.
1. Rename the copied column `#!python symbol_copy` to be `#!python symbol`.
1. Reorder the columns.

```python
>>> col_order = db.trade.columns.py()
>>> db.copy_column('trade', 'price', 'price_copy')
>>> db.apply_function('trade', 'price_copy', lambda x: x * 0.5)
>>> db.delete_column('trade', 'price')
>>> db.rename_column('trade', 'price_copy', 'price')
>>> db.reorder_columns(col_order)
```

## Next Steps

- [Query your database with Python](../../fundamentals/query/pyquery.md)
- [Compress/encrypt data](../compress-encrypt.md#persisting-database-partitions-with-various-configurations) for persisting database partitions.

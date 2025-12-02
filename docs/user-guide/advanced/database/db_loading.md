---
title: Load an existing database
description: How to load an existing database into a Python process
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, database, loading
---

# Load an existing database

_This page explains how to load an existing database into a Python process._

!!! tip "Tip: For the best experience, we recommend reading [Databases in PyKX](index.md) and [generate a database](db_gen.md) first."

By default, you can only load one database into a Python process when using PyKX. To automatically load a database when initializing the `#!python pykx.DB` class, set the database location as the path:

```python
>>> import pykx as kx
>>> db = kx.DB(path='/tmp/db')
>>> db.tables
['quote', 'trade']
```

To load a database after initialization, use the `#!python load` command as shown below:

```python
>>> import pykx as kx
>>> db = kx.DB()
>>> db.tables
>>> db.load('/tmp/db')
>>> db.tables
['quote', 'trade']
```

## Change the loaded database

To overwrite the database loaded and use another database if needed, use the `#!python  overwrite` keyword. 

In the below example, we are loading a new database `#!python /tmp/newdb` which in our case doesn't exist but mimics the act of loading a separate database:

```python
>>> db = kx.DB(path='/tmp/db')
>>> db.load(path='/tmp/newdb', overwrite=True)
```

## Next Steps

- [Modify the contents of your database](db_mgmt.md).
- [Query your database with Python](../../fundamentals/query/pyquery.md).
- [Compress/encrypt data](../compress-encrypt.md#persist-database-partitions-with-various-configurations) for persisting database partitions.

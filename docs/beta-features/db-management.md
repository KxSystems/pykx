# Database Management

!!! Warning

	This module is a Beta Feature and is subject to change. To enable this functionality for testing please follow the configuration instructions [here](../user-guide/configuration.md) setting `PYKX_BETA_FEATURES='true'`

## Introduction

The term Database Management as used here, refers to creating and maintaining [partitioned kdb+ databases](https://code.kx.com/q/kb/partition/). Go to [Q for Mortals](https://code.kx.com/q4m3/14_Introduction_to_Kdb+/#143-partitioned-tables) for more in-depth information about partitioned databases in kdb+.

A kdb+ database consists of one or more tables saved on-disk, where they are split into separate folders called partitions. These partitions are most often based on a temporal field within the dataset, such as date or month. Each table within the database must follow the same partition structure. 

We recommend using partitioned databases when the volume of data being handled exceeds ~100 million records.

## Functional walkthrough

This walkthrough will demonstrate the following steps:

1. Creating a database from a historical dataset.
1. Adding a new partition to the database.
1. Managing the on-disk database by:
	1. Renaming a table and column
	1. Creating a copy of a column to the database
	1. Applying a Python function to a column of the database
	1. Updating the data type of a column
1. Adding a new table to the most recent partition of the database.

All integrations with the `Database Management` functionality are facilitated through use of the `pykx.DB` class. To follow along with the example outlined below you can use the [companion notebook](../examples/db-management.ipynb). This uses a more complex table but runs the same commands. For full information on the functions available you can reference the [API section](../api/db.md).

### Creating a database

Create a dataset containing time-series data with multiple dates, and columns of various types:

```python
>>> import pykx as kx
>>> from datetime import date
>>> N = 100000
>>> dataset = kx.Table(data={
...     'date': kx.random.random(N, [date(2020, 1, 1), date(2020, 1, 2)]),
...     'sym': kx.random.random(N, ['AAPL', 'GOOG', 'MSFT']),
...     'price': kx.random.random(N, 10.0)
... })
```

Initialise the `DB` class. The expected input is the file path where you intend to save the partitioned database and its associated tables.

```python
>>> db = kx.DB(path = 'db')
```

Create the database using the `date` column as the partition, and add `dataset` as a table called `trade_data` within it.

```python
>>> db.create(dataset, 'trade_data', 'date', by_field = 'sym', sym_enum = 'symcol')
Writing Database Partition 2020.01.01 to table trade_data
Writing Database Partition 2020.01.02 to table trade_data
```

This now exists as a table and is saved to disk.

```python
>>> db.tables
['trade_data']
```

When a table is saved, an attribute is added to the `db` class for it. For our newly generated table, this is `db.trade_data`

```python
>>> db.trade_data
pykx.PartitionedTable(pykx.q('
date       sym  price    
-------------------------
2020.01.01 AAPL 7.055037 
2020.01.01 AAPL 3.907669 
2020.01.01 AAPL 2.20948  
2020.01.01 AAPL 7.839242 
2020.01.01 AAPL 0.8549648
..
')
```

### Adding a new partition to the database

Once a table has been generated, you can add more partitions to the database through reuse of the `create` method. In this case we are adding the new partition `2020.01.03` to the database.

```python
>>> N = 10000
>>> dataset = kx.Table(data={
...     'sym': kx.random.random(N, ['AAPL', 'GOOG', 'MSFT']),
...     'price': kx.random.random(N, 10.0)
... })
>>> db.create(dataset, 'trade_data', date(2020, 1, 3), by_field = 'sym', sym_enum = 'symcol')
Writing Database Partition 2020.01.03 to table trade_data
```

### Managing the database

This section covers updating the contents of a database. We will continue using the table created in the [Creating a database](#creating-a-database) section above.

The name of a table can be updated using the `rename_table` method. Below, we are updating the table `trade_data` to be called `trade`.

```python
>>> db.rename_table('trade_data', 'trade')
2023.12.08 09:54:22 renaming :/tmp/db/2020.01.01/trade_data to :/tmp/db/2020.01.01/trade
2023.12.08 09:54:22 renaming :/tmp/db/2020.01.02/trade_data to :/tmp/db/2020.01.02/trade
2023.12.08 09:54:22 renaming :/tmp/db/2020.01.03/trade_data to :/tmp/db/2020.01.03/trade
```

During the rename process, the attribute in the `db` class is also updated. 

```python
>>> db.trade
pykx.PartitionedTable(pykx.q('
date       sym  price
-------------------------
2020.01.01 AAPL 7.055037
2020.01.01 AAPL 3.907669
2020.01.01 AAPL 2.20948
2020.01.01 AAPL 7.839242
2020.01.01 AAPL 0.8549648
..
')
```

Renaming a column in a table is achieved using the `rename_column` method. For example, let's update the `sym` column in the `trade` table to be called `ticker`.

```python
>>> db.rename_column('trade', 'sym', 'ticker')
2023.12.08 10:06:27 renaming sym to ticker in `:/tmp/db/2020.01.01/trade
2023.12.08 10:06:27 renaming sym to ticker in `:/tmp/db/2020.01.02/trade
2023.12.08 10:06:27 renaming sym to ticker in `:/tmp/db/2020.01.03/trade
```

To safely apply a function to modify the `price` column within the database, first create a copy of the column.

```python
>>> db.copy_column('trade', 'price', 'price_copy')
2023.12.08 10:14:54 copying price to price_copy in `:/tmp/db/2020.01.01/trade
2023.12.08 10:14:54 copying price to price_copy in `:/tmp/db/2020.01.02/trade
2023.12.08 10:14:54 copying price to price_copy in `:/tmp/db/2020.01.03/trade
```

You can now apply a function to the copied column without the risk of losing the original data. Below we are modifying the copied column by multiplying the contents by 2.

```python
>>> db.apply_function('trade', 'price_copy', lambda x: 2*x)
2023.12.08 10:18:18 resaving column price_copy (type 9) in `:/tmp/db/2020.01.01/trade
2023.12.08 10:18:18 resaving column price_copy (type 9) in `:/tmp/db/2020.01.02/trade
2023.12.08 10:18:18 resaving column price_copy (type 9) in `:/tmp/db/2020.01.03/trade
>>> db.trade
pykx.PartitionedTable(pykx.q('
date       ticker price     price_copy
--------------------------------------
2020.01.01 AAPL   7.055037  14.11007  
2020.01.01 AAPL   3.907669  7.815337  
2020.01.01 AAPL   2.20948   4.418959  
2020.01.01 AAPL   7.839242  15.67848  
2020.01.01 AAPL   0.8549648 1.70993
..
')
```

Once you are happy with the new values within the `price_copy` column, you can safely delete the `price` column, then rename the `price_copy` column to be called `price`.

```python
>>> db.delete_column('trade', 'price')
2023.12.08 10:20:02 deleting column price from `:/tmp/db/2020.01.01/trade
2023.12.08 10:20:02 deleting column price from `:/tmp/db/2020.01.02/trade
2023.12.08 10:20:02 deleting column price from `:/tmp/db/2020.01.03/trade
>>> db.rename_column('trade', 'price_copy', 'price')
2023.12.08 10:06:27 renaming price_copy to price in `:/tmp/db/2020.01.01/trade
2023.12.08 10:06:27 renaming price_copy to price in `:/tmp/db/2020.01.02/trade
2023.12.08 10:06:27 renaming price_copy to price in `:/tmp/db/2020.01.03/trade
>>> db.trade
pykx.PartitionedTable(pykx.q('
date       ticker price
--------------------------
2020.01.01 AAPL   14.11007
2020.01.01 AAPL   7.815337
2020.01.01 AAPL   4.418959
2020.01.01 AAPL   15.67848
2020.01.01 AAPL   1.70993
..
')
```

To convert the data type of a column, you can use the `set_column_type` method. Currently the `price` column is the type `FloatAtom`. We will update this to be a type `RealAtom`.

```python
>>> db.set_column_type('trade', 'price', kx.RealAtom)
2023.12.08 10:28:28 resaving column price (type 8) in `:/tmp/db/2020.01.01/trade
2023.12.08 10:28:28 resaving column price (type 8) in `:/tmp/db/2020.01.02/trade
2023.12.08 10:28:28 resaving column price (type 8) in `:/tmp/db/2020.01.03/trade
```

### Adding a new table to the database

Now that you have successfully set up one table, you may want to add a second table named `quotes`. In this example, the `quotes` table only contains data for `2020.01.03`. We follow the same method as before and create the `quotes` table using the `create` method

```python
>>> quotes = kx.Table(data={
...     'sym': kx.random.random(N, ['AAPL', 'GOOG', 'MSFT']),
...     'open': kx.random.random(N, 10.0),
...     'high': kx.random.random(N, 10.0),
...     'low': kx.random.random(N, 10.0),
...     'close': kx.random.random(N, 10.0)
... })
>>> db.create(quotes, 'quotes', date(2020, 1, 3), by_field = 'sym', sym_enum = 'symcol')
Writing Database Partition 2020-01-03 to table quotes
```

As mentioned in the introduction, all tables within a database must contain the same partition structure. To ensure the new table can be accessed, the quotes table needs to exist in every partition within the database, even if there is no data for that partition. This is called backfilling data. For the partitions where the `quotes` table is missing, we use the `fill_database` method. 

```python
>>> db.fill_database()
Successfully filled missing tables to partition: :/tmp/db/2020.01.01
Successfully filled missing tables to partition: :/tmp/db/2020.01.02
```

Now that the database has resolved the missing tables within the partitions, we can view the new `quotes` table

```python
>>> db.quotes
pykx.PartitionedTable(pykx.q('
date       sym  open      high      low       close    
-------------------------------------------------------
2020.01.03 AAPL 7.456644  7.217498  5.012176  3.623649 
2020.01.03 AAPL 6.127973  0.4229592 7.450608  5.651364 
2020.01.03 AAPL 8.147475  4.459108  3.493555  5.78803  
2020.01.03 AAPL 5.812028  7.81659   5.395469  8.424176 
2020.01.03 AAPL 8.519148  1.18101   6.684017  8.376375
..
')
```

Finally, to view the amount of saved data you can count the number of rows per partition using `partition_count`

```python
>>> db.partition_count()
pykx.Dictionary(pykx.q('
          | quotes trade
----------| -------------
2020.01.01| 0      49859
2020.01.02| 0      50141
2020.01.03| 100000 100000
'))
```

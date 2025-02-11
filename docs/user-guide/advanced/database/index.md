---
title: Databases in PyKX
description: PyKX database creation and management
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, database, maintenance, management, generation
---

# Databases in PyKX

_This page explains the concept of databases in PyKX, including the creation and management of databases._

## What's a PyKX database?

In PyKX, the term database refers to a kdb+ database which can hold a set of [splayed](https://code.kx.com/q/kb/splayed-tables/) and [partitioned](https://code.kx.com/q/kb/partition/) tables.

### Splayed Database

A splayed kdb+ database consists of a single table stored on-disk with each column stored as a separate file rather than using a single file for the whole table. Tables of medium-size with < 100 million rows and many columns are good candidates for being stored as splayed tables, in particular when only a small subset of columns are being accessed often.

```bash
quotes
 ├── .d
 ├── price
 ├── sym
 └── time
```

!!! note "More information on splayed databases"

	The splayed database format used by PyKX has been used in production environments for decades. As such there is a significant amount of information available on the creation and use of these databases. Below are some articles.

	- [q knowledge base splayed databases](https://code.kx.com/q/kb/splayed-tables/)
	- [Q for Mortals splayed tables](https://code.kx.com/q4m3/14_Introduction_to_Kdb%2B/#142-splayed-tables)
	- [Basics of splayed tables](https://thinqkdb.wordpress.com/splayed-tables/)

### Partitioned Database

A partitioned kdb+ database consists of one or more tables saved on-disk, where they are split into separate folders called partitions. These partitions are most often based on a temporal field within the dataset, such as date or month. Each table within the database must follow the same partition structure.

A visual representation of a database containing 2 tables (trade and quote) partitioned by date would be as follows, where `#!python price`, `#!python sym`, `#!python time` in the quotes folder are columns within the table:

```bash
db
├── 2020.10.04
│   ├── quotes
│   │   ├── .d
│   │   ├── price
│   │   ├── sym
│   │   └── time
│   └── trades
│       ├── .d
│       ├── price
│       ├── sym
│       ├── time
│       └── vol
├── 2020.10.06
│   ├── quotes
..
└── sym
```

!!! note "More information on partitioned databases"

	The partitioned database format used by PyKX has been used in production environments for decades in many of the world's best-performing tier-1 investment banks. Today, there is a significant amount of information available on the creation and maintenance of these databases. Below are some articles related to their creation and querying.

	- [Blog: Partitioning data with kdb+](https://kx.com/blog/partitioning-data-in-kdb/)
	- [Q for Mortals Partitioned Tables](https://code.kx.com/q4m3/14_Introduction_to_Kdb%2B/#143-partitioned-tables)
	- [Partitioned Tables](https://thinqkdb.wordpress.com/partitioned-tables/)

## How to use databases in PyKX

Creating and managing databases is crucial for handling large amounts of data. The `#!python pykx.DB` module helps make these tasks easier, Pythonic, and more user-friendly.

PyKX Database API supports the following operations:

| **Operation**            | **Description**                                                                                   |
|:-------------------------|:--------------------------------------------------------------------------------------------------|
| [Generate](db_gen.md)    | Learn how to generate a new historical database using data from Python/q and expand it over time. |
| [Load](db_loading.md)    | Learn how to load existing databases and fix some common issues with databases.                   |
| [Manage](db_mgmt.md)     | Copy, change datatypes or names of columns, apply functions to columns, delete columns from a table, rename tables and backfill data. |

Check out a full breakdown of the [database API](../../../api/db.md).

## Next Steps

- Learn how to create a new database or update an existing one [here](db_gen.md).
- Learn how to load an existing database [here](db_loading.md).

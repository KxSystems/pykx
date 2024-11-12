---
title:  PyKX Glossary
description: Common terms explained for PyKX
date: June 2024
author: KX Systems, Inc.,
tags: glossary
---

# Glossary

_This page contains descriptions of commonly used terms in PyKX._

## Attributes
PyKX attributes are characteristics and features of PyKX objects that define their behavior and interaction within the Python environment. 

## Grouped attribute
The `#!python grouped` attribute ensures that all items in the `#!python Vector`/`#!python Table` column are stored in a different format to help reduce memory usage.

## HDB
An HDB is a mount for historical data within a database. It’s the ultimate repository for interval data. Learn how [HDB](https://code.kx.com/insights/1.11/enterprise/database/configuration/assembly/database.html) works in database configuration.

## IDB
An IDB serves as a mount to store interval data in a database. It collects data from a real-time database (RDB), retains it for a specified duration, such as 10 minutes, and then transfers the data to a historical database (HDB). Learn how [IDB](https://code.kx.com/insights/1.9/enterprise/database/configuration/assembly/database.html) works in database configuration.

## IPC
Interprocess Communication (IPC) forms a central mechanism by which you can connect to and query existing kdb+/q infrastructures. Read more about [communicating via IPC](../user-guide/advanced/ipc.md).

## kdb+ 
kdb+ is a powerful, ultra-fast column-based relational time series database (TSDB) with in-memory (IMDB) capabilities. Operating with temporal data types as a first class entity, the use of q and its query language qSQL against this database creates a highly performant time-series analysis tool. Learn more about [kdb+](https://code.kx.com/q/).

## Mount
In databases, mounting refers to making a set of databases available online. A mounted database is ready for use. A database may have three types of mounts: real-time (RDB), interval (IDB), and/or historic (HDB). Learn more about [mounts](https://code.kx.com/insights/1.9/enterprise/database/configuration/assembly/database.html#mounts) in database configuration.

## Multithreading
Multithreading means running multiple threads concurrently within a single process. It allows better resource utilization and improved responsiveness. When a program has multiple threads, the operating system switches between them rapidly, giving the illusion of parallel execution. Learn about [multithreading in PyKX](../user-guide/advanced/threading.md) and run an [example](../examples/threaded_execution/threading.md).

## Objects
PyKX objects are Python representations of kdb+ data structures that allow developers to interact with kdb+ databases, perform complex queries, and manipulate data efficiently. 

## Object storage 
Object storage is a data storage architecture designed to handle large amounts of unstructured data. A data storage system that manages data as objects is distinct from file hierarchy or block-based storage architectures. Object storage is ideal for unstructured data because it overcomes the scaling limitations of traditional file storage systems. The capacity for limitless scaling is why object storage is the backbone of cloud storage; major players like Amazon, Google, and Microsoft utilize object storage as their primary data storage solution. Learn [how to interact with PyKX objects](../user-guide/fundamentals/creating.md) or [how to index PyKX objects](../user-guide/fundamentals/indexing.md).

## Parallelization
Parallelization involves distributing computational tasks across multiple threads to improve performance and efficiency.

## Parted attribute
The `#!python parted` attribute is similar to the `#!python grouped` attribute with the additional requirement that each unique value must be adjacent to its other copies, where the grouped attribute allows them to be dispersed throughout the `#!python Vector`/`#!python Table`. 

## Partitioned database
A partitioned database is a database that is divided into smaller, more manageable units, improving scalability while maintaining security. These partitions can be created for various reasons, such as manageability, performance optimization, availability, or load balancing. Learn more about [creating and maintaining partitioned kdb+ databases](https://code.kx.com/q/kb/partition/). Go to [Q for Mortals](https://code.kx.com/q4m3/14_Introduction_to_Kdb+/#143-partitioned-tables) for in-depth information about partitioned databases in kdb+.

## Partitioning 
When writing a data table to a database, it must be partitioned to ensure compatibility with a kdb+ time series database. In PyKX, partitioning is managed through a `#!python Timestamp` column defined in the schema, and each table is required to have a `#!python Timestamp` column. Learn more about [partitioning tables across directories](https://code.kx.com/q/kb/partition/).

## Persisted database
A persisted database (or on-disk database) stores data on non-volatile storage like a hard drive or SSD, ensuring the data remains intact even after the application is closed or the system is restarted. In contrast, in-memory databases store data in RAM and lose all data when the system is powered down. Persisted databases are crucial for applications needing long-term data storage and reliability, such as financial systems, customer databases, and many web applications.

## Python byte object
A Python byte object is an immutable sequence of bytes, used to handle binary data. Each byte is an integer between 0 and 255. Byte objects are essential for tasks like file I/O and network communication.

## q 
q is a versatile vector programming language mainly used to query a kdb+ database. q is known both for its speed and expressiveness. [Learn more about q](https://code.kx.com/q/learn/) including [q terminology](https://code.kx.com/q/basics/glossary/).

## q/SQL 
q/SQL is a collection of SQL-like functions for interacting with a kdb+ database. Learn more about [q/SQL](https://code.kx.com/q4m3/9_Queries_q-sql/).

## RDB
Real-time event data is stored on an RDB mount of the database, before being moved to the interval database (IDB). Learn how [RDB](https://code.kx.com/insights/1.9/enterprise/database/configuration/assembly/database.html) works in database configuration.

## Schema
A database schema is a fundamental concept in database management. It describes how data is structured within a relational database. It serves as a blueprint for the database’s architecture, outlining the relationships between different entities, such as tables, columns, data types, views, stored procedures, primary keys, and foreign keys. Learn more about [API schema generation in PyKX](../api/schema.md) or [schema configuration in kdb Insights Enterprise](https://code.kx.com/insights/1.9/enterprise/database/configuration/assembly/schema.html).

## Sorted attribute
The `#!python sorted` attribute ensures that all items in the `#!python Vector`/`#!python Table` column are sorted in ascending order.

## Time-series analysis
Time-series analysis is a specific way of analyzing a sequence of data points collected over an interval of time. Unlike sporadic or random data collection, time-series analysis involves recording data points at consistent intervals within a set period. Use cases include finance, Internet of Things (IoT), and other domains where data evolves over time. PyKX time-series analysis uses q’s query language (qSQL) against kdb+. Learn more about [time-series models](https://code.kx.com/insights/1.9/api/machine-learning/q/analytics/api/variadic/timeseries.html).

## Thread
A thread is an independent sequence of instructions within a program that can be executed independently of other code. Threads share the same memory space as the process they belong to, allowing them to communicate and share data efficiently. In Python, the threading module provides an intuitive API for working with threads. Learn about [PyKX calling into q from multiple threads](../examples/threaded_execution/threading.md).

## Unique attribute
The `#!python unique` attribute ensures that all items in the `#!python Vector`/`#!python Table` column are unique (there are no duplicated values).

## Upsert
In the context of databases, upsert is an operation that combines both updating and inserting data into a table. When you perform an upsert, the database checks whether a record with a specific key already exists in the table. If a record with that key exists, the database updates the existing record with new values. If no record with that key exists, the database inserts a new record with the provided data. Learn more about [upsert](../api/pykx-execution/q.md#upsert).

## Vector
A vector is a mathematical concept used to represent quantities that have both magnitude (size) and direction. In other words, vectors are arrays of numerical values that represent points in multidimensional space. A vector is typically represented as an arrow in space, pointing from one point to another. Learn more about [using in-built methods on PyKX vectors](../examples/interface-overview.ipynb#using-in-built-methods-on-pykx-vectors) and [adding values to PyKX vectors/lists](../user-guide/fundamentals/indexing.md#assigning-and-adding-values-to-vectorslists).

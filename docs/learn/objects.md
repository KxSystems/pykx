---
title: PyKX Objects and Attributes
description: Objects explained for PyKX 
date: September 2024
author: KX Systems, Inc.,
tags: PyKX, object
---

# PyKX objects and attributes

_This page explains objects and attributes in PyKX._

## What are PyKX objects?

PyKX objects are Python representations of kdb+ data structures. They allow Python developers to interact with kdb+ databases, perform complex queries, and manipulate data efficiently.

When you call or connect to a q instance, it returns a PyKX object. This object is an instance of the [`#!python pykx.K`](../api/pykx-q-data/wrappers.md#pykx.wrappers.K) class or one of its subclasses, as documented on the [PyKX wrappers API](..//api/pykx-q-data/wrappers.md) page. 

PyKX objects act as wrappers around objects in q’s memory space within the Python process where PyKX (and your program) runs. These wrappers are efficient to create since they don’t require copying data out of q’s memory space. 

PyKX objects support various Python features like iteration, slicing, and calling, so converting them to other types (for example, from [`#!python pykx.Vector`](../api/pykx-q-data/wrappers.md#pykx.wrappers.Vector) to `#!python numpy.ndarray`) is often unnecessary.

Examples of PyKX objects:

- **Atoms**: Single values, such as integers, floats, or symbols.
- **Vectors**: Arrays of values of the same type.
- **Dictionaries**: Key-value pairs, where keys and values can be of different types.
- **Tables**: Collections of columns, where each column is a vector.
- **Lists**: These can contain elements of different types.

### How to use PyKX objects

To leverage the power of kdb+ within a Python environment, you can perform the following key operations with PyKX objects:

| **Operation**                                                | **Description** |
|--------------------------------------------------------------|-------------|
| [Create and convert](../user-guide/fundamentals/creating.md) | Create PyKX objects from and to various Python objects, such as lists, dictionaries, and NumPy arrays. |
| [Use](../user-guide/fundamentals/evaluating.md)              | Once created, interact with PyKX objects using familiar Pythonic syntax. For example [querying tables](../user-guide/fundamentals/query/pyquery.md) using Python. |
| [Index](../user-guide/fundamentals/indexing.md)              | Indexing PyKX objects allows you to access and manipulate elements within these objects, similar to how you would with standard Python sequences.|

## What are PyKX attributes?

Attributes are metadata that you attach to lists with special forms. They are also used on table columns to speed up retrieval for certain operations. PyKX can optimize based on the list structure implied by the attribute.

Attributes (except for  ``#!python `g#``) are descriptive rather than prescriptive. This means that by applying an attribute, you are asserting that the list has a special form, which PyKX will verify. It does not instruct PyKX to create or remake the list into its special form; that is your responsibility. If a list operation respects the form specified by the attribute, the attribute remains intact (except for  ``#!python `p#``). However, if an operation breaks the form, the attribute is removed from the result.

Learn how to [apply attributes](../user-guide/advanced/attributes.md) in PyKX.

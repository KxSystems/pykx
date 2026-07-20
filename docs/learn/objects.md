---
title: pykx Objects and Attributes
description: Objects explained for KDB-X Python 
date: September 2024
author: KX Systems, Inc.,
tags: KDB-X Python, object
---

# `pykx` objects and attributes

_This page explains objects and attributes in KDB-X Python._

## What are `pykx` objects?

`pykx` objects are Python representations of KDB-X data structures. They allow Python developers to interact with KDB-X databases, perform complex queries, and manipulate data efficiently.

When you call or connect to a q instance, it returns a KDB-X Python object. This object is an instance of the [`#!python pykx.K`](../api/pykx-q-data/wrappers.md#pykx.wrappers.K) class or one of its subclasses, as documented on the [KDB-X Python wrappers API](..//api/pykx-q-data/wrappers.md) page. 

`pykx` objects act as wrappers around objects in q’s memory space within the Python process where KDB-X Python (and your program) runs. These wrappers are efficient to create since they don’t require copying data out of q’s memory space. 

`pykx` objects support various Python features like iteration, slicing, and calling, so converting them to other types (for example, from [`#!python pykx.Vector`](../api/pykx-q-data/wrappers.md#pykx.wrappers.Vector) to `#!python numpy.ndarray`) is often unnecessary.

Examples of `pykx` objects:

- **Atoms**: Single values, such as integers, floats, or symbols.
- **Vectors**: Arrays of values of the same type.
- **Dictionaries**: Key-value pairs, where keys and values can be of different types.
- **Tables**: Collections of columns, where each column is a vector.
- **Lists**: These can contain elements of different types.

### How to use `pykx` objects

To leverage the power of KDB-X within a Python environment, you can perform the following key operations with `pykx` objects:

| **Operation**                                                | **Description** |
|--------------------------------------------------------------|-------------|
| [Create and convert](../user-guide/fundamentals/creating.md) | Create `pykx` objects from and to various Python objects, such as lists, dictionaries, and NumPy arrays. |
| [Use](../user-guide/fundamentals/evaluating.md)              | Once created, interact with `pykx` objects using familiar Pythonic syntax. For example [querying tables](../user-guide/fundamentals/query/pyquery.md) using Python. |
| [Index](../user-guide/fundamentals/indexing.md)              | Indexing `pykx` objects allows you to access and manipulate elements within these objects, similar to how you would with standard Python sequences.|

## What are `pykx` attributes?

Attributes are metadata that you attach to lists with special forms. They are also used on table columns to speed up retrieval for certain operations. KDB-X Python can optimize based on the list structure implied by the attribute.

Attributes (except for  ``#!python `g#``) are descriptive rather than prescriptive. This means that by applying an attribute, you are asserting that the list has a special form, which KDB-X Python will verify. It does not instruct KDB-X Python to create or remake the list into its special form; that is your responsibility. If a list operation respects the form specified by the attribute, the attribute remains intact (except for  ``#!python `p#``). However, if an operation breaks the form, the attribute is removed from the result.

Learn how to [apply attributes](../user-guide/advanced/attributes.md) in KDB-X Python.

---
title: Apply attributes in PyKX
description: How to use attributes in PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, PyKX objects, 
---

# Apply Attributes
_This page provides details on how to apply attributes in PyKX._

!!! tip "Tip: For the best experience, we recommend reading about [PyKX attributes](..//../learn/objects.md#what-are-pykx-attributes) first." 

In PyKX, you can apply attributes to various data structures, including `#!python Vector`/`#!python List` types, `#!python Tables`, and `#!python KeyedTable`s. To apply the attributes, call the `#!python sorted`, `#!python unique`, `#!python grouped`, and `#!python parted` methods on these objects.

### Sorted

The `#!python sorted` attribute ensures that all items in the `#!python Vector` / `#!python Table` column are sorted in ascending
order. This attribute will be removed if you append to the list with an item that is not in sorted
order.


!!! example "Example of applying the `sorted` attribute to a `Vector` by calling the `sorted` method on the `Vector`:"

```Python
>>> a = kx.q.til(10)
>>> a
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
>>> a.sorted()
pykx.LongVector(pykx.q('`s#0 1 2 3 4 5 6 7 8 9'))
```

### Unique

The `#!python unique` attribute ensures that all items in the `#!python Vector` / `#!python Table` column are unique (there are
no duplicated values). This attribute will be removed if you append to the list with an item that
is not unique.

!!! example "Example of applying the `unique` attribute to the first column of the table:"

```Python
>>> a = kx.Table(data = {
...     'a': kx.q.til(5),
...     'b': ['a', 'b', 'c', 'd', 'e']
... })
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j
b| s
'))
>>> a = a.unique()
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j   u
b| s
'))
```

### Grouped

The `#!python grouped` attribute ensures that all items in the `#!python Vector` / `#!python Table` column are stored in a
different format to help reduce memory usage. It creates a backing dictionary to store the value and
indexes that each value has within the list. 

Unlike other attributes, the `#!python grouped` attribute will be kept on all insert operations to the list. For instance, this is how a grouped list would be stored:

```q
// The list
`g#`a`b`c`a`b`b`c
// The backing dictionary
a| 0 3
b| 1 4 5
c| 2 6
```

!!! example "Example of applying the `#!python grouped` attribute to a specified column of a table:"

```Python
>>> a = kx.Table(data = {
...     'a': kx.q.til(5),
...     'b': ['a', 'a', 'b', 'b', 'b']
... })
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j
b| s
'))
>>> a = a.grouped('b')
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j
b| s   g
'))
```

### Parted

The `#!python parted` attribute is similar to the `#!python grouped` attribute with the additional requirement that each unique value must be adjacent to its other copies, where the grouped attribute allows them to be dispersed throughout the `#!python Vector` / `#!python Table`. 

When possible, the `#!python parted` attribute results in a larger performance gain than using the `#!python grouped` attribute. This attribute will be removed if you append to the list with an item that is not in the `#!python parted`
order.

```q
// Can be parted
`p#`a`a`a`e`e`b`b`c`c`c`d
// Has to be grouped as the `d symbols are not all contiguous within the vector
`g#`a`a`d`e`e`b`b`c`c`c`d
```

!!! example "Example of applying the `parted` attribute to multiple columns on a table:"

```Python
>>> a = kx.Table(data = {
...     'a': kx.q.til(5),
...     'b': ['a', 'a', 'b', 'b', 'b']
... })
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j
b| s
'))
>>> a = a.parted(['a', 'b'])
>>> kx.q.meta(a)
pykx.KeyedTable(pykx.q('
c| t f a
-| -----
a| j   p
b| s   p
'))
```

## Performance

When attributes are set on PyKX objects, various functions can use these attributes to speed up their
execution, by using different algorithms. For example, searching through a list without an attribute
requires checking every single value. However, setting the `#!python sorted` attribute allows a search algorithm
to use a binary search instead and then only a fraction of the values actually needs to be checked.

Examples of functions that can use attributes to speed up execution:

- Where clauses in `#!python select` and `#!python exec` templates run faster with `#!python where =`, `#!python where in` and `#!python where within`.
- Searching with [`#!python bin`](../../api/pykx-execution/q.md#bin), [`#!python distinct`](../../api/pykx-execution/q.md#distinct),
    [`#!python Find`](https://code.kx.com/q/ref/find/) and [`#!python in`](https://code.kx.com/q/ref/in/).
- Sorting with [`#!python iasc`](../../api/pykx-execution/q.md#iasc) or [`#!python idesc`](../../api/pykx-execution/q.md#idesc).

!!!Note
    Setting attributes consumes resources and is likely to improve performance on large lists.

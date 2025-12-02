---
title: Convert data types in PyKX 
description: Converting data types in PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, data, convert
---

# PyKX conversion considerations

_This page provides details on data types and conversions in PyKX._

PyKX attempts to make conversions between q and Python as seamless as possible.
However due to differences in their underlying implementations there are cases where 1 to 1 mappings are not possible.

## Data types and conversions

The key PyKX APIs around data types and conversions are outlined under:

* [Convert Pythonic data to PyKX](../../api/pykx-q-data/toq.md)
* [PyKX type wrappers](../../api/pykx-q-data/wrappers.md)
* [PyKX to Pythonic data type mapping](../../api/pykx-q-data/type_conversions.md)
* [Registering Custom Conversions](../../api/pykx-q-data/register.md)

## Text representation in PyKX

Handling and converting [text in PyKX](./text.md) requires consideration as there are some key differences between the `Symbol` and `Char` data types.

## Nulls and Infinites

Most q datatypes have the concepts of null, negative infinity, and infinity. Python does not have the concept of infinites and it's null behavior differs in implementation. The page [handling nulls and infinities](./nulls_and_infinities.md) details the needed considerations when dealing with these special values.

## Temporal data types

Converting [temporal data types](./temporal.md) in PyKX involves handling [timestamp/datetime](./temporal.md#timestampdatetime-types) types and [duration](./temporal.md#duration-types) types, each with specific considerations due to differences in how Python and q (the language used by kdb+) represent these data types.

## List conversion considerations

By default the library converts generic PyKX List objects `#!python pykx.List` to NumPy as an array of NumPy arrays. This conversion is chosen as it allows for the most flexible representation of data allowing ragged array representations and mixed lists of objects to be converted easily. However, this representation can be difficult to work with if/when dealing with multi-dimensional numeric data as is common in machine learning tasks for example.

As an example we can look at the conversion of a 3-Dimensional regularly shaped `#!python pykx.List` object to a NumPy array as follows:

```python
>>> import pykx as kx
>>> qlist =  kx.random.random([2, 2, 2], 5.0)
pykx.List(pykx.q('
3.453383  3.388243 0.8355005 4.325851
0.6168138 3.450051 3.849182  2.360245
'))
>>> qlist.np()
array([array([array([3.45338272, 3.3882429 ]), array([0.83550046, 4.32585143])],
             dtype=object)                                                      ,
       array([array([0.61681376, 3.45005117]), array([3.84918233, 2.36024517])],
             dtype=object)                                                      ],
      dtype=object)
```

This representation clearly is more difficult to handle than you might expect for a regularly shaped numeric dataset of single type. A keyword argument `#!python reshape` is provided to facilitate a better converted representation of these singularly typed N-Dimensional lists, for example:

```python
>>> import pykx as kx
>>> qlist = kx.random.random([2, 2, 2], 5.0)
>>> qlist.np(reshape=True)
array([[[3.45338272, 3.3882429 ],
        [0.83550046, 4.32585143]],
       [[0.61681376, 3.45005117],
        [3.84918233, 2.36024517]]])
```

Setting the `#!python reshape` keyword to `#!python True` checks if the input list is "rectangular" and contains only one data type before converting it to a single NumPy array by ['razing'](https://code.kx.com/q/ref/raze/) the data to a single array and reshaping the data in NumPy post conversion. 

This can be slow for nested arrays or many list elements. If you know the input and output shape of the data, you can pass this shape to the `#!python reshape` keyword like this:

```python
>>> import pykx as kx
>>> qlist = kx.random.random([10000, 100, 10], 10.0)
>>> qlist.np(reshape=[10000, 100, 10])
array([[[4.99088645, 9.20164969, 3.3486574 , ..., 9.28529354,
         7.78650336, 0.9355585 ],
        [9.49664481, 0.79703755, 8.41364461, ..., 5.28080439,
         7.3933825 , 7.40476901],
        [6.03204263, 9.40702084, 6.75116092, ..., 2.43375089,
         9.33645056, 8.56930709],
        ...
```

The performance boost from knowing the shape ahead of time is significant

```python
import pykx as kx
qlist = kx.random.random([10000, 100, 10], 10.0)
%timeit qlist.np(reshape=True)
# 974 ms ± 34.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
%timeit qlist.np(reshape=[10000, 100, 10])
# 81.2 ms ± 2.69 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

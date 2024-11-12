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

Most q datatypes have the concepts of null, negative infinity, and infinity. Python does not have the concept of infinites and it's null behaviour differs in implementation. The page [handling nulls and infinities](./nulls_and_infinities.md) details the needed considerations when dealing with these special values.

## Temporal data types

Converting [temporal data types](./temporal.md) in PyKX involves handling [timestamp/datetime](./temporal.md#timestampdatetime-types) types and [duration](./temporal.md#duration-types) types, each with specific considerations due to differences in how Python and q (the language used by kdb+) represent these data types.
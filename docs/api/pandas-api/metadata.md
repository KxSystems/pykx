---
title: pandas API metadata and aggregations
description: API reference for the pandas API metadata and aggregations methods of pykx.Table
author: KX Systems
date: October 2026
tags: pandas, table, api
---
# Metadata and aggregations

_Properties describing a table, detection of null values and aggregations which reduce each column or row to a single value. These methods are available on `pykx.Table` and `pykx.KeyedTable` objects._

These methods follow the behavior of the pandas method of the same name, not the q function of that name. The differences from the q functions are:

| Method | q function | Difference |
| --- | --- | --- |
| `sum`, `prod`, `min`, `max` | `sum`, `prd`, `min`, `max` | `skipna=False` returns null if the column contains a null. q always skips nulls. |
| `median` | `med` | Nulls are skipped. q counts a null as the smallest value. |
| `std` | `sdev`, `dev` | Normalized by N - 1 by default. `ddof` sets the divisor to N - `ddof`. |
| `count` | `count` | Counts the values which are not null. q counts every value. |
| `any` | `any` | Nulls are skipped by default. q treats a null as true. A null symbol is false. |
| `all` | `all` | A null never makes the result false. A null symbol is false. |
| `idxmax`, `idxmin` | `x?max x`, `x?min x` | Returns the index label rather than the position. |

`sem`, `skew`, `kurt` and `mode` have no q equivalent.

::: pykx.pandas_api.pandas_meta.PandasMeta
    rendering:
      show_root_heading: false
      show_root_toc_entry: false
      members_order: source

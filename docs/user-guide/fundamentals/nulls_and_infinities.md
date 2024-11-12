---
title: Convert nulls and infinities
description: How to handle nulls and infinities in PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, nulls, infinity
---

# Convert nulls and infinities

_This page explains how to handle nulls and infinities in PyKX._

PyKX handles nulls and infinities in ways that are subtly different from familiar libraries like NumPy, due to its q legacy.

PyKX provides typed null and infinity values for most types. [As shown in the q docs](https://code.kx.com/q/ref/#datatypes):

- nulls can be expressed as `#!python 0N` followed by a type character (or no type character for long integer null).
- infinities can be expressed as `#!python 0W` followed by a type character (or no type character for a long integer infinity).

Datatypes in q designate a particular value in their numeric range as null, and another two as positive and negative infinity. Most other languages, such as Python, have no way to represent infinity for anything other than IEEE floating point numbers, and where typed nulls exist, they will not also be a value in the range of the datatype (save for floats, which can be `#!python NaN`).

!!! example "For example, the q null short integer `#!python 0Nh` is stored as the value `#!python -32768` (i.e. the smallest possible signed 16 bit integer), and the q infinite short integer is stored as the value `#!python 32767` (i.e the largest possible signed 16 bit integer)."

Due to the design of nulls and infinites in q, there are some technical considerations - detailed on this page - regarding converting nulls and infinities between Python and q in either direction.

## Generation of null and infinite values

Here are some examples demonstrating how to create various null and infinite values.

### Null generation

Where possible, null values return the following:

```python
>>> import pykx as kx
>>> kx.LongAtom.null
pykx.LongAtom(pykx.q('0N'))
>>> kx.TimespanAtom.null
pykx.TimespanAtom(pykx.q('0Nn'))
>>> kx.GUIDAtom.null
pykx.GUIDAtom(pykx.q('00000000-0000-0000-0000-000000000000'))
>>> kx.SymbolAtom.null
pykx.SymbolAtom(pykx.q('`'))
```

Unsupported values return a `#!python NotImplemetedError` as below:

```python
>>> import pykx as kx
>>> kx.ByteAtom.null
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/wrappers.py", line 250, in null
    raise NotImplementedError('Retrieval of null values not supported for this type')
NotImplementedError: Retrieval of null values not supported for this type
```

### Infinite generation

Where possible, positive and negative infinite values return the following:

```python
>>> import pykx as kx
>>> kx.TimeAtom.inf
pykx.TimeAtom(pykx.q('0Wt'))
>>> kx.TimeAtom.inf_neg
pykx.TimeAtom(pykx.q('-0Wt'))
>>> kx.IntAtom.inf
pykx.IntAtom(pykx.q('0Wi'))
>>> kx.IntAtom.inf_neg
pykx.IntAtom(pykx.q('-0Wi'))
```

Unsupported values return a `#!python NotImplementedError`:

```python
>>> kx.SymbolAtom.inf
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/wrappers.py", line 274, in inf
    raise NotImplementedError('Retrieval of infinite values not supported for this type')
NotImplementedError: Retrieval of infinite values not supported for this type
```

## Checking for nulls and infinities

If you apply [the q function named null](https://code.kx.com/q/ref/null/) to most PyKX objects, it returns `#!python 1b` if the object is null. If it contains nulls, returns a collection of booleans whose shape matches the object. Like with any function from the `#!python .q` namespace, you can access it via the [context interface](../../api/pykx-execution/ctx.md): [`#!python q.null`](../../api/pykx-execution/q.md#null).

```python
>>> import pykx as kx
>>> kx.q.null(kx.q('0n'))
pykx.BooleanAtom(pykx.q('1b'))
>>> kx.q.null(('1 2 0n 3f'))
pykx.BooleanVector(pykx.q('0010b'))
```

[`#!python pykx.Atom`][pykx.Atom] objects provide the properties:

- `#!python is_null`: `#!python True` if the atom is a null value for it's type else `#!python False`.
- `#!python is_inf`: `#!python True` if the atom is an infinite value (positive or negative) for its type else `#!python False`.
- `#!python is_pos_inf`: `#!python True` if the atom is a positive infinite value for its type else `#!python False`.
- `#!python is_neg_inf`: `#!python True` if the atom is a negative infinite value for its type else `#!python False`.

`#!python is_inf`/`#!python is_pos_inf`/`#!python is_neg_inf` are always `#!python False` for types which do not have an infinite value in q, such as [`#!python pykx.SymbolAtom`][pykx.SymbolAtom].

```python
>>> kx.q('0w').is_inf
True
>>> kx.q('1f').is_inf
False
>>> kx.q('-0Wf').is_inf
True
>>> kx.q('-0Wf').is_pos_inf
False
>>> kx.q('-0Wf').is_neg_inf
True
>>> kx.q('1f').is_null
False
>>> kx.q('0n').is_null
True
```

Likewise, all [`#!python pykx.Collection`][pykx.Collection] objects provide the properties `#!python has_nulls` and `#!python has_infs`. They are `#!python True` if the collection has any nulls/infinities in it.

```python
>>> kx.q('0w,9?1f').has_infs
True
>>> kx.q('0w,9?1f').has_nulls
False
```

Some null values are unintuitive. For instance, the null value for a character in q is the space `#!python " "`, the null value for a symbol is the empty symbol, and the null value for a GUID is `#!python 00000000-0000-0000-0000-000000000000`. A char vector (i.e. q string) that has any spaces has `#!python has_nulls` set to `#!python True`.
See also the page with specifics on [temporal](./temporal.md) conversions.

## q to Python

### Null conversions

!!! note "Note"

    PyKX null conversion behaviour changed in version 3.0.0. The below table outlines the before and after conversions.

    === ".py()"

        | datatype  | q value | 2.* atom conversion | 2.* vector conversion | 3.* atom conversion | 3.* vector conversion |
        |-----------|---------|---------------------|-----------------------|---------------------|-----------------------|
        | guid      | `0Ng`   | `UUID(int=0)`       | `UUID(int=0)`         |                     |                       |
        | short     | `0Nh`   | `q('0Nh')`          | `q('0Nh')`            | `pd.NA`             | `pd.NA`               |
        | int       | `0Ni`   | `q('0Ni')`          | `q('0Ni')`            | `pd.NA`             | `pd.NA`               |
        | long      | `0Nj`   | `q('0N')`           | `q('0N')`             | `pd.NA`             | `pd.NA`               |
        | real      | `0Ne`   | `float('nan')`      | `float('nan')`        |                     |                       |
        | float     | `0n`    | `float('nan')`      | `float('nan')`        |                     |                       |
        | character | `" "`   | `b' '`              | `b' '`                |                     |                       |
        | symbol    | `` ` `` | `''`                | `''`                  |                     |                       |
        | timestamp | `0Np`   | `None`              | `q('0Np')`            | `pd.NaT`            | `pd.NaT`              |
        | month     | `0Nm`   | `None`              | `q('0Nm')`            | `pd.NaT`            | `pd.NaT`              |
        | date      | `0Nd`   | `None`              | `q('0Nd')`            | `pd.NaT`            | `pd.NaT`              |
        | timespan  | `0Nn`   | `pd.NaT`            | `q('0Nn')`            |                     | `pd.NaT`              |
        | minute    | `0Nu`   | `pd.NaT`            | `q('0Nu')`            |                     | `pd.NaT`              |
        | second    | `0Nv`   | `pd.NaT`            | `q('0Nv')`            |                     | `pd.NaT`              |
        | time      | `0Nt`   | `pd.NaT`            | `q('0Nt')`            |                     | `pd.NaT`              |

    === ".np()"

        | datatype  | q value | 2.* atom conversion     | 2.* vector conversion            | 3.* atom conversion              | 3.vector conversion |
        |-----------|---------|-------------------------|----------------------------------|----------------------------------|---------------------|
        | guid      | `0Ng`   | `UUID(int=0)`           | `UUID(int=0)`                    |                                  |                     |
        | short     | `0Nh`   | **1                     | `np.int16(-32768)`               | `np.int16(-32768)`               |                     |
        | int       | `0Ni`   | **1                     | `np.int32(-2147483648)`          | `np.int32(-2147483648)`          |                     |
        | long      | `0Nj`   | **1                     | `np.int64(-9223372036854775808)` | `np.int64(-9223372036854775808)` |                     |
        | real      | `0Ng`   | `np.float32('nan')`     | `np.float32('nan')`              |                                  |                     |
        | float     | `0n`    | `np.float64('nan')`     | `np.float64('nan')`              |                                  |                     |
        | character | `" "`   | `b' '`                  | `np.bytes_(' ')`                 |                                  |                     |
        | symbol    | `` ` `` | `''`                    | `''`                             |                                  |                     |
        | timestamp | `0Np`   | `np.datetime64('NaT')`  | `np.datetime64('NaT')`           |                                  |                     |
        | month     | `0Nm`   | `np.datetime64('NaT')`  | `np.datetime64('NaT')`           |                                  |                     |
        | date      | `0Nd`   | `np.datetime64('NaT')`  | `np.datetime64('NaT')`           |                                  |                     |
        | timespan  | `0Nn`   | `np.timedelta64('NaT')` | `np.timedelta64('NaT')`          |                                  |                     |
        | minute    | `0Nu`   | `np.timedelta64('NaT')` | `np.timedelta64('NaT')`          |                                  |                     |
        | second    | `0Nv`   | `np.timedelta64('NaT')` | `np.timedelta64('NaT')`          |                                  |                     |
        | time      | `0Nt`   | `np.timedelta64('NaT')` | `np.timedelta64('NaT')`          |                                  |                     |

        - **1 Errors: `NumPy does not support null atomic integral values for short int long`

    === ".pd()"

        | datatype  | q value | 2.* atom conversion | 2.* vector conversion | 3.* atom conversion | 3.* vector conversion |
        |-----------|---------|---------------------|-----------------------|---------------------|-----------------------|
        | guid      | `0Ng`   | `UUID(int=0)`       | `UUID(int=0)`         |                     |                       |
        | short     | `0Nh`   | **1                 | `pd.NA`               | `pd.NA`             |                       |
        | int       | `0Ni`   | **1                 | `pd.NA`               | `pd.NA`             |                       |
        | long      | `0Nj`   | **1                 | `pd.NA`               | `pd.NA`             |                       |
        | real      | `0Ne`   | `np.float32('nan')` | `np.float32('nan')`   |                     |                       |
        | float     | `0n`    | `np.float64('nan')` | `np.float64('nan')`   |                     |                       |
        | character | `" "`   | `b' '`              | `np.bytes_(' ')`      |                     |                       |
        | symbol    | `` ` `` | `''`                | `''`                  |                     |                       |
        | timestamp | `0Np`   | `pd.NaT`            | `pd.NaT`              |                     |                       |
        | month     | `0Nm`   | `pd.NaT`            | `pd.NaT`              |                     |                       |
        | date      | `0Nd`   | `pd.NaT`            | `pd.NaT`              |                     |                       |
        | timespan  | `0Nn`   | `pd.NaT`            | `pd.NaT`              |                     |                       |
        | minute    | `0Nu`   | `pd.NaT`            | `pd.NaT`              |                     |                       |
        | second    | `0Nv`   | `pd.NaT`            | `pd.NaT`              |                     |                       |
        | time      | `0Nt`   | `pd.NaT`            | `pd.NaT`              |                     |                       |

        - **1 Errors: `NumPy does not support null atomic integral values for short int long`

    === ".pa()"

        | datatype  | q value | 2.* atom conversion  | 2.* vector conversion                                 | 3.* atom conversion | 3.* vector conversion |
        |-----------|---------|---------------------|-------------------------------------------------------|---------------------|-----------------------|
        | guid      | `0Ng`   | `UUID(int=0)`       | **1                                                   |                     |                       |
        | short     | `0Nh`   | **2                 | **2                                                   |     `pd.NA`         |                       |
        | int       | `0Ni`   | **2                 | **2                                                   |     `pd.NA`         |                       |
        | long      | `0Nj`   | **2                 | **2                                                   |     `pd.NA`         |                       |
        | real      | `0Ne`   | `np.float32('nan')` | `pa.array([np.float32('nan')], type=pa.float32())[0]` |                     |                       |
        | float     | `0n`    | `np.float64('nan')` | `pa.array([np.float32('nan')], type=pa.float64())[0]` |                     |                       |
        | character | `" "`   | `b' '`              | `pa.array([b' '], pa.binary())[0]`                    |                     |                       |
        | symbol    | `` ` `` | `''`                | `pa.array([''], pa.string())[0]`                      |                     |                       |
        | timestamp | `0Np`   | `pd.NaT`            | **3                                                   |                     |                       |
        | month     | `0Nm`   | `pd.NaT`            | **3                                                   |                     |                       |
        | date      | `0Nd`   | `pd.NaT`            | **3                                                   |                     |                       |
        | timespan  | `0Nn`   | `pd.NaT`            | **4                                                   |                     |                       |
        | minute    | `0Nu`   | `pd.NaT`            | **4                                                   |                     |                       |
        | second    | `0Nv`   | `pd.NaT`            | **4                                                   |                     |                       |
        | time      | `0Nt`   | `pd.NaT`            | **4                                                   |                     |                       |

        - **1 Errors: `Could not convert UUID('00000000-0000-0000-0000-000000000000') with type UUID: did not recognize Python value type when inferring an Arrow data type`
        - **2 Errors: `NumPy does not support null atomic integral values for short int long`
        - **3 Errors: `pyarrow.lib.ArrowNotImplementedError: Unbound or generic datetime64 time unit`
        - **4 Errors: `pyarrow.lib.ArrowNotImplementedError: Unbound or generic timedelta64 time unit`

To convert vectors with the q types `#!python short`, `#!python int`, and `#!python long` to Python, you can use the following methods:

|**Method**|**Description**|
|----------|---------------|
|`.py` | Provides a list with the null values converted to the closest possible Python representation |
|`.np`| Provides a [masked array](https://numpy.org/doc/stable/reference/maskedarray.html) with the null values masked out, and the fill value set to the underlying value of the q null.|
|`.pd`| Provides a [Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) as per usual, but backed by an [IntegerArray](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.arrays.IntegerArray.html) instead of a regular `np.ndarray`.|
|`.pa`| Provides a [PyArrow Array](https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array) as per usual, which natively supports nullable integral vector data, so it simply has the indexes of the nulls stored in the array metadata.|

!!! note "Note"

    - **Real vectors** use the standard `#!python NaN` and `#!python inf` values, and so are handled by q, Python, NumPy, Pandas, and PyArrow in the same way with no special handling.

    - **Temporal vectors** use `#!python NaT` to represent null values in Python, NumPy, Pandas, and PyArrow represents null temporal values like it does for any other data type: by masking it out using the array metadata.

When converting a table from q to Python with one of the methods above, each column is transformed as an independent vector as described above.

The following provides an example of the masked array behavior outlined in the `#!python .np` method, which is additionally exhibited by the `#!python .pd` method.

```python
>>> import pykx as kx
>>> df = kx.q('([] til 10; 0N 5 10 15 0N 20 25 30 0N 35)').pd()
>>> print(df)
   x  x1
0  0  --
1  1   5
2  2  10
3  3  15
4  4  --
5  5  20
6  6  25
7  7  30
8  8  --
9  9  35
>>> kx.toq(df)
pykx.Table(pykx.q('
x x1
----
0
1 5
2 10
3 15
4
5 20
6 25
7 30
8
9 35
'))
```

An important example highlighting the limitations of Pandas DataFrames in displaying masked arrays within index columns is shown below. 

In this example, we convert a keyed table with one key column containing null values to a pandas DataFrame. As expected, the null mask is appropriately applied during the conversion.

```python
>>> keytab = kx.q.xkey('x',
...     kx.Table(data = {
...         'x': kx.q('1 2 0N'),
...         'x1': kx.q('1 0N 2'),
...         'x3': [1, 2, 3]})
...     )
>>> keytab
pykx.KeyedTable(pykx.q('
x| x1 x3
-| -----
1| 1  1 
2|    2 
 | 2  3 
'))
>>> keytab.pd()
    x1  x3
x         
 1   1   1
 2  --   2
--   2   3
```

However, when displaying with multi-index columns, the mask behaviour is not adhered to:

```python
>>> keytab = kx.q.xkey(['x', 'x1'],
...     kx.Table(data = {
...         'x': kx.q('1 2 0N'),
...         'x1': kx.q('1 0N 2'),
...         'x3': [1, 2, 3]})
...     )
>>> keytab
pykx.KeyedTable(pykx.q('
x x1| x3
----| --
1 1 | 1 
2   | 2 
  2 | 3 
'))
>>> keytab.pd()
                                           x3
x                    x1                      
 1                    1                     1
 2                   -9223372036854775808   2
-9223372036854775808  2                     3
```

To illustrate this as a limitation of Pandas rather than PyKX consider the following:

```python
>>> tab = kx.Table(data = {
...         'x': kx.q('1 2 0N'),
...         'x1': kx.q('1 0N 2'),
...         'x3': [1, 2, 3]})
>>> tab
pykx.Table(pykx.q('
x x1 x3
-------
1 1  1 
2    2 
  2  3 
'))
>>> df = tab.pd()
>>> df
   x  x1  x3
0  1   1   1
1  2  --   2
2 --   2   3
>>> df.set_index(['x'])
    x1  x3
x         
 1   1   1
 2  --   2
--   2   3
>>> df.set_index(['x', 'x1'])
                                           x3
x                    x1                      
 1                    1                     1
 2                   -9223372036854775808   2
-9223372036854775808  2                     3
```

Additional to the above inconsistency with Pandas you may also run into issues with the visual representations of masked arrays when displayed in Pandas DataFrames containing large numbers of rows. For example, consider the following case:

```python
>>> t = kx.q('([] time:.z.p;a:til 1000;b:9,999#0N)')
>>> t.pd()
                                 time    a                    b
0   2023-06-12 01:25:48.178532806    0                    9
1   2023-06-12 01:25:48.178532806    1 -9223372036854775808
2   2023-06-12 01:25:48.178532806    2 -9223372036854775808
3   2023-06-12 01:25:48.178532806    3 -9223372036854775808
4   2023-06-12 01:25:48.178532806    4 -9223372036854775808
..                            ...  ...                  ...
995 2023-06-12 01:25:48.178532806  995 -9223372036854775808
996 2023-06-12 01:25:48.178532806  996 -9223372036854775808
997 2023-06-12 01:25:48.178532806  997 -9223372036854775808
998 2023-06-12 01:25:48.178532806  998 -9223372036854775808
999 2023-06-12 01:25:48.178532806  999 -9223372036854775808
 
[1000 rows x 3 columns]    
```

While `#!python -9223372036854778080` represents an underlying PyKX Null value, for display purposes it's visually distracting. To display the DataFrame with the masked values, set its `#!python display.max_rows` to be longer than the length of the specified table. Notice the result below:

```python
>>> import pandas as pd
>>> t = kx.q('([] time:.z.p;a:til 1000;b:9,999#0N)')
>>> pd.set_option('display.max_rows', 1000)
>>> t.pd
                          time    a  b
0   2023-11-26 22:16:05.885992    0  9
1   2023-11-26 22:16:05.885992    1 --
2   2023-11-26 22:16:05.885992    2 --
3   2023-11-26 22:16:05.885992    3 --
4   2023-11-26 22:16:05.885992    4 --
5   2023-11-26 22:16:05.885992    5 --
..
```

!!! info "For more information on masked NumPy arrays and interactions with null representation data in Pandas, check out the following links:"

    - [NumPy masked arrays](https://numpy.org/doc/stable/reference/maskedarray.generic.html#filling-in-the-missing-data)
    - [Pandas working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
    - [Pandas nullable integer data types](https://pandas.pydata.org/docs/user_guide/integer_na.html#integer-na)

### Infinite Conversions

See also the page with specifics on [temporal](./temporal.md) conversions to explain further some of the difficulties around infinities while converting.

!!! note "Note"

    PyKX infinite conversion behaviour changed in version 3.0.0. The below tables outline the before and after conversions.

    #### Positive Infinity conversions

    === ".py()"

        | datatype  | q value | 2.* atom conversion                                  | 2.* vector conversion                               | 3.* atom conversion                                  | 3.* vector conversion                                |
        |-----------|---------|------------------------------------------------------|-----------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
        | short     | `0Wh`   | `q('0Wh')`                                           | `q('0Wh')`                                          | `float('inf')`                                       | `float('inf')`                                       |
        | int       | `0Wi`   | `q('0Wi')`                                           | `q('0Wi')`                                          | `float('inf')`                                       | `float('inf')`                                       |
        | long      | `0Wj`   | `q('0W')`                                            | `q('0W')`                                           | `float('inf')`                                       | `float('inf')`                                       |
        | real      | `0We`   | `float('inf')`                                       | `float('inf')`                                      |                                                      |                                                      |
        | float     | `0w`    | `float('inf')`                                       | `float('inf')`                                      |                                                      |                                                      |
        | timestamp | `0Wp`   | `datetime.datetime(2262, 4, 11, 23, 47, 16, 854775)` | `datetime.datetime(1707, 9, 22, 0, 12, 43, 145224)` |                                                      |                                                      |
        | month     | `0Wm`   | `2147484007`                                         | `2147484007`                                        |                                                      |                                                      |
        | date      | `0Wd`   | `2147494604`                                         | `2147494604`                                        |                                                      |                                                      |
        | timespan  | `0Wn`   | `datetime.timedelta(106751, 16, 854775, 0, 47, 23)`  | `datetime.timedelta(106751, 16, 854775, 0, 47, 23)` |                                                      |                                                      |
        | minute    | `0Wu`   | `datetime.timedelta(-3220, 4, 33138, 0, 5, 5)`       | `datetime.timedelta(-3220, 4, 33138, 0, 5, 5)`      |                                                      |                                                      |
        | second    | `0Wv`   | `datetime.timedelta(24855, 7, 0, 0, 14, 3)`          | `datetime.timedelta(24855, 7, 0, 0, 14, 3)`         |                                                      |                                                      |
        | time      | `0Wt`   | `datetime.timedelta(24, 23, 647000, 0, 31, 20)`      | `datetime.timedelta(24, 23, 647000, 0, 31, 20)`     |                                                      |                                                      |

    === ".np()"

        | datatype  |       | 2.* atom conversion                              | 2.* vector conversion                            | 3.* atom conversion             | 3.* vector conversion                            |
        |-----------|-------|--------------------------------------------------|--------------------------------------------------|---------------------------------|--------------------------------------------------|
        | short     | `0Wh` | **1                                              | `np.int16(32767)`                                | `np.int16(32767)`               |                                                  |
        | int       | `0Wi` | **1                                              | `np.int32(2147483647)`                           | `np.int32(2147483647)`          |                                                  |
        | long      | `0Wj` | **1                                              | `np.int64(9223372036854775807)`                  | `np.int64(9223372036854775807)` |                                                  |
        | real      | `0We` | `np.float32('inf')`                              | `np.float32('inf')`                              |                                 |                                                  |
        | float     | `0w`  | `np.float64('inf')`                              | `np.float64('inf')`                              |                                 |                                                  |
        | timestamp | `0Wp` | `np.datetime64('2262-04-11T23:47:16.854775807')` | `np.datetime64('1707-09-22T00:12:43.145224191')` |                                 |                                                  |
        | month     | `0Wm` | `np.datetime64('178958970-08')`                  | `np.datetime64('-178954971-04')`                 |                                 |                                                  |
        | date      | `0Wd` | `np.datetime64('5881610-07-11')`                 | `np.datetime64('-5877611-06-21')`                |                                 |                                                  |
        | timespan  | `0Wn` | `np.timedelta64(9223372036854775807, 'ns')`      | `np.timedelta64(9223372036854775807, 'ns')`      |                                 |                                                  |
        | minute    | `0Wu` | `np.timedelta64(2147483647, 'm')`                | `np.timedelta64(2147483647, 'm')`                |                                 |                                                  |
        | second    | `0Wv` | `np.timedelta64(2147483647, 's')`                | `np.timedelta64(2147483647, 's')`                |                                 |                                                  |
        | time      | `0Wt` | `np.timedelta64(2147483647, 'ms')`               | `np.timedelta64(2147483647, 'ms')`               |                                 |                                                  |

        - **1 Errors: `NumPy does not support infinite atomic integral values`

    === "pd()"

        | datatype  |       | 2.* atom conversion                             | 2.* vector conversion                           | 3.* atom conversion             | 3.* vector conversion                           |
        |-----------|-------|-------------------------------------------------|-------------------------------------------------|---------------------------------|-------------------------------------------------|
        | short     | `0Wh` | **1                                             | `np.int16(32767)`                               | `np.int16(32767)`               |                                                 |
        | int       | `0Wi` | **1                                             | `np.int32(2147483647)`                          | `np.int32(2147483647)`          |                                                 |
        | long      | `0Wj` | **1                                             | `np.int64(9223372036854775807)`                 | `np.int64(9223372036854775807)` |                                                 |
        | real      | `0We` | `np.float32('inf')`                             | `np.float32('inf')`                             |                                 |                                                 |
        | float     | `0w`  | `np.float64('inf')`                             | `np.float64('inf')`                             |                                 |                                                 |
        | timestamp | `0Wp` | `pd.Timestamp('2262-04-11T23:47:16.854775807')` | `pd.Timestamp('1707-09-22T00:12:43.145224191')` |                                 |                                                 |
        | month     | `0Wm` | **2 `Timestamp('178958970-08-01 00:00:00')`     | **2 `Timestamp('178958970-08-01 00:00:00')`     |                                 |                                                 |
        | date      | `0Wd` | **2 `Timestamp('5881610-07-11 00:00:00')`       | **2 `Timestamp('5881610-07-11 00:00:00')`       |                                 |                                                 |
        | timespan  | `0Wn` | `pd.Timedelta(9223372036854775807, 'ns')`       | `pd.Timedelta(9223372036854775807, 'ns')`       |                                 |                                                 |
        | minute    | `0Wu` | **2 `Timedelta('1491308 days 02:07:00')`        | **2 `Timedelta('1491308 days 02:07:00')`        |                                 |                                                 |
        | second    | `0Wv` | `pd.Timedelta(2147483647, 's')`                 | `pd.Timedelta(2147483647, 's')`                 |                                 |                                                 |
        | time      | `0Wt` | `pd.Timedelta(2147483647, 'ms')`                | `pd.Timedelta(2147483647, 'ms')`                |                                 |                                                 |

        - **1 Errors: `NumPy does not support infinite atomic integral values
        - **2 Errors: `Values out of range` Pandas constructors block creation of these values

    === ".pa()"

        | datatype  |       | 2.* atom conversion                             | 2.* vector conversion                                                                       | 3.* atom conversion             | 3.* vector conversion                                        |
        |-----------|-------|-------------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------|--------------------------------------------------------------|
        | short     | `0Wh` | **1                                             | `<pyarrow.Int16Scalar: 32767>`                                                              | `np.int16(32767)`               |                                                              |
        | int       | `0Wi` | **1                                             | `<pyarrow.Int32Scalar: 2147483647>`                                                         | `np.int32(2147483647)`          |                                                              |
        | long      | `0Wj` | **1                                             | `<pyarrow.Int64Scalar: 9223372036854775807>`                                                | `np.int64(9223372036854775807)` |                                                              |
        | real      | `0We` | `np.float32('inf')`                             | `<pyarrow.FloatScalar: inf>`                                                                |                                 |                                                              |
        | float     | `0w`  | `np.float64('inf')`                             | `<pyarrow.DoubleScalar: inf>`                                                               |                                 |                                                              |
        | timestamp | `0Wp` | `pd.Timestamp('2262-04-11T23:47:16.854775807')` | `<pyarrow.TimestampScalar: '1707-09-22T00:12:43.145224191'>`                                |                                 |                                                              |
        | month     | `0Wm` | **2 `Timestamp('178958970-08-01 00:00:00')`     | **3                                                                                         |                                 |                                                              |
        | date      | `0Wd` | **2 `Timestamp('5881610-07-11 00:00:00')`       | **4                                                                                         |                                 |                                                              |
        | timespan  | `0Wn` | `pd.Timedelta(9223372036854775807, 'ns')`       | `<pyarrow.DurationScalar: Timedelta('106751 days 23:47:16.854775807')>`                     |                                 |                                                              |
        | minute    | `0Wu` | **2 `Timedelta('1491308 days 02:07:00')`        | **5                                                                                         |                                 |                                                              |
        | second    | `0Wv` | `pd.Timedelta(2147483647, 's')`                 | `<pyarrow.DurationScalar: datetime.timedelta(days=24855, seconds=11647)>`                   |                                 |                                                              |
        | time      | `0Wt` | `pd.Timedelta(2147483647, 'ms')`                | `<pyarrow.DurationScalar: datetime.timedelta(days=24, seconds=73883, microseconds=647000)>` |                                 |                                                              |

        - **1 Errors: `NumPy does not support infinite atomic integral values`
        - **2 Errors: `Values out of range - Pandas constructors block them`
        - **3 Errors: `pyarrow.lib.ArrowNotImplementedError: Unsupported datetime64 time unit`
        - **4 Errors: `OverflowError: days=-2147472692; must have magnitude <= 999999999`
        - **5 Errors: `pyarrow.lib.ArrowNotImplementedError: Unsupported timedelta64 time unit`

    #### Negative Infinity conversions

    === ".py()"

        | datatype  |        | 2.* atom conversion                                  | 2.* vector conversion                               | 3.* atom conversion                                 | 3.* vector conversion |
        |-----------|--------|------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------|
        | short     | `-0Wh` | `q('-0Wh')`                                          | `q('-0Wh')`                                         | `float('-inf')`                                     | `float('-inf')`       |
        | int       | `-0Wi` | `q('-0Wi')`                                          | `q('-0Wi')`                                         | `float('-inf')`                                     | `float('-inf')`       |
        | long      | `-0Wj` | `q('-0W')`                                           | `q('-0W')`                                          | `float('-inf')`                                     | `float('-inf')`       |
        | real      | `-0We` | `float('-inf')`                                      | `float('-inf')`                                     |                                                     |                       |
        | float     | `-0w`  | `float('-inf')`                                      | `float('-inf')`                                     |                                                     |                       |
        | timestamp | `-0Wp` | `datetime.datetime(2262, 4, 11, 23, 47, 16, 854774)` | `datetime.datetime(1707, 9, 22, 0, 12, 43, 145224)` | `datetime.datetime(1707, 9, 22, 0, 12, 43, 145224)` |                       |
        | month     | `-0Wm` | `-2147483287`                                        | `-2147483287`                                       |                                                     |                       |
        | date      | `-0Wd` | `-2147472690`                                        | `-2147472690`                                       |                                                     |                       |
        | timespan  | `-0Wn` | `datetime.timedelta(-106752, 43, 145224, 0, 12)`     | `datetime.timedelta(-106752, 43, 145224, 0, 12)`    |                                                     |                       |
        | minute    | `-0Wu` | `datetime.timedelta(3219, 55, 966861, 0, 54, 18)`    | `datetime.timedelta(3219, 55, 966861, 0, 54, 18)`   |                                                     |                       |
        | second    | `-0Wv` | `datetime.timedelta(-24856, 53, 0, 0, 45, 20)`       | `datetime.timedelta(-24856, 53, 0, 0, 45, 20)`      |                                                     |                       |
        | time      | `-0Wt` | `datetime.timedelta(-25, 36, 353000, 0, 28, 3)`      | `datetime.timedelta(-25, 36, 353000, 0, 28, 3)`     |                                                     |                       |

    === "np()"

        | datatype  |        | 2.* atom conversion                              | 2.* vector conversion                            | 3.* atom conversion                              | 3.* vector conversion |
        |-----------|--------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|-----------------------|
        | short     | `-0Wh` | **1                                               | `np.int16(-32767)`                              | `np.int16(-32767)`                               |                       |
        | int       | `-0Wi` | **1                                              | `np.int32(-2147483647)`                          | `np.int32(-2147483647)`                          |                       |
        | long      | `-0Wj` | **1                                              | `np.int64(-9223372036854775807)`                 | `np.int64(-9223372036854775807)`                 |                       |
        | real      | `-0We` | `np.float32('-inf')`                             | `np.float32('-inf')`                             |                                                  |                       |
        | float     | `-0w`  | `np.float64('-inf')`                             | `np.float64('-inf')`                             |                                                  |                       |
        | timestamp | `-0Wp` | `np.datetime64('1677-09-21T00:12:43.145224193')` | `np.datetime64('1707-09-22T00:12:43.145224193')` | `np.datetime64('1707-09-22T00:12:43.145224193')` |                       |
        | month     | `-0Wm` | `np.datetime64('-178954971-06')`                 | `np.datetime64('-178954971-06')`                 |                                                  |                       |
        | date      | `-0Wd` | `np.datetime64('-5877611-06-23')`                | `np.datetime64('-5877611-06-23')`                |                                                  |                       |
        | timespan  | `-0Wn` | `np.timedelta64(-9223372036854775807, 'ns')`     | `np.timedelta64(-9223372036854775807, 'ns')`     |                                                  |                       |
        | minute    | `-0Wu` | `np.timedelta64(-2147483647, 'm')`               | `np.timedelta64(-2147483647, 'm')`               |                                                  |                       |
        | second    | `-0Wv` | `np.timedelta64(-2147483647, 's')`               | `np.timedelta64(-2147483647, 's')`               |                                                  |                       |
        | time      | `-0Wt` | `np.timedelta64(-2147483647, 'ms')`              | `np.timedelta64(-2147483647, 'ms')`              |                                                  |                       |

        - **1 Errors: `NumPy does not support infinite atomic integral values`

    === ".pd()"

        | datatype  |        | 2.* atom conversion                             | 2.* vector conversion                           | 3.* atom conversion                             | 3.* vector conversion |
        |-----------|--------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-----------------------|
        | short     | `-0Wh` | **1                                             | `np.int16(-32767)`                              | `np.int16(-32767)`                              |                       |
        | int       | `-0Wi` | **1                                             | `np.int32(-2147483647)`                         | `np.int32(-2147483647)`                         |                       |
        | long      | `-0Wj` | **1                                             | `np.int64(-9223372036854775807)`                | `np.int64(-9223372036854775807)`                |                       |
        | real      | `-0We` | `np.float32('-inf')`                            | `np.float32('-inf')`                            |                                                 |                       |
        | float     | `-0w`  | `np.float64('-inf')`                            | `np.float64('-inf')`                            |                                                 |                       |
        | timestamp | `-0Wp` | `pd.Timestamp('1677-09-21T00:12:43.145224193')` | `pd.Timestamp('1707-09-22 00:12:43.145224193')` | `pd.Timestamp('1707-09-22 00:12:43.145224193')` |                       |
        | month     | `-0Wm` | **2 `Timestamp('-178954971-06-01 00:00:00')`    | **2 `Timestamp('-178954971-06-01 00:00:00')`    |                                                 |                       |
        | date      | `-0Wd` | **2 `Timestamp('-5877611-06-23 00:00:00')`      | **2 `Timestamp('-5877611-06-23 00:00:00')`      |                                                 |                       |
        | timespan  | `-0Wn` | `pd.Timedelta(-9223372036854775807, 'ns')`      | `pd.Timedelta(-9223372036854775807, 'ns')`      |                                                 |                       |
        | minute    | `-0Wu` | **2 `Timedelta('-1491309 days +21:53:00')`      | **2 `Timedelta('-1491309 days +21:53:00')`      |                                                 |                       |
        | second    | `-0Wv` | `pd.Timedelta(-2147483647, 's')`                | `pd.Timedelta(-2147483647, 's')`                |                                                 |                       |
        | time      | `-0Wt` | `pd.Timedelta(-2147483647, 'ms')`               | `pd.Timedelta(-2147483647, 'ms')`               |                                                 |                       |

        - **1 Errors: `NumPy does not support infinite atomic integral values`
        - **2 Errors: `Values out of range` Pandas constructors block creation of these values

    === ".pa()"

        | datatype  |        | 2.* atom conversion                             | 2.* vector conversion                                                                        | 3.* atom conversion                             | 3.* vector conversion |
        |-----------|--------|-------------------------------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------|-----------------------|
        | short     | `-0Wh` | **1                                             | `<pyarrow.Int16Scalar: -32767>`                                                              | `np.int16(-32767)`                              |                       |
        | int       | `-0Wi` | **1                                             | `<pyarrow.Int32Scalar: -2147483647>`                                                         | `np.int32(-2147483647)`                         |                       |
        | long      | `-0Wj` | **1                                             | `<pyarrow.Int64Scalar: -9223372036854775807>`                                                | `np.int64(-9223372036854775807)`                |                       |
        | real      | `-0We` | `np.float32('-inf')`                            | `<pyarrow.FloatScalar: -inf>`                                                                |                                                 |                       |
        | float     | `-0w`  | `np.float64('-inf')`                            | `<pyarrow.DoubleScalar: -inf>`                                                               |                                                 |                       |
        | timestamp | `-0Wp` | `pd.Timestamp('1677-09-21T00:12:43.145224193')` | `<pyarrow.TimestampScalar: '1707-09-22 00:12:43.145224193'>`                                 | `pd.Timestamp('1707-09-22 00:12:43.145224193')` |                       |
        | month     | `-0Wm` | **2 `Timestamp('-178954971-06-01 00:00:00')`    |  **3                                                                                         |                                                 |                       |
        | date      | `-0Wd` | **2 `Timestamp('-5877611-06-23 00:00:00')`      |  **4                                                                                         |                                                 |                       |
        | timespan  | `-0Wn` | `pd.Timedelta(-9223372036854775807, 'ns')`      | `<pyarrow.DurationScalar: Timedelta('-106752 days +00:12:43.145224193')>`                    |                                                 |                       |
        | minute    | `-0Wu` | **2 `Timedelta('-1491309 days +21:53:00')`      | **5                                                                                          |                                                 |                       |
        | second    | `-0Wv` | `pd.Timedelta(-2147483647, 's')`                | `<pyarrow.DurationScalar: datetime.timedelta(days=-24856, seconds=74753)>`                   |                                                 |                       |
        | time      | `-0Wt` | `pd.Timedelta(-2147483647, 'ms')`               | `<pyarrow.DurationScalar: datetime.timedelta(days=-25, seconds=12516, microseconds=353000)>` |                                                 |                       |

        - **1 Errors: `NumPy does not support infinite atomic integral values`
        - **2 Errors: `Values out of range - Pandas constructors block them`
        - **3 Errors: `pyarrow.lib.ArrowNotImplementedError: Unsupported datetime64 time unit`
        - **4 Errors: `OverflowError: days=-2147472690; must have magnitude <= 999999999`
        - **5 Errors: `pyarrow.lib.ArrowNotImplementedError: Unsupported timedelta64 time unit`

#### Infinite weirdness

Other than real/float infinities, which follow the IEEE standard for infinities and so are ignored in this section, infinite values in kdb+ do not behave how you would expect them to. PyKX opts to expose their behavior as-is, since the alternatives (error for infinities, or always expose them as their underlying values) are undesirable. For this reason you should take care when using them.

Arithmetic operations on infinities are applied directly to the underlying values. As such, adding 1 to many positive infinities in q will result in the null for that type, as the value overflows and becomes the smallest value in that type's range. Subtracting 1 from positive infinities merely yields the second largest number for that type. For instance, `#!python 2147483646 == q('0Wi') - 1`.

## Python to q

Wherever possible, the conversions from Q to Python are symmetric. Therefore, you can apply in reverse most of the conversions described in the previous section. For instance, if you convert a NumPy masked array with dtype `#!python np.int32` to q, the masked values will be represented by int null (`0Ni`) in q.

## Performance

By default, whenever PyKX converts a q vector to a Python representation (e.g. a NumPy array) it checks where the nulls (if any) are located. This requires operating on every element of the array, which can be rather expensive. 

If you know ahead of time that your q vector/table has no nulls in it, you can provide the keyword argument `#!python has_nulls=False` to `#!python .py`/`#!python .np`/`#!python .pd`/`#!python .pa`. This will skip the null-check. If you set this keyword argument to false, but there are still nulls in the data, they will come through as the underlying values from q, for example, `#!python -32768` for a short integer.

By default `#!python has_nulls` is `#!python None`. You can set it to `#!python True` to always handle the data as if it contains nulls, regardless of whether it actually does. This can improve consistency in some cases, for instance by having all int vectors be converted to NumPy masked arrays instead of normal NumPy arrays when there are no nulls, and masked arrays when there are nulls.

!!! tip "Tip: you can also use the keyword argument `#!python raw=True` for the `#!python py`/`#!python np`/`#!python pd`/`#!python pa` methods for improved performance - albeit this affects more than just how nulls are handled. See [the performance doc page](../advanced/performance.md) for more details about raw conversions."

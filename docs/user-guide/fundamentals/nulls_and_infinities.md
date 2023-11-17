# Handling nulls and infinities

PyKX and its management of nulls and infinities inherited from q operate differently in subtle ways from familiar libraries such as Numpy.

PyKX provides typed null and infinity values for most types. [As shown in the q docs](https://code.kx.com/q/ref/#datatypes), nulls can be expressed as `0N` followed by a type character (or no type character for long integer null), while infinities can be expressed as `0W` followed by a type character (or no type character for a long integer infinity).

Datatypes in q designate a particular value in their numeric range as null, and another two as positive and negative infinity. Most other languages, such as Python, have no way to represent infinity for anything other than IEEE floating point numbers, and where typed nulls exist, they will not also be a value in the range of the datatype (save for floats, which can be `NaN`).

For example, the q null short integer `0Nh` is stored as the value `-32768` (i.e. the smallest possible signed 16 bit integer), and the q infinite short integer is stored as the value `32767` (i.e the largest possible signed 16 bit integer).

Due to the design of nulls and infinites in q, there are some technical considerations - detailed on this page - regarding converting nulls and infinities between Python and q in either direction.

## Generation of null and infinite values

To facilitate the generation of null and infinite values there are a number of properties for `pykx.Atom` objects which allow this to be completed Pythonically. In all cases this requires access to [licensed mode](../advanced/modes.md). The following examples show the generation of various null and infinite values.

### Null generation

Where possible null values can be returned to you as follows:

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

Unsupported values will return a `NotImplemetedError` as follows:

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

Where possible positive and negative infinite values can be returned to you as follows:

```python
>>> import pykx as kx
>>> kx.TimeAtom.inf
pykx.TimeAtom(pykx.q('0Wt'))
>>> -kx.TimeAtom.inf
pykx.TimeAtom(pykx.q('-0Wt'))
>>> kx.IntAtom.inf
pykx.IntAtom(pykx.q('0Wi'))
>>> -kx.IntAtom.inf
pykx.IntAtom(pykx.q('-0Wi'))
```

Unsupported values will return a `NotImplementedError` as follows:

```python
>>> kx.SymbolAtom.inf
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/wrappers.py", line 274, in inf
    raise NotImplementedError('Retrieval of infinite values not supported for this type')
NotImplementedError: Retrieval of infinite values not supported for this type
```

## Checking for nulls and infinities

[The q function named null](https://code.kx.com/q/ref/null/) can be applied to most PyKX objects, and will return if the object is null by returning `1b`, or if it contains nulls by returning a collection of booleans whose shape matches the object. Like with any function from the `.q` namespace, it can be accessed via the [context interface](../../api/pykx-execution/ctx.md): [`q.null`](../../api/pykx-execution/q.md#null)).

```python
>>> import pykx as kx
>>> kx.q.null(kx.q('0n'))
pykx.BooleanAtom(pykx.q('1b'))
>>> kx.q.null(('1 2 0n 3f'))
pykx.BooleanVector(pykx.q('0010b'))
```

[`pykx.Atom`][pykx.Atom] objects provide the properties `is_null` and `is_inf`. These are `True` if the atom is a null value for its type, or an infinite value (positive or negative) for its type, respectively, and `False` otherwise. `is_inf` is always `False` for types which do not have an infinite value in q, such as symbols.

```python
>>> kx.q('0w').is_inf
True
>>> kx.q('0W').is_inf
True
>>> kx.q('1f').is_inf
False
>>> kx.q('0n').is_null
True
>>> kx.q('0N').is_null
True
>>> kx.q('1f').is_null
False
```

Likewise, all [`pykx.Collection`][pykx.Collection] objects provide the properties `has_nulls` and `has_infs`. They are `True` if the collection has any nulls/infinities in it.

```python
>>> kx.q('0w,9?1f').has_infs
True
>>> kx.q('0w,9?1f').has_nulls
False
```

Some null values are unintuitive. For instance, the null value for a character in q is the space `" "`, the null value for a symbol is the empty symbol, and the null value for a GUID is `00000000-0000-0000-0000-000000000000`. A char vector (i.e. q string) that has any spaces in it will have `has_nulls` set to `True`.

## q to Python

Vectors with the q types `short`, `int`, and `long` can be converted to Python in the following ways:

- `.py` provides a list with the null values left as `pykx.K` objects - thin wrappers around the objects in q's memory.
- `.np` provides a [masked array](https://numpy.org/doc/stable/reference/maskedarray.html) with the null values masked out, and the fill value set to the underlying value of the q null.
- `.pd` provides a [Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) as per usual, but backed by an [IntegerArray](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.arrays.IntegerArray.html) instead of a regular `np.ndarray`.
- `.pa` provides a [PyArrow Array](https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array) as per usual, which natively supports nullable integral vector data, so it simply has the indexes of the nulls stored in the array metadata.

Real vectors use the standard `NaN` and `inf` values, and so are handled by q, Python, Numpy, Pandas, and PyArrow in the same way with no special handling.

Temporal vectors use `NaT` to represent null values in Numpy and Pandas, `None` to represent them in pure Python, and PyArrow represents null temporal values like it does for any other data type: by masking it out using the array metadata.

When converting a table from q to Python with one of the methods above, each column will be transformed as an independent vector as described above.

The following provides an example of the masked array behavior outlined in the `.np` method described above which is additionally exhibited by the `.pd` method.

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

For more information on masked Numpy arrays and interactions with null representation data in Pandas see the following links

- [Numpy masked arrays](https://numpy.org/doc/stable/reference/maskedarray.generic.html#filling-in-the-missing-data)
- [Pandas working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Pandas nullable integer data types](https://pandas.pydata.org/docs/user_guide/integer_na.html#integer-na)


## Python to q

Wherever practical the conversions from q to Python are symmetric, so most of the conversions detailed in the section above work in reverse too. For instance, if you convert a Numpy masked array with dtype `np.int32` to q, the masked values will be represented by int null (`0Ni`) in q.


## Performance

By default, whenever PyKX converts a q vector to some Python representation (e.g. a Numpy array) it checks where the nulls (if any) are located. This requires operating on every element of the array, which can be rather expensive. If you know ahead of time that your q vector/table has no nulls in it, you can provide the keyword argument `has_nulls=False` to `.py`/`.np`/`.pd`/`.pa`. This will skip the null-check. If you set this keyword argument to false, but there are still nulls in the data, they will come through as the underlying values from q, e.g. `-32768` for a short integer.

By default `has_nulls` is `None`. It can be set to `True` to always handle the data as if it contains nulls, regardless of whether it actually does. This can improve consistency in some cases, for instance by having all int vectors be converted to Numpy masked arrays instead of normal Numpy arrays when there are no nulls, and masked arrays when there are nulls.

You can also use the keyword argument `raw=True` for the `py`/`np`/`pd`/`pa` methods for improved performance - albeit this affects more than just how nulls are handled. See [the performance doc page](../advanced/performance.md) for more details about raw conversions.

## Infinite Weirdness

Other than real/float infinities, which follow the IEEE standard for infinities and so are ignored in this section, infinite values in kdb+ do not behave how you would expect them to. PyKX opts to expose their behavior as-is, since the alternatives (error for infinities, or always expose them as their underlying values) are undesirable. For this reason you should take care when using them.

Arithmetic operations on infinities are applied directly to the underlying values. As such, adding 1 to many positive infinities in q will result in the null for that type, as the value overflows and becomes the smallest value in that type's range. Subtracting 1 from positive infinities merely yields the second largest number for that type. For instance, `2147483646 == q('0Wi') - 1`.

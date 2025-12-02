# PyKX to Pythonic data type mapping

A breakdown of each of the `pykx.K` types and their analogous `Python`, `NumPy`, `Pandas`, and `PyArrow` types.

??? "Cheat Sheet: `Python`, `NumPy`, `PyArrow`"

	| PyKX type                       | Python type | Numpy dtype     | PyArrow type   |
	| ------------------------------- | ----------- | --------------- | -------------- |
	| [List](#pykxlist)               | list        | object          | Not Supported  |
	| [Boolean](#pykxbooleanatom)     | bool        | bool            | Not Supported  |
	| [GUID](#pykxguidatom)           | uuid4       | uuid4           | uuid4          |
	| [Byte](#pykxbyteatom)           | int         | uint8           | uint8          |
	| [Short](#pykxshortatom)         | int         | int16           | int16          |
	| [Int](#pykxintatom)             | int         | int32           | int32          |
	| [Long](#pykxlongatom)           | int         | int64           | int64          |
	| [Real](#pykxrealatom)           | float       | float32         | FloatArray     |
	| [Float](#pykxfloatatom)         | float       | float64         | DoubleArray    |
	| [Char](#pykxcharatom)           | bytes       | \\\|S1          | BinaryArray    |
	| [Symbol](#pykxsymbolatom)       | str         | object          | StringArray    |
	| [Timestamp](#pykxtimestampatom) | datetime    | datetime64[ns]  | TimestampArray |
	| [Month](#pykxmonthatom)         | date        | datetime64[M]   | Not Supported  |
	| [Date](#pykxdateatom)           | date        | datetime64[D]   | Date32Array    |
	| [Timespan](#pykxtimespanatom)   | timedelta   | timedelta64[ns] | DurationArray  |
	| [Minute](#pykxminuteatom)       | timedelta   | timedelta64[m]  | Not Supported  |
	| [Second](#pykxsecondatom)       | timedelta   | timedelta64[s]  | DurationArray  |
	| [Time](#pykxtimeatom)           | timedelta   | timedelta64[ms] | DurationArray  |
	| [Dictionary](#pykxdictionary)   | dict        | Not Supported   | Not Supported  |
	| [Table](#pykxtable)             | dict        | records         | Table          |

??? "Cheat Sheet: `Pandas 1.*`, `Pandas 2.*`, `Pandas 2.* PyArrow backed`"

	**Note:** Creating PyArrow backed Pandas objects uses `as_arrow=True` using NumPy arrays as an intermediate data format.

	| PyKX type                       | Pandas 1.\* dtype | Pandas 2.\* dtype | Pandas 2.\* as_arrow=True dtype |
	| ------------------------------- | ----------------- | ----------------- | ------------------------------- |
	| [List](#pykxlist)               | object            | object            | object                          |
	| [Boolean](#pykxbooleanatom)     | bool              | bool              | bool[pyarrow]                   |
	| [GUID](#pykxguidatom)           | object            | object            | object                          |
	| [Byte](#pykxbyteatom)           | uint8             | uint8             | uint8[pyarrow]                  |
	| [Short](#pykxshortatom)         | int16             | int16             | int16[pyarrow]                  |
	| [Int](#pykxintatom)             | int32             | int32             | int32[pyarrow]                  |
	| [Long](#pykxlongatom)           | int64             | int64             | int64[pyarrow]                  |
	| [Real](#pykxrealatom)           | float32           | float32           | float[pyarrow]                  |
	| [Float](#pykxfloatatom)         | float64           | float64           | double[pyarrow]                 |
	| [Char](#pykxcharatom)           | bytes8            | bytes8            | fixed_size_binary[1][pyarrow]   |
	| [Symbol](#pykxsymbolatom)       | object            | object            | string[pyarrow]                 |
	| [Timestamp](#pykxtimestampatom) | datetime64[ns]    | datetime64[ns]    | timestamp[ns][pyarrow]          |
	| [Month](#pykxmonthatom)         | datetime64[ns]    | datetime64[s]     | timestamp[s][pyarrow]           |
	| [Date](#pykxdateatom)           | datetime64[ns]    | datetime64[s]     | timestamp[s][pyarrow]           |
	| [Timespan](#pykxtimespanatom)   | timedelta64[ns]   | timedelta64[ns]   | duration[ns][pyarrow]           |
	| [Minute](#pykxminuteatom)       | timedelta64[ns]   | timedelta64[s]    | duration[s][pyarrow]            |
	| [Second](#pykxsecondatom)       | timedelta64[ns]   | timedelta64[s]    | duration[s][pyarrow]            |
	| [Time](#pykxtimeatom)           | timedelta64[ns]   | timedelta64[ms]   | duration[ms][pyarrow]           |
	| [Dictionary](#pykxdictionary)   | Not Supported     | Not Supported     | Not Supported                   |
	| [Table](#pykxtable)             | DataFrame         | DataFrame         | DataFrame                       |

## `pykx.List`

=== "Python"
	A python list of mixed types will be converted into a `pykx.List`.

	```Python
	>>> pykx.List([1, b'foo', 'bar', 4.5])
	pykx.List(pykx.q('
	1
	"foo"
	`bar
	4.5
	'))
	```

	Calling `.py()` on a `pykx.List` will return a generic python list object where each object is converted into its analogous python type.

	```Python
	>>> pykx.List([1, b'foo', 'bar', 4.5]).py()
	[1, b'foo', 'bar', 4.5]
	```

=== "Numpy"
	A numpy list with `dtype==object` containing data of mixed types will be converted into a `pykx.List`

	```Python
	>>> pykx.List(np.array([1, b'foo', 'bar', 4.5], dtype=object))
	pykx.List(pykx.q('
	1
	"foo"
	`bar
	4.5
	'))
	```

	Calling `.np()` on a `pykx.List` object will return a numpy `ndarray` with `dtype==object` where each element has been converted into its closest analogous python type.

	```Python
	>>> pykx.List([1, b'foo', 'bar', 4.5]).np()
	array([1, b'foo', 'bar', 4.5], dtype=object)
	```

=== "Pandas"
	Calling `.pd()` on a `pykx.List` object will return a pandas `Series` with `dtype==object` where each element has been converted into its closest analogous python type.

	```Python
	>>> pykx.List([1, b'foo', 'bar', 4.5]).pd()
	0         1
	1    b'foo'
	2       bar
	3       4.5
	dtype: object
	```

## `pykx.BooleanAtom`

**Python**

The python bool type will be converted into a `pykx.BooleanAtom`.

```Python
>>> pykx.BooleanAtom(True)
pykx.BooleanAtom(pykx.q('1b'))
```

Calling `.py()` on a `pykx.BooleanAtom` will return a python bool object.

```Python
>>> pykx.BooleanAtom(True).py()
True
```

## `pykx.BooleanVector`

=== "Python"
	A list of python bool types will be converted into a `pykx.BooleanVector`.

	```Python
	>>> pykx.BooleanVector([True, False, True])
	pykx.BooleanVector(pykx.q('101b'))
	```

	Calling `.py()` on a `pykx.BooleanVector` will return a list of python bool objects.

	```Python
	>>> pykx.BooleanVector([True, False, True]).py()
	[True, False, True]
	```

=== "Numpy, Pandas, PyArrow"
	Converting a `pykx.BoolVector` will result in an array of objects with the `bool` `dtype`, arrays of that `dtype` can also be converted into `pykx.BoolVector` objects.

## `pykx.GUIDAtom`

=== "Python"
	The python uuid4 type from the `uuid` library will be converted into a `pykx.GUIDAtom`.

	```Python
	>>> from uuid import uuid4
	>>> pykx.GUIDAtom(uuid4())
	pykx.GUIDAtom(pykx.q('012e8fb7-52c4-49e6-9b4e-93aa625ca3d7'))
	```

	Calling `.py()` on a `pykx.GUIDAtom` will return a python uuid4 object.

	```Python
	>>> pykx.GUIDAtom(uuid4()).py()
	UUID('d16f9f3f-2a57-4dfd-818e-04c9c7a53584')
	```

## `pykx.GUIDVector`

=== "Python"
	A list of python uuid4 types from the `uuid` library will be converted into a `pykx.GUIDVector`.

	```Python
	>>> pykx.GUIDVector([uuid4(), uuid4()])
	pykx.GUIDVector(pykx.q('542ccbef-8aa1-4433-804a-7928172ec2d4 ff6f89fb-1aec-4073-821a-ce281ca6263e'))
	```

	Calling `.py()` on a `pykx.GUIDVector` will return a list of python uuid4 objects.

	```Python
	>>> pykx.GUIDVector([uuid4(), uuid4()]).py()
	[UUID('a3b284fc-5f31-4ba2-b521-fa8b5c309e02'), UUID('95ee9044-3930-492c-96f2-e336110de023')]
	```

=== "Numpy, Pandas, PyArrow"
	Each of these will return an array of their respective object types around a list of uuid4 objects.

## `pykx.ByteAtom`

=== "Python"
	The python int type will be converted into a `pykx.ByteAtom`.

	Float types will also be converted but the decimal will be truncated away and no rounding done.

	```Python
	>>> pykx.ByteAtom(1.0)
	pykx.ByteAtom(pykx.q('0x01'))
	>>> pykx.ByteAtom(1.5)
	pykx.ByteAtom(pykx.q('0x01'))
	```

	Calling `.py()` on a `pykx.ByteAtom` will return a python int object.

	```Python
	>>> pykx.ByteAtom(1.5).py()
	1
	```

## `pykx.ByteVector`

=== "Python"
	A list of python int types will be converted into a `pykx.ByteVector`.

	Float types will also be converted but the decimal will be truncated away and no rounding done.

	```Python
	>>> pykx.ByteVector([1, 2.5])
	pykx.ByteVector(pykx.q('0x0102'))
	```

	Calling `.py()` on a `pykx.ByteVector` will return a list of python int objects.

	```Python
	>>> pykx.ByteVector([1, 2.5]).py()
	[1, 2]
	```

=== "Numpy, Pandas, PyArrow"
	Converting a `pykx.ByteVector` will result in an array of objects with the `uint8` `dtype`, arrays of that `dtype` can also be converted into `pykx.ByteVector` objects.

## `pykx.ShortAtom`

=== "Python"
	The python int type will be converted into a `pykx.ShortAtom`.

	Float types will also be converted but the decimal will be truncated away and no rounding done.

	```Python
	>>> pykx.ShortAtom(1)
	pykx.ShortAtom(pykx.q('1h'))
	>>> pykx.ShortAtom(1.5)
	pykx.ShortAtom(pykx.q('1h'))
	```

	Calling `.py()` on a `pykx.ShortAtom` will return a python int object.

	```Python
	>>> pykx.ShortAtom(1.5).py()
	1
	```

## `pykx.ShortVector`

=== "Python"
	A list of python int types will be converted into a `pykx.ShortVector`.

	Float types will also be converted but the decimal will be truncated away and no rounding done.

	```Python
	>>> pykx.ShortVector([1, 2.5])
	pykx.ShortVector(pykx.q('1 2h'))
	```

	Calling `.py()` on a `pykx.ShortVector` will return a list of python int objects.

	```Python
	>>> pykx.ShortVector([1, 2.5]).py()
	[1, 2]
	```

=== "Numpy, Pandas, PyArrow"
	Converting a `pykx.ShortVector` will result in an array of objects with the `int16` `dtype`, arrays of that `dtype` can also be converted into `pykx.ShortVector` objects.

## `pykx.IntAtom`

=== "Python"
	The python int type will be converted into a `pykx.IntAtom`.

	Float types will also be converted but the decimal will be truncated away and no rounding done.

	```Python
	>>> pykx.IntAtom(1)
	pykx.IntAtom(pykx.q('1i'))
	>>> pykx.IntAtom(1.5)
	pykx.IntAtom(pykx.q('1i'))
	```

	Calling `.py()` on a `pykx.IntAtom` will return a python int object.

	```Python
	>>> pykx.IntAtom(1.5).py()
	1
	```

## `pykx.IntVector`

=== "Python"
	A list of python int types will be converted into a `pykx.IntVector`.

	Float types will also be converted but the decimal will be truncated away and no rounding done.

	```Python
	>>> pykx.IntVector([1, 2.5])
	pykx.IntVector(pykx.q('1 2i'))
	```

	Calling `.py()` on a `pykx.IntVector` will return a list of python int objects.

	```Python
	>>> pykx.IntVector([1, 2.5]).py()
	[1, 2]
	```

=== "Numpy, Pandas, PyArrow"
	Converting a `pykx.IntVector` will result in an array of objects with the `int32` `dtype`, arrays of that `dtype` can also be converted into `pykx.IntVector` objects.

## `pykx.LongAtom`

=== "Python"
	The python int type will be converted into a `pykx.LongAtom`.

	Float types will also be converted but the decimal will be truncated away and no rounding done.

	```Python
	>>> pykx.LongAtom(1)
	pykx.LongAtom(pykx.q('1'))
	>>> pykx.LongAtom(1.5)
	pykx.LongAtom(pykx.q('1'))
	```

	Calling `.py()` on a `pykx.LongAtom` will return a python int object.

	```Python
	>>> pykx.LongAtom(1.5).py()
	1
	```

## `pykx.LongVector`

=== "Python"
	A list of python int types will be converted into a `pykx.LongVector`.

	Float types will also be converted but the decimal will be truncated away and no rounding done.

	```Python
	>>> pykx.LongVector([1, 2.5])
	pykx.LongVector(pykx.q('1 2'))
	```

	Calling `.py()` on a `pykx.LongVector` will return a list of python int objects.

	```Python
	>>> pykx.LongVector([1, 2.5]).py()
	[1, 2]
	```

=== "Numpy, Pandas, PyArrow"
	Converting a `pykx.LongVector` will result in an array of objects with the `int64` `dtype`, arrays of that `dtype` can also be converted into `pykx.LongVector` objects.

## `pykx.RealAtom`

=== "Python"
	The python float and int types will be converted into a `pykx.RealAtom`.

	```Python
	>>> pykx.RealAtom(2.5)
	pykx.RealAtom(pykx.q('2.5e'))
	```

	Calling `.py()` on a `pykx.RealAtom` will return a python float object.

	```Python
	>>> pykx.RealAtom(2.5).py()
	2.5
	```

## `pykx.RealVector`

=== "Python"
	A list of python int and float types will be converted into a `pykx.RealVector`.

	```Python
	>>> pykx.RealVector([1, 2.5])
	pykx.RealVector(pykx.q('1 2.5e'))
	```

	Calling `.py()` on a `pykx.RealVector` will return a list of python float objects.

	```Python
	>>> pykx.RealVector([1, 2.5]).py()
	[1.0, 2.5]
	```

=== "Numpy, Pandas"
	Converting a `pykx.RealVector` will result in an array of objects with the `float32` `dtype`, arrays of that `dtype` can also be converted into `pykx.RealVector` objects.


=== "PyArrow"
	This will return a `PyArrow` array with the FloatArray type.

## `pykx.FloatAtom`

=== "Python"
	The python float and int types will be converted into a `pykx.FloatAtom`.

	```Python
	>>> pykx.FloatAtom(2.5)
	pykx.FloatAtom(pykx.q('2.5'))
	```

	Calling `.py()` on a `pykx.FloatAtom` will return a python float object.

	```Python
	>>> pykx.FloatAtom(2.5).py()
	2.5
	```

## `pykx.FloatVector`

=== "Python"
	A list of python int and float types will be converted into a `pykx.FloatVector`.

	```Python
	>>> pykx.FloatVector([1, 2.5])
	pykx.FloatVector(pykx.q('1 2.5'))
	```

	Calling `.py()` on a `pykx.FloatVector` will return a list of python float objects.

	```Python
	>>> pykx.FloatVector([1, 2.5]).py()
	[1.0, 2.5]
	```

=== "Numpy, Pandas"
	Converting a `pykx.FloatVector` will result in an array of objects with the `float64` `dtype`, arrays of that `dtype` can also be converted into `pykx.FloatVector` objects.

=== "PyArrow"
	This will return a `PyArrow` array with the DoubleArray type.

## `pykx.CharAtom`

=== "Python"
	The python bytes type with length 1 will be converted into a `pykx.CharAtom`.

	```Python
	>>> pykx.CharAtom(b'a')
	pykx.CharAtom(pykx.q('"a"'))
	```

	Calling `.py()` on a `pykx.CharAtom` will return a python bytes object.

	```Python
	>>> pykx.CharAtom(b'a').py()
	b'a'
	```

## `pykx.CharVector`

=== "Python"
	The python bytes type with length greater than 1 will be converted into a `pykx.CharVector`.

	```Python
	>>> pykx.CharVector(b'abc')
	pykx.CharVector(pykx.q('"abc"'))
	```

	Calling `.py()` on a `pykx.CharVector` will return a python bytes object.

	```Python
	>>> pykx.CharVector(b'abc').py()
	b'abc'
	```

=== "Numpy"
	Calling `.np()` on a `pykx.CharVector` will return a numpy `ndarray` with `dtype` `'|S1'`.

	```Python
	>>> pykx.CharVector(b'abc').np()
	array([b'a', b'b', b'c'], dtype='|S1')
	```

	Converting a `ndarray` of this `dtype` will create a `pykx.CharVector`.

	```Python
	>>> pykx.CharVector(np.array([b'a', b'b', b'c'], dtype='|S1'))
	pykx.CharVector(pykx.q('"abc"'))
	```
=== "Pandas"
	Calling `.pd()` on a `pykx.CharVector` will return a pandas `Series` with `dtype` `bytes8`.

	```Python
	>>> pykx.CharVector(b'abc').pd()
	0    b'a'
	1    b'b'
	2    b'c'
	dtype: bytes8
	```

	Converting a `Series` of this `dtype` will create a `pykx.CharVector`.

	```Python
	>>> pykx.CharVector(pd.Series([b'a', b'b', b'c'], dtype=bytes))
	pykx.CharVector(pykx.q('"abc"'))
	```
=== "PyArrow"
	Calling `.pa()` on a `pykx.CharVector` will return a pyarrow `BinaryArray`.

	```Python
	<pyarrow.lib.BinaryArray object at 0x7f44cc099c00>
	[
	  61,
	  62,
	  63
	]
	```

## `pykx.SymbolAtom`

=== "Python"
	The python string type will be converted into a `pykx.SymbolAtom`.

	```Python
	>>> pykx.toq('symbol')
	pykx.SymbolAtom(pykx.q('`symbol'))
	```

	Calling `.py()` on a `pykx.SymbolAtom` will return a python string object.

	```Python
	>>> pykx.toq('symbol').py()
	'symbol'
	```

## `pykx.SymbolVector`

=== "Python"
	A list of python string types will be converted into a `pykx.SymbolVector`.

	```Python
	>>> pykx.SymbolVector(['a', 'b', 'c'])
	pykx.SymbolVector(pykx.q('`a`b`c'))
	```

	Calling `.py()` on a `pykx.SymbolVector` will return a list of python string objects.

	```Python
	>>> pykx.SymbolVector(['a', 'b', 'c']).py()
	['a', 'b', 'c']
	```

=== "Numpy"
	Calling `.np()` on a `pykx.SymbolVector` will return a numpy `ndarray` of python strings with `dtype` `object`.

	```Python
	>>> pykx.SymbolVector(['a', 'b', 'c']).np()
	array(['a', 'b', 'c'], dtype=object)
	```

	Converting a `ndarray` of `dtype` `object` will create a `pykx.SymbolVector`.

	```Python
	>>> pykx.SymbolVector(np.array(['a', 'b', 'c'], dtype=object))
	pykx.SymbolVector(pykx.q('`a`b`c'))
	```

=== "Pandas"
	Calling `.pd()` on a `pykx.SymbolVector` will return a pandas `Series` with `dtype` `object`.

	```Python
	>>> pykx.SymbolVector(['a', 'b', 'c']).pd()
	0    a
	1    b
	2    c
	dtype: object
	```

=== "PyArrow"
	Calling `.pa()` on a `pykx.SymbolVector` will return a pyarrow `StringArray`.

	```Python
	>>> pykx.SymbolVector(['a', 'b', 'c']).pa()
	<pyarrow.lib.StringArray object at 0x7f44cc323fa0>
	[
	  "a",
	  "b",
	  "c"
	]
	```

## `pykx.TimestampAtom`

=== "Python"
	The python datetime type will be converted into a `pykx.TimestampAtom`.

	```Python
	>>> kx.TimestampAtom(datetime(2150, 10, 22, 20, 31, 15, 70713))
	pykx.TimestampAtom(pykx.q('2150.10.22D20:31:15.070713000'))
	```

	Calling `.py()` on a `pykx.TimestampAtom` will return a python datetime object.

	```Python
	>>> kx.TimestampAtom(datetime(2150, 10, 22, 20, 31, 15, 70713)).py()
	datetime.datetime(2150, 10, 22, 20, 31, 15, 70713)
	```

## `pykx.TimestampVector`

=== "Python"
	A list of python `datetime` types will be converted into a `pykx.TimestampVector`.

	```Python
	>>> kx.TimestampVector([datetime(2150, 10, 22, 20, 31, 15, 70713), datetime(2050, 10, 22, 20, 31, 15, 70713)])
	pykx.TimestampVector(pykx.q('2150.10.22D20:31:15.070713000 2050.10.22D20:31:15.070713000'))
	```

	Calling `.py()` on a `pykx.TimestampVector` will return a list of python `datetime` objects.

	```Python
	>>> kx.TimestampVector([datetime(2150, 10, 22, 20, 31, 15, 70713), datetime(2050, 10, 22, 20, 31, 15, 70713)]).py()
	[datetime.datetime(2150, 10, 22, 20, 31, 15, 70713), datetime.datetime(2050, 10, 22, 20, 31, 15, 70713)]
	```

=== "Numpy"
	Calling `.np()` on a `pykx.TimestampVector` will return a numpy `ndarray` with `dtype` `datetime64[ns]`.

	```Python
	>>> kx.TimestampVector([datetime(2150, 10, 22, 20, 31, 15, 70713), datetime(2050, 10, 22, 20, 31, 15, 70713)]).np()
	array(['2150-10-22T20:31:15.070713000', '2050-10-22T20:31:15.070713000'],
	      dtype='datetime64[ns]')
	```

	Converting a `ndarray` of `dtype` `datetime64[ns]` will create a `pykx.TimestampVector`.

	```Python
	>>> kx.TimestampVector(np.array(['2150-10-22T20:31:15.070713000', '2050-10-22T20:31:15.070713000'], dtype='datetime64[ns]'))
	pykx.TimestampVector(pykx.q('2150.10.22D20:31:15.070713000 2050.10.22D20:31:15.070713000'))
	```

=== "Pandas"
	Calling `.pd()` on a `pykx.TimestampVector` will return a pandas `Series` with `dtype`:
	
	1. `datetime64[ns]`:

		```python
		>>> kx.TimestampVector([datetime(2150, 10, 22, 20, 31, 15, 70713), datetime(2050, 10, 22, 20, 31, 15, 70713)]).pd()
		0   2150-10-22 20:31:15.070713
		1   2050-10-22 20:31:15.070713
		dtype: datetime64[ns]
		```

	2. `timestamp[ns][pyarrow]` in pandas>=2.0 with `as_arrow=True`:

		```python
		>>> kx.TimestampVector([datetime(2150, 10, 22, 20, 31, 15, 70713), datetime(2050, 10, 22, 20, 31, 15, 70713)]).pd(as_arrow=True)
		0    2150-10-22 20:31:15.070713
		1    2050-10-22 20:31:15.070713
		dtype: timestamp[ns][pyarrow]
		```

=== "PyArrow"
	Calling `.pa()` on a `pykx.TimestampVector` will return a pyarrow `TimestampArray`.

	```Python
	>>> kx.TimestampVector([datetime(2150, 10, 22, 20, 31, 15, 70713), datetime(2050, 10, 22, 20, 31, 15, 70713)]).pa()
	<pyarrow.lib.TimestampArray object at 0x7f6428f0dde0>
	[
	  2150-10-22 20:31:15.070713000,
	  2050-10-22 20:31:15.070713000
	]
	```

## `pykx.MonthAtom`

=== "Python"
	The python date type will be converted into a `pykx.MonthAtom`.

	```Python
	>>> from datetime import date
	>>> kx.MonthAtom(date(1972, 5, 1))
	pykx.MonthAtom(pykx.q('1972.05m'))
	```

	Calling `.py()` on a `pykx.MonthAtom` will return a python date object.

	```Python
	>>> kx.MonthAtom(date(1972, 5, 1)).py()
	datetime.date(1972, 5, 1)
	```

## `pykx.MonthVector`

=== "Python"
	A list of python `date` types will be converted into a `pykx.MonthVector`.

	```Python
	>>> kx.MonthVector([date(1972, 5, 1), date(1999, 5, 1)])
	pykx.MonthVector(pykx.q('1972.05 1999.05m'))
	```

	Calling `.py()` on a `pykx.MonthVector` will return a list of python `date` objects.

	```Python
	>>> kx.MonthVector([date(1972, 5, 1), date(1999, 5, 1)]).py()
	[datetime.date(1972, 5, 1), datetime.date(1999, 5, 1)]
	```

=== "Numpy"
	Calling `.np()` on a `pykx.MonthVector` will return a numpy `ndarray` with `dtype` `datetime64[M]`.

	```Python
	>>> kx.MonthVector([date(1972, 5, 1), date(1999, 5, 1)]).np()
	array(['1972-05', '1999-05'], dtype='datetime64[M]')
	```

	Converting a `ndarray` of `dtype` `datetime64[M]` will create a `pykx.MonthVector`.

	```Python
	>>> kx.MonthVector(np.array(['1972-05', '1999-05'], dtype='datetime64[M]'))
	pykx.MonthVector(pykx.q('1972.05 1999.05m'))
	```

=== "Pandas"
	Calling `.pd()` on a `pykx.MonthVector` will return a pandas `Series` with `dtype`:

	1. `datetime64[ns]` in `pandas<2.0`:
   
		```python
		>>> kx.MonthVector([date(1972, 5, 1), date(1999, 5, 1)]).pd()
		0   1972-05-01
		1   1999-05-01
		dtype: datetime64[ns]
		```

	2.  `datetime64[s]` in `pandas>=2.0`:

		```python
		>>> kx.MonthVector([date(1972, 5, 1), date(1999, 5, 1)]).pd()
		0   1972-05-01
		1   1999-05-01
		dtype: datetime64[s]
		```

	3. `timestamp[s][pyarrow]` in `pandas>=2.0` with `as_arrow=True`:

		```python
		>>> kx.MonthVector([date(1972, 5, 1), date(1999, 5, 1)]).pd(as_arrow=True)
		0    1972-05-01 00:00:00
		1    1999-05-01 00:00:00
		dtype: timestamp[s][pyarrow]
		```

## `pykx.DateAtom`

=== "Python"
	The python date type will be converted into a `pykx.DateAtom`.

	```Python
	>>> kx.DateAtom(date(1972, 5, 31))
	pykx.DateAtom(pykx.q('1972.05.31'))
	```

	Calling `.py()` on a `pykx.DateAtom` will return a python date object.

	```Python
	>>> kx.DateAtom(date(1972, 5, 31)).py()
	datetime.date(1972, 5, 31)
	```

## `pykx.DateVector`

=== "Python"
	A list of python `date` types will be converted into a `pykx.DateVector`.

	```Python
	>>> kx.DateVector([date(1972, 5, 1), date(1999, 5, 1)])
	pykx.DateVector(pykx.q('1972.05.01 1999.05.01'))
	```

	Calling `.py()` on a `pykx.DateVector` will return a list of python `date` objects.

	```Python
	>>> kx.DateVector([date(1972, 5, 1), date(1999, 5, 1)]).py()
	[datetime.date(1972, 5, 1), datetime.date(1999, 5, 1)]
	```

=== "Numpy"
	Calling `.np()` on a `pykx.DateVector` will return a numpy `ndarray` of python strings with `dtype` `datetime64[D]`.

	```Python
	>>> kx.DateVector([date(1972, 5, 1), date(1999, 5, 1)]).np()
	array(['1972-05-01', '1999-05-01'], dtype='datetime64[D]')
	```

	Converting a `ndarray` of `dtype` `datetime64[D]` will create a `pykx.DateVector`.

	```Python
	>>> kx.DateVector(np.array(['1972-05-01', '1999-05-01'], dtype='datetime64[D]'))
	pykx.DateVector(pykx.q('1972.05.01 1999.05.01'))
	```

=== "Pandas"
	Calling `.pd()` on a `pykx.DateVector` will return a pandas `Series` with `dtype`:

	1.  `datetime64[ns]` in `pandas<2.0`:

		```python
		# pandas<2.0
		>>> kx.DateVector([date(1972, 5, 1), date(1999, 5, 1)]).pd()
		0   1972-05-01
		1   1999-05-01
		dtype: datetime64[ns]
		```

	2. `datetime64[s]` in `pandas>=2.0`:

		```python
		>>> kx.DateVector([date(1972, 5, 1), date(1999, 5, 1)]).pd()
		0   1972-05-01
		1   1999-05-01
		dtype: datetime64[s]
		```

	3. `timestamp[s][pyarrow]` in `pandas>=2.0` with `as_arrow=True`:

		```python
		>>> kx.DateVector([date(1972, 5, 1), date(1999, 5, 1)]).pd(as_arrow=True)
		0    1972-05-01 00:00:00
		1    1999-05-01 00:00:00
		dtype: timestamp[s][pyarrow]
		```

=== "PyArrow"
	Calling `.pa()` on a `pykx.DateVector` will return a pyarrow `Date32Array`.

	```Python
	>>> kx.DateVector([date(1972, 5, 1), date(1999, 5, 1)]).pa()
	<pyarrow.lib.Date32Array object at 0x7f6428f0de40>
	[
	  1972-05-01,
	  1999-05-01
	]
	```

## `pykx.Datetime` types

=== "Python and Numpy"
	These types are deprecated and can only be accessed using the `raw` key word argument.

	Converting these types to python will return a float object or a `float64` object in numpy's case.

	```Python
	>>> kx.q('0001.02.03T04:05:06.007 0001.02.03T04:05:06.007').py(raw=True)
	[-730085.8297915857, -730085.8297915857]
	>>> kx.q('0001.02.03T04:05:06.007 0001.02.03T04:05:06.007').np(raw=True)
	array([-730085.82979159, -730085.82979159])
	>>> kx.q('0001.02.03T04:05:06.007 0001.02.03T04:05:06.007').np(raw=True).dtype
	dtype('float64')
	```

## `pykx.TimespanAtom`

=== "Python"
	The python `timedelta` type will be converted into a `pykx.TimespanAtom`.

	```Python
	>>> from datetime import timedelta
	>>> kx.TimespanAtom(timedelta(days=43938, seconds=68851, microseconds=664551))
	pykx.TimespanAtom(pykx.q('43938D19:07:31.664551000'))
	```

	Calling `.py()` on a `pykx.TimespanAtom` will return a python `timedelta` object.

	```Python
	>>> kx.TimespanAtom(timedelta(days=43938, seconds=68851, microseconds=664551)).py()
	datetime.timedelta(days=43938, seconds=68851, microseconds=664551)
	```

## `pykx.TimespanVector`

=== "Python"
	A list of python `timedelta` types will be converted into a `pykx.TimespanVector`.

	```Python
	>>> kx.TimespanVector([timedelta(days=43938, seconds=68851, microseconds=664551), timedelta(days=43938, seconds=68851, microseconds=664551)])
	pykx.TimespanVector(pykx.q('43938D19:07:31.664551000 43938D19:07:31.664551000'))
	```

	Calling `.py()` on a `pykx.TimespanVector` will return a list of python `timedelta` objects.

	```Python
	>>> kx.TimespanVector([timedelta(days=43938, seconds=68851, microseconds=664551), timedelta(days=43938, seconds=68851, microseconds=664551)]).py()
	[datetime.timedelta(days=43938, seconds=68851, microseconds=664551), datetime.timedelta(days=43938, seconds=68851, microseconds=664551)]
	```

=== "Numpy"
	Calling `.np()` on a `pykx.TimespanVector` will return a numpy `ndarray` of python strings with `dtype` `timedelta64[ns]`.

	```Python
	>>> kx.TimespanVector([timedelta(days=43938, seconds=68851, microseconds=664551), timedelta(days=43938, seconds=68851, microseconds=664551)]).np()
	array([3796312051664551000, 3796312051664551000], dtype='timedelta64[ns]')
	```

	Converting a `ndarray` of `dtype` `datetime64[ns]` will create a `pykx.TimespanVector`.

	```Python
	>>> kx.TimespanVector(np.array([3796312051664551000, 3796312051664551000], dtype='timedelta64[ns]'))
	pykx.TimespanVector(pykx.q('43938D19:07:31.664551000 43938D19:07:31.664551000'))
	```

=== "Pandas"
	Calling `.pd()` on a `pykx.TimespanVector` will return a pandas `Series` with `dtype`:

	1.  `timedelta64[ns]`:

		```python
		>>> kx.TimespanVector([timedelta(days=43938, seconds=68851, microseconds=664551), timedelta(days=43938, seconds=68851, microseconds=664551)]).pd()
		0   43938 days 19:07:31.664551
		1   43938 days 19:07:31.664551
		dtype: timedelta64[ns]
		```

	2.  `duration[ns][pyarrow]` in `pandas>=2.0` with `as_arrow=True`:

		```python
		>>> kx.TimespanVector([timedelta(days=43938, seconds=68851, microseconds=664551), timedelta(days=43938, seconds=68851, microseconds=664551)]).pd(as_arrow=True)
		0    43938 days 19:07:31.664551
		1    43938 days 19:07:31.664551
		dtype: duration[ns][pyarrow]
		```

=== "PyArrow"
	Calling `.pa()` on a `pykx.TimespanVector` will return a pyarrow `DurationArray`.

	```Python
	>>> kx.TimespanVector([timedelta(days=43938, seconds=68851, microseconds=664551), timedelta(days=43938, seconds=68851, microseconds=664551)]).pa()
	<pyarrow.lib.DurationArray object at 0x7f6428f0dea0>
	[
	  3796312051664551000,
	  3796312051664551000
	]
	```

## `pykx.MinuteAtom`

=== "Python"
	The python `timedelta` type will be converted into a `pykx.MinuteAtom`.

	```Python
	>>> kx.MinuteAtom(timedelta(minutes=216))
	pykx.MinuteAtom(pykx.q('03:36'))
	```

	Calling `.py()` on a `pykx.MinuteAtom` will return a python `timedelta` object.

	```Python
	>>> kx.MinuteAtom(timedelta(minutes=216)).py()
	datetime.timedelta(seconds=12960)
	```

## `pykx.MinuteVector`

=== "Python"
	A list of python `timedelta` types will be converted into a `pykx.MinuteVector`.

	```Python
	>>> kx.MinuteVector([timedelta(minutes=216), timedelta(minutes=67)])
	pykx.MinuteVector(pykx.q('03:36 01:07'))
	```

	Calling `.py()` on a `pykx.MinuteVector` will return a list of python `timedelta` objects.

	```Python
	>>> kx.MinuteVector([timedelta(minutes=216), timedelta(minutes=67)]).py()
	[datetime.timedelta(seconds=12960), datetime.timedelta(seconds=4020)]
	```

=== "Numpy"
	Calling `.np()` on a `pykx.MinuteVector` will return a numpy `ndarray` of python strings with `dtype` `timedelta64[m]`.

	```Python
	>>> kx.MinuteVector([timedelta(minutes=216), timedelta(minutes=67)]).np()
	array([216,  67], dtype='timedelta64[m]')
	```

	Converting a `ndarray` of `dtype` `timedelta64[m]` will create a `pykx.MinuteVector`.

	```Python
	>>> kx.MinuteVector(np.array([216,  67], dtype='timedelta64[m]'))
	pykx.MinuteVector(pykx.q('03:36 01:07'))
	```

=== "Pandas"
	Calling `.pd()` on a `pykx.MinuteVector` will return a pandas `Series` with `dtype`:

    1. `timedelta64[ns]` in `pandas<2.0`:

		```python
		>>> kx.MinuteVector([timedelta(minutes=216), timedelta(minutes=67)]).pd()
		0   0 days 03:36:00
		1   0 days 01:07:00
		dtype: timedelta64[ns]
		```

	2. `timedelta64[s]` in `pandas>=2.0`:

		```python
		>>> kx.MinuteVector([timedelta(minutes=216), timedelta(minutes=67)]).pd()
		0   0 days 03:36:00
		1   0 days 01:07:00
		dtype: timedelta64[s]
		```

	3.  `duration[s][pyarrow]` in `pandas>=2.0` with `as_arrow=True`:
   
		```python
		>>> kx.MinuteVector([timedelta(minutes=216), timedelta(minutes=67)]).pd(as_arrow=True)
		0    0 days 03:36:00
		1    0 days 01:07:00
		dtype: duration[s][pyarrow]
		```

## `pykx.SecondAtom`

=== "Python"
	The python `timedelta` type will be converted into a `pykx.SecondAtom`.

	```Python
	>>> kx.SecondAtom(timedelta(seconds=13019))
	pykx.SecondAtom(pykx.q('03:36:59'))
	```

	Calling `.py()` on a `pykx.SecondAtom` will return a python `timedelta` object.

	```Python
	>>> kx.SecondAtom(timedelta(seconds=13019)).py()
	datetime.timedelta(seconds=13019)
	```

## `pykx.SecondVector`

=== "Python"
	A list of python `timedelta` types will be converted into a `pykx.SecondVector`.

	```Python
	>>> kx.SecondVector([timedelta(seconds=13019), timedelta(seconds=1019)])
	pykx.SecondVector(pykx.q('03:36:59 00:16:59'))
	```

	Calling `.py()` on a `pykx.SecondVector` will return a list of python `timedelta` objects.

	```Python
	>>> kx.SecondVector([timedelta(seconds=13019), timedelta(seconds=1019)]).py()
	[datetime.timedelta(seconds=13019), datetime.timedelta(seconds=1019)]
	```

=== "Numpy"
	Calling `.np()` on a `pykx.SecondVector` will return a numpy `ndarray` of python strings with `dtype` `timedelta64[s]`.

	```Python
	>>> kx.SecondVector([timedelta(seconds=13019), timedelta(seconds=1019)]).np()
	array([13019,  1019], dtype='timedelta64[s]')
	```

	Converting a `ndarray` of `dtype` `timedelta64[s]` will create a `pykx.SecondVector`.

	```Python
	>>> kx.SecondVector(np.array([13019,  1019], dtype='timedelta64[s]'))
	pykx.SecondVector(pykx.q('03:36:59 00:16:59'))
	```

=== "Pandas"
	Calling `.pd()` on a `pykx.SecondVector` will return a pandas `Series` with `dtype`:

	1. `timedelta64[ns]` in `pandas<2.0`:
		```python
		# pandas<2.0
		>>> kx.SecondVector([timedelta(seconds=13019), timedelta(seconds=1019)]).pd()
		0   0 days 03:36:59
		1   0 days 00:16:59
		dtype: timedelta64[ns]
		```

	2. `timedelta64[s]` in `pandas>=2.0`:

		```python
		>>> kx.SecondVector([timedelta(seconds=13019), timedelta(seconds=1019)]).pd()
		0   0 days 03:36:59
		1   0 days 00:16:59
		dtype: timedelta64[s]
		```

 	3. `duration[s][pyarrow]` in `pandas>=2.0` with `as_arrow=True`:

		```python
		>>> kx.SecondVector([timedelta(seconds=13019), timedelta(seconds=1019)]).pd(as_arrow=True)
		0    0 days 03:36:59
		1    0 days 00:16:59
		dtype: duration[s][pyarrow]
		```

=== "PyArrow"
	Calling `.pa()` on a `pykx.SecondVector` will return a pyarrow `DurationArray`.

	```Python
	>>> kx.SecondVector([timedelta(seconds=13019), timedelta(seconds=1019)]).pa()
	<pyarrow.lib.DurationArray object at 0x7f6428f0e020>
	[
	  13019,
	  1019
	]
	```

## `pykx.TimeAtom`

=== "Python"
	The python `timedelta` type will be converted into a `pykx.TimeAtom`.

	```Python
	>>> kx.TimeAtom(timedelta(seconds=59789, microseconds=214000))
	pykx.TimeAtom(pykx.q('16:36:29.214'))
	```

	Calling `.py()` on a `pykx.TimeAtom` will return a python `timedelta` object.

	```Python
	>>> kx.TimeAtom(timedelta(seconds=59789, microseconds=214000)).py()
	datetime.timedelta(seconds=59789, microseconds=214000)
	```

## `pykx.TimeVector`

=== "Python"
	A list of python `timedelta` types will be converted into a `pykx.TimeVector`.

	```Python
	>>> kx.TimeVector([timedelta(seconds=59789, microseconds=214000), timedelta(seconds=23789, microseconds=214000)])
	pykx.TimeVector(pykx.q('16:36:29.214 06:36:29.214'))
	```

	Calling `.py()` on a `pykx.TimeVector` will return a list of python `timedelta` objects.

	```Python
	>>> kx.TimeVector([timedelta(seconds=59789, microseconds=214000), timedelta(seconds=23789, microseconds=214000)]).py()
	[datetime.timedelta(seconds=59789, microseconds=214000), datetime.timedelta(seconds=23789, microseconds=214000)]
	```

=== "Numpy"

	Calling `.np()` on a `pykx.TimeVector` will return a numpy `ndarray` of python strings with `dtype` `timedelta64[ms]`.

	```Python
	>>> kx.TimeVector([timedelta(seconds=59789, microseconds=214000), timedelta(seconds=23789, microseconds=214000)]).np()
	array([59789214, 23789214], dtype='timedelta64[ms]')
	```

	Converting a `ndarray` of `dtype` `timedelta64[ms]` will create a `pykx.TimeVector`.

	```Python
	>>> kx.TimeVector(np.array([59789214, 23789214], dtype='timedelta64[ms]'))
	pykx.TimeVector(pykx.q('16:36:29.214 06:36:29.214'))
	```

=== "Pandas"

	Calling `.pd()` on a `pykx.TimeVector` will return a pandas `Series` with `dtype`:

	1.  `timedelta64[ns]` in `pandas<2.0`:

		```python
		>>> kx.TimeVector([timedelta(seconds=59789, microseconds=214000), timedelta(seconds=23789, microseconds=214000)]).pd()
		0   0 days 16:36:29.214000
		1   0 days 06:36:29.214000
		dtype: timedelta64[ns]
		```

	2. `timedelta[ms]` in `pandas>=2.0`:

		```python
		>>> kx.TimeVector([timedelta(seconds=59789, microseconds=214000), timedelta(seconds=23789, microseconds=214000)]).pd()
		0   0 days 16:36:29.214000
		1   0 days 06:36:29.214000
		dtype: timedelta64[ms]
		```

	3. `duration[ms][pyarrow]` in `pandas>=2.0` with `as_arrow=True`:

		```python
		>>> kx.TimeVector([timedelta(seconds=59789, microseconds=214000), timedelta(seconds=23789, microseconds=214000)]).pd(as_arrow=True)
		0    0 days 16:36:29.214000
		1    0 days 06:36:29.214000
		dtype: duration[ms][pyarrow]
		```

=== "PyArrow"
	Calling `.pa()` on a `pykx.TimeVector` will return a pyarrow `DurationArray`.

	```Python
	>>> kx.TimeVector([timedelta(seconds=59789, microseconds=214000), timedelta(seconds=23789, microseconds=214000)]).pa()
	<pyarrow.lib.DurationArray object at 0x7f643021ff40>
	[
	  59789214,
	  23789214
	]
	```

## `pykx.Dictionary`

=== "Python"
	A python `dict` type will be converted into a `pykx.Dictionary`.

	```Python
	>>> kx.Dictionary({'foo': b'bar', 'baz': 3.5, 'z': 'prime'})
	pykx.Dictionary(pykx.q('
	foo| "bar"
	baz| 3.5
	z  | `prime
	'))
	```

	Calling `.py()` on a `pykx.Dictionary` will return a python `dict` object.

	```Python
	>>> kx.Dictionary({'foo': b'bar', 'baz': 3.5, 'z': 'prime'}).py()
	{'foo': b'bar', 'baz': 3.5, 'z': 'prime'}
	```

## `pykx.Table`

=== "Python"
	Calling `.py()` on a `pykx.Table` will return a python `dict` object.

	```Python
	>>> kx.Table(data={
	...     'a': kx.random.random(10, 10),
	...     'b': kx.random.random(10, 10)}).py()
	{'a': [5, 6, 4, 1, 3, 3, 7, 8, 2, 1], 'b': [8, 1, 7, 2, 4, 5, 4, 2, 7, 8]}
	```

=== "Numpy"
	Calling `.np()` on a `pykx.Table` will return a numpy `record` array of the rows of the table with each type converted to it closest analogous numpy type.

	```Python
	>>> kx.Table(data={
	...     'a': kx.random.random(10, 10),
	...     'b': kx.random.random(10, 10)}).np()
	rec.array([(9, 9), (9, 7), (2, 6), (5, 6), (4, 4), (2, 7), (5, 8), (8, 4),
        	   (7, 4), (9, 6)],
	           dtype=[('a', '<i8'), ('b', '<i8')])
	```

=== "Pandas"
	Calling `.pd()` on a `pykx.Table` will return a pandas `DataFrame` with each column being converted to its closest pandas `dtype`.

	```Python
	>>> kx.Table(data={
	...     'a': kx.random.random(10, 10),
	...     'b': kx.random.random(10, 10)}).pd()
	   a  b
	0  1  9
	1  0  7
	2  5  7
	3  1  1
	4  0  9
	5  0  1
	6  1  0
	7  7  8
	8  6  8
	9  3  3
	```

	Converting a `pandas` `DataFrame` object will result in a `pykx.Table` object.

	```Python
	>>> kx.Table(pd.DataFrame({'a': [x for x in range(10)], 'b': [float(x) for x in range(10)]}))
	pykx.Table(pykx.q('
	a b
	---
	0 0
	1 1
	2 2
	3 3
	4 4
	5 5
	6 6
	7 7
	8 8
	9 9
	'))
	```

=== "PyArrow"
	Calling `.pa()` on a `pykx.Table` will return a pyarrow `Table`.

	```Python
	>>> kx.Table(data={
	...     'a': kx.random.random(10, 10),
	...     'b': kx.random.random(10, 10)}).pa()
	pyarrow.Table
	a: int64
	b: int64
	----
	a: [[0,7,3,3,6,8,2,3,8,9]]
	b: [[5,7,5,6,7,0,2,1,8,1]]
	```

	Converting a `pyarow` `Table` object will result in a `pykx.Table` object.

	```Python
	>>> kx.Table(pa.Table.from_arrays([[1, 2, 3, 4], [5, 6, 7, 8]], names=['a', 'b']))
	pykx.Table(pykx.q('
	a b
	---
	1 5
	2 6
	3 7
	4 8
	'))
	```

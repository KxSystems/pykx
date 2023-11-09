# Interacting with PyKX objects

In order to use the power of q and the functionality provided by PyKX a user must at some point interact with a PyKX object. At it's most basic level these items are allocated C representations of q/kdb+ objects within a memory space managed by q. Keeping the data in this format allows it to be used directly for query/analytic execution in q without any translation overhead.

There are a number of ways to generate PyKX objects:

1. Explicitly converting from a Python object to a PyKX object
2. By evaluating q code using `kx.q`
3. By retrieving a named entity from q's memory
4. Through query of an external q session

Getting the data to a PyKX format provides you with the ability to easily interact with these objects using q or the analytic functionality provided by PyKX, however, having data in this format is not suitable for all use-cases. For example, should a function require a Pandas DataFrame as input then a PyKX object must be converted to a Pandas DataFrame. This is supported using methods provided for the majority of PyKX objects, these are covered below.

## Generating PyKX objects

### Explicitly converting from Pythonic objects to PyKX objects

The most simplistic method of creating a PyKX object is to convert an analogous Pythonic type to a PyKX object. This is facilitated through the use of the functions `pykx.toq` which allows conversions from Python, Numpy, Pandas and PyArrow types to PyKX objects, open the tabs which are of interest to you to see some examples of these conversions

??? Note "Specifying target types"

	When converting Pythonic objects to PyKX types users can make use of the `ktype` named argument. Users converting lists/atomic elements should use [PyKX types](../../api/pykx-q-data/type_conversions.md), if converting Pandas DataFrames or PyArrow Tables users can make use of the `ktype` argument with a dictionary input mapping the column name to the [PyKX type](../../api/pykx-q-data/type_conversions.md).

=== "Python"

	```python
	>>> import pykx as kx
	>>> pyatom = 2
	>>> pylist = [1, 2, 3]
	>>> pydict = {'x': [1, 2, 3], 'y': {'x': 3}}
	>>>
	>>> kx.toq(pyatom)
	pykx.LongAtom(pykx.q('2'))
	>>> kx.toq(pylist)
	pykx.LongVector(pykx.q('1 2 3'))
	>>> kx.toq(pylist, kx.FloatVector)
	pykx.FloatVector(pykx.q('1 2 3f'))
	>>> kx.toq(pydict)
	pykx.Dictionary(pykx.q('
	x| (1;2;3)
	y| (,`x)!,3
	'))
	```

=== "Numpy"

	```python
	>>> import pykx as kx
	>>> import numpy as np
	>>> nparray1 = np.array([1, 2, 3])
	>>> nparray2 = np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')
	>>> nparray3 = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
	>>>
	>>> kx.toq(nparray1)
	pykx.LongVector(pykx.q('1 2 3'))
 	>>> kx.toq(nparray1, kx.FloatVector)
	pykx.FloatVector(pykx.q('1 2 3f'))
	>>> kx.toq(nparray2)
	pykx.DateVector(pykx.q('2007.07.13 2006.01.13 2010.08.13'))
	>>> kx.toq(nparray3)
	pykx.List(pykx.q('
	1 2 3
	4 5 6
	'))
	```

=== "Pandas"

	```python
	>>> import pykx as kx
	>>> import pandas as pd
	>>> import numpy as np
	>>> pdseries1 = pd.Series([1, 2, 3])
	>>> pdseries2 = pd.Series([1, 2, 3], dtype=np.int32)
	>>> df = pd.DataFrame.from_dict({'x': [1, 2], 'y': ['a', 'b']})
	>>> kx.toq(pdseries1)
	pykx.LongVector(pykx.q('1 2 3'))
	>>> kx.toq(pdseries1, kx.FloatVector)
	pykx.FloatVector(pykx.q('1 2 3f'))
	>>> kx.toq(pdseries2)
	pykx.IntVector(pykx.q('1 2 3i'))
	>>> kx.toq(df)
	pykx.Table(pykx.q('
	x y
	---
	1 a
	2 b
	'))
	>>> kx.toq(df).dtypes
	pykx.Table(pykx.q('
	columns type           
	-----------------------
	x       "kx.LongAtom"  
	y       "kx.SymbolAtom"
	'))
	>>> kx.toq(df, ktype={'x': kx.FloatVector}).dtypes
	pykx.Table(pykx.q('
	columns type           
	-----------------------
	x       "kx.FloatAtom" 
	y       "kx.SymbolAtom"
	'))
	```

=== "PyArrow"

	```python
	>>> import pykx as kx
	>>> import pyarrow as pa
	>>> arr = pa.array([1, 2, None, 3])
	>>> nested_arr = pa.array([[], None, [1, 2], [None, 1]])
	>>> dict_arr = pa.array([{'x': 1, 'y': True}, {'z': 3.4, 'x': 4}])
	>>> kx.toq(arr)
	pykx.FloatVector(pykx.q('1 2 0n 3'))
	>>> kx.toq(nested_arr)
	pykx.List(pykx.q('
	`float$()
	::
	1 2f
	0n 1
	'))
	>>> kx.toq(dict_arr)
	pykx.List(pykx.q('
	x y  z
	--------
	1 1b ::
	4 :: 3.4
	'))
	>>>
	>>> n_legs = pa.array([2, 4, 5, 100])
	>>> animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
	>>> names = ["n_legs", "animals"]
	>>> tab = pa.Table.from_arrays([n_legs, animals], names=names)
	>>> kx.toq(tab)
	pykx.Table(pykx.q('
	n_legs animals      
	--------------------
	2      Flamingo     
	4      Horse        
	5      Brittle stars
	100    Centipede    
	'))
	>>> kx.toq(tab).dtypes
	pykx.Table(pykx.q('
	columns type           
	-----------------------
	n_legs  "kx.LongAtom"  
	animals "kx.SymbolAtom"
	'))
	>>> kx.toq(tab, {'animals': kx.CharVector}).dtypes
	pykx.Table(pykx.q('
	columns type           
	-----------------------
	n_legs  "kx.LongAtom"  
	animals "kx.CharVector"
	'))
	```

### Generating data using PyKX inbuilt functions

For users who wish to generate objects directly but who are not familiar with q and want to quickly prototype functionality a number of helper functions can be used.

Create a vector of random floating point precision values

```python
>>> kx.random.random(3, 10.0)
pykx.FloatVector(pykx.q('9.030751 7.750292 3.869818'))
```

Additionally, users when generating random data can use PyKX null/infinite data to create data across larger data ranges as follows

```python
>>> kx.random.random(2, kx.GUIDAtom.null)
pykx.GUIDVector(pykx.q('8c6b8b64-6815-6084-0a3e-178401251b68 5ae7962d-49f2-404d-5aec-f7c8abbae288'))
>>> kx.random.random(3, kx.IntAtom.inf)
pykx.IntVector(pykx.q('986388794 824432196 2022020141i'))
```

Create a two-dimensional list of random symbol values

```python
>>> kx.random.random([2, 3], ['a', 'b', 'c'])
pykx.List(pykx.q('
b b c
b a b
'))
```

Create a table of tabular data generated using random data

```python
>>> N = 100000
>>> table = kx.Table(
...     data = {'sym': kx.random.random(N, ['AAPL', 'MSFT']),
...             'price': kx.random.random(N, 100.0),
...             'size': 1+kx.random.random(N, 100)})
>>> table.head()
pykx.Table(pykx.q('
sym  price    size
------------------
MSFT 49.34749 50  
MSFT 23.31342 96  
AAPL 63.1368  36  
AAPL 98.71169 7   
AAPL 68.98055 94  
'))
```

### Evaluating q code using `kx.q`

For users more familiar with q it is possible to evaluate q code to generate PyKX objects, this can be done as follows

```python
>>> kx.q('til 10')
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
```

More information on the usage of `kx.q` can be found by following the documentation guide [here](evaluating.md)

### By retrieving a named entity from q's memory

As noted at the start of this guide PyKX objects exist in a memory space accessed and controlled by interactions with q, as such items which are created in q may not be immediately available as Python objects. For example if a named variable in q has been created as a side effect of a function call or explicitly created by a user it can be retrieved based on this name as follows.

```python
>>> kx.q('t:([]5?1f;5?1f)')            # Generate a named variable in a single object
pykx.Identity(pykx.q('::'))
>>> kx.q('{k::5?1f;k*x}',2)            # Generate a global variable k as a side effect
pykx.FloatVector(pykx.q('0.7855048 1.034182 1.031959 0.8133284 0.3561677'))
>>> kx.q['t']
pykx.Table(pykx.q('
x          x1       
--------------------
0.4931835  0.3017723
0.5785203  0.785033 
0.08388858 0.5347096
0.1959907  0.7111716
0.375638   0.411597 
'))
>>> kx.q['k']
pykx.FloatVector(pykx.q('0.3927524 0.5170911 0.5159796 0.4066642 0.1780839'))
```

### Through query of an external q session

PyKX provides an IPC interface allowing users to query and retrieve data from a q server. Assuming that a user has a q server with no username/password exposed on port 5000 it is possible to run synchronous and asynchronous events against this server as follows:

```python
>>> conn = kx.QConnection('localhost', 5000)    # Open a connection to the q server
>>> conn('til 10')                               # Execute a command server side
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
>>> conn.qsql.select('tab', where = 'x=`a')      # Query using qsql statement
pykx.Table(pykx.q('
x x1         x2        
-----------------------
a 0.481804   0.8112026 
a 0.4301331  0.04881728
a 0.8664098  0.9006991 
a 0.5281112  0.8505909 
a 0.06494865 0.8196014 
a 0.5464707  0.8187707 
a 0.9601549  0.6919292 
a 0.9256041  0.340393  
a 0.8276669  0.9456963 
a 0.5930176  0.8649262 
a 0.4746581  0.3114364 
a 0.8608133  0.8132478 
a 0.01668426 0.274227  
a 0.707851   0.2439194 
a 0.7632325  0.6568734 
a 0.927445   0.9625156 
a 0.1247049  0.3714973 
a 0.3992327  0.3550381 
a 0.7263287  0.3615143 
a 0.02810674 0.481821  
..
'))
```

## Converting PyKX objects to Pythonic Types

As mentioned above PyKX objects can be created and interacted with using q functionality, once the data is in a position to be used by Python it may be more appropriate to convert it to a Python, Numpy, Pandas or PyArrow representation. This is facilitated through the use of the following methods:

| Method   | Description                      |
|----------|----------------------------------|
| `*.py()` | Convert a PyKX object to Python  |
| `*.np()` | Convert a PyKX object to Numpy   |
| `*.pd()` | Convert a PyKX object to Pandas  |
| `*.pa()` | Convert a PyKX object to PyArrow |

The following provides some examples of this functionality in use:

```python
import pykx as kx
qarr = kx.q('til 5')
>>> qarr.py()
[0, 1, 2, 3, 4]
>>> qarr.np()
array([0, 1, 2, 3, 4])
>>> qarr.pd()
0    0
1    1
2    2
3    3
4    4
dtype: int64
>>> qarr.pa()
<pyarrow.lib.Int64Array object at 0x7ffabf2f4fa0>
[
  0,
  1,
  2,
  3,
  4
]
>>>
>>> qtab = kx.q('([]5?1f;5?1f)')
>>> qtab
pykx.Table(pykx.q('
x         x1       
-------------------
0.439081  0.4707883
0.5759051 0.6346716
0.5919004 0.9672398
0.8481567 0.2306385
0.389056  0.949975 
'))
>>> qtab.np()
rec.array([(0.43908099, 0.47078825), (0.57590514, 0.63467162),
           (0.59190043, 0.96723983), (0.84815665, 0.23063848),
           (0.38905602, 0.94997503)],
          dtype=[('x', '<f8'), ('x1', '<f8')])
>>> qtab.pd()
          x        x1
0  0.439081  0.470788
1  0.575905  0.634672
2  0.591900  0.967240
3  0.848157  0.230638
4  0.389056  0.949975
>>> qtab.pa()
pyarrow.Table
x: double
x1: double
```

!!! warning "Precision Loss Considerations"

	Care should be taken in particular when converting q temporal data to Python native data types. As Python temporal data types only support microsecond precision roundtrip conversions will reduce temporal granularity for q data.

		```python
		>>> import pykx as kx
		>>> qtime = kx.q('first 1?0p')
		>>> qtime
		pykx.TimestampAtom(pykx.q('2001.08.17D03:16:23.736627552'))
		>>> kx.toq(qtime.py())
		pykx.TimestampAtom(pykx.q('2001.08.17D03:16:23.736627000'))
		```

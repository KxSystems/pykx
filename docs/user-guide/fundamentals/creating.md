---
title: Create and convert PyKX objects
description: How to generate PyKX objects
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, PyKX objects, 
---

# Create and convert PyKX objects

_This page provides details on how to generate and convert PyKX objects._

!!! tip "Tip: For the best experience, we recommend reading [PyKX objects and attributes](..//../learn/objects.md) first." 

To use the power of q and the functionality provided by PyKX, at some point you must interact with PyKX objects. At their most basic level, objects are allocated C representations of q/kdb+ objects within a memory space managed by q. Keeping the data in this format allows it to be used directly for query/analytic execution in q without any translation overhead.

## 1. Create PyKX objects

There are five ways to create PyKX objects:

- a. [Convert Python objects to PyKX objects](#1a-convert-python-objects-to-pykx-objects)
- b. [Generate data using PyKX inbuilt functions](#1b-generate-data-using-pykx-inbuilt-functions)
- c. [Evaluate q code using `#!python kx.q`](#1c-evaluate-q-code-using-kxq)
- d. [Assign Python data to q's memory](#1d-assign-python-data-to-qs-memory)
- e. [Retrieve a named entity from q's memory](#1e-retrieve-a-named-entity-from-qs-memory)
- f. [Query an external q session](#1f-query-an-external-q-session)

### 1.a Convert Python objects to PyKX objects

The simplest way to create a PyKX object is by converting a similar Python type into a PyKX object. You can do this with the `#!python pykx.toq function`, which supports conversions from Python, NumPy, Pandas, PyArrow, and PyTorch (Beta) types to PyKX objects. Open the tabs that interest you to see conversion examples:

??? Note "Specify target types"

	When converting Pythonic objects to PyKX types, you can use the `ktype` named argument:

	- To convert lists/atomic elements, use [PyKX types](../../api/pykx-q-data/type_conversions.md);
	- To convert Pandas DataFrames or PyArrow Tables, use the `#!python ktype` argument with a dictionary input mapping the column name to the [PyKX type](../../api/pykx-q-data/type_conversions.md).

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

=== "PyTorch (Beta)"

	When converting data from PyTorch types to PyKX support is only provided for `#!python torch.Tensor` object conversions to PyKX at this time and requires setting of the configuration `PYKX_BETA_FEATURES=True` as shown below

	```python
	>>> import os
	>>> os.environ['PYKX_BETA_FEATURES'] = 'True'
	>>> import pykx as kx
	>>> import torch
	>>> pt = torch.Tensor([1, 2, 3])
	tensor([1., 2., 3.])
	>>> ptl = torch.Tensor([[1, 2, 3], [4, 5, 6]])
	tensor([[1., 2., 3.],
	        [4., 5., 6.]])
	>>> kx.toq(pt)
	pykx.RealVector(pykx.q('1 2 3e'))
	>>> kx.toq(ptl)
	pykx.List(pykx.q('
	1 2 3
	4 5 6
	'))
	```

By default, when you convert Python strings to PyKX, they are returned as `#!python pykx.SymbolAtom` objects. This ensures a clear distinction between `#!python str` (string) and `#!python byte` objects. However, you might prefer Python strings to be returned as `#!python pykx.CharVector` objects, to achieve memory efficiency or greater flexibility in analytic development. To do this, use the keyword argument `#!python strings_as_char`, which ensures that all `#!python str` objects are converted to `#!python pykx.CharVector` objects.

```python
>>> import pykx as kx
>>> kx.toq('str', strings_as_char=True)
pykx.CharVector(pykx.q('"str"'))
>>> kx.toq({'a': {'b': 'test'}, 'b': 'test1'}, strings_as_char=True)
pykx.Dictionary(pykx.q('
a| (,`b)!,"test"
b| "test1"
'))
```

### 1.b Generate data using PyKX inbuilt functions

For users who want to generate objects directly but are not familiar with q, and wish to quickly prototype this functionality, several helper functions are available.

Create a vector of random floating point precision values:

```python
>>> kx.random.random(3, 10.0)
pykx.FloatVector(pykx.q('9.030751 7.750292 3.869818'))
```

Additionally, when generating random data, you can use PyKX null/infinite data to create data across larger data ranges as follows:

```python
>>> kx.random.random(2, kx.GUIDAtom.null)
pykx.GUIDVector(pykx.q('8c6b8b64-6815-6084-0a3e-178401251b68 5ae7962d-49f2-404d-5aec-f7c8abbae288'))
>>> kx.random.random(3, kx.IntAtom.inf)
pykx.IntVector(pykx.q('986388794 824432196 2022020141i'))
```

Create a two-dimensional list of random symbol values:

```python
>>> kx.random.random([2, 3], ['a', 'b', 'c'])
pykx.List(pykx.q('
b b c
b a b
'))
```

Create a table of tabular data generated using random data:

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

For retrieval of current temporal information, call the `#!python date`, `#!python time`, and `#!python timestamp` type objects as follows:

```python
>>> kx.DateAtom('today')
pykx.DateAtom(pykx.q('2024.01.05'))
>>> kx.TimeAtom('now')
pykx.TimeAtom(pykx.q('16:22:12.178'))
>>> kx.TimestampAtom('now')
pykx.TimestampAtom(pykx.q('2024.01.05T16:22:21.012631000'))
```

### 1.c Evaluate q code using `#!python kx.q`

If you're more familiar with q, generate PyKX objects by evaluating q code: 

```python
>>> kx.q('til 10')
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
```

Documentation guide on [how to use `kx.q`](evaluating.md).

### 1.d Assign Python data to q's memory

Assignment of data from Python's memory space to q can take a number of forms:

- Using Python `__setitem__` syntax on the `kx.q` method: (_Suggested_)

	```python
	>>> kx.q['data'] = np.array([10, 20, 30])
	>>> kx.q['data']
	pykx.LongVector(pykx.q('10 20 30'))
	```

- Setting data to q explicitly through set/assignment in q: (_Available_)

	```python
	>>> kx.q('{data::x}', np.array([15, 25, 35]))
	pykx.Identity(pykx.q('::'))
	>>> kx.q['data']
	pykx.LongVector(pykx.q('15 25 35'))
	>>> kx.q.set('data', np.array([20, 30, 40]))
	pykx.SymbolAtom(pykx.q('`data'))
	>>> kx.q['data']
	pykx.LongVector(pykx.q('20 30 40'))
	```

- Using Python `__setattr__` syntax on the `kx.q` object: (_Discouraged_)

	```python
	>>> kx.q.data = np.array([30, 40, 50])
	>>> kx.q.data
	pykx.LongVector(pykx.q('30 40 50'))
	```

??? Note "Why `__setattr__` is discouraged"

	Data retrieval using `__getattr__` on the `kx.q` object is designed for use with the PyKX [context interface](../../api/pykx-execution/ctx.md). To comply with round-trip retrieval the assignment completed with `__setattr__` syntax persists data to a name with a leading `.`.

	To see the effect of this in practice we can look at the following example:

	```python
	>>> import pykx as kx
	>>> kx.q.data = [100, 200, 300]
	>>> kx.q['data']
	QError: data
	>>> kx.q['.data']
	pykx.LongVector(pykx.q('100 200 300'))
	```

### 1.e Retrieve a named entity from q's memory

As PyKX objects exist in a memory space accessed and controlled by interactions with q, the items created in q may not be immediately available as Python objects. For example, if you created a named variable in q as a side effect of a function call or just explicitly created it, you can retrieve it by its name:

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

### 1.f Query an external q session

PyKX provides an IPC interface allowing users to query and retrieve data from a q server. If you have a q server with no username/password exposed on `#!python port 5000`, it's possible to run synchronous and asynchronous events against this server:

```python
>>> conn = kx.QConnection('localhost', 5000)    # Open a connection to the q server
>>> conn('til 10')                              # Execute a command server side
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
>>> conn['tab'] = kx.q('([]100?`a`b;100?1f;100?1f)') # Generate a table on the server
>>> conn.qsql.select('tab', where = 'x=`a')     # Query using qsql statement
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

## 2. Convert PyKX objects to Pythonic types

Converting data to a PyKX format allows for easy interaction with these objects using q or the analytic functionality provided by PyKX. However, this format may not be suitable for all use cases. For instance, if a function requires a Pandas DataFrame as input, a PyKX object must be converted to a Pandas DataFrame.

Once the data is ready for use in Python, it may be more appropriate to convert it into a representation using Python, NumPy, Pandas, PyArrow, or PyTorch (Beta) by using the following methods:

| **Method**      | **Description**                  |
|-----------------|----------------------------------|
| `*.py()`        | Convert a PyKX object to Python  |
| `*.np()`        | Convert a PyKX object to Numpy   |
| `*.pd()`        | Convert a PyKX object to Pandas  |
| `*.pa()`        | Convert a PyKX object to PyArrow |
| `*.pt()` (Beta) | Convert a PyKX object to PyTorch |
    
??? example "Example"

	```python
	>>> import os
	>>> os.environ['PYKX_BETA_FEATURES'] = 'True'
	>>> import pykx as kx
	>>> qarr = kx.q('til 5')
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
	>>> qarr.pt()
	tensor([0, 1, 2, 3, 4])
	>>>
	>>> qtab = kx.Table(data={
	...     'x': kx.random.random(5, 1.0),
	...     'x1': kx.random.random(5, 1.0),
	... })
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

!!! warning "Precision loss considerations"

	Special care is needed when converting q temporal data to Python native data types. Since Python temporal data types only support microsecond precision, roundtrip conversions reduce the temporal granularity of q data.

	```python
	>>> import pykx as kx
	>>> qtime = kx.TimestampAtom('now')
	>>> qtime
	pykx.TimestampAtom(pykx.q('2024.01.05D03:16:23.736627552'))
	>>> kx.toq(qtime.py())
	pykx.TimestampAtom(pykx.q('2024.01.05D03:16:23.736627000'))
	```

	See our [Conversion considerations for temporal types](../fundamentals/conversion_considerations.md#temporal-data-types) section for further details.

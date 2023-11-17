# PyKX Changelog

!!! Note

	The changelog presented here outlines changes to PyKX when operating within a Python environment specifically, if you require changelogs associated with PyKX operating under a q environment see [here](./underq-changelog.md).

## PyKX 2.2.0

#### Release Date

2023-11-09

!!! Warning

	PyKX 2.2.0 presently does not include a Python 3.11 release for MacOS x86 and Linux x86 architectures, this will be rectified in an upcoming patch release.

### Additions

- Addition of `agg` method for application of aggregation functions on `pykx.Table` and `pykx.GroupbyTable` objects

	```python
	>>> import pykx as kx
	>>> import numpy as np
	>>> import statistics
        >>> def mode(x):
	...     return statistics.mode(x)
	>>> tab = kx.Table(data={
	...     'x': kx.random.random(1000, 10),
	...     'x1': kx.random.random(1000, 10.0)})
	>>> tab.agg(mode)
	pykx.Dictionary(pykx.q('
	x | 6
	x1| 2.294631
	'))
	>>> tab.agg(['min', 'mean'])
	pykx.KeyedTable(pykx.q('
	function| x     x1         
	--------| -----------------
	min     | 0     0.009771725
	mean    | 4.588 5.152194   
	'))
	>>> 
	>>> group_tab = kx.Table(data={
	...     'x': kx.random.random(1000, ['a', 'b']),
	...     'y': kx.random.random(1000, 10.0)})
	>>> group_tab.groupby('x').agg('mean')
	pykx.KeyedTable(pykx.q('
	x| y       
	-| --------
	a| 5.239048
	b| 4.885599
	'))
	>>> group_tab.groupby('x').agg(mode)
	pykx.KeyedTable(pykx.q('
	x| y       
	-| --------
	a| 1.870281
	b| 4.46898 
	'))
	```

- Addition of the ability for users to run `min`, `max`, `mean`, `median`, `sum` and `mode` methods on vector objects within PyKX.

	```python
	>>> import pykx as kx
	>>> random_vec = kx.random.random(5, 3, seed=20)
	pykx.LongVector(pykx.q('0 1 0 1 1'))
	>>> random_vec.mode()
	pykx.LongVector(pykx.q(',1'))
	>>> random_vec.mean()
	pykx.FloatAtom(pykx.q('0.6'))
	```

- Addition of the ability for users to assign objects to `pykx.*Vector` and `pykx.List` objects

	```python
	>>> import pykx as kx
	>>> qvec = kx.q.til(10)
	>>> qvec
	pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
	>>> qvec[3] = 45
	>>> qvec
	pykx.LongVector(pykx.q('0 1 2 45 4 5 6 7 8 9'))
	>>> qvec[-1] = 20
	>>> qvec
	pykx.LongVector(pykx.q('0 1 2 45 4 5 6 7 8 20'))
	```

- Users can now assign/update keys of a `pykx.Dictionary` object using an in-built `__setitem__` method as follows

	```python
	>>> import pykx as kx
	>>> pykx_dict = kx.toq({'x': 1})
	>>> pykx_dict
	pykx.Dictionary(pykx.q('x| 1'))
	>>> pykx_dict['x1'] = 2
	>>> pykx_dict
	pykx.Dictionary(pykx.q('
	x | 1
	x1| 2
	'))
	>>> for i in range(3):
	...     pykx_dict['x']+=i
	...
	>>> pykx_dict
	pykx.Dictionary(pykx.q('
	x | 4
	x1| 2
	'))
	```

- Addition of `null` and `inf` properties for `pykx.Atom` objects allowing for Pythonic retrieval of nulls and infinities

	```python
	>>> import pykx as kx
	>>> kx.FloatAtom.null
	pykx.FloatAtom(pykx.q('0n'))
	>>> kx.GUIDAtom.null
	pykx.GUIDAtom(pykx.q('00000000-0000-0000-0000-000000000000'))
	>>> kx.IntAtom.inf
	pykx.IntAtom(pykx.q('0Wi'))
	>>> -kx.IntAtom.inf
	pykx.IntAtom(pykx.q('-0Wi'))
	```

- Users can now use the environment variables `PYKX_UNLICENSED="true"` or `PYKX_LICENSED="true"` set this as part of configuration within their `.pykx-config` file to allow `unlicensed` or `licensed` mode to be the default behaviour on initialisation for example:

	```python
	>>> import os
	>>> os.environ['PYKX_UNLICESED'] = "true"
	>>> import pykx as kx
	>>> kx.toq([1, 2, 3])
	pykx.List._from_addr(0x7fee46000a00)
	```

- Addition of `append` and `extend` methods to `pykx.*Vector` and `pykx.List` objects

	```python
	>>> import pykx as kx
	>>> qvec = kx.q.til(5)
	>>> qvec.append(100)
	>>> qvec
	pykx.LongVector(pykx.q('0 1 2 3 4 100'))
	>>> qvec.extend([1, 2, 3])
	>>> qvec
	pykx.LongVector(pykx.q('0 1 2 3 4 100 1 2 3'))
	```

- Addition of `debug` keyword argument to the `__call__` method on `EmbeddedQ` and `QConnection` objects to provide backtraces on `q` code.

	```python
	>>> import pykx as kx
	>>> kx.q('{[x] a: 5; b: til a; c: til x; b,c}', b'foo', debug=True)
	backtrace:
	  [3]  (.q.til)

	  [2]  {[x] a: 5; b: til a; c: til x; b,c}
	                               ^
	  [1]  (.Q.trp)

	      [0]  {[pykxquery] .Q.trp[value; pykxquery; {2@"backtrace:
	                    ^
	",.Q.sbt y;'x}]}
	Traceback (most recent call last):
	  File "<stdin>", line 1, in <module>
	  File "...\site-packages\pykx\embedded_q.py", line 226, in __call__
	    return factory(result, False)
	  File "pykx\\_wrappers.pyx", line 504, in pykx._wrappers._factory
	  File "pykx\\_wrappers.pyx", line 497, in pykx._wrappers.factory
	pykx.exceptions.QError: type
	```

- Added feature to extract individual elements of both `TimestampAtom` and `TimestampVector` in a pythonic way including:
  
	* `date` - DateAtom / DateVector
	* `time` - TimeAtom / TimeVector
	* `year` - IntAtom / IntVector
	* `month` - IntAtom / IntVector
	* `day` - IntAtom / IntVector
	* `hour` - IntAtom / IntVector
	* `minute` - IntAtom / IntVector
	* `second` - IntAtom / IntVector

	```python
	>>> timestamp_atom = kx.q('2023.10.25D16:42:01.292070013')
	
	>>> timestamp_atom.time
	pykx.TimeAtom(pykx.q('16:42:01.292'))
	>>> timestamp_atom.date
	pykx.DateAtom(pykx.q('2023.10.25'))
	>>> timestamp_atom.minute
	pykx.IntAtom(pykx.q('42i'))

	>>> timestamp_atom_2 = kx.q('2018.11.09D12:21:08.456123789')
	>>> timestamp_vector = kx.q('enlist', timestamp_atom, timestamp_atom_2)
	
	>>> timestamp_vector.time
	pykx.TimeVector(pykx.q('16:42:01.292 12:21:08.456'))
	>>> timestamp_vector.date
	pykx.DateVector(pykx.q('2023.10.25 2018.11.09'))
	>>> timestamp_vector.hour
	pykx.IntVector(pykx.q('16 12i'))
	```

- Addition of `poll_recv_async` to `RawQConnection` objects to support asynchronous polling.

### Fixes and Improvements

- Fix to allow users to use Python functions when operating on a `pykx.GroupbyTable` with an `apply` function

	```python
	>>> import pykx as kx
	>>> import statistics
	>>> def mode(x):
	...    return statistics.mode(x)
	>>> tab = kx.q('([]sym:`a`b`a`a;1 1 0 0)')
	>>> tab.groupby('sym').apply(mode)
	pykx.KeyedTable(pykx.q('
	sym| x
	---| -
	a  | 0
	b  | 1
	'))
	```

- Added debug dependency for `find-libpython` that can be installed using `pip install "pykx[debug]"`. This dependency can be used to help find `libpython` in the scenario that `pykx.q` fails to find it.
- Usage of the `QARGS` to enable/disable various elements of kdb Insights functionality has been formalised, outlined [here](../user-guide/configuration.md). For example users can now use `QARGS="--no-objstor"` to disable object storage capabilities.

- Failure to initialise PyKX with `exp` or `embedq` license errors will now prompt users to ask if they wish to download an appropriate license following expiry or use of an invalid license

	=== "'exp' License Prompt"

		```python
		Your PyKX license has now expired.

		Captured output from initialization attempt:
		    '2023.10.18T13:27:59.719 licence error: exp

		Would you like to renew your license? [Y/n]:
		```

	=== "'embedq' License Prompt"

		```python
		You appear to be using a non kdb Insights license.

		Captured output from initialization attempt:
		    '2023.10.18T13:27:59.719 licence error: embedq

		Running PyKX in the absence of a kdb Insights license has reduced functionality.
		Would you like to install a kdb Insights personal license? [Y/n]:
		```

	=== "'upd' License Prompt"

		```python
		Your installed license is out of date for this version of PyKX and must be updated.

		Captured output from initialization attempt:
		    '2023.10.18T13:27:59.719 licence error: upd

		Would you like to install an updated kdb Insights personal license? [Y/n]:
		```

- PyKX sets `PYKX_EXECUTABLE` to use when loading embedded q to prevent errors if launched using a different Python executable than that which will be found in `PATH`

- Jupyter Notebook:
	- Removal of `FutureWarning` when displaying tables and dictionaries.
	- Revert issue causing results to be displayed as pointer references rather than Python objects in unlicensed mode.
	- `%%q` magic now suppresses displaying of `::`. 
	- `%%q` magic addition of `--display` option to have `display` be called on returned items in place of the default `print`.

- `PyKXReimport` now additionally unsets/resets: `PYKX_SKIP_UNDERQ`, `PYKX_EXECUTABLE`, `PYKX_DIR`
- When attempting to deserialize unsupported byte representations `pykx.deserialize` would result in a segmentation fault, this has been updated such that an error message is now raised.

	```python
	>>> import pykx as kx
	>>> kx.deserialize(b'invalid byte string')
	Traceback (most recent call last):
	  File "<stdin>", line 1, in <module>
	  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/serialize.py", line 123, in deserialize
	    return _deserialize(data)
	  File "pykx/_wrappers.pyx", line 131, in pykx._wrappers.deserialize
	  File "pykx/_wrappers.pyx", line 135, in pykx._wrappers.deserialize
	pykx.exceptions.QError: Failed to deserialize supplied non PyKX IPC serialized format object
	```

- Fixed an issue when using multiple asynchronous `QConnection` connected to multiple servers.
- Users can now access the length of and index into `pykx.CharAtom` objects to align with Pythonic equivalent data

	```python
	>>> qatom = kx.CharAtom('a')
	>>> len(qatom)
	1
	>>> qatom[0]
	pykx.CharAtom(pykx.q('"a"'))
	```

## PyKX 2.1.2

#### Release Date

2023-10-24

### Fixes and Improvements

- Fix to issue where functions retrieved using the Context Interface with names `update/delete/select/exec` would result in an `AttributeError`

	=== "Behavior prior to change"

		```python
		>>> import pykx as kx
		>>> kx.q.test
		<pykx.ctx.QContext of .test with [ctx]>
		>>> kx.q.test.ctx.update(1)
		Traceback (most recent call last):
		  File "<stdin>", line 1, in <module>
		  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/ctx.py", line 121, in __getattr__
		    raise AttributeError(f'{key}: {self._unsupported_keys_with_msg[key]}')
		AttributeError: update: Usage of 'update' function directly via 'q' context not supported, please consider using 'pykx.q.qsql.update'
		```

	=== "Behavior post change"

		```python
		>>> import pykx as kx
		>>> kx.q.test
		<pykx.ctx.QContext of .test with [ctx]>
		>>> kx.q.test.ctx.update(1)
		pykx.LongAtom(pykx.q('2'))
		```

## PyKX 2.1.1

#### Release Date

2023-10-10

### Fixes and Improvements

- Fix to regression in PyKX 2.1.0 where execution of `from pykx import *` would result in the following behaviour

	```
	>>> from pykx import *
	...
	AttributeError: module 'pykx' has no attribute 'PyKXSerialized'
	```

## PyKX 2.1.0

#### Release Date

2023-10-09

### Additions

- Added functionality to the CSV Reader to allow for the input of data structures while defining column types. For example,
the following reads a CSV file and specifies the types of the three columns named `x1`, `x2` and `x3` to be of type `Integer`, `GUID` and `Timestamp`.

	```python
	>>> table = q.read.csv('example.csv', {'x1':kx.IntAtom,'x2':kx.GUIDAtom,'x3':kx.TimestampAtom})
	```

- Conversions from Pandas Dataframes and PyArrow tables using `pykx.toq` can now specify the `ktype` argument as a dictionary allowing selective type conversions for defined columns

	```python
	>>> import pykx as kx
	>>> import pandas as pd
	>>> df = pd.DataFrame.from_dict({'x': [1, 2], 'y': ['a', 'b']})
	>>> kx.toq(df).dtypes
	pykx.Table(pykx.q('
	columns type
	-----------------------
	x       "kx.LongAtom"
	y       "kx.SymbolAtom"
	'))
	>>> kx.toq(df, ktype={'x': kx.FloatAtom}).dtypes
	pykx.Table(pykx.q('
	columns type
	-----------------------
	x       "kx.FloatAtom"
	y       "kx.SymbolAtom"
	'))
	```

- Addition of the ability for users to run an `apply` method on vector objects within PyKX allowing the application of Python/PyKX functionality on these vectors directly

	```python
	>>> import pykx as kx
	>>> random_vec = kx.random.random(2, 10.0, seed=100)
	>>> random_vec
	pykx.FloatVector(pykx.q('8.909647 3.451941'))
	>>> random_vec.apply(lambda x:x+1)
	pykx.FloatVector(pykx.q('9.909647 4.451941'))
	>>> def func(x, y):
	...     return x+y
	>>> random_vec.apply(func, y=2)
	pykx.FloatVector(pykx.q('10.909647 5.451941'))
	```

- Notebooks will HTML print tables and dictionaries through the addition of `_repr_html_`. Previous `q` style output is still available using `print`.
- Added [`serialize` and `deserialize`](../api/serialize.html) as base methods to assist with the serialization of `K` objects for manual use over IPC.
- Added support for `pandas` version `2.0`.

!!! Warning "Pandas 2.0 has deprecated the `datetime64[D/M]` types."

    Due to this change it is not always possible to determine if the resulting q Table should
    use a `MonthVector` or a `DayVector`. In the scenario that it is not possible to determine
    the expected type a warning will be raised and the `DayVector` type will be used as a
    default.

### Fixes and Improvements

- Empty PyKX keyed tables can now be converted to Pandas DataFrames, previously this would raise a `ValueError`

	```python
	>>> import pykx as kx
	>>> df = kx.q('0#`a xkey ([]a:1 2 3;b:3 4 5)').pd()
	>>> df
	Empty DataFrame
	Columns: [b]
	Index: []
	>>> df.index.name
	'a'
	>>> kx.toq(df)
	pykx.KeyedTable(pykx.q('
	a| b
	-| -
	'))
	```

- Fix to issue introduced in 2.0.0 where indexing of `pykx.Table` returned incorrect values when passed negative/out of range values

	=== "Behavior prior to change"

		```python
		>>> import pykx as kx
		>>> tab = kx.Table(data={"c1": list(range(3))})
		>>> tbl[-1]
		pykx.Table(pykx.q('
		c1
		--

		'))
		>>> tab[-4]
		pykx.Table(pykx.q('
		c1
		--

		'))
		>>> tab[3]
		pykx.Table(pykx.q('
		c1
		--

		'))
		```

	=== "Behavior post change"

		```python
		>>> import pykx as kx
		>>> tab = kx.Table(data={"c1": list(range(3))})
		>>> tab[-1]
		pykx.Table(pykx.q('
		c1
		--
		2
		'))
		>>> tab[-4]
		...
		IndexError: index out of range
		>>> tab[3]
		...
		IndexError: index out of range
		```

- Fix to issue where PyKX would not initialize when users with a [`QINIT`](https://code.kx.com/q/basics/by-topic/#environment) environment variable set which pointed to a file contained a `show` statement
- Retrieval of `dtypes` with tables containing `real` columns will now return `kx.RealAtom` for the type rather than incorrectly returning `kx.ShortAtom`
- Users with [`QINIT`](https://code.kx.com/q/basics/by-topic/#environment) environment variable would previously load twice on initialization within PyKX
- Users installing PyKX under q on Windows had been missing installation of required files using `pykx.install_into_QHOME()`

### Dependency Updates

- The version of `Cython` used to build `PyKX` was updated to the full `3.0.x` release version.

## PyKX 2.0.1

#### Release Date

2023-09-21

### Fixes and Improvements

- User input based license initialization introduced in 2.0.0 no longer expects user input when operating in a non-interactive modality, use of PyKX in this mode will revert to previous behavior
- Use of the environment variables `QARGS='--unlicensed'` or `QARGS='--licensed'` operate correctly following regression in 2.0.0
- Fix to issue where `OSError` would be raised when `close()` was called on an IPC connection which has already disconnected server side

## PyKX 2.0.0

#### Release Date

2023-09-18

- PyKX 2.0.0 major version increase is required due to the following major changes which are likely to constitute breaking changes
	- Pandas API functionality is enabled permanently which will modify data indexing and retrieval of `pykx.Table` objects. Users should ensure to review and test their codebase before upgrading.
	- EmbedPy replacement functionality for PyKX under q is now non-beta for Linux and MacOS installations, see [here](underq-changelog.md) for full information on 2.0.0 changelog.

### Additions

- [Pandas API](../user-guide/advanced/Pandas_API.ipynb) is enabled by default allowing users to treat PyKX Tables similarly to Pandas Dataframes for a limited subset of Pandas like functionality. As a result of this change the environment variable `PYKX_ENABLE_PANDAS_API` is no longer required.
- Addition of file based configuration setting allowing users to define profiles for various PyKX modalities through definition of the file `.pykx.config` see [here](../user-guide/configuration.md) for more information.
- Addition of new PyKX license installation workflow for users who do not have a PyKX license allowing for installation of personal licenses via a form based install process. This updated flow is outlined [here](../getting-started/installing.md).
- Addition of a new module `pykx.license` which provides functionality for the installation of licenses, checking of days to expiry and validation that the license which PyKX is using matches the file/base64 string the user expects. For more information see [here](../api/license.md).


- Addition of `apply` and `groupby` methods to PyKX Tables allowing users to perform additional advanced analytics for example:

	```python
	>>> import pykx as kx
	>>> N = 1000000
	>>> tab = kx.Table(data = {
	...       'price': kx.random.random(N, 10.0),
	...       'sym': kx.random.random(N, ['a', 'b', 'c'])
	...       })
	>>> tab.groupby('sym').apply(kx.q.sum)
	pykx.KeyedTable(pykx.q('
	sym| price
	---| --------
	a  | 166759.4
	b  | 166963.6
	c  | 166444.1
	'))
	```

- Addition of a new module `pykx.random` which provides functionality for the generation of random data and setting of random seeds. For more information see [here](../api/random.md)

	```python
	>>> import pykx as kx
	>>> kx.random.random(5, 1.0, seed=123)
	pykx.FloatVector(pykx.q('0.1959057 0.06460555 0.9550039 0.4991214 0.3207941'))
	>>> kx.random.seed(123)
	>>> kx.random.random(5, 1.0)
	pykx.FloatVector(pykx.q('0.1959057 0.06460555 0.9550039 0.4991214 0.3207941'))
	>>> kx.random.random([3, 4], ['a', 'b', 'c'])
	pykx.List(pykx.q('
	b c a b
	b a b a
	a a a a
	'))
	```

- Addition of a new module `pykx.register` which provides functionality for the addition of user specified type conversions for Python objects to q via the function `py_toq` for more information see [here](../api/pykx-q-data/register.md). The following is an example of using this function

	```python
	>>> import pykx as kx
	>>> def complex_conversion(data):
	...     return kx.q([data.real, data.imag])
	>>> kx.register.py_toq(complex, complex_conversion)
	>>> kx.toq(complex(1, 2))
	pykx.FloatVector(pykx.q('1 2f'))
	```

- Support for fixed length string dtype with Numpy arrays

	```python
	>>> import pykx as kx
	>>> import numpy as np
	>>> kx.toq(np.array([b'string', b'test'], dtype='|S7'))
	pykx.List(pykx.q('
	"string"
	"test"
	'))
	```

### Fixes and Improvements

- Update to environment variable definitions in all cases to be prefixed with `PYKX_*`
- Return of Pandas API functions `dtypes`, `columns`, `empty`, `ndim`, `size` and `shape` return `kx` objects rather than Pythonic objects
- Removed GLIBC_2.34 dependency for conda installs
- Removed the ability for users to incorrectly call `pykx.q.{select/exec/update/delete}` with error message now suggesting usage of `pykx.q.qsql.{function}`
- Fixed behavior of `loc` when used on `KeyedTable` objects to match the pandas behavior.
- Addition of warning on failure to link the content of a users `QHOME` directory pointing users to documentation for warning suppression
- Update to PyKX foreign function handling to support application of Path objects as first argument i.e. ```q("{[f;x] f x}")(lambda x: x)(Path('test'))```
- SQL interface will attempt to automatically load on Windows and Mac
- Attempts to serialize `pykx.Foreign`, `pykx.SplayedTable` and `pykx.PartitionedTable` objects will now result in a type error fixing a previous issue where this could result in a segmentation fault.
- Messages mistakenly sent to a PyKX client handle are now gracefully ignored.
- Application of Pandas API `dtypes` operations return a table containing `column` to `type` mappings with `PyKX` object specific types rather than Pandas/Python types

	=== "Behavior prior to change"

		```python
		>>> table = kx.Table([[1, 'a', 2.0, b'testing', b'b'], [2, 'b', 3.0, b'test', b'a']])
		>>> print(table)
		x x1 x2 x3        x4
		--------------------
		1 a  2  "testing" b
		2 b  3  "test"    a
		>>> table.dtypes
		x       int64
		x1     object
		x2    float64
		x3     object
		x4        |S1
		dtype: object
		```

	=== "Behavior post change"

		```python
		>>> table = kx.Table([[1, 'a', 2.0, b'testing', b'b'], [2, 'b', 3.0, b'test', b'a']])
		>>> print(table)
		x x1 x2 x3        x4
		--------------------
		1 a  2  "testing" b
		2 b  3  "test"    a
		>>> table.dtypes
		pykx.Table(pykx.q('
		columns type
		-----------------------
		x       "kx.LongAtom"
		x1      "kx.SymbolAtom"
		x2      "kx.FloatAtom"
		x3      "kx.CharVector"
		x4      "kx.CharAtom"
		'))
		```

- Fixed an issue where inequality checks would return `False` incorrectly

	=== "Behavior prior to change"

		```python
		>>> import pykx as kx
		>>> kx.q('5') != None
		pykx.q('0b')
		```

	=== "Behavior post change"

		```python
		>>> import pykx as kx
		>>> kx.q('5') != None
		pykx.q('1b')
		```

### Breaking Changes

- Pandas API functionality is enabled permanently which will modify data indexing and retrieval. Users should ensure to review and test their codebase before upgrading.

## PyKX 1.6.3

#### Release Date

2023-08-18

### Additions

- Addition of argument `return_info` to `pykx.util.debug_environment` allowing user to optionally return the result as a `str` rather than to stdout

### Fixes and Improvements

- Fixed Pandas API use of `ndim` functionality which should return `2` when interacting with tables following the expected Pandas behavior.
- Fixed an error when using the Pandas API to update a column with a `Symbols`, `Characters`, and `Generic Lists`.
- Prevent attempting to pass wrapped Python functions over IPC.
- Support IPC payloads over 4GiB.

## PyKX 1.6.2

#### Release Date

2023-08-15

### Additions

- Added `to_local_folder` kwarg to `install_into_QHOME` to enable use of `pykx.q` without write access to `QHOME`.
- Added [an example](../examples/threaded_execution/README.md) that shows how to use `EmbeddedQ` in a multithreaded context where the threads need to modify global state.
- Added [PYKX_NO_SIGINT](../user-guide/configuration.md#environment-variables) environment variable.

### Fixes and Improvements

- Fixed an issue causing a crash when closing `QConnection` instances on Windows.
- Updated q 4.0 libraries to 2023.08.11. Note: Mac ARM release remains on 2022.09.30.
- Fix [Jupyter Magic](../getting-started/q_magic_command.ipynb) in local mode.
- Fix error when binding with [FFI](https://github.com/KxSystems/ffi) in `QINIT`.
- Fix issue calling `peach` with `PYKX_RELEASE_GIL` set to true when calling a Python function.

## PyKX 1.6.1

#### Release Date

2023-07-19

### Additions

- Added `sorted`, `grouped`, `parted`, and `unique`. As methods off of `Tables` and `Vectors`.
- Added `PyKXReimport` class to allow subprocesses to reimport `PyKX` safely.
    - Also includes `.pykx.safeReimport` in `pykx.q` to allows this behavior when running under q as well.
- Added environment variables to specify a path to `libpython` in the case `pykx.q` cannot find it.

### Fixes and Improvements

- Fixed memory leaks within the various `QConnection` subclasses.
- Added deprecation warning around the discontinuing of support for Python 3.7.
- Fixed bug in Jupyter Notebook magic command.
- Fixed a bug causing `np.ndarray`'s to not work within `ufuncs`.
- Fixed a memory leak within all `QConnection` subclasses. Fixed for both `PyKX` as a client and as a server.
- Updated insights libraries to 4.0.2
- Fixed `pykx.q` functionality when run on Windows.
- Fixed an issue where reimporting `PyKX` when run under q would cause a segmentation fault.
- Updated the warning message for the insights core libraries failing to load to make it more clear that no error has occurred.

## PyKX 1.6.0

#### Release Date

2023-06-16

### Additions

- Added `merge_asof` to the Pandas like API.
    - See [here](../user-guide/advanced/Pandas_API.ipynb#tablemerge_asof) for details of supported keyword arguments and limitations.
- Added `set_index` to the Pandas like API.
    - See [here](../user-guide/advanced/Pandas_API.ipynb##setting-indexes) for details of supported keyword arguments and limitations.
- Added a set of basic computation methods operating on tabular data to the Pandas like API. See [here](../user-guide/advanced/Pandas_API.ipynb#computations) for available methods and examples.
- `pykx.util.debug_environment` added to help with import errors.
- q vector type promotion in licensed mode.
- Added `.pykx.toraw` to `pykx.q` to enable raw conversions (e.g. `kx.toq(x, raw=True)`)
- Added support for Python `3.11`.
    - Support for PyArrow in this python version is currently in Beta.
- Added the ability to use `kx.RawQConnection` as a Python based `q` server using `kx.RawQConnection(port=x, as_server=True)`.
    - More documentation around using this functionality can be found [here](../examples/server/server.md).

### Fixes and Improvements

- Improved error on Windows if `msvcr100.dll` is not found
- Updated q libraries to 2023.04.17
- Fixed an issue that caused `q` functions that shared a name with python key words to be inaccessible using the context interface.
    - It is now possible to access any `q` function that uses a python keyword as its name by adding an underscore to the name (e.g. `except` can now be accessed using `q.except_`).
- Fixed an issue with `.pykx.get` and `.pykx.getattr` not raising errors correctly.
- Fixed an issue where `deserializing` data would sometimes not error correctly.
- Users can now add new column(s) to an in-memory table using assignment when using the Pandas like API.

	```python
	>>> import os
	>>> os.environ['PYKX_ENABLE_PANDAS_API'] = 'true'
	>>> import pykx as kx
	>>> import numpy as np
	>>> tab = kx.q('([]100?1f;100?1f)')
	>>> tab['x2'] = np.arange(0, 100)
	>>> tab
	pykx.Table(pykx.q('
	x           x1         x2
	-------------------------
	0.1485357   0.1780839  0
	0.4857547   0.3017723  1
	0.7123602   0.785033   2
	0.3839461   0.5347096  3
	0.3407215   0.7111716  4
	0.05400102  0.411597   5
	..
	'))
	```

## PyKX 1.5.3

#### Release Date

2023-05-18

### Additions

- Added support for Pandas `Float64Index`.
- Wheels for ARM64 based Macs are now available for download.

## PyKX 1.5.2

#### Release Date

2023-04-30

### Additions

- Added support for ARM 64 Linux.

## PyKX 1.5.1

#### Release Date

2023-04-28

### Fixes and Improvements

- Fixed an issue with `pykx.q` that caused errors to not be raised properly under q.
- Fixed an issue when using `.pykx.get` and `.pykx.getattr` that caused multiple calls to be made.

## PyKX 1.5.0

#### Release Date

2023-04-17

### Additions

- Added wrappers around various `q` [system commands](https://code.kx.com/q/basics/syscmds/).
- Added `merge` method to tables when using the `Pandas API`.
- Added `mean`/`median`/`mode` functions to tables when using the `Pandas API`.
- Added various functions around type conversions on tables when using the `Pandas API`.

### Fixes and Improvements

- Fix to allow GUIDs to be sent over IPC.
- Fix an issue related to IPC connection using compression.
- Improved the logic behind loading `pykx.q` under a `q` process allowing it to run on MacOS and Linux in any environment that `EmbedPy` works in.
- Fix an issue that cause the default handler for `SIGINT` to be overwritten.
- `pykx.toq.from_callable` returns a `pykx.Composition` rather than `pykx.Lambda`. When executed returns an unwrapped q object.
- Fixed conversion of Pandas Timestamp objects.
- Fixed an issue around the `PyKX` `q` magic command failing to load properly.
- Fixed a bug around conversions of `Pandas` tables with no column names.
- Fixed an issue around `.pykx.qeval` not returning unwrapped results in certain scenarios.

## PyKX 1.4.2

#### Release Date

2023-03-08

### Fixes and Improvements

- Fixed an issue that would cause `EmbeddedQ` to fail to load.

## PyKX 1.4.1

#### Release Date

2023-03-06

### Fixes and Improvements

- Added constructors for `Table` and `KeyedTable` objects to allow creation of these objects from dictionaries and list like objects.
- Fixed a memory leak around calling wrapped `Foreign` objects in `pykx.q`.
- Fixed an issue around the `tls` keyword argument when creating `QConnection` instances, as well as a bug in the unlicensed behavior of `SecureQConnection`'s.

## PyKX 1.4.0

#### Release Date

2023-01-23

### Additions

- Addition of a utility function `kx.ssl_info()` to retrieve the SSL configuration when running in unlicensed mode (returns the same info as kx.q('-26!0') with a license).
- Addition of a utility function `kx.schema.builder` to allow for the generation of `pykx.Table` and `pykx.KeyedTable` types with a defined schema and zero rows, this provides an alternative to writing q code to create an empty table.
- Added helper functions for inserting and upserting to `k.Table` instances. These functions provide new keyword arguments to run a test insert against the table or to enforce that the schema of the new row matches the existing table.
- Added environment variable `PYKX_NOQCE=1` to skip the loading of q Cloud Edition in order to speed up the import of PyKX.
- Added environment variable `PYKX_LOAD_PYARROW_UNSAFE=1` to import PyArrow without the "subprocess safety net" which is here to prevent some hard crashes (but is slower than a simple import).
- Addition of method `file_execute` to `kx.QConnection` objects which allows the execution of a local `.q` script on a server instance as outlined [here](../user-guide/advanced/ipc.md#file_execution).
- Added `kx.RawQConnection` which extends `kx.AsyncQConnection` with extra functions that allow a user to directly poll the send and receive selectors.
- Added environment variable `PYKX_RELEASE_GIL=1` to drop the [`Python GIL`](https://wiki.python.org/moin/GlobalInterpreterLock) on calls into embedded q.
- Added environment variable `PYKX_Q_LOCK=1` to enable a Mutex Lock around calls into q, setting this environment variable to a number greater than 0 will set the max length in time to block before raising an error, a value of '-1' will block indefinitely and will not error, any other value will cause an error to be raised immediately if the lock cannot be acquired.
- Added `insert` and `upsert` methods to `Table` and `KeyedTable` objects.

### Fixes and Improvements

- Fixed `has_nulls` and `has_infs` properties for subclasses of `k.Collection`.
- Improved error output of `kx.QConnection` objects when an error is raised within the context interface.
- Fixed `.py()` conversion of nested `k.Dictionary` objects and keyed `k.Dictionary` objects.
- Fixed unclear error message when querying a `QConnection` instance that has been closed.
- Added support for conversions of non C contiguous Numpy arrays.
- Fixed conversion of null `GUIDAtom`'s to and from Numpy types.
- Improved performance of converting `q` enums to pandas Categoricals.

### Beta Features

- Added support for a Pandas like API around `Table` and `KeyedTable` instances, documentation for the specific functionality can be found [here](../user-guide/advanced/Pandas_API.ipynb).
- Added `.pykx.setdefault` to `pykx.q` which allows the default conversion type to be set without using environment variables.

## PyKX 1.3.2

#### Release Date

2023-01-06

### Features and Fixes

- Fixed support for using TLS with `SyncQConnection` instances.

## PyKX 1.3.1

#### Release Date

2022-11-16

### Features and Fixes

- Added environment variable `PYKX_Q_LIB_LOCATION` to specify a path to load the PyKX q libraries from.
    - Required files in this directory
        - If you are using the kdb+/q Insights core libraries they all must be present within this folder.
        - The `read.q`, `write.q`, and `csvutil.q` libraries that are bundled with PyKX.
        - A `q.k` that matches the version of `q` you are loading.
        - There must also be a subfolder (`l64` / `m64` / `w64`) based on the platform you are using.
            - Within this subfolder a copy of these files must also be present.
                - `libq.(so / dylib)` / `q.dll`.
                - `libe.(so / dylib)` / `e.dll`.
                - If using the Insights core libraries their respective shared objects must also be present here.
- Updated core q libraries
    - PyKX now supports M1 Macs
    - OpenSSLv3 support
- Added ability to specify maximum length for IPC error messages. The default is 256 characters and this can be changed by setting the `PYKX_MAX_ERROR_LENGTH` environment variable.

## PyKX 1.3.0

#### Release Date

2022-10-20

### Features and Fixes

- Support for converting `datetime.datetime` objects with time zone information into `pykx.TimestampAtom`s and `pykx.TimestampVector`s.
- Added a magic command to run cells of q code in a Jupyter Notebook. The addition of `%%q` at the start of a Jupyter Notebook cell will allow a user to execute q code locally similarly to loading a q file.
- Added `no_ctx` key word argument to `pykx.QConnection` instances to disable sending extra queries to/from q to manage the context interface.
- Improvements to SQL interface for PyKX including the addition of support for prepared statements, execution of these statements and retrieval of inputs see [here](../api/query.md#pykx.query.SQL) for more information.
- Fix to memory leak seen when converting Pandas Dataframes to q tables.
- Removed unnecessary copy when sending `q` objects over IPC.

### Beta Features

- EmbedPy replacement functionality `pykx.q` updated significantly to provide parity with embedPy from a syntax perspective. Documentation of the interface [here](../pykx-under-q/intro.md) provides API usage. Note that initialization requires the first version of Python to be retrieved on a users `PATH` to have PyKX installed. Additional flexibility with respect to installation location is expected in `1.4.0` please provide any feedback to `pykx@kx.com`

## PyKX 1.2.2

#### Release Date

2022-10-01

### Features and Fixes

- Fixed an issue causing the timeout argument for `QConnection` instances to not work properly.

## PyKX 1.2.1

#### Release Date

2022-09-27

### Features and Fixes

- Added support for OpenSSLv3 for IPC connections created when in 'licensed' mode.
- Updated conversion functionality for timestamps to support conversions within Pandas 1.5.0

## PyKX 1.2.0

#### Release Date

2022-09-01

### Features and Fixes

- Support for converting any python type to a `q` Foreign object has been added.
- Support for converting Pandas categorical types into `pykx.EnumVector` type objects.
- Support for q querying against Pandas/PyArrow tables through internal conversion to q representation and subsequent query. `kx.q.qsql.select(<pd.DataFrame>)`
- Support for casting Python objects prior to converting into K objects. (e.g. `kx.IntAtom(3.14, cast=True)` or `kx.toq("3.14", ktype=kx.FloatAtom, cast=True)`).
- Support usage of Numpy [`__array_ufunc__`'s](https://numpy.org/doc/stable/reference/ufuncs.html) directly on `pykx.Vector` types.
- Support usage of Numpy `__array_function__`'s directly on `pykx.Vector` types (Note: these will return a Numpy ndarray object not an analogous `pykx.K` object).
- Improved performance of `pykx.SymbolVector` conversion into native Python type (e.g. `.py()` conversion for `pykx.SymbolVector`'s).
- Improved performance and memory usage of various comparison operators between `K` types.
- Improved performance of various `pykx.toq` conversions.
- `pykx.Vector` types will now automatically enlist atomic types instead of erroring.
- Fixed conversions of Numpy float types into `pykx.FloatAtom` and `pykx.RealAtom` types.
- Fixed conversion of `None` Python objects into analogous null `K` types if a `ktype` is specified.
- Added `event_loop` parameter to `pykx.AsyncQConnection` that takes a running event loop as a parameter and allows the event loop to manage `pykx.QFuture` objects.

### Beta Features

- Added extra functionality to `pykx.q` related to the calling and use of python foreign objects directly within a `q` process.
- Support for [NEP-49](https://numpy.org/neps/nep-0049.html), which allows Numpy arrays to be converted into `q` Vectors without copying the underlying data. This behavior is opt-in and you can do so by setting the environment variable `PYKX_ALLOCATOR` to 1, "1" or True or by adding the flag `--pykxalloc` to the `QARGS` environment variable. Note: This feature also requires a python version of at least 3.8.
- Support the ability to trigger early garbage collection of objects in the `q` memory space by adding `--pykxgc` to the QARGS environment variable, or by setting the `PYKX_GC` environment variable to 1, "1" or True.

## PyKX 1.1.1

#### Release Date

2022-06-13

### Features & Fixes

- Added ability to skip symlinking `$QHOME` to `PyKX`'s local `$QHOME` by setting the environment variable `IGNORE_QHOME`.

## PyKX 1.1.0

#### Release Date

2022-06-07

### Dependencies

- The dependency on the system library `libcurl` has been made optional for Linux. If it is missing on Linux, a warning will be emitted instead of an error being raised, and the KX Insights Core library `kurl` will not be fully loaded. Windows and macOS are unaffected, as they don't support the KX Insights Core features to begin with.

### Features & Fixes

- Splayed and partitioned tables no longer emit warnings when instantiated.
- Added `pykx.Q.sql`, which is a wrapper around [KXI Core SQL](https://code.kx.com/insights/core/sql.html#sql-language-support).
- `.pykx.pyexec` and `.pykx.pyeval` no longer segfault when called with a character atom.
- Updated several `pykx.toq` tests so that they would not randomly fail.
- Fixed error when pickling `pykx.util.BlockManager` in certain esoteric situations.
- Fixed `pandas.MultiIndex` objects created by PyKX having `pykx.SymbolAtom` objects within them - now they have `str` objects instead, as they normally would.
- Upgraded the included KX Insights Core libraries to version 3.0.0.
- Added `pykx.toq.from_datetime_date`, which converts `datetime.date` objects into any q temporal atom that can represent a date (defaulting to a date atom).
- Fixed error when user specifies `-s` or `-q` in `$QARGS`.
- Fixed recursion error when accessing a non-existent attribute of `pykx.q` while in unlicensed mode. Now an attribute error is raised instead.
- Fixed build error introduced by new rules enforced by new versions of setuptools.
- Added `pykx.Anymap`.
- Fixed support for `kx.lic` licenses.
- The KXIC libraries are now loaded after q has been fully initialized, rather than during the initialization. This significantly reduces the time it takes to import PyKX.
- PyKX now uses a single location for `$QHOME`: its `lib` directory within the installed package. The top-level contents of the `$QHOME` directory (prior to PyKX updating the env var when embedded q is initialized) will be symlinked into PyKX's `lib` directory, along with the content of any subdirectories under `lib` (e.g. `l64`, `m64`, `w64`). This enables loading scripts and libraries located in the original `$QHOME` directory during q initialization.
- Improved performance (both execution speed and memory usage) of calling `np.array` on `pykx.Vector` instances. The best practice is still to use the `np` method instead of calling `np.array` on the `pykx.Vector` instance.
- `pykx.Vector` is now a subclass of `collections.abc.Sequence`.
- `pykx.Mapping` is not a subclass of `collections.abc.Mapping`.
- Split `pykx.QConnection` into `pykx.SyncQConnection` and `pykx.AsyncQConnection` and added support for asynchronous IPC with `q` using `async`/`await`. Refer to [the `pykx.AsyncQConnection` docs](../api/ipc.md#pykx.ipc.AsyncQConnection) for more details.
- Pandas dataframes containing Pandas extension arrays not originally created as Numpy arrays would result in errors when attempting to convert to q. For example a Dataframe with index of type `pandas.MultiIndex.from_arrays` would result in an error in conversion.
- Improved performance of converting `pykx.SymbolVector` to `numpy.array` of strings, and also the conversion back from a `numpy.array` of `strings` to a `q` `SymbolVector`.
- Improved performance of converting `numpy.array`'s of `dtype`s `datetime64`/`timedelta64 ` to the various `pykx.TemporalTypes`.

## PyKX 1.0.1

#### Release Date

2022-03-18

### Deprecations & Removals

- The `sync` parameter for `pykx.QConnection` and `pykx.QConnection.__call__` has been renamed to the less confusing name `wait`. The `sync` parameter remains, but its usage will result in a `DeprecationWarning` being emitted. The `sync` parameter will be removed in a future version.

### Features & Fixes
- Updated to stable classifier (`Development Status :: 5 - Production/Stable`) in project metadata. Despite this update being done in version 1.0.1, version 1.0.0 is still the first stable release of PyKX.
- PyKX now provides source distributions (`sdist`). It can be downloaded from PyPI using `pip download --no-binary=:all: --no-deps pykx`. As noted in [the installation docs](../getting-started/installing.md#supported-environments), installations built from the source will only receive support on a best-effort basis.
- Fixed Pandas NaT conversion to q types. Now `pykx.toq(pandas.NaT, ktype=ktype)` produces a null temporal atom for any given `ktype` (e.g. `pykx.TimeAtom`).
- Added [a doc page for limitations of embedded q](../user-guide/advanced/limitations.md).
- Added a test to ensure large vectors are correctly handled (5 GiB).
- Always use synchronous queries internally, i.e. fix `QConnection(sync=False)`.
- Disabled the context interface over IPC. This is a temporary measure that will be reversed once q function objects are updated to run in the environment they were defined in by default.
- Reduced the time it takes to import PyKX. There are plans to reduce it further, as `import pykx` remains fairly slow.
- Updated to [KXI Core 2.1](https://code.kx.com/insights/core/release-notes/previous.html#210) & rename `qce` -> `kxic`.
- Misc test updates.
- Misc doc updates.

## PyKX 1.0.0

#### Release Date

2022-02-14

### Migration Notes

To switch from Pykdb to PyKX, you will need to update the name of the dependency from `pykdb` to `pykx` in your `pyproject.toml`/`requirements.txt`/`setup.cfg`/etc. When Pykdb was renamed to PyKX, its version number was reset. The first public release of PyKX has the version number 1.0.0, and will employ [semantic versioning](https://semver.org/).

Pay close attention to the renames listed below, as well as the removals. Many things have been moved to the top-level, or otherwise reorganized. A common idiom with Pykdb was the following:

```python
from pykdb import q, k
```

It is recommended that the following be used instead:

```python
import pykx as kx
```

This way the many attributes at the top-level can be easily accessed without any loss of context, for example:

```python
kx.q # Can be called to execute q code
kx.K # Base type for objects in q; can be used to convert a Python object into a q type
kx.SymbolAtom # Type for symbol atoms; can be used to convert a `str` or `bytes` into a symbol atom
kx.QContext # Represents a q context via the PyKX context interface
kx.QConnection # Can be called to connect to a q process via q IPC
kx.PyKXException # Base exception type for exceptions specific to PyKX and q
kx.QError # Exception type for errors that occur in q
kx.LicenseException # Exception type raised when features that require a license are used without
kx.QHOME # Path from which to load q files, set by $QHOME environment variable
kx.QARGS # List of arguments provided to the embedded q instance at startup, set by $QARGS environment variable
# etc.
```

You can no longer rely on the [context](../api/pykx-execution/ctx.md) being reset to the global context after each call into embedded q, however IPC calls are unaffected.

### Renames
- Pykdb has been renamed to PyKX. `Pykdb` -> `PyKX`; `PYKDB` -> `PYKX`; `pykdb` -> `pykx`.
- The `adapt` module has been renamed to `toq`, and it can be called directly. Instead of `pykdb.adapt.adapt(x)` one should write `pykx.toq(x)`.
- The `k` module has been renamed to `wrappers`. All wrapper classes can be accessed from the top-level, i.e. `pykx.K`, `pykx.SymbolAtom`, etc.
- The "module interface" (`pykdb.module_interface`) has been renamed to the "context interface" (`pykx.ctx`). All `pykx.Q` instances (i.e. `pykx.q` and all `pykx.QConnection` instances) have a `ctx` attribute, which is the global `QContext` for that `pykx.Q` instance. Usually, one need not directly access the global context. Instead, one can access its subcontexts directly e.g. `q.dbmaint` instead of `q.ctx.dbmaint`.
- `KdbError` (and its subclasses) have been renamed to `QError`
- `pykdb.ctx.KdbContext` has been renamed to `pykx.ctx.QContext`, and is available from the top-level, i.e. `pykx.QContext`.
- The `Connection` class in the IPC module has been renamed to `QConnection`, and is now available at the top-level, i.e. `pykx.QConnection`.
- The q type wrapper `DynamicLoad` has been renamed to `Foreign` (`pykdb.k.DynamicLoad` -> `pykx.Foreign`).

### Deprecations & Removals
- The `pykdb.q.ipc` attribute has been removed. The IPC module can be accessed directly instead at `pykx.ipc`, but generally one will only need to access the `QConnection` class, which can be accessed at the top-level: `pykx.QConnection`.
- The `pykdb.q.K` attribute has been removed. Instead, `K` types can be used as constructors for that type by leveraging the `toq` module. For example, instead of `pykdb.q.K(x)` one should write `pykx.K(x)`. Instead of `pykx.q.K(x, k_type=pykx.k.SymbolAtom)` one should write `pykx.SymbolAtom(x)` or `pykx.toq(x, ktype=pykx.SymbolAtom)`.
- Most `KdbError`/`QError` subclasses have been removed, as identifying them is error prone, and we are unable to provide helpful error messages for most of them.
- The `pykx.kdb` singleton class has been removed.

### Dependencies
- More Numpy, Pandas, and PyArrow versions are supported. Current `pandas~=1.0`, `numpy~=1.20,<1.22`, and `pyarrow>=3.0.0` are supported. PyArrow remains an optional dependency.
- A dependency on `find-libpython~=0.2` was added. This is only used when running PyKX under a q process (see details in the section below about new alpha features).
- A dependency on the system library `libcurl` was added for Linux. This dependency will be made optional in a future release.

### Features & Fixes
- The `pykx.Q` class has been added as the base class for `pykx.EmbeddedQ` (the class for `pykx.q`) and `pykx.QConnection`.
- The `pykx.EmbeddedQ` process now persists its [context](../api/pykx-execution/ctx.md) between calls.
- The console now works over IPC.
- The query module now works over IPC. Because `K` objects hold no reference to the `q` instance that created them (be it local or over IPC), `K` tables no longer have `select`/`exec`/`update`/`delete` methods with themselves projected in as the first argument. That is to say, instead of writing `t.select(...)`, write `q.qsql.select(t, ...)`, where `q` is either `pykx.q` or an instance of `pykx.QConnection`, and `t` was obtained from `q`.
- The context interface now works over IPC.
- Nulls and infinities are now handled as nulls and infinities, rather than as their underlying values. `pykx.Atom.is_null`, `pykx.Atom.is_inf`, `pykx.Collection.has_nulls`, and `pykx.Collection.has_infs` have been added. Numpy, Pandas, and PyArrow handles integral nulls with masked arrays, and they handle temporal nulls with `NaT`. `NaN` continues to be used for real/float nulls. The general Python representation (from `.py()`) uses `K` objects for nulls and infinities.
- Calling `bool` on `pykx.K` objects now either raises a `TypeError`, or return the unambiguously correct result. For ambiguous cases such as `pykx.Collection` instances, use `.any()`, `.all()`, or a length check instead.
- Assignment to q reserved words or the q context now raises a `pykx.PyKXException`.
- `pykx.toq.from_list` (previously `pykdb.adapt.adapt_list`) now works in unlicensed mode.
- `q.query` and `q.sql` are now placeholders (set to `None`). The query interface can be accessed from `q.qsql`.
- Ternary `pow` now raises `TypeError` for `RealNumericVector` and `RealNumericAtom`.
- `QContext` objects are now context handlers, e.g. `with pykx.q.dbmaint: # operate in .dbmaint within this block`. This context handler supports arbitrary nesting.
- `__getitem__` now raises a `pykx.LicenseException` when used in unlicensed mode. Previously it worked for a few select types only. If running in unlicensed mode, one should perform all q indexing in the connected q process, and all Python indexing after converting the `K` object to a Python/Numpy/Pandas/PyArrow object.
- `pykx.QConnection` (previously `pykdb.ipc.Connection`) objects now have an informative/idiomatic repr.
- Calls to `pykx.q` now support up to 8 arguments beyond the required query at position 0, similar to calling `pykx.QConnection` instances. These arguments are applied to the result of the query.
- Embedded q is now used to count the number of rows a table has.
- All dynamic linking to `libq` and `libe` has been replaced by dynamic loading. As a result, the modules previously known as `adapt` and `adapt_unlicensed` have been unified under `pykx.toq`.
- PyKX now attempts to initialize embedded q when `pykx` is imported, rather than when `pykx.q` is first accessed. As a result, the error-prone practice of supplying the `pykx.kdb` singleton class with arguments for embedded q is now impossible.
- Arguments for embedded q can now be supplied via the environment variable `$QARGS` in the form of command-line arguments. For example, `QARGS='--unlicensed'` causes PyKX to enter unlicensed mode when it is started, and `QARGS='-o 8'` causes embedded q to use an offset from UTC of 8 hours. These could be combined as `QARGS='--unlicensed -o 8'`.
- Added the `--licensed` startup flag (to be provided via the `$QARGS` environment variable), which can be used to raise a `pykx.PyKXException` (rather than emitting a warning) if PyKX fails to start in licensed mode (likely because of a missing/invalid q license).
- PyKX Linux wheels are now [PEP 600](https://peps.python.org/pep-0600/) compliant, built to the `manylinux_2_17` standard.
- Misc other bug fixes.
- Misc doc improvements.

### Performance Improvements

- Converting nested lists from q to Python is much faster.
- Internally, PyKX now calls q functions with arguments directly instead of creating a `pykx.Function` instance then calling it. This results in modest performance benefits in some cases.
- The context interface no longer loads every element of a context when the context is first accessed, thereby removing the computation spike, which could be particularly intense for large q contexts.

### New Alpha Features

!!! danger "Alpha features are subject to change"

    Alpha features are not stable will be subject to changes without notice. Use at your own risk.

- q can now load PyKX by loading the q file `pykx.q`. `pykx.q` can be copied into `$QHOME` by running `pykx.install_into_QHOME()`. When loaded into q, it will define the `.pykx` namespace, which notably has `.pykx.exec` and `.pykx.pyeval`. This allows for Python code to be run within q libraries and applications without some of the limitations of embedded q such as the lack of the q main loop, or the lack of timers. When q loads `pykx.q`, it attempts to source the currently active Python environment by running `python`, then fetching the environment details from it.

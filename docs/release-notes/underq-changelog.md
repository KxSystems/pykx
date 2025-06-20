# PyKX under q Changelog

This changelog provides updates from PyKX 2.0.0 and above, for information relating to versions of PyKX prior to this version see the changelog linked below.

!!! Note

	The changelog presented here outlines changes to PyKX when operating within a q environment specifically, if you require changelogs associated with PyKX operating within a Python environment see [here](./changelog.md).

## PyKX 3.1.3

#### Release Date

2025-06-12

### Fixes and Improvements

- Dashboards integration now lists missing libraries by name.

    === "Behavior prior to change"

        ```q
        q).pykx.dash.util.getFunction[""]
        'Required libraries for PyKX Dashboards integration not found
            [0]  .pykx.dash.util.getFunction[""]
                ^
        ```

    === "Behavior post change"

        ```q
        q).pykx.dash.util.getFunction[""]
        'Required libraries for PyKX Dashboards integration not found: ast2json
            [0]  .pykx.dash.util.getFunction[""]
                ^ 
        ```

- Resolved `double free or corruption (out)` error when `pykx.q` was loaded in `QINIT` or `q.q`.
- `pykx.q` load time has been roughly halved.

### Deprecations & Removals

- `.pykx.console[]` has been removed on Windows due to incompatibility. Will now error with `'.pykx.console is not available on Windows` if called.

## PyKX 3.1.1

#### Release Date

2025-02-14

### Fixes and Improvements

- Fix throwing of errors for `.pykx.safeReimport`, now throws an error instead of returning a value.
  
	=== "Behavior prior to change"

		```q
		q).pykx.safeReimport {1+`this}
		"type"'
		```

	=== "Behavior post change"
	
		```q
		q).pykx.safeReimport {1+`this}
		'type
		  [2]  /home/user/q/pykx.q:1714: .pykx.safeReimport@:{'x}
                                                              ^
 		  [1]  /home/user/q/pykx.q:1714: .pykx.safeReimport:
 		  setenv'[envlist;envvals];
 		  $[r 0;{'x};::] r 1
  		  ^
  		  }
		q.pykx))
		```

## PyKX 3.1.0

#### Release Date

2025-02-11

### Additions

- Addition of `.pykx.typepy` which returns an objects datatype as a `CharVector` after being passed to python.

	```q
	q).pykx.typepy til 10  // Data type conversion set to default
	"<class 'numpy.ndarray'>"

	q).pykx.util.defaultConv:"pd"  // Set data type conversion to pandas
	q).pykx.typepy til 10
	"<class 'pandas.core.series.Series'>"
	```

### Fixes and Improvements

- Using `.pykx.toq`/`.pykx.toq0` now return the q representation of an object when passed a wrapped type conversion object

	=== "Behaviour prior to change"

		```q
		q).pykx.toq .pykx.topd ([] a:1 2 3)
		enlist[`..pandas;;][...
		```

	=== "Behaviour post change"

		```q
		q).pykx.toq .pykx.topd[([] a:1 2 3)]
		a
		-
		1
		2
		3
		```

- When using `.pykx.toq`/`.pykx.toq0`, passing compositions such as `any` now returns the data as the appropriate object

	=== "Behaviour prior to change"

		```q
		q).pykx.toq any
		'Expected foreign object for call to .pykx.toq
		```

	=== "Behaviour post change"

		```q
		q).pykx.toq any
		max$["b"]
		```

- When failing to find a file loaded with `.pykx.loadPy` the name of the file which was loaded is now included in the error message

	=== "Behaviour prior to change"

		```q
		q).pykx.loadPy "file.py"
		'FileNotFoundError(2, 'No such file or directory')
		```

	=== "Behaviour post change"

		```q
		q).pykx.loadPy "file.py"
		'FileNotFoundError(file.py, 'No such file or directory')
		```

- When attempting to load PyKX after embedPy has already been loaded, an error will be thrown and PyKX will not continue to load.

### Beta Features

- Added ability for users to convert between PyKX numeric vectors or N-Dimensional Lists and PyTorch Tensor objects using the `.pykx.topt` function.

	```python
	q).pykx.eval["lambda x:print(type(x))"].pykx.topt enlist til 10;
	<class 'torch.Tensor'>
	```

## PyKX 3.0.1

#### Release Date

2024-12-04

### Additions

- Added a function `.pykx.loadPy` which can be used to execute/load a Python file following expected Python syntax, this removes limitations which exist due to use of q parsing syntax when loading a Python file using `system"l /path/to/file.p"` for example

	```q
	q)\cat /path/to/file.p
	"def func(x: int,"
	"         y: int"
	"): -> None"
	"    return x+y"
	q).pykx.loadPy["/path/to/file.p"]
	q)f:.pykx.get[`func;<]
	q)f[10;20]
	30
	```

## PyKX 3.0.0

#### Release Date

2024-11-12

### Additions

- Added `cloud_libraries` kwarg to `install_into_QHOME` allowing installation of the kdb Insights cloud libraries to QHOME.
- Addition of support for new environment variable `PYKX_USE_FIND_LIBPYTHON` which will use the Python package [`find_libpython`](https://pypi.org/project/find-libpython/) to specify the location from which `libpython.[so|dll]` will be taken.

### Fixes and Improvements

- Addition of function `.pykx.toq0` to support conversion of Python strings to q strings rather than q symbols as is the default behaviour

	```q
	q)pystr:.pykx.eval["\"test\""]
	q).pykx.toq0[pystr]
	`test
	q).pykx.toq0[pystr;1b]
	"test"
	```

- Fix for `install_into_QHOME` with `overwrite_embedpy=True`. Previously loading PyKX through use of `p)` would fail.

	=== "Behaviour prior to change"

		```q
		q)p)print(1+1)
		'pykx.q. OS reports: No such file or directory
		  [3]  /home/user/q/p.k:1: \l pykx.q
		```

	=== "Behaviour post change"

		```q
		q)p)print(1+1)
		2
		```

- Fix to minor memory leak when accessing attributes or retrieving global variables from Python objects. The following operations would lead to this behaviour

	```q
	q)np:.pykx.import[`numpy]
	q)np`:array # Accessing an attribute caused a leak
	q).pykx.console[]
	>>> variable = 100
	>>> quit()
	q).pykx.get`variable # Accessing a global in this way caused a leak
	```

- When loading on Linux loading of `qlog` no longer loads the logging functionality into the `.pykx` namespace and instead loads it to the `.com_kx_log` namespace as expected under default conditions.

	=== "Behaviour prior to change"

		```q
		q)@[{get x;1b};`.pykx.configure;0b]
		1b
		q)@[{get x;1b};`.com_kx_log.configure;0b]
		0b
		```

	=== "Behaviour post change"

		```q
		q)@[{get x;1b};`.pykx.configure;0b]
		0b
		q)@[{get x;1b};`.com_kx_log.configure;0b]
		1b
		```

## PyKX 2.5.4

#### Release Date

2024-10-22

### Fixes and Improvements

- `.pykx.util.loadfile` now loads a file using it's full path unless it contains a space. This is to avoid issues loading scripts which are sensitive to their working directory.

## PyKX 2.5.3

#### Release Date

2024-08-20

### Fixes and Improvements

- Previously PyKX conversions of generic lists (type 0h) would convert this data to it's `raw` representation rather than it's `python` representation as documented. This had the effect of restricting the usability of some types within PyKX under q in non-trivial use-cases. With the `2.5.2` changes to more accurately represent `raw` data at depth this became more obvious as an issue.

	=== "Behaviour prior to change"

		```q
		q).pykx.version[]
		"2.5.2"
		q).pykx.print .pykx.eval["lambda x:x"](`test;::;first 1?0p)
		[b'test', None, 49577290277400616]
		```

	=== "Behaviour post change"

		```q
		q).pykx.print .pykx.eval["lambda x:x"]
		['test', None, datetime.datetime(2002, 1, 25, 11, 16, 58, 871372)]
		```

## PyKX 2.5.0

#### Release Date

2024-05-15

### Fixes and Improvements

- When loading PyKX under from a source file path containing a space initialisation would fail with an `nyi` error message, this has now been resolved.

## PyKX 2.4.1

#### Release Date

2024-03-27

### Fixes and Improvements

- When loading PyKX under q users who had previously loaded [embedPy](https://github.com/KxSystems/embedPy) into their process would cause a segfault of unspecified origin. With this release we have added a warning prior to loading of PyKX which specifies that if a value of `.p.e` has been specified which does not match that expected of PyKX a user should consider installing PyKX under q fully:

	```q
	q)\l p.q     // Load embedPy
	q)\l pykx.q
	Warning: Detected invalid '.p.e' function definition expected for PyKX.
	Have you loaded another Python integration first?

	Please consider full installation of PyKX under q following instructions at:
	https://code.kx.com/pykx/pykx-under-q/intro.html#installation
	```

## PyKX 2.3.1

#### Release Date

2024-02-07

### Fixes and Improvements

- `.pykx.eval` is now variadic allowing an optional second parameter to be passed to define return type. Previously would error with `rank`.

	=== "Behavior prior to change"

		```q
		q).pykx.eval["lambda x: x";<] 7
		'rank
		[0]  .pykx.eval["lambda x: x";<] 7
		```

	=== "Behavior post change"

		```q
        q).pykx.eval["lambda x: x";<] 7
		7
		```

- Wraps which have a return type assigned using `<` or `>` are now considered wraps and can be unwrapped:

	=== "Behavior prior to change"

		```q
		q).pykx.util.isw .pykx.eval["lambda x: x"][<]
		0b
		q).pykx.unwrap  .pykx.eval["lambda x: x"][<]
		{$[type[x]in 104 105 112h;util.foreignToq unwrap x;x]}.[code[foreign]]`.pykx.util.parseArgsenlist
		```

	=== "Behavior post change"

		```q
		q).pykx.util.isw .pykx.eval["lambda x: x"][<]
		1b
		q).pykx.unwrap  .pykx.eval["lambda x: x"][<]
		foreign
		```

- `.pykx.qcallable` and `.pykx.pycallable` can now convert wraps which already have return types assigned:

	=== "Behavior prior to change"

		```q
		q).pykx.qcallable[.pykx.eval["lambda x: x"][<]]`
		'Could not convert provided function to callable with q return
		q).pykx.print .pykx.pycallable[.pykx.eval["lambda x: x"][>]]
		'Could not convert provided function to callable with Python return
		```

	=== "Behavior post change"

		```q
		q).pykx.qcallable[.pykx.eval["lambda x: x"][<]]`test
		`test
		q).pykx.print .pykx.wrap .pykx.pycallable[.pykx.eval["lambda x: x"][>]]`test
		test
		```

## PyKX 2.3.0

#### Release Date

2024-01-22

### Fixes and Improvements

- A bug was fixed when using `.pykx.console`, it is now possible to access python variables set using the console with `.pykx.(eval|pyexec|pyeval)` functions.

	=== "Behavior prior to change"

		```q
        q) .pykx.console[]
        >>> a = 10
        >>> quit()
        q) .pykx.eval["a"]`
        'NameError("name 'a' is not defined")
          [1]  /.../q/pykx.q:968: .pykx.eval:{wrap pyeval x}
		```

	=== "Behavior post change"

		```q
        q) .pykx.console[]
        >>> a = 10
        >>> quit()
        q) .pykx.eval["a"]`
        10
		```

## PyKX 2.2.2

#### Release Date

2023-12-07

### Fixes and Improvements

- When loaded in a q process loading `pykx.q` would not allow `Ctrl+C` (SIGINT) interruption.

## PyKX 2.2.1

#### Release Date

2023-11-30

### Fixes and Improvements

- `.pykx.print` was using `repr` representation for some objects. Now consistently calls `print`.
- `.pykx.safeReimport` now resets environment variables correctly before throwing any error raised by the function supplied to it.
- Wrapped Python objects being supplied as arguments to functions were being converted according to `.pykx.util.defaultConv`. Now are left untouched:

	=== "Behavior prior to change"

		```q
		q)\l pykx.q
		q)np:.pykx.import `numpy;
		q)r:np[`:random.rand;1;2];
		q).pykx.print r
		array([[0.03720163, 0.72012121]])
		q).pykx.print .pykx.eval["lambda x: x"] r
		array([array([0.03720163, 0.72012121])], dtype=object)
		q).pykx.setdefault"py"
		q).pykx.print .pykx.eval["lambda x: x"] r
		[[0.037201634310417564, 0.7201212148535847]]
		```

	=== "Behavior post change"

		```q
		q).pykx.print r
		array([[0.59110368, 0.52612429]])
		q).pykx.print .pykx.eval["lambda x: x"] r
		array([[0.59110368, 0.52612429]])
		q).pykx.setdefault"py"
		q).pykx.print .pykx.eval["lambda x: x"] r
		array([[0.59110368, 0.52612429]])
		```
- q hsym will convert correctly to `pathlib.PosixPath` rather than `str`:

	=== "Behavior prior to change"

		```q
		q).pykx.eval["lambda x: print(type(x))"] `:/path/to/somewhere;
		<class 'str'>
		```

	=== "Behavior post change"

		```q
		q).pykx.eval["lambda x: print(type(x))"] `:/path/to/somewhere;
		<class 'pathlib.PosixPath'>
		```

## PyKX 2.2.0

#### Release Date

2023-11-09

### Additions

- Addition of `PYKX_EXECUTABLE` environment/configuration variable to allow control of which Python executable is used under q.

### Fixes and Improvements

- Failure to access and load PyKX resulting in an `os` error now returns Python backtrace outlining the underlying Python error allowing for easier debugging

	=== "Behavior prior to change"

		```q
		q)\l pykx.q
		'os
		  [4]  \python3 -c "import pykx; print(pykx.config.pykx_dir)" 2>/dev/null
		       ^
		```

	=== "Behavior post change"

		```q
		q)\l pykx.q
		Traceback (most recent call last):
		  File "<string>", line 1, in <module>
		  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/__init__.py", line 27, in <module>
		    from . import core
		  File "pykx/core.pyx", line 6, in init pykx.core
		    from .util import num_available_cores
		  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/util.py", line 8, in <module>
		    import pandas as pd
		  File "/usr/local/anaconda3/lib/python3.8/site-packages/pandas/__init__.py", line 16, in <module>
		    raise ImportError(
		ImportError: Unable to import required dependencies:
		numpy: cannot import name 'SystemRandom' from 'random' (/Users/projects/pykx/src/pykx/random.py)
		Traceback (most recent call last):
		  File "<string>", line 1, in <module>
		  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/__init__.py", line 27, in <module>
		    from . import core
		  File "pykx/core.pyx", line 6, in init pykx.core
		    from .util import num_available_cores
		  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/util.py", line 8, in <module>
		    import pandas as pd
		  File "/usr/local/anaconda3/lib/python3.8/site-packages/pandas/__init__.py", line 16, in <module>
		    raise ImportError(
		ImportError: Unable to import required dependencies:
		numpy: cannot import name 'SystemRandom' from 'random' (/Users/projects/pykx/src/pykx/random.py)
		'os
		  [4]  \python3 -c "import pykx; print('PYKX_DIR: ' + str(pykx.config.pykx_dir))"
		```

- Fixed `type` error if converting dictionaries or keyed tables with conversion set to `default`
- On load now sets `PYKX_SKIP_UNDERQ` rather than deprecated `SKIP_UNDERQ`
- `safeReimport` now additionally unsets/resets: `PYKX_DEFAULT_CONVERSION`, `PYKX_SKIP_UNDERQ`, `PYKX_EXECUTABLE`, `PYKX_DIR`

## PyKX 2.1.0

#### Release Date

2023-10-09

### Fixes and Improvements

- Update to default conversion logic for q objects passed to PyKX functions to more closely match embedPy based conversion expectations.For version <=2.0 conversions of KX lists would produce N Dimensional Numpy arrays of singular type. This results in issues when applying to many analytic libraries which rely on lists of lists rather than singular N Dimensional arrays. Additionally q tables and keyed tables would be converted to Numpy recarrays, these are now converted to Pandas DataFrames. To maintain previous behavior please set the following environment variable `PYKX_DEFAULT_CONVERSION="np"`.

	=== "Behaviour prior to change"

		```q
		q).pykx.eval["lambda x:print(type(x))"](10?1f;10?1f)
		<class 'numpy ndarray'>
		q).pykx.eval["lambda x:print(type(x))"]([]10?1f;10?1f)
		<class 'numpy.recarray'>
		```

	=== "Behaviour post change"

		```q
		q).pykx.eval["lambda x:print(type(x))"](10?1f;10?1f)
		<class 'list'>
		q).pykx.eval["lambda x:print(type(x))"]([]10?1f;10?1f)
		<class 'pandas.core.frame.DataFrame'>
		```

## PyKX 2.0.0

#### Release Date

2023-09-18

### Additions

- Addition of `.pykx.qcallable` and `.pykx.pycallable` functions which allow wrapping of a foreign Python callable function returning the result as q or Python foreign respectively.
- Addition of `.pykx.version` allowing users to programmatically access their version from a q process.
- Addition of `.pykx.debug` namespace containing copies of useful process initialization information specific to usage within a q environment
- Addition of function `.pykx.debugInfo` which returns a string representation of useful information when debugging issues with the the use of PyKX within the q environment
- Added the ability for users to print the return of a `conversion` object

	```q
	q).pykx.print .pykx.topd ([]5?1f;5?1f)
	          x        x1
	0  0.613745  0.493183
	1  0.529481  0.578520
	2  0.691610  0.083889
	3  0.229662  0.195991
	4  0.691953  0.375638
	```

### Fixes and Improvements

- Application of object setting on a Python list returns generic null rather than wrapped foreign object.
- Use of environment variables relating to `PyKX under q` must use `"true"` as accepted value, previously any value set for such environment variables would be supported.
- Fixed an issue where invocation of `.pykx.print` would not return results to stdout.
- Fixed an issue where `hsym`/`Path` style objects could not be passed to Python functions

	```q
	q).pykx.eval["lambda x: x"][`:test]`
	`:test	
	```

- Resolution to memory leak incurred during invocation of `.pykx.*eval` functions relating to return of Python foreign objects to q.
- Fixed an issue where segmentation faults could occur for various function if non Python backed foreign objects are passed in place of Python backed foreign

	```q
	q).pykx.toq .pykx.util.load[(`foreign_to_q;1)]
	'Provided foreign object is not a Python object
	```

- Fixed an issue where segmentation faults could occur through repeated invocation of `.pykx.pyexec`

	```q
	q)do[1000;.pykx.pyexec"1+1"]
	```

- Removed the ability when using PyKX under q to allow users set reserved Python keywords to other values

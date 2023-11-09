# PyKX under q Changelog

This changelog provides updates from PyKX 2.0.0 and above, for information relating to versions of PyKX prior to this version see the changelog linked below.

!!! Note

	The changelog presented here outlines changes to PyKX when operating within a q environment specifically, if you require changelogs associated with PyKX operating within a Python environment see [here](./changelog.md).

## PyKX 2.2.0

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

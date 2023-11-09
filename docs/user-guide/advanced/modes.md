# Modes of Operation

PyKX exists to supersede all previous interfaces between q and Python, this document outlines the various conditions under which PyKX can operate and the limitations/requirements which are imposed under these distinct operating modalities, specifically this document breaks down the following:

- PyKX within a Python session
  - Operating with a valid KX License
  - Operating in the absence of a valid KX License
- PyKX within a q session with a valid KX License

## PyKX within a Python session

PyKX operating within a Python session is intended to offer a replacement for [qPython](https://github.com/exxeleron/qPython) and [PyQ](https://github.com/kxsystems/pyq). In order to facilitate replacement of qPython PyKX provides a mode of operation for IPC based communication which allows for the creation of IPC connections and the conversion of data from Pythonic representations to kx objects, this IPC only modality is referred to as `"Unlicensed mode"` within the documentation. The following outline the differences between `"Licensed"` and `"Unlicensed"` operation.

The following table outlines some of the key differences between the two operating modes

| Feature                                                                      | With a PyKX Enabled License | Without a PyKX Enabled License |
|------------------------------------------------------------------------------|-----------------------------|--------------------------------|
| Convert objects from q to Pythonic types and vice-versa                      | :material-check:            | :material-check:               |
| Query synchronously and asynchronously an existing q server via IPC          | :material-check:            | :material-check:               |
| Query synchronously and asynchronously an existing q server with TLS enabled | :material-check:            | :material-close:               |
| Interact with PyKX tables via a Pandas like API                              | :material-check:            | :material-close:               |
| Can run arbitrary q code within a Python session                             | :material-check:            | :material-close:               |
| Display PyKX/q objects within a Python session                               | :material-check:            | :material-close:               |
| Load kdb+ Databases within a Python session                                  | :material-check:            | :material-close:               |
| Can read/write JSON, CSV and q formats to/from disk                          | :material-check:            | :material-close:               |
| Access to Python classes for SQL, schema creation custom data conversion     | :material-check:            | :material-close:               |
| Run Python within a q session using PyKX under q                             | :material-check:            | :material-close:               |
| Full support for nulls, infinities, data slicing and casting                 | :material-check:            | :material-close:               |
| Production Support                                                           | :material-check:            | :material-close:               |

### Operating in the absence of a KX License

Unlicensed mode is a feature-limited mode of operation for PyKX which aims to replace qPython, which has the benefit of not requiring a valid q license (except for the q license required to run the remote q process that PyKX will connect to in this mode).

This mode cannot run q embedded within it, and so it lacks the ability to run q code within the local Python process, and also every feature that depends on running q code. Despite this limitation, it provides the following features (which are all also available in licensed mode):

- Conversions from Python to q
  - With the exception of Python callable objects
- Conversions from q to Python
- [A q IPC interface](../../api/ipc.md)

### Operating with a valid KX License

Licensed mode is the standard mode of operation of PyKX, wherein it is running under a Python process [with a valid q license](../../getting-started/installing.md#licensing-code-execution-for-pykx). This modality aims to replace PyQ as the Python first library for KX. All PyKX features are available in this mode.

The following are the differences provided through operation with a valid KX License

1. Users can execute PyKX/q functionality directly within a Python session
2. PyKX objects can be represented in a human readable format rather than as a memory address, namely

	=== "Licensed mode"
	
		```python
		>>> kx.q('([]til 3;3?0Ng)')
		pykx.Table(pykx.q('
		x x1                                  
		--------------------------------------
		0 8c6b8b64-6815-6084-0a3e-178401251b68
		1 5ae7962d-49f2-404d-5aec-f7c8abbae288
		2 5a580
		```

	=== "Unlicensed mode"

		```python
		>>> conn.q('([]til 3;3?0Ng)')
		pykx.Table._from_addr(0x7f5b72ef8860)
		```

3. PyKX objects can be introspected through indexing

	=== "Licensed mode"

		```python
		>>> import pykx as kx
		>>> tab = kx.q('til 10')
		>>> tab[1:6]
		pykx.LongVector(pykx.q('1 2 3 4 5'))
		```

	=== "Unlicensed mode"

		```python
		>>> py = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		>>> kx.toq(py)
		pykx.List._from_addr(0x7fae68f00a00)
		>>> kx.toq(py)[1:6]
		Traceback (most recent call last):
		  File "<stdin>", line 1, in <module>
		  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/wrappers.py", line 1166, in __getitem__
		    raise LicenseException('index into K object')
		pykx.exceptions.LicenseException: A valid q license must be in a known location (e.g. `$QLIC`) to index into K object.
		```

4. Users can cast between kx object types explicitly

	=== "Licensed mode"

		```python
		>>> import pykx as kx
		>>> py = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		>>> kx.K.cast(kx.toq(py), kx.FloatVector)
		pykx.FloatVector(pykx.q('0 1 2 3 4 5 6 7 8 9f'))
		```

	=== "Unlicensed mode"

		```python
		>>> import pykx as kx
		>>> py = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		>>> kx.K.cast(kx.toq(py), kx.FloatVector)
		Traceback (most recent call last):
		  File "<stdin>", line 1, in <module>
		  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/wrappers.py", line 419, in cast
		    return ktype(self)
		  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/wrappers.py", line 246, in __new__
		    return toq(x, ktype=None if cls is K else cls, cast=cast) # TODO: 'strict' and 'cast' flags
		  File "pykx/toq.pyx", line 2543, in pykx.toq.ToqModule.__call__
		  File "pykx/toq.pyx", line 470, in pykx.toq.from_pykx_k
		pykx.exceptions.LicenseException: A valid q license must be in a known location (e.g. `$QLIC`) to directly convert between K types..
		```

5. Access to the following classes/functionality are supported when running in the licensed modality but not unlicensed, note this is not an exhaustive list
	1. kx.q.sql
	2. kx.q.read
	3. kx.q.write
	4. kx.q.schema
	5. kx.q.console
6. [Pandas API](Pandas_API.ipynb) functionality for interactions with and PyKX Table objects
6. Keyed tables can be converted to equivalent Numpy types
7. All types can be disambiguated, generic null can be discerned from a projection null, and similar for regular vs splayed tables.
8. Numpy list object conversion when operating with a valid PyKX license are optimized relative to unlicensed mode.
9. The `is_null`, `is_inf`, `has_nulls`, and `has_infs` methods of `K` objects are only supported when using a license.

### Choosing to run with/without a license

Users can choose to initialise PyKX under one of these modalities explicitly through the use of the `QARGS` environment variable as follows:

| Modality argument| Description|
|------------------|----------|
| `--unlicensed`   | Starts PyKX in unlicensed mode. No license check will be performed, and no warning will be emitted at start-up if embedded q initialization fails. |
| `--licensed`     | Raise a `PyKXException` (as opposed to emitting a `PyKXWarning`) if embedded q initialization fails.

In addition to the PyKX specific start-up arguments `QARGS` also can be used to set the standard [q command-line arguments](https://code.kx.com/q/basics/cmdline/).

Alternatively for users who wish to make use of PyKX in unlicensed mode they can set the environment variable `PYKX_UNLICENSED="true"` or define this in their `.pykx-config` file as outlined [here](../configuration.md).


## PyKX within a q session

Fully described [here](../../pykx-under-q/intro.md) the ability to use PyKX within a q session directly is intended to provide the ability to replace [embedPy](https://github.com/kxsystems/embedpy) functionally with an updated and more flexible interface. Additionally it provides the ability to use Python functionality within a q environment which does not have the central limitations that exist for PyKX as outlined [here](limitations.md), namely Python code can be used in conjunction with timers and subscriptions within a q/kdb+ ecosystem upon which are reliant on these features of the language.

Similar to the use of PyKX in it's licensed modality PyKX running under q requires a user to have access to an appropriate license containing the `insights.lib.pykx` and `insights.lib.embedq` licensing flags.

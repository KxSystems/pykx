---
title:  PyKX modes of operation
description: Operating PyKX in Python and q
date: June 2024
author: KX Systems, Inc.,
tags: PyKX, q language, Python, licensed, unlincensed,
---

# Modes of operation

_This page explains how to operate PyKX in Python and q, with or without a KDB Insights license._

PyKX can operate in different modes. Each mode has its limitations and requirements, so we're breaking them down into the following:

1. PyKX within Python
     - 1.a Unlicensed
     - 1.b Licensed
2. PyKX within q with a kdb Insights license

## 1. PyKX within Python

The purpose of operating PyKX within a Python session is to replace [qPython](https://github.com/exxeleron/qPython) and [PyQ](https://github.com/kxsystems/pyq). Within Python, PyKX has two modes of operation:

- `#!python Licensed` (this means you have a kdb Insights license with PyKX enabled)
- `#!python Unlicensed` (this means you don't have a kdb Insights license or a license in which PyKX is not enabled)

The main difference between the two is that the `#!python Unlicensed` mode is for IPC-based communication. This mean that it allows to create IPC connections and convert data from Pythonic representations to PyKX objects. 

The following table outlines more key differences:

| **Feature**                                                                  | **Licensed**          | **Unlicensed**         |
| :--------------------------------------------------------------------------- | :-------------------- | :--------------------- |
| Convert objects from q to Pythonic types and vice-versa                      | :material-check:      | :material-check:       |
| Query synchronously and asynchronously a q server via IPC                    | :material-check:      | :material-check:       |
| Query synchronously and asynchronously a q server with TLS enabled           | :material-check:      | :material-close:       |
| Interact with PyKX tables via a Pandas like API                              | :material-check:      | :material-close:       |
| Run arbitrary q code within a Python session                                 | :material-check:      | :material-close:       |
| Display PyKX/q objects within a Python session                               | :material-check:      | :material-close:       |
| Load kdb+ Databases within a Python session                                  | :material-check:      | :material-close:       |
| Read/write JSON, CSV and q formats to/from disk                              | :material-check:      | :material-close:       |
| Access to Python classes for SQL, schema creation, custom data conversion    | :material-check:      | :material-close:       |
| Run Python within a q session using PyKX under q                             | :material-check:      | :material-close:       |
| Full support for nulls, infinities, data slicing and casting                 | :material-check:      | :material-close:       |
| Production support                                                           | :material-check:      | :material-close:       |

### 1.a Running in Unlicensed mode

Unlicensed mode is a feature-limited mode of operation for PyKX. Its aim is to replace qPython, which has the benefit of not requiring a valid q license (except for the q license required to run the remote q process that PyKX connects to in this mode).

This mode cannot run q embedded within it. Also, it lacks the ability to run q code within the local Python process or any functionality that depends on running q code. Despite this limitation, it provides the following features (which are all also available in licensed mode):

- Conversions from Python to q, except Python-callable objects
- Conversions from q to Python
- [A q IPC interface](../../api/ipc.md)

### 1.b Running in Licensed mode

Licensed mode is the standard way to operate PyKX, wherein it's running under a Python process [with a valid q license](../../getting-started/installing.md#licensing-code-execution-for-pykx). This modality aims to replace PyQ as the Python-first library for KX. All PyKX features are available in this mode.

The differences provided through operating with a valid kdb Insights license are:

1. You can execute PyKX/q functionalities directly within a Python session.
2. PyKX objects can be represented in a human readable format rather than as a memory address, namely:

	=== "Licensed mode"
	
		```python
		>>> kx.q('([]til 3;3?0Ng)')
		pykx.Table(pykx.q('
		x x1                                  
		--------------------------------------
		0 8c6b8b64-6815-6084-0a3e-178401251b68
		1 5ae7962d-49f2-404d-5aec-f7c8abbae288
		2 5a580
		'))
		```

	=== "Unlicensed mode"

		```python
		>>> conn.q('([]til 3;3?0Ng)')
		pykx.Table._from_addr(0x7f5b72ef8860)
		```

3. You can analyze PyKX objects through indexing:

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

4. Licensed mode allows users to cast between PyKX object types. Unlicensed mode doesn't support this, showing an error as below:

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
		pykx.exceptions.LicenseException: A valid q license must be in a known location (e.g. `$QLIC`) to directly convert between K types.
		```

5. Only licensed mode supports the classes/functionalities below. This is not an exhaustive list:
	1. kx.q.sql
	2. kx.q.read
	3. kx.q.write
	4. kx.q.schema
	5. kx.q.console
6. [Pandas API](Pandas_API.ipynb) functionality for interactions with and PyKX Table objects.
6. You can convert keyed tables to equivalent Numpy types.
7. All types can be disambiguated, generic null can be discerned from a projection null, and similar for regular vs splayed tables.
8. Numpy list object conversion is optimized only in licensed mode.
9. Only licensed mode grants users access to the `#!python is_null`, `#!python is_inf`, `#!python has_nulls`, and `#!python has_infs` methods of `#!python K` objects.

### How to choose between Licensed and Unlicensed

You can choose to initialise PyKX under one of these modes through the use of the `#!python QARGS` environment variable as follows:

| **Mode argument** | **Description**                                                                                                                          |
| :---------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| `--unlicensed`    | Starts PyKX in unlicensed mode. No license check is performed, and no warning is emitted at start-up if embedded q initialization fails. |
| `--licensed`      | Raises a `PyKXException` (as opposed to emitting a `PyKXWarning`) if embedded q initialization fails.                                    |

In addition to the PyKX specific start-up arguments, you can also use `#!python QARGS` to set the standard [q command-line arguments](https://code.kx.com/q/basics/cmdline/).

Alternatively, if you wish to access PyKX in unlicensed mode, you set the environment variable `#!python PYKX_UNLICENSED="true"` or define this in your `#!python .pykx-config` file as outlined [here](../configuration.md).


## 2. PyKX within q

Fully described [here](../../pykx-under-q/intro.md), the ability to use PyKX within a q session allows you to achieve the following:

- Replace [embedPy](https://github.com/kxsystems/embedpy) functionally with an updated, more flexible interface.
- Use Python within a q environment without the [limitations for PyKX](../../help/issues.md).
- Use Python code in conjunction with timers and subscriptions within a q/kdb+ ecosystem.

Similar to the use of PyKX in licensed mode, PyKX running under q requires a user to have access to an appropriate license containing the `#!python insights.lib.pykx` and `#!python insights.lib.embedq` licensing flags.

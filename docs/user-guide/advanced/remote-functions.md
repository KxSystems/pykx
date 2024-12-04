---
title: PyKX Remote Functions
description: How to execute Python functions on q servers in PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, performance, parallelization, secondary q threads, multithreading, peach
---

# Remote Function Execution

_This page explains how to execute Python functions on q servers in PyKX._

Remote Functions let you define Python functions within your Python environment which can interact with kdb+ data on a q process. Once defined, these functions are registered to a [remote session object](../../api/remote.md) along with any Python dependencies which need to be imported. The [remote session object](../../api/remote.md) establishes and manages the remote connection to the kdb+/q server.

To execute kdb+/q functions using PyKX, go to [PyKX under q](../../pykx-under-q/intro.md)

## Requirements and limitations

Before you start:

- Make sure all necessary Python requirements are installed on the client server. For this functionality you need `#!python dill>=0.2`.
- Confirm that the kdb+/q server you connect to can load PyKX under q.
- Ensure that you have the correct versions of Python library dependencies in your kdb+/q environment at runtime.
- Run the following command:

```bash
pip install pykx[remote]
```

## Functional walkthrough

This walkthrough demonstrates the following steps:

1. Initialize a q/kdb+ server loading PyKX under q on a specified port.
1. Import PyKX and generate a remote session object which denotes the process against which the Python functions will be executed.
1. Define a number of Python functions which will be executed on the remote q/kdb+ server.

### Initializea q/kdb+ server with PyKX under q

This step ensures you have a q process running with PyKX under q, as well as having a kdb+ table available to query. If you have this already, proceed to the next step.

Ensure that you have q installed. If you do not have this installed please follow the guide provided [here](https://code.kx.com/q/learn/install/), retrieving your license following the instructions provided [here](https://kx.com/kdb-insights-personal-edition-license-download).

Install PyKX under q using the following command.

```bash
python -c "import pykx;pykx.install_into_QHOME()"
```

Start the q process to which you will execute your functions.

```bash
q pykx.q -p 5050
```

Create a table which you will use within your Python analytics defined below.

```q
q)N:1000
q)tab:([]sym:N?`AAPL`MSFT`GOOG`FDP;price:100+N?100f;size:10+N?100)
```

Set a requirement for users to provide a username/password if you wish to add security to your q process.

```q
.z.pw:{[u;p]$[(u~`user)&p~`password;1b;0b]}
```

### Import PyKX and create a session

Create a session object from a Python environment of your choice, which establishes and manages the remote connection to the kdb+/q server.

```python
>>> import pykx as kx
>>> session = kx.remote.session(host='localhost', port=5050, username='user', password='password')
```

### Define and execute Python functions using a session

Tag the Python functions you want to run on the remote server using the `#!python kx.remote.function` decorator. This registers the functions on the `#!python session` object you have just created. 

=== "Zero argument function"

	```python
	>>> @kx.remote.function(session)
	... def zero_arg_function():
	...     return 10
	>>> zero_arg_function()
	pykx.LongAtom(pykx.q('10'))
	```

=== "Single argument function"

	```python
	>>> @kx.remote.function(session)
	... def single_arg_function(x):
	...     return x+10
	>>> single_arg_function(10)
	pykx.LongAtom(pykx.q('20'))
	```

=== "Multi argument function"

	```python
	>>> @kx.remote.function(session)
	... def multi_arg_function(x, y):
	...     return x+y
	>>> multi_arg_function(10, 20)
	pykx.LongAtom(pykx.q('30'))
	```

Add any Python libraries which need to be available when executing the function(s) you have just defined. You can achieve this in three ways:

1. Adding the `#!python libraries` keyword when generating your session object
1. Using `#!python session.libraries` on an existing session to import required libraries before defining your function
1. Importing libraries within the body of the function being executed

Examples of each of these methods can be seen below:

=== "Libraries being defined at initialisation"

	```python
	>>> import pykx as kx
	>>> session = kx.remote.session(port=5050, libraries={'kx': pykx})
	```

=== "Library addition functionality"

	```python
	>>> session.libraries({'np': 'numpy', 'kx': 'pykx'})
	>>> @function(session)
	... def dependent_function(x, y, z):
	...     return kx.q.mavg(4, np.linspace(x, y, z))
	>>> dependent_function(0, 10, 10)
	pykx.FloatVector(pykx.q('0 0.5555556 1.111111 2.222222 3...'))
	```

=== "Defining imports within function body"

	```python
	>>> @function(remote_session)
	... def dependent_function(x, y, z):
	...     import pykx as kx
	...     import numpy as np
	...     return kx.q.mavg(4, np.linspace(x, y, z))
	>>> dependent_function(0, 10, 10)
        pykx.FloatVector(pykx.q('0 0.5555556 1.111111 2.222222 3...'))
	```

While both are valid, we suggest using `#!python libraries` as a method or keyword as it allows for pre-checking of the libraries prior to definition of the function and will be expanded over time to include additional validation.

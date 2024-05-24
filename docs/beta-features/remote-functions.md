# Remote Function Execution

!!! Warning

	This module is a Beta Feature and is subject to change. To enable this functionality for testing please follow the configuration instructions [here](../user-guide/configuration.md) setting `PYKX_BETA_FEATURES='true'`

## Introduction

Remote Functions let you define Python functions within your Python environment which can interact with kdb+ data on a q process. Once defined, these functions are registered to a [remote session object]() along with any Python dependencies which need to be imported. The [remote session object]() establishes and manages the remote connection to the kdb+/q server.

To execute kdb+/q functions using PyKX, please see [PyKX under q](../pykx-under-q/intro.md)

## Requirements and limitations

To run this functionality, the kdb+/q server you connect to must have the ability to load PyKX under q. It is your responsibility to ensure the version and existence of Python library dependencies are correct in your kdb+/q environment at runtime. 

Users must additionally ensure that they have all Python requirements installed on the client server, in particular `dill>=0.2` is required for this functionality.

It can be installed using the following command:

```bash
pip install pykx[beta]
```

## Functional walkthrough

This walkthrough will demonstrate the following steps:

1. Initialize a q/kdb+ server loading PyKX under q on a specified port.
1. Import PyKX and generate a remote session object which denotes the process against which the Python functions will be executed
1. Define a number of Python functions which will be executed on the remote q/kdb+ server.

### Initializing a q/kdb+ server with PyKX under q

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
>>> import os
>>> os.environ['PYKX_BETA_FEATURES'] = 'true'
>>> from pykx.remote import session
>>> remote_session = session()
>>> remote_session.create(host='localhost', port=5050, username='user', password='password')
```

### Defining and Executing Python functions using a session

Tag the Python functions you want to run on the remote server using the `kx.remote.function` decorator. This registers the functions on the `remote_session` object you have just created. 

=== "Single Argument Function"

	```python
	>>> from pykx.remote import function
	>>> @function(remote_session)
	... def single_arg_function(x):
	...     return x+10
	>>> single_arg_function(10)
	pykx.LongAtom(pykx.q('20'))
	```

=== "Multi Argument Function"

	```python
	>>> from pykx.remote import function
	>>> @function(remote_session)
	... def multi_arg_function(x, y):
	...     return x+y
	>>> multi_arg_function(10, 20)
	pykx.LongAtom(pykx.q('30'))
	```

Add any Python libraries which need to be available when executing the function(s) you have just defined. You can achieve this in two ways:

1. Using `session.add_library` to import required libraries before defining your function
1. Importing libraries within the body of the function being executed

Both examples can be seen below

=== "Library addition functionality"

	```python
	>>> remote_session.add_library('numpy', 'pykx')
	>>> @function(remote_session)
	... def dependent_function(x, y, z):
	...     return pykx.q.mavg(4, numpy.linspace(x, y, z))
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

While both are valid, we suggest using `add_library` as it allows for pre-checking of the libraries prior to definition of the function and will be expanded over time to include additional validation.

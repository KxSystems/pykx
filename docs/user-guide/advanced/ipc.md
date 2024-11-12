---
title: Communicate via IPC
description: Use PyKX via IPC
date: June 2024
author: KX Systems, Inc.,
tags: PyKX, IPC, 
---

# Communicate via IPC

_This page explains how to use PyKX to communicate with q processes via IPC._

Interprocess Communication (IPC) forms a central mechanism by which you can connect to and query existing kdb+/q infrastructures.

The processes to which users are connecting and running queries often connect into a central server/gateway that contains vast amounts of historical data.

There are 4 main types of IPC connections in PyKX.

| **Connection Name**                                                   | **When it's often used**                                                                                                                   |
| :-------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| [`kx.SyncQConnection`](../../api/ipc.md#pykx.ipc.SyncQConnection)     | When you need to retrieve data from a server.                                                                                              |
| [`kx.AsyncQConnection`](../../api/ipc.md#pykx.ipc.AsyncQConnection)   | When you need to integrate with Python's `asyncio` library or when integration running queries on an event loop.                           |
| [`kx.SecureQConnection`](../../api/ipc.md#pykx.ipc.SecureQConnection) | When you need to connect to a kdb+/q server which has TLS enabled.                                                                         |
| [`kx.RawQConnection`](../../api/ipc.md#pykx.ipc.RawQConnection)       | Used when more fine-grained control is required by a user to handle when messages are read, also used if emulating a q server from Python. |

In the below sections you will learn more about these connections and how to

- Establish a connection to an existing kdb+/q process
- Run analytics/queries on existing kdb+/q processes
- Reconnect to a process
- Execute a local file
- Integrate with Python asynchronous frameworks
- Create your own IPC Server using PyKX 

!!! Note "To run the examples"

	Before we get started the following sections will make use of a q process running on port 5050.

	To emulate this you can download [this file](scripts/server.py) and run it as follows:

	```python
	>>> import pykx as kx
	>>> import subprocess
	>>> with kx.PyKXReimport():
	...     server = subprocess.Popen(
	...         ('python', 'server.py'),
	...         stdin=subprocess.PIPE,
	...         stdout=subprocess.DEVNULL,
	...         stderr=subprocess.DEVNULL,
	...         )
	...     time.sleep(2)
	```

!!!Warning

	This emulated server is less flexible and performant than a typical q server and as such, for best results use a q process for testing.

Once you're done you can shut down the server as follows

```python
>>> server.stdin.close()
>>> server.kill()
```

## Connect to an existing system

You can connect to processes in two ways

1. Direct connection creation and management
2. Connections established within a `#!python with` statement

The documentation below also shows you how to servers with additional requirements for establishing a connection, such as requiring a username/password or only allowing TLS encrypted connections.

### Connect directly

!!! Tip "Close connections"

	It is best practice to close connections to processes once you have finished with them.

In the below examples you can connect to a process on port 5050 and run a query.

- Establish a connection to the server on port 5050, run a query and close the connection

	```python
	>>> conn = kx.SyncQConnection('localhost', 5050)
	>>> print(conn('1+1').py())
	2
	>>> conn.close()
	```

- Establish a connection using an `#!python kx.AsyncQConnection`, run a query and close the connection

	```python
	>>> conn = await kx.AsyncQConnection('localhost', 5050)
	>>> print(await conn('til 10').py())
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	>>> conn.close()
	```

### Connect using a with statement

To reduce the need to manually open/close connections, use the `#!python with` statement. This will automatically close the connection following execution:

- Query a server on port 5050, run a query and automatically close the connection

	```python
	>>> with kx.SyncQConnection('localhost', 5050) as conn:
	...     print(conn('1+1').py())
	2
	```

- Establish a connection using an `#!python kx.AsyncQConnection`, run a query and automatically close the connection

	```python
	>>> async with kx.AsyncQConnection('localhost', 5050) as conn:
	...     print(await conn('1+1'))
	2
	```

### Connect to a restricted server

You can authenticate on protected servers during connection creation using the optional parameters `username` and `password`

```python
>>> with kx.SyncQConnection('localhost', 5050, username='user', password='pass') as conn:
...     print(conn('1+1').py())
2
```

If establishing a connection to a server where TLS encryption is required you can either use the `tls` keyword when establishing your [`kx.SyncQConnection`](../../api/ipc.md#pykx.ipc.SyncQConnection)/[`kx.AsyncQConnection`](../../api/ipc.md#pykx.ipc.AsyncQConnection) instances, or use an instance of [`kx.SecureQConnection`](../../api/ipc.md#pykx.ipc.SecureQConnection)

=== "Using a sync connection"

	```python
	>>> with kx.SyncQConnection('localhost', 5050, tls=True) as conn:
	...     print(conn('1+1'))
	2
	```

=== "Using a secure connection"

	```python
	>>> with kx.SecureQConnection('localhost', 5050) as conn:
	...     print(conn('1+1'))
	2
	```

## Run analytics on an existing system

Once you have established a connection to your existing system there are various ways that you can run analytics or pass data to the server. The following breaks down the most important approaches

- Call the connection directly
- Using the context interface to call server side functions directly

### Call the connection directly

The most basic method of doing this is through direct calls against the connection object as has been used in the previous section and can be seen as follows:

```python
>>> with kx.SyncQConnection('localhost', 5050) as conn:
...    print(conn('1+1').py())
2
```

In this case any `#!python q` code can be used, for example querying a table on the remote server using qSQL:

```python
>>> with kx.SyncQConnection('localhost', 5050) as conn:
...     print(conn('select from tab where x=`a, x1>0.9995').pd())
   x        x1  x2
0  a  0.999522   3
1  a  0.999996   8
2  a  0.999742   2
3  a  0.999641   6
4  a  0.999515   1
5  a  0.999999   3
```

You can call the connection object with an associated function and supplied parameters, for example:

```python
>>> with kx.SyncQConnection(port=5050) as conn:
...     print(conn('{x+y+z}', 1, 2, 3))
6
```

### Call a named function on the server

Using the "Context Interface", you can call namespaced functions on a remote server. This sends a message before executing a function to validate whether the function being called exists.

In the below examples we will make use of two functions registered on a server. To facilitate this testing you can first set these functions on the server explicitly as follows

```python
>>> with kx.SyncQConnection(port=5050) as conn:
...     conn('.test.addition:{x+y}')
...     conn('.test.namespace.subtraction:{x-y}')
```

Firstly you can call the function `#!python .test.addition` directly:

```python
>>> with kx.SyncQConnection(port=5050) as conn:
...     print(conn.test.addition(4, 2))
6
```

Next you can call the function `#!python .test.namespace.subtraction` which uses a nested namespace:

```python
>>> with kx.SyncQConnection(port=5050) as conn:
...     print(conn.test.namespace.subtraction(4, 2))
2
```

In the case that you do not have access to a named function/namespace you will receive an `#!python AttributeError`:

```python
>>> with kx.SyncQConnection(port=5050) as conn:
...     print(conn.test.unavailable(4, 2))
AttributeError: 'pykx.ctx.QContext' object has no attribute 'unavailable'
QError: '.test.unavailable
```

For more information on the context interface and how to use your q code Python first see [here](context_interface.md)

### Run a local Python function on a server

While not explicitly part of the IPC module of PyKX the ability to run your local Python functions on remote servers makes use of the IPC logic provided by PyKX heavily. Outlined in full detail [here](remote-functions.md), this functionality works by sending across to your server instructions to import relevant libraries, evaluate the function being run and pass data to this function for execution.

In the examples below we can see the registration and use of these functions in practice where the `#!python kx.remote.session` objects are a form of IPC connection. In each case the function is defined in your local session but executed remotely:

=== "Zero argument function"

	```python
	>>> import pykx as kx
	>>> session = kx.remote.session(host='localhost', port=5050)
	>>> @kx.remote.function(session)
	... def zero_arg_function():
	...     return 10
	>>> zero_arg_function()
	pykx.LongAtom(pykx.q('10'))
	```

=== "Single argument function"

	```python
	>>> import pykx as kx
	>>> session = kx.remote.session(host='localhost', port=5050)
	>>> @kx.remote.function(session)
	... def single_arg_function(x):
	...     return x+10
	>>> single_arg_function(10)
	pykx.LongAtom(pykx.q('20'))
	```

=== "Multi argument function"

	```python
	>>> import pykx as kx
	>>> session = kx.remote.session(host='localhost', port=5050)
	>>> @kx.remote.function(session)
	... def multi_arg_function(x, y):
	...     return x+y
	>>> multi_arg_function(10, 20)
	pykx.LongAtom(pykx.q('30'))
	```

## Reconnect to a kdb+ server

When a server with active connections becomes unavailable, restarts, or suffers an outage, all active connections will need to reconnect whenever the server recovers. This could mean closing an existing stale connection and reconnecting using the same credentials.

PyKX allows you to manually configure reconnection attempts for clients connecting to servers using the #!python reconnection_attempts keyword argument. The following example shows the output of when attempting to make use of a connection which has been cancelled and is subsequently re-established:

```python
>>> conn = kx.SyncQConnection(port=5050, reconnection_attempts=5)
>>> conn('1+1')  # after this call the server on port 5050 is shutdown for 2 seconds
pykx.LongAtom(pykx.q('2'))
>>> conn('1+2')
WARNING: Connection lost attempting to reconnect.
Failed to reconnect, trying again in 0.5 seconds.
Failed to reconnect, trying again in 1.0 seconds.
Connection successfully reestablished.
pykx.LongAtom(pykx.q('3'))
```

While configuring `reconnection_attempts` allows you to perform an exponential backoff starting with a delay of 0.5 seconds and multiplying by 2 at each attempt for users wishing to have more control over how reconnection attempts are processed can modify the following keywords

- `reconnection_delay`: The initial delay between the first and second reconnection attempts
- `reconnection_function`: The function/lambda which is used to change the delay between reconnections

As an example take the following where connection which when created sets a delay of 1 second between each connection attempt

```python
>>> conn = kx.SyncQConnection(port=5050, reconnection_attempts=5, reconnection_delay=1, reconnection_function=lambda x:x)
>>> conn('1+1')  # after this call the server on port 5050 is shutdown for 3 seconds
pykx.LongAtom(pykx.q('2'))
>>> conn('1+2')
WARNING: Connection lost attempting to reconnect.
Failed to reconnect, trying again in 1.0 seconds.
Failed to reconnect, trying again in 1.0 seconds.
Failed to reconnect, trying again in 1.0 seconds.
Connection successfully reestablished.
pykx.LongAtom(pykx.q('3'))
```

To read more about reconnection options see the parameters of the [`kx.SyncQConnection`](../../api/ipc.md#pykx.ipc.SyncQConnection) class in the API documentation [here](../../api/ipc.md#pykx.ipc.SyncQConnection).

## Execute a file on a server

In addition to executing code remotely via explicit calls to various [`kx.SyncQConnection`]((../../api/ipc.md#pykx.ipc.SyncQConnection) instances, you can also pass the name of a locally available file to these instances for remote execution. This allows you to package larger code updates as q files for reuse/persistence locally while testing against a remote process.

This is possible provided that the file contains all necessary logic for execution, or the server has the required libraries and associated files to support the execution. In the below examples we will use a file created locally called `file.q` which can be generated as follows:

```python
>>> with open('file.q', 'w') as file:
...     file.write('''
...                .test.namespace.variable:2;
...                .test.namespace.function:{x+y};
...                ''')
```

Here's an example of how to use this functionality on both a synchronous and asynchronous use case.


=== "Synchronous"

	```python
	>>> with kx.SyncQConnection(port = 5050) as q:
	...     q.file_execute('file.q')
	...     ret = q('.test.namespace.variable')
	>>> ret.py()
	2
	```
=== "Asynchronous"

	```python
	>>> async with kx.AsyncQConnection('localhost', 5050) as q:
	...     q.file_execute('file.q')
	...     ret = await q('.test.namespace.function')
	>>> ret
	pykx.Lambda(pykx.q('{x+y}'))
	```


To read more about the file execution API functionality see [here](../../api/ipc.md#pykx.ipc.QConnection.file_execute).

## Communicate asynchronously

When talking about asynchronous communication between `#!python  Python` and `#!python q` there are two ways this can be interpreted, we will deal with these cases separately.

1. Attempting to send Asynchronous messages to a `#!python q` processes which don't expect a response
2. Integrating IPC workflows with Python's `#!python asyncio` library

### Send messages without expecting a response
 
To send messages to a q process without a response you do _not_ need to use a [`kx.AsyncQConnection`](../../api/ipc.md#pykx.ipc.AsyncQConnection) instance, sending messages to a q process without anticipation of response is facilitated through the `#!python wait` keyword which should be set to `#!python False` in the case you are not expecting a response from the q server. Calls made with this keyword set will return `#!python pykx.Identity` objects

```python
>>>  with kx.SyncQConnection('localhost', 5050) as q:
...      ret = q('1+1', wait=False)
>>> ret
pykx.Identity(pykx.q('::'))
```

### Integrate with Python Async libraries

To make integrate with Python's async libraries such as `#!python asyncio` with `#!python PyKX`, you must use a [`kx.AsyncQConnection`](../../api/ipc.md#pykx.ipc.AsyncQConnection). When calling an instance of an [`kx.AsyncQConnection`](../../api/ipc.md#pykx.ipc.AsyncQConnection), the query is sent to the `#!python q` server and control is immediately handed back to the running Python program. The `#!python __call__` function returns a [`kx.QFuture`](../../api/ipc.md##pykx.ipc.QFuture) instance that can later be awaited on to block until it receives a result.

If you're using a third-party library that runs an eventloop to manage asynchronous calls, ensure you use the `#!python event_loop` keyword argument to pass the event loop into the [`kx.AsyncQConnection`](../../api/ipc.md#pykx.ipc.AsyncQConnection) instance. This allows the eventloop to properly manage the returned [`kx.QFuture`](../../api/ipc.md##pykx.ipc.QFuture) objects and its lifecycle.

```python
async with kx.AsyncQConnection('localhost', 5001, event_loop=asyncio.get_event_loop()) as q:
    fut = q('til 10') # returns a QFuture that can later be awaited on, this future is attached to the event loop
    await fut # await the future object to get the result
```

If you're using a [`kx.AsyncQConnection`](../../api/ipc.md#pykx.ipc.AsyncQConnection) to make q queries that respond in a [deferred manner](https://code.kx.com/q/basics/ipc/#async-message-set), you must make the call using the `#!python reuse=False` parameter. This parameter helps to make the query over a dedicated `#!python pykx.AsyncQConnection` instance that is closed upon the result being received.

```python
async with kx.AsyncQConnection('localhost', 5001, event_loop=asyncio.get_event_loop()) as q:
    fut = q('query', wait=False, reuse=False) # query a q process that is going to return a deferred result
    await fut # await the future object to get the result
```

## Create your own IPC Server using PyKX

There are several cases where providing the ability for users to open IPC connections to Python processes via the q native IPC protocol provides advantages. In particular if you are looking to manage infrastructure in Python which kdb+ users are likely to communicate with using q.

The [`server.py`](scripts/server.py) file that you may have called at the start of this page makes use of this functionality and specifically uses a [`kx.RawQConnection`](../../api/ipc.md#pykx.ipc.RawQConnection) to allow connections to be made, this script is defined in plain text as follows:

```python
import asyncio
import sys


import pykx as kx

port = 5010
if len(sys.argv)>1:
    port = int(sys.argv[1])


def qval_sync(query):
    res = kx.q.value(query)
    print("sync")
    print(f'{query}\n{res}\n')
    return res


def qval_async(query):
    res = kx.q.value(query)
    print("async")
    print(f'{query}\n{res}\n')


async def main():
    kx.q.z.pg = qval_sync
    kx.q.z.ps = qval_async
    kx.q('@[system"l ",;"s.k_";{show "Failed to load SQL"}]')
    kx.q('tab:([]1000?`a`b`c;1000?1f;1000?10)')
    async with kx.RawQConnection(port=port, as_server=True, conn_gc_time=20.0) as q:
        print('Server Initialized')
        while True:
            q.poll_recv()


if __name__ == "__main__":
    asyncio.run(main())
```

Notably the definition of [`kx.RawQConnection`](../../api/ipc.md#pykx.ipc.RawQConnection) uses the keyword `#!python as_server=True` to indicate that it should anticipate external connections, and the tight while loop running `#!python q.poll_recv` will manage the execution of incoming queries. It is also worth noting that in the definition of the `#!python main` function that you can set and specify both the `#!python kx.q.z.pg` and `#!python kx.q.z.ps` functions which manage how messages are handled in synchronous and asynchronous cases.

For a full breakdown on `#!python kx.RawQConnection` type connections see [here](../../api/ipc.md#pykx.ipc.RawQConnection)

## Next Steps

- [Deep dive into how to execute Python functions remotely](remote-functions.md)
- [Create your first database](database/db_gen.md)
- [Query data using Python](../fundamentals/query/pyquery.md)

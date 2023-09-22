# Communicating via IPC

q IPC connections are often used to connect into a central server / gateway that contains large amounts
of historical data. PyKX `QConnection` instances provide a way to connect into these servers and directly query
the data within them. This allows users to access data within a running q process, optionally convert it into
a Python object and then locally perform analysis / transformations to the data within python. For licensed users
the local object can be used within embedded q, for unlicensed users they will first have to convert it to a
python type with one of the helper methods (`.py()`/`.np()`/`.pd()`/`.pa()`). This allows users to get
the best of both worlds where they can harness the power of q as well as the power of other existing python
libraries to perform analysis and modifications to q data.

## Modalities of use for IPC

Using the IPC module is available to both `licensed` and `unlicensed` users. Using a QConnection instance
is the only way for an unlicensed user to run `q` code directly within PyKX. When using a
`QConnection` instance in unlicensed mode you must convert the resulting value back into a python
type before it is usable. In licensed mode the resulting value can be directly modified and used
within Embedded Q without first converting it. For both licensed and unlicensed users this module can be
used to replace the functionality of [`qPython`](https://github.com/exxeleron/qPython).

```python
# Licensed mode
with pykx.SyncQConnection('localhost', 5001) as q:
    result = q('til 10')
    print(result)
    print(result.py())

0 1 2 3 4 5 6 7 8 9
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

```python
# Unlicensed mode
with pykx.SyncQConnection('localhost', 5001) as q:
    result = q('til 10')
    print(result)
    print(result.py())

pykx.LongVector._from_addr(0x7fcab6800b80)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Methods of Instantiating

There are two ways to create each subclass of [`pykx.QConnection`][], the first is to directly instantiate
the connection instance and the second option is to create them in the form of a context interface. Using
the context interface method of declaring these [`pykx.QConnection`][] instances should be preferred as it will
ensure that the connection instance is properly closed automatically when leaving the scope of the context.

Manually creating a `QConnection`

```python
q = pykx.SyncQConnection('localhost' 5001) # Directly instantiate a QConnection instance
q(...) # Make some queries
q.close() # Must manually ensure it is closed when no longer needed
```

Using a context interface to create and manage the `QConnection`

```python
with pykx.SyncQConnection('localhost', 5001) as q:
    q(...) # Make some queries
# QConnection is automatically closed here
```

## Performance Considerations

When querying [`pykx.Table`][] instances on the remote process you should avoid directly calling the table object as
that will result in the entirety of the table being sent over IPC and then loaded within the `Python` process.
You should ensure that when querying tables over IPC that you are applying sufficient filters to your query, 
so that you limit the amount of data being converted and transfered between processes.

## Execution Contexts

Functions pulled in over IPC execute locally within PyKX by default using embedded q.
[Symbolic functions][pykx.SymbolicFunction] can be used to execute in a different context instead,
such as over IPC in the q instance where the function was originally defined. The
[context interface](../../api/pykx-execution/ctx.md) provides symbolic functions for all functions accessed through it by
default.

In the following example, `q` is a [`pykx.QConnection`][] instance.

The following call to the q function [`save`](../../api/pykx-execution/q.md#save) executes locally using embedded q,
because `q('save')` returns a regular [`pykx.Function`][] object.

```python
with pykx.SyncQConnection('localhost', 5001) as q:
    q('save')('t') # Executes locally within Embedded q
```

When [`save`](../../api/pykx-execution/q.md#save) is accessed through the [context interface](../../api/pykx-execution/ctx.md), it is a
[`pykx.SymbolicFunction`][] object instead, which means it is simultaneously an instance of
[`pykx.Function`][] and [`pykx.SymbolAtom`][]. When it is executed, the function retrived within
its execution context using its symbol value, and so it is executed in the q server where
[`save`](../../api/pykx-execution/q.md#save) is defined.

```python
with pykx.SyncQConnection('localhost', 5001) as q:
    q.save('t') # Executes in the q server over IPC
```

Alternatively, one can simply access & use the function by name manually within a single query.
This differs from the first case because the query includes the argument for [`save`](../../api/pykx-execution/q.md#save),
and so what is returned is the result of calling [`save`](../../api/pykx-execution/q.md#save) with the argument `t`,
rather than the [`save`](../../api/pykx-execution/q.md#save) function itself.

```python
with pykx.SyncQConnection('localhost', 5001) as q:
    q('save', 't') # Executes in the q server over IPC
```

## Asynchronous Execution

In order to make asynchronous queries to `q` with `PyKX` a [`pykx.AsyncQConnection`][] must be used. When an
instance of an [`pykx.AsyncQConnection`][] is called the query will be sent to the `q` server and control
will be immediately handed back to the running Python program. The `__call__` function returns a
[`pykx.QFuture`][] instance that can later be awaited on to block until a result has been received.

If you are using a third party library that runs an eventloop to manage asynchronous calls, you must ensure
you use the `event_loop` keyword argument to pass the event loop into the [`pykx.AsyncQConnection`][] instance.
This will allow the eventloop to properly manage the returned [`pykx.QFuture`][] objects.

```python
async with pykx.AsyncQConnection('localhost', 5001, event_loop=asyncio.get_event_loop()) as q:
    fut = q('til 10') # returns a QFuture that can later be awaited on, this future is attached to the event loop
    await fut # await the future object to get the result
```

If you are using an [`pykx.AsyncQConnection`][] to make q queries that respond in a [deferred manner](https://code.kx.com/q/basics/ipc/#async-message-set)
, you must make the call using the `reuse=False` parameter. By using this parameter the query will be made over
a dedicated [`pykx.AsyncQConnection`][] instance that is closed upon the result being received.

```python
async with pykx.AsyncQConnection('localhost', 5001, event_loop=asyncio.get_event_loop()) as q:
    fut = q('query', wait=False, reuse=False) # query a q process that is going to return a deferred result
    await fut # await the future object to get the result
```

## File Execution

In addition to the ability to execute code remotely using explicit calls to the various [`pykx.QConnection`][] instances, it is also possible to pass to these instances the name of a file available locally which can be executed on the remote server. This is supported under the condition that the file being executed remotely contains all of the required logic to be executed, or the server contains sufficient libraries and associated files to allow execution to occur.

The following provide and example of the usage of this functionality on both a syncronous and asyncronous use-case.

```python
with pykx.SyncQConnection(port = 5000) as q:
    q.file_execute('/absolute/path/to/file.q')
    ret = q('.test.variable.set.in.file.q', return_all=True)
```

```python
async with pykx.AsyncQConnection('localhost', 5001) as q:
    q.file_execute('../relative/path/to/file.q')
    ret = await q('.test.variable.set.in.file.q')
```

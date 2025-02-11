---
title:  PyKX as q server
description: PyKX as q server example 
date: October 2024
author: KX Systems, Inc.,
tags: PyKX, q, server
---

# Use PyKX as a `#!python q` Server

_This example provides a quick start for setting up PyKX as a `#!python q` server that other `#!python q` and PyKX sessions can connect to._

To follow along, feel free to download this <a href="./archive.zip" download>zip archive</a> that contains a copy of the python script and this writeup.

## Quick start

To run this example, run the `#!python server.py` script to launch a `#!python PyKX` server on port 5000. Alternatively, run `#!python server_async.py` to run an asynchronous version of the server. 

The server prints out any queries it receives as well as the result of executing the query before replying.

```bash
python server.py
// or
python server_async.py
```
## Extra configuration options

### User validation

You can add a function to validate users when they try to connect to the server. You can do so by overriding the `#!python .z.pw` function. By default all connection attempts will be accepted.

The function receives 2 arguments when a user connects:

 - username
 - password (if no password is provided `#!python None`/`#!python ::` will be passed in place of a password).

!!! note "Important! You need to override the function using `#!python EmbeddedQ` not on the q connection."

Here is an example of overriding it using a Python function as a validation function:

```python
def validate(user, password):
    if password == 'password':
        return True # Correct password allow the connection
    return False # Incorrect password deny the connection

kx.q.z.pw = validate
```

Here is an example of overriding it using a q function as a validation function:

```q
kx.q.z.pw = kx.q('{[user; password] $[password=`password; 1b; 0b]}')
```

### Message handler

You can override the message handler to apply custom logic to incoming queries. By default, it returns
the result of calling `#!python kx.q.value()` on the incoming query. This function will be passed a `#!python CharVector`
containing the incoming query.

!!! note "Important! You need to override the function using `#!python EmbeddedQ` not on the q connection."

Here is an example of overriding it using a Python function as a message handler:

```python
def qval(query):
    res = kx.q.value(query)
    print(f'{query}\n{res}\n')
    return res

kx.q.z.pg = qval
```

Here is an example of overriding it using a q function as a message handler:

```q
kx.q.z.pg = kx.q('{[x] show x; show y: value x; y}')
```

For async messages, manage `#!python kx.q.z.ps` in the same fashion.

### Connection garbage collection frequency

One of the keyword arguments to use when creating a server is `#!python conn_gc_time`. This argument takes
a float as input and the value denotes how often the server will attempt to clear old closed connections.

By default the value is `#!python 0.0` and this will cause the list of connections to be cleaned on every call
to `#!python poll_recv`. With lots of incoming connections, this can deteriorate the performance. If you
set the `#!python conn_gc_time` to `#!python 10.0` then this clean-up happens every 10 seconds.

!!! Note

    [reval](../../api/pykx-execution/q.md#reval) will not impose read only exection on a PyKX server as Python manages the sockets rather than `q`.
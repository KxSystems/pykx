# Using PyKX as a `q` Server

The purpose of this example is to provide a quick start for setting up PyKX as a `q` server that other
`q` and PyKX sessions can connect to.

To follow along with this example please feel free to download this <a href="./archive.zip" download>zip archive</a> that contains a copy of the python script and this writeup.

## Quick start

To run this example simply run the `server.py` script and it will launch a `PyKX` server on port 5000 or
you can run `server_async.py` to run an asyncronous version of the server.
The server will print out any queries it receives as well as the result of executing the query before replying.

```bash
python server.py
// or
python server_async.py
```

## Extra Configuration Options

### User Validation

It is possible to add a function to validate users when they try to connect to the server. This can
be done by overriding the `.z.pw` function. By default all connection attempts will be accepted.

The function will be passed 2 arguments when a user connects, the first will be the username, and the
second will be the password (if no password is provided `None`/`::` will be passed in place of a password).

Note: The function needs to be overridden using `EmbeddedQ` not on the q connection.

Here is an example of overriding it using a python function as a validation function.

```python
def validate(user, password):
    if password == 'password':
        return True # Correct password allow the connection
    return False # Incorrect password deny the connection

kx.q.z.pw = validate
```

Here is an example of overriding it using a q function as a validation function.

```q
kx.q.z.pw = kx.q('{[user; password] $[password=`password; 1b; 0b]}')
```

### Message Handler

The message handler can be overridden to apply custom logic to incoming queries. By default it just returns
the result of calling `kx.q.value()` on the incoming query. This function will be passed a `CharVector`
containing the incoming query.

Note: The function needs to be overridden using `EmbeddedQ` not on the q connection.

Here is an example of overriding it using a python function as a message handler.

```python
def qval(query):
    res = kx.q.value(query)
    print(f'{query}\n{res}\n')
    return res

kx.q.z.pg = qval
```

Here is an example of overriding it using a q function as a message handler.

```q
kx.q.z.pg = kx.q('{[x] show x; show y: value x; y}')
```

For async messages `kx.q.z.ps` can be managed in the same fashion.

### Connection Garbage Collection Frequency

One of the keyword arguments you can use when creating a server is `conn_gc_time` this argument takes
a float as input and the value denotes how often the server will attempt to clear old closed connections.
By default the value is 0.0 and this will cause the list of connections to be cleaned on every call
to `poll_recv`, with lots of incoming connections this can cause performance to deteriorate. If you
set the `conn_gc_time` to `10.0` then this clean-up will happen at most every 10 seconds.

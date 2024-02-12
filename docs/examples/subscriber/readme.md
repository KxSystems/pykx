# PyKX Subscribing to a `q` Process

The purpose of this example is to provide a quickstart for setting up a python process using `PyKX` to subscribe to a running q process.

To follow along with this example please feel free to download this <a href="./archive.zip" download>zip archive</a> that contains a copy of the python script and this writeup.

## Quickstart

This example creates a python subscriber to a q process, that appends data received to the end of a table.

Here we have:

1. A q process running on port 5001
2. A Python process subscribing to the q process

### Start the required q processes

```q
// run q
$ q -p 5001
q)
```

### Start the pykx subscriber

```bash
// run the subscriber which will automatically connect
$ python subscriber.py
// you can also run the asnychronous example with
$ python subscriber_async.py
```

### Outcome

What should be observed on invocation of the above is that the q process should have the variable `py_server` set to the handle of the python process once the python process connects. Once this variable is set you can send rows of the table to the python process and they will be appended as they are recieved.

```q
// run q
$ q -p 5001
q)
```

q process is started.

```bash
// run the subscriber which will automatically connect
$ python subscriber.py
===== Initital Table =====
a b
---
4 8
9 1
2 9
7 5
0 4
1 6
9 6
2 1
1 8
8 5
===== Initital Table =====

```

Python process is started with a table, and it connects to the q server and sets the `py_server` variable.

```q
q)py_server[1 2]

```

Send a new table row (1, 2) to the python process from q.

```python
Recieved new table row from q: 1 2
a b
---
4 8
9 1
2 9
7 5
0 4
1 6
9 6
2 1
1 8
8 5
1 2
```

The new row has been appended to the table.

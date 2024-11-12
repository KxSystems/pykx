---
title:  PyKX installation guide
description: Subscriber Examples
date: October 2024
author: KX Systems, Inc.,
tags: subscriber, synchronous, asynchronous, PyKX
---
# PyKX Subscribing to a `q` Process

_This example demonstrates using `PyKX` to setup a python process as a subscriber to data messages published from a q process._

## Pre-requisites

A kdb+ license is required to complete this example. [Sign-up for a license](https://code.kx.com/q/learn/licensing/).

The following python libraries are required to run this example:

1. pykx
1. asyncio

The source code for this example is available in the examples directory here:

1. [Synchronous subscriber](https://github.com/KxSystems/pykx/blob/main/examples/subscriber/subscriber.py)
1. [Asynchronous subscriber](https://github.com/KxSystems/pykx/blob/main/examples/subscriber/subscriber_async.py)

## Summary of steps

Both example scripts for setting up a subscriber follow the same steps:

1. Start a q process running with some open port (5001 is used for the example, but you may choose any open port).
1. Run the python subscriber by executing the script from the github repository.

### Run the subscriber example

1. Begin by running a q process with an open port:

    ```q
    // run q
    $ q -p 5001
    q)
    ```
1. In a separate terminal start a python process running the subscriber script:

    ```bash
    // run the subscriber, which connects automatically
    $ python subscriber.py
    ```
    The python process opens an IPC connection to the q process and sets a new global variable on the q process as part of the main function:

    ```q
        async def main():
            global table
            async with kx.RawQConnection(port=5001) as q:
                print('===== Initial Table =====')
                print(table)
                print('===== Initial Table =====')
                await q('py_server:neg .z.w')
                await main_loop(q)
    ```
    The q process now has the variable `py_server` set to the handle of the python process once the python process connects. 
    
1. Once this variable is set, you can send rows of the table to the python process and they are appended as they are received.

    ```bash
    // run the subscriber, which automatically connects
    $ python subscriber.py
    ===== Initial Table =====
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
    ===== Initial Table =====

    ```

1. As the Python process is initiated, it connects to the q server and sets the `py_server` variable and creates the initial table.

    ```q
    q)py_server[1 2]

    ```

1. Send a new table row (1, 2) to the python process from q.

    ```python
    Received new table row from q: 1 2
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

### Run the asynchronous subscriber example

1. Begin by running a q process with an open port:

    ```q
    // run q
    $ q -p 5001
    q)
    ```
1. In a separate terminal start a python process running the asynchronous subscriber script:

    ```bash
    // run the asynchronous subscriber which automatically connects
    $ python subscriber_async.py
    ```
     The python process opens an IPC connection to the q process and sets a new global variable on the q process as part of the main function:

    ```q
        async def main():
            global table
            async with kx.RawQConnection(port=5001) as q:
                print('===== Initial Table =====')
                print(table)
                print('===== Initial Table =====')
                await q('py_server:neg .z.w')
                await main_loop(q)
    ```
    The q process now has the variable `py_server` set to the handle of the python process once the python process connects. 
    
1. Once this variable is set, you can send rows of the table to the python process and they are appended as they are received.

    ```bash
    // run the subscriber, which automatically connects
    $ python subscriber_async.py
    ===== Initial Table =====
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
    ===== Initial Table =====

    ```

1. As the Python process is initiated, it connects to the q server and sets the `py_server` variable and creates the initial table.

    ```q
    q)py_server[1 2]

    ```

1. Send a new table row (1, 2) to the python process from q.

    ```python
    Received new table row from q: 1 2
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


## Summary

This example has demonstrated how to initiate a q process, subscribe to an existing table, and append rows to it either synchronously or asynchronously.

## Next steps

Check out more examples such as:

- [Real-Time Streaming]
- [Compression and Encryption]

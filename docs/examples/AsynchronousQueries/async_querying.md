---
title: Asynchronous Querying Example
date: July 2025
author: KX Systems, Inc.
tags: PyKX, q, asyncio, IPC, asynchronous
---

# PyKX Calling into multiple q servers without blocking

_This example provides a quick start for setting up a Python process using `PyKX` to call into 
multiple q servers without blocking each other._

To follow along, feel free to download this <a href="./archive.zip" download>zip archive</a> that 
contains a copy of the python scripts and this writeup.

## Quickstart

This example creates a python process that sends 2 queries meant to simulate long running queries to 
two separate q servers to show how to query q servers without blocking using `PyKX`.

### Run the Example

The example uses 2 servers opened up on ports 5050 and 5051, these servers can be opened with the 
commands `$ q -p 5050` and `$ q -p 5051` respectively.
The script `async_query.py` can then be run to send the two queries simultaneously with 
`$ python async_query.py`.

### Outcome

The script will send the two queries and print their results followed by the total time taken 
by the script.

The first query takes 10 seconds and the second takes 5 seconds to complete showing that both 
queries were processed without blocking eachother.

```bash
0 1 2 3 4 5 6 7 8 9 10 11 12
0 1 2 3 4 5 6 7 8 9
took 10.001731808 seconds
```

### Important notes on usage of `QConnections`

While the `#!python with` syntax for `QConnection` objects is useful for sending one shot requests it 
should be avoided where possible when repeatedly querying the same server. This is because 
connecting to q servers is blocking and the closing of `QConnection` objects is also blocking which 
will cause other queries to be delayed in certain cases. There is a simple class called 
`ConnectionManager` provided in this example to handle opening connections to servers and allow 
querying them by supplying a port alongside the query and any arguments. This class will also clean 
up the stored `QConnection` objects when its `#!python with` block ends, much like the normal
`QConnection` objects do themselves.

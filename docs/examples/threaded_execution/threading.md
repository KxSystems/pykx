---
title: Multithreaded Execution Example
description: Example of PyKX Calling into q from multiple threads
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, threading, python, asyncio, multithreaded
---

# PyKX Calling into q from multiple threads

_This example provides a quick start for setting up a Python process using `#!python PyKX` to call into `#!python EmbeddedQ` from multiple threads._

To follow along, feel free to download this <a href="./archive.zip" download>zip archive</a> that contains a copy of the python scripts and this writeup.

## Quickstart

This example creates a Python process that spawns multiple tasks or threads to subscribe to a `#!python q` process over IPC. Upon receiving a new row, it upserts the row to a local table. There are two scripts included: 

- `#!python asyncio_threading.py`, which uses asyncio tasks running on separate threads.
- `#!python threads.py`, which uses the Python threading library to spawn threads directly.

### Run the example

```bash
$ python asyncio_threading.py
// or
$ python threads.py
```

### Outcome

This command prints the initial table at startup. Once all the threads or tasks have upserted their received rows to the table, it prints the final table:

```
$ python asyncio_threading.py
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
a  b
-----
4  8
9  1
2  9
7  5
0  4
1  6
9  6
2  1
1  8
8  5
7  63
11 13
80 89
43 50
96 35
35 83
28 31
96 12
83 16
77 33
..
```

### Important note on usage

Since using `#!python PYKX_THREADING` creates a background thread to run the calls into `#!python q`, the
background thread must be shutdown when finished. The easiest way to ensure this is done is by using
a `#!python try` - `#!python finally` block around the entrypoint to your script. This ensures that even in the
event of an error, the background thread shuts down correctly so Python can exit.

```
import os
os.environ['PYKX_THREADING'] = '1'
os.environ['PYKX_BETA_FEATURES'] = '1'
import pykx as kx

def main():
    ...

    
if __name__ == '__main__':
    try:
        main()
    finally:
        # Must shutdown the background thread to properly exit
        kx.shutdown_thread()
```

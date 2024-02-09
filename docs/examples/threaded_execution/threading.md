# PyKX Calling into q from multiple threads

The purpose of this example is to provide a quickstart for setting up a python process using `PyKX`
to call into `EmbeddedQ` from multiple threads.

To follow along with this example please feel free to download this
<a href="./archive.zip" download>zip archive</a> that contains a copy of the python scripts and this
writeup.

## Quickstart

This example creates a python process that creates multiple tasks/threads that subscribe to a `q`
process over IPC and upon recieving a new row upsert it to a local table. There are 2 scripts
included: `asyncio_threading.py` and `threads.py`, the first uses asyncio tasks running on
seperate threads and the second example uses the python `threading` library directly to spawn
threads.


### Running the example

```bash
$ python asyncio_threading.py
// or
$ python threads.py
```

### Outcome

The inital table will be printed upon starting the program, once all the threads/tasks have
upserted all of the rows they have recieved to the table the final table will be printed.

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

### Important Note on usage

Since using `PYKX_THREADING` creates a background thread to run the calls into `q`, the
background thread must be shutdown when finished. The easiest way to ensure this is done is by using
a `try` - `finally` block around the entrypoint to your script. This will ensure that even in the
event of an error the background thread will still be shutdown correctly so python can exit.

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

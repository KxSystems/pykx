# PyKX Calling into q from multiple threads

The purpose of this example is to provide a quickstart for setting up a python process using `PyKX`
to call into EmbeddedQ from multiple threads.

To follow along with this example please feel free to download this
<a href="./archive.zip" download>zip archive</a> that contains a copy of the python script and this
writeup.

## Quickstart

This example creates a python process that calls into `q` from multiple threads. When using the
included `pykxthreading` library these threads will be able to modify state when calling into `kx.q`.
The base `EmbeddedQ` object within PyKX normally only allows the main thread to make these state
changes.


### Start the PyKX threaded execution example

```bash
$ python threaded_execution.py
```

### Outcome

In this simple example the output of `kx.q.til(...)` will be output to the console, where each thread
is given a different number of elements to print.

```
$ python threaded_execution.py
0 1
0 1 2
0 1 2 3 4
0 1 2 3 4 5 6
0 1 2 3
0 1 2 3 4 5 6 7
0 1 2 3 4 5
0 1 2 3 4 5 6 7 8
0 1 2 3 4 5 6 7 8 9
0 1 2 3 4 5 6 7 8 9 10 11
0 1 2 3 4 5 6 7 8 9 10
```

### Important Note on usage

Since the `pykxthreading` library creates a background thread to run the calls into `EmbeddedQ`, the
background thread must be shutdown when finished. The easiest way to ensure this is done is by using
a `try` - `finally` block around the entrypoint to your script. This will ensure that even in the
event of an error the background thread will still be shutdown correctly so python can exit.

```
def main():
    ...

    
if __name__ == '__main__':
    try:
        main()
    finally:
        # Must shutdown the background thread to properly exit
        shutdown_q()
```

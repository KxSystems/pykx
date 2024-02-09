# Multi-Threaded Execution

!!! Warning

	This module is a Beta Feature and is subject to change. To enable this functionality for testing please follow the configuration instructions [here](../user-guide/configuration.md) setting `PYKX_BETA_FEATURES='true'` and `PYKX_THREADING='true'`.

## Introduction

One major limitation of `EmbeddedQ` when using python with multi-threading is that only the main 
thread (the thread that imports PyKX and loads `libq`) is allowed to modify state within `EmbeddedQ`.
However if you wanted to use one of Pythons multi-threading libraries whether that is the `threading`
library or `asyncio` or any other library that allows Python to utilise multiple threads at once, 
and have those threads modify state in some way; whether that be to upsert a row to a global table,
open `QConnection` instances or any other use case that requires the threads to modify state. You 
would not be able to do that by default in PyKX.

This beta feature allows these use cases to become possible by spawning a background thread that all 
calls into `EmbeddedQ` will be run on. This background thread is created at the `C` level using 
`libpthread` with lightweight future objects to ensure the lowest overhead possible for passing 
calls onto a secondary thread. This allows multi-threaded programs to modify state within the spawned 
threads safely, without losing out on performance.


!!! Note
    
    While using `PyKX Threading` it is not possible to also use the functionality within `pykx.q`,
    it is also not possible to have q call back into Python.

## How to enable

This beta feature requires an extra opt-in step. While the overhead for offloading calls onto a secondary 
thread is low, there will always be a cost to forcing a thread context switch to process a call into 
`EmbeddedQ`. Therefore you will need to enable both the `PYKX_BETA_FEATURES` environment variable as 
well as the `PYKX_THREADING` environment variable.

!!! Warning

    Because using `PyKX Threading` spawns a background thread to run all queries to `EmbeddedQ`, you
    must ensure that you call `kx.shutdown_thread()` at the end of your script to ensure that this
    background thread is properly shutdown at the end. If you fail to do this the background thread will
    be left running after the script is finished. The best way to ensure this always happens is to start
    a main function for your script within a `try` - `finally` block.


```Python
import os
import asyncio
os.environ['PYKX_THREADING'] = '1'
os.environ['PYKX_BETA_FEATURES'] = '1'
import pykx as kx

def main(): # Your scripts entry point
    ...

if __name__ == '__main__':
    try:
        main()
    finally:
        kx.shutdown_thread() # This will be called if the script completes normally or errors early
```

## More complete examples

More examples showing this functionality in use can be found  [here](../examples/threaded_execution/threading.md).

---
title: Multithreaded Execution
description: Learn how multithreaded integration for PyKX works
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, threading, python, asyncio, multithreaded, noupdate
---

# Multithreaded execution

_This page explains how to execute q code on multiple threads._

When used in its default configuration, PyKX does not support multithreaded execution of q code on Python threads. This limitation arises because only the main thread (the one importing PyKX and loading `#!python libq`) can modify the state in PyKX’s assigned memory.

As a result, PyKX’s integration with Python’s multithreading libraries, such as [`#!python threading`](https://docs.python.org/3/library/threading.html), [`#!python asyncio`](https://docs.python.org/3/library/asyncio.html), is restricted. This also affects other Python libraries that utilize multiple threads simultaneously, including [`#!python streamlit`](https://streamlit.io/), which uses multiple threads to manage data caching. Read more information about [PyKX’s integration with Streamlit](streamlit.md).

Use cases for multithreading with PyKX:

- **Upserting Data**: Insert or update data in a global table from multiple sources.
- **Querying Multiple Processes**: Open `#!python QConnection` instances to query several processes simultaneously and combine their results.

If you don’t configure PyKX for multithreading, you might encounter a `#!python noupdate` error. To avoid this, consider enabling the feature described here. This feature allows multithreading by creating a background thread that `#!python loads` libq. 

All calls to q from other threads are run on this background thread, created using `#!python libpthread` for minimal overhead. This setup enables safe state modification in multithreaded programs with minimal performance impact.

## Before enabling

Before globally enabling this functionality, consider the following:

- **Concurrency Cost**: While the overhead for offloading calls onto a secondary thread is low, there will always be a cost in forcing a thread context switch. As such single-threaded performance is faster at the cost of concurrency.
- **Memory-Safe Use**: While using `#!python PYKX_THREADING` it's not possible nor memory safe to have `#!python q` call back into Python; this could result in memory corruption or side-effects which may not be immediately obvious.
- **Shutdown**: When using `#!python PYKX_THREADING`, it creates a background thread for running queries to `#!python q`. Make sure to call `#!python kx.shutdown_thread()` at the end of your script to properly close this thread. If you don’t, the thread will remain running in the background after the script finishes. To avoid this, it’s best to start your `#!python main` function within a `#!python try` - `#!python finally` block.

## How to enable multithreaded execution

By default, PyKX doesn't start with multithreading support enabled. To enable this feature, you must set `#!python PYKX_THREADING=True` during [configuration](../configuration.md). You can do this either as an environment variable or by adding this configuration to a `#!python .pykx-config` file as outlined [here](../configuration.md#configuration-file).

## Example usage

The following example shows the basic structure suggested for using this functionality:

```Python
import os
import asyncio
os.environ['PYKX_THREADING'] = '1'
import pykx as kx

def main(): # Your scripts entry point
    ...

if __name__ == '__main__':
    try:
        main()
    finally:
        kx.shutdown_thread() # This will be called if the script completes normally or errors early
```

- A more complete worked example can be found [here](../../examples/threaded_execution/threading.md).

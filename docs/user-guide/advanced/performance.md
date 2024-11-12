---
title: PyKX Performance 
description: How to optimize PyKX 
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, perfromance, paralellization, secondary q threads, multithreading, peach
---

# Performance tips

_This page includes PyKX performance optimization tips, including insights on parallelization, secondary q threads, multithreading, and peach._

To get the best performance out of PyKX, follow these guidelines. Note that this page focuses on efficiently interfacing between Python and q, rather than optimizing Python or q individually.

## General guidelines

1. **Avoid Unnecessary Conversions**. Avoid converting K objects with their `#!python .py`/`#!python .np`/`#!python .pd` methods unless necessary. Often, the K object itself is sufficient.

1. **Avoid Nested Columns** when converting q tables into Pandas dataframes, as this currently incurs data copy.

1. **Do as Little Work as Necessary**. Convert only what is needed. For example, instead of converting an entire q table to a dataframe, convert only the required columns into Numpy arrays by indexing into the [`#!python pykx.Table`][pykx.Table] and calling `#!python .np` on the columns. Use select statements and indexing to send only the necessary subset of data over an IPC connection.

1. **Prefer `#!python .np` and `#!python .pd` Over `#!python .py`**. Use Numpy/Pandas conversions to avoid data copying where possible. Converting objects with `.py` always incurs a data copy and may not always be possible (for example, some K objects return themselves when `.py` is called, such as [`pykx.Function`][pykx.Function]) instances.

1. **Use `#!python raw=True` for Performance**. When performance is more important than the richness of the output, use the `#!python raw=True` keyword argument. This can be more efficient by skipping certain adjustments, such as:

    - Temporal epoch adjustments from `#!python 2000-01-01` to `#!python 1970-01-01`.
    - Converting q `#!python GUIDs` to Python `#!python UUID` objects (they will come through as complex numbers instead).
    - Converting bytes into strings.

1. **Let q do the heavy lifting.** When using licensed mode, prefer q code and functions (like `#!python q.avg`, `#!python q.sdev`) over pure Python code. This is similar to using Numpy functions for Numpy arrays instead of pure Python.

    - Numpy functions on K vectors converted to Numpy arrays perform well, even with conversion overhead.
    - When using an IPC connection to a remote q process, use q code to pre-process data and reduce the workload on Python.
    - Avoid converting large data from Python to q. Conversions from q to Python (via Numpy) often avoid data copying, but Python to q conversions always copy the data.

## Parallelization

Parallelization involves distributing computational tasks across multiple threads to improve performance and efficiency. 
Use the following methods if you want to allow PyKX to handle large-scale data processing tasks efficiently by utilizing the available computational resources: secondary q threads, multithreading, or `#!python peach`.

### Secondary q threads

PyKX starts embedded q with as many secondary q threads enabled as are available. q automatically uses these threads to parallelize some computations as it deems appropriate. You can use the `#!python QARGS` environment variable to provide command-line arguments and other startup flags to q/PyKX, including the number of secondary threads:

```sh
QARGS='-s 0' python # disable secondary threads
```

```sh
QARGS='-s 12' python # use 12 secondary threads by default
```

- The value set using `#!python -s` sets both the default and the maximum available to the process; you can't change it after importing PyKX.
- `#!python pykx.q.system.max_num_threads` shows the maximum number of threads and cannot be changed.
- `#!python pykx.q.system.num_threads` shows the current number of threads in use. It starts at the maximum value but can be set to a lower number.


### Multithreading

By default, PyKX doesn’t support calling q from multiple threads in a Python process due to the Global Interpreter Lock [GIL](https://wiki.python.org/moin/GlobalInterpreterLock). Enabling the `#!python PYKX_RELEASE_GIL` environment variable drops the GIL when calling q, making it unsafe to call q from multiple threads. To ensure thread safety, you can also enable the `#!python PYKX_Q_LOCK` environment variable, which adds a re-entrant lock around q. Learn [how to enable multithreaded execution](threading.md) and set up a Python process using PyKX to [call into EmbeddedQ from multiple threads](../../examples/threaded_execution/threading.md)

### Peach

Using the [`#!python peach`](../../api/pykx-execution/q.md#peach) function in q to call Python is not supported unless you enable the `#!python PYKX_RELEASE_GIL` setting. Without enabling this setting, the process will hang indefinitely.

For example, calling from Python into q into Python works normally:

```python
>>> kx.q('{x each 1 2 3}', lambda x: range(x))
pykx.List(pykx.q('
,0
0 1
0 1 2
'))
```
But, by default, using `#!python peach` to call from Python into q and back into Python hangs:

```python
>>> kx.q('{x peach 1 2 3}', lambda x: range(x)) # Warning: will hang indefinitely
```

However, if you enable `#!python PYKX_RELEASE_GIL`, it works:

```python
>>> import os
>>> os.environ['PYKX_RELEASE_GIL'] = '1'
>>> import pykx as kx
>>> kx.q('{x peach 1 2 3}', lambda x: range(x))
pykx.List(pykx.q('
,0
0 1
0 1 2
'))
```

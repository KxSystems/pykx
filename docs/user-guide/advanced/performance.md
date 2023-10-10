# Performance considerations

To get the best performance out of PyKX, follow the guidelines explained on this page. Note that this page doesn't concern itself with getting the best performance out of Python itself, or out of q itself. Rather this page is focused on how to interface between the two most efficiently.

- Avoid converting K objects with their `.py`/`.np`/`.pd`/etc. methods. Oftentimes the K object itself is sufficient for the task at hand.
- **Do as little work as necessary:**
    - When conversion is necessary, only convert what is really needed. For instance, instead of converting an entire q table to a dataframe, perhaps only a subset of the columns need to be converted into Numpy arrays. You could get these columns by indexing into the [`pykx.Table`][pykx.Table], then calling `.np` on the columns returned.
    - When using an IPC connection, make use of select statements and indexing to only send the subset of the data you want to process in Python over the IPC connection.
- Prefer using `.np` and `.pd` over `.py`. If a conversion must happen, try to stick to the Numpy/Pandas conversions which avoid copying data where possible. Converting objects with `.py` will always incur a data copy (if the conversion is possible at all - n.b. some K objects return themselves when `.py` is called on them, such as [`pykx.Function`][pykx.Function]) instances.
- Convert with the keyword argument `raw=True` when performance is more important than the richness of the output. Using a raw conversion can be much more efficient in many cases by not doing some work, such as adjusting the temporal epoch from 2000-01-01 to 1970-01-01, turning q GUIDs into Python `UUID` objects (instead they will come through as complex numbers, as that is the only widely available 128 bit type), converting bytes into strings, and more.
- Avoid nested columns when converting q tables into Pandas dataframes, as this currently incurs a data copy.
- **Let q do the heavy lifting:**
    - When running in licensed mode, make use of q code and q functions (e.g. `q.avg`, `q.sdev`, etc.) instead of pure Python code. This is similar to how you should use Numpy functions to operate on Numpy arrays instead of pure Python code. Note that the performance of Numpy functions on K vectors that have been converted to Numpy arrays is often comparable, even when including the conversion overhead.
    - When using an IPC connection to a remote q process, consider using q code to offload some of the work to the q process by pre-processing the data in q.
- Avoid converting large amounts of data from Python to q. Conversions from q to Python (via Numpy) can often avoid copying data, but conversions from Python to q always incur a copy of the data.

## Parallelization

### Secondary q threads

PyKX starts embedded q with as many secondary q threads enabled as are available. These threads are automatically used by q to parallelize some computations as it deems appropriate. The `QARGS` environment variable can be used to provide command-line arguments and other startup flags to q/PyKX, including the number of secondary threads:

```sh
QARGS='-s 0' python # disable secondary threads
```

```sh
QARGS='-s 12' python # use 12 secondary threads by default
```

The value set using `-s` provides both the default, and the maximum available to the process - it cannot be changed after PyKX has been imported.

PyKX exposes this maximum value as `pykx.q.system.max_num_threads`, which cannot be assigned to. The current number of secondary threads being used by q is exposed as `pykx.q.system.num_threads`. It is initially equal to `pykx.q.system.max_num_threads`, but can be assigned to a lower value.

### Multi-threading

By default PyKX does not currently support calling into q from multiple threads within a Python process simultaneously.
The [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) generally prevents this from occurring.

However enabling the `PYKX_RELEASE_GIL` environment variable will cause the Python Global Interpreter Lock to be dropped when calling into `q`.
Caution must be used when calling into q from multiple threads if this environment variable is set as it will no longer be thread safe, you can optionally also
enable the `PYKX_Q_LOCK` environment variable as well which will add an extra re-entrant lock around embedded q to ensure two threads cannot access `q`'s memory in an unsafe manner.

## Peach

Having q use [`peach`](../../api/pykx-execution/q.md#peach) to call into Python is not supported unless `PYKX_RELEASE_GIL` is enabled, and will hang indefinitely.

For example, calling from Python into q into Python works normally:

```python
>>> kx.q('{x each 1 2 3}', lambda x: range(x))
pykx.List(pykx.q('
,0
0 1
0 1 2
'))
```

But by default calling from Python into q into Python using `peach` hangs:

```python
>>> kx.q('{x peach 1 2 3}', lambda x: range(x)) # Warning: will hang indefinitely
```

However if `PYKX_RELEASE_GIL` is enabled this will work:

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

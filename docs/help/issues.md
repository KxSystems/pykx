---
title: Issues and Limitations
description: Known issues that occur when using python and limitations when using q embedded in python
maintained by: KX Systems, Inc.
date: Aug 2024
tags: PyKX, issues, embedded
---

# Issues and Limitations

_This page details known issues and functional limitations when using PyKX either as q embedded in a python process via the `#!python import pykx` command, or as a python processes embedded in q via `#!q \l pykx.q`._

## PyKX
### Known issues
* Enabling the NEP-49 NumPy allocators will often segfault when running in a multiprocess setting.
* The timeout value is always set to 0 when using PYKX_Q_LOCK.
* Enabling PYKX_ALLOCATOR and using PyArrow tables can cause segfaults.
* `#!python kurl` functions require their `#!python options` dictionary to have mixed type values. Add a `#!python None` value to bypass: `#!python {'': None, ...}`
* `#!python None` and `#!python pykx.Identity(pykx.q('::'))` do not pass through to single argument Python functions set under q as outlined in this example:
```python
>>> def func(n=2):
...    return n
...
>>> kx.q('func', None)
pykx.LongAtom(pykx.q('2'))
>>> kx.q('func', kx.q('::'))
pykx.LongAtom(pykx.q('2'))
```

### Limitations
Embedding q in a Python process imposes some restrictions on functionality. The embedded q process does not run the main loop that it would when running natively, hence it is limited in usage of q IPC and q timers.

#### IPC
The embedded q process cannot be used to respond to q IPC requests as a server. Callback functions such as .z.pg defined within a Python process will not operate as expected. Here is an example demonstrating this:

In a Python process, start a q IPC server:
```python
>>> import pykx as kx
>>> kx.q('\\p 5001')
pykx.Identity(pykx.q('::'))
```

Now, in a Python or q process attempt to connect to the above embedded q server.
```python
>>> import pykx as kx
>>> q = kx.QConnection(port=5001) # Attempt to create a q connection to a PyKX embedded q instance
# This process is now hung indefinitely as the embedded q server cannot respond
```

```q
q)h:hopen`::5001 /Attempting to create an IPC connection to a PyKX embedded q instance
/This process is now hung indefinitely as the embedded q server cannot respond
```

#### Timers
Timers in q rely on the main loop of the standalone executable so they will not work on the q process embedded in python.
```python
>>> import pykx as kx
>>> kx.q('.z.ts:{0N!x}') # Set callback function which should be run on a timer
>>> kx.q('\t 1000') # Set timer to tick every 1000ms
pykx.Identity(pykx.q('::')) # No output follows because the timer never ticks
```
Attempting to use the timer callback function directly using PyKX will raise an AttributeError:
```python
>>> kx.q.z.ts
AttributeError: ts: .z.ts is not exposed through the context interface because there is no main loop in the embedded q process
```

## Python embedded in a q process

### Limitations
Controlling object return and conversion between a q process and its embedded python instance requires the use of several special characters. In order to use these characters as parameters of functions, as opposed to operations on objects, there are specific steps to be followed.

#### Return characters: `#!q <`, `#!q >`, and `#!q *`
During function definition you must specify a return type in order to use the return characters as parameters for the function.
```q
q)f:.pykx.eval["lambda x: x";<] /Specify < to return output as a q object. *, < and > can now be used as parameters to function f
q)f[*]
*
```

#### Conversion characters: ``#!q ` `` and ``#!q `. ``
During function definition either define the return type (as above) or use the `#!q .pykx.tok` function:
```q
q).pykx.eval["lambda x: x"][`]` /throws error
'Provided foreign object is not a Python object
q).pykx.eval["lambda x: x";<][`] /defining the return type using < allows use of ` as a parameter
`
q).pykx.eval["lambda x: x"][.pykx.tok[`]]` /wrapping input in function tok allows use of ` as a parameter
`
```

#### q default parameter `#!q ::`
When you execute a q function that has no user defined parameters the accepted q style is to use `#!q []` (e.g. `#!q f:{1+1};f[] /outputs 2`). During execution q will use the generic null as the value passed:
```q
q)(::)~{x}[] /x parameter received by lambda is the generic null ::
1b
```

Using `#!q ::` as an argument to PyKX functions presents some difficulties:
```q
q)f:.pykx.eval["lambda x: x";<]
q)f[::] /the Python process cannot tell the difference between f[] and f[::] so throws an error
'TypeError("<lambda>() missing 1 required positional argument: 'x'")
  [0]  f[::]
```

You can avoid this by wrapping the input in `#!q .pykx.tok`:
```q
q)(::)~f[.pykx.tok[::]]
1b
```

Python functions defined with 0 parameters will run without issues as they will ignore the automatically added `#!q ::`:
```q
p)def noparam():return 7
q)f:.pykx.get[`noparam;<]
q)f[]
7
q)f[::] /equivalent
7
```

## Pandas

### Known issues
#### Changes in `DataFrame.equals` behavior from Pandas 2.2.0

In Pandas 2.2.0, a difference was introduced in how `DataFrame.equals` handles DataFrames with different `_mgr` types.

**Example:**
```python
>>> import pandas as pd
>>> import pykx as kx

>>> df1 = pd.DataFrame({'cl': ['foo']})
>>> df2 = kx.q('([] cl:enlist `foo)').pd()

>>> df2.equals(df1)
True

>>> df1.equals(df2) # Prior to Pandas 2.2.0, this would also evaluate to True
False
```

**Cause:**  
Pandas now checks the type of the `_mgr` (dataframes manager) property. PyKX uses a custom `_mgr` implementation for performance optimization.

```python
>>> type(df1._mgr)
<class 'pandas.core.internals.managers.BlockManager'>

>>> type(df2._mgr)
<class 'pykx.util.BlockManagerUnconsolidated'>
```

**Workaround:**
Comparing the full contents of DataFrames irrespective of `_mgr` types works regardless of order of df1 and df2 in the comparison. To do so, use one of the following approaches:
```python
>>> assert (df2 == df1).all().all()  # Element-wise comparison
>>> pd.testing.assert_frame_equal(df2, df1)  # Pandas' built-in test
>>> assert df1.compare(df2).empty  # Check if there are no differences
```

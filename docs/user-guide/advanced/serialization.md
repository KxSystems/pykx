---
title: Serialization and De-serialization
description: Learn how to serialize and de-serialize in PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, python, serialize, de-serialize
---

# Serialize and de-serialize data

_This page explains how to use PyKX to serialize and de-serialize kdb+/q data structures directly to and from Python byte objects._

There are two main ways to serialize/de-serialize data with PyKX:

- By interfacing with Python's [`pickle`](https://docs.python.org/3/library/pickle.html) library to persist data to disk in a Python friendly format.
- By using the [`kx.serialize`](../../api/serialize.md) module to prepare data in q IPC data format.

!!! Warning

	In all cases de-serializing data can be risky. Proceed only if you’re sure the data comes from a trusted source.


## Serialization using pickle

Serializing data is extremely useful in cases where you need to convert a data object into a format that is easily transmittable, such as storing data or transferring it to a remote process. When serializing your PyKX data in most cases it is suggested that you make use of the integration between PyKX and Pickle.

In the following three examples you can see the serialization and de-serialization of various PyKX objects:

1. PyKX Table

	```python
	>>> import pykx as kx
	>>> import pickle
	>>> table = kx.Table([[1, 2, 3]])
	>>> print(table)
	x x1 x2
	-------
	1 2  3 
	>>> pdump = pickle.dumps(table)
	>>> print(pdump)
	b'\x80\x04\x95\xf5\x00..'
	>>> print(pickle.loads(pdump))
	x x1 x2
    -------
    1 2  3
	```

2. PyKX Float Vector

	```python
	>>> import pykx as kx
	>>> import pickle
	>>> qvec = kx.random.random(10, 2.0)
	>>> print(qvec)
	0.7855048 1.034182 1.031959 0.8133284 0.3561677 0.6035445 1.570066 1.069419 1..
	>>> pdump = pickle.dumps(qvec)
	>>> print(pdump)
	b'\x80\x04\x95\n\x..'
	>>> print(pickle.loads(pdump))
	0.7855048 1.034182 1.031959 0.8133284 0.3561677 0.6035445 1.570066 1.069419 1..
	```

3. PyKX List

	```python
	>>> import pykx as kx
	>>> import pickle
	>>> import uuid
	>>> qlist = kx.toq([1, 'b', uuid.uuid4()])
	>>> print(qlist)
	1
	`b
	7c667128-4ebd-45da-971c-38d5c54e36e1
	>>> pdump = pickle.dumps(qlist)
	>>> print(pdump)
	b'\x80\x04\x95\xd7..'
	>>> print(pickle.loads(pickle.dumps(qlist)))
	1
	`b
	7c667128-4ebd-45da-971c-38d5c54e36e1
	```

## Serialization using `kx.serialize`

While using `#!python pickle` will be sufficient in most cases, there will be times where you are required to convert data to or from the q IPC format byte representation. Using the `#!python kx.serialize` and `#!python kx.deserialize` functions will provide better performance in these situations.

Unlike with `#!python pickle`, which returns the byte representation immediately on serialization, PyKX allows the generation of this byte object to be deferred by creating a [`memoryview`](https://docs.python.org/3/library/stdtypes.html#memoryview). Deserialization can be completed directly from this `#!python memoryview` or from the raw byte objects

Similar to the examples in the previous section in the below we will serialize and deserialize various PyKX objects:

1. PyKX Table

	```python
	>>> import pykx as kx
	>>> table = kx.Table([[1, 2, 3]])
	>>> print(table)
	x x1 x2
	-------
	1 2  3
	>>> sertab = kx.serialize(table)
	>>> sertab
	<pykx.serialize.serialize at 0x147d744a0>
	>>> sertab.copy()
	b'\x01\x00\x00\x00I\..'
	>>> print(kx.deserialize(sertab))
	x x1 x2
	-------
	1 2  3
	>>> print(kx.deserialize(sertab.copy())
	x x1 x2
	-------
	1 2  3 
    ```

2. PyKX Float Vector

	```python
	>>> import pykx as kx
	>>> import pickle
	>>> qvec = kx.random.random(10, 2.0)
	>>> print(qvec)
	0.7855048 1.034182 1.031959 0.8133284 0.3561677 0.6035445 1.570066 1.069419 1..
	>>> servec = kx.serialize(qvec)
	>>> print(servec)
	<pykx.serialize.serialize object at 0x11dff7b90>
	>>> print(servec.copy())
	b'\x01\x00\x00\x00^..'
	>>> print(kx.deserialize(servec))
	0.7855048 1.034182 1.031959 0.8133284 0.3561677 0.6035445 1.570066 1.069419 1..
	>>> print(kx.deserialize(servec.copy()))
	0.7855048 1.034182 1.031959 0.8133284 0.3561677 0.6035445 1.570066 1.069419 1..
	```

3. PyKX List

	```python
	>>> import pykx as kx
	>>> import pickle
	>>> import uuid
	>>> qlist = kx.toq([1, 'b', uuid.uuid4()])
	>>> print(qlist)
	1
	`b
	7c667128-4ebd-45da-971c-38d5c54e36e1
	>>> serlist = kx.serialize(qlist)
	>>> print(serlist)
    <pykx.serialize.serialize object at 0x147d93590>
    >>> print(serlist.copy())
	b'\x01\x00\x00\x00..'
	>>> print(kx.deserialize(serlist))
	1
	`b
	7c667128-4ebd-45da-971c-38d5c54e36e1
	>>> print(kx.deserialize(serlist.copy()))
	1
	`b
	7c667128-4ebd-45da-971c-38d5c54e36e1
	```

## What are the limitations?

Serialization of PyKX objects is limited to objects which are purely generated from kdb+/q data. Serialization of `pykx.Foreign` objects, for example, is not supported as these represent underlying objects defined in C of arbitrary complexity.

```python
>>> import pykx as kx
>>> import pickle
>>> pickle.dumps(kx.Foreign(1))
TypeError: Unable to serialize pykx.Foreign objects
```

Similarly, you cannot serialize on-disk representations of tabular data such as `pykx.SplayedTable` and `pykx.PartitionedTable`.

## Next Steps

- [Learn how to interact via IPC](ipc.md)
- [Learn how to call q functions in a Python first way](context_interface.md)

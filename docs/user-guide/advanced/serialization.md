# Serialization and de-serialization

PyKX allows users to serialize and de-serialize kdb+/q data structures directly to and from Python byte objects. Interoperating with Pythons [`pickle`](https://docs.python.org/3/library/pickle.html) library this allows users to persist and retrieve objects generated or accessed via PyKX into entities which can be saved to disk or sent via IPC to another process.

While the application of serialization and de-serialization can be completed using q code directly within PyKX it is advised that users leverage [`pickle.dumps`](https://docs.python.org/3/library/pickle.html#pickle.dumps) and [`pickle.loads`](https://docs.python.org/3/library/pickle.html#pickle.loads) when attempting to interact with serialized representations of kdb+/q data for usage within a Python only environment. 

!!! Warning

	De-serialization of data is not inherently secure, if you are de-serializing data please only do so if retrieved from a trusted source.

## Limitations

Serialization of PyKX objects is limited to objects which are purely generated from kdb+/q data. Serialization of `pykx.Foreign` objects, for example, is not supported as these represent underlying objects defined in C of arbitrary complexity.

```python
>>> import pykx as kx
>>> import pickle
>>> pickle.dumps(kx.Foreign(1))
TypeError: Unable to serialize pykx.Foreign objects
```

Similarly on-disk representations of tabular data such as `pykx.SplayedTable` and `pykx.PartitionedTable` cannot be serialized.

## Examples

The following are examples showing the serialization and de-serialization of PyKX objects with

1. PyKX Table

	```python
	>>> import pykx as kx
	>>> import pickle
	>>> table = kx.Table([[1, 2, 3]])
	>>> print(table)
	x x1 x2
	-------
	1 2  3 
	>>> print(pickle.loads(pickle.dumps(table)))
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
	>>> print(pickle.loads(pickle.dumps(qvec)))
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
	540bad66-0838-46ca-b5eb-b4bab5e32228
	>>> print(pickle.loads(pickle.dumps(qlist)))
	1
	`b
	540bad66-0838-46ca-b5eb-b4bab5e32228
	```

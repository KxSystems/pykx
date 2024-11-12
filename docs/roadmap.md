# PyKX Roadmap

This page outlines areas of development focus for the PyKX team to provide you with an understanding of the development direction of the library. This is not an exhaustive list of all features/areas of focus but should give you a view on what to expect from the team over the coming months. Additionally this list is subject to change, particularly for any example code provided based on the complexity of the features and any customer feature requests raised following the publishing of this list.

If you need a feature that's not included in this list please let us know by raising a [Github issue](https://github.com/KxSystems/pykx/issues)!

## Upcoming Changes

- Addition of support for q primitives as methods off PyKX Vector and Table objects. Syntax for this will be similar to the following:

	```python
	>>> import pykx as kx
	>>> N = 1000
	>>> vec = kx.random.random(N, 100.0)
	>>> vec.mavg(3)
	>>> vec.abs()
	```

- Performance improvements for conversions from PyKX Vector objects to Numpy arrays through enhanced use of C++ over Cython.
- Configurable initialisation logic in the absence of a license. Thus allowing users who have their own workflows for license access to modify the instructions for their users.

## Future

- Tighter integration between PyKX/q objects and PyArrow arrays/Tables
- Expansion of supported datatypes for translation to/from PyKX
- Data pre-processing and statistics modules for operation on PyKX tables and vector objects

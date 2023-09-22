# Differences and upgrade considerations from embedPy

As outlined [here](intro.md) PyKX provides users with the ability to execute Python code within a q session similar to [embedPy](https://github.com/kxsystems/embedpy). This document outlines points of consideration when upgrading from embedPy to PyKX under q both with respect to the function mappings between the two interfaces and differences in their behaviour.

## Functional differences

### q symbol and string support 

EmbedPy does not allow users to discern between q string and symbol types when converting to Python. In both cases these are converted to `str` objects in Python. As a result round trip conversions are not supported in embedPy for symbols, PyKX does support such round trip operations:

=== "embedPy"

	```q
	q).p.set[`a;"test"]
	q)"test"~.p.get[`a]`
	1b
	q).p.set[`b;`test]
	q)`test~.p.get[`b]`
	0b
	```

=== "PyKX"

	```q
	q).pykx.set[`a;"test"]
	q)"test"~.pykx.get[`a]`
	1b
	q).pykx.set[`b;`test]
	q)`test~.pykx.get[`b]`
	1b
	```


### Python object type support

EmbedPy contains a fundamental limitation with respect to the data formats that are supported when converting betwen q and Python. Namely that all q objects when passed to Python functions use the analagous Python/Numpy representation. This limitation means that a user of embedPy must handle their own data conversions when handling Pandas or PyArrow objects.

PyKX natively supports data conversions from q to Python, Numpy, Pandas and PyArrow and as such can support workflows which previously required users to manually control these conversions, for example:

```q
q).pykx.print .pykx.eval["lambda x:type(x)"] .pykx.topd ([]10?1f)

```

## Functionality mapping

The following table describes the function mapping from PyKX to embedPy for various elements of the supported functionality within embedPy, where a mapping supported this will be explicitly noted. Where workarounds exist these are additionally noted.

| Description                                                           | PyKX                            | embedPy         |
|-----------------------------------------------------------------------|---------------------------------|-----------------|
| Library loading                                                       | `\l pykx.q`                     | `\l p.q`        |
| Importing Python Libraries as wrapped Python objects                  | `.pykx.import`                  | `.p.import`     |
| Setting objects in Python Memory                                      | `.pykx.set`                     | `.p.set`        |
| Retrieving Python objects from Memory                                 | `.pykx.get`                     | `.p.get`        |
| Converting Python objects to q                                        | `.pykx.toq`                     | `.p.py2q`       |
| Execute Python code returning as intermediary q/Python object         | `.pykx.eval`                    | `.p.eval`       |
| Execute Python code returning a q object                              | `.pykx.qeval`                   | `.p.qeval`      |
| Execute Python code returning a Python foreign object                 | `.pykx.pyeval`                  | `.p.eval`       |
| Retrieve a printable representation of a supplied PyKX/q objext       | `.pykx.repr`                    | `.p.repr`       |
| Set an attribute on a supplied Python object                          | `.pykx.setattr`                 | `.p.setattr`    |
| Retrieve an attribute from a supplied Python object                   | `.pykx.getattr`                 | `.p.getattr`    |
| Convert a Python foreign object to a wrapped object for conversion    | `.pykx.wrap`                    | `.p.wrap`       |
| Convert a wrapped Python object to a Python foreign object            | `.pykx.unwrap`                  | `.p.unwrap`     |
| Print a Python object to standard out                                 | `.pykx.print`                   | `.p.print`      |
| Import a Python library as a Python foreign object                    | `.pykx.pyimport`                | `.p.pyimport`   |
| Generate a callable Python function returning a Python foreign object | `.pykx.pycallable`              | `.p.pycallable` |
| Generate a callable Python function returning a q result              | `.pykx.qcallable`               | `.p.qcallable`  |
| Interactive Python help string                                        |  Unsupported                    | `.p.help`       |
| Retrieval of Python help string as a q string                         |  Unsupported                    | `.p.helpstr`    |
| Convert a q object to a Python foreign object                         |  Unsupported                    | `.p.q2py`       |
| Create a Python closure using a q function                            |  Unsupported                    | `.p.closure`    |
| Create a Python generator using a q function                          |  Unsupported                    | `.p.generator`  |

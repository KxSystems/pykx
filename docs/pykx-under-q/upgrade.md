# Differences and upgrade considerations from embedPy

As outlined [here](intro.md) PyKX provides users with the ability to execute Python code within a q session similar to [embedPy](https://github.com/kxsystems/embedpy). This document outlines points of consideration when upgrading from embedPy to PyKX under q both with respect to the function mappings between the two interfaces and differences in their behavior.

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
| Retrieve a printable representation of a supplied PyKX/q object       | `.pykx.repr`                    | `.p.repr`       |
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

## PyKX under q benefits over embedPy

PyKX under q provides a number of key functional benefits over embedPy alone when considering the generation of workloads that integrate Python and q code. The following are the key functional/feature updates which provide differentiation between the two libraries

1. Flexibility in supported data formats and conversions
2. Python code interoperability
3. Access to PyKX in it's Python first modality

### Flexibility in supported data formats and conversions

EmbedPy contains a fundamental limitation with respect to the data formats that are supported when converting between q and Python. Namely that all q objects when passed to Python functions use the analogous Python/NumPy representation. This limitation means that a user of embedPy who require data to be in a Pandas/PyArrow format need to handle these conversions manually.

As PyKX supports Python, NumPy, Pandas and PyArrow data formats this improves the flexibility of workflows that can be supported, for example PyKX will by default convert q tables to Pandas DataFrames when passed to a Python function as follows

```q
q).pykx.eval["lambda x:type(x)"] ([]10?1f;10?1f)
<class 'pandas.core.frame.DataFrame'>
```

Additional to this a number of helper functions are provided to allow users to selectively choose the target data formats which are used when passing to multivariable functions, for example

```q
q).pykx.eval["lambda x, y:print(type(x), type(y))"][.pykx.tonp ([]10?1f);.pykx.topd til 10];
<class 'numpy.recarray'> <class 'pandas.core.series.Series'>
```

This flexibility makes integration with custom libraries easier to manage.

### Python interoperability

For users that are working to integrate tightly their Python code and q code prototyping Python functions for use within embedPy could be difficult. Users are required when defining their functions either to provide them as a string with appropriate tab/indent usage to a `.p.e` as follows

```q
q).p.e"def func(x):\n\treturn x+1"
q)pyfunc:.pykx.get[`func;<]
q)pyfunc[2]
3
```

Alternatively users could create a `.py`/`.p` file and access their functions using ```.pykx.import[`file_name]``` or `\l file_name.p` respectively.

While these solutions provide provide a method of integrating your Python code they are not intuitive to a user versed both in Python and q.

PyKX provides a function `.pykx.console` which allows users within a q session to run a Python "console" to generate their functions/variables for use within their q code. The following example uses PyKX 2.3.0.

```q
q).pykx.console[]
>>> def func(x):
...     return x+1
...
>>> quit()
q)pyfunc:.pykx.get[`func;<]
q)pyfunc[2]
3
```

This change allows users to iterate development of their analytics faster than when operating with embedPy.

### Access to PyKX in it's Python first modality

Following on from the Python interoperability section above access to PyKX itself as a Python module provides significant flexibility to users when developing analytics for use within a q session.

With embedPy when q/kdb+ data is passed to Python for the purposes of completing "Python first" analysis there is a requirement that that analysis fully uses Python libraries that are available to a user and can not get performance benefits from having access to q/kdb+.

Take for example a case where a user wishes to run a Python function which queries a table available in their q process using SQL and calculates the mean value for all numeric columns.

```q
q)tab:([]100?`a`b`c;100?1f;100?1f;100?0Ng)
q).pykx.console[]
>>> import pykx as kx
>>> def pyfunction(x):
...     qtab = kx.q.sql('SELECT * from tab where x=$1', x)
...     return qtab.mean(numeric_only=True)
>>> quit()
q)pyfunc:.pykx.get[`pyfunction;<]
q)pyfunc `a
x1| 0.5592623
x2| 0.486176
```

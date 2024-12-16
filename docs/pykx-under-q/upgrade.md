---
title:  Upgrade from embedPy
description: How to upgrade from embedPy to PyKX within q
date: June 2024
author: KX Systems, Inc.,
tags: embedPy, PyKX, q,
---


# Upgrade from embedPy

_This page outlines differences and function mappings when upgrading from embedPy to PyKX in a q session._

Just like [PyKX](../getting-started/what_is_pykx.md), [embedPy](https://github.com/kxsystems/embedpy) is a tool that allows to execute Python code and call Python functions. 

## Functional differences

### q symbol and string support 

EmbedPy doesn't allow users to discern between q `#!python string` and `#!python symbol` types when converting to Python. In both cases, these are converted to `#!python str` objects in Python. As a result, embedPy doesn't support round-trip conversions for symbols, but PyKX does:

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

### Functionality mapping

The following table describes function mapping from PyKX to embedPy:

| Description                                                           | PyKX                            | embedPy         |
|-----------------------------------------------------------------------|---------------------------------|-----------------|
| Load library                                                          | `\l pykx.q`                     | `\l p.q`        |
| Import Python Libraries as wrapped Python objects                     | `.pykx.import`                  | `.p.import`     |
| Set objects in Python Memory                                          | `.pykx.set`                     | `.p.set`        |
| Retrieve Python objects from Memory                                   | `.pykx.get`                     | `.p.get`        |
| Convert Python objects to q                                           | `.pykx.toq`                     | `.p.py2q`       |
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
| Retrieve Python help string as a q string                             |  Unsupported                    | `.p.helpstr`    |
| Convert a q object to a Python foreign object                         |  Unsupported                    | `.p.q2py`       |
| Create a Python closure using a q function                            |  Unsupported                    | `.p.closure`    |
| Create a Python generator using a q function                          |  Unsupported                    | `.p.generator`  |

## PyKX under q benefits over embedPy

When generating workloads that integrate Python and q code, PyKX under q provides a few key functional benefits over embedPy alone:

1. [Flexibility in supported data formats and conversions](#1-flexibility-in-supported-data-formats-and-conversions)
2. [Python code interoperability](#2-python-interoperability)
3. [Access to PyKX as a Python module](#3-access-to-pykx-as-a-python-module)

### 1. Flexibility in supported data formats and conversions

When using EmbedPy to convert data between q and Python, there’s a fundamental limitation related to supported data formats. Specifically, when passed to Python functions, q objects use the analogous Python/NumPy representation. This means that if an embedPy user requires data in a Pandas/PyArrow format, they need to convert it manually.

As PyKX supports Python, NumPy, Pandas, and PyArrow data formats, it improves the workflow coverage and flexibility. For instance, PyKX by default converts q tables to Pandas DataFrames when passed to a Python function as follows:

```q
q).pykx.eval["lambda x:type(x)"] ([]10?1f;10?1f)
<class 'pandas.core.frame.DataFrame'>
```

Additionally, PyKX provides helper functions, allowing you to choose the target data formats used when passing to multivariable functions. For example:

```q
q).pykx.eval["lambda x, y:print(type(x), type(y))"][.pykx.tonp ([]10?1f);.pykx.topd til 10];
<class 'numpy.recarray'> <class 'pandas.core.series.Series'>
```

This flexibility makes integration with custom libraries significantly easier to manage.

### 2. Python interoperability

If you wish to integrate Python and q code, prototyping Python functions for use within embedPy could be difficult. When defining your functions, you need to either provide them as a string with appropriate tab/indent usage to a `#!python .p.e` as follows:

```q
q).p.e"def func(x):\n\treturn x+1"
q)pyfunc:.pykx.get[`func;<]
q)pyfunc[2]
3
```

Alternatively, you could create a `#!python .py`/`#!python .p` file and access your functions using ```#!python .pykx.import[`file_name]``` or `#!python \l file_name.p` respectively.

Both solutions are not intuitive to users versed both in Python and q.

That's why PyKX provides a Python `#!python .pykx.console` function that you can run within a q session to generate your functions/variables. The following example uses PyKX 2.3.0:

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

This function allows you to iterate your analytics development faster than when operating with embedPy.

### 3. Access to PyKX as a Python module

Access to PyKX in its Python-first mode adds more flexibility to users who develop analytics to use within q.

With embedPy, when you pass q/kdb+ data to Python to complete a "Python-first" analysis, you're restricted to your Python libraries and can't get performance benefits from having access to q/kdb+.

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

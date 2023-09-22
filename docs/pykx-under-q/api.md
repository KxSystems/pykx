# pykx.q Library Reference Card

This page documents the functions found in the `pykx.q` q library that are available.

This library can be installed by calling a helper function within `PyKX`, this function will move
all the required files and libraries into your `QHOME` directory.

```python
import pykx as kx
kx.install_into_QHOME()
```

or equivalently using only command line

```python
python -c "import pykx;pykx.install_into_QHOME()"
```

If you previously had `embedPy` installed pass:

```python
python -c "import pykx;pykx.install_into_QHOME(overwrite_embedpy=True)"
```

If you cannot edit files in `QHOME` you can copy the files to your local folder and load `pykx.q` from there:

```bash
python -c "import pykx;pykx.install_into_QHOME(to_local_folder=True)"
```

Gain access to the `.pykx` namespace within the `q` session

```q
q)\l pykx.q
```

**PyKX q API functionality:**

<div markdown="1" class="typewriter">
**.pykx.**

**General:**
[console                           open an interactive Python REPL](#pykxconsole)
[version                           retrieve PyKX version](#pykxversion)
[print                             print a Python object directly to stdout](#pykxprint)
[repr                              evaluate the Python function repr() on supplied Python object](#pykxrepr)
[debugInfo                         print useful process debug information to q session](#pykxdebuginfo)

**Data Conversions:**
[setdefault                        define the default conversion for KX objects to Python](#pykxsetdefault)
[toq                               convert an (un)wrapped `PyKX` foreign object into a q type](#pykxtoq)
[tok                               tag a q object to be indicate conversion to a Pythonic PyKX object when called in Python](#pykxtok)
[topy                              tag a q object to be indicate conversion to a Python object when called in Python](#pykxtopy)
[tonp                              tag a q object to be indicate conversion to a Numpy object when called in Python](#pykxtonp)
[topd                              tag a q object to be indicate conversion to a Pandas object when called in Python](#pykxtopd)
[topa                              tag a q object to be indicate conversion to a PyArrow object when called in Python](#pykxtopa)
[toraw                             tag a q object to be indicate conversion to a raw representation object when called in Python](#pykxtoraw)

**Evaluation and Execution:**
[eval                              evaluate a string as Python code returning a wrapped foreign object](#pykxeval)
[pyeval                            evaluate a string as Python code returning a foreign object](#pykxpyeval)
[qeval                             evaluate a string as Python code returning a q object](#pykxqeval)
[pyexec                            execute a string as Python code in Python memory](#pykxpyexec)

**Python Library Integration:**
[import                            import a Python library and store as a wrapped foreign object](#pykximport)
[pyimport                          import a Python library and store as a foreign object](#pykxpyimport)

**Callable Object Generation:**
[qcallable                         convert a Python foreign object to a callable function which returns a q result](#pykxqcallable)
[pycallable                        convert a Python foreign object to a callable function which returns a Python result](#pykxpycallable)

**Object Setting and Retrieval:**
[set                               set a q object as a named object in Python memory](#pykxset)
[setattr                           set an attribute of a Python object](#pykxsetattr)
[get                               retrieve a named item from the Python memory](#pykxget)
[getattr                           retrieve an attribute of a Python object](#pykxgetattr)

**Foreign and PyKX object Handling:**
[wrap                              convert a foreign object generated from Python execution to a callable q object](#pykxwrap)
[unwrap                            convert a wrapped foreign object generated from this interface into a python foreign](#pykxunwrap)

**.q.**

**Python Function Argument Utilities:**
[pykw                              allow users to apply individual keywords to a Python function](#pykw)
[pyarglist                         allow users to apply a list of arguments to a Python function, equivalent to `*args`](#pyarglist)
[pykwargs                          allow users to apply a dictionary of named arguments to a Python function, equivalent to `*kwargs`](#pykwargs)

</div>


<!-- Do not edit anything in this file below this line. Edits must be made to source files or they will be overwritten! -->
<!-- AUTO GENERATED DOCUMENTATION -->


## `.pykx.console`


_Open an interactive python REPL from within a q session similar to launching python from the command line._

```q
.pykx.console[]
```

**Returns:**

type | description
-----|------------
`::` | This function has no explicit return but execution of the function will initialise a Python REPL.

**Example:**

```q
Enter PyKX console and evaluate Python code
q).pykx.console[]
>>> 1+1
2
>>> list(range(10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> quit()
q)

// Enter PyKX console setting q objects using PyKX
q).pykx.console[]
>>> import pykx as kx
>>> kx.q['table'] = kx.q('([]2?1f;2?0Ng;2?`3)'
>>> quit()
q)table
x         x1                                   x2 
--------------------------------------------------
0.439081  49f2404d-5aec-f7c8-abba-e2885a580fb6 mil
0.5759051 656b5e69-d445-417e-bfe7-1994ddb87915 igf
```

## `.pykx.debugInfo`


_Library and environment information which can be used for environment debugging_

```q
.pykx.debugInfo[]
```

**Returns:**

type   | description
-------|------------
`list` | A list of strings containing information useful for debugging

**Example:**

```q
q).pykx.debugInfo[]
"**** PyKX information ****"
"pykx.args: ()"
"pykx.qhome: /usr/local/anaconda3/envs/qenv/q"
"pykx.qlic: /usr/local/anaconda3/envs/qenv/q"
"pykc.licensed: True"
..
```

## `.pykx.eval`


_[Evaluates](https://docs.python.org/3/library/functions.html#eval) a `string` as python code and return the result as a wrapped `foreign` type._

```q
.pykx.eval[pythonCode]
```

**Parameters:**

name         | type      | description |
-------------|-----------|-------------|
`pythonCode` | string    | A string of Python code to be executed returning the result as a wrapped foreign object. |

**Return:**

type | description
-----|------------
`composition` | A wrapped foreign object which can be converted to q or Python objects

```q
// Evaluate the code and return as a wrapped foreign object
q).pykx.eval"1+1"
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist

// Evaluate the code and convert to Python foreign
q).pykx.eval["1+1"]`.
foreign

// Evaluate the code and convert to a q object
q).pykx.eval["lambda x: x + 1"][5]`
6
```

## `.pykx.get`


_Retrieve a named item from the Python memory_

```q
.pykx.get[objectName]
```

**Parameters:**

name          | type      | description |
--------------|-----------|-------------|
`objectName` | symbol    | A named entity to retrieve from Python memory as a wrapped q foreign object. |

**Return:**

type          | description
--------------|------------
`composition` | A wrapped foreign object which can be converted to q or Python objects

```q
// Set an item in Python memory and retrieve using .pykx.get
q).pykx.set[`test;til 10]
q).pykx.get[`test]
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist

// Convert to q and Python objects
q).pykx.get[`test]`
0 1 2 3 4 5 6 7 8 9

// Retrieve an item defined entirely using Python
q).pykx.pyexec"import numpy as np"
q).pykx.pyexec"a = np.array([1, 2, 3])"
q).pykx.get[`a]`
1 2 3
```

## `.pykx.import`


_Import a Python library and store as a wrapped foreign object to allow use in q projections/evaluation._

```q
.pykx.import[libName]
```

**Parameters:**

name      | type   | description |
----------|--------|-------------|
`libName` | symbol | The name of the Python library/module to imported for use |

**Return:**

type          | description
--------------|------------
`composition` | Returns a wrapped foreign object associated with an imported library on success, otherwise will error if library/module cannot be imported.

```q
// Import numpy for use as a q object named numpy
q)np:.pykx.import`numpy
q).pykx.print np
<module 'numpy' from '/usr/local/lib64/python3.9/site-packages/numpy/__init__.py'>

// Use a function from within the numpy library using attribute retrieval
q).pykx.print np[`:arange]
<built-in function arange>
q)np[`:arange][10]`
0 1 2 3 4 5 6 7 8 9
```

## `.pykx.print`


_Print a python object directly to stdout. This is equivalent to calling `print()` on the object in Python._

```q
.pykx.print[pythonObject]
print[pythonObject]
```

**Parameters:**

name           | type              | description |
---------------|-------------------|-------------|
`pythonObject` | (wrapped) foreign | A Python object retrieved from the Python memory space, if passed a q object this will be 'shown' |

**Return:**

type | description
-----|------------
`::` | Will print the output to stdout but return null

!!! Note

        For back compatibility with embedPy this function is also supported in the shorthand form `print` which uses the `.q` namespace. To not overwrite `print` in your q session and allow use only of the longhand form `.pykx.print` set the environment variable `UNSET_PYKX_GLOBALS` to any value.

```q
// Use a wrapped foreign object
q)a: .pykx.eval"1+1"
q).pykx.print a
2

// Use a foreign object
q)a: .pykx.eval"'hello world'"
q).pykx.print a`.
hello world

// Use a q object
q).pykx.print til 5
0 1 2 3 4

// Print the return of a conversion object
q).pykx.print .pykx.topd ([]5?1f;5?0b)
          x     x1
0  0.178084  False
1  0.301772   True
2  0.785033   True
3  0.534710  False
4  0.711172  False
```

## `.pykx.toq`


_Convert an (un)wrapped `PyKX` foreign object into an analogous q type._

```q
.pykx.toq[pythonObject]
```

**Parameters:**

name           | type                   | description |
---------------|------------------------|-------------|
`pythonObject` | foreign/composition    | A foreign Python object or composition containing a Python foreign to be converted to q

**Return:**

type  | description
------|------------
`any` | A q object converted from Python

```q
// Convert a wrapped PyKX foreign object to q
q)show a:.pykx.eval["1+1"]
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist
q).pykx.toq a
2

// Convert an unwrapped PyKX foreign object to q
q)show b:a`.
foreign
q).pykx.toq b
2
```

## `.pykx.pycallable`


_Convert a Python foreign object to a callable function which returns a Python foreign result_

```q
.pykx.pycallable[pyObject]
```

**Parameters:**

name         | type      | description                                  
-------------|-----------|-------------
`pyObject`   | `foreign` | A Python object representing an underlying callable function

**Returns:**

type      | description
----------|------------
`foreign` | The return of the Python callable function as a foreign object

**Example:**

```q
q)wrappedPy:.pykx.import[`numpy;`:arange]
q)show setCallable:.pykx.pycallable[wrappedPy][1;3]
foreign
q).pykx.print setCallable
[1 2]
```

## `.pykx.pyeval`


_[Evaluates](https://docs.python.org/3/library/functions.html#eval) a `CharVector` as python code and return the result as a `q` foreign._

```q
.pykx.pyeval[pythonCode]
```

**Parameters:**

name         | type     | description |
-------------|----------|-------------|
`pythonCode` | `string` | A string of Python code to be evaluated returning the result as a q foreign object. |

**Return:**

 type      | description |
-----------|-------------|
 `foreign` | The return of the Python string evaluation returned as a q foreign. |

```q
// evaluate a Python string
q).pykx.pyeval"1+1"
foreign

// Use a function defined in Python taking a single argument
q).pykx.pyeval["lambda x: x + 1"][5]
foreign

// Use a function defined in Python taking multiple arguments
q).pykx.pyeval["lambda x, y: x + y"][4;5]
foreign
```

## `.pykx.pyexec`


_[Executes](https://docs.python.org/3/library/functions.html#exec) a `string` as python code in Python memory._

```q
.pykx.pyexec[pythonCode]
```

**Parameters:**

name         | type      | description |
-------------|-----------|-------------|
`pythonCode` | string    | A string of Python code to be executed. |

**Return:**

 type | description |
------|-------------|
 `::` | Returns generic null on successful execution, will return an error if execution of Python code is unsuccessful. |

```q
// Execute valid Python code
q).pykx.pyexec"1+1"
q).pykx.pyexec"a = 1+1"

// Evaluate the Python code returning the result to q
q).pykx.qeval"a"
2

// Attempt to execute invalid Python code
q).pykx.pyexec"1+'test'"
'TypeError("unsupported operand type(s) for +: 'int' and 'str'")
  [0]  .pykx.pyexec["1+'test'"]
       ^
```

## `.pykx.qcallable`


_Convert a Python foreign object to a callable function which returns a q result_

```q
.pykx.qcallable[pyObject]
```

**Parameters:**

name         | type      | description
-------------|-----------|-------------
`pyObject`   | `foreign` | A Python object representing an underlying callable function

**Returns:**

type  | description
------|------------
`any` | The return of the Python callable function as an appropriate q object

**Example:**

```q
q)wrappedPy:.pykx.import[`numpy;`:arange]
q)show setCallable:.pykx.pycallable[wrappedPy][1;3]
foreign
q).pykx.print setCallable
[1 2]
```

## `.pykx.qeval`


_[Evaluates](https://docs.python.org/3/library/functions.html#eval) a `CharVector` in Python returning the result as a q object._

```q
.pykx.qeval[pythonCode]
```

**Parameters:**

name         | type      | description |
-------------|-----------|-------------|
`pythonCode` | string    | A string of Python code to be evaluated returning the result as a q object. |

**Return:**

type  | description |
------|-------------|
`any` | The return of the Python string evaluation returned as a q object. |

```q
// evaluate a Python string
q).pykx.qeval"1+1"
2

// Use a function defined in Python taking a single argument
q).pykx.qeval["lambda x: x + 1"][5]
6

// Use a function defined in Python taking multiple arguments
q).pykx.qeval["lambda x, y: x + y"][4;5]
9
```

## `.pykx.repr`


_Evaluate the python function `repr()` on an object retrieved from Python memory_

```q
.pykx.repr[pythonObject]
```

**Parameters:**

name           | type  | description |
---------------|-------|-------------|
`pythonObject` | `any` | A Python object retrieved from the Python memory space, if passed a q object this will retrieved using [`.Q.s1`](https://code.kx.com/q/ref/dotq/#qs1-string-representation). |

**Return:**

type     | description
---------|------------
`string` | The string representation of the Python/q object

```q
// Use a wrapped foreign object
q)a: .pykx.eval"1+1"
q).pykx.repr a
,"2"

// Use a foreign object
q)a: .pykx.eval"'hello world'"
q).pykx.repr a`.
"hello world"

// Use a q object
q).pykx.repr til 5
"0 1 2 3 4"
```

## `.pykx.set`


_Set a q object to a named and type specified object in Python memory_

```q
.pykx.set[objectName;qObject]
```

**Parameters:**

name         | type     | description |
-------------|----------|-------------|
`objectName` | `symbol` | The name to be associated with the q object being persisted to Python memory |
`qObject`    | `any`    | The q/Python entity that is to be stored to Python memory

**Return:**

type | description
-----|------------
`::` | Returns null on successful execution

```q
// Set a q array of guids using default behaviour
q).pykx.set[`test;3?0Ng]
q)print .pykx.get`test
[UUID('3d13cc9e-f7f1-c0ee-782c-5346f5f7b90e')
 UUID('c6868d41-fa85-233b-245f-55160cb8391a')
 UUID('e1e5fadd-dc8e-54ba-e30b-ab292df03fb0')]

// Set a q table as pandas dataframe
q).pykx.set[`test;.pykx.topd ([]5?1f;5?1f)]
q)print .pykx.get`test
          x        x1
0  0.301772  0.392752
1  0.785033  0.517091
2  0.534710  0.515980
3  0.711172  0.406664
4  0.411597  0.178084

// Set a q table as pyarrow table
q).pykx.set[`test;.pykx.topa ([]2?0p;2?`a`b`c;2?1f;2?0t)]
q)print .pykx.get`test
pyarrow.Table
x: timestamp[ns]
x1: string
x2: double
x3: duration[ns]
----
x: [[2002-06-11 11:57:24.452442976,2001-12-28 01:34:14.199305176]]
x1: [["c","a"]]
x2: [[0.7043314231559634,0.9441670505329967]]
x3: [[2068887000000,41876091000000]]
```

## `.pykx.setattr`


_Set an attribute of a Python object, this is equivalent to calling Python's [setattr(f, a, x)](https://docs.python.org/3/library/functions.html#setattr) function_

```q
.pykx.setattr[pythonObject;attrName;attrObj]
```

**Parameters:**

name           | type                  | description |
---------------|-----------------------|-------------|
`pythonObject` | `foreign/composition` | The Python object on which the defined attribute is to be set |
`attrName`     | `symbol`              | The name to be associated with the set attribute |
`attrObject`   | `any`                 | The object which is to be set as an attribute associated with `pythonObject` |

**Returns:**

type | description |
-----|-------------|
`::` | Returns generic null on successful execution otherwise returns the error message raised

**Example:**

```q
// Define a Python object to which attributes can be set
q).pykx.pyexec"aclass = type('TestClass', (object,), {'x': pykx.LongAtom(3), 'y': pykx.toq('hello')})";
q)a:.pykx.get`aclass

// Retrieve an existing attribute to show defined behaviour
q)a[`:x]`
3

// Retrieve a named attribute that doesn't exist
q)a[`:r]`

// Set an attribute 'r' and retrieve the return
q).pykx.setattr[a; `r; til 4]
q)a[`:r]`
0 1 2 3
q).pykx.print a[`:r]
[0 1 2 3]

// Set an attribute 'k' to be a Pandas type
q).pykx.setattr[a;`k;.pykx.topd ([]2?1f;2?0Ng;2?`2)]
q)a[`:k]`
x         x1                                   x2
-------------------------------------------------
0.4931835 0a3e1784-0125-1b68-5ae7-962d49f2404d mi
0.5785203 5aecf7c8-abba-e288-5a58-0fb6656b5e69 ig
q).pykx.print a[`:k]
          x                                    x1  x2
0  0.493183  0a3e1784-0125-1b68-5ae7-962d49f2404d  mi
1  0.578520  5aecf7c8-abba-e288-5a58-0fb6656b5e69  ig

// Attempt to set an attribute against an object which does not support this behaviour
q)arr:.pykx.eval"[1, 2, 3]"
q).pykx.setattr[arr;`test;5]
'AttributeError("'list' object has no attribute 'test'")
  [1]  /opt/kx/pykx.q:218: .pykx.util.setattr:
  cx:count x;
  util.setAttr[unwrap x 0;x 1;;x 2]
  ^
    $[cx>4;
```

## `.pykx.setdefault`


_Define the default conversion type for KX objects when converting from q to Python_

```q
.pykx.setdefault[conversionFormat]
```

**Parameters:**

name               | type   | description |
-------------------|--------|-------------|
`conversionFormat` | string | The Python data format to which all q objects when passed to Python will be converted. |

**Returns:**

type | description |
-----|-------------|
`::` | Returns generic null on successful execution and updates variable `.pykx.util.defaultConv`

??? "Supported Options"

    The following outline the supported conversion types and the associated values which can be passed to set these values

    Conversion Format                                              | Accepted inputs              |
    ---------------------------------------------------------------|------------------------------|
    [Numpy](https://numpy.org/)                                    | `"np", "numpy", "Numpy"`     |
    [Pandas](https://pandas.pydata.org/docs/user_guide/index.html) | `"pd", "pandas", "Pandas"`   |
    [Python](https://docs.python.org/3/library/datatypes.html)     | `"py", "python", "Python"`   |
    [PyArrow](https://arrow.apache.org/docs/python/index.html)     | `"pa", "pyarrow", "PyArrow"` |
    [K](../api/pykx-q-data/type_conversions.md)                    | `"k", "q"`                   |


```q
// Default value on startup is "np"
q).pykx.util.defaultConv
"np"

// Set default value to Pandas
q).pykx.setdefault["Pandas"]
q).pykx.util.defaultConv
"pd"
```

## `.pykx.tok`


_Tag a q object to be indicate conversion to a Pythonic PyKX object when called in Python_

```q
.pykx.tok[qObject]
```

**Parameters:**

name      | type    | description |
----------|---------|-------------|
`qObject` | `any`   | A q object which is to be defined as a PyKX object in Python. |

**Return:**

type         | description
-------------|------------
`projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a PyKX type object. |

```q
// Denote that a q object once passed to Python should be managed as a PyKX object
q).pykx.tok til 10
enlist[`..k;;][0 1 2 3 4 5 6 7 8 9]

// Pass a q object to Python with default conversions and return type
q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
<class 'numpy.ndarray'>

// Pass a q object to Python treating the Python object as a PyKX object
q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.tok til 10
<class 'pykx.wrappers.LongVector'>
```

## `.pykx.tonp`


_Tag a q object to be indicate conversion to a Numpy object when called in Python_

```q
.pykx.tonp[qObject]
```

**Parameters:**

name      | type    | description |
----------|---------|-------------|
`qObject` | `any`   | A q object which is to be defined as a Numpy object in Python. |

**Return:**

type         | description
-------------|------------
`projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a Numpy type object. |

```q
// Denote that a q object once passed to Python should be managed as a Numpy object
q).pykx.tonp til 10
enlist[`..numpy;;][0 1 2 3 4 5 6 7 8 9]

// Update the default conversion type to be non numpy
q).pykx.util.defaultConv:"py"

// Pass a q object to Python with default conversions and return type
q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
<class 'list'>

// Pass a q object to Python treating the Python object as a Numpy Object
q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.tonp til 10
<class 'numpy.ndarray'>
```

## `.pykx.topa`


_Tag a q object to be indicate conversion to a PyArrow object when called in Python_

```q
.pykx.topa[qObject]
```

**Parameters:**

name      | type    | description |
----------|---------|-------------|
`qObject` | `any`   | A q object which is to be defined as a PyArrrow object in Python. |

**Return:**

type         | description
-------------|------------
`projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a PyArrow type object. |

```q
// Denote that a q object once passed to Python should be managed as a PyArrow object
q).pykx.topa til 10
enlist[`..pyarrow;;][0 1 2 3 4 5 6 7 8 9]

// Pass a q object to Python with default conversions and return type
q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
<class 'numpy.ndarray'>

// Pass a q object to Python treating the Python object as a PyArrow Object
q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.topa til 10
<class 'pyarrow.lib.Int64Array'>
```

## `.pykx.topd`


_Tag a q object to be indicate conversion to a Pandas object when called in Python_

```q
.pykx.topd[qObject]
```

**Parameters:**

name      | type    | description |
----------|---------|-------------|
`qObject` | `any`   | A q object which is to be defined as a Pandas object in Python. |

**Return:**

type         | description
-------------|------------
`projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a Pandas type object. |

```q
// Denote that a q object once passed to Python should be managed as a Pandas object
q).pykx.topd til 10
enlist[`..pandas;;][0 1 2 3 4 5 6 7 8 9]


// Pass a q object to Python with default conversions and return type
q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
<class 'numpy.ndarray'>

// Pass a q object to Python treating the Python object as a Pandas Object
q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.topd til 10
<class 'pandas.core.series.Series'>
```

## `.pykx.topy`


_Tag a q object to be indicate conversion to a Python object when called in Python_

```q
.pykx.topy[qObject]
```

**Parameters:**

name      | type    | description |
----------|---------|-------------|
`qObject` | `any`   | A q object which is to be defined as a Python object in Python. |

**Return:**

type         | description
-------------|------------
`projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a Python type object. |

```q
// Denote that a q object once passed to Python should be managed as a Python object
q).pykx.topy til 10
enlist[`..python;;][0 1 2 3 4 5 6 7 8 9]

// Pass a q object to Python with default conversions and return type
q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
<class 'numpy.ndarray'>

// Pass a q object to Python treating the Python object as a Python Object
q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.topy til 10
<class 'list'>
```

## `.pykx.toraw`


_Tag a q object to be indicate a raw conversion when called in Python_

```q
.pykx.toraw[qObject]
```

**Parameters:**

name      | type    | description |
----------|---------|-------------|
`qObject` | `any`   | A q object which is to be converted in its raw form in Python. |

**Return:**

type         | description
-------------|------------
`projection` | A projection which is used to indicate that once the q object is passed to Python for evaluation is should be treated as a raw object. |

```q
// Denote that a q object once passed to Python should be managed as a Numpy object
q).pykx.toraw til 10
enlist[`..raw;;][0 1 2 3 4 5 6 7 8 9]

// Pass a q object to Python with default conversions and return type
q).pykx.print .pykx.eval["lambda x: type(x)"]til 10
<class 'numpy.ndarray'>

// Pass a q object to Python treating the Python object as a raw Object
q).pykx.print .pykx.eval["lambda x: type(x)"] .pykx.toraw til 10
<class 'list'>
```

## `.pykx.unwrap`


_Convert a wrapped foreign object generated from this interface into a python foreign._

```q
.pykx.unwrap[wrapObj]
```

**Parameters:**

 name      | type                | description |
-----------|---------------------|-------------|
 `wrapObj` | composition/foreign | A (un)wrapped Python foreign object. |

**Returns:**

 type      | description |
-----------|-------------|
 `foreign` | The unwrapped representation of the Python foreign object. |

```q
// Generate an object which returns a wrapped Python foreign
q).pykx.set[`test;.pykx.topd ([]2?0p;2?`a`b`c;2?1f;2?0t)]
q)a:.pykx.get`test
q)show a
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist

// Unwrap the wrapped object
q).pykx.unwrap a
foreign
```

## `.pykx.version`


_Retrieve the version of PyKX presently being used by a q process_

```q
.pykx.version[]
```

**Return:**

type     | description
---------|------------
`string` | The version number of PyKX installed within the users q session

```q
q).pykx.version[]
"2.0.0"
```

## `.pykx.wrap`


_Convert a foreign object generated from Python execution to a callable `q` object._

```q
.pykx.wrap[pyObject]
```

**Parameters:**

name       | type      | description |
-----------|-----------|-------------|
`pyObject` | `foreign` | A Python object which is to be converted to a callable q object. |

**Returns:**

type          | description |
--------------|-------------|
`composition` | The Python object wrapped such that it can be called using q |

```q
// Create a q foreign object in Python
q)a:.pykx.pyeval"pykx.Foreign([1, 2, 3])"
q)a
foreign
q).pykx.print a
[1, 2, 3]

// Wrap the foreign object and convert to q
q)b:.pykx.wrap a
q)b
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist
q)b`
1 2 3
```

## pyarglist


_Allow users to apply a list of arguments to a Python function, equivalent to `*args`_

```q
pyarglist argList
```

!!! Warning

	This function will be set in the root `.q` namespace

**Parameters:**

name       | type   | description
-----------|--------|------------
`argList`  | `list` | List of opsitional arguments

**Return:**

type         | description
-------------|------------
`projection` | A projection which when used with a wrapped callable Python

**Example:**

The following example shows the usage of `pyarglist` with a Python function and
various configurations of it's use

```q
q)p)import numpy as np
q)p)def func(a=1,b=2,c=3,d=4):return np.array([a,b,c,d,a*b*c*d])
q)qfunc:.pykx.get[`func;<] / callable, returning q
q)qfunc[pyarglist 1 1 1 1]          / full positional list specified
1 1 1 1 1
q)qfunc[pyarglist 1 1]              / partial positional list specified
1 1 3 4 12
q)qfunc[1;1;pyarglist 2 2]          / mix of positional args and positional list
1 1 2 2 4
q)qfunc[pyarglist 1 1;`d pykw 5]    / mix of positional list and keyword args
1 1 3 5 15
```

## pykw


_Allow users to apply individual keywords to a Python function_

```q
`argName pykw argValue
```

!!! Warning

     This function will be set in the root `.q` namespace

**Parameters:**

name       | type     | description
-----------|----------|------------
`argName`  | `symbol` | Name of the keyword argument to be applied
`argValue` | `any`    | Value to be applied as a keyword

**Return:**

type         | description
-------------|------------
`projection` | A projection which when used with a wrapped callable Python 

**Example:**

The following example shows the usage of `pykw` with a Python function

```q
q)p)import numpy as np
q)p)def func(a=1,b=2,c=3,d=4):return np.array([a,b,c,d,a*b*c*d])
q)qfunc:.pykx.get[`func;<] / callable, returning q
q)qfunc[`d pykw 1;`c pykw 2;`b pykw 3;`a pykw 4] / all keyword args specified
4 3 2 1 24
q)qfunc[1;2;`d pykw 3;`c pykw 4]   / mix of positional and keyword args
```

## pykwargs


_Allow users to apply a dictionary of named arguments to a Python function, equivalent to `*kwargs`_

```q
pykwargs argDict
```

!!! Warning

     This function will be set in the root `.q` namespace 

**Parameters:**

name       | type   | description
-----------|--------|------------
`argDict`  | `dict` | A dictionary of named keyword arguments mapped to their value

**Return:**

type         | description
-------------|------------
`projection` | A projection which when used with a wrapped callable Python

**Example:**

The following example shows the usage of `pykwargs` with a Python function and
various configurations of it's use

```q
q)p)import numpy as np
q)p)def func(a=1,b=2,c=3,d=4):return np.array([a,b,c,d,a*b*c*d])
q)qfunc:.pykx.get[`func;<] / callable, returning q
q)qfunc[pykwargs`d`c`b`a!1 2 3 4]             / full keyword dict specified
4 3 2 1 24
q)qfunc[2;2;pykwargs`d`c!3 3]                 / mix of positional args and keyword dict
2 2 3 3 36
q)qfunc[`d pykw 1;`c pykw 2;pykwargs`a`b!3 4] / mix of keyword args and keyword dict
3 4 2 1 24
```

## `.pykx.pyimport`


_Import a Python library and store as a foreign object._

```q
.pykx.pyimport[libName]
```

**Parameters:**

name      | type   | description |
----------|--------|-------------|
`libName` | symbol | The name of the Python library/module to imported for use |

**Return:**

type      | description
----------|------------
`foreign` | Returns a foreign object associated with an imported library on success, otherwise will error if library/module cannot be imported.

```q
// Import numpy for use as a q object named numpy
q)np:.pykx.pyimport`numpy
q).pykx.print np
<module 'numpy' from '/usr/local/lib64/python3.9/site-packages/numpy/__init__.py'>
```

## `.pykx.getattr`


_Retrieve an attribute or property form a foreign Python object returning another foreign._

```q
.pykx.getattr[pythonObject;attrName]
```

**Parameters:**

name           | type                  | description
---------------|-----------------------|-------------
`pythonObject` | `foreign/composition` | The Python object from which the defined attribute is to be retrieved.
`attrName`     | `symbol`              | The name of the attribute to be retrieved.

**Returns:**

type      | description
----------|------------
`foreign` | An unwrapped foreign object containing the retrieved

!!! Note

    Application of this function is equivalent to calling Python's [`getattr(f, 'x')`](https://docs.python.org/3/library/functions.html#getattr) function.

    The wrapped foreign objects provide a shorthand version of calling `.pykx.getattr`. Through the use of the ````:x``` syntax for attribute/property retrieval

**Example:**

```q
// Define a class object from which to retrieve Python attributes
q).pykx.pyexec"aclass = type('TestClass', (object,), {'x': pykx.LongAtom(3), 'y': pykx.toq('hello')})";

// Retrieve the class object from Python as a q foreign
q)show a:.pykx.get[`aclass]`.
foreign

// Retrieve an attribute from the Python foreign
q).pykx.getattr[a;`y]
foreign

// Print the Python representation of the foreign object
q)print .pykx.getattr[a;`y]
hello

// Retrieve the attribute from a Python foreign and convert to q
q).pykx.wrap[.pykx.getattr[a;`y]]`
`hello
```

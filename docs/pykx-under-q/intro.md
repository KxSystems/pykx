# Using PyKX within a q session

## Introduction

As described in the majority of the documentation associated with PyKX, the principal intended usage of the library is as Python first interface to the programming language q and it's underlying database kdb+. However as described in the limitations section [here](../user-guide/advanced/limitations.md) not all use-cases can be satisfied with this modality. In particular software relying on the use of active subscriptions such as real-time analytic engines or any functionality reliant on timers in q cannot be run from Python directly without reimplementing this logic Pythonically.

As such a modality is distributed with PyKX which allows Python functionality to be run from within a q session. This is achieved through the creation of a domain-specific language (DSL) which allows for the execution and manipulation of Python objects within a q session. Providing this functionality allows users proficient in kdb+/q to build applications which embed machine learning/data science libraries within production q infrastructures and allows users to use plotting libraries to visualise the outcomes of their analyses.

## Getting started

### Prerequisites

To make use of PyKX running embedded within a q session a user must have the following set up

1. The user has access to a running `q` environment, follow the q installation guide [here](https://code.kx.com/q/learn/install/) for more information.
2. The user is permissioned to run PyKX with access to a license containing the feature flags `insights.lib.pykx` and `insights.lib.embedq` For more information see [here](../getting-started/installing.md).

### Installation

To facilitate the execution of Python code within a q session a user must first install the PyKX library and the q script used to drive this embedded feature into their `$QHOME` location. This can be done as follows.

1. Install the PyKX library following the instructions [here](../getting-started/installing.md).
2. Run the following command to install the `pykx.q` script:

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

### Initialisation

Once installation has been completed a user should be in a position to initialise the library as follows

```q
q)\l pykx.q
q).pykx
console    | {pyexec"pykx.console.PyConsole().interact(banner='', exitmsg='')"}
getattr    | code
get        | {[f;x]r:wrap f x 0;$[count x:1_x;.[;x];]r}[code]enlist
setattr    | {i.load[(`set_attr;3)][unwrap x;y;i.convertArg[i.toDefault z]`.]}
set        | {i.load[(`set_global;2)][x; i.convertArg[i.toDefault y]`.]}
print      | {$[type[x]in 104 105 112h;i.repr[0b] unwrap x;show x];}
repr       | {$[type[x]in 104 105 112h;i.repr[1b] unwrap x;.Q.s x]}
import     | {[f;x]r:wrap f x 0;$[count x:1_x;.[;x];]r}[code]enlist
..
```

## Using the library

Usage of the functionality provided by this library can range in complexity from the simple execution of Python code through to the generation of streaming applications containing machine learning models. The following documentation section outlines the use of this library under various use-case agnostic scenarios

### Evaluating and Executing Python code

#### Executing Python code

This interface allows a user to execute Python code a variety of ways:

1. Executing directly using the `.pykx.pyexec` function

	This is incredibly useful if there is a requirement to script execution of Python code within a library

	```q
	q).pykx.pyexec"import numpy as np"
	q).pykx.pyexec"array = np.array([0, 1, 2, 3])"
	q).pykx.pyexec"print(array)"
	[0 1 2 3]
	```

2. Usage of the PyKX console functionality

	This is useful when interating within a q session and needing to prototype some functionality in Python

	```q
	q).pykx.console[]
	>>> import numpy as np
	>>> print(np.linspace(0, 10, 5))
	[ 0.   2.5  5.   7.5 10. ]
	>>> quit()
	q)
	```

3. Execution through use of a `p)` prompt

	Provided as a way to embed execution of Python code within a q script, additionally this provides backwards compatibility with PyKX.

	```q
	q)p)import numpy as np
	q)p)print(np.arange(1, 10, 2))
	[1 3 5 7 9]
	```

4. Loading of a `.p` file

	This is provided as a method of executing the contents of a Python file in bulk.

	```q
	$ cat test.p
	def func(x, y):
            return(x+y)
    $ q pykx.q
    q)\l test.p
    q).pykx.get[`func]
    {[f;x].pykx.i.pykx[f;x]}[foreign]enlist
	```

#### Evaluating Python code

The evaluation of Python code can be completed using PyKX by passing a string of Python code to a variety of functions.

??? "Differences between evaluation and execution"

	Python evaluation (unlike Python execution) does not allow side effects. Any attempt at variable assignment or class definition will signal an error. To execute a string performing side effects, use `.pykx.pyexec` or `.p.e`.

	[Difference between eval and exec in Python](https://stackoverflow.com/questions/2220699/whats-the-difference-between-eval-exec-and-compile)

To evaluate Python code and return the result to `q`, use the function `.pykx.qeval`.

```q
q).pykx.qeval"1+2"
3
```

Similarly to evaluate Python code and return the result as a `foreign` object denoting the underlying Python object

```q
q)show a:.pykx.pyeval"1+2"
foreign
q)print a
3
```

Finally to return a hybrid representation which can be manipulated to return the q or Python representation you can run the following

```q
q)show b:.pykx.eval"1+2"
{[f;x].pykx.i.pykx[f;x]}[foreign]enlist
q)b`       // Convert to a q object
3
q)b`.      // Convert to a Python foreign
foreign
```

## Interacting with PyKX objects

### Foreign objects

At the lowest level, Python objects are represented in q as foreign objects, which contain pointers to objects in the Python memory space.

Foreign objects can be stored in variables just like any other q datatype, or as part of lists, dictionaries or tables. They will display as foreign when inspected in the q console or using the string (or .Q.s) representation.

**Serialization:** Kdb+ cannot serialize foreign objects, nor send them over IPC: they live in the embedded Python memory space. To pass these objects over IPC, first convert them to q.

### PyKX objects

Foreign objects cannot be directly operated on in q. Instead, Python objects are typically represented as PyKX objects, which wrap the underlying foreign objects. This provides the ability to get and set attributes, index, call or convert the underlying foreign object to a q object.

Use .pykx.wrap to create an PyKX object from a foreign object.

```q
q)x
foreign
q)p:.pykx.wrap x
q)p           /how an PyKX object looks
{[f;x].pykx.i.pykx[f;x]}[foreign]enlist
```

More commonly, PyKX objects are retrieved directly from Python using one of the following functions:

function       | argument                                         | example
---------------|--------------------------------------------------|-----------------------
`.pykx.import` | symbol: name of a Python module or package, optional second argument is the name of an object within the module or package | ``np:.pykx.import`numpy``
`.pykx.get`    | symbol: name of a Python variable in `__main__`  | ``v:.pykx.get`varName``
`.pykx.eval`   | string: Python code to evaluate                  | `x:.pykx.eval"1+1"`

**Side effects:** As with other Python evaluation functions and noted previously, `.pykx.eval` does not permit side effects.

### Converting data

Given `obj`, an PyKX object representing Python data, we can get the underlying data (as foreign or q) using

```q
obj`. / get data as foreign
obj`  / get data as q
```

For example:

```q
q)x:.pykx.eval"(1,2,3)"
q)x
{[f;x].pykx.i.pykx[f;x]}[foreign]enlist
q)x`.
foreign
q)x`
1 2 3
```

### `None` and identity

Python `None` maps to the q identity function `::` when converting from Python to q (and vice versa).

There is one important exception to this. When calling Python functions, methods or classes with a single q data argument, passing `::` will result in the Python object being called with _no_ arguments, rather than a single argument of `None`. See the section below on _Zero-argument calls_ for how to explicitly call a Python callable with a single `None` argument.

### Getting attributes and properties

Given `obj`, an PyKX object representing a Python object, we can get an attribute or property directly using

```q
obj`:attr         / equivalent to obj.attr in Python
obj`:attr1.attr2  / equivalent to obj.attr1.attr2 in Python
```

These expressions return PyKX objects, allowing users to chain operations together.

```q
obj[`:attr1]`:attr2  / equivalent to obj.attr1.attr2 in Python
```

e.g.

```bash
$ cat class.p
class obj:
    def __init__(self,x=0,y=0):
        self.x = x
        self.y = y
```

```q
q)\l class.p
q)obj:.pykx.eval"obj(2,3)"
q)obj[`:x]`
2
q)obj[`:y]`
3
```

### Setting attributes and properties

Given `obj`, an PyKX object representing a Python object, we can set an attribute or property directly using

```q
obj[:;`:attr;val]  / equivalent to obj.attr=val in Python
```

e.g.

```q
q)obj[`:x]`
2
q)obj[`:y]`
3
q)obj[:;`:x;10]
q)obj[:;`:y;20]
q)obj[`:x]`
10
q)obj[`:y]`
20
```

### Indexing

Given `lst`, an PyKX object representing an indexable container object in Python, we can access the element at index `i` using

```q
lst[@;i]    / equivalent to lst[i] in Python
```

We can set the element at index `i` (to object `x`) using

```q
lst[=;i;x]  / equivalent to lst[i]=x in Python
```

These expressions return PyKX objects, e.g.

```q
q)lst:.pykx.eval"[True,2,3.0,'four']"
q)lst[@;0]`
1b
q)lst[@;-1]`
`four
q)lst'[@;;`]2 1 0 3
3f
2
1b
`four
q)lst[=;0;0b];
q)lst[=;-1;`last];
q)lst`
0b
2
3f
`last
```

### Getting methods

Given `obj`, an PyKX object representing a Python object, we can access a method directly using

```q
obj`:method  / equivalent to obj.method in Python
```

Presently the calling of PyKX objects representing Python methods is only supported in such a manner that the return of evaluation is a PyKX object.

For example

```q
q)np:.pykx.import`numpy
q)np`:arange
{[f;x].pykx.i.pykx[f;x]}[foreign]enlist
q)arange:np`:arange                   / callable returning PyKX object
q)arange 12
{[f;x].pykx.i.pykx[f;x]}[foreign]enlist
q)arange[12]`
0 1 2 3 4 5 6 7 8 9 10 11
```

### PyKX function API

Using the function API, PyKX objects can be called directly (returning PyKX objects) or declared callable returning q or `foreign` data.

Users explicitly specify the return type as q or foreign, the default is as a PyKX object.

Given `func`, an `PyKX` object representing a callable Python function or method, we can carry out the following operations:

```q
func                   / func is callable by default (returning PyKX)
func arg               / call func(arg) (returning PyKX)
func[<]                / declare func callable (returning q)
func[<]arg             / call func(arg) (returning q)
func[<;arg]            / equivalent
func[>]                / declare func callable (returning foreign)
func[>]arg             / call func(arg) (returning foreign)
func[>;arg]            / equivalent
```

**Chaining operations** Returning another PyKX object from a function or method call, allows users to chain together sequences of operations.  We can also chain these operations together with calls to `.pykx.import`, `.pykx.get` and `.pykx.eval`.


### PyKX examples

Some examples

```bash
$ cat test.p # used for tests
class obj:
    def __init__(self,x=0,y=0):
        self.x = x # attribute
        self.y = y # property (incrementing on get)
    @property
    def y(self):
        a=self.__y
        self.__y+=1
        return a
    @y.setter
    def y(self, y):
        self.__y = y
    def total(self):
        return self.x + self.y
```

```q
q)\l test.p
q)obj:.pykx.get`obj / obj is the *class* not an instance of the class
q)o:obj[]           / call obj with no arguments to get an instance
q)o[`:x]`
0
q)o[;`]each 5#`:x
0 0 0 0 0
q)o[:;`:x;10]
q)o[`:x]`
10
q)o[`:y]`
1
q)o[;`]each 5#`:y
3 5 7 9 11
q)o[:;`:y;10]
q)o[;`]each 5#`:y
10 13 15 17 19
q)tot:o[`:total;<]
q)tot[]
30
q)tot[]
31
```

```q
q)np:.pykx.import`numpy
q)v:np[`:arange;12]
q)v`
0 1 2 3 4 5 6 7 8 9 10 11
q)v[`:mean;<][]
5.5
q)rs:v[`:reshape;<]
q)rs[3;4]
0 1 2  3
4 5 6  7
8 9 10 11
q)rs[2;6]
0 1 2 3 4  5
6 7 8 9 10 11
q)np[`:arange;12][`:reshape;3;4]`
0 1 2  3
4 5 6  7
8 9 10 11
```

```q
q)stdout:.pykx.import[`sys]`:stdout.write
q)stdout `$"hello\n";
hello
q)stderr:.pykx.import[`sys;`:stderr.write]
q)stderr `$"goodbye\n";
goodbye
```

```q
q)oarg:.pykx.eval"10"
q)oarg`
10
q)ofunc:.pykx.eval["lambda x:2+x";<]
q)ofunc[1]`
3
q)ofunc oarg
12
q)p)def add2(x,y):return x+y
q)add2:.pykx.get[`add2;<]
q)add2[1;oarg]
11
```

### Function argument types

One of the distinct differences that PyKX has over the previous incarnation of embedded interfacing with Python in q PyKX is support for a much wider variety of data type conversions between q and Python.

In particular the following types are supported:

1. Python native objects
2. Numpy objects
3. Pandas objects
4. PyArrow objects
5. PyKX objects

By default when passing a q object to a callable function it will be converted to it's underlying Numpy equivalent representation. This will be the case for all types including tabular structures which are converted to numpy records.

For example:

```q
q)typeFunc:.pykx.eval"lambda x:print(type(x))"
q)typeFunc 1; 
<class 'numpy.int64'>
q)typeFunc til 10;
<class 'numpy.ndarray'>
q)typeFunc ([]100?1f;100?1f);
<class 'numpy.recarray'>
```

The default behaviour of the conversions which are undertaken when making function/method calls is controlled through the definition of `.pykx.i.defaultConv`

```q
q).pykx.i.defaultConv
"np"
```

This can have one of the following values:

| Python type | Value |
|-------------|-------|
| Python      | "py"  |
| Numpy       | "np"  |
| Pandas      | "pd"  |
| PyArrow     | "pa"  |
| PyKX        | "k"   |

Taking the examples above for numpy we can update the default types across all function calls

```q
q)typeFunc:.pykx.eval"lambda x:print(type(x))"
q).pykx.i.defaultConv:"py"
q)typeFunc 1;
<class 'int'>
q)typeFunc til 10;
<class 'list'>
q)typeFunc ([]100?1f;100?1f);
<class 'dict'>

q).pykx.i.defaultConv:"pd"
q)typeFunc 1;
<class 'numpy.int64'>
q)typeFunc til 10;
<class 'pandas.core.series.Series'>
q)typeFunc ([]100?1f;100?1f);
<class 'pandas.core.frame.DataFrame'>

q).pykx.i.defaultConv:"pa"
q)typeFunc 1;
<class 'numpy.int64'>
q)typeFunc til 10;
<class 'pyarrow.lib.Int64Array'>
q)typeFunc ([]100?1f;100?1f);
<class 'pyarrow.lib.Table'>

q).pykx.i.defaultConv:"k"
q)typeFunc 1;
<class 'pykx.wrappers.LongAtom'>
q)typeFunc til 10;
<class 'pykx.wrappers.LongVector'>
q)typeFunc ([]100?1f;100?1f);
<class 'pykx.wrappers.Table'>
```

Alternatively individual arguments to functions can be modified using the `.pykx.to*` functionality, for example in the following:

```q
q)typeFunc:.pykx.eval"lambda x,y: [print(type(x)), print(type(y))]"
q)typeFunc[til 10;til 10];                          // Simulate passing both arguments with defaults
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
q)typeFunc[til 10].pykx.topd til 10;               // Pass in the second argument as Pandas series
<class 'numpy.ndarray'>
<class 'pandas.core.series.Series'>
q)typeFunc[.pykx.topa([]100?1f);til 10];           // Pass in first argument as PyArrow Table
<class 'pyarrow.lib.Table'>
<class 'numpy.ndarray'>
q)typeFunc[.pykx.tok til 10;.pykx.tok ([]100?1f)]; // Pass in two PyKX objects
<class 'pykx.wrappers.LongVector'>
<class 'pykx.wrappers.Table'>
```

### Setting Python variables

Variables can be set in Python `__main__` using `.pykx.set`

```q
q).pykx.set[`var1;42]
q).pykx.qeval"var1"
42
q).pykx.set[`var2;{x*2}]
q)qfunc:.pykx.get[`var2;<]
{[f;x].pykx.i.pykx[f;x]}[foreign]enlist
q)qfunc[3]
6
```

## Function calls


Python allows for calling functions with

-   A variable number of arguments
-   A mixture of positional and keyword arguments
-   Implicit (default) arguments

All of these features are available through the PyKX function-call interface.
Specifically:

-   Callable PyKX objects are variadic
-   Default arguments are applied where no explicit arguments are given
-   Individual keyword arguments are specified using the (infix) `pykw` operator
-   A list of positional arguments can be passed using `pyarglist` (like Python \*args)
-   A dictionary of keyword arguments can be passed using `pykwargs` (like Python \*\*kwargs)

**Keyword arguments last** We can combine positional arguments, lists of positional arguments, keyword arguments and a dictionary of keyword arguments. However, _all_ keyword arguments must always follow _any_ positional arguments. The dictionary of keyword arguments (if given) must be specified last.


### Example function calls

```q
q)p)import numpy as np
q)p)def func(a=1,b=2,c=3,d=4):return np.array([a,b,c,d,a*b*c*d])
q)qfunc:.pykx.get[`func;<] / callable, returning q
```

Positional arguments are entered directly.
Function calling is variadic, so later arguments can be excluded.

```q
q)qfunc[2;2;2;2]   / all positional args specified
2 2 2 2 16
q)qfunc[2;2]       / first 2 positional args specified
2 2 3 4 48
q)qfunc[]          / no args specified
1 2 3 4 24
q)qfunc[2;2;2;2;2] / error if too many args specified
'TypeError('func() takes from 0 to 4 positional arguments but 5 were given')
  [0]  qfunc[2;2;2;2;2] / error if too many args specified
       ^
```

Individual keyword arguments can be specified using the `pykw` operator (applied infix).
Any keyword arguments must follow positional arguments, but the order of keyword arguments does not matter.

```q
q)qfunc[`d pykw 1;`c pykw 2;`b pykw 3;`a pykw 4] / all keyword args specified
4 3 2 1 24
q)qfunc[1;2;`d pykw 3;`c pykw 4]   / mix of positional and keyword args
1 2 4 3 24
q)qfunc[`a pykw 2;`b pykw 2;2;2]   / error if positional args after keyword args
'TypeError("func() got multiple values for argument 'a'")
  [0]  qfunc[`a pykw 1;pyarglist 2 2 2] 
       ^
q)qfunc[`a pykw 2;`a pykw 2]       / error if duplicate keyword args
'Expected only unique key names for keyword arguments in function call
  [0]  qfunc[`a pykw 2;`a pykw 2]  
       ^
```

A list of positional arguments can be specified using `pyarglist` (similar to Python’s \*args).
Again, keyword arguments must follow positional arguments.

```q
q)qfunc[pyarglist 1 1 1 1]          / full positional list specified
1 1 1 1 1
q)qfunc[pyarglist 1 1]              / partial positional list specified
1 1 3 4 12
q)qfunc[1;1;pyarglist 2 2]          / mix of positional args and positional list
1 1 2 2 4
q)qfunc[pyarglist 1 1;`d pykw 5]    / mix of positional list and keyword args
1 1 3 5 15
q)qfunc[pyarglist til 10]           / error if too many args specified
'TypeError('func() takes from 0 to 4 positional arguments but 10 were given')
  [0]  qfunc[pyarglist til 10]      / error if too many args specified
       ^
q)qfunc[`a pykw 1;pyarglist 2 2 2]  / error if positional list after keyword args
'TypeError("func() got multiple values for argument 'a'")
  [0]  qfunc[`a pykw 1;pyarglist 2 2 2] 
       ^
```


A dictionary of keyword arguments can be specified using `pykwargs` (similar to Python’s \*\*kwargs).
If present, this argument must be the _last_ argument specified.

```q
q)qfunc[pykwargs`d`c`b`a!1 2 3 4]             / full keyword dict specified
4 3 2 1 24
q)qfunc[2;2;pykwargs`d`c!3 3]                 / mix of positional args and keyword dict
2 2 3 3 36
q)qfunc[`d pykw 1;`c pykw 2;pykwargs`a`b!3 4] / mix of keyword args and keyword dict
3 4 2 1 24
q)qfunc[pykwargs`d`c!3 3;2;2]                 / error if keyword dict not last
'pykwargs last
q)qfunc[pykwargs`a`a!1 2]                     / error if duplicate keyword names
'dupnames
```

All 4 methods can be combined in a single function call, as long as the order follows the above rules.

```q
q)qfunc[4;pyarglist enlist 3;`c pykw 2;pykwargs enlist[`d]!enlist 1]
4 3 2 1 24
```

!!! warning "`pykw`, `pykwargs`, and `pyarglist`"

    Before defining functions containing `pykw`, `pykwargs`, or `pyarglist` within a script, the file `p.q` must be loaded explicitly. 
    Failure to do so will result in errors `'pykw`, `'pykwargs`, or `'pyarglist`.

### Zero-argument calls

In Python these two calls are _not_ equivalent:

```python
func()       #call with no arguments
func(None)   #call with argument None
```

!!! warning "PyKX function called with `::` calls Python with no arguments"

    Although `::` in q corresponds to `None` in Python, if an PyKX function is called with `::` as its only argument, the corresponding Python function will be called with _no_ arguments.

To call a Python function with `None` as its sole argument, retrieve `None` as a foreign object in q and pass that as the argument.

```q
q)pynone:.pykx.eval"None"
q)pyfunc:.pykx.eval["print"]
q)pyfunc pynone;
None
```

Python         | form                      | q
---------------|---------------------------|-----------------------
`func()`       | call with no arguments    | `func[]` or `func[::]`
`func(None)`   | call with argument `None` | `func[.pykx.eval"None"]`

!!! info "Q functions applied to empty argument lists"

    The _rank_ (number of arguments) of a q function is determined by its _signature_,
    an optional list of arguments at the beginning of its definition.
    If the signature is omitted, the default arguments are as many of
    `x`, `y` and `z` as appear, and its rank is 1, 2, or 3.

    If it has no signature, and does not refer to `x`, `y`, or `z`, it has rank 1.
    It is implicitly unary.
    If it is then applied to an empty argument list, the value of `x` defaults to `(::)`.

    So `func[::]` is equivalent to `func[]` – and in Python to `func()`, not `func[None]`.

### Printing or returning object representation


`.pykx.repr` returns the string representation of a Python object, either PyKX or foreign. This representation can be printed to stdout using `.pykx.print`. The usage of this function with a q object

```q
q)x:.pykx.eval"{'a':1,'b':2}"
q).pykx.repr x
"{'a': 1, 'b': 2}"
q).pykx.print x
{'a': 1, 'b': 2}

q).pykx.repr ([]5?1f;5?1f)
"x         x1       \n-------------------\n0.3017723 0.3927524\n0.785033  0.5..
q).pykx.print ([]5?1f;5?1f)
x         x1        
--------------------
0.6137452 0.4931835 
0.5294808 0.5785203 
0.6916099 0.08388858
0.2296615 0.1959907 
0.6919531 0.375638  
```

### Aliases in the root


For convenience, `pykx.q` defines `print` in the default namespace of q, as aliases for `.pykx.print`. To prevent the aliasing of this function please set either:

1. `UNSET_PYKX_GLOBALS` as an environment variable.
2. `unsetPyKXGlobals` as a command line argument when initialising your q session.

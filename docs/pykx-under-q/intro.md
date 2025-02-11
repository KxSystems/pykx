---
title:  PyKX within q 
description: How to use PyKX in a q session
date: June 2024
author: KX Systems, Inc.,
tags: PyKX, q, setup,
---

# How to use PyKX within q

_This page provides details on how to run PyKX within a q session, including how to evaluate and execute Python code, how to interact with objects, and how to call a function._

!!! tip "Tip: For the best experience, we recommend reading [Why upgrade from embedPy](../pykx-under-q/upgrade.md) first." 

## Introduction

PyKX is a Python-first interface to the programming language q and its underlying database kdb+. To overcome a few [limitations](../help/issues.md), PyKX allows you to run Python within q, similarly to [embedPy](https://github.com/kxsystems/embedpy). The ability to execute and manipulate Python objects within a q session helps two types of users in the following ways:

 - kdb+/q users can build applications which embed machine learning/data science libraries in production q infrastructures.
 - Users of Python plotting libraries can visualize and explore the outcomes of their analyses.

## Getting started

### Prerequisites

Before you run PyKX within q, make sure you:

1. Have access to a running `#!python q` environment. [Follow [the q installation guide](https://code.kx.com/q/learn/install/).]
2. Have [installed](../getting-started/installing.md) the licensed version of PyKX.

### Install

Run the following command to install the `#!python pykx.q` script into your `#!python $QHOME` directory:

```python
python -c "import pykx;pykx.install_into_QHOME()"
```

If you previously had `#!python embedPy` installed, pass:

```python
python -c "import pykx;pykx.install_into_QHOME(overwrite_embedpy=True)"
```

If you cannot edit the files in `#!python QHOME`, copy them to your local folder and load `#!python pykx.q` from there:

```bash
python -c "import pykx;pykx.install_into_QHOME(to_local_folder=True)"
```

### Initialize

Initialize the library as follows:

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

## How to use the library

Use this library to complete a wide variety of tasks, from the simple execution of Python code through to the generation of streaming applications containing machine learning models. The next sections outline various use-case-agnostic scenarios that you can follow.

### Evaluate and Execute Python

??? "Differences between evaluation and execution"

	Python evaluation (unlike Python execution) does not allow side effects. Any attempt at variable assignment or class definition signals an error. To execute a string with side effects, use `#!python .pykx.pyexec` or `#!python .p.e`.

	[Difference between eval and exec in Python](https://stackoverflow.com/questions/2220699/whats-the-difference-between-eval-exec-and-compile)

??? info "What’s a Python side effect?"

    A Python function has side effects if it might do more than return a value, for example, modify the state or interact with external entities/systems in a noticeable way. Such effects could manifest as changes to input arguments, modifications to global variables, file operations, or network communications.

#### Evaluate Python code

To evaluate Python code with PyKX, pass a string of Python code to a variety of PyKX functions as shown below.

For example, if you want to evaluate and return the result to `#!python q`, use the function `#!python .pykx.qeval`:

```q
q).pykx.qeval"1+2"
3
```
Similarly, to evaluate Python code and return the result as a `#!python foreign` object denoting the underlying Python object, use:

```q
q)show a:.pykx.pyeval"1+2"
foreign
q)print a
3
```
Finally, to return a hybrid representation that you can edit to return the q or Python representation, run the following:

```q
q)show b:.pykx.eval"1+2"
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist
q)b`       // Convert to a q object
3
q)b`.      // Convert to a Python foreign
foreign
```

#### Execute Python code

This interface allows you to execute Python code in a variety of ways:

a) Execute directly with the `#!python .pykx.pyexec` function

This is incredibly useful if you need to script execution of Python code within a library:

```q
q).pykx.pyexec"import numpy as np"
q).pykx.pyexec"array = np.array([0, 1, 2, 3])"
q).pykx.pyexec"print(array)"
[0 1 2 3]
```

b) Use the PyKX console functionality

This is useful when interacting within a q session and you need to prototype a functionality in Python:

```q
q).pykx.console[]
>>> import numpy as np
>>> print(np.linspace(0, 10, 5))
[ 0.   2.5  5.   7.5 10. ]
>>> quit()
q)
```

c) Use a `#!python p)` prompt

This way of embedding the execution of Python code within a q script also provides backwards compatibility with embedPy:

```q
q)p)import numpy as np
q)p)print(np.arange(1, 10, 2))
[1 3 5 7 9]
```

d) Load a `#!python .p` file

This is a method of executing the contents of a Python file in bulk:

```q
$ cat test.p
def func(x, y):
        return(x+y)
$ q pykx.q
q)\l test.p
q).pykx.get[`func]
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist
```

In some cases the `#!python .p` file being loaded does not contain syntax which can be parsed by q (used by `q)\l test.p` example), as such there is available a `#!q .pykx.loadPy` function:

```q
$ cat test.p
def func(x,
         y
): -> None
    return x+y
$ q pykx.q
q)\l test.p
'SyntaxError('unexpected EOF while parsing...
q).pykx.loadPy["test.p"]
q)f:.pykx.get[`func;<]
q)f[1;2]
3
```

### Interact with PyKX objects

#### Foreign objects

At the lowest level, Python objects are represented in q as foreign objects, which contain pointers to objects in the Python memory space.

You can store foreign objects in variables just like any other q datatype, or as part of lists, dictionaries or tables. They will show up as foreign when inspected in the q console or using the string (or .Q.s) representation.

??? "Serialization and IPC"

	Kdb+ cannot serialize foreign objects, nor send them over IPC. Foreign objects live in the embedded Python memory space. To pass them over IPC, first you have to convert them to q.

#### Create PyKX objects

q doesn't allow you to operate directly with foreign objects. Instead, Python objects are represented as PyKX objects, which wrap the underlying foreign objects. This helps to get and set attributes, index, call or convert the underlying foreign object to a q object.

Use `#!python .pykx.wrap` to create a PyKX object from a foreign object.

```q
q)x
foreign
q)p:.pykx.wrap x
q)p           /how a PyKX object looks
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist
```

To retrieve PyKX objects directly from Python, choose between the following functions:

**Function**   | **Argument**                                     | **Example**
---------------|--------------------------------------------------|-----------------------
`.pykx.import` | symbol: name of a Python module or package, optional second argument is the name of an object within the module or package | ``np:.pykx.import`numpy``
`.pykx.get`    | symbol: name of a Python variable in `__main__`  | ``v:.pykx.get`varName``
`.pykx.eval`   | string: Python code to evaluate                  | `x:.pykx.eval"1+1"`


!!! warning "Side effects"

	As with other Python evaluation functions, `#!python .pykx.eval` does not allow side effects.

#### Convert data

For `#!python obj`, a PyKX object representing Python data, to obtain the underlying data (as foreign object or q) use:

```q
obj`. / get data as foreign
obj`  / get data as q
```

For example:

```q
q)x:.pykx.eval"(1,2,3)"
q)x
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist
q)x`.
foreign
q)x`
1 2 3
```

#### `#!python None` and identity

Python `#!python None` maps to the q identity function `#!python ::` when converting from Python to q (and vice versa).

!!! warning "Exception!"

	When calling Python functions, methods or classes with a single q data argument, passing `::` results in the Python object being called with _no arguments_, rather than a single argument of `None`. See the [Zero-argument calls](#zero-argument-calls) section for how to call a Python object with a single `None` argument.

#### Get attributes and properties

Given `#!python obj`, a PyKX object representing a Python object, you can get an attribute or property by using:

```q
obj`:attr         / equivalent to obj.attr in Python
obj`:attr1.attr2  / equivalent to obj.attr1.attr2 in Python
```

These expressions return PyKX objects, allowing you to chain operations together:

```q
obj[`:attr1]`:attr2  / equivalent to obj.attr1.attr2 in Python
```

For example:

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

#### Set attributes and properties

Given `#!python obj`, a PyKX object representing a Python object, you can set an attribute or property by using:

```q
obj[:;`:attr;val]  / equivalent to obj.attr=val in Python
```

For example:

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

#### How to index

Given `#!python lst`, a PyKX object representing an indexable container object in Python, you can access the element at index `#!python i` by using:

```q
lst[@;i]    / equivalent to lst[i] in Python
```

Set the element at index `#!python i` (to object `#!pythonx`) with this command:

```q
lst[=;i;x]  / equivalent to lst[i]=x in Python
```

These expressions return PyKX objects, for instance:

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

#### Get methods

Given `#!python obj`, a PyKX object representing a Python object, you can access a method by using:

```q
obj`:method  / equivalent to obj.method in Python
```

When calling PyKX objects representing Python methods, the return of evaluation is a PyKX object. For example:

```q
q)np:.pykx.import`numpy
q)np`:arange
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist
q)arange:np`:arange                   / callable returning PyKX object
q)arange 12
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist
q)arange[12]`
0 1 2 3 4 5 6 7 8 9 10 11
```

#### PyKX function API

Use the function API to achieve the following:

- Call PyKX objects (to get PyKX objects).
- Declare PyKX objects callable (to get q or `#!python foreign` data).

The default return is a PyKX object. For q or foreign return type, you need to specify it. 

Given `#!python func`, a `#!python PyKX` object representing a callable Python function or method, you can carry out the following operations:

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

!!! info "How to chain operations?" 
    
    To chain together sequences of operations, return another PyKX object from a function or method call. Alternatively, call `.pykx.import`, `.pykx.get` and `.pykx.eval`.


#### PyKX examples

=== "Example #1"

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
=== "Example #2"

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
=== "Example #3"

    ```q
    q)stdout:.pykx.import[`sys]`:stdout.write
    q)stdout `$"hello\n";
    hello
    q)stderr:.pykx.import[`sys;`:stderr.write]
    q)stderr `$"goodbye\n";
    goodbye
    ```
=== "Example #4"

    ```q
    q)oarg:.pykx.eval"10"
    q)oarg`
    10
    q)ofunc:.pykx.eval["lambda x:2+x";<]
    q)ofunc[1]
    3
    q)ofunc oarg
    12
    q)p)def add2(x,y):return x+y
    q)add2:.pykx.get[`add2;<]
    q)add2[1;oarg]
    11
    ```

#### Function argument types

PyKX supports data type conversions between q and Python for Python native objects, NumPy objects, Pandas objects, PyArrow objects, and PyKX objects.

By default, when passing a q object to a callable function, it's converted to the most "natural" analogous type, as detailed below:

- PyKX/q generic list objects become Python lists.
- PyKX/q table/keyed table objects become Pandas equivalent DataFrames.
- All other PyKX/q objects become their analogous NumPy equivalent types.

!!! Warning

	Prior to PyKX 2.1.0, all conversions from q objects to Python would convert to their NumPy equivalent. To achieve this now, set the environment variable `PYKX_DEFAULT_CONVERSION="np"`

For function/method calls, control the default behavior of the conversions by setting `#!python .pykx.util.defaultConv`:

```q
q).pykx.util.defaultConv
"default"
```
You can apply one of the following values:

|**Python type**|Default|Python|NumPy|Pandas|PyArrow|PyKX|
|---------------|-------|------|-----|------|-------|----|
|**Value**:     |"default"|"py"|"np"|"pd"|"pa"|"k"|  


In the example below, we start with NumPy and update the default types across all function calls:

=== "NumPy"

    ```q
    q).pykx.typepy 1; 
    "<class 'numpy.int64'>"
    q).pykx.typepy til 10;
    "<class 'numpy.ndarray'>"
    q).pykx.typepy (10?1f;10?1f)
    "<class 'list'>"
    q).pykx.typepy ([]100?1f;100?1f);
    "<class 'pandas.core.frame.DataFrame'>"
    ```
=== "Python"

    ```q
    q).pykx.util.defaultConv:"py"
    q).pykx.typepy 1;
    "<class 'int'>"
    q).pykx.typepy til 10;
    "<class 'list'>"
    q).pykx.typepy ([]100?1f;100?1f);
    "<class 'dict'>"
    ```
=== "Pandas"

    ```q
    q).pykx.util.defaultConv:"pd"
    q).pykx.typepy 1;
    "<class 'numpy.int64'>"
    q).pykx.typepy til 10;
    "<class 'pandas.core.series.Series'>"
    q).pykx.typepy ([]100?1f;100?1f);
    "<class 'pandas.core.frame.DataFrame'>"
    ```
=== "PyArrow"

    ```q
    q).pykx.util.defaultConv:"pa"
    q).pykx.typepy 1;
    "<class 'numpy.int64'>"
    q).pykx.typepy til 10;
    "<class 'pyarrow.lib.Int64Array'>"
    q).pykx.typepy ([]100?1f;100?1f);
    "<class 'pyarrow.lib.Table'>"
    ```
=== "PyKX"

    ```q
    q).pykx.util.defaultConv:"k"
    q).pykx.typepy 1;
    "<class 'pykx.wrappers.LongAtom'>"
    q).pykx.typepy til 10;
    "<class 'pykx.wrappers.LongVector'>"
    q).pykx.typepy ([]100?1f;100?1f);
    "<class 'pykx.wrappers.Table'>"
    ```

Alternatively, to modify individual arguments to functions, use the `#!python .pykx.to*` functionality:

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

#### Set Python variables

You can set variables in Python `#!python __main__` by using `#!python .pykx.set`:

```q
q).pykx.set[`var1;42]
q).pykx.qeval"var1"
42
q).pykx.set[`var2;{x*2}]
q)qfunc:.pykx.get[`var2;<]
{[f;x].pykx.util.pykx[f;x]}[foreign]enlist
q)qfunc[3]
6
```

### Function calls

Python allows you to call functions with:

-   A variable number of arguments
-   A mixture of positional and keyword arguments
-   Implicit (default) arguments

This is available in the PyKX function-call interface, as detailed below:

-   Callable PyKX objects are variadic (they accept a variable number of arguments).
-   Default arguments are applied where no explicit arguments are given.
-   Individual keyword arguments are specified using the (infix) `#!python pykw` operator.
-   A list of positional arguments can be passed using `#!python pyarglist` (like Python \*args).
-   A dictionary of keyword arguments can be passed using `#!python pykwargs` (like Python \*\*kwargs).

!!! info "Keyword arguments last" 
    
    You can combine positional arguments, lists of positional arguments, keyword arguments, and a dictionary of keyword arguments. However, _all_ keyword arguments must always follow _any_ positional arguments. The dictionary of keyword arguments (if given) must be specified _last_.


#### Examples

```q
q)p)import numpy as np
q)p)def func(a=1,b=2,c=3,d=4):return np.array([a,b,c,d,a*b*c*d])
q)qfunc:.pykx.get[`func;<] / callable, returning q
```

Enter positional arguments directly. Function calling is variadic, so you can exclude later arguments:

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

Specify individual keyword arguments with the `#!python pykw` operator (applied infix). The order of keyword arguments doesn't matter.

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

To specify a list of positional arguments, use `#!python pyarglist` (similar to Python’s \*args).
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

You can specify a dictionary of keyword arguments by using `#!python pykwargs` (similar to Python’s \*\*kwargs).
If present, this argument must be the _last_ argument.

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

You can combine all four methods in a single function call if the order follows the above rules.

```q
q)qfunc[4;pyarglist enlist 3;`c pykw 2;pykwargs enlist[`d]!enlist 1]
4 3 2 1 24
```

!!! warning "`pykw`, `pykwargs`, and `pyarglist`"

    Before defining functions containing `pykw`, `pykwargs`, or `pyarglist` within a script, you must explicitly load the file `p.q`. Failure to do so results in errors.

#### Zero-argument calls

In Python these two calls are _not_ equivalent:

```python
func()       #call with no arguments
func(None)   #call with argument None
```

!!! warning "PyKX function called with `::` calls Python with no arguments"

    Although `::` in q corresponds to `None` in Python, if a PyKX function is called with `::` as its only argument, the corresponding Python function will be called with _no_ arguments.

To call a Python function with `#!python None` as its sole argument, retrieve `#!python None` as a foreign object in q and pass that as the argument:

```q
q)pynone:.pykx.eval"None"
q)pyfunc:.pykx.eval["print"]
q)pyfunc pynone;
None
```

**Python**     | **Form**                  | **q**
---------------|---------------------------|-----------------------
`func()`       | call with no arguments    | `func[]` or `func[::]`
`func(None)`   | call with argument `None` | `func[.pykx.eval"None"]`

!!! info "q functions applied to empty argument lists"

    The _rank_ (number of arguments) of a q function is determined by its _signature_,
    an optional list of arguments at the beginning of its definition.
    If the signature is omitted, the default arguments are as many of
    `x`, `y` and `z` as appear, and its rank is 1, 2, or 3.

    If it has no signature, and does not refer to `x`, `y`, or `z`, it has rank 1.
    It is implicitly unary.
    If it is then applied to an empty argument list, the value of `x` defaults to `(::)`.

    So `func[::]` is equivalent to `func[]` – and in Python to `func()`, not `func[None]`.

#### Print or return

`#!python .pykx.repr` returns the string representation of a Python object, either PyKX or foreign. You can print this representation to `#!python stdout` by using `#!python .pykx.print`. Here's how to use this function with a q object:

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

## Next steps

- Use the [pykx.q Library Reference Card](../pykx-under-q/api.md).
- [Upgrade from embedPy](../pykx-under-q/upgrade.md).

# Evaluating q code with PyKX

There are a number of ways to manipulate PyKX objects and evaluate q code in PyKX, for example:

1. By calling `pykx.q` directly, e.g. `pykx.q('10 {x,sum -2#x}/ 0 1')`
2. By dropping into the [interactive console][pykx.QConsole]
3. By making use of keyword functions provided by `q`, e.g. `pykx.q.til(10)`
4. Over [IPC][pykx.QConnection]

The first three methods evaluate the code locally within the Python process, and are not available without a q license. The final method evaluates the code in a separate q process, and can be used with or without a q license provided the server to which your PyKX instance is connected is appropriately licensed.

!!! Warning

    Functions pulled in over IPC are executed locally in PyKX, see the [IPC documentation](../../api/ipc.md)
    for more information on how to ensure `q` code is executed on the server and not locally.

## PyKX Objects

Calling a q instance or a connection to a q instance will return what is commonly referred to as a *PyKX object*. A PyKX object is an instance of the [`pykx.K`][pykx.K] class, or one of its subclasses. These classes are documented on the [PyKX wrappers API doc](../../api/pykx-q-data/wrappers.md) page.

PyKX objects are wrappers around objects in q's memory space within the Python process that PyKX (and your program that uses PyKX) runs in. These wrappers are cheap to make as they do not require copying any data out of q's memory space.

These PyKX objects support a variety of Python features (e.g. iteration, slicing, calling, etc.), and so oftentimes converting them to other types (e.g. a [`pykx.Vector`][pykx.Vector] to a `numpy.ndarray`) is unnecessary.

## Calling q using `pykx.q`

For users familiar with writing kdb+/q code use of the method `pykx.q` (or more commonly in this documentation `kx.q`) allows the evaluation of q code to take place providing the return of the function as a `PyKX` object. This method is variadic in nature and its usage comes in two forms:

1. Direct evaluation of single lines of code
2. The application of functions taking multiple arguments


### Direct evaluation of single lines of code

```python
>>> import pykx as kx
>>> kx.q('til 10')
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
>>> kx.q('.test.var:5?1f')
pykx.Identity(pykx.q('::'))
>>> kx.q('.test.var')
pykx.FloatVector(pykx.q('0.06165008 0.285799 0.6684724 0.9133033 0.1485357'))
```

### Application of functions taking multiple arguments

As noted above the `pykx.q` functionality is variadic in nature, in the case that the first argument is a function the N following arguments will be treated as arguments to that function. Of particular note is that these arguments can be Python or PyKX objects, all objects passed to a q function will be converted to a PyKX object using the method `pykx.toq` for example:

```python
>>> import pykx as kx
>>> import pandas as pd
>>> kx.q('{x+y}', 2, 4)
pykx.LongAtom(pykx.q('6'))
>>> kx.q('{[f;x]2*f x}', kx.q('{x+4}'), 3)   # Use a mixture of q/Python objects
pykx.LongAtom(pykx.q('14'))
>>> kx.q('{x,y}', kx.q('([]5?1f;5?1f)'), pd.DataFrame.from_dict({'x':[1.2, 1.3], 'x1': [2.0, 3.0]}))
pykx.Table(pykx.q('
x          x1       
--------------------
0.4269177  0.5339515
0.7704774  0.9387084
0.01594028 0.3027801
0.3573039  0.4448492
0.02547383 0.4414491
1.2        2        
1.3        3     
```

!!! Note

	The application of arguments to functions within PyKX is limited to a maximum of 8 arguments. This limitation is imposed by the evaluation of q code.

## Using the q console within PyKX

For users more comfortable prototyping q code within a q terminal it is possible within a Python terminal to run an emulation of a q session directly in Python through use of the `kx.q.console` method.

```python
>>> import pykx as kx
>>> kx.q.console()
q)til 10
0 1 2 3 4 5 6 7 8 9
q)\\
>>>
```

!!! Note

    This is not a fully featured q terminal, it has the same core [limitations](../advanced/limitations.md) that PyKX has when it comes to the running of timers and subscriptions.

## Using q keywords

Consider the following q function that checks if a given number is prime:

```q
{$[x in 2 3;1;x<2;0;{min x mod 2_til 1+floor sqrt x}x]}
```

We can evaluate it through `q` to obtain a [`pykx.Lambda`](../../api/pykx-q-data/wrappers.md) object. This object can then be called as a Python function:

```python
import pykx as kx

is_prime = kx.q('{$[x in 2 3;1;x<2;0;{min x mod 2_til 1+floor sqrt x}x]}')
assert is_prime(2)
assert is_prime(127)
assert not is_prime(128)
```

Arguments to the function are converted to [`pykx.K`][pykx.K] objects via the [`pykx.toq`][pykx.toq] module, and so the arguments can be anything supported by that module, i.e. any Python type `X` for which a `pykx.toq.from_X` function exists (barring some caveats - see the [`pykx.toq`][pykx.toq] documentation).

For instance, we can apply the `each` adverb to `is_prime` and then provide it a range of numbers to check like so:

```python
>>> is_prime.each(range(10))
pykx.LongVector(q('0 0 1 1 0 1 0 1 0 0'))
```

Then we could pass that into [`pykx.q.where`](../../api/pykx-execution/q.md#where)

```python
>>> kx.q.where(is_prime.each(range(10)))
pykx.LongVector(pykx.q('2 3 5 7'))
```

Context is persisted between embedded calls to q, but not calls over IPC.

```python
>>> kx.q('\d .abc') # change to the `.abc` context
pykx.Identity(pykx.q('::'))
>>> kx.q('xyz: 1 2 3') # set variable `xyz` within the `.abc` context
pykx.Identity(pykx.q('::'))
>>> kx.q('.abc.xyz')
pykx.LongVector(pykx.q('1 2 3'))
>>> kx.q('\d .') # change back to the default `.` global context
pykx.Identity(pykx.q('::'))
>>> kx.q('xyz: 4 5 6') # set variable `xyz` within the `.` global context
pykx.Identity(pykx.q('::'))
>>> kx.q('.abc.xyz')
pykx.LongVector(pykx.q('1 2 3'))
>>> kx.q('xyz')
pykx.LongVector(pykx.q('4 5 6'))
>>> q = kx.QConnection('localhost', 5001)
>>> q('\d .abc')
pykx.Identity(pykx.q('::'))
>>> q('xyz: 1 2 3')
pykx.Identity(pykx.q('::'))
>>> q('.abc.xyz')
pykx.exceptions.QError: .abc.xyz # `xyz` was not set within the `.abc` context.
>>> q('xyz')
pykx.LongVector(pykx.q('1 2 3')) # It got set within the global context
```

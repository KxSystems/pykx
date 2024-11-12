---
title: Use PyKX objects
description: How to use PyKX objects and evaluate q code with PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, PyKX objects, 
---

# Use PyKX objects and evaluate q code with PyKX

_This page provides details on how to use PyKX objects and how to evaluate q code with PyKX._

!!! tip "Tip: For the best experience, we recommend reading [PyKX objects and attributes](..//../learn/objects.md) and [Create and convert PyKX objects](creating.md) first." 

There are four ways to manipulate PyKX objects and evaluate q code in PyKX:

- a. By calling `#!python pykx.q` directly, for example, `#!python pykx.q('10 {x,sum -2#x}/ 0 1')`
- b. By dropping into the [interactive console][pykx.QConsole]
- c. By using `#!python q` keyword functions, for example, `#!python pykx.q.til(10)`
- d. Over [IPC][pykx.QConnection]

The first three methods evaluate the code locally within the Python process and require a q license. The final method evaluates the code in a separate q process and can be used with or without a q license, provided the server your PyKX instance is connected to is appropriately licensed.

!!! Warning

    Functions pulled in over IPC are executed locally in PyKX. Go to the [IPC documentation](../../api/ipc.md)
    for more information on how to ensure the `q` code is executed on the server and not locally.

## a. Call q using `#!python pykx.q`

For users familiar with kdb+/q code, the `#!python pykx.q` (or `#!python kx.q`) method allows the evaluation of q code to take place providing the return of the function as a `#!python PyKX` object. This method is variadic, meaning it can accept a variable number of arguments. You can use in two different ways:”

1. Direct evaluation of single lines of code
2. Application of functions that take multiple arguments


### a.1 Direct evaluation of single lines of code

```python
>>> import pykx as kx
>>> kx.q('til 10')
pykx.LongVector(pykx.q('0 1 2 3 4 5 6 7 8 9'))
>>> kx.q('.test.var:5?1f')
pykx.Identity(pykx.q('::'))
>>> kx.q('.test.var')
pykx.FloatVector(pykx.q('0.06165008 0.285799 0.6684724 0.9133033 0.1485357'))
```

### a.2 Application of functions taking multiple arguments

If the first argument of `#!python pykx.q` is a function, the `#!python N` following arguments are treated as arguments to that function. Arguments can be Python or PyKX objects. All objects passed to a q function are converted to a PyKX object using the method `#!python pykx.toq`. For example:

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

Users wishing to debug failed evaluation of q code can do so, either by globally setting the environment variable `#!python PYKX_QDEBUG` or through a `#!python debug` keyword:

=== "Global Setting"

	```python
	>>> import os
	>>> os.environ['PYKX_QDEBUG'] = 'True'
	>>> import pykx as kx
	>>> kx.q('{x+1}', 'a')
	backtrace:
	  [2]  {x+1}
        	 ^
	  [1]  (.Q.trp)

	  [0]  {[pykxquery] .Q.trp[value; pykxquery; {if[y~();:(::)];2@"backtrace:
        	            ^
	",.Q.sbt y;'x}]}
	Traceback (most recent call last):
	  File "<stdin>", line 1, in <module>
	  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/embedded_q.py", line 230, in __call__
	    return factory(result, False)
	  File "pykx/_wrappers.pyx", line 493, in pykx._wrappers._factory
	  File "pykx/_wrappers.pyx", line 486, in pykx._wrappers.factory
	pykx.exceptions.QError: type
	```

=== "Per call debugging"

	```python
	>>> import pykx as kx
	>>> kx.q('{x+1}', 'a', debug=True)
	backtrace:
	  [2]  {x+1}
	         ^
	  [1]  (.Q.trp)

	  [0]  {[pykxquery] .Q.trp[value; pykxquery; {if[y~();:(::)];2@"backtrace:
	                    ^
	",.Q.sbt y;'x}]}
	Traceback (most recent call last):
	  File "<stdin>", line 1, in <module>
	  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/embedded_q.py", line 230, in __call__
	    return factory(result, False)
	  File "pykx/_wrappers.pyx", line 493, in pykx._wrappers._factory
	  File "pykx/_wrappers.pyx", line 486, in pykx._wrappers.factory
	pykx.exceptions.QError: type
	```

## b. Use the q console within PyKX

For users more comfortable prototyping q code within a q terminal, it's possible within a Python terminal to run an emulation of a q session directly in Python through the `#!python kx.q.console` method:

```python
>>> import pykx as kx
>>> kx.q.console()
q)til 10
0 1 2 3 4 5 6 7 8 9
q)\\
>>>
```

!!! Note

    This is not a fully-featured q terminal. It shares the same core [limitations](../../help/issues.md) as PyKX, particularly regarding the running of timers and subscriptions.

## c. Use q keywords

Consider the following q function that checks if a given number is prime:

```q
{$[x in 2 3;1;x<2;0;{min x mod 2_til 1+floor sqrt x}x]}
```

You can evaluate it through `#!python q` to obtain a [`#!python pykx.Lambda`](../../api/pykx-q-data/wrappers.md) object. You can then call this object as a Python function:

```python
import pykx as kx

is_prime = kx.q('{$[x in 2 3;1;x<2;0;{min x mod 2_til 1+floor sqrt x}x]}')
assert is_prime(2)
assert is_prime(127)
assert not is_prime(128)
```

To convert arguments to the function to [`#!python pykx.K`][pykx.K] objects, use the [`#!python pykx.toq`][pykx.toq] module. The arguments can be anything supported by that module, for example, any Python type `#!python X` for which a `#!python ykx.toq.from_X` function exists (barring some caveats mentioned in the [`#!python pykx.toq`][pykx.toq] documentation).

For instance, you can apply the `#!python each` adverb to `#!python is_prime` and then provide it a range of numbers to check:

```python
>>> is_prime.each(range(10))
pykx.LongVector(q('0 0 1 1 0 1 0 1 0 0'))
```

Then you could pass that into [`pykx.q.where`](../../api/pykx-execution/q.md#where):

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

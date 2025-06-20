---
title:  Import existing q functions
description: How to use q functions in a Pythonic way in PyKX
date: October 2024
author: KX Systems, Inc.,
tags: interface, q, PyKX
---

# Import existing q functions

For you and many users of PyKX the q programming language may not be your primary language of choice when developing analytics and applications. There are a number of circumstances under which access to q functionality or the ability to use functions written in q may be critical to your use-case:

- When dealing with large volumes of kdb+ data or operations where vector analytic performance is of paramount importance.
- When you wish to make use of existing q code/libraries in a Python first way.
- When you need access to functions of the q language directly.

The below sections make use of what is known as "The Context Interface". In q, a context (known as a namespace when at the top level) is an organizational structure which is used to organize code into libraries/common utilities. For more information on contexts/namespaces in q please refer to [Chapter 12 of Q for Mortals](https://code.kx.com/q4m3/12_Workspace_Organization/). PyKX exposes these contexts as special [`kx.QContext`](../../api/pykx-execution/ctx.md#pykx.ctx.QContext) objects. These context objects have attributes for their members, which can either be sub-contexts or K objects. For example:

* `#!python pykx.q.Q` is a KdbContext instance for the builtin `#!q .Q` context/namespace
* `#!python pykx.q.ctxA.ctxB` is a KdbContext instance for the `#!q .ctxA.ctxB` context
* `#!python pykx.q.ctxA.ctxB.kObject` is a pykx.K instance for the `#!q .ctxA.ctxB.kObject` K object

## Use the in-built q functionality

When you start a q process there are 4 namespaces loaded which provide useful functionality to users of PyKX.

| **Namespace** | **Contents**                                                    | **Link**                                |
| :------------ | :-------------------------------------------------------------- | :-------------------------------------- |
| `#!q .q`      | Fundamental keywords and operators of the q language.           | [link](https://code.kx.com/q/ref/)      |
| `#!q .Q`      | Miscellaneous tooling for database interactions, debugging etc. | [link](https://code.kx.com/q/ref/dotq/) |
| `#!q .z`      | Environment and callback configuration functionality            | [link](https://code.kx.com/q/ref/dotz/) |
| `#!q .j`      | Functionality for the serialization/deserialization of json     | [link](https://code.kx.com/q/ref/dotj/) |

These namespaces are loaded by default and accessible as follows 

```python
>>> kx.q.max    # Access the max native function
>>> kx.q.Q.qt   # Access the .Q.qt function
>>> kx.q.z.p    # Access the .z.p function
>>> kx.q.j.k    # Access the .j.k function
```

As can be seen above, just as in q, the .q context is accessible at the top-level, so for instance instead of accessing pykx.q.q.md5, you can access it as pykx.q.md5.

!!! Note

	Some q builtins cannot be accessed like this such as or and not as these result in Python syntax errors. Such functions can instead be accessed either with getattr, or by evaluating them as q code (e.g. pykx.q('not')).

## Using executed q code Python first

Much of the code you write or need to access will come from executed code locally in your process or will be contained in scripts which you have access to. The flow chart below shows the hierarchy of search/loading that happens when PyKX cannot find a requested context.

```mermaid
graph LR
  A([`pykx.q.ctxA`<br>is accessed.]);
  B{Is `.ctxA` <br>defined in<br>memory in q?};
  C{Is `.ctxA` <br>defined in<br>memory in q?};
  D["Search for a<br>matching script<br>(details below)."];
  E{Has a<br>matching script<br>been found?};
  F([`AttributeError`<br>is raised.]);
  G([`QContext`<br>is returned.]);
  H["Switch to the<br>context `.ctxA`<br>(`system &quot;d .ctxA&quot;`),
    <br>execute the script,<br>then switch back."];

  A --> B;
  B --No--> D;
  B --Yes--> G;
  C --No--> F;
  C --Yes--> G;
  E --No--> F;
  E --Yes--> H;
  D --> E;
  H --> C;
```

As described in the flow chart if a context is found to exist within the `q` memory space this will be presented to the user as the `kx.QContext`. Take for example the below case where you have defined two q functions and wish to access them in a Python first manner:

```python
>>> import pykx as kx
>>> kx.q('.test.function0:{x+y}')
>>> kx.q('.test.function1:{x-y}')
>>> kx.q.test
<pykx.ctx.QContext of .test with [function0, function1]>
```

If the namespace/context you are requesting doesn't exist in the `q` memory space then a search is carried out for a script matching the supplied context which if found is executed. The search logic is outlined in the expandable section below.

??? Note "File Search Path"

	When the context interface cannot find a namespace (i.e. a top-level context) that is being accessed it attempts to find a q/k script that has a matching name. This process is done via a depth first search of a tree where each node corresponds to part of the path, and each leaf corresponds to a possible file. Only the first file found that exists is executed. If none of the files exist then an `AttributeError` is raised.

	The layers of the tree are as follows:

	- Each of the paths in `pykx.q.paths`/`pykx.ipc.Connection(...).paths` (which defaults to `pykx.ctx.default_paths`)
	- `.` prefix or not
	- The name of the attribute accessed (i.e. `pykx.q.script` -> `script`)
	- `.q` or `.k`
	- No trailing `_` or a trailing `_` ([n.b. why a q/k script path would end with an underscore](https://code.kx.com/q/basics/syscmds/#_-hide-q-code))

	So for example if `pykx.q.script` was accessed, the context `.script` was not defined in memory in q, and `paths` was set to `['.', pykx.qhome]` (where `pykx.qhome == pathlib.Path('/opt/kdb')`), then the following paths would be checked in order until one is found to exist, or they have all been checked:

	1.  `./.script.q`
	2.  `./.script.q_`
	3.  `./.script.k`
	4.  `./.script.k_`
	5.  `./script.q`
	6.  `./script.q_`
	7.  `./script.k`
	8.  `./script.k_`
	9.  `/opt/kdb/.script.q`
	10. `/opt/kdb/.script.q_`
	11. `/opt/kdb/.script.k`
	12. `/opt/kdb/.script.k_`
	13. `/opt/kdb/script.q`
	14. `/opt/kdb/script.q_`
	15. `/opt/kdb/script.k`
	16. `/opt/kdb/script.k_`

To show the script search logic in action you can first write a file to the `#!python kx.qhome` location used by PyKX containing a namespace matching the name of the script

```python
>>> demo_extension_source = '''
... \d .demo_extension
... N:100
... test_data:([]N?`a`b`c;N?1f;N?10;N?0b)
... test_function:{[data]
...   analytic_keys :`max_x1`avg_x2`med_x3;
...   analytic_calcs:(
...     (max;`x1);
...     (avg;`x2);
...     (med;`x3));
...   ?[data;
...     ();
..     k!k:enlist `x;
...     analytic_keys!analytic_calcs
...     ]
...   }
.. '''
>>>
>>> demo_extension_filename = kx.qhome/'demo_extension.q'
>>> with open(demo_extension_filename, 'w') as f:
...     f.write(demo_extension_source)
```

Now that your script is available as a file `demo_extension.q` you can access and use the functions as follows:

```python
>>> kx.q.demo_extension
<pykx.ctx.QContext of .demo_extension with [N, test_data, test_function]>
>>> kx.q.demo_extension.test_data
pykx.Table(pykx.q('
x x1         x2 x3
------------------
c 0.2086614  2  0
a 0.9907116  1  1
a 0.5794801  8  1
b 0.9029713  8  0
a 0.2011578  1  0
..
'))
>>> kx.q.demo_extension.test_function
pykx.SymbolicFunction(pykx.q('`.demo_extension.test_function'))
>>> kx.q.demo_extension.test_function(kx.q.demo_extension.test_data)
pykx.KeyedTable(pykx.q('
x| max_x1    avg_x2   med_x3
-| -------------------------
a| 0.9907116 4.74359  1
b| 0.9550901 4.580645 1
c| 0.9830794 4.433333 0
'))
```

## Extend where PyKX searches for scripts

In addition to the default search locations you can add additional locations to be searched through appending of additional search paths to the `kx.q.paths` list which is used in the search.

The following shows a practical example of this, accessing a file `my_context.q` at a new location `/tmp/files`. In this example you can see the behavior if attempting to access a namespace without this location set for search:

```python
>>> import pykx as kx
>>> from pathlib import Path
>>> kx.q.my_context
Traceback (most recent call last):
  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/__init__.py", line 132, in __getattr__
..
>>> kx.q.paths.append(Path('/tmp/files'))
>>> kx.q.my_context
<pykx.ctx.QContext of .my_context with [func]>
```

If PyKX fails to find a script an `#!python AttributeError` will be raised, the expanding section below provides an example of this

??? Note "Failed to find a script"

	```python
	>>> kx.q.test_extension
	Traceback (most recent call last):
	  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/__init__.py", line 118, in __getattr__
	    self.__getattribute__('_register')(name=key)
	  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/__init__.py", line 189, in _register
	    path = _first_resolved_path([''.join(x) for x in it.product(
	  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/__init__.py", line 47, in _first_resolved_path
	    raise FileNotFoundError(f'Could not find any of the following files:\n{unfound_paths}')
	FileNotFoundError: Could not find any of the following files:
	.test_extension.q
	.test_extension.q_
	/usr/local/anaconda3/lib/python3.8/site-packages/pykx/lib/test_extension.k
	/usr/local/anaconda3/lib/python3.8/site-packages/pykx/lib/test_extension.k_

	The above exception was the direct cause of the following exception:

	Traceback (most recent call last):
	  File "<stdin>", line 1, in <module>
	  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/__init__.py", line 122, in __getattr__
	    raise attribute_error from inner_error
	  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/__init__.py", line 115, in __getattr__
	    return ctx.__getattr__(key)
	  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/ctx.py", line 267, in __getattr__
	    raise AttributeError(
	AttributeError: 'pykx.ctx.QContext' object has no attribute 'test_extension'
	```

## Use functions retrieved with the Context Interface

Functions returned by the context interface are provided as [`pykx.SymbolicFunction`][pykx.SymbolicFunction] instances.

These objects are symbol atoms whose symbol is a named function (with a fully-qualified name). They can be called like regular [`pykx.Function`][pykx.Function] objects, but unlike regular [`pykx.Function`][pykx.Function] objects, they will execute in the `pykx.Q` instance (also known as its "execution context") in which it was defined.

The following shows an example of the retrieval of a function from a context vs defining the function itself:

* Function retrieved via a context

	```python
	>>> kx.q.extension.func
	pykx.SymbolicFunction(pykx.q('`.extension.func'))
	>>> kx.q.extension.func(2)
	pykx.LongAtom(pykx.q('3'))
	```

* Function defined locally and retrieved

	```python
	>>> qfunc = kx.q('{x+1}')
	>>> qfunc
	pykx.Lambda(pykx.q('{x+1}'))
	>>> qfunc(2)
	pykx.LongAtom(pykx.q('3'))
	```

## Use Contexts via IPC

The context interface is also supported against remote processes thus allowing you to run analytic operations Python first against a remote kdb+/q server. The syntax and operational restrictions outlined in the previous sections also exist for the IPC instance which you can call as follows

```python
>>> with kx.SyncQConnection(port=5050) as conn:
...     print(conn.max([1, 2, 3, 4]))
4
>>> with kx.SyncQConnection(port=5050) as conn:
...     conn('.test.func:{x+1}')
...     print(conn.test.func(10))
11
```

!!! Warning "Performance Impact"

	The context interface adds overhead to remote queries as it requires the retrieval of information about the members of namespaces prior to execution, optimal use of IPC connections should limit access to this functionality by setting `#!python no_ctx=True`

## Best practice for organize scripts

For efficient automatic script loading, each q/k script should only define at most one public context. The name of the context should be equivalent to the name of the file without the file extension. For instance, `script.q` should place its definitions within the `#!q .script` namespace. This ensures the context will be defined and ensures that when the context interface executes a script to load a context, it doesn't load in more contexts than intended.

When these best practices cannot be followed it may be impossible to use the automatic loading of scripts via the context interface. In that case we can resort to manually loading scripts either by executing the q code `#!q system "l <path to script>"`, or by calling `#!python pykx.q._register` with the path to the script.

When switching contexts within a script, one should always save the context they were in prior to their context switch, and then switch back into it afterwards, rather than explicitly switching into the global context.

## Next Steps

- [Learn how to interact via IPC](ipc.md)
- [Query a database using Python](../fundamentals/query/pyquery.md)

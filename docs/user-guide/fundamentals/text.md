---
title: Convert text in PyKX 
description: How to convert text in PyKX
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, text, 
---

# Convert text in PyKX

_This page provides details on how to represent, handle, and convert text in PyKX._

In PyKX, text can be represented in various ways. Here are the basic building blocks for handling text within the library:

| **Type**            | **Description**                                                                                  | **Example Generation**   |
|---------------------|--------------------------------------------------------------------------------------------------|------------------------------|
| `pykx.SymbolAtom`   | A symbol atom in PyKX is an irreducible atomic entity storing an arbitrary number of characters. | ```pykx.q('`test')```        |
| `pykx.SymbolVector` | A symbol vector is a collected list of symbol atoms.                                             | ```pykx.q('`test`vector')``` |
| `pykx.CharAtom`     | A char atom holds a single ASCII or 8-but unicode character stored as 1 byte.                    | `pykx.q('"a"')`              |
| `pykx.CharVector`   | A char vector is a collected list of char vectors.                                               | `pykx.q('"test"')`           |

!!! info "Head to our [Text data](https://code.kx.com/q4m3/2_Basic_Data_Types_Atoms/#24-text-data) section for a deeper dive into the underlying text representation."

## Convert text to/from PyKX

To convert Pythonic text data to PyKX objects, use the `#!python pykx.SymbolAtom` and `#!python pykx.CharVector` functions as shown below:

```python
>>> import pykx as kx
>>> pystring = 'test string'
>>> kx.SymbolAtom(pystring)
pykx.SymbolAtom(pykx.q('`test string'))
>>> kx.CharVector(pystring)
pykx.CharVector(pykx.q('"test string"'))
```

Alternatively, you use the automatic conversion function `#!python pykx.toq` which takes an incoming Python type and converts it to its analogous PyKX type. The following table shows the mapping between the two types:

| **Python Type**| **PyKX Type**                  |
|-------------|-----------------------------------|
| `str`       | `pykx.SymbolAtom`                 |
| `byte`      | `pykx.CharAtom`/`pykx.CharVector` |

```python
>>> import pykx as kx
>>> kx.toq('string')
pykx.SymbolAtom(pykx.q('`string'))
>>> kx.toq(b'bytes')
pykx.CharVector(pykx.q('"bytes"'))
>>> kx.toq(b'a')
pykx.CharAtom(pykx.q('"a"'))
```

When using the `#!python pykx.toq` function, you can specify the target type for your data as shown below. This can be useful when selectively converting data:

```python
>>> import pykx as kx
>>> kx.toq('string', kx.CharVector)
pykx.CharVector(pykx.q('"string"'))
>>> kx.toq(b'bytes', kx.SymbolAtom)
pykx.SymbolAtom(pykx.q('`bytes'))
```

The `#!python pykx.toq` conversion is used by default when passing Python data to PyKX functions, for example:

```python
>>> import pykx as kx
>>> kx.q('{(x;y)}', 'string', b'bytes')
pykx.List(pykx.q('
`string
"bytes"
'))
```

The `strings_as_chars` parameter can be set to `True` to force the conversion of strings to `CharVectors` instead of `SymbolAtoms`:

```python
>>> kx.toq('test', strings_as_char=False)
pykx.SymbolAtom(pykx.q('`test'))
>>> kx.toq('test', strings_as_char=True)
pykx.CharVector(pykx.q('"test"'))
```

## PyKX Under q

For more information on executing Python code in a q process see [evaluate and execute python](../../pykx-under-q/intro.html#evaluate-and-execute-python)

Using text conversion under q we can convert PyKX text objects into q. This function call converts the Python `str` into a `SymbolAtom`

```q
q)\l pykx.q
q)s:.pykx.eval["'testtest'"]
q).pykx.toq s
`testtest
q)type .pykx.toq s
-11h
```

Calling `.pykx.toq` on this `tuple` returns a `SymbolVector`

```q
q)\l pykx.q
q)s:.pykx.eval["('test1', 'test2')"]
q).pykx.toq s
`testtest`test2
q)type .pykx.toq s
11h
```

If you explicitly want to convert a `str` into a `CharVector` you can use the function `.pykx.toq[;1b]`

```q
q)\l pykx.q
q)s:.pykx.eval["'testtest'"]
q).pykx.toq0[;1b] s
"testtest"
q)type .pykx.toq0[;1b] s
10h
```

Calling `.pykx.toq0[;1b]` on the `tuple` below returns a list of `CharVectors`

```q
q)\l pykx.q
q)s:.pykx.eval["('test1', 'test2')"]
q).pykx.toq0[;1b] s
"testtest"
"test2"
q)type .pykx.toq0[;1b] s
0h
q)x:.pykx.toq0[;1b] s;type x[1]
10h
```

When using `.pykx.qeval` for text conversions the default `.pykx.toq` logic is applied

```q
q).pykx.qeval"'testtest'"
q)`testtest
```

A backtick `` ` ``  can be used to convert a PyKX object to q. This uses the same underlying logic as .pykx.toq:

```q
q)s:.pykx.eval["('test1', 'test2')"]
q)s`
`test1`test2
```

For more detail on text conversion under q see our page on [.pykx.toq0](../../pykx-under-q/api.md#pykxtoq0).

## Differences between `Symbol` and `Char` data objects

While there may appear to be limited differences between `#!python Symbol` and `#!python Char` representations of objects, the choice of underlying representation can have an impact on the performance and memory profile of many applications of PyKX. This section will describe a number of these differences and their impact in various scenarios.

Although `#!python Symbol` and `#!python Char` representations of objects might seem similar, the choice between them can significantly affect the performance and memory usage of many PyKX applications. This section exploreS the impact of these differences in various scenarios.


### Text access and mutability

The individual characters which comprise a `#!python pykx.SymbolAtom` object are not directly accessible by a user; this limitation does not exist for `#!python pykx.CharVector` objects. For example, it's possible to retrieve slices of a `#!python pykx.CharVector`:

```python
>>> import pykx as kx
>>> charVector = kx.CharVector('test')
>>> charVector
pykx.CharVector(pykx.q('"test"'))
>>> charVector[1:]
pykx.CharVector(pykx.q('"est"'))
>>> symbolAtom = kx.SymbolAtom('test')
>>> symbolAtom
pykx.SymbolAtom(pykx.q('`test'))
>>> symbolAtom[1:]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'SymbolAtom' object is not subscriptable
```

Similarly `#!python pykx.CharVector` type objects are mutable while `#!python pykx.SymbolAtom` type objects are not:

```python
>>> import pykx as kx
>>> charVector = kx.CharVector('test')
>>> kx.q('{x[0]:"r";x}', charVector)
pykx.CharVector(pykx.q('"rest"'))
```

### Memory considerations

When dealing with Symbol type objects, note that they are never deallocated once generated. You can notice this through growth of the `#!python syms` key of `#!python kx.q.Q.w` as follows:

```python
>>> kx.q.Q.w()['syms']
pykx.LongAtom(pykx.q('2790'))
>>> kx.SymbolAtom('test')
pykx.SymbolAtom(pykx.q('`test'))
>>> kx.q.Q.w()['syms']
pykx.LongAtom(pykx.q('2791'))
>>> kx.SymbolAtom('testing')
pykx.SymbolAtom(pykx.q('`testing'))
>>> kx.q.Q.w()['syms']
pykx.LongAtom(pykx.q('2792'))
```

This is important as overuse of symbols can result in increased memory requirements for your processes. Symbols as such are best used when dealing with highly repetitive text data.

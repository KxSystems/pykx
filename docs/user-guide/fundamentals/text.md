# Text representation in PyKX 

Within PyKX text can be represented in a number of ways that you will encounter when using the library. The following are the basic building blocks for text within PyKX, a deeper dive into the underlying text representation can be found [here](https://code.kx.com/q4m3/2_Basic_Data_Types_Atoms/#24-text-data):

| Type                | Description                                                                                      | Example Generation           |
|---------------------|--------------------------------------------------------------------------------------------------|------------------------------|
| `pykx.SymbolAtom`   | A symbol atom in PyKX is an irreducible atomic entity storing an arbitrary number of characters. | ```pykx.q('`test')```        |
| `pykx.SymbolVector` | A symbol vector is a collected list of symbol atoms.                                             | ```pykx.q('`test`vector')``` |
| `pykx.CharAtom`     | A char atom holds a single ASCII or 8-but unicode character stored as 1 byte.                    | `pykx.q('"a"')`              |
| `pykx.CharVector`   | A char vector is a collected list of char vectors                                                | `pykx.q('"test"')`           |

## Converting text to/from PyKX

Pythonic text data can be converted to PyKX objects directly through use of the `pykx.SymbolAtom` and `pykx.CharVector` functions as shown below

```python
>>> import pykx as kx
>>> pystring = 'test string'
>>> kx.SymbolAtom(pystring)
pykx.SymbolAtom(pykx.q('`test string'))
>>> kx.CharVector(pystring)
pykx.CharVector(pykx.q('"test string"'))
```

Alternatively you can make use of the automatic conversion function `pykx.toq` which will take an incoming Python type and convert it to its analagous PyKX type. The following table shows the mapping which is used

| Python Type | PyKX Type                         |
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

When using the `pykx.toq` function it is possible to specify the target type for your data as shown below, this can be useful when selectively converting data

```python
>>> import pykx as kx
>>> kx.toq('string', kx.CharVector)
pykx.CharVector(pykx.q('"string"'))
>>> kx.toq(b'bytes', kx.SymbolAtom)
pykx.SymbolAtom(pykx.q('`bytes'))
```

An important note on the above when using PyKX functions is that the `pykx.toq` conversion will be used by default when passing Python data to these functions, for example:

```python
>>> import pykx as kx
>>> kx.q('{(x;y)}', 'string', b'bytes')
pykx.List(pykx.q('
`string
"bytes"
'))
```

## Differences between `Symbol` and `Char` data objects

While there may appear to be limited differences between `Symbol` and `Char` representations of objects, the choice of underlying representation can have an impact on the performance and memory profile of many applications of PyKX. This section will describe a number of these differences and their impact in various scenarios.

### Text access and mutability

The individual characters which comprise a `pykx.SymbolAtom` object are not directly accessible by a user, this limitation does not exist for `pykx.CharVector` objects. For example it is possible to retrieve slices of a `pykx.CharVector`

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

Similarly `pykx.CharVector` type objects are mutable while `pykx.SymbolAtom` type objects are not

```python
>>> import pykx as kx
>>> charVector = kx.CharVector('test')
>>> kx.q('{x[0]:"r";x}', charVector)
pykx.CharVector(pykx.q('"rest"'))
```

### Memory considerations

An important point of note when dealing with Symbol type objects is that these are never deallocated once generated, this can be seen through growth of the `syms` key of `kx.q.Q.w` as follows

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

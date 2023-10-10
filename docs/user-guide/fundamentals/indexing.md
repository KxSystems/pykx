# Indexing PyKX Objects

## An introduction to indexing within PyKX

Indexing in q works differently than you may be used to, and that behavior largely carries over into PyKX for indexing K objects. For more information about how indexing in q works (and by extension, how indexing K objects in PyKX work), refer to the following sections of the q tutorial book [Q For Mortals](https://code.kx.com/q4m3/):

- [Indexing](https://code.kx.com/q4m3/3_Lists/#34-indexing)
- [Iterated Indexing and Indexing at Depth](https://code.kx.com/q4m3/3_Lists/#38-iterated-indexing-and-indexing-at-depth)
- [Indexing with Lists](https://code.kx.com/q4m3/3_Lists/#39-indexing-with-lists)
- [Elided Indices](https://code.kx.com/q4m3/3_Lists/#310-elided-indices)

Indexes used on K objects in PyKX are converted to equivalent K objects in q using the [toq module](../../api/pykx-q-data/toq.md), just like any other Python to q conversion. To guarantee that the index used against a K object is what you intend it to be, you may perform the conversion of the index yourself before applying it. When K objects are used as the index for another K object, the index object is applied to the [`pykx.Collection`][pykx.Collection] object as they would be in q; i.e. as described in Q For Mortals.

The following provides some examples of applying indexing to various q objects:

## Basic Array Indexing

Indexing in PyKX spans elements `0` to element `N-1` where `N` is the length of the object being indexed. 

### Single element indexing

Single element indexing works similarly to any other standard Python sequence. Similar to Numpy PyKX supports negative indices to allow retrieval of indexes at the end of an array. For example:

```python
>>> x = kx.q.til(10)
>>> x[2]
pykx.LongAtom(pykx.q('2'))
>>> x[-2]
pykx.LongAtom(pykx.q('8'))
```

Similar to Numpy indexing an array out of bounds will result in an `IndexError` being raised.

```python
>>> x = kx.q.til(5)
>>> x[6]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/wrappers.py", line 1165, in __getitem__
    return q('@', self, _idx_to_k(key, _wrappers.k_n(self)))
  File "/usr/local/anaconda3/lib/python3.8/site-packages/pykx/wrappers.py", line 212, in _idx_to_k
    raise IndexError('index out of range')
IndexError: index out of range
```

N Dimensional list arrays can also be manipulated using single element indexing as follows

```python
>>> x = kx.q('4 4#16?1f')
>>> x
pykx.List(pykx.q('
0.5294808 0.6916099 0.2296615 0.6919531 
0.4707883 0.6346716 0.9672398 0.2306385 
0.949975  0.439081  0.5759051 0.5919004 
0.8481567 0.389056  0.391543  0.08123546
'))
>>> x[0][3]
pykx.FloatAtom(pykx.q('0.6919531'))
```

### Slicing

Slicing arrays in PyKX is more simplistic than the functionality provided by Numpy. Arrays of N dimensions are indexed using `obj[start:stop:step]` semantics. This slice syntax operates where `start` is the starting index, `stop` is the stopping index and `step` is the number of steps between the elements where `step` is non zero

```python
>>> x = kx.q.til(10)
>>> x[2:4]
pykx.LongVector(pykx.q('2 3'))
>>> x[5:]
pykx.LongVector(pykx.q('5 6 7 8 9'))
>>> x[:8:2]
pykx.LongVector(pykx.q('0 2 4 6'))

>>> x = kx.q('4 4#16?1f')
>>> x[:2]
pykx.List(pykx.q('
0.1477547 0.274227  0.5635053 0.883823 
0.2439194 0.6718125 0.8639591 0.8439807
'))
```

## Indexing Non Array Objects

In addition to being able to index and slice PyKX vector and list objects it is also possible to apply index and slicing semantics on PyKX Table objects. Application of slice/index semantics on tabular objects will return table like objects

```python
>>> import pandas as pd
>>> from random import random
>>> from uuid import uuid4
>>> df = pd.DataFrame.from_dict({
    'x':  [random() for _ in range(10)],
    'x1': [random() for _ in range(10)],
    'x2': [uuid4() for _ in range(10)]
})
>>> tab = kx.toq(df)
>>> tab
pykx.Table(pykx.q('
x           x1         x2                                  
-----------------------------------------------------------
0.1872634   0.4176994  c9555cdf-57db-28a8-bf6c-67f6ee711a5f
0.8416288   0.01920741 3ecca92c-aae6-f796-38c9-80f4da70a89d
0.7250709   0.8761714  6417d4b3-3fc6-e35a-1c34-8c5c3327b1e8
0.481804    0.7575856  4040cd34-c49e-587b-e546-e1342bf1dd85
0.9351307   0.6030223  e8327955-bd9a-246a-0b17-fbf8f05fd28a
0.7093398   0.1811364  54a1959c-997c-6c57-1ff2-a0f3e845f01d
0.9452199   0.2329662  008bded0-3383-f19b-1d18-abfb199b1ac1
0.7092423   0.250046   6f54a161-49e7-f707-0054-626b867fb02f
0.002184472 0.0737272  f294c3cb-a6da-e15d-c8e0-3a848d2abf10
0.06670537  0.3186642  cd17ee98-c089-10a3-8992-d437a566f081
'))
>>> tab[3]
x           x1         x2
-----------------------------------------------------------
0.481804    0.7575856  4040cd34-c49e-587b-e546-e1342bf1dd85
'))
>>> tab[:5]
pykx.Table(pykx.q('
x         x1         x2                                  
---------------------------------------------------------
0.1872634 0.4176994  c9555cdf-57db-28a8-bf6c-67f6ee711a5f
0.8416288 0.01920741 3ecca92c-aae6-f796-38c9-80f4da70a89d
0.7250709 0.8761714  6417d4b3-3fc6-e35a-1c34-8c5c3327b1e8
0.481804  0.7575856  4040cd34-c49e-587b-e546-e1342bf1dd85
0.9351307 0.6030223  e8327955-bd9a-246a-0b17-fbf8f05fd28a
'))
>>> tab[0:8:2]
pykx.Table(pykx.q('
x         x1        x2                                  
--------------------------------------------------------
0.1872634 0.4176994 c9555cdf-57db-28a8-bf6c-67f6ee711a5f
0.7250709 0.8761714 6417d4b3-3fc6-e35a-1c34-8c5c3327b1e8
0.9351307 0.6030223 e8327955-bd9a-246a-0b17-fbf8f05fd28a
0.9452199 0.2329662 008bded0-3383-f19b-1d18-abfb199b1ac1
'))
```

